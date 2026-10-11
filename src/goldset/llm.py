"""The LLM client used for judging: model verification, balance, cost estimate, and a bounded batch runner."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings

# DeepSeek's published rates, US dollars per million tokens, cache-miss input.
RATES_USD_PER_MTOK = {
    "deepseek-flash": {"in": 0.28, "out": 0.42},
    "deepseek-v4-pro": {"in": 0.55, "out": 2.19},
}
# Retries for a transient provider failure. 429 and 5xx only: a 400 or a 401 will fail identically forever, and retrying
# it wastes the wall-clock budget an answer is held to.
RETRIES = 3
BACKOFF_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 20.0

DEFAULT_RATE = {"in": 0.60, "out": 2.40}  # a deliberately pessimistic fallback for an unlisted model
USD_PER_GBP = 1.27

# Rough but honest: about 4 characters per token for English prose.
CHARS_PER_TOKEN = 4


class LlmError(RuntimeError):
    """Anything that stopped a request. Never carries the key or the full prompt."""

    def __init__(self, *args: object, usage: dict | None = None) -> None:
        super().__init__(*args)
        self.usage = usage or {}


@dataclass
class Reply:
    """One completion, with everything the caller needs to price and attribute it."""

    content: str
    model: str = ""
    fingerprint: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Prompt tokens the provider served from its own cache.
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        return (self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0

    def cost_usd(self, model: str | None = None) -> float:
        """Priced from the same rate table the pre-flight estimate uses, so an estimate and an actual are comparable.
        Cached prompt tokens are charged at a tenth of the input rate, which is DeepSeek's published discount; if
        that changes the number moves, and it is better to be explicit and wrong than silent.
        """
        rate = RATES_USD_PER_MTOK.get(model or self.model or "", DEFAULT_RATE)
        fresh = max(0, self.prompt_tokens - self.cached_tokens)
        return ((fresh * rate["in"] + self.cached_tokens * rate["in"] * 0.1
                 + self.completion_tokens * rate["out"]) / 1_000_000)  # fmt: skip


@dataclass(frozen=True)
class Estimate:
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    gbp: float
    model: str
    basis: str

    def line(self) -> str:
        return (
            f"{self.calls:,} calls, about {self.input_tokens:,} input and {self.output_tokens:,} output "
            f"tokens on {self.model}: about ${self.usd:.2f} (~GBP {self.gbp:.2f}). {self.basis}"
        )


def estimate(prompts: list[str], model: str, output_tokens_each: int = 24) -> Estimate:
    """What a batch will cost, before any of it is sent."""
    rate = RATES_USD_PER_MTOK.get(model, DEFAULT_RATE)
    input_tokens = sum(len(p) for p in prompts) // CHARS_PER_TOKEN
    output_tokens = len(prompts) * output_tokens_each
    usd = (input_tokens * rate["in"] + output_tokens * rate["out"]) / 1_000_000
    return Estimate(
        calls=len(prompts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=round(usd, 4),
        gbp=round(usd / USD_PER_GBP, 4),
        model=model,
        basis=f"~{CHARS_PER_TOKEN} chars/token; rates {rate['in']}/{rate['out']} USD per Mtok"
        + ("" if model in RATES_USD_PER_MTOK else " (model not in the rate table: pessimistic fallback)"),
    )


@dataclass
class Client:
    """A minimal OpenAI-compatible chat client. Holds the key; never renders it."""

    key: str = field(default="", repr=False)
    base: str = ""
    model: str = ""
    timeout: float = 90.0
    concurrency: int = 4
    # Every model id and fingerprint that has answered this client, filled in by `_observe`.
    served_models: set[str] = field(default_factory=set, repr=False)
    fingerprints: set[str] = field(default_factory=set, repr=False)
    # Sent as `reasoning_effort` when set.
    reasoning_effort: str = ""

    def __post_init__(self) -> None:
        self.key = self.key or settings.deepseek_key
        self.base = (self.base or settings.deepseek_base).rstrip("/")
        self.model = self.model or settings.bulk_model

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _headers(self) -> dict[str, str]:
        # No e-mail address, no machine name: the User-Agent identifies the project and nothing else.
        return {"Authorization": f"Bearer {self.key}", "User-Agent": "on-the-record/goldset"}

    async def models(self) -> list[str]:
        """The model ids the account can actually use."""
        if not self.configured:
            raise LlmError("no API key is configured")
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            response = await http.get(f"{self.base}/models", headers=self._headers())
            if response.status_code != 200:
                raise LlmError(f"/models returned {response.status_code}")
            data = response.json()
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))

    async def balance(self) -> dict[str, Any]:
        """Remaining credit, where the provider exposes it. Absent is not an error."""
        if not self.configured:
            raise LlmError("no API key is configured")
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            response = await http.get(f"{self.base}/user/balance", headers=self._headers())
            if response.status_code != 200:
                return {"available": None, "note": f"balance endpoint returned {response.status_code}"}
            return response.json()

    async def available_usd(self) -> float | None:
        """The balance as a number, or None if the provider does not say. Never raises on a missing field."""
        try:
            data = await self.balance()
        except LlmError:
            return None
        for info in data.get("balance_infos") or []:
            if str(info.get("currency", "")).upper() == "USD":
                try:
                    return float(info.get("total_balance"))
                except (TypeError, ValueError):
                    return None
        return None

    async def affordable(
        self, estimate_usd: float, floor_usd: float | None = None, confirm_gbp: float | None = None
    ) -> tuple[bool, str]:
        """Whether this run may go ahead: cheap enough to take freely, and inside any balance floor."""
        floor = settings.min_balance_usd if floor_usd is None else floor_usd
        ceiling = settings.spend_confirm_gbp if confirm_gbp is None else confirm_gbp
        gbp = estimate_usd / USD_PER_GBP

        if ceiling > 0 and gbp >= ceiling:
            return False, (
                f"this run is estimated at GBP {gbp:.2f} (${estimate_usd:.2f}), at or above the "
                f"GBP {ceiling:.2f} that needs approval first. Pass --min-balance/--confirm-spend to "
                f"override once it is approved."
            )

        available = await self.available_usd()
        if floor > 0:
            if available is None:
                return False, f"the provider did not report a balance, and the floor is ${floor:.2f}"
            if available - estimate_usd < floor:
                return False, (
                    f"balance ${available:.2f} minus the estimated ${estimate_usd:.2f} leaves "
                    f"${available - estimate_usd:.2f}, below the ${floor:.2f} floor"
                )
        left = "" if available is None else f", about ${available - estimate_usd:.2f} left afterwards"
        return True, f"estimated GBP {gbp:.2f} (${estimate_usd:.2f}), under the GBP {ceiling:.2f} bar{left}"

    async def complete(self, prompt: str, *, system: str = "", max_tokens: int = 64,
                       temperature: float = 0.0) -> str:  # fmt: skip
        """The text only. Kept byte-identical in behaviour for every existing caller."""
        return (await self.chat(prompt, system=system, max_tokens=max_tokens,
                                temperature=temperature)).content  # fmt: skip

    async def chat(self, prompt: str, *, system: str = "", max_tokens: int = 64,
                   temperature: float = 0.0, json_object: bool = False) -> Reply:  # fmt: skip
        if not self.configured:
            raise LlmError("no API key is configured")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_object:
            # DeepSeek's JSON mode guarantees syntactically valid JSON, not a particular shape.
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            data = None
            last = ""
            for attempt in range(RETRIES + 1):
                try:
                    response = await http.post(f"{self.base}/chat/completions", headers=self._headers(),
                                               json=payload)  # fmt: skip
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last = f"{type(exc).__name__}"
                    if attempt == RETRIES:
                        raise LlmError(f"chat/completions unreachable after {attempt + 1} attempts ({last})") from exc
                    await asyncio.sleep(BACKOFF_SECONDS * (2 ** attempt))
                    continue
                if response.status_code == 200:
                    data = response.json()
                    break
                # The body can echo the request; only the status is safe to surface.
                last = f"HTTP {response.status_code}"
                retryable = response.status_code == 429 or 500 <= response.status_code < 600
                if not retryable or attempt == RETRIES:
                    raise LlmError(f"chat/completions returned {response.status_code}"
                                   + (f" after {attempt + 1} attempts" if attempt else ""))  # fmt: skip
                # Honour Retry-After when the provider sends one; it knows its own window better than we do.
                wait = BACKOFF_SECONDS * (2 ** attempt)
                header = response.headers.get("retry-after")
                if header:
                    try:
                        wait = max(wait, min(float(header), MAX_BACKOFF_SECONDS))
                    except ValueError:
                        pass
                await asyncio.sleep(wait)
            if data is None:  # pragma: no cover - the loop either breaks with data or raises
                raise LlmError(f"chat/completions produced no response ({last})")
        # Record WHICH model actually answered, not which alias was asked for.
        self._observe(data)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"unexpected response shape ({type(exc).__name__})") from exc
        # A reasoning model that runs out of budget while thinking returns "" with finish_reason "length".
        usage = data.get("usage") or {}
        if not (content or "").strip():
            raise LlmError(
                f"empty reply (finish_reason {choice.get('finish_reason')!r}); "
                "a reasoning model needs a larger max_tokens or reasoning_effort='none'",
                usage={**usage, "model": data.get("model") or self.model},
            )
        return Reply(
            content=content,
            model=str(data.get("model") or self.model),
            fingerprint=str(data.get("system_fingerprint") or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            # DeepSeek's name for it.
            cached_tokens=int(usage.get("prompt_cache_hit_tokens")
                              or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        )

    def _observe(self, data: dict[str, Any]) -> None:
        """Remember the served model id and fingerprint from a response. Never raises: provenance is worth having and
        is not worth failing a judgment over.
        """
        try:
            served = str(data.get("model") or "").strip()
            fingerprint = str(data.get("system_fingerprint") or "").strip()
        except Exception:  # noqa: BLE001 - a malformed body is the caller's problem, not provenance's
            return
        if served:
            self.served_models.add(served)
        if fingerprint:
            self.fingerprints.add(fingerprint)

    def provenance(self) -> dict[str, Any]:
        """What actually served this client's calls, for the report and for `save_judgments`."""
        return {
            "requested_model": self.model,
            "served_models": sorted(self.served_models),
            "system_fingerprints": sorted(self.fingerprints),
            "reasoning_effort": self.reasoning_effort or "(provider default)",
            "one_model": len(self.served_models) <= 1,
        }

    async def batch(self, prompts: list[str], *, system: str = "", max_tokens: int = 64) -> list[str | None]:
        """Run prompts with bounded concurrency. A failure yields None for that prompt, never an exception."""
        gate = asyncio.Semaphore(self.concurrency)

        async def one(prompt: str) -> str | None:
            async with gate:
                try:
                    return await self.complete(prompt, system=system, max_tokens=max_tokens)
                except Exception:  # noqa: BLE001 - one bad prompt must not lose the other 2,000
                    return None

        return list(await asyncio.gather(*(one(p) for p in prompts)))


# A STANDALONE 0-3.
_GRADE = re.compile(r"(?<![\d.])[0-3](?![\d.])")


def parse_grade(text: str | None) -> int | None:
    """The first standalone 0-3 in the reply, or None. Models add a sentence however firmly told not to."""
    if not text:
        return None
    if (match := _GRADE.search(text)) is not None:
        return int(match.group(0))
    return None


def parse_json_object(text: str | None) -> dict | None:
    """The first JSON object in the reply. Models wrap JSON in prose and in code fences."""
    if not text:
        return None
    start, depth = None, 0
    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
                return value if isinstance(value, dict) else None
    return None
