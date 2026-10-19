"""A blind, rate-limited panel of judges from different model families."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Where a provider's key is read from.
_DOTENV_CACHE: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    global _DOTENV_CACHE
    if _DOTENV_CACHE is None:
        found: dict[str, str] = {}
        path = Path(__file__).resolve().parents[2] / ".env"
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                found[name.strip().casefold()] = value.strip().strip('"').strip("'")
        _DOTENV_CACHE = found
    return _DOTENV_CACHE


KEY_NAMES: dict[str, tuple[str, ...]] = {
    "deepseek": ("DEEPSEEK_KEY", "DEEPSEEK_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "CLAUDE_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GOOGLEAPIKEY", "GEMINI_API_KEY"),
}

# The family that GENERATES answers.
GENERATOR_FAMILY = "deepseek"


@dataclass(frozen=True)
class Provider:
    """One judge: where it lives, what it costs in time, and how far it may be trusted."""

    name: str
    family: str
    base: str
    model: str
    # Requests a minute this provider tolerates.
    per_minute: int
    # Requests a day, where the provider publishes one. 0 means no stated daily cap.
    per_day: int = 0
    # OpenAI-compatible providers share one client; the others need their own request shape.
    openai_compatible: bool = True
    # True where the provider is known to be unreliable or geo-restricted: a failure is logged and skipped rather than
    # raised, so one flaky member cannot take down the panel.
    fragile: bool = False
    note: str = ""

    @property
    def min_interval_s(self) -> float:
        """The gap to leave between requests. The whole point of the rate limiter."""
        return 60.0 / self.per_minute if self.per_minute > 0 else 0.0

    def key(self) -> str:
        for name in KEY_NAMES.get(self.name, ()):
            # Case-insensitive: the maintainer's key is spelled `googleapikey` in.env.
            for candidate in (name, name.lower(), name.upper()):
                if value := os.environ.get(candidate, "").strip():
                    return value
                if value := _dotenv().get(candidate.casefold(), "").strip():
                    return value
        return ""

    @property
    def available(self) -> bool:
        return bool(self.key())

    def unavailable_reason(self) -> str:
        if self.available:
            return ""
        names = " or ".join(KEY_NAMES.get(self.name, ("?",)))
        return f"no key: set {names} in .env"


# The panel.
PROVIDERS: tuple[Provider, ...] = (
    Provider("anthropic", "anthropic", "https://api.anthropic.com/v1", "claude-sonnet-5",
             per_minute=50, openai_compatible=False,
             note="The owner's intended primary judge. NO KEY IS PRESENT. Use a Claude 5 model when one is "
                  "added -- `claude-3-5-sonnet` is two generations old."),
    Provider("deepseek", "deepseek", "https://api.deepseek.com/v1", "deepseek-flash",
             per_minute=60,
             note="The incumbent relevance judge, validated in Session 2 and re-checked in E1."),
    # `/v1beta/openai` is Gemini's documented OpenAI-compatible path, which is why it is used — not because `/v1beta`
    # fails.
    Provider("google", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3-flash-preview",
             per_minute=4, per_day=18, openai_compatible=False, fragile=True,
             note="Free tier is very tight and model availability varies by country; the owner reports it "
                  "failing. Marked fragile: skipped on error, never blocking."),
)

BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}


class RateLimiter:
    """One request at a time per provider, with at least `min_interval` between them."""

    def __init__(self, min_interval_s: float, per_day: int = 0,
                 sleep=asyncio.sleep, clock=time.monotonic):  # fmt: skip
        self.min_interval_s = max(0.0, min_interval_s)
        self.per_day = per_day
        self._sleep, self._clock = sleep, clock
        self._lock = asyncio.Lock()
        # None until the first request.
        self._last: float | None = None
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return bool(self.per_day) and self.used >= self.per_day

    async def acquire(self) -> None:
        """Wait until another request is allowed, then record it."""
        if self.exhausted:
            raise QuotaExhausted(f"the day's cap of {self.per_day} requests is used up")
        async with self._lock:
            if self.min_interval_s and self._last is not None:
                waited = self._clock() - self._last
                if waited < self.min_interval_s:
                    await self._sleep(self.min_interval_s - waited)
            self._last = self._clock()
            self.used += 1


class QuotaExhausted(RuntimeError):
    """A provider's stated daily cap is spent. Raised before the request, never after a ban."""


@dataclass
class Verdict:
    """One provider's answer to one prompt, and what it cost to get."""

    provider: str
    family: str
    reply: str | None
    error: str = ""
    seconds: float = 0.0
    served_model: str = ""

    @property
    def ok(self) -> bool:
        return self.reply is not None and not self.error


@dataclass
class PanelResult:
    """Every provider's independent answer to one prompt."""

    prompt_id: str
    verdicts: list[Verdict] = field(default_factory=list)

    def by_provider(self) -> dict[str, Verdict]:
        return {v.provider: v for v in self.verdicts}

    @property
    def answered(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.ok]

    def agreement(self, parse) -> dict[str, Any]:
        """Whether the providers that answered agree, once their replies are parsed to a label."""
        labels = {v.provider: parse(v.reply) for v in self.answered}
        usable = {p: v for p, v in labels.items() if v is not None}
        distinct = set(usable.values())
        return {
            "labels": labels,
            "answered": len(self.answered),
            "unanimous": len(distinct) <= 1 and bool(usable),
            "distinct": sorted(str(d) for d in distinct),
            "split": len(distinct) > 1,
        }


def listed_models(provider: Provider, *, timeout: float = 10.0) -> set[str]:
    """The model ids the provider's /models endpoint lists right now. Raises on any failure: a check that could not run
    has not passed.
    """
    import httpx

    response = httpx.get(f"{provider.base.rstrip('/')}/models", timeout=timeout,
                         headers={"Authorization": f"Bearer {provider.key()}"})  # fmt: skip
    response.raise_for_status()
    rows = response.json().get("data") or []
    return {str(row.get("id", "")).removeprefix("models/") for row in rows if isinstance(row, dict)}


def verify_model(provider: Provider) -> str | None:
    """None when `provider.model` is on the provider's live list; otherwise why it cannot be trusted."""
    try:
        ids = listed_models(provider)
    except Exception as exc:  # noqa: BLE001 - reported as the reason the judge is refused
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return f"{provider.name} refused the key (HTTP {status}): it is invalid, revoked or a placeholder"
        return f"could not list {provider.name}'s models ({type(exc).__name__}{f', HTTP {status}' if status else ''})"
    if provider.model not in ids:
        return (f"{provider.model!r} is not on {provider.name}'s live /models list ({len(ids)} listed): "
                "retired or renamed")
    return None


def panel(names: Sequence[str] | None = None, *, exclude_family: str = "") -> list[Provider]:
    """The providers to use: available, and not from a family we must not judge with."""
    chosen = [BY_NAME[n] for n in names] if names else list(PROVIDERS)
    return [p for p in chosen if p.available and p.family != exclude_family]


def unavailable(names: Sequence[str] | None = None) -> dict[str, str]:
    """provider -> why it cannot be used. Reported, so a missing panel member is never silent."""
    chosen = [BY_NAME[n] for n in names] if names else list(PROVIDERS)
    return {p.name: p.unavailable_reason() for p in chosen if not p.available}


def primary(available: Sequence[Provider], exclude_family: str = "") -> Provider | None:
    """The authoritative judge: the first available, non-fragile provider from another family."""
    for provider in available:
        if not provider.fragile and provider.family != exclude_family:
            return provider
    return None
