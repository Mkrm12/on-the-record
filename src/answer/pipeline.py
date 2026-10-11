"""One path from a question to a verified answer."""

from __future__ import annotations

import hashlib
import json
import logging
import time

import psycopg as _psycopg
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from answer import cache as K
from answer import keys as _keys
from answer import cards as C
from answer import generate as G
from answer import names as N
from answer import prompts as P
from answer import router as RO
from answer import routes as RT
from answer import sealing as SG
from answer import trace as TR
from answer.contract import Abstention, Answer, Provenance
from app import sealed
from retrieval import hybrid as H

log = logging.getLogger("answer.pipeline")

ROOT = Path(__file__).resolve().parents[2]
FROZEN_PATH = ROOT / "config" / "retrieval_frozen.json"


class PipelineError(RuntimeError):
    """The pipeline could not be assembled — no frozen config, no GPU, no key."""


@dataclass(frozen=True)
class Frozen:
    """The frozen retrieval configuration and its hash, read from disk."""

    config: dict[str, Any]
    config_hash: str
    recipe: str
    frozen_at: str = ""
    adopted_arms: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path = FROZEN_PATH) -> Frozen:
        if not path.exists():
            raise PipelineError(f"no frozen retrieval config at {path}; E1 must freeze before E3 can answer")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            config=data["config"], config_hash=data["config_hash"],
            recipe=data.get("embedding_recipe", ""), frozen_at=data.get("frozen_at", ""),
            adopted_arms=tuple(data.get("adopted_arms") or ()),
        )  # fmt: skip

    def search_config(self) -> Any:
        """A `SearchConfig` built from the frozen dict, with any key this build does not know about refused."""
        from retrieval.hybrid import SearchConfig

        known = {f.name for f in __import__("dataclasses").fields(SearchConfig)}
        unknown = sorted(set(self.config) - known)
        if unknown:
            raise PipelineError(
                f"the frozen config sets {unknown}, which this build of SearchConfig does not have. The hash "
                f"{self.config_hash} would no longer describe what runs. Reconcile before answering.")
        # And the other direction: a knob the file does not name is decided by a default nobody froze, so the hash would
        # describe a configuration other than the one running.
        missing = sorted(known - set(self.config) - {"filters"})
        if missing:
            raise PipelineError(
                f"the frozen config does not name {missing}, so this build's defaults would decide them and the hash "
                f"{self.config_hash} would not describe what runs. Re-freeze (or amend) before answering.")
        return SearchConfig(**self.config)

    def verify_hash(self) -> bool:
        """Whether the stored hash still matches the stored config."""
        return frozen_hash(self.config) == self.config_hash


def frozen_hash(config: dict[str, Any]) -> str:
    """The frozen configuration's hash, exactly as `scripts/uplift.py freeze` computes it: sha256 over
    `json.dumps(sort_keys=True, default=str)` with the default separators, truncated to 16 hex characters.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def amend_frozen(path: Path, changes: dict[str, Any], *, reason: str, evidence: str,
                 date: str | None = None) -> dict[str, Any]:
    """Change named knobs of FROZEN retrieval, recording why — the only sanctioned way to."""
    import time as _time

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    config = dict(data["config"])
    if frozen_hash(config) != data["config_hash"]:
        raise PipelineError(f"{path.name}: the stored hash does not match the stored config (a hand edit?); amending "
                            "over it would give an unrecorded change a valid hash. Reconcile first.")
    if not str(reason or "").strip() or not str(evidence or "").strip():
        raise PipelineError("an amendment needs a reason and the evidence it rests on")
    from retrieval.hybrid import SearchConfig

    knobs = {f.name for f in __import__("dataclasses").fields(SearchConfig)} - {"filters"}
    unknown = sorted(set(changes) - knobs)
    if unknown:
        raise PipelineError(f"{unknown} is not a frozen SearchConfig knob")
    changed = {k: [config.get(k), v] for k, v in changes.items() if k not in config or config[k] != v}
    if not changed:
        raise PipelineError("the amendment changes nothing")
    config.update(changes)
    new_hash = frozen_hash(config)
    Frozen(config=config, config_hash=new_hash, recipe=data.get("embedding_recipe", "")).search_config()  # validates
    entry = {"date": date or _time.strftime("%Y-%m-%d"), "changed": changed, "previous_hash": data["config_hash"],
             "config_hash": new_hash, "reason": reason.strip(), "evidence": evidence.strip()}
    data["config"], data["config_hash"] = config, new_hash
    data.setdefault("amendments", []).append(entry)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return entry


# Re-exported so existing callers keep working.
normalise_question = _keys.normalise_question
question_hash = _keys.question_hash
CORPUS_VERSION = _keys.CORPUS_VERSION


@dataclass
class Runtime:
    """The assembled pieces. Built once per process, reused per question."""

    searcher: Any
    client: Any
    frozen: Frozen
    model_id: str = ""
    verifier: Any = None
    verifier_name: str = "deterministic"
    verifier_precision: str = ""
    spend_usd: float = 0.0
    calls: int = 0
    cached_prompt_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    # Whether the router may consult the model for questions the rules leave as plain `retrieval`.
    route_with_model: bool = False
    # The statement timeout actually used when searching, in ms.
    serving_timeout_ms: int = 20_000

    def provenance(self, route: str, filters: dict[str, Any] | None = None,
                   prompt_version: str = P.PROMPT_VERSION) -> Provenance:  # fmt: skip
        return Provenance(
            route=route, resolved_filters=filters or {}, corpus_version=sealed.served_label(CORPUS_VERSION),
            retrieval_config_hash=self.frozen.config_hash, recipe=self.frozen.recipe,
            prompt_version=prompt_version, model_id=self.model_id or getattr(self.client, "model", "unknown"),
            code_version=_keys.code_version(),
        )  # fmt: skip

    def score_batch(self):
        """The entailment callable, or None when no verifier is reachable."""
        if self.verifier is None:
            return None
        verifier = self.verifier

        async def score(evidence_texts, claim_texts):
            pairs = list(zip(list(evidence_texts), list(claim_texts), strict=True))
            scores = await verifier.verify(pairs)
            # A pair the service could not score comes back None.
            return [(-1.0 if s is None else s) for s in scores]

        return score


def to_search_filters(filters: Any) -> Any:
    """The question's own scope — House, section, sitting date or range — as retrieval filters."""
    from retrieval.hybrid import SearchFilters

    if filters is None:
        return SearchFilters()
    get = (lambda k: filters.get(k)) if isinstance(filters, dict) else (lambda k: getattr(filters, k, None))
    date_from = get("on_date") or get("from_date")
    date_to = get("on_date") or get("to_date")
    year = get("year")
    if not (date_from or date_to) and year:
        from datetime import date as _date

        date_from, date_to = _date(int(year), 1, 1), _date(int(year), 12, 31)
    if isinstance(date_from, str):
        from datetime import date as _date

        date_from = _date.fromisoformat(date_from[:10])
    if isinstance(date_to, str):
        from datetime import date as _date

        date_to = _date.fromisoformat(date_to[:10])
    house, section = get("house"), get("section")
    return SearchFilters(date_from=date_from, date_to=date_to,
                         houses=(house,) if house else (), sections=(section,) if section else ())


# Below this many hits a scoped search is too narrow and is widened to the whole corpus.
WIDEN_BELOW = 3


def retrieval_scope(filters: Any) -> Any:
    """The scope retrieval is given for this question: its DATE, with House and section when it also names them — and
    nothing when it names no date.
    """
    scope = to_search_filters(filters)
    if not (scope.date_from or scope.date_to):
        from retrieval.hybrid import SearchFilters

        return SearchFilters()
    return scope


# Cards for a question the router could not date.
UNDATED_TOP_K = 10
# The overview route's card budget.
OVERVIEW_TOKEN_BUDGET = 16_000


def deck_size(route: str, *, dated: bool, top_k: int = 10) -> tuple[int, int]:
    """(hits to retrieve, card token budget) for a question: the deck scales with the question's scope."""
    if route == "overview":
        return RT.OVERVIEW_POOL, OVERVIEW_TOKEN_BUDGET
    return (top_k if dated else max(top_k, UNDATED_TOP_K)), C.DEFAULT_TOKEN_BUDGET


async def search_deck(runtime: Runtime, question: str, top_k: int = 10,
                      token_budget: int = C.DEFAULT_TOKEN_BUDGET,
                      filters: Any = None) -> tuple[C.Deck, Any]:  # fmt: skip
    """Retrieval under the frozen config, formatted into cards, scoped by `filters` when given (a `SearchFilters`).
    """
    import dataclasses as _dcf

    config = runtime.frozen.search_config()
    if filters:
        config = _dcf.replace(config, filters=filters)
    if top_k > config.pool:
        # The reranker pool caps what can come back — `collapse(limit=pool)` runs BEFORE `top_k` — so the overview
        # route's 60 was silently 20, and the broadest questions got the same-sized deck as a pinpoint one.
        config = _dcf.replace(config, pool=top_k)
    if runtime.serving_timeout_ms:
        # `replace`, not assignment: SearchConfig is a FROZEN dataclass, and assigning raised FrozenInstanceError on
        # every single question — a whole 223-item run failed in 3.1 seconds because of it.
        import dataclasses as _dc

        config = _dc.replace(config, statement_timeout_ms=runtime.serving_timeout_ms)
    try:
        result = await runtime.searcher.search(question, top_k=top_k, config=config)
    except H.RetrievalError as exc:
        # ONE retry, and only for a cancelled statement.
        if "QueryCanceled" not in str(exc):
            raise
        log.warning("the search was cancelled; retrying once (%s)", exc)
        result = await runtime.searcher.search(question, top_k=top_k, config=config)
    hits = list(result.hits)
    # Each card names its speaker as the record printed them, never by a title taken since ("Lord Gove" addressing the
    # Commons).
    speakers: dict[str, str] = {}
    if hits:
        async with runtime.searcher.pool.connection() as conn:
            speakers = await N.as_printed(conn, [h.speech_id for h in hits])
    return C.build_deck(hits, budget_tokens=token_budget, speakers=speakers), result


async def answer(runtime: Runtime, question: str, *, route: str = "retrieval", top_k: int = 10,
                 variant_key: str | None = None, facts: list[dict] | None = None,
                 filters: dict[str, Any] | None = None, budget: G.Budget | None = None,
                 token_budget: int = C.DEFAULT_TOKEN_BUDGET) -> G.Generated:  # fmt: skip
    """Retrieve, generate, verify. The single entry point every caller uses."""
    deck, result = await search_deck(runtime, question, top_k, token_budget)
    variant = P.variant(variant_key)
    generated = await G.answer_question(
        question, deck, runtime.client, runtime.provenance(route, filters, variant.version),
        variant_key=variant.key, facts=facts, score_batch=runtime.score_batch(),
        budget=budget, verifier_name=runtime.verifier_name, precision=runtime.verifier_precision,
    )  # fmt: skip
    generated.route = route

    if generated.budget:
        runtime.spend_usd += generated.budget.spent_usd
        runtime.calls += generated.budget.calls
    runtime.prompt_tokens += generated.actual_prompt_tokens
    # 's trace row.
    runtime.trace.append({
        "question": question, "route": route, "variant": variant.key,
        "outcome": ("abstain" if generated.answer.abstention else
                    ("clarify" if generated.answer.facets.get("clarify") else "answer")),
        "abstention": generated.answer.abstention.value if generated.answer.abstention else None,
        "cards": len(deck.cards), "cards_dropped": deck.dropped,
        "deck_tokens_estimated": deck.used_tokens,
        "prompt_tokens": generated.actual_prompt_tokens,
        "token_estimate_error": round(generated.token_estimate_error, 4),
        "claims_generated": generated.claims_generated,
        "claims_rejected": generated.claims_rejected,
        "claims_verified": generated.claims_verified,
        "claims_dropped": generated.claims_dropped,
        "repaired": generated.repaired, "fell_back": generated.fell_back,
        "degraded": generated.degraded,
        "seconds": round(generated.budget.elapsed, 3) if generated.budget else None,
        "usd": round(generated.budget.spent_usd, 6) if generated.budget else None,
        "served_model": generated.served_model, "fingerprint": generated.fingerprint,
        "retrieval_hits": len(result.hits),
    })
    return generated


async def _prewarm(pool) -> None:
    """Pull the vector index into the OS page cache before the first question."""
    try:
        async with pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            for relation in ("chunk_embedding_idx", "chunk"):
                cur = await conn.execute("SELECT pg_prewarm(%s, 'read')", (relation,))
                blocks = (await cur.fetchone())[0]
                log.info("prewarmed %s blocks of %s", blocks, relation)
            await conn.commit()
    except Exception as exc:  # noqa: BLE001
        log.info("could not prewarm the index (%s); the first queries will be colder", type(exc).__name__)


async def _check_index_form(pool, wanted: str) -> str | None:
    """Warn, loudly, when the live HNSW index is not the form the dense leg's ORDER BY is written for."""
    from retrieval import vector_index as V

    try:
        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT pg_get_indexdef(to_regclass(%s))", (V.INDEX_NAME,))).fetchone()
            await conn.rollback()
    except Exception as exc:  # noqa: BLE001 - a check that cannot run is reported, never fatal
        log.warning("could not read %s's definition (%s)", V.INDEX_NAME, type(exc).__name__)
        return None
    live = V.form_of(row[0] if row else None)
    if live != wanted:
        log.warning("%s is %s, but frozen retrieval queries the %s form: every dense search will be a sequential "
                    "scan. Run: uv run python scripts/switch_vector_index.py run --form %s",
                    V.INDEX_NAME, live or "missing or of neither form", wanted, wanted)
    return live


async def build_runtime(database: str = "hansard_v2", *, model: str | None = None,
                        with_verifier: bool = False, spend_cap_usd: float | None = None) -> tuple[Runtime, Any]:
    """Assemble everything, returning the runtime and the pool to close."""
    from app.config import settings
    from app.db import make_pool
    from goldset.llm import Client
    from psycopg.conninfo import make_conninfo
    from retrieval.gpu_client import GpuClient
    from retrieval.hybrid import HybridSearcher

    frozen = Frozen.load()
    if not frozen.verify_hash():
        log.warning("the frozen config's hash does not match its contents; provenance would be wrong")
    pool = make_pool(make_conninfo(settings.pg_dsn, dbname=database))
    await pool.__aenter__()
    searcher = HybridSearcher(pool, GpuClient())
    await searcher.refresh()
    await _prewarm(pool)
    await _check_index_form(pool, frozen.search_config().dense_index)
    # reasoning_effort="none": this task assembles a JSON object from cards already in front of it, and deepseek models
    # are reasoning models.
    client = Client(model=model or settings.bulk_model, reasoning_effort="none")
    if not client.configured:
        raise PipelineError("no DeepSeek key is configured; E3 cannot generate")

    runtime = Runtime(searcher=searcher, client=client, frozen=frozen, model_id=client.model)
    if with_verifier:
        try:
            # The service, not the model.
            probe = GpuClient()
            health = await probe.health() if hasattr(probe, "health") else {}
            runtime.verifier = probe
            info = (health or {}).get("verifier") or {}
            runtime.verifier_name = str(info.get("name") or "minicheck-via-gpu-service")
            runtime.verifier_precision = str(info.get("precision") or "")
            log.info("verifier reachable over HTTP: %s", info or "loads on first /verify call")
        except Exception as exc:  # noqa: BLE001 - degrading is the designed behaviour
            # The message, not just the class.
            log.warning("verifier unavailable, paraphrases will be dropped: %s: %s",
                        type(exc).__name__, " ".join(str(exc).split())[:300])
    return runtime, pool

# --- the one entry point ---------------------------------------------------------------------------------


@dataclass
class Asked:
    """What answering one question produced, and everything the trace needs to record it."""

    answer: Any
    route: str
    route_by: str = "rules"
    route_why: str = ""
    # The fact the router read from the question's words; None for anything that is not a fact lookup.
    category: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    generated: Any = None
    set_result: Any = None
    overview: Any = None
    attempts: list = field(default_factory=list)
    trace_id: int | None = None
    retrieval_ms: int = 0
    # How many passages the search returned.
    retrieval_hits: int | None = None
    # How many cards the generator was shown, and how many sentences they held.
    cards: int | None = None
    card_sentences: int | None = None
    # Cards the token budget left out of the deck.
    cards_dropped: int | None = None
    # How many of those sentences carry an id a claim may cite.
    cards_citable: int | None = None
    # The retrieval scope actually used (date, House, section), and whether it had to be widened to the whole corpus
    # because it left fewer than WIDEN_BELOW hits.
    retrieval_scope: dict[str, Any] = field(default_factory=dict)
    retrieval_widened: bool = False
    # Set by an evaluation after the deck is built: whether a speech the gold item names as relevant is in it, and
    # where.
    gold_evidence_retrieved: bool | None = None
    gold_evidence_rank: int | None = None
    # Whether anything the pool graded merely RELATED (grade 1) was shown.
    related_evidence_retrieved: bool | None = None
    total_ms: int = 0
    failed_reason: str | None = None
    # Why the exit guard withheld the answer (each thing it held dated inside the sealed window).
    sealed_withheld: list[str] = field(default_factory=list)
    # The best claim's relevance to the question (the cross-encoder), whatever the gate decided; None when the gate did
    # not run (no claims, another route, no scorer).
    claim_relevance: float | None = None

    @property
    def claim_counts(self) -> dict[str, int | None]:
        """Claims generated, verified and dropped: the retrieval route's `Generated`, or summed over an overview's
        debates.
        """
        if self.generated is not None:
            g = self.generated
            return {"claims_generated": g.claims_generated, "claims_verified": g.claims_verified,
                    "claims_dropped": g.claims_dropped, "claims_rejected": g.claims_rejected}  # fmt: skip
        if self.overview is not None:
            o = self.overview
            return {"claims_generated": o.claims_generated, "claims_verified": o.per_debate_claims,
                    "claims_dropped": o.claims_dropped, "claims_rejected": None}  # fmt: skip
        return {"claims_generated": None, "claims_verified": None, "claims_dropped": None, "claims_rejected": None}

    @property
    def degraded_reason(self) -> str:
        """Why this answer was produced without the full verifier, or "" — a different system from the verified one,
        which an evaluation must count, not average in.
        """
        generated = self.generated
        if generated is not None and getattr(generated, "degraded", False) and not getattr(generated, "generator_failed", ""):
            return getattr(generated, "degraded_reason", "") or "degraded (no reason recorded)"
        return getattr(self.overview, "degraded_reason", "") if self.overview is not None else ""

    @property
    def outcome(self) -> str:
        if self.failed_reason:
            return "error"
        if self.answer is None:
            return "error"
        if self.answer.abstention:
            return "abstain"
        if (self.answer.facets or {}).get("clarify"):
            return "clarify"
        return "answer"


# The verifier proves a quote EXISTS and belongs to its speaker; nothing asked whether a claim is ABOUT the question.
CLAIM_RELEVANCE_FLOOR = -7.65


async def relevance_gate(runtime: Any, question: str, answer: Answer | None, *, record: Any = None) -> Answer | None:
    """The answer, or a decline when not one of its claims is about the question."""
    claims = list(answer.showable_claims) if answer is not None and not answer.abstention else []
    gpu = getattr(getattr(runtime, "searcher", None), "gpu", None)
    if not claims or gpu is None:
        return answer
    try:
        scores = [s for s in await gpu.rerank(question, [c.text for c in claims]) if s is not None]
    except Exception as exc:  # noqa: BLE001 - unavailable: the answer stands, as retrieval itself degrades
        log.warning("the claim-relevance check could not run (%s); the answer stands unchecked", type(exc).__name__)
        return answer
    if scores and record is not None:
        record.claim_relevance = round(max(scores), 4)
    if not scores or max(scores) >= CLAIM_RELEVANCE_FLOOR:
        return answer
    log.info("claim relevance: best %.2f < %.2f; declining a near miss", max(scores), CLAIM_RELEVANCE_FLOOR)
    return Answer(question=question, provenance=answer.provenance, abstention=Abstention.NOT_IN_RECORD,
                  abstention_detail=G.READER_REASONS[Abstention.NOT_IN_RECORD])


# At or above this many distinct members using the quoted words, "who said them?" has no single answer.
COMMON_PHRASE_SPEAKERS = 10


async def common_phrase_clarification(conn, question: str, provenance: Any) -> Answer | None:
    """A clarification when the question asks who said words many members use, else None."""
    from answer import resolve as RS
    from answer import router as RO

    phrase = RO.quoted_phrase(question)
    if not phrase:
        return None
    speakers = await RS.phrase_speakers(conn, phrase)
    if speakers < COMMON_PHRASE_SPEAKERS:
        return None
    note = (f"Those words were used by {speakers} different members in the record — a common or procedural form — so "
            f"who said them has no single answer. Say which debate, date or member you mean.")
    return Answer(question=question, provenance=provenance, coverage_note=note, rendered=note,
                  facets={"clarify": ["which debate, date or member do you mean?"], "reason": note})


def clarify_answer(question: str, provenance: Any, *, why: str, clarify: tuple[str, ...] | list[str]) -> Answer:
    """The router's question back ("which MP do you mean?"), stated to the reader."""
    def sentence(text: str) -> str:
        text = " ".join(str(text).split()).rstrip(".")
        first = next((i for i, ch in enumerate(text) if ch.isalpha()), None)  # "'the prime minister' …" too
        return text[:first] + text[first].upper() + text[first + 1:] if first is not None else text

    asked = " ".join(q if q.endswith("?") else f"{q}." for q in (sentence(c) for c in clarify) if q)
    rendered = " ".join(part for part in (f"{sentence(why)}." if why else "", asked) if part)
    return Answer(question=question, provenance=provenance, facets={"clarify": list(clarify), "reason": why},
                  coverage_note=why, rendered=rendered)  # fmt: skip


def request_failure(result: Asked) -> str:
    """Why this request FAILED, or "": the request itself raised, or the generator was never reached — on the retrieval
    route, or on every debate of an overview. A failed request's decline is the system's ("could not be reached"),
    never the model's: it is traced as an error, never cached, and an evaluation records it as failed rather than
    grading it (a 401 on every call was graded "over_abstention" and blamed on the prompt).
    """
    if result.failed_reason:
        return result.failed_reason
    for part in (result.generated, result.overview):
        failed = getattr(part, "generator_failed", "") if part is not None else ""
        if failed:
            return f"generator: {failed}"
    return ""


# Said when an answer reached the exit with nothing a reader could be shown: a decline in the system's words.
NOTHING_SHOWABLE = ("The system found material for this question but produced nothing it could show, so it gives no "
                    "answer rather than an empty one.")


def reader_ready(answer: Answer | None) -> Answer | None:
    """Every answer leaves `ask` with text a reader can be shown, or as a decline that says why — never empty."""
    if answer is None or answer.abstention is not None or (answer.rendered or "").strip():
        return answer
    text = answer.render()
    if text:
        return Answer.model_validate({**answer.model_dump(), "rendered": text})
    log.error("an answer reached the exit with nothing showable (facets %s); declined", sorted(answer.facets or {}))
    return Answer(question=answer.question, provenance=answer.provenance, abstention=Abstention.INSUFFICIENT_EVIDENCE,
                  abstention_detail=NOTHING_SHOWABLE)  # fmt: skip


# Fact categories that name a division, and so can meet several divisions sharing one title on one day.
_DIVISION_FACTS = ("division_result", "division_tally", "division_vote", "division_debate")


def _division_description(title: str | None, motion: str | None, limit: int = 110) -> str | None:
    """How a reader tells same-titled divisions apart: the motion where it says more than the title, else the title.
    """
    text = " ".join((motion or "").split())
    if text and text.rstrip(".") != (title or "").rstrip("."):
        return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return title


async def structured_answer(conn, question: str, decision: Any, provenance: Any) -> tuple[Answer | None, list[str]]:
    """The structured route: E2's fact engine and no model. (answer, notes), or (None, notes) to fall back to
    retrieval.
    """
    from answer import facts as F
    from answer import resolve as RS

    category = decision.category
    params, notes = await RS.parameters(conn, category, question, decision.filters)
    extra_note = ""
    if not params and category in _DIVISION_FACTS:
        tied, why = await RS.division_candidates(conn, question, house=decision.filters.house,
                                                 when=decision.filters.on_date, number=RS.division_number(question))
        if len(tied) > 1:
            # `resolve.parameters` returns {} when ANYTHING required is unresolved, so an ambiguous division never gets
            # as far as the member — and `division_vote` needs both.
            member: dict[str, Any] = {}
            if category == "division_vote":
                mnis, member_why = await RS.mnis_id(conn, question)
                notes.append(f"mnis_id: {member_why}")
                if mnis is None:
                    return None, notes
                member = {"mnis_id": mnis}
            candidates = []
            for candidate_id in tied[:8]:
                try:
                    candidates.append((candidate_id, await F.answer(conn, category,
                                                                    {"division_id": candidate_id, **member})))
                except F.FactError as exc:
                    notes.append(f"candidate {candidate_id} could not be answered: {exc}")
                    return None, notes
            answered = [(i, fact) for i, fact in candidates if fact.ok]
            # The answer-bearing fields ONLY: `division_id` and `division_number` differ between any two candidates by
            # definition, so comparing whole values would report disagreement every time.
            distinct = {((fact.value or {}).get("result"),) if category == "division_result" else fact.answer_fields()
                        for _, fact in answered}
            if len(distinct) == 1 and category == "division_result" and len(answered) > 1:
                facts_ = [fact for _, fact in answered]
                caveat = " ".join(dict.fromkeys(f.caveat for f in facts_ if f.caveat))
                sentence = F.several_results_sentence(facts_)
                notes.append(f"{len(facts_)} candidate divisions share one result ({why})")
                return Answer(
                    question=question, provenance=provenance,
                    structured_facts=tuple(f.to_contract() for f in facts_),
                    coverage_note=caveat, rendered=f"{sentence} {caveat}".strip(),
                ), notes  # fmt: skip
            if len(distinct) == 1:
                params = {"division_id": answered[0][0], **member}
                if len(answered) < len(candidates):
                    extra_note = (f"{len(candidates)} divisions on that day match this description; this member is "
                                  f"recorded voting in {len(answered)} of them.")
                else:
                    extra_note = (f"{len(candidates)} divisions on that day match this description, and they give "
                                  f"the same answer.")
                notes.append(f"{extra_note} ({why})")
            elif len(distinct) > 1:
                # Which one is meant is asked — and what the record holds for EACH is stated, because a clarification
                # used to render as nothing (six dev answers the judge saw as empty).
                cur = await conn.execute("SELECT id, number, title, motion_text, house, division_date FROM division "
                                         "WHERE id = ANY(%s)", ([i for i, _ in answered],))  # fmt: skip
                rows = await cur.fetchall()
                # A Lords division's title is often just the bill's name; its motion says which amendment it was.
                meta = {row[0]: (row[1], _division_description(row[2], row[3])) for row in rows}
                where = {"house": rows[0][4], "date": rows[0][5]} if rows else {}
                facts_ = [fact for _, fact in answered]
                asserted = category not in F.NAME_ONLY_WHEN_DIFFERENT  # Named, not answered
                return Answer(
                    question=question, provenance=provenance,
                    facets={"clarify": [{"division_id": i} for i, _ in answered[:8]],
                            "reason": f"{why}; the answer differs between them"},
                    structured_facts=tuple(f.to_contract() for f in facts_) if asserted else (),
                    coverage_note=(" ".join(dict.fromkeys(f.caveat for f in facts_ if f.caveat)) if asserted else ""),
                    rendered=F.candidates_sentence(category, facts_, meta, **where),
                ), notes  # fmt: skip
            elif category == "division_vote":
                # Recorded in none of them: that is what the record says, and searching speeches cannot improve on it.
                return Answer(
                    question=question, provenance=provenance, abstention=Abstention.NOT_IN_RECORD,
                    abstention_detail=(f"That member is not recorded in the division lists of any of the "
                                       f"{len(candidates)} divisions matching this description on that day."),
                ), notes  # fmt: skip
    if not params:
        return None, notes
    try:
        fact = await F.answer(conn, category, params)
    except F.FactError as exc:
        return None, [*notes, f"the fact engine refused: {exc}"]
    if fact.ok:
        sentence = fact.sentence()
        return Answer(
            question=question, provenance=provenance, structured_facts=(fact.to_contract(),),
            coverage_note=" ".join(x for x in (fact.caveat, extra_note) if x),
            rendered=f"{sentence} {extra_note}".strip() if extra_note else sentence,
        ), notes  # fmt: skip
    if fact.needs_clarification:
        return Answer(
            question=question, provenance=provenance,
            facets={"clarify": list(fact.clarify), "reason": "the record holds more than one answer"},
            coverage_note=fact.sentence(),
        ), notes  # fmt: skip
    return None, [*notes, fact.missing or "the fact engine found no value"]


async def ask(runtime: Runtime, conn, question: str, *, category: str | None = None,
              variant_key: str | None = None, use_cache: bool = True, store: bool = True,
              budget: G.Budget | None = None, run_id: str | None = None,
              session_label: str | None = None, gold_item_id: str | None = None,
              gold_split: str | None = None, gold_category: str | None = None,
              keep_question: bool = False, top_k: int = 10,
              gold_speech_ids: frozenset[str] | None = None,
              related_speech_ids: frozenset[str] | None = None,
              rewrite: bool = True) -> Asked:  # fmt: skip
    """One question in, one verified answer out, with a cache lookup, a route, and a trace row written either way.
    """
    del category  # deliberately unused: see the docstring
    started = time.perf_counter()
    budget = budget or G.Budget()
    variant = P.variant(variant_key)

    # 1. route.
    decision = await RO.route_with_model(question, runtime.client if runtime.route_with_model else None)
    filters = decision.filters.as_dict()
    provenance = runtime.provenance(decision.route, filters, variant.version)

    result = Asked(answer=None, route=decision.route, route_by=decision.by, route_why=decision.why,
                  filters=filters, category=decision.category)  # fmt: skip

    async def finish(answer, *, cache_hit: bool = False, failed: str | None = None) -> Asked:
        answer = reader_ready(answer)  # Never an empty answer, on any route
        # The one exit every answer passes is checked against the record's own dates, so nothing dated inside the sealed
        # window reaches a reader whatever an inner layer did.
        answer, withheld = await SG.guard(conn, answer)
        result.sealed_withheld = result.sealed_withheld or withheld
        result.answer = answer
        result.cache_hit = cache_hit
        result.failed_reason = failed
        # A generator that was never reached is a failed request, not a decline the model made.
        result.failed_reason = request_failure(result) or None
        result.total_ms = int((time.perf_counter() - started) * 1000)
        generated = result.generated
        trace = TR.Trace(
            query_hash=question_hash(question, filters, decision.route),
            route=decision.route, route_by=decision.by, route_why=decision.why[:400],
            resolved_filters=filters,
            corpus_version=sealed.served_label(CORPUS_VERSION), retrieval_config_hash=runtime.frozen.config_hash,
            recipe=runtime.frozen.recipe, prompt_version=variant.version, model_id=runtime.model_id,
            verifier=runtime.verifier_name, verifier_precision=runtime.verifier_precision or None,
            outcome=result.outcome,
            abstention_reason=(answer.abstention.value if answer is not None and answer.abstention else None),
            cache_hit=cache_hit,
            degraded=bool(generated and generated.degraded),
            degraded_reason=((generated.degraded_reason or None) if generated and generated.degraded else None),
            failed_reason=result.failed_reason,  # the request's own failure, or the generator's
            question=question if keep_question else None,
            retrieval_hits=result.retrieval_hits,
            cards=result.cards, card_sentences=result.card_sentences, cards_dropped=result.cards_dropped,
            generation_ms=(generated.generation_ms or None) if generated else None,
            verify_ms=(generated.verify_ms or None) if generated else None,
            gold_evidence_retrieved=result.gold_evidence_retrieved,
            gold_evidence_rank=result.gold_evidence_rank,
            # the retrieval route's counts, or an overview's summed — named one by one, so the structural test that
            # every trace column has a writer can read them
            claims_generated=result.claim_counts["claims_generated"],
            claims_rejected=result.claim_counts["claims_rejected"],
            claims_verified=result.claim_counts["claims_verified"],
            claims_dropped=result.claim_counts["claims_dropped"],
            repairs=(1 if generated and generated.repaired else 0),
            rewrites=sum(1 for a in result.attempts if getattr(a, "rewritten_from", None)),
            retrieval_ms=result.retrieval_ms or None, total_ms=result.total_ms,
            prompt_tokens=(generated.actual_prompt_tokens if generated else None),
            completion_tokens=(generated.actual_completion_tokens if generated else None),
            # 's prompt-cache saving.
            cached_tokens=(generated.actual_cached_tokens if generated else None),
            served_model=(generated.served_model if generated else None),
            fingerprint=(generated.fingerprint if generated else None),
            cost_usd=round(budget.spent_usd, 6), llm_calls=budget.calls,
            run_id=run_id, session_label=session_label, gold_item_id=gold_item_id,
            gold_split=gold_split, gold_category=gold_category, is_evaluation=bool(gold_item_id),
        )
        result.trace_id = await TR.record(conn, trace)
        # Commit per question, and roll back a poisoned transaction first.
        try:
            if conn.info.transaction_status == _psycopg.pq.TransactionStatus.INERROR:
                await conn.rollback()
            else:
                await conn.commit()
        except Exception as exc:  # noqa: BLE001 - never let bookkeeping sink the answer
            log.warning("could not close the transaction (%s)", type(exc).__name__)
        runtime.spend_usd += budget.spent_usd
        runtime.calls += budget.calls
        return result

    # 1b.
    if SG.question_is_sealed(decision.filters):
        return await finish(SG.scope_answer(question, provenance))

    # 2. the cache.
    if use_cache:
        try:
            key = K.Key(
                query_hash=question_hash(question, filters, decision.route),
                corpus_version=sealed.served_label(CORPUS_VERSION), retrieval_config_hash=runtime.frozen.config_hash,
                recipe=runtime.frozen.recipe, prompt_version=variant.version, model_id=runtime.model_id,
            )
            hit = await K.lookup(conn, key)
            if hit and hit.get("answer_json"):
                # psycopg hands back `jsonb` already decoded, as a dict — `model_validate_json` wants text and rejected
                # it, so the row was found, its hit count incremented, and the answer thrown away.
                payload = hit["answer_json"]
                cached = (Answer.model_validate_json(payload) if isinstance(payload, (str, bytes, bytearray))
                          else Answer.model_validate(payload))  # fmt: skip
                return await finish(cached, cache_hit=True)
        except Exception as exc:  # noqa: BLE001 - a broken cache must never stop an answer
            log.warning("cache lookup failed (%s: %s); answering fresh", type(exc).__name__,
                        " ".join(str(exc).split())[:160])  # fmt: skip

    # 3. the clarify routes, which need neither retrieval nor a model.
    if decision.route == "clarify":
        return await finish(clarify_answer(question, provenance, why=decision.why, clarify=decision.clarify))
    if decision.route == "abstain":
        return await finish(Answer(question=question, provenance=provenance,
                                  abstention=Abstention.NOT_IN_RECORD, abstention_detail=decision.why))  # fmt: skip

    try:
        # 4. the structured route: E2's fact engine, no model at all.
        if decision.route == "structured" and decision.category:
            answered, notes = await structured_answer(conn, question, decision, provenance)
            if answered is not None:
                return await finish(answered)
            # Unresolved parameters are not a failure to answer, they are a failure to *identify* what was asked.
            log.info("structured route fell back to retrieval: %s", "; ".join(n for n in notes if n))

        # 5. the set route: SQL only, no model at all.
        if decision.route == "set":
            set_result = await RT.set_route(conn, question, provenance, filters=decision.filters)
            result.set_result = set_result
            return await finish(set_result.answer)

        # 5b.
        if decision.route in ("retrieval", "overview"):
            clarified = await common_phrase_clarification(conn, question, provenance)
            if clarified is not None:
                return await finish(clarified)

        # 6. everything else retrieves.
        retrieval_started = time.perf_counter()
        # The question's own date scopes the search (see `retrieval_scope` for the measurement).
        scope = retrieval_scope(decision.filters)
        # The deck scales with the scope.
        pool, budget_tokens = deck_size(decision.route, dated=bool(scope), top_k=top_k)
        deck, search_result = await search_deck(runtime, question, pool, token_budget=budget_tokens,
                                                 filters=scope or None)  # fmt: skip
        if scope and len(search_result.hits) < WIDEN_BELOW:
            # Too narrow — the date may be the topic ("the events of 7 July 2005") rather than when it was said, or fall
            # outside every window.
            pool, budget_tokens = deck_size(decision.route, dated=False, top_k=top_k)
            deck, search_result = await search_deck(runtime, question, pool, token_budget=budget_tokens)
            result.retrieval_widened = True
        result.retrieval_scope = {k: str(v) for k, v in scope.as_dict().items()} if scope else {}
        result.retrieval_hits = len(search_result.hits)
        result.cards = len(deck.cards)
        result.cards_dropped = deck.dropped
        # Everything the model READS, not what it may cite.
        result.card_sentences = sum(len(c.sentences) for c in deck.cards)
        result.cards_citable = len(deck.chunk_ids)
        if gold_speech_ids or related_speech_ids:
            # Scored HERE, on the deck the generator is about to be shown, not on the search result: cards are what it
            # sees, and a hit dropped by the card budget was not given to it however well it ranked.
            shown = [str(getattr(card, "speech_id", "")) for card in deck.cards]
            if gold_speech_ids:
                result.gold_evidence_retrieved = False
                for position, speech_id in enumerate(shown, start=1):
                    if speech_id in gold_speech_ids:
                        result.gold_evidence_retrieved = True
                        result.gold_evidence_rank = position
                        break
            if related_speech_ids:
                result.related_evidence_retrieved = any(s in related_speech_ids for s in shown)
        result.retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        if decision.route == "overview":
            overview = await RT.overview_route(
                conn, question, list(search_result.hits), runtime.client, provenance,
                score_batch=runtime.score_batch(), budget=budget,
                verifier_name=runtime.verifier_name, precision=runtime.verifier_precision,
                variant_key=variant.key,
            )  # fmt: skip
            result.overview = overview
            answer = overview.answer
        else:
            # 6. retrieval, multi_hop, timeline and structured all generate from cards, with the corrective loop.
            async def do_search(query: str):
                if query == question:
                    return deck
                built, found = await search_deck(runtime, query, pool, filters=scope or None)
                if scope and len(found.hits) < WIDEN_BELOW:
                    built, _ = await search_deck(runtime, query, pool)
                return built

            async def do_answer(query: str, built):
                return await G.answer_question(
                    query, built, runtime.client, provenance, variant_key=variant.key,
                    score_batch=runtime.score_batch(), budget=budget,
                    verifier_name=runtime.verifier_name, precision=runtime.verifier_precision,
                )  # fmt: skip

            if rewrite:
                corrected = await RT.with_corrections(do_search, do_answer, question, budget=budget)
                result.generated, result.attempts = corrected.generated, corrected.attempts
            else:
                result.generated = await do_answer(question, deck)
            # A verified answer none of whose claims is about the question is a near miss, not an answer.
            answer = await relevance_gate(runtime, question, result.generated.answer, record=result)
    except Exception as exc:  # noqa: BLE001 - a failed request is traced as an error, never as an answer
        log.warning("answering failed (%s: %s)", type(exc).__name__, " ".join(str(exc).split())[:200])
        # Roll back before tracing: a database error leaves the transaction unusable, and the trace row is written on
        # the same connection.
        try:
            await conn.rollback()
        except Exception as rollback_exc:  # noqa: BLE001 - the connection may be gone; say so, then trace anyway
            log.warning("could not roll back after a failed request (%s); the trace write may fail too",
                        type(rollback_exc).__name__)
        return await finish(
            Answer(question=question, provenance=provenance, abstention=Abstention.INSUFFICIENT_EVIDENCE,
                   abstention_detail=f"the request failed ({type(exc).__name__})"),
            failed=f"{type(exc).__name__}: {' '.join(str(exc).split())[:160]}",
        )  # fmt: skip

    # 7. store, but only what is fit to be reused — and as a reader will be shown it.
    answer = reader_ready(answer)
    # Guarded BEFORE it can be stored.
    answer, withheld = await SG.guard(conn, answer)
    result.sealed_withheld = withheld
    if store and answer is not None and not request_failure(result) and not withheld:
        try:
            await K.store(
                conn, answer, filters=filters, verifier=runtime.verifier_name or "deterministic",
                verifier_precision=runtime.verifier_precision,
                claims_dropped=(result.claim_counts["claims_dropped"] or 0),
                degraded=bool(result.generated and result.generated.degraded),
                total_ms=int((time.perf_counter() - started) * 1000),
                retrieval_ms=result.retrieval_ms or None,
                prompt_tokens=(result.generated.actual_prompt_tokens if result.generated else None),
                cost_usd=round(budget.spent_usd, 6),
                gold_item_id=gold_item_id, gold_split=gold_split, gold_category=gold_category,
                run_id=run_id, session_label=session_label, is_evaluation=bool(gold_item_id),
            )  # fmt: skip
        except K.CacheError as exc:
            # A refusal to cache is not a failure to answer.
            log.info("not cached: %s", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache write failed (%s: %s)", type(exc).__name__,
                        " ".join(str(exc).split())[:160])  # fmt: skip

    return await finish(answer)

