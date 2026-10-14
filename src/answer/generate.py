"""Answer generation and the verifier loop."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from answer import cards as C
from answer import prompts as P
from answer.contract import Abstention, Answer, Citation, Claim, Provenance, quote_matches
from answer.verify import Evidence, Scope, verified_answer

log = logging.getLogger("answer.generate")

# How many tokens a single answer may spend, prompt plus completion, across every call it makes.
TOKEN_BUDGET = 24_000
# 's wall clock.
WALL_CLOCK_SECONDS = 15.0
# 's ceiling on corrective turns.
MAX_CORRECTIONS = 2
# Completion ceiling per call.
MAX_COMPLETION_TOKENS = 2_400

# The sufficiency thresholds.
MIN_CARDS = 1
MIN_TOP_SCORE = 0.0
# Below this many matched sentences the evidence may still be enough, but the answer must say what is missing.
THIN_EVIDENCE_SENTENCES = 3

# The cross-encoder score at or below which the deck's BEST passage is scored as actively non-relevant, so the answer
# carries a caveat saying so.
WEAK_RERANK_SCORE = -2.0
# Declining every weak match before generation removed two harmful near misses but fired on 22 dev decks and cost seven
# partial answers, leaving over-abstention at 16.8%.
WEAK_MATCH_DECLINES = False


class GenerationError(RuntimeError):
    """The generator could not run at all — no client, no cards. Not the same as an answer that abstained."""


# What an unpriceable call is charged per token: well above any rate in the table, so a budget that cannot price a call
# stops early rather than overspends blind. $10 per million tokens.
_PESSIMISTIC_USD_PER_TOKEN = 10.0 / 1_000_000


@dataclass
class Budget:
    """What one answer may spend. Checked before every call, never after."""

    tokens: int = TOKEN_BUDGET
    seconds: float = WALL_CLOCK_SECONDS
    corrections: int = MAX_CORRECTIONS
    usd: float | None = None

    started: float = field(default_factory=time.perf_counter)
    spent_tokens: int = 0
    # Calls the budget could not price, charged pessimistically.
    unpriced_calls: int = 0
    spent_usd: float = 0.0
    used_corrections: int = 0
    calls: int = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.seconds - self.elapsed)

    def exhausted(self) -> str:
        """Why no further call may be made, or "" when one may. A string, so the abstention can quote it."""
        if self.elapsed >= self.seconds:
            return f"wall-clock budget of {self.seconds:.0f}s exhausted after {self.elapsed:.1f}s"
        if self.spent_tokens >= self.tokens:
            return f"token budget of {self.tokens:,} exhausted at {self.spent_tokens:,}"
        if self.usd is not None and self.spent_usd >= self.usd:
            return f"spend budget of ${self.usd:.2f} exhausted at ${self.spent_usd:.4f}"
        return ""

    def record(self, reply: Any) -> None:
        self.calls += 1
        self.spent_tokens += getattr(reply, "total_tokens", 0)
        try:
            self.spent_usd += reply.cost_usd()
        except Exception as exc:  # noqa: BLE001 - an unpriceable model must not fail the answer...
            # ...but it must not count as $0 either.
            self.unpriced_calls += 1
            self.spent_usd += _PESSIMISTIC_USD_PER_TOKEN * getattr(reply, "total_tokens", 0)
            log.warning("could not price a model call (%s); charged pessimistically", type(exc).__name__)


@dataclass
class Sufficiency:
    """Whether to try at all, and what to say if not."""

    enough: bool
    reason: Abstention | None = None
    detail: str = ""
    thin: bool = False
    cards: int = 0
    top_score: float = 0.0
    # The best cross-encoder score in the deck.
    top_rerank: float | None = None
    # The deck's best passage was scored as non-relevant by the cross-encoder.
    weak_match: bool = False


def sufficiency(deck: C.Deck) -> Sufficiency:
    """The gate before any spend."""
    if not deck.cards:
        return Sufficiency(False, Abstention.NOT_IN_RECORD,
                           "nothing in the record matched this question")  # fmt: skip
    top = max((card.score for card in deck.cards), default=0.0)
    if len(deck.cards) < MIN_CARDS or top < MIN_TOP_SCORE:
        return Sufficiency(False, Abstention.INSUFFICIENT_EVIDENCE,
                           f"only {len(deck.cards)} passage(s) matched, and the best scored {top:.3f}",
                           cards=len(deck.cards), top_score=top)  # fmt: skip
    matched = sum(len(card.matched_chunk_ids) for card in deck.cards)
    scores = [card.rerank_score for card in deck.cards if card.rerank_score is not None]
    best_rerank = max(scores) if scores else None
    weak = best_rerank is not None and best_rerank <= WEAK_RERANK_SCORE
    if weak and WEAK_MATCH_DECLINES:
        return Sufficiency(False, Abstention.INSUFFICIENT_EVIDENCE,
                           f"the best passage found scored {best_rerank:.2f} for relevance to this question, below "
                           f"{WEAK_RERANK_SCORE}", cards=len(deck.cards), top_score=top, top_rerank=best_rerank,
                           weak_match=True)  # fmt: skip
    return Sufficiency(True, thin=matched < THIN_EVIDENCE_SENTENCES or weak, cards=len(deck.cards),
                       top_score=top, top_rerank=best_rerank, weak_match=weak)  # fmt: skip


# --- parsing the model's reply ---------------------------------------------------------------------------


@dataclass
class Rejected:
    """A claim that never became a claim, and why. Counted, so a prompt that produces unusable output is visible as a
    number rather than as a quiet shortfall in answers.
    """

    index: int
    why: str
    raw: dict[str, Any] = field(default_factory=dict)


def parse_reply(text: str, deck: C.Deck) -> tuple[list[Claim], list[Rejected], dict[str, Any]]:
    """Turn the model's JSON into contract claims, rejecting anything that will not validate."""
    rejected: list[Rejected] = []
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        return [], [Rejected(-1, f"the reply was not JSON ({type(exc).__name__})")], {}
    if not isinstance(data, dict):
        return [], [Rejected(-1, f"the reply was JSON but not an object ({type(data).__name__})")], {}

    if data.get("answer") is False or data.get("abstention"):
        return [], [], {"abstention": data.get("abstention"), "why": data.get("why", ""),
                        "coverage_note": data.get("coverage_note", "")}  # fmt: skip

    by_speech = {card.speech_id: card for card in deck.cards}
    claims: list[Claim] = []
    for index, raw in enumerate(data.get("claims") or []):
        if not isinstance(raw, dict):
            rejected.append(Rejected(index, "claim was not an object"))
            continue
        speech_id = str(raw.get("speech_id") or "").strip()
        card = by_speech.get(speech_id)
        if card is None:
            # A speech id that was never shown is the citation-fabrication case, caught before the contract even sees
            # it, because without a card there is no evidence to check the quote against.
            rejected.append(Rejected(index, f"cited speech {speech_id!r} was not in the evidence", raw))
            continue
        ids = raw.get("chunk_ids") or []
        if not isinstance(ids, list):
            ids = [ids]
        try:
            chunk_ids = tuple(dict.fromkeys(int(i) for i in ids))
        except (TypeError, ValueError):
            rejected.append(Rejected(index, "chunk_ids were not integers", raw))
            continue
        outside = [i for i in chunk_ids if i not in deck.chunk_ids]
        if not chunk_ids or outside:
            rejected.append(Rejected(index, f"chunk_ids {outside or 'missing'} were not retrieved", raw))
            continue
        quote = str(raw.get("quote") or "").strip()
        # Named claim_text, not text: `text` is this function's own parameter (the whole reply), and shadowing it inside
        # the loop worked only because the JSON is parsed before the loop starts.
        claim_text = str(raw.get("text") or "").strip()
        # A `said` claim must contain its own quote, or it is a `summary`.
        kind = "said" if str(raw.get("kind", "summary")).lower() == "said" else "summary"
        if kind == "said" and quote and not quote_matches(quote, claim_text):
            kind = "summary"
        try:
            claims.append(Claim(
                id=f"c{index + 1}",
                kind=kind,
                text=claim_text,
                speaker_name=raw.get("speaker_name") or card.speaker,
                date=card.date, house=card.house, section=card.section,
                citations=(Citation(speech_id=speech_id, chunk_ids=chunk_ids,
                                    quote=quote,
                                    hansard_url=card.url),),
            ))  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - a validation error is a rejected claim, not a failed answer
            rejected.append(Rejected(index, _why(exc), raw))
    return claims, rejected, {"coverage_note": data.get("coverage_note", "")}


def _why(exc: Exception) -> str:
    """The actual reason a claim would not validate, not the header above it."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            messages = []
            for error in errors():
                where = ".".join(str(p) for p in error.get("loc", ()) if p != "__root__")
                message = str(error.get("msg", "")).removeprefix("Value error, ").strip()
                messages.append(f"{where}: {message}" if where else message)
            if messages:
                return "; ".join(messages)[:400]
        except Exception:  # noqa: BLE001 - fall through to the string form
            pass
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"


def evidence_from_deck(deck: C.Deck) -> dict[str, Evidence]:
    """What the verifier compares claims against: the cards themselves, which are the record's own words."""
    return {
        card.speech_id: Evidence(
            speech_id=card.speech_id, text=card.text, speaker_name=card.speaker,
            date=card.date, house=card.house,
            chunk_ids=frozenset(i for i in card.chunk_ids if i >= 0),
        )
        for card in deck.cards
    }


# --- the extractive fallback -----------------------------------------------------------------------------


def extractive(deck: C.Deck, limit: int = 3, prefer: set[int] | None = None) -> list[Claim]:
    """Verified quotes with attribution, built without a model at all."""
    preferred = prefer or set()

    def order(card: C.Card) -> tuple:
        return (0 if preferred & set(card.chunk_ids) else 1, card.rank)

    out: list[Claim] = []
    for card in sorted(deck.cards, key=order):
        for sentence in sorted(card.sentences, key=lambda s: 0 if s.chunk_id in preferred else 1):
            if sentence.chunk_id < 0:
                continue
            if not sentence.matched and sentence.chunk_id not in preferred:
                continue
            text = sentence.text.strip()
            if not text:
                continue
            try:
                out.append(Claim(
                    id=f"x{len(out) + 1}", kind="said", text=text,
                    speaker_name=card.speaker, date=card.date, house=card.house, section=card.section,
                    citations=(Citation(speech_id=card.speech_id, chunk_ids=(sentence.chunk_id,),
                                        quote=text, hansard_url=card.url),),
                ))  # fmt: skip
            except Exception:  # noqa: BLE001 - a sentence the contract refuses (two sentences in one chunk) is skipped
                continue
            if len(out) >= limit:
                return out
    return out


# --- the flow --------------------------------------------------------------------------------------------


@dataclass
class Generated:
    """An answer and an account of how it was produced. The account is not decoration:'s trace row is built from it,
    and "how many claims did verification drop" is one of the numbers that tells us whether the prompt is working.
    """

    answer: Answer
    variant: str = ""
    route: str = "retrieval"
    sufficiency: Sufficiency | None = None
    claims_generated: int = 0
    claims_rejected: int = 0
    claims_verified: int = 0
    claims_dropped: int = 0
    repaired: bool = False
    fell_back: bool = False
    # (claim text, why) for every claim verification refused.
    drops: list[tuple[str, str]] = field(default_factory=list)
    degraded: bool = False
    # WHY it was degraded (the verifier's own words, e.g.
    degraded_reason: str = ""
    # The provider failed — a refused key, an outage, a timeout — and the answer below is a decline the model never
    # made.
    generator_failed: str = ""
    # The model's own "what the evidence does not cover" note.
    model_note: str = ""
    # The model's own reason for declining.
    model_why: str = ""
    budget: Budget | None = None
    rejections: list[Rejected] = field(default_factory=list)
    served_model: str = ""
    fingerprint: str = ""
    estimated_prompt_tokens: int = 0
    actual_prompt_tokens: int = 0
    # What the model wrote, as the provider counted it.
    actual_completion_tokens: int = 0
    # Prompt tokens the provider served from its own cache, billed at a tenth of the input rate. asks for this saving to
    # be recorded, and it is the only way to know whether keeping the system prompt byte-stable bought anything.
    actual_cached_tokens: int = 0
    # Stage latencies. asks for them and the trace had the columns with nothing writing them, so both read NULL across
    # every row of every run — found by checking whether each column varies, not by reading one.
    generation_ms: int = 0
    verify_ms: int = 0

    @property
    def token_estimate_error(self) -> float:
        """How wrong `cards.estimate_tokens` was, as a signed fraction. Recorded rather than assumed: the budget is
        built on that estimate, so its error is the budget's error.
        """
        if not self.actual_prompt_tokens:
            return 0.0
        return (self.estimated_prompt_tokens - self.actual_prompt_tokens) / self.actual_prompt_tokens


def _ms(since: float) -> int:
    """Wall-clock milliseconds since `since`. asks for stage latencies; `trace.liveness` reported both of them DEAD on
    the run after I added the fields and wired the trace row, because nothing assigned them.
    """
    return int((time.perf_counter() - since) * 1000)


# F5: what a READER is told when the model declines for a reason that asserts absence.
READER_REASONS: dict[Abstention, str] = {
    Abstention.NOT_IN_RECORD: "The passages found in the parliamentary record do not answer this question.",
    Abstention.INSUFFICIENT_EVIDENCE: ("The passages found touch on this, but they do not contain enough to answer it "
                                       "reliably."),
    Abstention.OUTSIDE_WINDOW: "The question falls outside the periods this record covers.",
}


def reader_reason(reason: Abstention, model_why: str) -> str:
    return READER_REASONS.get(reason, model_why)


def _abstain(question: str, provenance: Provenance, reason: Abstention, detail: str,
             note: str = "") -> Answer:  # fmt: skip
    return Answer(question=question, provenance=provenance, abstention=reason,
                  abstention_detail=detail, coverage_note=note)  # fmt: skip


def _failure_lines(dropped: tuple[tuple[Claim, str], ...], rejected: list[Rejected]) -> str:
    lines = [f'- claim "{claim.text[:110]}" — {why}' for claim, why in dropped[:6]]
    lines += [f"- a claim was unusable — {r.why}" for r in rejected[:4]]
    return "\n".join(lines) or "- no claim survived checking"


async def answer_question(question: str, deck: C.Deck, client: Any, provenance: Provenance,
                          *, variant_key: str | None = None, facts: list[dict] | None = None,
                          scope: Scope | None = None, score_batch: Any = None,
                          budget: Budget | None = None, verifier_name: str = "deterministic",
                          precision: str = "") -> Generated:  # fmt: skip
    """One question, one answer, with at most one repair. Never returns unverified text."""
    variant = P.variant(variant_key)
    budget = budget or Budget()
    provenance = provenance.model_copy(update={"prompt_version": variant.version})

    enough = sufficiency(deck)
    if not enough.enough:
        return Generated(
            answer=_abstain(question, provenance, enough.reason or Abstention.INSUFFICIENT_EVIDENCE,
                            enough.detail),
            variant=variant.key, sufficiency=enough, budget=budget,
        )  # fmt: skip

    evidence = evidence_from_deck(deck)
    fact_block = C.render_structured_facts(facts)
    rendered_cards = deck.render(lean=variant.lean_cards)
    system = variant.system()
    user = variant.user(question, rendered_cards, fact_block)
    estimated = C.estimate_tokens(system + user)

    blocked = budget.exhausted()
    if blocked:
        return Generated(
            answer=_abstain(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                            f"stopped before generating: {blocked}"),
            variant=variant.key, sufficiency=enough, budget=budget,
        )  # fmt: skip

    generation_ms = 0
    verify_ms = 0
    call_started = time.perf_counter()
    try:
        reply = await client.chat(user, system=system, max_tokens=MAX_COMPLETION_TOKENS,
                                 temperature=0.0, json_object=True)  # fmt: skip
        generation_ms += _ms(call_started)
    except Exception as exc:  # noqa: BLE001 - a provider failure abstains; it never invents an answer
        # The message, not just the class.
        usage = getattr(exc, "usage", None)
        if usage:
            budget.calls += 1
            budget.spent_tokens += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
            from goldset.llm import Reply as _Reply

            try:
                budget.spent_usd += _Reply(
                    content="", model=str(usage.get("model") or ""),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                ).cost_usd()  # fmt: skip
            except Exception as price_exc:  # noqa: BLE001 - an unpriceable failure must not mask the real error
                budget.unpriced_calls += 1
                log.warning("could not price a failed call (%s)", type(price_exc).__name__)
        detail = " ".join(str(exc).split())[:200] or type(exc).__name__
        log.warning("generation failed: %s: %s", type(exc).__name__, detail)
        return Generated(
            answer=_abstain(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                            f"the generator could not be reached ({detail})"),
            variant=variant.key, sufficiency=enough, budget=budget, degraded=True,
            degraded_reason=f"the generator could not be reached ({detail})",
            generator_failed=f"{type(exc).__name__}: {detail}",
        )  # fmt: skip
    budget.record(reply)

    claims, rejected, extra = parse_reply(reply.content, deck)
    if extra.get("abstention"):
        try:
            reason = Abstention(str(extra["abstention"]))
        except ValueError:
            reason = Abstention.INSUFFICIENT_EVIDENCE
        return Generated(
            # The reason for declining stays (it is what corrects a false premise); the model's coverage note does not
            # reach the reader here either — see the answered path below.
            answer=_abstain(question, provenance, reason,
                            reader_reason(reason, str(extra.get("why") or "")[:500])),
            model_why=str(extra.get("why") or "")[:500],
            model_note=str(extra.get("coverage_note") or ""),
            variant=variant.key, sufficiency=enough, budget=budget, claims_rejected=len(rejected),
            rejections=rejected, served_model=reply.model, fingerprint=reply.fingerprint,
            estimated_prompt_tokens=estimated, actual_prompt_tokens=reply.prompt_tokens,
            actual_completion_tokens=reply.completion_tokens,
            actual_cached_tokens=reply.cached_tokens,
        )  # fmt: skip

    # The model's own coverage note is never shown.
    model_note = str(extra.get("coverage_note") or "")
    note = ""
    if enough.weak_match:
        # A different thing from "few passages": the search DID return passages and the cross-encoder scored the best of
        # them as not relevant to the question.
        note = (note + " The closest passages found were only weakly related to this question, "
                       "so this answer may not be what you meant.").strip()
    elif enough.thin:
        note = (note + " Only a few passages matched this question, so the answer may be partial.").strip()

    draft = Answer(question=question, provenance=provenance, claims=tuple(claims),
                   retrieved_chunk_ids=deck.chunk_ids, coverage_note=note) if claims else None  # fmt: skip
    outcome = None
    if draft is not None:
        verify_started = time.perf_counter()
        checked, outcome = await verified_answer(draft, evidence, scope, score_batch,
                                                verifier_name, precision)  # fmt: skip
        verify_ms += _ms(verify_started)
    else:
        checked = None

    result = Generated(
        answer=checked or _abstain(question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
                                   "the generator produced no usable claim", note),
        variant=variant.key, sufficiency=enough, budget=budget, model_note=model_note,
        generation_ms=generation_ms, verify_ms=verify_ms,
        claims_generated=len(claims), claims_rejected=len(rejected), rejections=rejected,
        claims_verified=len(outcome.verified) if outcome else 0,
        claims_dropped=len(outcome.dropped) if outcome else 0,
        drops=[(c.text[:160], why) for c, why in (outcome.dropped if outcome else ())],
        degraded=bool(outcome and outcome.degraded),
        degraded_reason=(outcome.degraded_reason if outcome and outcome.degraded else ""), served_model=reply.model,
        fingerprint=reply.fingerprint, estimated_prompt_tokens=estimated,
        actual_prompt_tokens=reply.prompt_tokens,
        actual_completion_tokens=reply.completion_tokens,
        actual_cached_tokens=reply.cached_tokens,
    )  # fmt: skip

    # --- the single repair ------------------------------------------------------------------------------
    lost_everything = not (outcome and outcome.verified)
    if lost_everything and budget.used_corrections < budget.corrections and not budget.exhausted():
        budget.used_corrections += 1
        failures = _failure_lines(outcome.dropped if outcome else (), rejected)
        try:
            repair_started = time.perf_counter()
            repair = await client.chat(
                user + "\n\n" + P.REPAIR_INSTRUCTION.format(failures=failures),
                system=system, max_tokens=MAX_COMPLETION_TOKENS, temperature=0.0, json_object=True,
            )
            budget.record(repair)
            fixed, refused, extra2 = parse_reply(repair.content, deck)
            result.repaired = True
            result.claims_rejected += len(refused)
            if extra2.get("abstention"):
                try:
                    reason = Abstention(str(extra2["abstention"]))
                except ValueError:
                    reason = Abstention.INSUFFICIENT_EVIDENCE
                result.answer = _abstain(question, provenance, reason,
                                         str(extra2.get("why") or "")[:500], note)  # fmt: skip
                return result
            if fixed:
                draft2 = Answer(question=question, provenance=provenance, claims=tuple(fixed),
                                retrieved_chunk_ids=deck.chunk_ids, coverage_note=note)  # fmt: skip
                result.generation_ms += _ms(repair_started)
                verify2_started = time.perf_counter()
                checked2, outcome2 = await verified_answer(draft2, evidence, scope, score_batch,
                                                           verifier_name, precision)  # fmt: skip
                result.verify_ms += _ms(verify2_started)
                result.claims_generated += len(fixed)
                result.claims_verified = len(outcome2.verified)
                result.claims_dropped += len(outcome2.dropped)
                result.drops += [(c.text[:160], why) for c, why in outcome2.dropped]
                result.degraded = result.degraded or outcome2.degraded
                if outcome2.degraded and not result.degraded_reason:
                    result.degraded_reason = outcome2.degraded_reason
                if outcome2.verified:
                    result.answer = checked2
                    return result
        except Exception as exc:  # noqa: BLE001 - a failed repair falls through to the fallback
            log.warning("repair failed: %s", type(exc).__name__)

    # --- extractive fallback, then abstention -----------------------------------------------------------
    if not (outcome and outcome.verified):
        # The sentences the model cited, even from claims that were rejected: its choice of line is signal about
        # relevance even when its paraphrase failed.
        cited = {i for claim, _ in (outcome.dropped if outcome else ()) for i in claim.cited_chunk_ids}
        cited |= {i for r in rejected for i in (r.raw.get("chunk_ids") or []) if isinstance(i, int)}
        quotes = extractive(deck, prefer=cited)
        if quotes:
            draft3 = Answer(question=question, provenance=provenance, claims=tuple(quotes),
                            retrieved_chunk_ids=deck.chunk_ids,
                            coverage_note=(note + " No summary could be verified, so the most relevant passages "
                                           "are quoted directly.").strip())  # fmt: skip
            verify3_started = time.perf_counter()
            checked3, outcome3 = await verified_answer(draft3, evidence, scope, score_batch,
                                                       verifier_name, precision)  # fmt: skip
            result.verify_ms += _ms(verify3_started)
            if outcome3.verified:
                result.answer = checked3
                result.fell_back = True
                result.claims_verified = len(outcome3.verified)
                return result
        result.answer = _abstain(
            question, provenance, Abstention.INSUFFICIENT_EVIDENCE,
            "no claim could be verified against the record" + (f"; {budget.exhausted()}" if budget.exhausted() else ""),
            note,
        )  # fmt: skip
    return result
