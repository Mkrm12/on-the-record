"""The verifier: nothing reaches a reader unverified."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

from answer.contract import (REPORTING_VERBS, Abstention, Answer, Claim, Verification, missing_numbers,
                             quote_matches, speaker_lead)

log = logging.getLogger("answer.verify")

# MiniCheck's support probability above which a paraphrase is accepted.
SUPPORT_THRESHOLD = 0.5

# Claim kinds that may be shown on the deterministic checks alone when no entailment model is available.
DEGRADABLE_KINDS = frozenset({"said"})


class EvidenceMissing(RuntimeError):
    """The verifier was asked to check a claim against text it was not given."""


@dataclass(frozen=True)
class Evidence:
    """The record's own version of one cited speech. Everything here comes from SQL."""

    speech_id: str
    text: str
    speaker_name: str | None = None
    mnis_id: int | None = None
    date: Date | None = None
    house: str | None = None
    chunk_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass
class Scope:
    """What the answer claims to cover. A claim dated outside it is out of scope however true it is."""

    first_date: Date | None = None
    last_date: Date | None = None
    houses: frozenset[str] = field(default_factory=frozenset)

    def contains(self, when: Date | None, house: str | None) -> bool:
        if when is not None:
            if self.first_date and when < self.first_date:
                return False
            if self.last_date and when > self.last_date:
                return False
        if house and self.houses and house not in self.houses:
            return False
        return True


# --- the five deterministic checks ----------------------------------------------------------------------


def check_citations(claim: Claim, retrieved: frozenset[int] | set[int]) -> tuple[bool, dict]:
    """Every cited chunk was retrieved for this question."""
    if not retrieved:
        return (False, {"why": "nothing was retrieved, so no citation can be confirmed"})
    outside = sorted(claim.cited_chunk_ids - set(retrieved))
    return (not outside, {"cited_but_not_retrieved": outside} if outside else {})


def check_quotes(claim: Claim, evidence: dict[str, Evidence]) -> tuple[bool, dict]:
    """Each citation's quote appears verbatim in the speech it names."""
    failures = []
    for citation in claim.citations:
        source = evidence.get(citation.speech_id)
        if source is None:
            raise EvidenceMissing(
                f"claim {claim.id} cites speech {citation.speech_id!r} but no evidence for it was supplied; "
                "marking it unsupported would hide a caller bug behind a verdict about the claim")
        if not quote_matches(citation.quote, source.text):
            failures.append({"speech_id": citation.speech_id, "quote": citation.quote[:120]})
    return (not failures, {"quotes_not_in_source": failures} if failures else {})


def check_numbers(claim: Claim, evidence: dict[str, Evidence]) -> tuple[bool, dict]:
    """Every figure the claim states appears in the text it cites."""
    sources = " ".join(evidence[c.speech_id].text for c in claim.citations if c.speech_id in evidence)
    absent = missing_numbers(claim.text, sources)
    return (not absent, {"numbers_not_in_evidence": sorted(absent)} if absent else {})


def check_attribution(claim: Claim, evidence: dict[str, Evidence]) -> tuple[bool, dict]:
    """The claim's speaker is the speaker of the speech it cites, according to SQL."""
    if claim.speaker_mnis_id is None and not claim.speaker_name:
        return (True, {"note": "the claim attributes nothing, so there is nothing to misattribute"})
    problems = []
    for citation in claim.citations:
        source = evidence.get(citation.speech_id)
        if source is None:
            raise EvidenceMissing(f"claim {claim.id} cites speech {citation.speech_id!r} with no evidence given")
        if claim.speaker_mnis_id is not None and source.mnis_id is not None:
            if claim.speaker_mnis_id != source.mnis_id:
                problems.append({"speech_id": source.speech_id, "claimed": claim.speaker_mnis_id,
                                 "record": source.mnis_id})  # fmt: skip
            continue
        # No mnis id on one side or the other: fall back to the name, and say that is what happened.
        if claim.speaker_name and source.speaker_name:
            if claim.speaker_name.strip().casefold() not in source.speaker_name.strip().casefold() \
                    and source.speaker_name.strip().casefold() not in claim.speaker_name.strip().casefold():
                problems.append({"speech_id": source.speech_id, "claimed": claim.speaker_name,
                                 "record": source.speaker_name, "matched_on": "name"})  # fmt: skip
    return (not problems, {"misattributed": problems} if problems else {})


def check_window(claim: Claim, evidence: dict[str, Evidence], scope: Scope | None) -> tuple[bool, dict]:
    """The claim sits inside the window and House the answer says it covers."""
    problems: dict[str, Any] = {}
    for citation in claim.citations:
        source = evidence.get(citation.speech_id)
        if source is None:
            raise EvidenceMissing(f"claim {claim.id} cites speech {citation.speech_id!r} with no evidence given")
        if claim.date and source.date and claim.date != source.date:
            problems.setdefault("date_disagrees_with_record", []).append(
                {"speech_id": source.speech_id, "claimed": str(claim.date), "record": str(source.date)})
        if claim.house and source.house and claim.house != source.house:
            problems.setdefault("house_disagrees_with_record", []).append(
                {"speech_id": source.speech_id, "claimed": claim.house, "record": source.house})
        if scope and not scope.contains(source.date, source.house):
            problems.setdefault("outside_the_answer_scope", []).append(
                {"speech_id": source.speech_id, "date": str(source.date), "house": source.house})
    return (not problems, problems)


DETERMINISTIC = ("citation_ok", "quote_ok", "numbers_ok", "attribution_ok", "window_ok")


def check_deterministically(claim: Claim, evidence: dict[str, Evidence],
                            retrieved: frozenset[int] | set[int],
                            scope: Scope | None = None) -> Verification:  # fmt: skip
    """All five, with their reasons. Every one runs: knowing a claim fails three checks is worth more than knowing it
    failed the first.
    """
    detail: dict[str, Any] = {}
    citation_ok, why = check_citations(claim, retrieved)
    detail.update({f"citation.{k}": v for k, v in why.items()})
    quote_ok, why = check_quotes(claim, evidence)
    detail.update({f"quote.{k}": v for k, v in why.items()})
    numbers_ok, why = check_numbers(claim, evidence)
    detail.update({f"numbers.{k}": v for k, v in why.items()})
    attribution_ok, why = check_attribution(claim, evidence)
    detail.update({f"attribution.{k}": v for k, v in why.items()})
    window_ok, why = check_window(claim, evidence, scope)
    detail.update({f"window.{k}": v for k, v in why.items()})
    return Verification(
        citation_ok=citation_ok, quote_ok=quote_ok, numbers_ok=numbers_ok,
        attribution_ok=attribution_ok, window_ok=window_ok,
        checked_by="deterministic", detail=detail,
    )  # fmt: skip


# --- entailment, and what to do when it is not there -----------------------------------------------------


# Reporting verbs a claim uses to attribute.
_REPORTING = REPORTING_VERBS


def strip_attribution(text: str, speaker: str | None) -> str:
    """The claim's substance, with its leading "X said that" removed."""
    body = text.strip()
    # Who the claim names, by `contract.speaker_lead` — the renderer's definition too: the printed name, the present
    # name of a member shown "(now …)", and the shorter forms — the surname (Hansard prints "Mr Speaker" where the
    # roster says "Sir Lindsay Hoyle"), an honorific with it, a peer's title without the designation — each only when a
    # reporting verb follows, so "May" is never stripped from "May I begin".
    lead = speaker_lead(body, speaker)
    if lead is None:
        return body
    rest = body[len(lead):].lstrip(" ,:")
    for verb in _REPORTING:
        if rest.casefold().startswith(verb):
            stripped = rest[len(verb):].lstrip(" ,:")
            # Never strip away the whole claim: if nothing substantive is left, the original is what gets checked.
            return stripped[0].upper() + stripped[1:] if len(stripped) > 1 else body
    return body


def entailment_inputs(claim: Claim, evidence: dict[str, Evidence]) -> tuple[str, str]:
    """The (evidence, claim) pair to send to MiniCheck: the full cited speeches, and the claim's substance."""
    joined = "\n".join(evidence[c.speech_id].text for c in claim.citations if c.speech_id in evidence)
    return (joined, strip_attribution(claim.text, claim.speaker_name))


@dataclass
class Outcome:
    """What the verifier concluded about a whole answer."""

    verified: tuple[Claim, ...] = ()
    dropped: tuple[tuple[Claim, str], ...] = ()
    degraded: bool = False
    degraded_reason: str = ""
    verifier: str = ""
    precision: str = ""

    @property
    def counts(self) -> dict[str, int]:
        return {"claims": len(self.verified) + len(self.dropped),
                "verified": len(self.verified), "dropped": len(self.dropped)}  # fmt: skip


async def verify_claims(claims: tuple[Claim, ...] | list[Claim], evidence: dict[str, Evidence],
                  retrieved: frozenset[int] | set[int], scope: Scope | None = None,
                  score_batch=None, verifier_name: str = "deterministic",
                  precision: str = "", threshold: float = SUPPORT_THRESHOLD) -> Outcome:  # fmt: skip
    """Check every claim, and say what happened to each."""
    checked: list[tuple[Claim, Verification]] = []
    dropped: list[tuple[Claim, str]] = []

    for claim in claims:
        verification = check_deterministically(claim, evidence, retrieved, scope)
        if not verification.deterministic_passed:
            failed = [name for name in DETERMINISTIC if getattr(verification, name) is False]
            dropped.append((claim.model_copy(update={"verification": verification}),
                            f"failed {', '.join(failed)}"))  # fmt: skip
            continue
        checked.append((claim, verification))

    degraded, degraded_reason = False, ""
    if score_batch is None:
        degraded, degraded_reason = True, "no entailment model available"
        surviving: list[Claim] = []
        for claim, verification in checked:
            if claim.kind in DEGRADABLE_KINDS:
                verification.entailment_verdict = "unavailable"
                verification.checked_by = f"{verifier_name} (deterministic only)"
                surviving.append(claim.model_copy(update={"verification": verification}))
            else:
                verification.entailment_verdict = "unavailable"
                dropped.append((claim.model_copy(update={"verification": verification}),
                                f"{claim.kind} claims need entailment, and no model was available"))  # fmt: skip
        return Outcome(tuple(surviving), tuple(dropped), degraded, degraded_reason,
                       f"{verifier_name} (deterministic only)", precision)  # fmt: skip

    pairs = [entailment_inputs(claim, evidence) for claim, _ in checked]
    try:
        scores = []
        if pairs:
            produced = score_batch([p[0] for p in pairs], [p[1] for p in pairs])
            # Awaited when the callable is a coroutine function, which the real one is: MiniCheck is an HTTP request.
            scores = list(await produced if inspect.isawaitable(produced) else produced)
    except Exception as exc:  # noqa: BLE001 - a failing model degrades; it does not pass everything
        log.warning("entailment failed (%s: %s); degrading", type(exc).__name__,
                    " ".join(str(exc).split())[:200])  # fmt: skip
        return await verify_claims(claims, evidence, retrieved, scope, None, verifier_name, precision, threshold)

    if len(scores) != len(pairs):
        raise EvidenceMissing(
            f"the entailment model returned {len(scores)} scores for {len(pairs)} claims; a mismatched batch "
            "would attach each score to the wrong claim")

    verified: list[Claim] = []
    for (claim, verification), score in zip(checked, scores, strict=True):
        verification.entailment_score = float(score)
        verification.entailment_verdict = "supported" if score >= threshold else "unsupported"
        verification.checked_by = verifier_name
        updated = claim.model_copy(update={"verification": verification})
        if verification.showable:
            verified.append(updated)
        else:
            dropped.append((updated, f"entailment {score:.4f} < {threshold}"))
    return Outcome(tuple(verified), tuple(dropped), degraded, degraded_reason, verifier_name, precision)


async def verified_answer(answer: Answer, evidence: dict[str, Evidence], scope: Scope | None = None,
                    score_batch=None, verifier_name: str = "deterministic",
                    precision: str = "") -> tuple[Answer, Outcome]:  # fmt: skip
    """An answer rebuilt from what passed, with `rendered` filled in — or an abstention if nothing passed."""
    outcome = await verify_claims(answer.claims, evidence, answer.retrieved_chunk_ids, scope,
                                  score_batch, verifier_name, precision)  # fmt: skip
    kept = outcome.verified
    has_other_content = bool(answer.structured_facts or answer.roster or answer.facets)

    # Rebuilt through the validators, not with `model_copy`.
    def rebuilt(**changes) -> Answer:
        return Answer.model_validate({**answer.model_dump(), **changes})

    if not kept and not has_other_content:
        return (rebuilt(
            claims=(), rendered="", abstention=Abstention.INSUFFICIENT_EVIDENCE,
            abstention_detail=(f"{len(outcome.dropped)} claim(s) did not pass verification: "
                               + "; ".join(reason for _, reason in outcome.dropped[:3])),
        ), outcome)  # fmt: skip

    note = answer.coverage_note
    if outcome.degraded:
        extra = f"Checked without the entailment model ({outcome.degraded_reason}); paraphrased claims were dropped."
        note = f"{note} {extra}".strip()
    staged = rebuilt(claims=[c.model_dump() for c in kept], rendered="", coverage_note=note)
    return (rebuilt(claims=[c.model_dump() for c in kept], rendered=staged.render(),
                    coverage_note=note), outcome)  # fmt: skip
