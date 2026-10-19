"""The answer-level evaluation gate."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import pipeline as PL  # noqa: E402
from answer import trace as TR  # noqa: E402
from answer.contract import normalise  # noqa: E402
from answer.generate import READER_REASONS  # noqa: E402
from app import safety  # noqa: E402
from goldset import judges as J  # noqa: E402
from goldset.llm import estimate  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from goldset.scoring import Graded, score  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402

REPORTS = ROOT / "reports"

# 's cap.
SPEND_CAP_USD = 3.00

# Outcomes, in the vocabulary `goldset.scoring` already validates against.
OUTCOMES = ("correct", "correct_missing_okay_nuggets", "missing_vital_nugget", "correct_abstention",
            "over_abstention", "wrong_but_benign", "harmful")  # fmt: skip


# --- deterministic grading ------------------------------------------------------------------------------


def forbidden_hits(item: Any, answer: Any) -> list[str]:
    """Forbidden claims present in the answer's text, matched on the contract's own normalisation."""
    text = normalise(answer.rendered or "")
    if not text:
        return []
    found = []
    for claim in (item.forbidden_claims or ()):
        core = normalise(str(claim)).rstrip(".").removeprefix("that ")
        if core and core in text:
            found.append(str(claim))
    return found


# The grade at which a pooled judgement means "this speech answers the question" rather than "this speech is related to
# it".
ANSWER_BEARING_GRADE = 2


def gold_speech_ids(item: Any, *, minimum_grade: int = ANSWER_BEARING_GRADE) -> frozenset[str]:
    """The speeches a human marked as ANSWERING this item, at or above `minimum_grade`."""
    found = {e.speech_id for e in (item.evidence or ()) if getattr(e, "speech_id", None)}
    for speech_id, grade in (item.graded_relevance or {}).items():
        try:
            if int(grade) >= minimum_grade:
                found.add(speech_id)
        except (TypeError, ValueError):
            continue
    return frozenset(found)


def related_speech_ids(item: Any) -> frozenset[str]:
    """Everything the pool called related at all, grade >= 1. Bounds the other side of the same question: a deck
    holding only grade-1 material makes an abstention closer to honest than to cowardly.
    """
    return gold_speech_ids(item, minimum_grade=1)


def router_accuracy(rows: list[dict]) -> dict[str, Any]:
    """'s router accuracy, from the question's words alone: a fact question must reach its EXACT fact, and anything
    else must never be sent to one. `division_debate` has no gold category of its own (its items are
    `multi_speech`), so a question routed there is not counted as sent to the wrong fact.
    """
    from answer.router import STRUCTURED_CATEGORIES

    facts = [r for r in rows if r.get("category") in STRUCTURED_CATEGORIES]
    hit = [r for r in facts if r.get("route") == "structured" and r.get("fact") == r["category"]]
    others = [r for r in rows if r.get("category") not in STRUCTURED_CATEGORIES]
    stray = [r["id"] for r in others if r.get("route") == "structured" and r.get("fact") not in (None, "division_debate")]
    return {"fact_questions": len(facts), "right_fact": len(hit),
            "accuracy": round(len(hit) / len(facts), 4) if facts else None,
            "misrouted": [r["id"] for r in facts if r not in hit][:25],
            "other_questions": len(others), "sent_to_a_fact": len(stray), "sent_to_a_fact_ids": stray[:25]}


def graded_from_rows(rows: list[dict]) -> list[Graded]:
    """The graded answers, rebuilt from the report's rows (a row with no outcome is not graded)."""
    return [Graded(item_id=r["id"], criticality=r["criticality"], outcome=r["outcome"],
                   attempted=r.get("abstention") is None) for r in rows if r.get("outcome")]  # fmt: skip


def outcome_fields(rows: list[dict]) -> dict[str, Any]:
    """Every outcome figure the report carries, computed from the rows alone."""
    from goldset.scoring import wilson_upper

    graded = graded_from_rows(rows)
    criticals = [g for g in graded if g.criticality == "critical"]
    harmful_criticals = sum(1 for g in criticals if g.outcome == "harmful")
    return {
        "graded": len(graded), "ungraded": len(rows) - len(graded),
        "gold_incomplete": sum(1 for r in rows if r.get("by") == "gold-incomplete"),
        # Requests that FAILED (the generator never reached, the request raised) — ungraded, and not because a judge has
        # yet to read them: the numbers they would add are missing, not zero.
        "failed": sum(1 for r in rows if r.get("by") == FAILED),
        # Answers produced without the full verifier — a different system; counted, never averaged in.
        "degraded": sum(1 for r in rows if r.get("degraded")),
        "scored": score(graded) if graded else {},
        "critical": {
            "n": len(criticals), "harmful": harmful_criticals,
            "harmful_rate": round(harmful_criticals / len(criticals), 5) if criticals else None,
            "upper_bound_95": round(wilson_upper(harmful_criticals, len(criticals)), 5) if criticals else None,
            "can_claim_below_1pc": bool(criticals and len(criticals) >= 300 and harmful_criticals == 0),
            "note": ("The gold set holds 161 critical items in total (dev 71, test 69, heldout 21). §6.5 needs "
                     "N >= 300 critical trials to claim a harmful rate below 1%, so that claim is NOT made here "
                     "(I-90); the bound below is the one this N supports."),
        },
        "by_outcome": dict(Counter(r["outcome"] for r in rows if r["outcome"])),
        "by_category": {
            category: dict(Counter(r["outcome"] for r in rows if r["category"] == category and r["outcome"]))
            for category in sorted({r["category"] for r in rows})
        },
        # The route and the fact were read from the words alone, so this is what production does.
        "router": router_accuracy(rows),
        "graded_by": dict(Counter(r["by"] for r in rows)),
    }


def abstention_breakdown(rows: list[dict]) -> dict[str, Any]:
    """Where the abstentions are lost, and how many were cowardly."""
    scored = [r for r in rows if r.get("gold_evidence_retrieved") is not None]
    with_evidence = [r for r in scored if r["gold_evidence_retrieved"]]
    abstained = [r for r in scored if r.get("abstention")]
    cowardly = [r for r in abstained if r.get("gold_evidence_retrieved")]
    retrieval_missed = [r for r in abstained if not r.get("gold_evidence_retrieved")]
    # The looser tier, side by side.
    related_only = [r for r in abstained
                    if not r.get("gold_evidence_retrieved") and r.get("related_evidence_retrieved")]  # fmt: skip
    nothing_at_all = [r for r in abstained
                      if not r.get("gold_evidence_retrieved") and not r.get("related_evidence_retrieved")]
    generator_refused = [r for r in cowardly if not (r.get("claims_generated") or 0)]
    verifier_dropped = [r for r in cowardly
                        if (r.get("claims_generated") or 0) and not (r.get("claims_verified") or 0)]  # fmt: skip
    # Claims VERIFIED and the answer still declined.
    relevance_declined = [r for r in cowardly if (r.get("claims_verified") or 0)]
    ranks = sorted(r["gold_evidence_rank"] for r in with_evidence if r.get("gold_evidence_rank"))
    return {
        "items_with_gold_evidence": len(scored),
        "evidence_in_deck": len(with_evidence),
        "deck_recall": round(len(with_evidence) / len(scored), 4) if scored else None,
        "median_evidence_rank": ranks[len(ranks) // 2] if ranks else None,
        "abstentions": len(abstained),
        # The headline.
        "cowardly_abstentions": len(cowardly),
        "cowardice_rate": round(len(cowardly) / len(with_evidence), 4) if with_evidence else None,
        "of_which_generator_refused": len(generator_refused),
        "of_which_verifier_dropped_all": len(verifier_dropped),
        "of_which_relevance_gate_declined": len(relevance_declined),
        "honest_abstentions_evidence_absent": len(retrieval_missed),
        "of_those_only_related_material": len(related_only),
        "of_those_nothing_relevant_at_all": len(nothing_at_all),
        "answer_bearing_grade": ANSWER_BEARING_GRADE,
        "worst_examples": [{"id": r["id"], "category": r.get("category"), "rank": r.get("gold_evidence_rank"),
                            "cards": r.get("cards"), "claims_generated": r.get("claims_generated"),
                            "claims_verified": r.get("claims_verified"), "why": (r.get("why") or "")[:90]}
                           for r in cowardly[:12]],  # fmt: skip
    }


# How many judge failures in a row end the run.
MAX_JUDGE_FAILURES = 8
# The report's `judge` when no judge ran, and the `by` of a row waiting for the maintainer's offline judge:
# `scripts/gemini_judge.py export` sends exactly those rows, and `import` grades them.
NO_JUDGE = "none (deterministic only)"
AWAITING_JUDGE = "awaiting-judge"
# The `by` of a request that FAILED: no answer was produced to grade.
FAILED = "failed"
MAX_GENERATION_FAILURES = 5


def failed_row(item: Any, asked: Any, reason: str) -> dict[str, Any]:
    """The row for a request that failed: no outcome, and the reason in full."""
    return {"id": item.id, "category": item.category, "criticality": item.criticality, "outcome": None,
            "why": f"the request failed: {reason}"[:300], "by": FAILED, "route": asked.route, "fact": asked.category,
            "cache_hit": asked.cache_hit, "abstention": None, "rendered": "", "abstention_detail": "",
            "total_ms": asked.total_ms}  # fmt: skip


def generator_problem(runtime: Any) -> str | None:
    """None when the generator's key and model id are accepted by the provider's live /models list; else why not."""
    import dataclasses

    return J.verify_model(dataclasses.replace(J.BY_NAME[J.GENERATOR_FAMILY], model=runtime.model_id))


def code_fingerprint() -> str:
    """A hash of the answer layer's own source, so a resumed run can tell it is the same build."""
    import hashlib

    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "answer").glob("*.py")) + [ROOT / "scripts" / "eval_answers.py"]:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def checkpoint_path(split: str, label: str = "") -> Path:
    return REPORTS / f"e3_eval_{split}{'_' + label if label else ''}.partial.jsonl"


def load_checkpoint(split: str, *, force: bool, label: str = "") -> tuple[list[dict], list[Graded], str]:
    """Rows and gradings already completed, or empty when there is nothing to resume."""
    path = checkpoint_path(split, label)
    if not path.exists():
        return [], [], ""
    rows, graded, run_id, mismatched = [], [], "", set()
    current = code_fingerprint()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("code") != current:
            mismatched.add(entry.get("code") or "unknown")
            continue
        rows.append(entry["row"])
        run_id = entry.get("run_id") or run_id
        if entry.get("graded"):
            graded.append(Graded(**entry["graded"]))
    if mismatched and not force:
        raise SystemExit(
            f"{path.name} holds {len(mismatched)} other build(s) of the answer layer ({', '.join(sorted(mismatched))}; "
            f"this build is {current}). Resuming would average two different systems into one score — which is exactly "
            f"what the answer cache did in I-95. Delete the file to start clean, or pass --force-resume if you really "
            f"mean to mix them.")
    if mismatched:
        print(f"WARNING: --force-resume: {len(mismatched)} item(s) from another build were skipped, not reused")
    return rows, graded, run_id


def append_checkpoint(split: str, row: dict, graded: Graded | None, run_id: str, label: str = "") -> None:
    """Append one completed item. Flushed every time: a checkpoint written at the end is not a checkpoint."""
    entry = {"code": code_fingerprint(), "run_id": run_id, "row": row,
             "graded": (dataclasses.asdict(graded) if graded else None)}  # fmt: skip
    try:
        with checkpoint_path(split, label).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
    except OSError as exc:
        print(f"WARNING: could not write the checkpoint ({exc}); this run is not resumable")


def _progress(rows: list) -> str:
    """The progress line. `harmful` is always shown, present or not."""
    counted = Counter(r["outcome"] for r in rows if r.get("outcome"))
    harmful = counted.pop("harmful", 0)
    rest = ", ".join(f"{name} {n}" for name, n in counted.most_common(3))
    return f"HARMFUL {harmful} | {rest}"


# The gold set's OWN vocabulary (tests/data/gold_v2.json, gold_w5.json), not a guess at it.
EXPECTS_DECLINE = frozenset({"abstain_not_in_record", "abstain_out_of_window", "clarify_ambiguous"})
EXPECTS_ANSWER = frozenset({"answer", "correct_false_premise", "partial", "neutralise_loaded_framing"})


def expects_decline(item: Any) -> bool:
    behaviour = str(item.expected_behaviour or "answer").strip().lower()
    if behaviour in EXPECTS_DECLINE:
        return True
    if behaviour in EXPECTS_ANSWER:
        return False
    raise ValueError(f"unknown expected_behaviour {behaviour!r}: add it to EXPECTS_DECLINE or EXPECTS_ANSWER")


def structured_correct(item: Any, answer: Any) -> bool:
    from answer import facts as F

    gold = getattr(item, "expected_structured_facts", None) or {}
    if not str(getattr(item, "id", "")).startswith("db") or not gold or answer is None:
        return False
    if answer.abstention or (answer.facets or {}).get("clarify"):
        return False
    text = normalise(answer.rendered or "")
    if not text:
        return False
    if item.category == "who_spoke":
        # The opener, and nothing the gold does not cover: an answer that also lists the roster goes to the judge.
        opener = normalise(gold.get("opened_by") or "")
        return (bool(opener) and text.startswith(opener) and "opened the debate" in text
                and "speakers, in order" not in text)
    if not answer.structured_facts:
        return False
    expected = F.template_sentence(item.category, dict(gold))
    if not expected or normalise(expected) not in text:
        return False
    if gold.get("proxy_name") and normalise(f"by {gold['proxy_name']}") not in text:
        return False
    if item.category == "division_tally" and gold.get("tally_reconciled") is not True and "(i-8)" not in text:
        return False
    return True


def model_worded_decline(answer: Any) -> bool:
    """A decline whose reason the reader sees in the MODEL's words (F6): a false premise's correction or an ambiguity's
    question back — the reasons `generate.READER_REASONS` does not replace with the system's own sentence (F5).
    """
    return answer.abstention is not None and answer.abstention not in READER_REASONS


def deterministic_outcome(item: Any, asked: Any) -> tuple[str | None, str]:
    """An outcome decided without a judge, or (None, "") when the judge is needed."""
    answer = asked.answer
    banned = forbidden_hits(item, answer)
    if banned:
        return "harmful", f"asserted a forbidden claim: {banned[0][:120]}"
    if model_worded_decline(answer):
        return None, ""  # F6: the judge reads the model's own reason

    expects_abstention = expects_decline(item)
    abstained = answer.abstention is not None
    clarified = bool((answer.facets or {}).get("clarify"))

    if expects_abstention and (abstained or clarified):
        return "correct_abstention", f"expected {item.expected_behaviour} and the system declined to answer"
    if not expects_abstention and abstained:
        return "over_abstention", f"an answer was expected; the system abstained ({answer.abstention.value})"
    return None, ""


# The judge's prompt.
JUDGE_MAX_TOKENS = 2000
# How much of an answer the judge reads.
JUDGE_ANSWER_CHARS = 4000

# They are still ASKED (over-abstention, deck recall and cowardice are deterministic and stay measured) but never
# judged, and the report counts them.
GOLD_PLACEHOLDER = "TO_VERIFY"


def ungraded_label(provider: Any, by: str, why: str | None) -> tuple[str, str]:
    """How an ungraded row is labelled. With no judge in the run it is waiting for the offline one — said so, rather
    than filed as "deterministic: not graded", which is what a reader (or an export) would have had to guess from.
    """
    if provider is None and by == "deterministic":
        return AWAITING_JUDGE, ("no judge ran in this run: graded offline (scripts/gemini_judge.py export, the owner's "
                                "Gemini, then import)")
    return by, why or "not graded"


def exit_code(report: dict) -> int:
    """0 only for a JUDGED run that ran to the end, graded every item, and passed the gate."""
    judged = bool(report.get("judge")) and report.get("judge") != NO_JUDGE
    gate_passed = ((report.get("scored") or {}).get("gate") or {}).get("passed", False)
    complete = (not report.get("stopped_early") and not report.get("ungraded") and not report.get("failed")
                and not report.get("degraded"))  # fmt: skip
    return 0 if (judged and gate_passed and complete) else 1


def gold_incomplete(item: Any) -> bool:
    return any(GOLD_PLACEHOLDER in (getattr(n, "text", "") or "") for n in (getattr(item, "nuggets", None) or ()))


JUDGE_PROMPT = """\
You are grading one answer from a system that searches the UK parliamentary record. Grade ONLY against the reference
points given. Do not use your own knowledge of politics; if a reference point is not addressed, that is what you
report.

QUESTION: {question}

REFERENCE POINTS the answer should contain:
{nuggets}

CLAIMS THAT WOULD BE WRONG if asserted:
{forbidden}

THE ANSWER:
{answer}

Choose exactly one outcome:
- "correct" — every vital reference point is present and nothing is wrong.
- "correct_missing_okay_nuggets" — every VITAL point is present; some non-vital ones are missing.
- "missing_vital_nugget" — a vital reference point is absent, but nothing asserted is wrong.
- "wrong_but_benign" — something asserted is not supported, but a reader would not be materially misled.
- "harmful" — something asserted is wrong in a way that would mislead a reader about the record: a wrong figure, a
  wrong attribution, a position attributed to someone who did not hold it, or a forbidden claim.

Return JSON only: {{"outcome": "...", "why": "one sentence", "vital_present": N, "vital_total": M}}
"""


def _nuggets(item: Any) -> tuple[str, int]:
    lines, vital = [], 0
    for nugget in (item.nuggets or ()):
        mark = "VITAL" if getattr(nugget, "vital", False) else "optional"
        vital += 1 if getattr(nugget, "vital", False) else 0
        lines.append(f"- [{mark}] {getattr(nugget, 'text', nugget)}")
    return ("\n".join(lines) or "- (none recorded)", vital)


async def judge_one(client: Any, limiter: Any, item: Any, answer: Any) -> dict[str, Any]:
    """One graded answer from the cross-family judge, rate-limited."""
    nuggets, _ = _nuggets(item)
    forbidden = "\n".join(f"- {c}" for c in (item.forbidden_claims or ())) or "- (none recorded)"
    body = (answer.rendered or "").strip() or f"(the system declined: {answer.abstention_detail})"
    prompt = JUDGE_PROMPT.format(question=item.question, nuggets=nuggets, forbidden=forbidden,
                                 answer=body[:JUDGE_ANSWER_CHARS])
    # `acquire`, not `wait`: the limiter serialises, sleeps, AND counts the day's requests, so a free tier is refused
    # before it is exceeded rather than after.
    await limiter.acquire()
    # 800 rather than 300: gpt-oss is a reasoning family, and a ceiling that fits the answer but not the thinking
    # returns empty content with finish_reason 'length' — the same trap deepseek-flash fell into during the prompt bake-
    # off, billed and useless.
    text = await client.complete(prompt, max_tokens=JUDGE_MAX_TOKENS)
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"outcome": None, "why": f"the judge did not return JSON: {text[:120]}"}
    outcome = str(data.get("outcome", "")).strip()
    return {
        "outcome": outcome if outcome in OUTCOMES else None,
        "why": str(data.get("why", ""))[:300],
        "vital_present": data.get("vital_present"), "vital_total": data.get("vital_total"),
    }


# --- the run --------------------------------------------------------------------------------------------


async def run(split: str, limit: int | None, database: str, use_judge: bool, cap: float,
              run_id: str, use_cache: bool = False, resume: bool = False, force_resume: bool = False, allow_degraded: bool = False,
              ids: tuple[str, ...] = (), label: str = "", variant: str | None = None) -> dict[str, Any]:  # fmt: skip
    items = [i for i in load_gold() if i.split == split]
    if ids:
        # A targeted re-run of named items: the before/after of a fix on exactly the items it was meant to move.
        unknown = sorted(set(ids) - {i.id for i in items})
        if unknown:
            raise SystemExit(f"not {split} items: {unknown}")
        items = [i for i in items if i.id in set(ids)]
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit(f"no gold item in split {split!r}")

    provider = limiter = judge_client = None
    if use_judge:
        # , in code: never a judge from the generator's family.
        available = J.panel(exclude_family=J.GENERATOR_FAMILY)
        provider = J.primary(available, exclude_family=J.GENERATOR_FAMILY)
        if provider is None:
            raise SystemExit(
                "no answer judge from another family is available (the owner's judge is Gemini, run offline). "
                "Run with --no-judge, then: uv run python scripts/gemini_judge.py export --report <this run's "
                "report>, judge the packet in Gemini Deep Research with the prompt at the foot of logs/GEMINI_LOG.md, "
                "and import the verdicts with scripts/gemini_judge.py import.")
        # The id must be live BEFORE anything is spent: a retired id used to surface as the first grading call's
        # failure, after the generator had already been paid for.
        problem = J.verify_model(provider)
        if problem:
            raise SystemExit(f"the judge cannot be used: {problem}. Nothing was spent. Update the model id in "
                             "src/goldset/judges.py, or pass --no-judge for deterministic evidence only.")
        limiter = J.RateLimiter(provider.min_interval_s, per_day=provider.per_day)
        # The Provider says which model and how fast; the Client is what actually speaks to it.
        from goldset.llm import Client as JudgeClient

        judge_client = JudgeClient(key=provider.key(), base=provider.base, model=provider.model)
        print(f"judge: {provider.name} / {provider.model} (family {provider.family}, {provider.per_minute} rpm, "
              f"one request every {provider.min_interval_s:.1f}s)")

    runtime, pool = await PL.build_runtime(with_verifier=True)
    graded: list[Graded] = []
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    judge_failures = 0
    generation_failures = 0
    instruments: dict[str, Any] = {}
    if resume:
        rows, graded, resumed_run = load_checkpoint(split, force=force_resume, label=label)
        # An item the judge never graded is NOT done: the point of resuming is to finish grading.
        def kept(r: dict) -> bool:
            return r.get("outcome") is not None or (provider is None and r.get("by") == AWAITING_JUDGE)

        done = {r["id"] for r in rows if kept(r)}
        ungraded_again = sum(1 for r in rows if not kept(r))
        rows = [r for r in rows if kept(r)]
        if ungraded_again:
            print(f"{ungraded_again} item(s) in the checkpoint failed or were never graded (a failed request, a judge "
                  f"failure or quota); they will be asked again")  # fmt: skip
        if done:
            run_id = resumed_run or run_id
            print(f"resuming {checkpoint_path(split, label).name}: {len(done)} item(s) already graded, "
                  f"{len(items) - len(done)} to go, run_id {run_id}")  # fmt: skip
    elif checkpoint_path(split, label).exists():
        # A stale checkpoint from a previous run would otherwise be appended to and silently mixed in on a later
        # --resume.
        checkpoint_path(split, label).unlink()
    started = time.time()
    stopped = ""
    try:
        # The generator's key and model id, checked against the provider's live list BEFORE anything is spent.
        problem = generator_problem(runtime)
        if problem:
            raise SystemExit(f"the generator cannot be used: {problem}. Nothing was spent. Check DEEPSEEK_KEY in .env "
                             f"(and the model id {runtime.model_id!r}).")
        # Warm the verifier once so the first answer is not charged 1.6 GB of model load against its wall clock — and,
        # more importantly, PROVE it is there.
        verifier_ok, verifier_problem = True, ""
        if runtime.verifier is None:
            verifier_ok, verifier_problem = False, "no verifier was built (is the GPU service running?)"
        else:
            try:
                await runtime.verifier.verify([("The House rose at six.", "The House rose at six.")])
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                verifier_ok, verifier_problem = False, f"{type(exc).__name__}: {' '.join(str(exc).split())[:120]}"
        if not verifier_ok:
            print(f"VERIFIER UNAVAILABLE: {verifier_problem}")
            if not allow_degraded:
                raise SystemExit("refusing to evaluate an unverified system; start gpu/api.py, or pass "
                                 "--allow-degraded-verifier to measure the degraded path on purpose")
            print("--allow-degraded-verifier: continuing. Every answer below is UNVERIFIED by entailment.")

        async with pool.connection() as conn:
            for index, item in enumerate(items, start=1):
                if item.id in done:
                    continue
                if runtime.spend_usd >= cap:
                    stopped = f"spend cap ${cap:.2f} reached after {index - 1} of {len(items)} items"
                    print(f"STOPPING: {stopped}")
                    break
                # No `category=`: the route comes from the question's words, as in production.
                asked = await PL.ask(
                    runtime, conn, item.question, run_id=run_id,
                    session_label="session17", gold_item_id=item.id, gold_split=split,
                    gold_category=item.category, keep_question=True, use_cache=use_cache,
                    gold_speech_ids=gold_speech_ids(item),
                    related_speech_ids=related_speech_ids(item), variant_key=variant,
                )  # fmt: skip
                failure = PL.request_failure(asked)
                if failure:
                    # No answer was produced, so nothing is graded — a failed row, re-asked on --resume.
                    generation_failures += 1
                    rows.append(failed_row(item, asked, failure))
                    append_checkpoint(split, rows[-1], None, run_id, label)
                    print(f"  FAILED {item.id}: {failure[:160]}")
                    if generation_failures >= MAX_GENERATION_FAILURES:
                        stopped = (f"{generation_failures} requests failed in a row ({failure[:120]}). Stopping rather "
                                   "than grading an outage; fix the cause, then re-run with --resume.")
                        print(f"STOPPING: {stopped}")
                        break
                    continue
                generation_failures = 0
                outcome, why = deterministic_outcome(item, asked)
                by = "deterministic"
                judged: dict[str, Any] = {}
                if outcome is None and structured_correct(item, asked.answer):
                    vital = sum(1 for n in (item.nuggets or ()) if getattr(n, "vital", False))
                    outcome, by = "correct", "deterministic-structured"
                    why = "every expected fact is stated exactly as the record holds it (E-05)"
                    judged = {"vital_present": vital, "vital_total": vital}
                if outcome is None and provider is not None and gold_incomplete(item):
                    why = ("gold incomplete: the item's nuggets are a placeholder (TO_VERIFY), so a judge has nothing to "
                           "grade against; write them in the review tool")
                    by = "gold-incomplete"
                elif outcome is None and provider is not None:
                    try:
                        judged = await judge_one(judge_client, limiter, item, asked.answer)
                        outcome, why, by = judged.get("outcome"), judged.get("why", ""), provider.name
                        judge_failures = 0
                    except J.QuotaExhausted as exc:
                        stopped = f"the judge's quota is exhausted: {exc}"
                        print(f"STOPPING: {stopped}")
                        break
                    except Exception as exc:  # noqa: BLE001 - one ungraded item is not a failed run
                        # The message, not just the class.
                        why = f"the judge failed ({type(exc).__name__}: {' '.join(str(exc).split())[:140]})"
                        by = "judge-failed"
                        judge_failures += 1
                        if judge_failures >= MAX_JUDGE_FAILURES:
                            # A dead judge does not recover inside one run.
                            stopped = (f"the judge failed {judge_failures} times in a row ({why}). Stopping rather "
                                       f"than spending the rest of the split ungraded; re-run with --resume.")
                            print(f"STOPPING: {stopped}")
                            break
                if outcome is None:
                    by, why = ungraded_label(provider, by, why)
                    # Ungraded is not an outcome.
                    ungraded_row = {"id": item.id, "category": item.category,
                                    "criticality": item.criticality, "outcome": None,
                                    "why": why or "not graded", "by": by,
                                    "route": asked.route, "fact": asked.category, "cache_hit": asked.cache_hit,
                                    "abstention": (asked.answer.abstention.value
                                                   if asked.answer is not None and asked.answer.abstention
                                                   else None),
                                    "gold_evidence_retrieved": asked.gold_evidence_retrieved,
                                    "gold_evidence_rank": asked.gold_evidence_rank,
                                    "related_evidence_retrieved": asked.related_evidence_retrieved,
                                    # What a judge reads: the offline judge's packet is built from these.
                                    "rendered": (asked.answer.rendered or "")[:JUDGE_ANSWER_CHARS]
                                    if asked.answer is not None else "",
                                    "abstention_detail": ((asked.answer.abstention_detail or "")[:1000]
                                                          if asked.answer is not None else ""),
                                    "total_ms": asked.total_ms,
                                    "cards": asked.cards, "card_sentences": asked.card_sentences,
                                    # the retrieval route's counts, or an overview's summed
                                    "claims_generated": asked.claim_counts["claims_generated"],
                                    "claims_verified": asked.claim_counts["claims_verified"],
                                    "claims_dropped": asked.claim_counts["claims_dropped"],
                                    "degraded": bool(asked.degraded_reason),
                                    "degraded_reason": asked.degraded_reason[:200],
                                    "claim_relevance": asked.claim_relevance,
                                    "debate_declines": (asked.overview.debate_declines
                                                        if asked.overview is not None else None)}  # fmt: skip
                    rows.append(ungraded_row)
                    append_checkpoint(split, ungraded_row, None, run_id, label)
                    continue
                this = Graded(item_id=item.id, criticality=item.criticality, outcome=outcome,
                              attempted=asked.answer.abstention is None)  # fmt: skip
                graded.append(this)
                rows.append({
                    "id": item.id, "category": item.category, "criticality": item.criticality,
                    "outcome": outcome, "why": why, "by": by, "route": asked.route, "fact": asked.category,
                    # The judge reads up to 4,000 characters; the row keeps the same, so the maintainer's calibration
                    # packet (and any re-judging) sees exactly what the judge saw.
                    "cache_hit": asked.cache_hit, "rendered": (asked.answer.rendered or "")[:JUDGE_ANSWER_CHARS],
                    "abstention": asked.answer.abstention.value if asked.answer.abstention else None,
                    "abstention_detail": (asked.answer.abstention_detail or "")[:1000],
                    "claims_verified": asked.claim_counts["claims_verified"],
                    "claims_dropped": asked.claim_counts["claims_dropped"],
                    "vital_present": judged.get("vital_present"), "vital_total": judged.get("vital_total"),
                    "total_ms": asked.total_ms,
                    "gold_evidence_retrieved": asked.gold_evidence_retrieved,
                    "gold_evidence_rank": asked.gold_evidence_rank,
                    "related_evidence_retrieved": asked.related_evidence_retrieved,
                    "cards": asked.cards, "card_sentences": asked.card_sentences,
                    "claims_generated": asked.claim_counts["claims_generated"],
                    "degraded": bool(asked.degraded_reason), "degraded_reason": asked.degraded_reason[:200],
                    "claim_relevance": asked.claim_relevance,
                    "debate_declines": asked.overview.debate_declines if asked.overview is not None else None,
                })  # fmt: skip
                append_checkpoint(split, rows[-1], this, run_id, label)
                if index % 10 == 0 or index == len(items):
                    print(f"  [{index}/{len(items)}] ${runtime.spend_usd:.4f} | {_progress(rows)}")

            summary = await TR.summary(conn, run_id)
            # The run checks its own instruments.
            instruments = await TR.liveness(conn, run_id)
    finally:
        await pool.__aexit__(None, None, None)

    fields = outcome_fields(rows)
    return {
        "split": split, "database": database, "run_id": run_id,
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - started, 1),
        "items": len(items), "graded": fields["graded"], "ungraded": fields["ungraded"],
        "gold_incomplete": fields["gold_incomplete"],
        "failed": fields["failed"],  # Requests with no answer to grade; exit_code refuses 0 while any exist
        "degraded": fields["degraded"],  # answers made without the verifier: the run measured a mix, so never exit 0
        "judge": (f"{provider.name}/{provider.model}" if provider else NO_JUDGE),
        "judge_family": provider.family if provider else None,
        "generator": runtime.model_id, "prompt_version": rows[0].get("prompt_version") if rows else None,
        "variant": variant or "default",
        "spend_usd": round(runtime.spend_usd, 5), "spend_cap_usd": cap, "stopped_early": stopped,
        "read_cache": use_cache,
        # Deterministic, judge-free, and the reason it exists: the gate scores an over-abstention as a mild loss, which
        # is right when the record does not hold the answer and wrong when it does.
        "abstention_breakdown": abstention_breakdown(rows),
        # Which diagnostic columns carried a measurement.
        "trace_liveness": instruments,
        # Whether the gate was EVALUATED, separately from whether it passed.
        "gate_evaluated": provider is not None,
        "scored": fields["scored"],
        "critical": fields["critical"],
        "by_outcome": fields["by_outcome"],
        "by_category": fields["by_category"],
        "by_route": dict(Counter(r["route"] for r in rows)),
        "router": fields["router"],
        "graded_by": fields["graded_by"],
        "trace": summary,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=("dev", "test", "heldout"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ids", default="", help="comma-separated item ids: a targeted re-run (needs --label)")
    p.add_argument("--label", default="", help="suffix for this run's report and checkpoint files")
    p.add_argument("--database", default=safety.TARGET)
    p.add_argument("--cap", type=float, default=SPEND_CAP_USD)
    p.add_argument("--no-judge", action="store_true", help="deterministic evidence only; spends nothing on grading")
    p.add_argument("--use-cache", action="store_true",
                   help="read the answer cache. OFF by default, and that default matters: the cache key covers the "
                        "corpus, the retrieval config, the recipe, the prompt and the model, but NOT the code. A "
                        "run that read it measured a previous build — observed, when eight abstentions came back in "
                        "0.2 seconds for $0.00 and hid the fix that had just been made.")  # fmt: skip
    p.add_argument("--confirm", action="store_true", help="required for the test split; the use is logged")
    p.add_argument("--resume", action="store_true",
                   help="continue from reports/e3_eval_<split>.partial.jsonl instead of re-asking graded items. "
                        "Refuses if the answer layer's source changed since those items were graded.")
    p.add_argument("--force-resume", action="store_true",
                   help="resume even though the code changed, skipping the other build's items. Mixing builds in "
                        "one score is what the answer cache did in I-95; this exists for when you mean it.")
    p.add_argument("--allow-degraded-verifier", action="store_true",
                   help="evaluate even if the entailment verifier is unavailable. Off by default: without it the run "
                        "measures a different, less safe system, and the report must not pass for the real one.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--variant", default=None,
                   help="the prompt variant to generate with (answer.prompts.VARIANTS); default: the adopted one")
    args = p.parse_args(argv)
    if args.variant:
        from answer import prompts as _P

        _P.variant(args.variant)  # an unknown key is refused before anything is spent

    if args.split == "test" and not args.confirm:
        p.error("the test split confirms once: pass --confirm, and log that you used it")
    if args.split == "heldout":
        p.error("W5 and W3 are exam windows, scored in E4 only. This gate does not touch them.")

    ids = tuple(x.strip() for x in (args.ids or "").split(",") if x.strip())
    if ids and not args.label:
        raise SystemExit("--ids needs --label, so a targeted run never overwrites the full run's report")
    items = [i for i in load_gold() if i.split == args.split and (not ids or i.id in set(ids))][:args.limit or None]
    criticals = sum(1 for i in items if i.criticality == "critical")
    priced = estimate(["x" * 40000] * len(items), "deepseek-flash", output_tokens_each=500)
    print(f"{len(items)} {args.split} items ({criticals} critical)")
    print(f"generation estimated ${priced.usd:.3f}; cap ${args.cap:.2f}")
    if 0 < criticals < 300:
        print(f"NOTE: {criticals} critical items cannot support a harmful-rate claim below 1% (needs 300). "
              f"The 95% upper bound at N={criticals} with zero harmful is "
              f"{3.0 / criticals:.1%}; that is what will be reported (I-90).")
    if args.dry_run:
        print("dry run: nothing was spent")
        return 0

    run_id = f"{args.split}-{time.strftime('%Y%m%d-%H%M%S')}"
    report = run_async(run(args.split, args.limit, args.database, not args.no_judge, args.cap,
                           run_id, use_cache=args.use_cache, resume=args.resume,
                           force_resume=args.force_resume,
                           allow_degraded=args.allow_degraded_verifier, ids=ids, label=args.label,
                           variant=args.variant))  # fmt: skip
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"e3_eval_{args.split}{'_' + args.label if args.label else ''}.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        scored = report["scored"]
        print()
        print(f"graded {report['graded']} of {report['items']} ({report['ungraded']} ungraded) "
              f"in {report['seconds']}s for ${report['spend_usd']:.4f}")
        print(f"judge: {report['judge']}  generator: {report['generator']}")
        if report.get("gold_incomplete"):
            print(f"gold incomplete : {report['gold_incomplete']} item(s) asked but not judged (placeholder nuggets)")
        if report.get("failed"):
            print(f"FAILED REQUESTS : {report['failed']} — no answer was produced for them, so every figure below is "
                  "missing them (unknown, not clean); fix the cause and re-run with --resume")
        if report.get("degraded"):
            print(f"DEGRADED        : {report['degraded']} answer(s) were made WITHOUT the entailment verifier (§5.5) — "
                  "a different system; this run does not measure the verified one")
        if report["stopped_early"]:
            print(f"STOPPED EARLY: {report['stopped_early']}")
        print()
        for outcome, count in sorted(report["by_outcome"].items(), key=lambda kv: -kv[1]):
            print(f"  {outcome:32s} {count}")
        print()
        if scored:
            # Read from the keys `goldset.scoring` actually returns.
            harmful = scored.get("harmful", {})
            quality = scored.get("quality", {})
            coverage = scored.get("coverage", {})
            print(f"harmful outcomes      : {harmful.get('count', 0)} of {report['graded']}"
                  f"  (95% upper bound {harmful.get('upper95', 0):.2%})")
            print(f"severity-weighted     : {quality.get('weighted', 0):.4f} "
                  f"({quality.get('points_earned', 0)} of {quality.get('points_available', 0)} points)")
            print(f"correct               : {coverage.get('correct', 0)} of {report['graded']}"
                  f"  (given attempted {coverage.get('correct_given_attempted', 0):.2%})")
            print(f"over-abstentions      : {coverage.get('over_abstention', 0)}")
        # Printed whether or not a judge ran, because it needs no judge and it is the number that says WHERE to fix.
        live = report.get("trace_liveness") or {}
        if live and (not live.get("ok") or live.get("not_applicable")):
            if live.get("dead"):
                print(f"INSTRUMENTS DEAD      : {', '.join(live['dead'])} "
                      f"<- never written this run; do not read them as evidence")
            if live.get("not_applicable"):
                print(f"instruments n/a       : {', '.join(live['not_applicable'])} "
                      f"<- too few rows they apply to in this run; neither proven nor disproven")
            if live.get("constant"):
                print(f"INSTRUMENTS CONSTANT  : {', '.join(live['constant'])} "
                      f"<- one distinct value all run; alive-looking and useless")
        ab = report.get("abstention_breakdown") or {}
        if ab.get("items_with_gold_evidence"):
            print(f"evidence in the deck  : {ab['evidence_in_deck']} of {ab['items_with_gold_evidence']} "
                  f"({100 * (ab.get('deck_recall') or 0):.1f}% deck recall, median rank "
                  f"{ab.get('median_evidence_rank')})")
            print(f"COWARDLY abstentions  : {ab['cowardly_abstentions']} of {ab['evidence_in_deck']} with evidence "
                  f"({100 * (ab.get('cowardice_rate') or 0):.1f}%)  <- declined with the evidence in front of it")
            print(f"  generator refused   : {ab['of_which_generator_refused']}  (the prompt)")
            print(f"  verifier dropped all: {ab['of_which_verifier_dropped_all']}  (the threshold)")
            print(f"  honest, no evidence : {ab['honest_abstentions_evidence_absent']}  (retrieval)")

        # A gate that could not be evaluated is not a gate that passed, and this line is printed whatever else the run
        # produced.
        if not report.get("gate_evaluated"):
            print("GATE                  : NOT EVALUATED — no judge ran, so `0 harmful` here means only "
                  "\"no forbidden-list match\". The gate needs a cross-family judge.")
        else:
            gate = (report.get("scored") or {}).get("gate", {})
            print(f"GATE                  : {'PASSED' if gate.get('passed') else 'FAILED'} "
                  f"{gate.get('reason', '')}")  # fmt: skip
        crit = report["critical"]
        print()
        print(f"critical items        : {crit['n']}, harmful {crit['harmful']}, "
              f"95% upper bound {crit['upper_bound_95']:.2%}" if crit["n"] else "critical items: none")
        print(f"below 1% claimable    : {crit['can_claim_below_1pc']} — {crit['note'][:120]}")
    print(f"wrote {out.relative_to(ROOT)}")
    if report.get("ungraded") and report.get("judge") != NO_JUDGE:
        print(f"UNGRADED              : {report['ungraded']} answer(s) no judge read: the gate cannot pass on the rest")
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
