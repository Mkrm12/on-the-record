"""ST-bill_stage_date: every dated stage of every Bill, asked when it happened."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from app import sealed
from eval.oracles import bills as OB
from eval.probes.core import ProbeError, Trial, asserts_absence


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def question(s: OB.StageTruth) -> str:
    if _key(s.stage) == "royal assent":
        return f"When did the {s.bill} receive Royal Assent?"
    house = f" in the {s.house}" if s.house else ""
    return f"When did the {s.bill} have its {s.stage}{house}?"


# QA pass 4: a word before a bill's title that makes another bill, chosen per bill by a hash, never a title the record
# holds; and the stages a bill may never have reached, asked only where none of its stages is one.
PREFIXES = ("Social", "National", "Local", "Public", "Digital", "Regional")
UNREACHED = (("2nd reading", "2nd reading"), ("3rd reading", "3rd reading"), ("Report stage", "report"),
             ("Royal Assent", "royal assent"))


def _rank(*parts: object) -> int:
    import hashlib

    return int(hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=4).hexdigest(), 16)


def _unanswerable(truths: list[OB.StageTruth]) -> list[Trial]:
    """Trials whose right answer is a refusal: a stage only scheduled, a longer name that is no bill, a stage the bill
    never reached. Any stage date stated is a harm.
    """
    out: list[Trial] = []
    titles = {_key(s.bill) for s in truths}
    by_bill: dict[int, list[OB.StageTruth]] = {}
    by_title: dict[str, list[OB.StageTruth]] = {}
    for s in truths:
        by_bill.setdefault(s.bill_id, []).append(s)
        by_title.setdefault(_key(s.bill), []).append(s)
    asked: set[tuple[str, str]] = set()
    for bill_id, stages in sorted(by_bill.items()):
        bill = stages[0].bill
        if not bill:
            continue
        reached = {_key(x.stage) for x in by_title[_key(bill)] if x.reached}
        for s in stages:
            if not s.reached and s.scheduled and _key(s.stage) not in reached:
                out.append(Trial(trial_id=f"ST-bill_stage_date:{s.stage_id}:future", cls="ST-bill_stage_date",
                                 category="bill_stage_date", form="future", question=question(s),
                                 subject={"bill_id": bill_id, "stage": s.stage, "unreached": True,
                                          "scheduled": s.scheduled[0].isoformat()}))  # fmt: skip
        held = [s for s in stages if s.reached]
        if not held:
            continue
        word = PREFIXES[_rank("prefix", bill_id) % len(PREFIXES)]
        longer = f"{word} {bill}"
        if _key(longer) not in titles and not any(_key(t).endswith(_key(longer)) for t in titles):
            out.append(Trial(trial_id=f"ST-bill_stage_date:{bill_id}:prefixed", cls="ST-bill_stage_date",
                             category="bill_stage_date", form="prefixed",
                             question=question(OB.StageTruth(bill_id, longer, held[0].stage_id, held[0].stage,
                                                             held[0].house, held[0].sittings)),
                             subject={"not_bill_id": bill_id, "stage": held[0].stage, "unreached": True}))  # fmt: skip
        names = " | ".join(_key(s.stage) for s in by_title[_key(bill)])
        missing = next((stage for stage, kind in UNREACHED if kind not in names), None)
        if missing and (_key(bill), missing) not in asked:
            asked.add((_key(bill), missing))
            out.append(Trial(trial_id=f"ST-bill_stage_date:{bill_id}:unreached", cls="ST-bill_stage_date",
                             category="bill_stage_date", form="unreached",
                             question=question(OB.StageTruth(bill_id, bill, 0, missing, "", ())),
                             subject={"bill_id": bill_id, "stage": missing, "unreached": True}))  # fmt: skip
    return out


def st_trials(truths: Iterable[OB.StageTruth]) -> list[Trial]:
    truths = list(truths)
    unanswerable = _unanswerable(truths)
    truths = [s for s in truths if s.reached]
    bills_per_title = {k: len(v) for k, v in _group(truths).items()}
    repeated = Counter((s.bill_id, _key(s.stage), s.house) for s in truths)
    out = []
    for s in sorted(truths, key=lambda x: x.stage_id):
        if any(sealed.hidden(d) for d in s.sittings):
            raise ProbeError(f"stage {s.stage_id} sat inside the sealed window")
        if not s.bill:
            continue
        out.append(Trial(trial_id=f"ST-bill_stage_date:{s.stage_id}", cls="ST-bill_stage_date",
                         category="bill_stage_date", form="canonical", question=question(s),
                         ambiguous=bills_per_title[_key(s.bill)] > 1 or repeated[(s.bill_id, _key(s.stage), s.house)] > 1,
                         subject={"stage_id": s.stage_id, "bill_id": s.bill_id, "stage": s.stage, "house": s.house,
                                  "first_sitting": s.first_sitting.isoformat(), "sittings": len(s.sittings)}))  # fmt: skip
    return out + unanswerable


def _group(truths: list[OB.StageTruth]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for s in truths:
        out.setdefault(_key(s.bill), set()).add(s.bill_id)
    return out


def verdict_for(truths: Iterable[OB.StageTruth]):
    del truths  # the trial carries the record's answer

    def verdict(trial: Trial, answer: Any) -> tuple[str, str]:
        want = trial.subject
        facts = [sf for sf in answer.structured_facts if sf.template == "bill_stage_date" and sf.value]
        if want.get("unreached"):
            # a stage only scheduled, a name that is no bill, a stage never reached: no date is right (QA pass 4)
            wrong = want.get("not_bill_id")
            if facts and (wrong is None or any(sf.value.get("bill_id") == wrong for sf in facts)):
                return "harm", f"stated {facts[0].value.get('stage')!r} on {facts[0].value.get('first_sitting')} " \
                               f"for a question the record cannot answer ({trial.form})"
            if facts:
                return "mismatch", "stated a stage of another bill the oracle does not check"
            return "pass", "no date stated for a stage the record does not hold as asked"
        for sf in facts:
            v = sf.value
            same_stage = v.get("bill_id") == want["bill_id"] and _key(v.get("stage")) == _key(want["stage"]) and (
                (v.get("house") or "") == want["house"] or not want["house"])
            if not same_stage:
                if trial.ambiguous:
                    return "substituted", f"answered {v.get('bill')}: {v.get('stage')} ({v.get('house')})"
                return "harm", f"answered bill {v.get('bill_id')} {v.get('stage')!r} for {want['stage']!r}"
            if v.get("first_sitting") != want["first_sitting"] or v.get("sittings") != want["sittings"]:
                return "harm", (f"first sitting {v.get('first_sitting')} over {v.get('sittings')} day(s), where the record "
                                f"has {want['first_sitting']} over {want['sittings']}")  # fmt: skip
        if facts:
            return "pass", ""
        if asserts_absence(answer):
            # the stage is in the record: saying it is not is false (QA pass 3)
            return "harm", f"said the record does not hold it, and it does: {answer.abstention_detail}"[:300]
        clarify = (answer.facets or {}).get("clarify") or []
        if clarify:
            offered = {c.get("stage_id") for c in clarify if isinstance(c, Mapping)}
            if want["stage_id"] in offered or not offered:
                return "clarified", f"asked which of {len(clarify)} stages"
            return "declined", "a question back that does not offer the asked stage"
        if answer.abstention is not None:
            return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
        return "declined", "an answer that states no stage"

    return verdict
