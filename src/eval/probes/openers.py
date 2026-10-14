"""MR-08: every debate the Chair opens, asked "who opened it?"."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from app import sealed
from eval.oracles import openers as OO
from eval.probes.core import ProbeError, Trial, asserts_absence
from eval.probes.divisions import pretty

FORM = "Who opened the debate on {heading} in the {house} on {day}?"


def _key(heading: str) -> str:
    return re.sub(r"\s+", " ", heading or "").strip().lower()


def mr08_trials(truths: Iterable[OO.DebateTruth]) -> list[Trial]:
    truths = list(truths)
    shared = Counter((t.house, t.date, _key(t.heading)) for t in truths)
    out = []
    for t in sorted(truths, key=lambda x: x.debate_id):
        if sealed.hidden(t.date):
            raise ProbeError(f"debate {t.debate_id} is dated inside the sealed window")
        if not (t.presiding_first and t.opener and t.heading):
            continue
        out.append(Trial(trial_id=f"MR-08:{t.debate_id}", cls="MR-08", category="opener", route="set",
                         form="canonical", ambiguous=shared[(t.house, t.date, _key(t.heading))] > 1,
                         question=FORM.format(heading=t.heading, house=t.house, day=pretty(t.date)),
                         subject={"debate_id": t.debate_id, "opener": t.opener.mnis_id,
                                  "presiding": sorted(t.presiding_ids),
                                  # members who spoke as members at some point: a member who also chaired part of the
                                  # debate (the Speaker-elect) is not only the Chair
                                  "members": sorted({c.mnis_id for c in t.contributions
                                                     if c.mnis_id and not OO.presiding(c.attributed)})}))  # fmt: skip
    return out


def verdict_for(truths: Iterable[OO.DebateTruth]):
    del truths  # the trial carries what the verdict needs; the signature matches the other probe modules

    def verdict(trial: Trial, answer: Any) -> tuple[str, str]:
        target = trial.subject["debate_id"]
        named = (answer.facets or {}).get("opener")
        if named:
            mnis, printed = named.get("mnis_id"), named.get("speaker_name") or ""
            if named.get("debate_id") != target:
                if trial.ambiguous:
                    return "substituted", f"answered for debate {named.get('debate_id')}, which shares the heading"
                return "harm", f"named the opener of debate {named.get('debate_id')} for a question about {target}"
            if mnis is not None and mnis == trial.subject["opener"] and not OO.presiding(printed):
                return "pass", ""
            only_chaired = mnis is not None and mnis in trial.subject["presiding"] and mnis not in trial.subject.get(
                "members", ())
            if only_chaired or OO.presiding(printed):
                return "harm", f"named the Chair ({printed or mnis}) as the debate's opener"
            return "mismatch", (f"named member {mnis} ({printed}); the record's first member, not presiding, is "
                                f"{trial.subject['opener']}")  # fmt: skip
        clarify = (answer.facets or {}).get("clarify") or []
        offered = {c.get("debate_id") for c in clarify if isinstance(c, Mapping)}
        if clarify:
            if target in offered:
                return "clarified", f"asked which of {len(offered)} debates, the asked one among them"
            return "declined", "a question back that does not offer the asked debate"
        if asserts_absence(answer):
            # the debate has a member's speech that opens it: saying the record holds none is false (QA pass 3)
            return "harm", f"said the record does not hold it, and it does: {answer.abstention_detail}"[:300]
        if answer.abstention is not None:
            return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
        return "declined", "an answer that names no opener"

    return verdict
