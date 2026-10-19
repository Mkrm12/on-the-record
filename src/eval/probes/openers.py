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
    for t in truths:
        if sealed.hidden(t.date):
            raise ProbeError(f"debate {t.debate_id} is dated inside the sealed window")
    threads = OO.threads(truths)
    # The question names a House, a day and a heading, not a section: items under one heading in two sections that day
    # are as ambiguous as two items in one.
    shared = Counter((t.house, t.date, _key(t.heading)) for t in threads)
    out = []
    for th in sorted(threads, key=lambda x: x.debate_ids[0]):
        if th.chair_business and th.heading and th.members:
            statement = th.chair_statement
            out.append(Trial(trial_id=f"MR-08:{th.debate_ids[0]}", cls="MR-08", category="opener", route="set",
                             form="canonical", ambiguous=shared[(th.house, th.date, _key(th.heading))] > 1,
                             question=FORM.format(heading=th.heading, house=th.house, day=pretty(th.date)),
                             subject={"debate_id": th.debate_ids[0], "debate_ids": list(th.debate_ids), "opener": None,
                                      "chair_business": True, "statement": statement.mnis_id if statement else None,
                                      "presiding": sorted(th.presiding_ids), "members": sorted(th.members)}))  # fmt: skip
            continue
        if th.opener is None and not th.chair_business and th.heading and th.members:
            # no member opened it: every member's contribution says nothing (QA pass 7)
            out.append(Trial(trial_id=f"MR-08:{th.debate_ids[0]}", cls="MR-08", category="opener", route="set",
                             form="canonical", ambiguous=shared[(th.house, th.date, _key(th.heading))] > 1,
                             question=FORM.format(heading=th.heading, house=th.house, day=pretty(th.date)),
                             subject={"debate_id": th.debate_ids[0], "debate_ids": list(th.debate_ids), "opener": None,
                                      "no_opener": True, "presiding": sorted(th.presiding_ids),
                                      "members": sorted(th.members)}))  # fmt: skip
            continue
        if not (th.presiding_first and th.opener and th.heading):
            continue
        out.append(Trial(trial_id=f"MR-08:{th.debate_ids[0]}", cls="MR-08", category="opener", route="set",
                         form="canonical", ambiguous=shared[(th.house, th.date, _key(th.heading))] > 1,
                         question=FORM.format(heading=th.heading, house=th.house, day=pretty(th.date)),
                         subject={"debate_id": th.debate_ids[0], "debate_ids": list(th.debate_ids),
                                  "opener": th.opener.mnis_id, "presiding": sorted(th.presiding_ids),
                                  # members who spoke as members at some point: a member who also chaired part of the
                                  # debate (the Speaker-elect) is not only the Chair
                                  "members": sorted(th.members)}))  # fmt: skip
    return out


def verdict_for(truths: Iterable[OO.DebateTruth]):
    threads = OO.threads(list(truths))
    thread_of = {did: th for th in threads for did in th.debate_ids}

    def judged(named: Mapping[str, Any], opener: int | None, presiding: Iterable[int], members: Iterable[int],
               in_thread: Iterable[str]) -> tuple[str, str]:  # fmt: skip
        mnis, printed = named.get("mnis_id"), named.get("speaker_name") or ""
        if mnis is not None and mnis == opener and not OO.presiding(printed):
            return "pass", ""
        if (mnis is not None and mnis in set(presiding) and mnis not in set(members)) or OO.presiding(printed):
            return "harm", f"named the Chair ({printed or mnis}) as the debate's opener"
        return "mismatch", (f"named member {mnis} ({printed}) for debate {named.get('debate_id')}; the record's first "
                            f"member there, not presiding, is {opener}")  # fmt: skip

    def verdict(trial: Trial, answer: Any) -> tuple[str, str]:
        target = trial.subject.get("debate_ids") or [trial.subject["debate_id"]]
        facets = answer.facets or {}
        listed = facets.get("openers")
        if listed:
            covered = set()
            for entry in listed:
                th = thread_of.get(entry.get("debate_id"))
                if th is None:
                    return "error", f"the answer names debate {entry.get('debate_id')}, which no truth covers"
                if entry.get("no_opener"):
                    if th.opener is not None or th.chair_business:
                        return "harm", (f"in a listing of {len(listed)}: said no member opened debate "
                                        f"{th.debate_ids[0]}, and the record says otherwise")  # fmt: skip
                    covered.add(th.debate_ids[0])
                    continue
                if th.chair_business or entry.get("chair_statement"):
                    if not (th.chair_business and entry.get("chair_statement")):
                        return "harm", (f"in a listing of {len(listed)}: debate {th.debate_ids[0]} is "
                                        f"{'the Chair' if th.chair_business else 'a member'}'s, and the answer says "
                                        f"{'the Chair' if entry.get('chair_statement') else 'a member'}'s")  # fmt: skip
                    covered.add(th.debate_ids[0])
                    continue
                outcome, why = judged(entry, th.opener.mnis_id if th.opener else None, th.presiding_ids, th.members,
                                      th.debate_ids)  # fmt: skip
                if outcome != "pass":
                    return "harm", f"in a listing of {len(listed)}: {why}"
                covered.add(th.debate_ids[0])
            if target[0] in covered:
                return "aggregate", f"every one of {len(listed)} items under the heading named with its opener, right"
            return "harm", f"a listing of {len(listed)} items that leaves out the asked debate {target[0]}"
        named = facets.get("opener")
        none = facets.get("no_opener")
        if none and none.get("debate_id") in target:
            if trial.subject.get("no_opener"):
                return "pass", "no member's speech opens it, and none does"
            if trial.subject.get("chair_business"):
                return "declined", "said no member opened it (true of the Chair's statement), not that it is the Chair's"
            return "harm", (f"said no member's speech opens it ({none.get('reason')}); the record's first member is "
                            f"{trial.subject['opener']}")  # fmt: skip
        if trial.subject.get("chair_business"):
            stated = facets.get("chair_statement")
            if named and named.get("debate_id") in target:
                return "harm", (f"named {named.get('speaker_name') or named.get('mnis_id')} as opening the Chair's own "
                                f"statement")  # fmt: skip
            if stated and stated.get("debate_id") in target:
                # by the Chair who made it: the Chair's printing, or the statement's own maker (QA pass 7, Med-1)
                if OO.presiding(stated.get("speaker_name")) or (
                        stated.get("mnis_id") is not None and stated.get("mnis_id") == trial.subject.get("statement")):
                    return "pass", "the Chair's own statement, named as the Chair's"
                return "harm", (f"called the statement the Chair's, made by {stated.get('speaker_name') or stated.get('mnis_id')}"
                                f", who is not the Chair who made it")  # fmt: skip
            if not named and not facets.get("clarify") and asserts_absence(answer):
                return "declined", "said no member's speech opens it, which is true of the Chair's statement"
        elif facets.get("chair_statement") and facets["chair_statement"].get("debate_id") in target:
            return "harm", "called a member's debate the Chair's own statement"
        if trial.subject.get("no_opener"):
            if named and named.get("debate_id") in target:
                if OO.presiding(named.get("speaker_name")) or named.get("mnis_id") in set(trial.subject["presiding"]):
                    return "harm", f"named the Chair ({named.get('speaker_name')}) as the debate's opener"
                return "mismatch", (f"named member {named.get('mnis_id')} as opener; by the record every member's "
                                    f"contribution says nothing")  # fmt: skip
            if asserts_absence(answer):
                return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
        if named:
            if named.get("debate_id") not in target:
                if trial.ambiguous:
                    return "substituted", f"answered for debate {named.get('debate_id')}, which shares the heading"
                return "harm", f"named the opener of debate {named.get('debate_id')} for a question about {target[0]}"
            return judged(named, trial.subject["opener"], trial.subject["presiding"], trial.subject.get("members", ()),
                          target)  # fmt: skip
        clarify = facets.get("clarify") or []
        offered = {c.get("debate_id") for c in clarify if isinstance(c, Mapping)}
        if clarify:
            if offered & set(target):
                return "clarified", f"asked which of {len(offered)} debates, the asked one among them"
            return "declined", "a question back that does not offer the asked debate"
        if asserts_absence(answer):
            # the debate has a member's speech that opens it: saying the record holds none is false (QA pass 3)
            return "harm", f"said the record does not hold it, and it does: {answer.abstention_detail}"[:300]
        if answer.abstention is not None:
            return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
        return "declined", "an answer that names no opener"

    return verdict
