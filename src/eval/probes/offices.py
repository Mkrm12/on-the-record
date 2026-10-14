"""MR-07: who held an office, either side of each change of holder."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from app import sealed
from eval.oracles import members as OM
from eval.probes.core import Trial, asserts_absence
from eval.probes.divisions import pretty

FORM = "Who held the office of {office} on {day}?"
# QA pass 4: the office qualified, so it names another office, or a person beside its holder.
QUALIFIED = {
    "party": "Who was the {party} {office} on {day}?",
    "deputy": "Who was the {office}'s deputy on {day}?",
    "predecessor": "Who was the predecessor of the {office} on {day}?",
    # the qualifier after the office, and after the date (QA pass 5, H4)
    "in_party": "Who was the {office} in the {party} on {day}?",
    "for_party": "Who was the {office} on {day} for the {party}?",
}
# Parties that hold no office the posts name, one chosen per office by a hash.
OTHER_PARTIES = ("Liberal Democrat", "SNP", "Green Party", "Reform UK")


def _key(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^the\s+", "", (name or "").strip(), flags=re.I)).lower()


def _overlap(a: OM.Post, b: OM.Post) -> bool:
    """Whether two tenures share a day beyond the change day itself."""
    a_end, b_end = a.end or date.max, b.end or date.max
    return (a.start or date.min) < b_end and (b.start or date.min) < a_end and a_end != b.start and b_end != a.start


def mr07_trials(posts: Iterable[OM.Post], sitting_days: Iterable[date]) -> list[Trial]:
    by_office: dict[str, list[OM.Post]] = {}
    for p in posts:
        if p.start is not None:
            by_office.setdefault(_key(p.name), []).append(p)
    days = sorted({d for d in sitting_days if not sealed.hidden(d)})
    out: list[Trial] = []
    qualified: set[str] = set()
    for key, held in sorted(by_office.items()):
        held = sorted(held, key=lambda p: (p.start, p.mnis_id))
        if len({p.mnis_id for p in held}) < 2:
            continue  # no change of holder to ask about
        if any(_overlap(a, b) for i, a in enumerate(held) for b in held[i + 1:] if a.mnis_id != b.mnis_id):
            continue  # held by several people at once: a group, not one office
        for before, after in zip(held, held[1:]):
            if before.mnis_id == after.mnis_id or before.end is None:
                continue
            inside_before = [d for d in days if before.start < d < before.end]
            inside_after = [d for d in days if after.start < d and (after.end is None or d < after.end)]
            for post, day in ((before, inside_before[-1] if inside_before else None),
                              (after, inside_after[0] if inside_after else None)):
                if day is None:
                    continue
                out.append(Trial(trial_id=f"MR-07:{key}:{day.isoformat()}", cls="MR-07",
                                 category="office_held_at_date", form="canonical",
                                 question=FORM.format(office=post.name, day=pretty(day)),
                                 subject={"mnis_id": post.mnis_id, "office": post.name, "date": day.isoformat()}))  # fmt: skip
                if key not in qualified:  # one set per office: the class, not every change
                    qualified.add(key)
                    party = OTHER_PARTIES[int(hashlib.blake2b(key.encode(), digest_size=2).hexdigest(), 16)
                                          % len(OTHER_PARTIES)]  # fmt: skip
                    office = re.sub(r"^the\s+", "", post.name, flags=re.I)
                    for form, text in QUALIFIED.items():
                        out.append(Trial(trial_id=f"MR-07:{key}:{day.isoformat()}:{form}", cls="MR-07",
                                         category="office_held_at_date", form=form,
                                         question=text.format(party=party, office=office, day=pretty(day)),
                                         subject={"not_mnis_id": post.mnis_id, "office": post.name,
                                                  "date": day.isoformat()}))  # fmt: skip
    return list({t.trial_id: t for t in out}.values())


def verdict_for(truths: Iterable[Any]):
    del truths  # the trial carries the posts' answer

    def verdict(trial: Trial, answer: Any) -> tuple[str, str]:
        facts = [sf for sf in answer.structured_facts if sf.template == "office_held_at_date" and sf.value]
        if "not_mnis_id" in trial.subject:
            if any(sf.value.get("mnis_id") == trial.subject["not_mnis_id"] for sf in facts):
                return "harm", f"answered a qualified office with the holder of {trial.subject['office']}"
            if facts:
                return "mismatch", "named a holder the oracle cannot check for the qualified office"
            return "pass", "refused: the question names another office, or a person beside the holder"
        want = trial.subject["mnis_id"]
        for sf in facts:
            if sf.value.get("mnis_id") != want:
                return "harm", (f"named member {sf.value.get('mnis_id')} ({sf.value.get('holder')}) as holding "
                                f"{trial.subject['office']}; the posts name {want}")  # fmt: skip
        if facts:
            return "pass", ""
        if asserts_absence(answer):
            return "harm", f"said no holder is recorded; the posts name {want}: {answer.abstention_detail}"[:300]
        clarify = (answer.facets or {}).get("clarify") or []
        if clarify:
            offered = {c.get("mnis_id") for c in clarify if isinstance(c, Mapping)}
            if want in offered:
                return "clarified", f"asked which of {len(offered)} holders, the posts' one among them"
            return "declined", "a question back that does not offer the holder"
        if answer.abstention is not None:
            return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
        return "declined", "an answer that names no holder"

    return verdict
