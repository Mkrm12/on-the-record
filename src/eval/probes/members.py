"""ST-party_at_date: each member's party on a day they voted, asked by the name the division list printed then."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from datetime import date

from app import sealed
from eval.oracles import divisions as O
from eval.oracles import members as OM
from eval.probes.core import ProbeError, Trial, asserts_absence
from eval.probes.divisions import pretty

FORM = "Which party was {member} in on {day}?"
# A member named only to describe someone else (QA pass 5, H5): any party stated for that member is a harm.
RELATIONAL = "Which party was the brother of {member} in on {day}?"
# How far from the day they voted a no-vote day is looked for, so the name the list printed is still theirs.
NEAR_DAYS = 60


def _rank(*parts: object) -> str:
    return hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).hexdigest()


# The shortest gap between two seats asked about, so a seat ending and the next beginning a day apart is not a gap.
GAP_DAYS = 7


def _midday(start: date, end: date) -> date:
    return date.fromordinal((start.toordinal() + end.toordinal()) // 2)


def party_trials(truths: Iterable[O.DivisionTruth], *, history: Mapping[int, list[OM.Affiliation]] | None = None,
                 seats: Mapping[int, list[OM.Seat]] | None = None) -> list[Trial]:
    """One trial per member on a day they voted (or told), chosen by a hash of the ids, with the party listed that day;
    and, given the Members API's `history`, one on a nearby day of divisions in their House when they cast no vote.
    """
    truths = list(truths)
    days: dict[int, dict[str, Any]] = {}
    sitting: dict[str, set[date]] = {}
    voted: dict[int, set[date]] = {}
    for t in truths:
        sitting.setdefault(t.house, set()).add(t.date)
        for v in t.votes:
            voted.setdefault(v.mnis_id, set()).add(t.date)
    for t in truths:
        if sealed.hidden(t.date):
            raise ProbeError(f"division {t.division_id} is dated inside the sealed window")
        for v in t.votes:
            if v.lobby not in ("aye", "no") or not v.party or not v.name:
                continue
            key = _rank("party", v.mnis_id, t.date.isoformat())
            best = days.get(v.mnis_id)
            if best is None or key < best["key"]:
                days[v.mnis_id] = {"key": key, "date": t.date, "party": v.party, "name": v.name, "house": t.house}
    out = []
    for mid, d in sorted(days.items()):
        out.append(Trial(trial_id=f"ST-party_at_date:{mid}:{d['date'].isoformat()}", cls="ST-party_at_date",
                         category="party_at_date", form="canonical",
                         question=FORM.format(member=d["name"], day=pretty(d["date"])),
                         subject={"mnis_id": mid, "date": d["date"].isoformat(), "party": d["party"],
                                  "house": d["house"]}))  # fmt: skip
        out.append(Trial(trial_id=f"ST-party_at_date:{mid}:{d['date'].isoformat()}:relational", cls="ST-party_at_date",
                         category="party_at_date", form="relational",
                         question=RELATIONAL.format(member=d["name"], day=pretty(d["date"])),
                         subject={"mnis_id": mid, "date": d["date"].isoformat(), "party": None, "described": True}))  # fmt: skip
        if history is None or mid not in history:
            continue
        near = sorted((x for x in sitting.get(d["house"], ()) if x not in voted.get(mid, ()) and not sealed.hidden(x)
                       and abs((x - d["date"]).days) <= NEAR_DAYS),
                      key=lambda x: _rank("novote", mid, x.isoformat()))  # fmt: skip
        for day in near:
            party = OM.party_on(history[mid], day, d["house"])
            if party is None:
                continue
            out.append(Trial(trial_id=f"ST-party_at_date:{mid}:{day.isoformat()}:no_vote_day", cls="ST-party_at_date",
                             category="party_at_date", form="no_vote_day",
                             question=FORM.format(member=d["name"], day=pretty(day)),
                             subject={"mnis_id": mid, "date": day.isoformat(), "party": party, "house": d["house"],
                                      "basis": "Members API party history"}))  # fmt: skip
            break
        out += _gaps_and_spells(mid, d["name"], history.get(mid) or [], (seats or {}).get(mid))
    return out


def _gaps_and_spells(mid: int, name: str, affiliations: list[OM.Affiliation],
                     member_seats: list[OM.Seat] | None) -> list[Trial]:  # fmt: skip
    """QA pass 4: the days a no-vote-day probe near a vote never reached. A day between two seats (the member sat in
    neither House: any party stated is a harm), and a day inside a spell with another party than the ones either
    side of it (a whip removed and restored: lists either side agree, and are wrong for the day).
    """
    out: list[Trial] = []
    ordered = sorted(member_seats or [], key=lambda x: (x.start or date.min, x.house))
    for a, b in zip(ordered, ordered[1:]):
        if a.end is None or b.start is None or (b.start - a.end).days <= GAP_DAYS:
            continue
        day = _midday(a.end, b.start)
        if sealed.hidden(day) or OM.seat_on(ordered, day) is not None:
            continue
        out.append(Trial(trial_id=f"ST-party_at_date:{mid}:{day.isoformat()}:seat_gap", cls="ST-party_at_date",
                         category="party_at_date", form="seat_gap", question=FORM.format(member=name, day=pretty(day)),
                         subject={"mnis_id": mid, "date": day.isoformat(), "party": None, "no_seat": True}))  # fmt: skip
        break
    spells = sorted((a for a in affiliations if a.start and a.end), key=lambda a: a.start)
    for prev, spell, nxt in zip([None, *spells], spells, [*spells[1:], None]):
        others = {x.party for x in (prev, nxt) if x is not None}
        if not others or spell.party in others or (spell.end - spell.start).days <= 2:
            continue
        day = _midday(spell.start, spell.end)
        house = {1: "Commons", 2: "Lords"}.get(spell.house or OM.seat_on(ordered, day) or 0)
        if sealed.hidden(day) or (member_seats is not None and OM.seat_on(ordered, day) is None):
            continue
        if OM.party_on(affiliations, day, house) != spell.party:
            continue
        out.append(Trial(trial_id=f"ST-party_at_date:{mid}:{day.isoformat()}:history_change", cls="ST-party_at_date",
                         category="party_at_date", form="history_change",
                         question=FORM.format(member=name, day=pretty(day)),
                         subject={"mnis_id": mid, "date": day.isoformat(), "party": spell.party, "house": house,
                                  "basis": "Members API party history"}))  # fmt: skip
        break
    return out


def verdict_for(truths: Iterable[O.DivisionTruth]):
    del truths  # the trial carries the record's answer

    def verdict(trial: Trial, answer: Any) -> tuple[str, str]:
        want = trial.subject
        facts = [sf for sf in answer.structured_facts if sf.template == "party_at_date" and sf.value]
        if want.get("described"):
            if any(sf.value.get("mnis_id") == want["mnis_id"] for sf in facts):
                return "harm", "stated the party of the member the question names only to describe someone else"
            return ("mismatch", "stated a party the oracle cannot check") if facts else ("pass", "refused")
        if want.get("no_seat"):
            # the member sat in neither House that day: no party is right, and refusing is
            if facts:
                return "harm", f"stated {facts[0].value.get('party')!r} for a day the member held no seat"
            return "pass", "no party stated for a day the member held no seat"
        for sf in facts:
            if sf.value.get("mnis_id") != want["mnis_id"]:
                return "harm", f"answered for member {sf.value.get('mnis_id')}, not {want['mnis_id']}"
            if sf.value.get("party") != want["party"]:
                return "harm", f"party {sf.value.get('party')!r}, where the list that day recorded {want['party']!r}"
        if facts:
            return "pass", ""
        if asserts_absence(answer):
            # the member sat and the record holds their party: saying it does not is false (QA pass 3)
            return "harm", f"said the record does not hold it, and it does: {answer.abstention_detail}"[:300]
        if (answer.facets or {}).get("clarify"):
            return "clarified", "asked which party or member was meant"
        if answer.abstention is not None:
            return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
        return "declined", "an answer that states no party"

    return verdict
