"""MR-16: present tense."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from app import sealed
from eval.oracles import divisions as O
from eval.oracles import members as OM
from eval.oracles import openers as OO
from eval.probes.core import Trial, asserts_absence
from eval.probes.offices import _key, _overlap

OFFICE_FORMS = {
    "canonical": "Who is the {office}?",
    "current": "Who is the current {office}?",
    "now": "Who holds the office of {office} now?",
}
# Named for the shared-rule tier (`eval.probes.core.SHARED_RULE`).
PARTY_FORMS = {
    "present_history": "Which party is {member} in?",
    "present_history_now": "What party does {member} belong to now?",
}
STAMP = re.compile(r"\bAs of the record up to (\d{1,2} [A-Z][a-z]+ \d{4})\b")


def _date(text: str) -> date | None:
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def last_sitting(raw: Path = OO.RAW) -> date | None:
    """The latest sitting day the raw download holds outside the sealed window. The window's own folder is skipped by
    name before anything inside it is listed, and a day inside the window filed anywhere else is skipped by date.
    """
    if not raw.exists():
        return None
    sealed_id = sealed.window().window_id
    days = []
    for window in sorted(raw.iterdir()):
        if not window.is_dir() or window.name == sealed_id:
            continue
        for day in window.iterdir():
            d = _date(day.name) if day.is_dir() else None
            if d is not None and not sealed.hidden(d):
                days.append(d)
    return max(days) if days else None


def office_trials(posts: Iterable[OM.Post], last: date) -> list[Trial]:
    by_office: dict[str, list[OM.Post]] = {}
    for p in posts:
        if p.start is not None:
            by_office.setdefault(_key(p.name), []).append(p)
    out: list[Trial] = []
    for key, held in sorted(by_office.items()):
        if any(_overlap(a, b) for i, a in enumerate(held) for b in held[i + 1:] if a.mnis_id != b.mnis_id):
            continue  # a group, not one office
        if any(last in (p.start, p.end) for p in held):
            continue  # a change on the day itself: whether a start or an end counts is not in the payload
        holders = {p.mnis_id for p in held if p.start < last and (p.end is None or last < p.end)}
        if len(holders) != 1:
            continue
        holder = holders.pop()
        name = next(p.name for p in held if p.mnis_id == holder)
        office = re.sub(r"^the\s+", "", name, flags=re.I)
        for form, text in OFFICE_FORMS.items():
            out.append(Trial(trial_id=f"MR-16:office:{key}:{form}", cls="MR-16", category="office_held_at_date",
                             form=form, question=text.format(office=office),
                             subject={"kind": "office", "mnis_id": holder, "office": name,
                                      "as_of": last.isoformat()}))  # fmt: skip
    return out


def _same(a: str | None, b: str | None) -> bool:
    return bool(a and b) and " ".join(a.split()).casefold() == " ".join(b.split()).casefold()


def party_trials(truths: Iterable[O.DivisionTruth], history: Mapping[int, list[OM.Affiliation]],
                 seats: Mapping[int, list[OM.Seat]], last: date) -> list[Trial]:  # fmt: skip
    # The latest list naming the member with a party, whatever the lobby: a Deputy Speaker is listed, not voting, as
    # "Deputy Speaker", and that is the record's latest statement of their party (MR-16's first run: Caroline Nokes).
    latest: dict[int, tuple[date, str, str, str]] = {}
    for t in truths:
        if sealed.hidden(t.date):
            continue
        for v in t.votes:
            if v.party and v.name:
                seen = latest.get(v.mnis_id)
                if seen is None or t.date > seen[0]:
                    latest[v.mnis_id] = (t.date, v.name, v.party, t.house)
    out: list[Trial] = []
    for mid, (_, name, listed, house) in sorted(latest.items()):
        if OM.seat_on(seats.get(mid), last) != OM.HOUSE_CODES[house]:
            continue
        party = OM.party_on(history.get(mid), last, house)
        if party is None:
            continue
        for form, text in PARTY_FORMS.items():
            out.append(Trial(trial_id=f"MR-16:party:{mid}:{form}", cls="MR-16", category="party_at_date", form=form,
                             question=text.format(member=name), disputed=not _same(party, listed),
                             subject={"kind": "party", "mnis_id": mid, "party": party, "house": house,
                                      "listed": listed, "as_of": last.isoformat()}))  # fmt: skip
    return out


def mr16_trials(truths: Iterable[O.DivisionTruth], *, posts: Iterable[OM.Post] | None = None,
                history: Mapping[int, list[OM.Affiliation]] | None = None,
                seats: Mapping[int, list[OM.Seat]] | None = None, raw: Path = OO.RAW) -> list[Trial]:  # fmt: skip
    last = last_sitting(raw)
    if last is None:
        return []
    posts = OM.posts() if posts is None else posts
    history = OM.history() if history is None else history
    seats = OM.seats() if seats is None else seats
    return office_trials(posts, last) + party_trials(truths, history, seats, last)


def verdict(trial: Trial, answer: Any) -> tuple[str, str]:
    want = trial.subject
    as_of = date.fromisoformat(want["as_of"])
    facts = [sf for sf in answer.structured_facts if sf.template == trial.category and sf.value]
    for sf in facts:
        if sf.value.get("mnis_id") != want["mnis_id"]:
            what = "holder" if want["kind"] == "office" else "member"
            return "harm", f"named {what} {sf.value.get('mnis_id')}; the raw biographies give {want['mnis_id']}"
        if want["kind"] == "party" and not _same(sf.value.get("party"), want["party"]):
            return "harm", f"party {sf.value.get('party')!r}; the history gives {want['party']!r} on {as_of}"
        stated = _date(str(sf.value.get("date") or ""))
        if stated is not None and stated != as_of:
            return "harm", f"answered for {stated}; the record's most recent sitting is {as_of}"
    if facts:
        stamp = STAMP.search(answer.rendered or "")
        if stamp is None:
            return "incomplete", "right, but no 'As of the record up to …' stamp"
        from datetime import datetime

        stamped = datetime.strptime(stamp.group(1), "%d %B %Y").date()
        if stamped != as_of:
            return "harm", f"stamped {stamped}; the record's most recent sitting is {as_of}"
        return "pass", ""
    if asserts_absence(answer):
        return "harm", f"said the record does not hold it; it does: {answer.abstention_detail}"[:300]
    if (answer.facets or {}).get("clarify"):
        return "clarified", "asked which was meant"
    if answer.abstention is not None:
        return "declined", f"{answer.abstention.value}: {answer.abstention_detail}"[:300]
    return "declined", "an answer that names no holder or party"


def verdict_for(truths: Iterable[Any]):
    del truths  # the trial carries the raw answer
    return verdict
