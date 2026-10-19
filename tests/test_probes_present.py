"""MR-16, the present-tense probe class, from raw payloads only."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from eval.oracles import divisions as O
from eval.oracles import members as OM
from eval.probes import present as PP

LAST = date(2026, 9, 17)


def _posts():
    return [OM.Post(1, "Chancellor of the Exchequer", date(2019, 7, 24), date(2026, 7, 5)),
            OM.Post(2, "Chancellor of the Exchequer", date(2026, 7, 5), None),
            # an office held by two people at once is a group, never asked
            OM.Post(3, "Minister of State (Home Office)", date(2025, 1, 1), None),
            OM.Post(4, "Minister of State (Home Office)", date(2025, 6, 1), None),
            # an office whose holder changes on the last day itself: whether the end counts is not in the payload
            OM.Post(5, "Leader of the House of Commons", date(2024, 7, 5), LAST),
            OM.Post(6, "Leader of the House of Commons", LAST, None),
            # an office nobody holds on the last day
            OM.Post(7, "Secretary of State for International Development", date(2019, 7, 24), date(2020, 9, 2))]


def _truths():
    vote = O.Vote(mnis_id=10, lobby="aye", is_teller=False, proxy_name=None, party="Labour", name="Ann Member")
    old = O.Vote(mnis_id=10, lobby="aye", is_teller=False, proxy_name=None, party="Labour", name="Ann Oldname")
    def truth(i, day, v):
        return O.DivisionTruth(division_id=i, house="Commons", date=day, number=1, title="T", ayes=1, noes=0,
                               ayes_listed=1, noes_listed=0, ayes_named=1, noes_named=0, tally_reconciled=True,
                               votes=(v,))
    return [truth(1, date(2024, 5, 9), old), truth(2, date(2026, 9, 15), vote)]


def _history():
    return {10: [OM.Affiliation("Labour", date(2015, 5, 7), None, 1)],
            11: [OM.Affiliation("Labour", date(2015, 5, 7), date(2026, 9, 1), 1),
                 OM.Affiliation("Independent", date(2026, 9, 1), None, 1)]}


def _seats():
    return {10: [OM.Seat(1, date(2015, 5, 7), None)], 11: [OM.Seat(1, date(2015, 5, 7), None)]}


def test_the_last_day_is_the_latest_raw_sitting_outside_the_sealed_window(tmp_path):
    for window, day in (("W1", "2024-07-30"), ("W0", "2026-09-17"), ("W0", "2026-09-01")):
        (tmp_path / window / day).mkdir(parents=True)
    assert PP.last_sitting(tmp_path) == LAST


def test_the_sealed_windows_folder_is_never_listed(tmp_path, monkeypatch):
    """Not filtered after reading: never opened. A day folder inside the window under ANY name is skipped too."""
    from app import sealed

    (tmp_path / sealed.window().window_id / "2026-05-05").mkdir(parents=True)
    (tmp_path / "W9" / "2026-05-06").mkdir(parents=True)  # a sealed day filed under another window
    (tmp_path / "W1" / "2024-07-30").mkdir(parents=True)
    seen = []
    real_iterdir = type(tmp_path).iterdir

    def spy(self):
        seen.append(self.name)
        return real_iterdir(self)

    monkeypatch.setattr(type(tmp_path), "iterdir", spy)
    assert PP.last_sitting(tmp_path) == date(2024, 7, 30)
    assert sealed.window().window_id not in seen


def test_office_trials_ask_only_offices_one_person_holds_strictly_on_the_last_day():
    trials = PP.office_trials(_posts(), LAST)
    offices = {t.subject["office"] for t in trials}
    assert offices == {"Chancellor of the Exchequer"}
    assert {t.subject["mnis_id"] for t in trials} == {2}
    assert all(t.cls == "MR-16" and t.category == "office_held_at_date" for t in trials)
    assert {t.form for t in trials} == set(PP.OFFICE_FORMS)
    assert all(t.subject["as_of"] == LAST.isoformat() for t in trials)


def test_party_trials_name_the_member_as_the_latest_list_printed_them_and_take_the_history_on_the_last_day():
    trials = PP.party_trials(_truths(), _history(), _seats(), LAST)
    assert trials and all(t.subject["mnis_id"] == 10 and t.subject["party"] == "Labour" for t in trials)
    assert all("Ann Member" in t.question for t in trials), "the latest printed name, not an older one"
    assert {t.form for t in trials} == set(PP.PARTY_FORMS)
    assert all(t.form in __import__("eval.probes.core", fromlist=["SHARED_RULE"]).SHARED_RULE for t in trials), \
        "the history is the route's rule at the record's edge too: reported in the shared-rule tier, never canonical"


def test_the_listed_party_is_the_latest_list_naming_the_member_whatever_the_lobby():
    """A Deputy Speaker is listed, not voting, with "Deputy Speaker" as the party; that is the record's latest
    statement of it, and the route reads it (MR-16's first run: Caroline Nokes).
    """
    voted = O.Vote(mnis_id=10, lobby="aye", is_teller=False, proxy_name=None, party="Labour", name="Ann Member")
    listed = O.Vote(mnis_id=10, lobby="no_vote", is_teller=False, proxy_name=None, party="Deputy Speaker",
                    name="Ann Member")
    truths = [O.DivisionTruth(division_id=i, house="Commons", date=day, number=1, title="T", ayes=1, noes=0,
                              ayes_listed=1, noes_listed=0, ayes_named=1, noes_named=0, tally_reconciled=True,
                              votes=(v,)) for i, day, v in ((1, date(2026, 9, 1), voted), (2, date(2026, 9, 15), listed))]
    trials = PP.party_trials(truths, _history(), _seats(), LAST)
    assert trials and all(t.subject["listed"] == "Deputy Speaker" and t.disputed for t in trials)


def test_a_member_whose_list_and_history_disagree_on_the_last_day_is_disputed():
    vote = O.Vote(mnis_id=11, lobby="aye", is_teller=False, proxy_name=None, party="Labour", name="Ben Member")
    truths = [O.DivisionTruth(division_id=3, house="Commons", date=date(2026, 9, 15), number=1, title="T", ayes=1,
                              noes=0, ayes_listed=1, noes_listed=0, ayes_named=1, noes_named=0, tally_reconciled=True,
                              votes=(vote,))]
    trials = PP.party_trials(truths, _history(), _seats(), LAST)
    assert trials and all(t.disputed for t in trials)


# --- the verdict
# ----------------------------------------------------------------------------------------------------------


def _answer(*, value=None, template="office_held_at_date", rendered="", abstention=None, clarify=None):
    facts = (SimpleNamespace(template=template, value=value),) if value else ()
    return SimpleNamespace(structured_facts=facts, rendered=rendered, abstention=abstention, abstention_detail="",
                           facets={"clarify": clarify} if clarify else {}, coverage_note="")


@pytest.fixture
def office_trial():
    return PP.office_trials(_posts(), LAST)[0]


STAMP = "As of the record up to 17 September 2026, its most recent sitting:"


def test_the_holder_with_the_stamp_passes(office_trial):
    answer = _answer(value={"mnis_id": 2, "date": "2026-09-17"}, rendered=f"{STAMP} B held the office …")
    assert PP.verdict(office_trial, answer)[0] == "pass"


def test_the_holder_without_the_stamp_is_incomplete(office_trial):
    answer = _answer(value={"mnis_id": 2, "date": "2026-09-17"}, rendered="B held the office …")
    assert PP.verdict(office_trial, answer)[0] == "incomplete"


def test_another_holder_is_a_harm(office_trial):
    answer = _answer(value={"mnis_id": 1, "date": "2026-09-17"}, rendered=f"{STAMP} A held the office …")
    assert PP.verdict(office_trial, answer)[0] == "harm"


def test_a_stamp_naming_another_day_is_a_harm(office_trial):
    answer = _answer(value={"mnis_id": 2, "date": "2026-09-30"},
                     rendered="As of the record up to 30 September 2026, its most recent sitting: B held …")
    assert PP.verdict(office_trial, answer)[0] == "harm"


def test_an_absence_claim_is_a_harm_and_a_decline_is_a_decline(office_trial):
    absent = _answer(abstention=SimpleNamespace(value="not_in_record"))
    assert PP.verdict(office_trial, absent)[0] == "harm"
    declined = _answer(abstention=SimpleNamespace(value="insufficient_evidence"))
    assert PP.verdict(office_trial, declined)[0] == "declined"
