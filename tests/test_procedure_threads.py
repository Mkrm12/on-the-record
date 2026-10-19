"""Debate threads: the records Hansard splits one sitting's business into are one debate."""

from __future__ import annotations

from datetime import date

import pytest

from procedure import threads as T

D = date(2020, 5, 20)


def _r(did, heading, order, *, house="Commons", day=D, section="Commons Chamber"):
    return T.Meta(debate_id=did, house=house, date=day, section=section, section_order=order, heading=heading)


def test_adjacent_records_under_one_heading_are_one_thread_in_record_order():
    found = T.build([_r("b", "Sovereign Grant Bill", 28), _r("a", "Sovereign Grant Bill", 27)])
    assert [t.debate_ids for t in found] == [("a", "b")] and found[0].thread_id == "a"


def test_the_same_heading_with_other_business_between_is_two_threads():
    found = T.build([_r("x", "Business of the House", 18), _r("y", "Northern Ireland Protocol: UK Approach", 19),
                     _r("z", "Business of the House", 20)])
    assert sorted(t.debate_ids for t in found) == [("x",), ("y",), ("z",)]


def test_a_gap_in_the_numbering_with_nothing_recorded_in_it_does_not_separate():
    """Record 22 on 23 March 2020 holds no contributions, so nothing recorded stands between 21 and 23."""
    found = T.build([_r("c1", "Coronavirus Bill", 21), _r("c2", "Coronavirus Bill", 23),
                     _r("ae", "A&E Departments: Staffing", 24)])
    assert sorted(t.debate_ids for t in found) == [("ae",), ("c1", "c2")]


def test_three_in_a_row_are_one_thread():
    found = T.build([_r(f"r{i}", "Student Loans", i) for i in (17, 18, 19)])
    assert [t.debate_ids for t in found] == [("r17", "r18", "r19")]


@pytest.mark.parametrize(("a", "b"), [
    ("Speaker\u2019s Statement", "Speaker's Statement"), ("Hybrid  Proceedings ", "Hybrid Proceedings"),
    ("STUDENT LOANS", "Student Loans"), ("Cyber Security and Resilience  (Network and Information Systems) Bill",
                                         "Cyber Security and Resilience (Network and Information Systems) Bill"),
])
def test_headings_are_compared_as_a_reader_reads_them(a, b):
    assert T.key(a) == T.key(b)
    assert len(T.build([_r("1", a, 1), _r("2", b, 2)])) == 1


def test_other_houses_days_and_sections_are_never_one_thread():
    found = T.build([_r("c", "Covid-19", 1), _r("l", "Covid-19", 2, house="Lords"),
                     _r("w", "Covid-19", 3, section="Westminster Hall"), _r("n", "Covid-19", 4, day=date(2020, 5, 21))])
    assert len(found) == 4


def test_a_record_with_no_place_in_the_days_order_is_never_merged():
    found = T.build([_r("p", "Covid-19", None), _r("q", "Covid-19", None), _r("s", "Covid-19", 3)])
    assert len(found) == 3


def test_a_record_with_no_heading_is_a_thread_of_its_own():
    found = T.build([_r("p", None, 1), _r("q", None, 2)])
    assert len(found) == 2


def test_every_record_belongs_to_exactly_one_thread():
    metas = [_r("a", "X", 1), _r("b", "X", 2), _r("c", "Y", 3), _r("d", "X", 4), _r("e", "X", 5), _r("f", None, 6)]
    found = T.build(metas)
    owned = [d for t in found for d in t.debate_ids]
    assert sorted(owned) == sorted(m.debate_id for m in metas) and len(owned) == len(set(owned))
    assert T.index(found)["e"].debate_ids == ("d", "e")


def test_no_two_threads_share_a_heading_unless_other_business_separates_them():
    metas = [_r("a", "X", 1), _r("b", "X", 2), _r("c", "Y", 3), _r("d", "X", 4)]
    assert T.violations(T.build(metas), metas) == []
    broken = [T.Thread(("a",), "Commons", D, "Commons Chamber", "X"), T.Thread(("b",), "Commons", D, "Commons Chamber",
                                                                               "X")]  # fmt: skip
    assert T.violations(broken, metas[:2]) == [("a", "b")]


# --- a bill's proceedings that day are one debate, whatever interrupts them ------------------------------ Measured on
# the masked record: 52 bill headings recur on a House-day-section with other business between, every one a stage
# resumed after a statement, an urgent question or oral questions.


def test_a_bills_proceedings_resumed_after_other_business_are_one_thread():
    found = T.build([_r("r18", "Financial Services and Markets Bill [HL]", 18, house="Lords", section="Lords Chamber"),
                     _r("st", "Local Government Reorganisation", 19, house="Lords", section="Lords Chamber"),
                     _r("r20", "Financial Services and Markets Bill [HL]", 20, house="Lords", section="Lords Chamber"),
                     _r("oq", "Sudan", 21, house="Lords", section="Lords Chamber"),
                     _r("r22", "Financial  Services and Markets Bill [HL]", 22, house="Lords", section="Lords Chamber")])
    assert sorted(t.debate_ids for t in found) == [("oq",), ("r18", "r20", "r22"), ("st",)]


@pytest.mark.parametrize("heading", ["Crime and Policing Bill", "Sovereign Grant Bill [Lords]", "Bus Services (No. 2) Bill"])
def test_every_form_of_a_bills_heading_joins(heading):
    found = T.build([_r("a", heading, 1), _r("b", "A Statement", 2), _r("c", heading, 3)])
    assert sorted(t.debate_ids for t in found) == [("a", "c"), ("b",)]


@pytest.mark.parametrize("heading", ["Business of the House", "Covid-19", "Crime and Policing Bill (Money)",
                                     "Crime and Policing Bill: Programme", "Billing Practices"])
def test_any_other_heading_separated_by_business_stays_separate(heading):
    found = T.build([_r("a", heading, 1), _r("b", "A Statement", 2), _r("c", heading, 3)])
    assert len(found) == 3


def test_a_bill_in_two_sections_that_day_is_two_threads():
    """The Lords Chamber and a Grand Committee sit apart: two sittings, never one debate."""
    found = T.build([_r("a", "Crime and Policing Bill", 1, section="Lords Chamber"),
                     _r("b", "Crime and Policing Bill", 1, section="Grand Committee")])
    assert len(found) == 2
