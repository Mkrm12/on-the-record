"""The sealed W5 window, one definition, masked everywhere until the W5 exams."""

from __future__ import annotations

import asyncio
from datetime import date, datetime

import pytest

from app import sealed as S


def test_the_window_is_w5_as_the_corpus_defines_it():
    w = S.window()
    assert (w.start, w.end, w.window_id) == (date(2026, 4, 1), date(2026, 7, 31), "W5")


@pytest.mark.parametrize("day,inside", [
    (date(2026, 4, 1), True), (date(2026, 7, 31), True), (date(2026, 5, 20), True),
    (date(2026, 3, 31), False), (date(2026, 8, 1), False), (date(2026, 9, 15), False),
    (datetime(2026, 6, 1, 12, 30), True), ("2026-06-01", True), ("2026-06-01T09:00:00", True), ("2025-06-01", False),
    (None, False), ("", False), ("June 2026", False), ("not a date", False), (20260601, False),
])
def test_is_sealed_takes_every_shape_a_date_arrives_in(day, inside):
    assert S.is_sealed(day) is inside


@pytest.mark.parametrize("start,end,overlaps,inside", [
    (date(2026, 5, 1), date(2026, 5, 31), True, True),  # all of it inside: answer with the scope statement
    (date(2026, 1, 1), date(2026, 12, 31), True, False),  # a year that spans it: answer the rest, masked
    (date(2026, 8, 1), date(2026, 9, 30), False, False),
    (None, date(2026, 5, 1), True, False),  # open-ended ranges overlap but are never inside
    (date(2026, 5, 1), None, True, False),
    (None, None, False, False),
])
def test_ranges_overlap_or_fall_inside(start, end, overlaps, inside):
    assert S.overlaps(start, end) is overlaps
    assert S.inside(start, end) is inside


def test_the_mask_is_on_unless_the_exam_lifts_it_and_only_for_its_own_run():
    assert S.masked()
    with S.lifted_for_exam("the W5 exam, retrieval level"):
        assert not S.masked()
    assert S.masked()


def test_a_lift_needs_a_reason_and_cannot_outlive_its_run():
    with pytest.raises(ValueError):
        with S.lifted_for_exam(""):
            pass
    with pytest.raises(RuntimeError), S.lifted_for_exam("the W5 exam"):
        raise RuntimeError("the run failed")
    assert S.masked()  # an exception inside the run still restores the mask


def test_a_lift_is_local_to_its_task():
    """ContextVar, not a global: a lift inside one task never unmasks a concurrent one."""
    seen = {}

    async def exam():
        with S.lifted_for_exam("the W5 exam"):
            await asyncio.sleep(0.01)
            seen["exam"] = S.masked()

    async def reader():
        await asyncio.sleep(0.005)
        seen["reader"] = S.masked()

    async def both():
        await asyncio.gather(exam(), reader())

    asyncio.run(both())
    assert seen == {"exam": False, "reader": True}


def test_the_sql_predicate_excludes_the_window_fails_closed_and_vanishes_when_lifted():
    sql, params = S.sql_exclusion("c.debate_date")
    assert sql == " AND (c.debate_date < %(sealed_from)s OR c.debate_date > %(sealed_to)s)"  # NULL: withheld
    assert params == {"sealed_from": date(2026, 4, 1), "sealed_to": date(2026, 7, 31)}
    with S.lifted_for_exam("the W5 exam"):
        assert S.sql_exclusion("c.debate_date") == ("", {})


@pytest.mark.parametrize("text,dates", [
    ("[2026-05-20] the House divided", ["2026-05-20"]),
    ("On 20 May 2026 the Lords voted; on 3 September 2026 they did not", ["2026-05-20"]),
    ("April to July 2026 is not yet available", []),  # the scope statement names months, never a sealed day
    ("on 31 March 2026 and 1 August 2026", []),
    ("2026-02-30 is not a date", []),
    # review pass 2: the forms the router reads, so the guard reads them too
    ("on 20th May 2026, and again on 1st June 2026", ["2026-05-20", "2026-06-01"]),
    ("Voting on May 20, 2026 and July 3rd 2026", ["2026-05-20", "2026-07-03"]),
    ("on 20 may 2026", ["2026-05-20"]),
])
def test_sealed_days_are_found_in_text(text, dates):
    assert [d.isoformat() for d in S.sealed_days_in(text)] == dates


def test_the_scope_statement_and_the_served_label():
    assert S.SCOPE_STATEMENT == "April to July 2026 is not yet available in this record."
    assert S.served_label("v2-w0w5") == "v2-w0w5+sealed:2026-04-01..2026-07-31"
    with S.lifted_for_exam("the W5 exam"):
        assert S.served_label("v2-w0w5") == "v2-w0w5"


def test_every_older_copy_of_the_window_reads_this_one():
    """Three modules carried their own copy of the dates. They read `app.sealed` now, so they cannot drift."""
    from eval import perturb
    from goldset import instrument

    assert perturb.SEALED == (S.window().start, S.window().end)
    assert instrument.mentions_sealed_window({"answer": "voted on 20 May 2026"})
    assert not instrument.mentions_sealed_window({"answer": "voted on 20 May 2025"})
