"""The coverage table, against a scratch Postgres holding a corpus this test defines."""

from __future__ import annotations

from datetime import date

import pytest

from answer import coverage as C
from ingest.runtime import run as run_async


@pytest.fixture
def coverage_db(scratch, scratch_dsn):
    """Two windows years apart, three sections, a recess, an unmapped speaker and one unreconciled division."""
    c = scratch
    c.execute("TRUNCATE coverage, corpus_window, debate, speech, division, division_vote, chunk CASCADE")
    c.execute("""INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role)
                 VALUES ('WA', 'v2', '2020-03-01', '2020-05-31', 'dev'),
                        ('WB', 'v2', '2026-04-01', '2026-04-30', 'heldout')""")
    # WA: a sitting on 2 March, then a 27-day hole, then 21 April — the shape of the covid recess.
    c.execute("""INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version)
                 VALUES ('a1', '2020-03-02', 'Commons', 'Business', 'Commons Chamber', 'WA', 'v2'),
                        ('a2', '2020-03-25', 'Commons', 'Business', 'Commons Chamber', 'WA', 'v2'),
                        ('a3', '2020-04-21', 'Commons', 'Business', 'Commons Chamber', 'WA', 'v2'),
                        ('a4', '2020-03-02', 'Commons', 'Debate',   'Westminster Hall', 'WA', 'v2'),
                        ('a5', '2020-03-02', 'Lords',   'Business', 'Lords Chamber',    'WA', 'v2'),
                        ('b1', '2026-04-15', 'Commons', 'Business', 'Commons Chamber', 'WB', 'v2')""")
    c.execute("""INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, is_boilerplate, window_id)
                 VALUES ('s1','a1',1,'A Member',1001,'One.',false,'WA'),
                        ('s2','a1',2,'B Member',1002,'Two.',false,'WA'),
                        ('s3','a1',3,'The Speaker',NULL,'Order.',true,'WA'),
                        ('s4','a2',1,'A Member',1001,'Three.',false,'WA'),
                        ('s5','a2',2,'An Unmapped Member',NULL,'Four.',false,'WA'),
                        ('s6','a3',1,'A Member',1001,'Five.',false,'WA'),
                        ('s7','a4',1,'B Member',1002,'Six.',false,'WA'),
                        ('s8','a5',1,'A Peer',1003,'Seven.',false,'WA'),
                        ('s9','b1',1,'A Member',1001,'Eight.',false,'WB')""")
    c.execute("""INSERT INTO division (id, source_id, debate_id, house, number, division_date, title,
                                      ayes, noes, result, ayes_listed, noes_listed, tally_reconciled)
                 VALUES (1, 1, 'a1', 'Commons', 1, '2020-03-02', 'D1', 10, 8, 'agreed', 12, 10, true),
                        (2, 2, 'a2', 'Commons', 2, '2020-03-25', 'D2', 10, 8, 'agreed', 11, 9, false)""")
    c.execute("""INSERT INTO division_vote (division_id, mnis_id, lobby)
                 VALUES (1, 1001, 'aye'), (1, 1002, 'no'), (2, 1001, 'aye')""")
    yield scratch_dsn
    c.execute("TRUNCATE coverage, corpus_window, debate, speech, division, division_vote, chunk CASCADE")


def call(fn, dsn, *args, **kwargs):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            result = await fn(conn, *args, **kwargs)
            await conn.commit()
            return result
        finally:
            await conn.close()

    return run_async(go())


def refreshed(dsn, window=None):
    call(C.refresh, dsn, window)
    return call(C.load, dsn)


def row_for(rows, window, house, section):
    return next(r for r in rows if (r.window_id, r.house, r.section) == (window, house, section))


# --- the refresh gets the arithmetic right ---------------------------------------------------------------


def test_one_row_per_window_house_and_section(coverage_db):
    rows = refreshed(coverage_db)
    assert {(r.window_id, r.house, r.section) for r in rows} == {
        ("WA", "Commons", "Commons Chamber"),
        ("WA", "Commons", "Westminster Hall"),
        ("WA", "Lords", "Lords Chamber"),
        ("WB", "Commons", "Commons Chamber"),
    }


def test_debates_counts_debates_not_debate_times_speech_rows(coverage_db):
    """The bug this pins. The aggregate LEFT JOINs speech, so `count(*)` returned the speech count as the debate count
    — and `stale` then flagged every window as out of date immediately after a successful refresh.
    """
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Commons Chamber")
    assert row.debates == 3, "a1, a2, a3 — not the six speeches inside them"
    assert row.sitting_days == 3


def test_contributions_exclude_boilerplate_and_speeches_do_not(coverage_db):
    """Both are stored. A per-window total that counted "Order." as a contribution would inflate every roster and every
    count template that reads from here.
    """
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Commons Chamber")
    assert row.speeches == 6
    assert row.contributions == 5
    assert row.boilerplate == 1


def test_unattributed_speeches_are_counted_rather_than_hidden(coverage_db):
    """A named speaker the mapper could not resolve is the reason a roster can be incomplete."""
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Commons Chamber")
    assert row.unattributed == 1, "s5 has a name and no mnis id"
    assert row.speakers == 2, "distinct mnis ids among non-boilerplate speeches"


def test_the_first_and_last_sitting_are_measured_not_declared(coverage_db):
    """A window declared to 31 May whose last sitting is 21 April is correct, and worth being able to see."""
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Commons Chamber")
    assert row.declared_start == date(2020, 3, 1) and row.declared_end == date(2020, 5, 31)
    assert row.first_sitting == date(2020, 3, 2) and row.last_sitting == date(2020, 4, 21)


def test_divisions_and_the_i8_count_are_per_section(coverage_db):
    """"Is this window's division data trustworthy" should be answerable from here without re-deriving it."""
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Commons Chamber")
    assert row.divisions == 2
    assert row.division_votes == 3
    assert row.unreconciled_divisions == 1


def test_the_role_is_denormalised_so_a_heldout_window_is_one_predicate(coverage_db):
    rows = refreshed(coverage_db)
    assert row_for(rows, "WB", "Commons", "Commons Chamber").heldout is True
    assert row_for(rows, "WA", "Commons", "Commons Chamber").heldout is False


def test_a_heldout_window_is_still_in_scope_but_can_be_excluded_from_metrics(coverage_db):
    """W5 is in the corpus, so a question inside it IS in scope. What must not happen is reporting its numbers, which
    is `Row.heldout`'s job rather than the scope filter's.
    """
    call(C.refresh, coverage_db)
    assert any(r.window_id == "WB" for r in call(C.load, coverage_db))
    assert all(r.window_id != "WB" for r in call(C.load, coverage_db, None, False))


def test_refresh_is_an_upsert_and_can_target_one_window(coverage_db):
    """Delete-and-insert would let a reader see a corpus with a window missing, and "we cover nothing for W4" is a
    worse answer than a stale count.
    """
    refreshed(coverage_db)
    call(C.refresh, coverage_db, "WA")
    rows = call(C.load, coverage_db)
    assert len(rows) == 4, "refreshing one window leaves the others in place"


def test_a_freshly_refreshed_table_is_not_stale(coverage_db):
    refreshed(coverage_db)
    assert call(C.stale, coverage_db) == []


def test_a_window_that_gained_debates_is_reported_stale(coverage_db):
    """The check the nightly delta needs: an unrefreshed window reports numbers quietly too small, and nothing else in
    the system would notice.
    """
    refreshed(coverage_db)
    import psycopg

    with psycopg.connect(coverage_db, autocommit=True) as conn:
        conn.execute("""INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version)
                        VALUES ('a9', '2020-05-06', 'Commons', 'New', 'Commons Chamber', 'WA', 'v2')""")
    assert call(C.stale, coverage_db) == ["WA"]


# --- the gap, which is the point of the table ------------------------------------------------------------


def test_the_longest_gap_is_found_with_its_dates(coverage_db):
    """The real W3 shape: the last March 2020 sitting is the 25th and the next is 21 April. That is a fact about 2020,
    not a fetch failure, and reporting it as a gap is how the system says so.
    """
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Commons Chamber")
    assert row.max_gap_days == 27
    assert row.max_gap_start == date(2020, 3, 25)
    assert row.max_gap_end == date(2020, 4, 21)


def test_a_section_with_one_sitting_has_no_gap(coverage_db):
    """Not a gap of zero, and not a gap spanning the window: with one sitting there is nothing to measure."""
    row = row_for(refreshed(coverage_db), "WA", "Commons", "Westminster Hall")
    assert row.max_gap_days is None


def test_gaps_are_listed_longest_first_above_a_threshold(coverage_db):
    refreshed(coverage_db)
    found = call(C.gaps, coverage_db, 14)
    assert [g["days"] for g in found] == [27]
    assert call(C.gaps, coverage_db, 28) == []


# --- the three standings --------------------------------------------------------------------------------


def test_a_sitting_day_is_reported_as_sitting(coverage_db):
    refreshed(coverage_db)
    result = call(C.standing, coverage_db, date(2020, 3, 2), "Commons")
    assert result["standing"] is C.Standing.SITTING
    assert result["windows"] == ["WA"], "deduplicated: one row per section matched the same window"


def test_a_day_inside_a_window_when_the_house_did_not_sit_is_not_outside_the_corpus(coverage_db):
    """The middle case, and the one most easily lost. 1 April 2020 is inside the window and inside the covid recess.
    Calling it `outside_window` would tell a reader the corpus does not cover a date it covers perfectly well;
    calling it a sitting day would invite an answer about a day with no proceedings.
    """
    refreshed(coverage_db)
    result = call(C.standing, coverage_db, date(2020, 4, 1), "Commons")
    assert result["standing"] is C.Standing.NO_SITTING
    assert result["windows"] == ["WA"]


def test_a_date_between_two_windows_is_outside_the_corpus(coverage_db):
    """The reason this table stores one row per window and not a span. 2023 sits inside
    min(first_sitting)..max(last_sitting) and inside no window, and a `BETWEEN` check would answer it confidently
    from the wrong decade.
    """
    refreshed(coverage_db)
    result = call(C.standing, coverage_db, date(2023, 6, 1), "Commons")
    assert result["standing"] is C.Standing.OUTSIDE_CORPUS


def test_an_out_of_scope_answer_says_what_is_covered_instead(coverage_db):
    """"Outside the window" is only actionable when the user can be told what they could ask about."""
    refreshed(coverage_db)
    result = call(C.standing, coverage_db, date(2023, 6, 1), "Commons")
    covered = {w["window_id"] for w in result["covered"]}
    assert covered == {"WA", "WB"}


def test_a_date_inside_a_window_but_in_the_other_house_is_judged_per_house(coverage_db):
    """WB has Commons sittings and no Lords ones, so April 2026 is outside the corpus for the Lords."""
    refreshed(coverage_db)
    assert call(C.standing, coverage_db, date(2026, 4, 15), "Commons")["standing"] is C.Standing.SITTING
    assert call(C.standing, coverage_db, date(2026, 4, 15), "Lords")["standing"] is C.Standing.OUTSIDE_CORPUS


def test_a_window_covers_a_declared_date_beyond_its_last_sitting(coverage_db):
    """Bounded by the declared window, not by the last sitting: the corpus was fetched for 20 May 2020 and there were
    no proceedings, which is a different statement from "out of scope".
    """
    refreshed(coverage_db)
    result = call(C.standing, coverage_db, date(2020, 5, 20), "Commons")
    assert result["standing"] is C.Standing.NO_SITTING


# --- the coverage note ----------------------------------------------------------------------------------


def test_a_single_window_note_names_its_dates_and_counts(coverage_db):
    refreshed(coverage_db)
    note = call(C.note, coverage_db, "Commons", "Commons Chamber", date(2020, 1, 1), date(2020, 12, 31))
    assert "Commons Chamber" in note and "2020-03-02 to 2020-04-21" in note
    assert "3 sitting days" in note and "5 contributions" in note


def test_a_multi_window_note_never_implies_continuous_coverage(coverage_db):
    """A reader who believed the span would take a silence about 2023 to mean nothing was said."""
    refreshed(coverage_db)
    note = call(C.note, coverage_db, "Commons")
    assert "separate periods" in note
    assert "2020-03-02 to 2020-04-21" in note and "2026-04-15" in note


def test_the_note_reports_a_long_break_in_sittings(coverage_db):
    refreshed(coverage_db)
    note = call(C.note, coverage_db, "Commons", "Commons Chamber")
    assert "2020-03-25 to 2020-04-21" in note and "27 days" in note


def test_the_note_warns_when_a_roster_may_be_incomplete(coverage_db):
    """One unmapped speaker in five contributions is 17%, and a roster built over that period will be missing someone.
    Saying so is the difference between an incomplete answer and a misleading one.
    """
    refreshed(coverage_db)
    note = call(C.note, coverage_db, "Commons", "Commons Chamber")
    assert "could not be matched to a member" in note


def test_the_note_counts_contributions_not_speeches(coverage_db):
    """"Order." is not a contribution, and a note that counted it would overstate the evidence."""
    refreshed(coverage_db)
    note = call(C.note, coverage_db, "Commons", "Commons Chamber", date(2020, 1, 1), date(2020, 12, 31))
    assert "5 contributions" in note, "s1, s2, s4, s5, s6"
    assert "6 contributions" not in note, "the boilerplate 'Order.' must not be counted"


def test_an_empty_scope_says_so_rather_than_inventing_a_range(coverage_db):
    refreshed(coverage_db)
    note = call(C.note, coverage_db, "Commons", "Grand Committee")
    assert "holds nothing" in note
