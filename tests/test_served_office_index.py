"""The index under `served.member_office`'s office key holds the view's own expression."""

from __future__ import annotations

import re

import pytest

from app import safety, sealed


def _view_key_expression(conn) -> str:
    definition = conn.execute("SELECT pg_get_viewdef('served.member_office'::regclass, true)").fetchone()[0]
    found = re.search(r"(regexp_replace\(lower\(btrim\(\w+\.speaker_office\)\).*?\)) AS office_key", definition, re.S)
    assert found, definition
    return re.sub(r"\b\w+\.speaker_office\b", "speaker_office", found.group(1))


def _index(conn) -> str:
    row = conn.execute("SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'speech_office_key_idx'"
                       ).fetchone()  # fmt: skip
    assert row, "migration 031's index is missing"
    return row[0]


def test_the_index_holds_the_view_s_own_office_key_expression(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    assert f"({_view_key_expression(scratch)})" in _index(scratch)


def test_the_index_holds_the_view_s_own_conditions(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    printed = scratch.execute("SELECT pg_get_viewdef('served.member_office'::regclass, true)").fetchone()[0]
    definition = re.sub(r"\b\w+\.(mnis_id|speaker_office)\b", r"\1", printed)
    index = _index(scratch)
    for condition in ("mnis_id IS NOT NULL", "speaker_office IS NOT NULL", "btrim(speaker_office) <> ''::text"):
        assert condition in index and condition in definition, (condition, index, definition)


def test_a_served_office_lookup_can_read_the_index(scratch, scratch_dsn):
    """With sequential scans priced out, a usable index is chosen; one that no longer matches the view cannot be. The
    speeches are the test's own and freshly analysed: which index the planner prefers depends on the statistics, and
    other tests' rows had left a full run's statistics unlike an isolated run's.
    """
    safety.require_test(scratch_dsn)
    if not sealed.masked():
        pytest.skip("the mask is lifted: the served views are not on the search path")
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WX', 'v2', '2020-03-01', '2020-03-31', 'dev')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                    "VALUES ('ix1', '2020-03-04', 'Commons', 'Questions', 'Commons Chamber', 'WX', 'v2')")
    scratch.execute("INSERT INTO speech (id, debate_id, seq, mnis_id, speaker_name, speaker_office, text, window_id) "
                    "SELECT 'ix1-' || n, 'ix1', n, 7000 + mod(n, 50), 'A Member', 'Minister ' || mod(n, 400), 'Words.', "
                    "'WX' FROM generate_series(1, 4000) AS n")
    scratch.execute("ANALYZE speech")
    scratch.execute(sealed.serving_sql())
    scratch.execute("SET enable_seqscan = off")
    plan = "\n".join(r[0] for r in scratch.execute(
        "EXPLAIN SELECT * FROM member_office WHERE office_key = 'prime minister' AND start_date <= '2020-03-04'"))
    scratch.execute("RESET enable_seqscan")
    assert "speech_office_key_idx" in plan, plan
