"""Old ordinary trace rows fold into a daily aggregate; evaluation rows and recent rows never move."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import psycopg
import pytest

from app import safety

_spec = importlib.util.spec_from_file_location("trace_retention", Path(__file__).resolve().parents[1] / "scripts" / "trace_retention.py")
TR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TR)


@pytest.fixture
def traces(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE answer_trace, answer_trace_daily")

    def add(days_ago: float, *, evaluation=False, route="retrieval", outcome="answer", ms=100, cost=0.001, calls=1):
        scratch.execute(
            "INSERT INTO answer_trace (query_hash, route, outcome, corpus_version, retrieval_config_hash, is_evaluation, "
            "total_ms, cost_usd, llm_calls, prompt_tokens, completion_tokens, degraded, cache_hit, created_at) "
            "VALUES ('h', %s, %s, 'test', 'cfg', %s, %s, %s, %s, 1000, 100, false, false, "
            "now() - make_interval(secs => %s))",
            (route, outcome, evaluation, ms, cost, calls, days_ago * 86400))
    return add


def rows(conn, sql):
    return conn.execute(sql).fetchall()


def test_old_ordinary_rows_are_folded_and_deleted_and_the_rest_stay(traces, scratch, scratch_dsn):
    for _ in range(3):
        traces(200)
    traces(200, outcome="abstain", ms=300)
    traces(200, evaluation=True)
    traces(5)
    result = TR.fold(scratch_dsn, 90)
    assert result == {"due": 4, "folded": 2, "deleted": 4}
    left = rows(scratch, "SELECT is_evaluation, count(*) FROM answer_trace GROUP BY 1 ORDER BY 1")
    assert left == [(False, 1), (True, 1)]
    daily = dict((r[0], r[1:]) for r in rows(scratch, "SELECT outcome, requests, total_ms_sum, total_ms_max, llm_calls "
                                                       "FROM answer_trace_daily"))
    assert daily == {"answer": (3, 300, 100, 3), "abstain": (1, 300, 300, 1)}


def test_evaluation_rows_are_never_folded_however_old(traces, scratch, scratch_dsn):
    traces(2000, evaluation=True)
    assert TR.fold(scratch_dsn, 1)["due"] == 0
    assert rows(scratch, "SELECT count(*) FROM answer_trace")[0][0] == 1


def test_a_second_run_folds_nothing_and_a_late_row_merges_additively(traces, scratch, scratch_dsn):
    traces(200)
    TR.fold(scratch_dsn, 90)
    assert TR.fold(scratch_dsn, 90) == {"due": 0, "folded": 0, "deleted": 0}
    traces(200)  # a late row for a day already folded
    TR.fold(scratch_dsn, 90)
    assert rows(scratch, "SELECT sum(requests) FROM answer_trace_daily")[0][0] == 2


def test_a_dry_run_changes_nothing(traces, scratch, scratch_dsn):
    traces(200)
    assert TR.fold(scratch_dsn, 90, dry_run=True) == {"due": 1, "folded": 0, "deleted": 0}
    assert rows(scratch, "SELECT count(*) FROM answer_trace")[0][0] == 1
    assert rows(scratch, "SELECT count(*) FROM answer_trace_daily")[0][0] == 0


def test_only_whole_days_are_folded(traces, scratch, scratch_dsn):
    """The cutoff is midnight N days back: a row from later on the boundary day waits for the next run."""
    traces(90.2)  # inside the boundary day or later, depending on the hour: never before midnight N days back
    traces(91.5)  # always before it
    result = TR.fold(scratch_dsn, 90)
    assert result["deleted"] in (1, 2)
    assert rows(scratch, "SELECT count(*) FROM answer_trace WHERE created_at >= date_trunc('day', now()) - interval '90 days'")[0][0] \
        == 2 - result["deleted"]


@pytest.mark.parametrize("days", [0, -1])
def test_a_retention_of_no_days_is_refused(scratch_dsn, days):
    with pytest.raises(SystemExit):
        TR.fold(scratch_dsn, days)


def test_the_rollback_is_refused():
    with pytest.raises(safety.WrongDatabase):
        TR.fold("postgresql://u:p@127.0.0.1:5432/hansard", 90)


def test_a_failed_fold_leaves_both_tables_as_they_were(traces, scratch, scratch_dsn, monkeypatch):
    traces(200)
    monkeypatch.setattr(TR, "SQL_DELETE", "DELETE FROM answer_trace WHERE no_such_column")
    with pytest.raises(psycopg.errors.UndefinedColumn):
        TR.fold(scratch_dsn, 90)
    assert rows(scratch, "SELECT count(*) FROM answer_trace")[0][0] == 1
    assert rows(scratch, "SELECT count(*) FROM answer_trace_daily")[0][0] == 0
