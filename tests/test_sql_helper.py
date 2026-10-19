"""`scripts/sql.py`: the one way a session or a review agent reads the record by hand."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import psycopg
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sql_helper", REPO / "scripts" / "sql.py")
SQL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SQL)


def _debate(conn, did, day):
    conn.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                 "VALUES ('WT', 't', '2020-01-01', '2026-12-31', 'dev_test') ON CONFLICT DO NOTHING")
    conn.execute("INSERT INTO debate (id, date, house, heading, window_id, corpus_version) "
                 "VALUES (%s, %s, 'Commons', 'A Debate', 'WT', 't')", (did, day))


def test_a_row_inside_the_sealed_window_is_never_shown(scratch, scratch_dsn):
    _debate(scratch, "OUT", "2026-03-10")
    _debate(scratch, "IN", "2026-06-01")  # inside W5 (config/sealed.toml)
    columns, rows = SQL.query("SELECT id FROM debate ORDER BY id", dsn=scratch_dsn)
    assert columns == ["id"] and rows == [("OUT",)]
    assert scratch.execute("SELECT count(*) FROM debate").fetchone() == (2,), "the planted row is there"


@pytest.mark.parametrize("sql", ["SELECT * FROM public.debate", "SELECT * FROM PUBLIC . speech",
                                 'SELECT * FROM "public"."division"', "SET search_path = public; SELECT 1",
                                 "SELECT set_config('search_path', 'public', false)",
                                 # a comment is whitespace to Postgres, and a Unicode-escaped name is the same name
                                 "SELECT * FROM public/* x */.debate", "SELECT * FROM public -- x\n.debate",
                                 'SELECT * FROM U&"\\0070ublic".debate'])
def test_reaching_past_the_mask_is_refused_by_name(scratch_dsn, sql):
    """The served views mask the window only while they are first on the search path: naming `public`, or moving the
    search path, reaches past them, so the helper refuses either rather than trust the reader.
    """
    with pytest.raises(SQL.Refused):
        SQL.query(sql, dsn=scratch_dsn)


@pytest.mark.parametrize("sql", [
    # QA pass 8 (Med 3): each of these reached past the mask, the reader role or the read-only transaction
    "SET ROLE NONE; SELECT 1", "SET SCHEMA 'public'; SELECT 1", "SET \"otr\".sealed_from = '2999-01-01'; SELECT 1",
    "COMMIT; BEGIN READ WRITE; SET ROLE NONE; SELECT 1",
    "SELECT query_to_xml('SELECT id FROM pub' || 'lic.division', true, true, '')",
    "SELECT table_to_xml('debate', true, true, '')", "SELECT * FROM dblink('x', 'SELECT 1') AS t(x int)",
    # one statement, and only a read
    "SELECT 1; SELECT 2", "INSERT INTO member_id_map (mnis_id, name) VALUES (1, 'x')", "DELETE FROM debate",
    "CREATE TABLE t (x int)", "DO $$ BEGIN PERFORM 1; END $$", "RESET ROLE",
])  # fmt: skip
def test_anything_but_one_read_is_refused_before_it_is_sent(scratch_dsn, sql):
    with pytest.raises(SQL.Refused):
        SQL.query(sql, dsn=scratch_dsn)


@pytest.mark.parametrize("sql", ["SELECT 1", "  -- a note\n  select 1", "WITH x AS (SELECT 1 AS n) SELECT n FROM x",
                                 "EXPLAIN SELECT 1", "SELECT 1;"])
def test_one_read_is_run(scratch_dsn, sql):
    SQL.query(sql, dsn=scratch_dsn)


def test_a_write_inside_a_read_is_refused_by_the_server(scratch_dsn):
    """A data-modifying WITH reads as a read to the name check; the read-only transaction refuses it."""
    with pytest.raises((psycopg.errors.ReadOnlySqlTransaction, psycopg.errors.InsufficientPrivilege)):
        SQL.query("WITH gone AS (DELETE FROM debate RETURNING id) SELECT count(*) FROM gone", dsn=scratch_dsn)


def test_the_database_is_the_target_never_the_old_record(monkeypatch):
    monkeypatch.setattr(SQL.safety, "TARGET", "hansard")
    with pytest.raises(SQL.Refused):
        SQL.run("SELECT 1")


def test_output_is_capped_in_rows_and_width():
    text = SQL.render(["a"], [("x" * 500,)] * 100, rows=3, width=20)
    lines = text.splitlines()
    assert lines[0] == "a" and len(lines) == 5 and all(len(line) <= 20 for line in lines[1:4])
    assert lines[-1] == "(100 rows, 3 shown)"
