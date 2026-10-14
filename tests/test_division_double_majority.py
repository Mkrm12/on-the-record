"""An English Votes for English Laws division is stored as recorded, and its result is never guessed."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from answer import facts as F
from app import safety
from ingest.divisions import DIVISION_COLUMNS, parse_commons, parse_lords, store
from ingest.runtime import run
from test_divisions import commons, lords, member

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "db" / "migrations" / "022_division_double_majority.sql"


def evel(aye_count, no_count, dm_ayes, dm_noes, kind="some EVEL type", country="England", **extra):
    return parse_commons(commons(aye_count=aye_count, no_count=no_count, EVELType=kind, EVELCountry=country,
                                 DoubleMajorityAyeCount=dm_ayes, DoubleMajorityNoCount=dm_noes, **extra))


# --- parsing ----------------------------------------------------------------------------------------------------


def test_the_double_majority_is_stored_in_columns_not_only_in_raw():
    d = evel(300, 200, 250, 150).division
    assert (d["evel_type"], d["evel_country"], d["double_majority_ayes"], d["double_majority_noes"]) == \
        ("some EVEL type", "England", 250, 150)
    assert d["result"] == "agreed" and not any("double_majority" in a for a in d["anomalies"])


def test_every_row_still_has_exactly_the_declared_columns():
    assert set(evel(300, 200, 250, 150).division) == set(DIVISION_COLUMNS)
    assert set(parse_lords(lords()).division) == set(DIVISION_COLUMNS)


@pytest.mark.parametrize("whole, subset, want", [
    ((300, 200), (100, 150), "whole=agreed,subset=negatived"),
    ((200, 300), (150, 100), "whole=negatived,subset=agreed"),
    ((250, 250), (150, 100), "whole=tied,subset=agreed"),
    ((300, 200), (120, 120), "whole=agreed,subset=tied"),
])  # fmt: skip
def test_majorities_that_disagree_withhold_the_result_and_say_why(whole, subset, want):
    d = evel(*whole, *subset).division
    assert d["result"] is None
    assert f"double_majority_disagrees:{want}" in d["anomalies"]


def test_majorities_that_agree_keep_the_result():
    assert evel(200, 300, 150, 180).division["result"] == "negatived"


@pytest.mark.parametrize("dm_ayes, dm_noes", [(250, None), (None, 150)])
def test_half_a_double_majority_withholds_the_result(dm_ayes, dm_noes):
    d = evel(300, 200, dm_ayes, dm_noes).division
    assert d["result"] is None and "double_majority_incomplete" in d["anomalies"]


def test_an_ordinary_division_is_parsed_exactly_as_before():
    d = parse_commons(commons(ayes=[member(1), member(2), member(3)], noes=[member(4), member(5)])).division
    assert (d["evel_type"], d["evel_country"], d["double_majority_ayes"], d["double_majority_noes"]) == (None,) * 4
    assert d["result"] == "agreed" and d["anomalies"] == []


def test_blank_evel_strings_are_absence():
    d = evel(300, 200, None, None, kind="  ", country="").division
    assert d["evel_type"] is None and d["evel_country"] is None and d["result"] == "agreed"


def test_the_raw_payload_still_keeps_the_source_fields():
    raw = evel(300, 200, 250, 150).division["raw"]
    assert raw["DoubleMajorityAyeCount"] == 250 and raw["EVELCountry"] == "England"


# --- storage ------------------------------------------------------------------------------------------------------


def stored_row(dsn, parsed):
    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.transaction(force_rollback=True):
                await store(conn, parsed)
                cur = await conn.execute("SELECT evel_type, evel_country, double_majority_ayes, double_majority_noes, "
                                         "result, anomalies FROM division WHERE id = %s", (parsed.division["id"],))
                return await cur.fetchone()
    return run(go())


def test_the_loader_writes_the_columns(scratch_dsn):
    safety.require_test(scratch_dsn)
    row = stored_row(scratch_dsn, evel(300, 200, 100, 150, DivisionId=999_201, ayes=[member(1)]))
    assert row[:5] == ("some EVEL type", "England", 100, 150, None)
    assert "double_majority_disagrees:whole=agreed,subset=negatived" in row[5]


@pytest.mark.parametrize("sql", [
    "INSERT INTO division (id, source_id, house, division_date, double_majority_ayes) "
    "VALUES (999301, 999301, 'Commons', '2016-01-05', -1)",
    "INSERT INTO division (id, source_id, house, division_date, double_majority_ayes, double_majority_noes) "
    "VALUES (1999302, 999302, 'Lords', '2016-01-05', 10, 5)",
])  # fmt: skip
def test_the_schema_refuses_an_impossible_double_majority(scratch, sql):
    with pytest.raises(psycopg.errors.CheckViolation):
        scratch.execute(sql)


def test_the_migration_backfills_rows_loaded_before_it_and_is_idempotent(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("""INSERT INTO division (id, source_id, house, division_date, ayes, noes, result, raw) VALUES
        (999401, 999401, 'Commons', '2016-01-05', 300, 200, 'agreed',
         '{"EVELType": "some EVEL type", "EVELCountry": "England", "DoubleMajorityAyeCount": 250,
           "DoubleMajorityNoCount": 150}'),
        (999402, 999402, 'Commons', '2024-01-05', 300, 200, 'agreed',
         '{"EVELType": "", "EVELCountry": "", "DoubleMajorityAyeCount": null, "DoubleMajorityNoCount": null}')""")
    for _ in range(2):
        scratch.execute(MIGRATION.read_text(encoding="utf-8"))
    rows = scratch.execute("SELECT id, evel_type, evel_country, double_majority_ayes, double_majority_noes "
                           "FROM division WHERE id IN (999401, 999402) ORDER BY id").fetchall()
    assert rows == [(999401, "some EVEL type", "England", 250, 150), (999402, None, None, None, None)]


# --- what a reader is told ----------------------------------------------------------------------------------------


def fact(dsn, division_id):
    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            return await F.division_result(conn, division_id)
    return run(go())


@pytest.fixture
def divisions(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    # Name lists that reconcile with the tellers, so the only caveat in play is the one under test.
    scratch.execute("""INSERT INTO division (id, source_id, house, number, division_date, ayes, noes, result,
                                             ayes_listed, noes_listed, evel_type, evel_country, double_majority_ayes,
                                             double_majority_noes)
        VALUES (999501, 999501, 'Commons', 12, '2016-01-05', 300, 200, NULL, 300, 200, 'some EVEL type', 'England',
                100, 150),
               (999502, 999502, 'Commons', 13, '2016-01-05', 300, 200, 'agreed', 300, 200, 'some EVEL type', 'England',
                250, 150),
               (999503, 999503, 'Commons', 14, '2024-01-05', 300, 200, 'agreed', 300, 200, NULL, NULL, NULL, NULL)""")
    return scratch_dsn


def test_a_withheld_result_is_never_rendered_as_none(divisions):
    got = fact(divisions, 999501)
    text = got.sentence()
    assert "None" not in text and "does not give a single result" in text
    assert "300" in text and "200" in text and "100" in text and "150" in text, "both counts, so the reader can see why"


def test_a_double_majority_division_says_both_counts(divisions):
    text = fact(divisions, 999502).sentence()
    assert "agreed to" in text and "250" in text and "150" in text and "England" in text


def test_an_ordinary_division_reads_exactly_as_before(divisions):
    assert fact(divisions, 999503).sentence() == \
        "The Commons divided on 5 January 2024 (division 14) and the motion was agreed to, 300 votes to 200."
