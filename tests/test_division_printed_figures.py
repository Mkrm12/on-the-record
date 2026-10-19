"""Hansard's printed figures beside the Votes API's, and answers that never choose between them silently."""

from __future__ import annotations

from datetime import date

import psycopg
import pytest

from answer import contract as K
from answer import facts as F
from answer import pipeline as PL
from answer import router as R
from app import safety
from ingest import printed_figures as PF
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e", recipe="r",
            prompt_version="p", model_id="m")


def record(house="Commons", day="2026-02-23T17:05:00", number="438", ayes=73, noes=286, **extra):
    return {"House": house, "Date": day, "Number": number, "AyesCount": ayes, "NoesCount": noes, **extra}


def test_the_printed_figures_are_read_by_house_day_and_number():
    found = PF.parse([record(), record(house="Lords", number="2", ayes=247, noes=165, day="2026-09-07T16:00:00")])
    assert found == {("Commons", date(2026, 2, 23), "438"): (73, 286), ("Lords", date(2026, 9, 7), "2"): (247, 165)}


def test_a_record_without_both_figures_or_dated_in_the_sealed_window_is_left_out():
    assert PF.parse([record(ayes=None), record(noes=None), record(day="2026-05-20T15:00:00")]) == {}


def test_two_records_under_one_key_that_disagree_are_left_out_never_picked():
    assert PF.parse([record(), record(noes=287)]) == {}
    assert PF.parse([record(), record()]) == {("Commons", date(2026, 2, 23), "438"): (73, 286)}


@pytest.fixture
def divisions(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    for vid, number, day, ayes, noes, result in ((2264, 438, "2026-02-23", 73, 256, "negatived"),
                                                 (2270, 439, "2026-02-23", 300, 200, "agreed")):
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed) VALUES (%s, %s, 'Commons', %s, %s, %s, %s, %s, %s, %s, %s)",
                        (vid, vid, number, day, f"Title {number}", ayes, noes, result, ayes, noes))  # fmt: skip
    return scratch


def test_the_schema_holds_both_or_neither_and_the_served_view_shows_them(divisions):
    with pytest.raises(psycopg.errors.CheckViolation):
        divisions.execute("UPDATE division SET printed_ayes = 73 WHERE id = 2264")
    divisions.execute("UPDATE division SET printed_ayes = 73, printed_noes = 286 WHERE id = 2264")
    columns = {r[0] for r in divisions.execute("SELECT column_name FROM information_schema.columns "
                                               "WHERE table_schema = 'served' AND table_name = 'division'")}
    assert {"printed_ayes", "printed_noes"} <= columns


def test_the_loader_writes_each_divisions_printed_figures_once_and_reports_what_it_matched(divisions):
    figures = PF.parse([record(), record(number="439", ayes=300, noes=200), record(number="999")])
    report = PF.apply(divisions, figures)
    assert report == {"matched": 2, "unmatched": 1, "differ": 1, "written": 2}
    rows = dict((r[0], (r[1], r[2])) for r in divisions.execute("SELECT id, printed_ayes, printed_noes FROM division"))
    assert rows == {2264: (73, 286), 2270: (300, 200)}
    assert PF.apply(divisions, figures)["written"] == 0, "idempotent"


def ask(dsn, question):
    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()

    return run(go())


@pytest.mark.parametrize("question", ["What were the Ayes and Noes in the Commons division on Title 438 on 23 February 2026?",
                                      "What was the result of the Commons division on Title 438 on 23 February 2026?"])
def test_where_the_sources_part_the_answer_states_both(divisions, scratch_dsn, question):
    divisions.execute("UPDATE division SET printed_ayes = 73, printed_noes = 286 WHERE id = 2264")
    answer, notes = ask(scratch_dsn, question)
    assert answer is not None, notes
    text = answer.rendered
    assert "286" in text and "256" in text and "Hansard" in text and "Votes" in text, text


def test_where_they_agree_nothing_is_added(divisions, scratch_dsn):
    divisions.execute("UPDATE division SET printed_ayes = 300, printed_noes = 200 WHERE id = 2270")
    answer, _ = ask(scratch_dsn, "What were the Ayes and Noes in the Commons division on Title 439 on 23 February 2026?")
    assert "Hansard" not in answer.rendered


def test_the_caveat_is_one_sentence_built_from_the_figures_alone():
    assert F.sources_caveat({"ayes": 73, "noes": 256, "printed_ayes": 73, "printed_noes": 286}) == (
        "The record's two sources differ on this division's figures: Hansard prints 73 Ayes and 286 Noes; the "
        "Votes service records 73 and 256.")
    assert F.sources_caveat({"ayes": 1, "noes": 2, "printed_ayes": 1, "printed_noes": 2}) == ""
    assert F.sources_caveat({"ayes": 1, "noes": 2}) == ""
    assert "Contents" in F.sources_caveat({"house": "Lords", "ayes": 246, "noes": 165, "printed_ayes": 247,
                                           "printed_noes": 165})
