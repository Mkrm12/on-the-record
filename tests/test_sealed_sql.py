"""The independent review found that the structured routes' SQL (facts, graph, roster, the set route) read the tables
unmasked, so for them the exit guard was the only layer, though `answer.sealing` claimed three. The fix is one choke
point, in Postgres: a `served` schema of views that leave out every row dated inside the sealed window.
"""

from __future__ import annotations

import psycopg
import pytest

from app import sealed
from app.db import make_pool
from corpus import Corpus, vec

INSIDE, OUTSIDE = "2026-05-20", "2026-09-08"
# Every table a route reads that holds the record's dated content, with the view that masks it.
SERVED = ("debate", "speech", "chunk", "chunk_context", "division", "division_vote", "bill_stage",
          "bill_stage_sitting", "coverage", "corpus_window", "member_office", "bill")  # fmt: skip
# Columns a served view adds after its table's own: what the mask did to a row, for the route to say.
ADDED = {"member_office": ["start_withheld", "end_withheld"]}


@pytest.fixture
def record(scratch):
    """The same kinds of row on each side of the window, so every view has something to hide and something to show.
    """
    c = Corpus(scratch)
    c.debate("open", "Housing", date=OUTSIDE)
    c.debate("shut", "Housing", date=INSIDE)
    c.speech("open", "s_open", ["Housing policy reform is needed now."], vectors=[vec((0, 1.0))])
    c.speech("shut", "s_shut", ["Housing policy reform was urged again."], vectors=[vec((0, 1.0))])
    for did, day in ((901, OUTSIDE), (902, INSIDE)):
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, ayes, noes) "
                        "VALUES (%s, %s, 'Commons', %s, 'Housing Bill: Second Reading', 3, 2)", (did, did, day))
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby) VALUES (%s, 7, 'aye')", (did,))
    scratch.execute("INSERT INTO bill (id, title) VALUES (5001, 'Housing Bill')")
    for sid, day, first, last in ((601, OUTSIDE, None, None), (602, INSIDE, None, None), (603, None, None, None),
                                  (604, None, "2026-03-30", "2026-04-02"), (605, "2026-03-01", None, None)):
        scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house, date, first_sitting, last_sitting) "
                        "VALUES (%s, 5001, 'Report stage', 'Commons', %s, %s, %s)", (sid, day, first, last))  # fmt: skip
    for sid, day in ((601, OUTSIDE), (602, INSIDE)):
        scratch.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, debate_id) "
                        "VALUES (%s, %s, 5001, %s, %s)", (sid, sid, day, "open" if day == OUTSIDE else "shut"))  # fmt: skip
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) VALUES "
                    "('WO', 'v2', '2026-09-01', '2026-09-30', 'dev'), ('WS', 'v2', '2026-04-01', '2026-07-31', 'heldout') "
                    "ON CONFLICT (window_id) DO NOTHING")  # fmt: skip
    return scratch


def served_conn(dsn):
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute(sealed.serving_sql())
    return conn


def count(conn, table, where=""):
    return conn.execute(f"SELECT count(*) FROM {table} {where}").fetchone()[0]


def test_every_served_view_hides_the_window_and_keeps_the_rest(record, scratch_dsn):
    with served_conn(scratch_dsn) as conn:
        assert conn.execute("SHOW search_path").fetchone()[0].startswith("served")
        assert count(conn, "debate") == count(record, "debate", f"WHERE date <> '{INSIDE}'")
        assert {r[0] for r in conn.execute("SELECT id FROM speech")} == {"s_open"}
        assert {r[0] for r in conn.execute("SELECT DISTINCT debate_id FROM chunk")} == {"open"}
        assert {r[0] for r in conn.execute("SELECT DISTINCT debate_id FROM chunk_context")} <= {"open"}
        assert {r[0] for r in conn.execute("SELECT id FROM division")} == {901}
        assert {r[0] for r in conn.execute("SELECT division_id FROM division_vote")} == {901}
        assert {r[0] for r in conn.execute("SELECT date::text FROM bill_stage_sitting")} == {OUTSIDE}
        assert "WS" not in {r[0] for r in conn.execute("SELECT window_id FROM corpus_window")}


def test_a_bill_stage_is_hidden_if_its_date_or_its_sittings_touch_the_window_and_an_undated_one_is_kept(
        record, scratch_dsn):  # fmt: skip
    """604 sat from 30 March to 2 April: its sittings reach into the window. 603 has no date at all: a stage the record
    has not dated carries nothing from any window, so it stays.
    """
    with served_conn(scratch_dsn) as conn:
        assert {r[0] for r in conn.execute("SELECT id FROM bill_stage")} == {601, 603, 605}


def test_the_whole_record_is_still_there_without_the_served_schema(record, scratch_dsn):
    with sealed.lifted_for_exam("test: the lifted search path"):
        with served_conn(scratch_dsn) as conn:
            assert {r[0] for r in conn.execute("SELECT id FROM division")} == {901, 902}
            assert {r[0] for r in conn.execute("SELECT id FROM speech")} == {"s_open", "s_shut"}


def test_a_served_view_fails_closed_when_the_window_is_not_set(record, scratch_dsn):
    """A connection that put `served` on its path without saying where the window is gets an error, never rows."""
    with psycopg.connect(scratch_dsn, autocommit=True) as conn:
        with pytest.raises(psycopg.Error):
            conn.execute("SELECT count(*) FROM served.division").fetchone()


@pytest.mark.parametrize("table", SERVED)
def test_each_served_view_has_exactly_its_table_s_columns(scratch, table):
    """A view written `SELECT *` freezes its columns when it is created. A later migration that adds a column to the
    table and does not recreate the view would make every route reading that column fail: this catches the drift.
    """
    def columns(schema):
        return [r[0] for r in scratch.execute("SELECT column_name FROM information_schema.columns WHERE "
                                              "table_schema = %s AND table_name = %s ORDER BY ordinal_position",
                                              (schema, table))]  # fmt: skip
    assert columns("served") == columns("public") + ADDED.get(table, [])


def test_every_pool_says_whether_it_serves_readers():
    with pytest.raises(TypeError):
        make_pool("postgresql://u@127.0.0.1/x")  # no `served=`: refused, so no caller can forget the mask
    assert make_pool("postgresql://u@127.0.0.1/x", served=True) is not None


def test_the_answer_runtime_s_pool_serves_readers():
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "src" / "answer" / "pipeline.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "make_pool"]
    assert calls and all(any(k.arg == "served" and getattr(k.value, "value", None) is True for k in c.keywords)
                         for c in calls)  # fmt: skip


def test_the_exit_guard_still_reads_the_true_dates_on_a_serving_connection(record, scratch_dsn):
    """The guard is the last layer, so it must see what the views hide: an answer that somehow cited a sealed speech is
    still withheld on a serving connection (its queries name `public.` tables).
    """
    from answer import sealing as SG
    from answer.contract import Answer, Citation, Claim, Provenance, Verification
    from ingest.runtime import run as run_async

    prov = Provenance(route="retrieval", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                      model_id="t")  # fmt: skip
    chunk = record.execute("SELECT id FROM chunk WHERE debate_id = 'shut' LIMIT 1").fetchone()[0]
    claim = Claim(id="c1", kind="said", text="Housing policy reform was urged again.", speaker_name="Ann Member",
                  citations=(Citation(speech_id="s_shut", chunk_ids=(chunk,), quote="Housing policy reform was urged again."),),
                  verification=Verification(quote_ok=True, citation_ok=True, window_ok=True))  # fmt: skip
    answer = Answer(question="q?", provenance=prov, claims=(claim,), rendered="Housing policy reform was urged.",
                    retrieved_chunk_ids=frozenset({chunk}))  # fmt: skip

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn, autocommit=True) as conn:
            await conn.execute(sealed.serving_sql())
            return await SG.guard(conn, answer)

    guarded, reasons = run_async(go())
    assert reasons and guarded.abstention is not None


def test_the_trace_records_what_the_guard_withheld(scratch, scratch_dsn):
    from answer import trace as TR
    from ingest.runtime import run as run_async

    assert "sealed_withheld" in TR.COLUMNS

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            tid = await TR.record(conn, TR.Trace(query_hash="h", route="retrieval", corpus_version="v",
                                                 retrieval_config_hash="r", sealed_withheld=["speech s1 2026-05-20"]))
            await conn.commit()
            return tid

    tid = run_async(go())
    assert tid is not None
    assert scratch.execute("SELECT sealed_withheld FROM answer_trace WHERE id = %s", (tid,)).fetchone()[0] == [
        "speech s1 2026-05-20"]


# --- member_office (QA pass 3, finding 6) ---------------------------------------------------------------------------
# The Hansard-attribution rows are aggregates over every speech (first and last day, and a count), and 190 of 559
# touched the window: 29 lay wholly inside it and 132 ended inside it.


@pytest.fixture
def offices(scratch):
    c = Corpus(scratch)
    for mid in (7, 8, 21, 22, 23, 24):
        scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (mid, f"Member {mid}"))  # fmt: skip
    for did, day in (("o1", "2026-03-10"), ("o2", "2026-05-20"), ("o3", "2026-09-08")):
        c.debate(did, "Treasury Questions", date=day)
    c.speech("o1", "sp1", ["Before."], mnis_id=7, office="The Chancellor of the Exchequer")
    c.speech("o2", "sp2", ["Inside."], mnis_id=7, office="The Chancellor of the Exchequer")
    c.speech("o2", "sp3", ["Inside only."], mnis_id=8, office="The Secretary of State for Sealed Affairs")
    c.speech("o3", "sp4", ["After."], mnis_id=7, office="The Chancellor of the Exchequer")
    for mid, office, start, end in ((21, "Minister for A", "2026-05-01", "2026-06-01"),  # wholly inside
                                    (22, "Minister for B", "2026-05-01", None),  # starts inside
                                    (23, "Minister for C", "2025-01-01", "2026-05-01"),  # ends inside
                                    (24, "Minister for D", "2025-01-01", "2025-06-01")):  # outside
        scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                        "VALUES (%s, %s, lower(%s), %s, %s, 'mnis_biography', 0)", (mid, office, office, start, end))
    scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) "
                    "VALUES (7, 'The Chancellor of the Exchequer', 'chancellor of the exchequer', '2026-03-10', "
                    "'2026-09-08', 'hansard_attribution', 3), (8, 'The Secretary of State for Sealed Affairs', "
                    "'secretary of state for sealed affairs', '2026-05-20', '2026-05-20', 'hansard_attribution', 1)")
    yield scratch
    scratch.execute("TRUNCATE member_office")


def test_hansard_attributed_offices_are_recomputed_from_the_masked_speeches(offices, scratch_dsn):
    with served_conn(scratch_dsn) as conn:
        rows = conn.execute("SELECT mnis_id, office_key, start_date::text, end_date::text, evidence FROM member_office "
                            "WHERE source = 'hansard_attribution' ORDER BY mnis_id").fetchall()  # fmt: skip
    assert rows == [(7, "chancellor of the exchequer", "2026-03-10", "2026-09-08", 2)], \
        "the window's speech is not counted, and an office seen only in the window is not there"


def test_a_biography_post_touching_the_window_is_clipped_to_its_edge_and_says_so(offices, scratch_dsn):
    with served_conn(scratch_dsn) as conn:
        rows = {r[0]: r[1:] for r in conn.execute(
            "SELECT mnis_id, start_date::text, end_date::text, start_withheld, end_withheld FROM member_office "
            "WHERE source = 'mnis_biography'")}  # fmt: skip
    assert 21 not in rows, "a post wholly inside the window is not served"
    assert rows[22] == ("2026-08-01", None, True, False), "begun inside: from the window's end, the start withheld"
    assert rows[23] == ("2025-01-01", "2026-03-31", False, True), "ended inside: to the window's start, the end withheld"
    assert rows[24] == ("2025-01-01", "2025-06-01", False, False)


def test_an_undated_office_question_says_a_withheld_date_is_not_yet_available(offices, scratch_dsn):
    """Listing the holders of an office lists their periods. A side the mask withheld is said to be unavailable, never
    shown as the clipped date, and the same query still runs on a connection that reads the whole record.
    """
    from answer import facts as F
    from ingest.runtime import run as run_async

    async def holders(serving):
        async with await psycopg.AsyncConnection.connect(scratch_dsn, autocommit=True) as conn:
            if serving:
                await conn.execute(sealed.serving_sql())
            return await F.office_held_at_date(conn, "minister for b", None)

    served = run_async(holders(True)).clarify
    assert served[0]["from"] is None and "not yet available" in served[0]["note"]
    whole = run_async(holders(False)).clarify
    assert whole[0]["from"] == "2026-05-01" and "note" not in whole[0]


# --- the exam lift reaches an open pool (QA pass 3, finding 12) -----------------------------------------------------
# The mask was set when a connection was made, so `lifted_for_exam` did not unmask a served pool already open, and a
# pool opened inside the lift stayed unmasked for its whole life, outliving the lift.


def test_a_served_pool_follows_the_lift_both_ways(record, scratch_dsn):
    from ingest.runtime import run as run_async

    async def path(pool):
        async with pool.connection() as conn:
            return (await (await conn.execute("SHOW search_path")).fetchone())[0]

    async def go():
        async with make_pool(scratch_dsn, served=True, min_size=1, max_size=1) as pool:
            before = await path(pool)
            with sealed.lifted_for_exam("test: the W5 exam run"):
                during = await path(pool)
            after = await path(pool)
        with sealed.lifted_for_exam("test: a pool opened inside the lift"):
            pool = make_pool(scratch_dsn, served=True, min_size=1, max_size=1)
            await pool.open()
            inside = await path(pool)
        try:
            outlived = await path(pool)
        finally:
            await pool.close()
        return before, during, after, inside, outlived

    before, during, after, inside, outlived = run_async(go())
    assert before.startswith("served") and after.startswith("served")
    assert "served" not in during and "served" not in inside
    assert outlived.startswith("served"), "a pool opened inside the lift is masked again once the lift ends"


# --- QA pass 4: bills (L8) and the reader role (L7), migration 028 ------------------------------------


@pytest.fixture
def bills(scratch):
    """A bill seen only inside the window, one seen on both sides, one only outside, and one never dated."""
    for bid, title in ((7001, "Sealed Only Bill"), (7002, "Both Sides Bill"), (7003, "Open Bill"), (7004, "Undated Bill")):
        scratch.execute("INSERT INTO bill (id, title, is_act, current_house, raw) VALUES (%s, %s, true, 'Lords', "
                        "'{\"stages\": [\"x\"]}'::jsonb)", (bid, title))  # fmt: skip
    scratch.execute("UPDATE bill SET last_update = '2026-05-20T10:00:00Z' WHERE id = 7002")  # inside the window
    for sid, bid, day in ((711, 7001, INSIDE), (721, 7002, OUTSIDE), (722, 7002, INSIDE), (731, 7003, OUTSIDE)):
        scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house) VALUES (%s, %s, '2nd reading', "
                        "'Commons')", (sid, bid))  # fmt: skip
        scratch.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date) VALUES (%s, %s, %s, %s)",
                        (sid, sid, bid, day))  # fmt: skip
    return scratch


def test_a_bill_seen_only_inside_the_window_is_not_served_and_one_that_touches_it_says_nothing_of_its_status(bills,
                                                                                                          scratch_dsn):
    """L8: the `bill` table had no served view, so 38 bills whose every dated stage is masked were still found, and a
    reader could learn they exist; `is_act` and `raw` (the Bills API's whole payload, every stage's date in it) were
    unmasked.
    """
    with served_conn(scratch_dsn) as conn:
        rows = {r[0]: r[1:] for r in conn.execute("SELECT id, is_act, current_house, raw FROM bill").fetchall()}
    assert 7001 not in rows, "a bill seen only inside the window is not served"
    assert rows[7002] == (None, None, {}), "status and payload withheld for a bill the window touches"
    assert rows[7003][0] is True and rows[7003][2] == {"stages": ["x"]}
    assert 7004 in rows, "a bill no stage dates carries nothing from any window"
    with served_conn(scratch_dsn) as conn:
        as_of = conn.execute("SELECT last_update::date FROM bill WHERE id = 7002").fetchone()[0]
    assert str(as_of) == "2026-03-31", ("an update inside the window reads as the day before it: no date leaks, and "
                                        "every served sitting after the window stays scheduled, as with the true date")


def test_the_reader_role_reads_no_unmasked_content_but_the_guard_still_reads_its_dates(record, scratch_dsn):
    """L7: `otr_reader` held SELECT on every public table, so it blocked writes but masked nothing. Now it reads only
    the served views, plus the ids and dates the exit guard checks.
    """
    with psycopg.connect(scratch_dsn, autocommit=True) as conn:
        conn.execute(sealed.serving_sql())
        conn.execute("SET ROLE otr_reader")
        for sql in ("SELECT title FROM public.division", "SELECT body FROM public.chunk", "SELECT raw FROM public.bill",
                    "SELECT heading FROM public.debate", "SELECT office FROM public.member_office"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(sql)
        assert conn.execute("SELECT count(*) FROM division").fetchone()[0] == 1, "the served view still reads"
        dated = conn.execute("SELECT s.id, d.date FROM public.speech s JOIN public.debate d ON d.id = s.debate_id "
                             "ORDER BY 1").fetchall()  # fmt: skip
        assert [r[0] for r in dated] == ["s_open", "s_shut"], "the guard sees every speech's date, the window's too"
        assert conn.execute("SELECT count(*) FROM public.chunk WHERE debate_date IS NOT NULL").fetchone()[0] >= 2
