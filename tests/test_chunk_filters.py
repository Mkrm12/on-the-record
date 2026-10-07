"""A chunk is born with its filter columns, and a changed sentence loses its whole embedding, recipe included."""

from __future__ import annotations

from datetime import date

import pytest

from app import safety
from ingest import chunk as C
from ingest.runtime import run

DAY = date(2025, 11, 19)


@pytest.fixture
def debate(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WF', 'test', '2025-11-01', '2025-11-30', 'dev_test')")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (42, 'Ms Test'), (43, 'Mr Other')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                    "VALUES ('dF', %s, 'Lords', 'A debate', 'Grand Committee', 'WF', 'test')", (DAY,))
    scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id, mnis_id) "
                    "VALUES ('sF', 'dF', 0, 'Ms Test', 'The first sentence stands. The second sentence stands too.', "
                    "'WF', 42)")
    return scratch


def chunk(scratch_dsn):
    from app.db import make_pool

    async def go():
        pool = make_pool(scratch_dsn, min_size=1, max_size=1)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                await conn.set_autocommit(True)
                await C.chunk_debate(conn, "dF", C.DEFAULT_WINDOW, C.Stats())
        finally:
            await pool.close()

    run(go())


def test_a_new_chunk_carries_its_filter_columns_without_any_sync(debate, scratch_dsn):
    chunk(scratch_dsn)
    rows = debate.execute("SELECT window_id, debate_date, house, section, mnis_id FROM chunk").fetchall()
    assert rows and set(rows) == {("WF", DAY, "Lords", "Grand Committee", 42)}


def test_the_filter_columns_agree_with_the_integrity_gate(debate, scratch_dsn):
    """The same comparison D02-D06 make, NULL-safe: nothing for the sync step to fix."""
    chunk(scratch_dsn)
    drift = debate.execute(
        "SELECT count(*) FROM chunk k JOIN debate d ON d.id = k.debate_id JOIN speech s ON s.id = k.speech_id "
        "WHERE (k.window_id, k.debate_date, k.house, k.section, k.mnis_id) "
        "IS DISTINCT FROM (d.window_id, d.date, d.house, d.section, s.mnis_id)").fetchone()[0]
    assert drift == 0


def test_a_remapped_speaker_follows_through_on_the_next_chunking(debate, scratch_dsn):
    chunk(scratch_dsn)
    debate.execute("UPDATE speech SET mnis_id = 43 WHERE id = 'sF'")
    chunk(scratch_dsn)
    assert {r[0] for r in debate.execute("SELECT mnis_id FROM chunk").fetchall()} == {43}


def _embed_everything(conn):
    from corpus import vec

    for (chunk_id,) in conn.execute("SELECT id FROM chunk").fetchall():
        conn.execute("UPDATE chunk SET embedding = %s, embedding_model = 'm', embedded_at = now(), "
                     "embedding_recipe = 'recipe-1' WHERE id = %s", (vec((0, 1.0)), chunk_id))


def test_a_changed_sentence_loses_its_whole_embedding_including_the_recipe(debate, scratch_dsn):
    chunk(scratch_dsn)
    _embed_everything(debate)
    debate.execute("UPDATE speech SET text = 'The first sentence stands. The second sentence has changed.' "
                   "WHERE id = 'sF'")
    chunk(scratch_dsn)
    rows = debate.execute("SELECT body, embedding IS NULL, embedding_model IS NULL, embedded_at IS NULL, "
                          "embedding_recipe IS NULL FROM chunk ORDER BY sent_idx").fetchall()
    unchanged, changed = rows[0], rows[1]
    assert unchanged[1:] == (False, False, False, False), "an unchanged sentence must keep its embedding"
    assert changed[1:] == (True, True, True, True), "a changed sentence kept part of an embedding it no longer has"


def test_rechunking_an_unchanged_debate_writes_nothing(debate, scratch_dsn):
    chunk(scratch_dsn)
    before = debate.execute("SELECT xmin::text FROM chunk ORDER BY id").fetchall()
    chunk(scratch_dsn)
    assert debate.execute("SELECT xmin::text FROM chunk ORDER BY id").fetchall() == before
