"""Measured on dev: 467 distinct members used the first phrase and 56 the second, against 0 for the ordinary "who said"
question. The test is the record's own count of speakers, not a list of phrases someone thought procedural.
"""

from __future__ import annotations

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import resolve as RS
from answer import router as R
from app import safety
from ingest.runtime import run

PROV = dict(route="retrieval", resolved_filters={}, corpus_version="v2", retrieval_config_hash="x", recipe="r",
            prompt_version="p", model_id="m")


@pytest.mark.parametrize("question, phrase", [
    ("Who said it is a pleasure to serve under your chairmanship?", "it is a pleasure to serve under your chairmanship"),
    ("Find the bit where they say the question is that the amendment be made",
     "the question is that the amendment be made"),
    ("Who said that we must take back control of our borders?", "we must take back control of our borders"),
    ("Which MP said 'a national disgrace on a grand scale' about water bills?", "a national disgrace on a grand scale"),
])  # fmt: skip
def test_the_quoted_words_are_found(question, phrase):
    assert R.quoted_phrase(question) == phrase


@pytest.mark.parametrize("question", ["Who said it?", "What did the Chancellor say about fuel duty?",
                                      "Who spoke in the debate on housing?", "Who said no?"])
def test_a_question_quoting_no_words_has_no_phrase(question):
    assert R.quoted_phrase(question) is None


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WP', 'test', '2025-01-01', '2025-12-31', 'dev_test')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                    "VALUES ('dP', '2025-06-01', 'Commons', 'A Debate', 'Westminster Hall', 'WP', 'test')")
    rows = [(n, "It is a pleasure to serve under your chairmanship, Sir Roger.") for n in range(1, 13)]
    rows += [(50, "Our water bills are a national disgrace on a grand scale."),
             (51, "It is a national disgrace on a grand scale, frankly.")]
    for n, (mnis_id, body) in enumerate(rows):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id, mnis_id) "
                        "VALUES (%s, 'dP', %s, %s, %s, 'WP', %s)", (f"s{n}", n, f"Member {mnis_id}", body, mnis_id))
        scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body, mnis_id) "
                        "VALUES (%s, 'dP', %s, 0, %s, %s, %s)", (f"s{n}", n, body, body, mnis_id))
    return scratch_dsn


def with_conn(dsn, fn, *args):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await fn(conn, *args)
        finally:
            await conn.close()
    return run(go())


def test_the_record_counts_who_used_the_words(record):
    assert with_conn(record, RS.phrase_speakers, "it is a pleasure to serve under your chairmanship") == 12
    assert with_conn(record, RS.phrase_speakers, "a national disgrace on a grand scale") == 2


def test_words_many_members_use_are_asked_about(record):
    answer = with_conn(record, PL.common_phrase_clarification,
                       "Who said it is a pleasure to serve under your chairmanship?", K.Provenance(**PROV))
    assert answer is not None and answer.facets.get("clarify")
    assert "12 different members" in answer.coverage_note


def test_words_a_few_members_used_are_answered_normally(record):
    assert with_conn(record, PL.common_phrase_clarification,
                     "Which MP said 'a national disgrace on a grand scale' about water bills?",
                     K.Provenance(**PROV)) is None


def test_a_question_with_no_quoted_words_asks_the_database_nothing():
    class _Refuses:
        async def execute(self, *a, **k):
            raise AssertionError("no phrase, no query")
    assert run(PL.common_phrase_clarification(_Refuses(), "What did the Chancellor say?", K.Provenance(**PROV))) is None
