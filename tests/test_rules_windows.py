"""The rules engine, one committed window at a time — the same answer as the whole-corpus pass, in bounded memory.
"""

from __future__ import annotations

import psycopg
import pytest

from app import safety
from ingest import rules as R
from ingest.runtime import run

CHAIR, MEMBER = "Mr Speaker", "Jim Shannon (Strangford) (DUP)"
SPEECHES = {
    # window, debate, speech, seq, speaker, text
    "WA": [("dA", "sA0", 0, CHAIR, "Order. I call the Minister."),
           ("dA", "sA1", 1, MEMBER, "I rise to present a petition on behalf of 563 constituents from East Grinstead."),
           ("dA", "sA2", 2, CHAIR, "Question accordingly agreed to.")],
    "WB": [("dB", "sB0", 0, CHAIR, "Order. I call the Minister."),
           ("dB", "sB1", 1, MEMBER, "The Government have failed the people of Strangford on this matter."),
           ("dB", "sB2", 2, CHAIR, "Question accordingly negatived.")],
}


@pytest.fixture
def corpus(scratch, scratch_dsn, monkeypatch):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE procedure_event CASCADE")
    for wid, start, end in (("WA", "2025-11-01", "2025-11-30"), ("WB", "2025-12-01", "2025-12-31")):
        scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                        "VALUES (%s, 'test', %s, %s, 'dev_test')", (wid, start, end))
        debate_id = SPEECHES[wid][0][0]
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                        "VALUES (%s, %s, 'Commons', 'Health Bill', 'Commons Chamber', %s, 'test')", (debate_id, start, wid))
        for did, sid, seq, speaker, text in SPEECHES[wid]:
            scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id) "
                            "VALUES (%s, %s, %s, %s, %s, %s)", (sid, did, seq, speaker, text, wid))
            scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body, is_boilerplate, "
                            "window_id) VALUES (%s, %s, %s, 0, %s, %s, false, %s)", (sid, did, seq, text, text, wid))
        # Planted damage in BOTH windows: a stale rule-made event, a member's sentence wrongly flagged, and a hand-made
        # event (rule IS NULL) that no rule may ever touch.
        speaker_speech = SPEECHES[wid][1][1]
        scratch.execute("INSERT INTO procedure_event (debate_id, speech_id, seq, event_type, raw_phrase, rule, char_start) "
                        "VALUES (%s, %s, 1, 'petition_presented', 'stale', 'retired_rule', 999)", (debate_id, speaker_speech))
        scratch.execute("INSERT INTO procedure_event (debate_id, speech_id, seq, event_type, raw_phrase, rule, char_start) "
                        "VALUES (%s, %s, 1, 'manual_note', 'by hand', NULL, 5)", (debate_id, speaker_speech))
        scratch.execute("UPDATE chunk SET is_boilerplate = true WHERE speech_id = %s", (speaker_speech,))

    from app.db import make_pool

    monkeypatch.setattr(R, "pool", make_pool(scratch_dsn, min_size=1, max_size=2))
    return scratch


def rules(**kw) -> R.RulesResult:
    return run(R.run_rules(**kw))


def fresh_pool(monkeypatch, dsn):
    from app.db import make_pool

    monkeypatch.setattr(R, "pool", make_pool(dsn, min_size=1, max_size=2))


def state(conn, window_id):
    """What the rules own, for one window: stale events, hand-made events, wrong flags."""
    return {
        "stale": conn.execute("SELECT count(*) FROM procedure_event pe JOIN debate d ON d.id = pe.debate_id "
                              "WHERE d.window_id = %s AND pe.rule = 'retired_rule'", (window_id,)).fetchone()[0],
        "manual": conn.execute("SELECT count(*) FROM procedure_event pe JOIN debate d ON d.id = pe.debate_id "
                               "WHERE d.window_id = %s AND pe.rule IS NULL", (window_id,)).fetchone()[0],
        "wrong_flags": conn.execute("SELECT count(*) FROM chunk WHERE window_id = %s AND is_boilerplate "
                                    "AND speech_id LIKE '%%1'", (window_id,)).fetchone()[0],
    }


def test_one_window_is_fixed_and_the_other_is_not_touched(corpus, monkeypatch, scratch_dsn):
    rules(windows=["WA"])
    assert state(corpus, "WA") == {"stale": 0, "manual": 1, "wrong_flags": 0}
    assert state(corpus, "WB") == {"stale": 1, "manual": 1, "wrong_flags": 1}, "a WA run reached into WB"


def test_every_window_is_fixed_by_a_full_run_and_hand_made_events_survive(corpus):
    result = rules()
    assert result.windows == ["WA", "WB"]
    for wid in ("WA", "WB"):
        assert state(corpus, wid) == {"stale": 0, "manual": 1, "wrong_flags": 0}


def test_a_second_run_changes_nothing(corpus, monkeypatch, scratch_dsn):
    rules()
    fresh_pool(monkeypatch, scratch_dsn)
    again = rules()
    assert (again.speech_rows_changed, again.stale_events_removed, again.chunk_flags_changed,
            again.debate_types_changed) == (0, 0, 0, 0)


def test_what_is_stored_is_what_the_rules_say(corpus):
    """The oracle: every stored flag, event and type equals the pure functions applied to that row alone."""
    rules()
    for sid, text, speaker, flag, reason in corpus.execute(
            "SELECT id, text, speaker_name, is_boilerplate, boilerplate_reason FROM speech").fetchall():
        want = R.boilerplate_reason(text, speaker)
        assert (flag, reason) == (want is not None, want), sid
        stored = sorted(r[0] for r in corpus.execute(
            "SELECT event_type FROM procedure_event WHERE speech_id = %s AND rule IS NOT NULL", (sid,)).fetchall())
        assert stored == sorted(e.event_type for e in R.find_events(text, speaker)), sid
    for body, speaker, flag in corpus.execute(
            "SELECT c.body, s.speaker_name, c.is_boilerplate FROM chunk c JOIN speech s ON s.id = c.speech_id").fetchall():
        assert flag == bool(R.sentence_boilerplate_reason(body, speaker)), body
    for did, heading, section, house, kind in corpus.execute(
            "SELECT id, heading, section, house, debate_type FROM debate").fetchall():
        events = {r[0] for r in corpus.execute("SELECT event_type FROM procedure_event WHERE debate_id = %s "
                                               "AND rule IS NOT NULL", (did,)).fetchall()}
        assert kind == R.classify_debate(heading, section, house, events), did


def test_the_counters_add_up_across_windows(corpus):
    result = rules(dry_run=True)
    assert result.speeches == 6 and result.chunks_total == 6
    assert result.events["question_agreed"] == 1 and result.events["question_negatived"] == 1
    assert sum(result.debate_types.values()) == 2


def test_a_dry_run_writes_nothing(corpus):
    rules(dry_run=True)
    for wid in ("WA", "WB"):
        assert state(corpus, wid) == {"stale": 1, "manual": 1, "wrong_flags": 1}


def test_a_failure_in_a_later_window_keeps_the_earlier_one_committed(corpus, monkeypatch):
    """Read from ANOTHER connection (the fixture's): the only way to tell a committed row from an uncommitted one.
    """
    real = R.find_events

    def explode(text, name=None):
        if text and "Strangford" in text:
            raise RuntimeError("a rule broke on WB")
        return real(text, name)

    monkeypatch.setattr(R, "find_events", explode)
    with pytest.raises(RuntimeError, match="WB"):
        rules()
    assert state(corpus, "WA") == {"stale": 0, "manual": 1, "wrong_flags": 0}, "WA's work was lost with WB's failure"
    assert state(corpus, "WB") == {"stale": 1, "manual": 1, "wrong_flags": 1}


def test_an_unknown_window_is_refused_before_anything_runs(corpus):
    with pytest.raises(SystemExit, match="NOPE"):
        rules(windows=["NOPE"])
    assert state(corpus, "WA")["stale"] == 1


def test_the_rules_engine_refuses_the_rollback(monkeypatch):
    from app.db import make_pool

    monkeypatch.setattr(R, "pool", make_pool("postgresql://u:p@127.0.0.1:5432/hansard"))
    with pytest.raises(safety.WrongDatabase):
        rules()


def test_the_window_loop_never_reads_the_whole_corpus():
    """Structural: every read is parameterised by a window. A future edit that reintroduces a whole-table read fails
    here rather than on the ten-year corpus's first rules pass.
    """
    for sql in (R.SQL_SPEECHES, R.SQL_CHUNKS, R.SQL_DEBATES, R.SQL_DELETE_STALE_EVENTS, R.SQL_UNFLAG_CHUNKS):
        assert "window_id = %s" in sql, sql


def test_the_chunks_are_streamed_not_fetched():
    import inspect

    source = inspect.getsource(R.classify_window)
    assert "cursor(name=" in source and "fetchall" not in source.split("SQL_CHUNKS")[1].split("\n")[0]


@pytest.mark.parametrize("bad", ["hansard"])
def test_a_protected_name_is_refused_even_with_a_scratch_looking_host(bad, monkeypatch):
    from app.db import make_pool

    monkeypatch.setattr(R, "pool", make_pool(f"host=127.0.0.1 dbname={bad} user=x"))
    with pytest.raises(safety.WrongDatabase):
        rules(dry_run=True)


def test_committed_state_is_visible_to_another_connection(corpus, scratch_dsn):
    rules()
    with psycopg.connect(scratch_dsn) as other:
        assert other.execute("SELECT count(*) FROM procedure_event WHERE rule = 'retired_rule'").fetchone()[0] == 0
