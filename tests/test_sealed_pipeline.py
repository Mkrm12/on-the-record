"""Nothing a reader is shown is dated inside the sealed window."""

from __future__ import annotations

import asyncio
from datetime import date

import psycopg
import pytest

from answer import pipeline as PL
from answer import sealing as SG
from answer.contract import (Abstention, Answer, Citation, Claim, Provenance, Roster, RosterEntry, StructuredFact,
                             Verification)
from app import sealed
from ingest.runtime import run

PROV = Provenance(route="retrieval", corpus_version="v2-w0w5", retrieval_config_hash="e57c6cdfd8793d76",
                  recipe="BAAI/bge-small-en-v1.5|full|v2", prompt_version="p1", model_id="deepseek-flash")


def claim(cid="c1", day=date(2026, 9, 8), speech="s_open", chunk=1):
    return Claim(id=cid, kind="said", text="Housing policy reform is needed now.", speaker_name="Ann Member", date=day,
                 citations=(Citation(speech_id=speech, chunk_ids=(chunk,), quote="Housing policy reform is needed now."),),
                 verification=Verification(quote_ok=True, citation_ok=True, window_ok=True))  # verified: showable


def answer(**kw):
    base = dict(question="What was said about housing?", provenance=PROV, claims=(claim(),),
                rendered="Ann Member said housing policy reform is needed now. [2026-09-08]")
    return Answer(**{**base, **kw})


OPEN = {"speech": {"s_open": date(2026, 9, 8)}, "debate": {"d_open": date(2026, 9, 8)}, "chunk": {1: date(2026, 9, 8)}}


# --- layer 2: the question -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("filters,inside", [
    ({"on_date": "2026-05-20"}, True),
    ({"from_date": "2026-05-01", "to_date": "2026-05-31", "year": 2026}, True),
    ({"from_date": "2026-01-01", "to_date": "2026-12-31", "year": 2026}, False),  # answered from the rest, masked
    ({"on_date": "2026-09-08"}, False),
    ({}, False),
])
def test_a_question_dated_inside_the_window_is_recognised(filters, inside):
    assert SG.question_is_sealed(filters) is inside
    with sealed.lifted_for_exam("the W5 exam (test)"):
        assert SG.question_is_sealed(filters) is False


def test_the_scope_answer_is_an_honest_decline_with_no_sealed_day_in_it():
    a = SG.scope_answer("What did MPs say on 20 May 2026?", PROV)
    assert a.abstention is Abstention.OUTSIDE_WINDOW
    assert a.rendered == sealed.SCOPE_STATEMENT and a.claims == () and a.structured_facts == ()
    assert not sealed.sealed_days_in(" ".join((a.rendered, a.abstention_detail, a.coverage_note)))


# --- layer 3: the exit ---------------------------------------------------------------------------------------------


def test_a_clean_answer_has_no_reasons():
    assert SG.reasons(answer(), OPEN) == []


@pytest.mark.parametrize("change,why", [
    (dict(claims=(claim(day=date(2026, 5, 20)),)), "claim c1"),
    (dict(claims=(claim(speech="s_sealed"),)), "s_sealed"),
    (dict(claims=(claim(chunk=2),)), "chunk 2"),
    (dict(retrieved_chunk_ids=frozenset({1, 2})), "chunk 2"),  # read, not cited: the model still saw it
    (dict(roster=Roster(debate_ids=("d_sealed",), confidence=1.0,
                        entries=(RosterEntry(speaker_name="Ann Member", contributions=1),))), "d_sealed"),
    (dict(structured_facts=(StructuredFact(template="division_result", params={"division_id": 7},
                                           value={"date": "2026-05-20", "result": "agreed"}),)), "2026-05-20"),
    (dict(structured_facts=(StructuredFact(template="bill_stage_date", params={"bill_id": 1},
                                           value={"first_sitting": date(2026, 4, 29)}),)), "2026-04-29"),
    (dict(rendered="The Lords voted on 20 May 2026."), "2026-05-20"),
    (dict(facets={"debates": [{"id": "d_open", "date": "2026-06-02"}]}), "2026-06-02"),
])
def test_anything_dated_inside_the_window_is_a_reason_to_withhold(change, why):
    dates = {"speech": {**OPEN["speech"], "s_sealed": date(2026, 5, 20)},
             "debate": {**OPEN["debate"], "d_sealed": date(2026, 5, 20)},
             "chunk": {**OPEN["chunk"], 2: date(2026, 5, 20)}}
    found = SG.reasons(answer(**change), dates)
    assert found and any(why in r for r in found), found


def test_nothing_is_withheld_while_the_exam_has_lifted_the_mask():
    with sealed.lifted_for_exam("the W5 exam (test)"):
        assert SG.reasons(answer(rendered="The Lords voted on 20 May 2026."), OPEN) == []


def test_the_guard_reads_the_records_own_dates_and_replaces_a_breach(scratch, scratch_dsn):
    from corpus import Corpus, vec

    c = Corpus(scratch)
    c.debate("d_open", "Housing", date="2026-09-08")
    c.debate("d_sealed", "Housing", date="2026-05-20")
    [open_chunk] = c.speech("d_open", "s_open", ["Housing policy reform is needed now."], vectors=[vec((0, 1))])
    [sealed_chunk] = c.speech("d_sealed", "s_sealed", ["Housing policy reform was urged again."], vectors=[vec((0, 1))])

    async def go(a):
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            return await SG.guard(conn, a)

    kept, why = run(go(answer(claims=(claim(chunk=open_chunk),), retrieved_chunk_ids=frozenset({open_chunk}))))
    assert why == [] and kept.claims
    held, why = run(go(answer(claims=(claim(speech="s_sealed", chunk=sealed_chunk, day=date(2026, 9, 8)),))))
    assert why and held.abstention is Abstention.OUTSIDE_WINDOW and held.rendered == sealed.SCOPE_STATEMENT


def test_a_guard_that_cannot_read_the_records_dates_withholds_rather_than_guesses():
    class Broken:
        async def execute(self, *a, **k):
            raise psycopg.OperationalError("the connection is gone")

    held, why = asyncio.run(SG.guard(Broken(), answer()))
    assert held.abstention is Abstention.OUTSIDE_WINDOW and any("could not read" in r for r in why)


# --- the pipeline: every path passes the guard ---------------------------------------------------------------------


class _Searcher:
    """Records whether anything was searched."""

    def __init__(self):
        self.calls = 0

    async def search(self, *a, **k):
        self.calls += 1
        raise AssertionError("a question dated inside the sealed window must never be searched")


def test_ask_answers_a_sealed_window_question_with_the_scope_statement_and_searches_nothing(scratch, scratch_dsn):
    searcher = _Searcher()
    runtime = PL.Runtime(searcher=searcher, client=None, frozen=PL.Frozen.load(), model_id="test-model")

    async def go():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            return await PL.ask(runtime, conn, "What did MPs say about housing on 20 May 2026?", use_cache=False,
                                store=False)

    asked = run(go())
    assert searcher.calls == 0
    assert asked.answer.abstention is Abstention.OUTSIDE_WINDOW and asked.answer.rendered == sealed.SCOPE_STATEMENT


def test_every_exit_of_ask_passes_the_guard_and_nothing_is_cached_before_it():
    import inspect

    source = inspect.getsource(PL.ask)
    finish = source[source.index("async def finish("):source.index("# 2. the cache")]
    assert "SG.guard(" in finish, "finish() must guard every answer it is handed"
    store = source[source.index("# 7. store"):]
    assert store.index("SG.guard(") < store.index("K.store("), "an answer is guarded before it can be cached"
    assert source.index("SG.question_is_sealed(") < source.index("K.lookup("), "a sealed question never reads the cache"


def test_masked_and_unmasked_answers_never_share_a_cache_key_or_a_provenance_label():
    runtime = PL.Runtime(searcher=None, client=None, frozen=PL.Frozen.load(), model_id="test-model")
    assert runtime.provenance("retrieval").corpus_version == sealed.served_label(PL.CORPUS_VERSION)
    with sealed.lifted_for_exam("the W5 exam (test)"):
        assert runtime.provenance("retrieval").corpus_version == PL.CORPUS_VERSION
