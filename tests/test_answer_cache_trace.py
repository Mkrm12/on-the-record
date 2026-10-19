"""The answer cache's read/write path and the trace table, against a real Postgres."""

from __future__ import annotations

import pytest

from answer import cache as K
from answer import trace as TR
from answer.contract import Answer, Citation, Claim, Provenance, Verification
from ingest.runtime import run as run_async

PROV = dict(route="retrieval", corpus_version="v2-w0w5", retrieval_config_hash="e57c6cdfd8793d76",
            recipe="bge-small-header-v1", prompt_version="p3-2026-09-29+brief", model_id="deepseek-flash",
            code_version="c0ffee123456")  # fmt: skip


def passing() -> Verification:
    return Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True,
                        window_ok=True, checked_by="test")  # fmt: skip


def answer(question="what did she say?", *, verified=True, abstain=None, **prov):
    claims = ()
    if abstain is None:
        claims = (Claim(id="c1", kind="said", text="The House rose at six o'clock.",
                        speaker_name="A Member",
                        citations=(Citation(speech_id="s1", chunk_ids=(10,),
                                            quote="The House rose at six o'clock."),),
                        verification=passing() if verified else Verification()),)  # fmt: skip
    return Answer(
        question=question, provenance=Provenance(**{**PROV, **prov}), claims=claims,
        retrieved_chunk_ids=frozenset({10, 11}),
        abstention=abstain, abstention_detail="nothing matched" if abstain else "",
    )  # fmt: skip


@pytest.fixture
def db(scratch, scratch_dsn):
    scratch.execute("TRUNCATE answer_cache, answer_trace")
    yield scratch_dsn
    scratch.execute("TRUNCATE answer_cache, answer_trace")


def call(fn, dsn, *args, **kwargs):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            result = await fn(conn, *args, **kwargs)
            await conn.commit()
            return result
        finally:
            await conn.close()

    return run_async(go())


def store(dsn, ans, **kw):
    kw.setdefault("verifier", "minicheck-flan-t5-large")
    return call(K.store, dsn, ans, **kw)


def lookup(dsn, ans):
    return call(K.lookup, dsn, K.Key.of(ans))


# --- the cache key --------------------------------------------------------------------------------------


def test_an_answer_is_found_under_its_own_provenance(db):
    ans = answer()
    store(db, ans)
    hit = lookup(db, ans)
    assert hit is not None and hit["outcome"] == "answer"


@pytest.mark.parametrize("field", ["corpus_version", "retrieval_config_hash", "recipe", "prompt_version",
                                  "model_id", "code_version"])  # fmt: skip
def test_any_change_of_provenance_is_a_miss(db, field):
    """The whole point of the key. An answer produced under a different corpus, retrieval config, recipe, prompt or
    model came from a system that no longer exists.
    """
    store(db, answer())
    assert lookup(db, answer(**{field: "something-else"})) is None


def test_a_different_question_is_a_miss(db):
    store(db, answer("what did she say?"))
    assert lookup(db, answer("what did he say?")) is None


def test_the_same_question_spelled_differently_is_a_hit(db):
    """Case, whitespace and a trailing question mark are spelling, not meaning."""
    store(db, answer("What did she say?"))
    assert lookup(db, answer("  what did she SAY  ")) is not None


def test_a_hit_bumps_the_hit_count(db):
    """wants a hit count per entry, and it carries no personal data — a count and a timestamp, never the asker."""
    ans = answer()
    store(db, ans)
    lookup(db, ans)
    lookup(db, ans)
    hit = lookup(db, ans)
    assert hit["hit_count"] == 3


def test_re_answering_under_identical_provenance_replaces_the_row(db):
    """Identical provenance means the same system answering the same question, so the newer run is the better record of
    it — and the hit count is deliberately preserved.
    """
    ans = answer()
    store(db, ans)
    lookup(db, ans)
    store(db, ans)
    hit = lookup(db, ans)
    assert hit["hit_count"] == 2, "the count survived the rewrite"


# --- what must never be cached --------------------------------------------------------------------------


def test_an_unverified_claim_is_refused(db):
    """The cache is the one place an unverified claim can outlive the process that failed to check it."""
    with pytest.raises(K.CacheError, match="did not pass verification"):
        store(db, answer(verified=False))


def test_an_answer_with_no_verifier_is_refused(db):
    with pytest.raises(K.CacheError, match="no verifier"):
        store(db, answer(), verifier="")


def test_an_abstention_is_cached(db):
    """"Nothing in the record" is a real answer and expensive to recompute, and caching it is what stops the system re-
    asking a provider the same unanswerable question.
    """
    from answer.contract import Abstention

    ans = answer(abstain=Abstention.NOT_IN_RECORD)
    store(db, ans)
    hit = lookup(db, ans)
    assert hit["outcome"] == "abstain" and hit["abstention_reason"] == "not_in_record"


def test_the_stored_answer_round_trips(db):
    """A cached answer must rebuild into the same object, or a hit serves something subtly different from a miss."""
    ans = answer()
    store(db, ans)
    hit = lookup(db, ans)
    payload = hit["answer_json"]
    rebuilt = (Answer.model_validate_json(payload) if isinstance(payload, (str, bytes, bytearray))
               else Answer.model_validate(payload))  # fmt: skip
    assert rebuilt.question == ans.question
    assert [c.text for c in rebuilt.claims] == [c.text for c in ans.claims]
    assert rebuilt.provenance.retrieval_config_hash == "e57c6cdfd8793d76"


# --- invalidation ---------------------------------------------------------------------------------------


def test_invalidation_keeps_what_matches_and_deletes_the_rest(db):
    """Expressed as "keep what matches" rather than "delete what changed", because the second form needs to know the
    old value and after a model swap nobody reliably remembers.
    """
    store(db, answer("a"))
    store(db, answer("b", model_id="an-older-model"))
    removed = call(K.invalidate, db, model_id="deepseek-flash")
    assert removed == 1
    assert lookup(db, answer("a")) is not None


def test_invalidating_with_no_current_value_is_refused(db):
    """It would delete the whole cache."""
    with pytest.raises(K.CacheError, match="at least one current value"):
        call(K.invalidate, db)


def test_stats_counts_what_the_cache_holds(db):
    from answer.contract import Abstention

    store(db, answer("a"))
    store(db, answer("b", model_id="m2"))
    store(db, answer("c", abstain=Abstention.NOT_IN_RECORD))
    found = call(K.stats, db)
    assert found["entries"] == 3 and found["abstentions"] == 1 and found["models"] == 2


# --- the trace ------------------------------------------------------------------------------------------


def trace(**kw):
    base = dict(query_hash="h1", route="retrieval", corpus_version="v2-w0w5",
                retrieval_config_hash="e57c6cdfd8793d76", outcome="answer")  # fmt: skip
    base.update(kw)
    return TR.Trace(**base)


def test_a_trace_row_is_written_for_an_answer(db):
    assert call(TR.record, db, trace()) is not None


@pytest.mark.parametrize("outcome", ["answer", "clarify", "abstain", "error"])
def test_every_outcome_the_table_allows_is_writable(db, outcome):
    """'error' exists in the trace and not in the cache: a request that fell over is a fact worth keeping and is
    exactly what must never be cached as though it were an answer.
    """
    reason = "not_in_record" if outcome == "abstain" else None
    assert call(TR.record, db, trace(outcome=outcome, abstention_reason=reason)) is not None


def test_an_invented_outcome_is_refused_by_the_schema(db):
    """`record` swallows the error and returns None rather than raising, because telemetry must not break the thing it
    observes — but nothing is written.
    """
    assert call(TR.record, db, trace(outcome="probably_fine")) is None


def test_an_invented_abstention_reason_is_refused(db):
    assert call(TR.record, db, trace(outcome="abstain", abstention_reason="could_not_be_bothered")) is None


def test_a_trace_write_never_raises(db):
    """The property that matters. A request that succeeded must not be reported as failed because the telemetry write
    failed, and a request that failed must not have its real error replaced by a database error.
    """
    assert call(TR.record, db, trace(route=None)) is None  # NOT NULL violation, swallowed


def test_the_question_is_stored_only_when_the_caller_asks(db):
    """: the question text is kept only when a config flag allows it."""
    call(TR.record, db, trace(query_hash="kept", question="what did she say?"))
    call(TR.record, db, trace(query_hash="dropped"))
    rows = call(_questions, db)
    assert rows == {"kept": "what did she say?", "dropped": None}


async def _questions(conn):
    cur = await conn.execute("SELECT query_hash, question FROM answer_trace")
    return {row[0]: row[1] for row in await cur.fetchall()}


def test_the_summary_separates_the_outcomes_that_never_reach_the_cache(db):
    """The reason the trace exists at all: abstentions, clarifications and errors are absent from `answer_cache`, and
    they are the population that distinguishes honest caution from a quiet malfunction.
    """
    call(TR.record, db, trace(run_id="r1", outcome="answer", cost_usd=0.001, llm_calls=1, total_ms=900))
    call(TR.record, db, trace(run_id="r1", outcome="abstain", abstention_reason="not_in_record", total_ms=50))
    call(TR.record, db, trace(run_id="r1", outcome="clarify", total_ms=40))
    call(TR.record, db, trace(run_id="r1", outcome="error", failed_reason="LlmError: HTTP 429", total_ms=10))
    call(TR.record, db, trace(run_id="other", outcome="answer"))
    found = call(TR.summary, db, "r1")["overall"]
    assert found["requests"] == 4
    assert (found["answers"], found["abstentions"], found["clarifications"], found["errors"]) == (1, 1, 1, 1)
    assert found["failed"] == 1
    assert float(found["cost_usd"]) == pytest.approx(0.001)


def test_the_summary_reports_latency_percentiles(db):
    for ms in (100, 200, 5000):
        call(TR.record, db, trace(run_id="lat", total_ms=ms))
    found = call(TR.summary, db, "lat")["overall"]
    assert found["median_ms"] == 200 and found["p95_ms"] == 5000


def test_rewrites_and_repairs_are_recorded(db):
    """A guard is only a guard if its use is visible afterwards."""
    call(TR.record, db, trace(run_id="g", rewrites=1, repairs=1))
    found = call(TR.summary, db, "g")["overall"]
    assert found["rewrites"] == 1 and found["repairs"] == 1


def test_an_answer_from_other_code_is_a_miss_even_with_everything_else_equal(db):
    """The key covered corpus, retrieval config, recipe, prompt and model but not the code, so a re-run served the
    previous build's abstentions in 0.2 s and hid the fix just made.
    """
    store(db, answer(code_version="build-A"))
    assert lookup(db, answer(code_version="build-A")) is not None
    assert lookup(db, answer(code_version="build-B")) is None


def test_the_code_version_is_a_fingerprint_of_what_decides_an_answer():
    from answer import keys

    first = keys.code_version()
    assert len(first) == 12 and all(c in "0123456789abcdef" for c in first)
    assert keys.code_version() == first, "stable within a process"


def test_the_fingerprint_covers_the_sql_that_decides_an_answer():
    """A migration that redefines either changes answers exactly as a code edit does, so the migrations are part of the
    fingerprint — or the cache serves what the current system would not say.
    """
    from answer import keys

    names = {p.as_posix() for p in keys.fingerprinted_files()}
    assert any(n.endswith("db/migrations/021_printed_name.sql") for n in names)
    assert any(n.endswith("src/answer/names.py") for n in names) and any(n.endswith("src/retrieval/hybrid.py") for n in names)
    assert any(n.endswith("config/retrieval_frozen.json") for n in names)


def test_every_trace_column_is_written_by_the_pipeline():
    """Every column the trace table has must be passed by the pipeline's one `Trace(...)` construction; a new column
    without a writer fails here.
    """
    import ast
    from pathlib import Path

    from answer import trace as TR

    source = Path(__file__).resolve().parents[1] / "src" / "answer" / "pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", getattr(n.func, "id", None)) == "Trace"]
    assert len(calls) == 1, "one construction of the trace row, or this test must learn about the other"
    passed = {kw.arg for kw in calls[0].keywords}
    assert set(TR.COLUMNS) - passed == set()


def test_the_cache_holds_no_column_nothing_writes(scratch):
    """Migration 018 dropped seven that migration 007 planned and nothing ever wrote."""
    import answer.cache as K

    have = {r[0] for r in scratch.execute("SELECT column_name FROM information_schema.columns "
                                     "WHERE table_name = 'answer_cache'").fetchall()}
    assert not have & {"attribution_failures", "citation_failures", "cited_debate_ids", "division_ids",
                       "entailment_min_score", "number_failures", "quote_failures"}
    assert set(K.KEY_COLUMNS) <= have
