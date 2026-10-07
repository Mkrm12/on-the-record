"""The answer cache's provenance and verification invariants."""

from __future__ import annotations

import pytest

CONFIG_HASH = "e57c6cdfd8793d76"  # what E1 froze retrieval at

BASE = dict(
    query_hash="h1",
    corpus_version="v2-w0w5",
    retrieval_config_hash=CONFIG_HASH,
    recipe="bge-small-header-v1",
    prompt_version="p1",
    model_id="deepseek-chat",
    verifier="minicheck-flan-t5-large",
    outcome="answer",
)


def insert(conn, **overrides):
    row = {**BASE, **overrides}
    row.setdefault("verified_at", "now()")
    cols = ", ".join(row)
    marks = ", ".join(["%s"] * len(row))
    conn.execute(f"INSERT INTO answer_cache ({cols}) VALUES ({marks})", list(row.values()))


@pytest.fixture
def cache(scratch):
    scratch.execute("TRUNCATE answer_cache")
    yield scratch
    scratch.execute("TRUNCATE answer_cache")


# --- what the table accepts ------------------------------------------------------------------------------


@pytest.mark.parametrize(("outcome", "reason"), [
    ("answer", None),
    ("clarify", None),
    ("abstain", "not_in_record"),
    ("abstain", "outside_window"),
    ("abstain", "ambiguous"),
    ("abstain", "false_premise"),
    ("abstain", "insufficient_evidence"),
])
def test_the_three_outcomes_and_every_abstention_reason_are_storable(cache, outcome, reason):
    """clarify is a first-class outcome, not a flavour of abstention: collapsing them would hide the case the system
    handles best.
    """
    insert(cache, query_hash=f"{outcome}-{reason}", outcome=outcome, abstention_reason=reason)
    assert cache.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 1


# --- what it refuses -------------------------------------------------------------------------------------


def test_an_answer_with_no_verifier_cannot_be_cached(cache):
    """A cache is the one place an unverified answer can outlive the process that failed to check it."""
    with pytest.raises(Exception, match="verifier"):
        insert(cache, verifier=None)


@pytest.mark.parametrize("field", ["corpus_version", "retrieval_config_hash", "recipe", "prompt_version",
                                  "model_id", "verified_at"])  # fmt: skip
def test_no_provenance_field_may_be_missing(cache, field):
    """Without these a cached answer from before the freeze is indistinguishable from one after it — which is the
    entire reason E1 hashed the retrieval config.
    """
    with pytest.raises(Exception, match=field):
        insert(cache, **{field: None})


def test_an_abstention_with_no_reason_is_refused(cache):
    """The bug this pins. `(outcome = 'abstain' AND reason IN (..."""
    with pytest.raises(Exception, match="abstention"):
        insert(cache, outcome="abstain", abstention_reason=None)


def test_a_row_with_no_outcome_is_refused(cache):
    """The same three-valued gap: `CHECK (outcome IN (...))` passes when outcome is NULL, so the column has to be NOT
    NULL in its own right.
    """
    with pytest.raises(Exception, match="outcome"):
        insert(cache, outcome=None)


def test_an_invented_abstention_reason_is_refused(cache):
    with pytest.raises(Exception, match="abstention"):
        insert(cache, outcome="abstain", abstention_reason="could_not_find_it")


def test_an_answer_may_not_carry_an_abstention_reason(cache):
    """Answering and abstaining at once is a hedge the user cannot act on."""
    with pytest.raises(Exception, match="abstention"):
        insert(cache, outcome="answer", abstention_reason="ambiguous")


def test_an_invented_outcome_is_refused(cache):
    with pytest.raises(Exception, match="outcome"):
        insert(cache, outcome="maybe")


def test_more_claims_verified_than_exist_is_refused(cache):
    with pytest.raises(Exception, match="verified"):
        insert(cache, claim_count=2, claims_verified=3)


# --- the cache key ---------------------------------------------------------------------------------------


def test_the_same_question_under_a_different_corpus_is_a_different_row(cache):
    """W5 added 351,989 chunks. The same question over 1,085,867 chunks and over 1,437,856 is not the same question,
    and serving the first as the second is the failure this key prevents.
    """
    insert(cache, corpus_version="v2-w0w4")
    insert(cache, corpus_version="v2-w0w5")
    assert cache.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 2


@pytest.mark.parametrize("field", ["retrieval_config_hash", "recipe", "prompt_version", "model_id"])
def test_any_provenance_change_is_a_cache_miss(cache, field):
    insert(cache)
    insert(cache, **{field: "something-else"})
    assert cache.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 2


def test_the_identical_answer_under_identical_provenance_is_one_row(cache):
    """The flip side: the key must still deduplicate, or the cache is an append-only log."""
    insert(cache)
    with pytest.raises(Exception, match="duplicate|unique"):
        insert(cache)


# --- the metadata is there to be queried -----------------------------------------------------------------


def test_cited_ids_are_searchable_by_containment(cache):
    """"Which cached answers cite this speech" is the first question asked when a chunk turns out to be wrong. Without
    the GIN index it is a sequential scan over every answer ever given.
    """
    insert(cache, query_hash="a", cited_chunk_ids=[10, 11], cited_speech_ids=["s1"])
    insert(cache, query_hash="b", cited_chunk_ids=[12], cited_speech_ids=["s2"])
    found = cache.execute("SELECT query_hash FROM answer_cache WHERE cited_chunk_ids @> ARRAY[11]::bigint[]")
    assert [r[0] for r in found] == ["a"]
    found = cache.execute("SELECT query_hash FROM answer_cache WHERE cited_speech_ids @> ARRAY['s2']")
    assert [r[0] for r in found] == ["b"]


def test_degradation_is_recorded_and_queryable(cache):
    """"How often did we fall back" must be answerable after the fact, not only from logs."""
    insert(cache, query_hash="ok")
    insert(cache, query_hash="fell-back", degraded=True, degraded_reason="gpu service unreachable")
    n = cache.execute("SELECT count(*) FROM answer_cache WHERE degraded").fetchone()[0]
    assert n == 1


def test_latency_per_stage_is_stored_so_a_p95_is_a_query(cache):
    for i, ms in enumerate([40, 55, 900]):
        insert(cache, query_hash=f"q{i}", route="lookup", retrieval_ms=ms, total_ms=ms + 10)
    p95 = cache.execute(
        "SELECT percentile_disc(0.95) WITHIN GROUP (ORDER BY total_ms) FROM answer_cache WHERE route='lookup'"
    ).fetchone()[0]
    assert p95 == 910


def test_the_frozen_hash_is_stored_verbatim(cache):
    insert(cache)
    assert cache.execute("SELECT retrieval_config_hash FROM answer_cache").fetchone()[0] == CONFIG_HASH
