"""No leg of any search returns a sentence dated inside the sealed window."""

from __future__ import annotations

from corpus import Corpus, vec
from test_retrieval_hybrid import FakeGpu, cfg, go

from app import sealed
from retrieval.hybrid import SearchConfig, SearchFilters

QUERY = vec((0, 1))


def two_windows(scratch):
    """A sentence in a sealed-window debate and one in an open debate, equally close to the query on every leg and
    worded differently (identical sentences would collapse into one hit and prove nothing).
    """
    c = Corpus(scratch)
    c.debate("open", "Housing", date="2026-09-08")
    c.debate("sealed", "Housing", date="2026-05-20")
    open_ids = c.speech("open", "s_open", ["Housing policy reform is needed now."], vectors=[QUERY])
    sealed_ids = c.speech("sealed", "s_sealed", ["Housing policy reform was urged again."], vectors=[QUERY])
    return open_ids[0], sealed_ids[0]


def test_the_fixture_is_not_vacuous_both_sentences_are_found_once_the_mask_is_lifted(scratch, scratch_dsn):
    open_id, sealed_id = two_windows(scratch)
    for config in (cfg(sparse=False, speaker=False), cfg(dense=False, speaker=False)):
        with sealed.lifted_for_exam("the W5 exam, retrieval level (test)"):
            result, _ = go(scratch_dsn, "housing policy reform", gpu=FakeGpu(), config=config, top_k=10)
        assert {open_id, sealed_id} <= _ids(result), config


def _ids(result):
    return {h.chunk_id for h in result.hits}


def test_a_search_never_returns_a_sealed_window_sentence(scratch, scratch_dsn):
    open_id, sealed_id = two_windows(scratch)
    result, _ = go(scratch_dsn, "housing policy reform", gpu=FakeGpu(), top_k=10)
    assert open_id in _ids(result) and sealed_id not in _ids(result)


def test_every_leg_is_masked_dense_keyword_and_speaker(scratch, scratch_dsn):
    open_id, sealed_id = two_windows(scratch)
    for config in (cfg(sparse=False, speaker=False), cfg(dense=False, speaker=False), cfg()):
        result, _ = go(scratch_dsn, "housing policy reform", gpu=FakeGpu(), config=config, top_k=10)
        assert sealed_id not in _ids(result), config


def test_a_date_filter_cannot_reach_inside_the_window(scratch, scratch_dsn):
    """A question dated inside the window gets nothing from retrieval: the pipeline answers it with the scope statement
    instead (tested in test_sealed_pipeline).
    """
    _, sealed_id = two_windows(scratch)
    from datetime import date

    scoped = cfg(filters=SearchFilters(date_from=date(2026, 5, 1), date_to=date(2026, 5, 31)))
    result, _ = go(scratch_dsn, "housing policy reform", gpu=FakeGpu(), config=scoped, top_k=10)
    assert sealed_id not in _ids(result) and result.hits == []


def test_only_the_exam_lift_sees_the_window(scratch, scratch_dsn):
    open_id, sealed_id = two_windows(scratch)
    with sealed.lifted_for_exam("the W5 exam, retrieval level (test)"):
        result, _ = go(scratch_dsn, "housing policy reform", gpu=FakeGpu(), top_k=10)
    assert {open_id, sealed_id} <= _ids(result)


def test_a_chunk_with_no_date_cannot_be_proven_outside_the_window_so_it_is_withheld(scratch, scratch_dsn):
    """Fails closed. Invariant D02 keeps every live chunk dated, so this costs nothing in production and stops a
    dateless row from leaking.
    """
    open_id, _ = two_windows(scratch)
    scratch.execute("UPDATE chunk SET debate_date = NULL WHERE id = %s", (open_id,))
    result, _ = go(scratch_dsn, "housing policy reform", gpu=FakeGpu(), top_k=10)
    assert open_id not in _ids(result)


def test_the_mask_is_part_of_the_predicate_not_a_filter_a_caller_chooses():
    empty = SearchFilters()
    assert not empty  # still "no filter": the plan for an unfiltered search is unchanged
    assert "c.debate_date < %(sealed_from)s OR c.debate_date > %(sealed_to)s" in empty.predicate("c")
    assert {"sealed_from", "sealed_to"} <= set(empty.params())
    assert SearchConfig().filters.predicate("c") == empty.predicate("c")
