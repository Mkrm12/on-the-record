"""The evaluation harness (scoring, aggregation, paired comparisons), checked with hand-worked numbers and a fake
searcher. The harness itself is what the Phase 2 gate rests on, so it is tested like the code it judges.
"""

import asyncio
import math
from types import SimpleNamespace

import pytest

from retrieval.evaluation import (
    QueryOutcome,
    aggregate,
    auc,
    outcome_from_result,
    paired_differences,
    percentile,
    run_variant,
    score_outcome,
    speech_ranking,
    standard_variants,
)
from retrieval.hybrid import SearchConfig


def hit(speech, also=(), rerank=None):
    return SimpleNamespace(speech_id=speech, also_in_speeches=tuple(also), rerank_score=rerank)


def outcome(qid, category, relevant, ranked, abstain=False, top_rerank=None):
    return QueryOutcome(qid, category, frozenset(relevant), list(ranked), abstain, top_rerank, {"total": 10.0}, [])


def test_the_speech_ranking_dedupes_in_order_and_counts_identical_passages_in_other_speeches():
    hits = [hit("a"), hit("b", also=["c"]), hit("a"), hit("d")]
    assert speech_ranking(hits) == ["a", "b", "c", "d"]


def test_scoring_one_query_by_hand():
    o = outcome("g1", "exact_term", {"x", "y"}, ["p", "x", "q", "r", "s", "t", "y"])
    s = score_outcome(o, k=10)
    assert (s.found, s.n, s.found5) == (2, 2, 1) and s.recall == 1.0 and s.rr == pytest.approx(0.5)
    assert s.ndcg == pytest.approx((1 / 1.5849625 + 1 / 3.0) / (1 + 1 / 1.5849625), rel=1e-6)  # gains at ranks 2 and 7
    assert score_outcome(o, k=5).found == 1 and score_outcome(o, k=5).recall == 0.5


def test_an_abstain_query_scores_nothing():
    s = score_outcome(outcome("g9", "abstain", set(), ["a"], abstain=True))
    assert s.n == 0 and s.recall is None and s.rr == 0.0


def test_aggregation_sums_found_over_total_per_category_and_overall_and_skips_abstain():
    outcomes = [
        outcome("g1", "exact_term", {"a"}, ["a"]),
        outcome("g2", "exact_term", {"b"}, ["x", "y"]),
        outcome("g3", "multi_speech", {"c", "d"}, ["d", "c"]),
        outcome("g4", "abstain", set(), ["z"], abstain=True),
    ]
    table = aggregate(outcomes, k=10)
    assert (table.by_category["exact_term"].found, table.by_category["exact_term"].n, table.by_category["exact_term"].queries) == (1, 2, 2)
    assert (table.by_category["multi_speech"].found, table.by_category["multi_speech"].n) == (2, 2)
    assert "abstain" not in table.by_category
    assert (table.overall.found, table.overall.n, table.overall.queries) == (3, 4, 3)
    assert table.overall.mrr == pytest.approx((1 + 0 + 1) / 3) and table.overall.recall == pytest.approx(0.75)


def test_paired_differences_line_up_by_query_id_and_only_use_answerable_queries():
    base = [outcome("g1", "c", {"a"}, ["x", "a"]), outcome("g2", "c", {"b"}, ["b"]), outcome("g3", "abstain", set(), ["q"], abstain=True)]
    other = [outcome("g2", "c", {"b"}, ["x", "b"]), outcome("g1", "c", {"a"}, ["a"]), outcome("g3", "abstain", set(), ["q"], abstain=True)]
    assert paired_differences(base, other, "rr") == pytest.approx([1 - 0.5, 0.5 - 1])  # g1 improves, g2 worsens
    assert paired_differences(base, other, "recall") == [0.0, 0.0]


def test_paired_differences_refuse_mismatched_query_sets():
    with pytest.raises(ValueError):
        paired_differences([outcome("g1", "c", {"a"}, ["a"])], [outcome("g2", "c", {"a"}, ["a"])], "rr")


def test_auc_is_the_chance_a_positive_outranks_a_negative():
    assert auc([3.0, 4.0], [1.0, 2.0]) == 1.0 and auc([1.0], [2.0]) == 0.0
    assert auc([2.0], [2.0]) == 0.5 and auc([1.0, 3.0], [2.0]) == 0.5
    assert auc([], [1.0]) is None and auc([1.0], []) is None


def test_percentile_is_the_nearest_rank_and_nan_for_nothing():
    values = [float(i) for i in range(1, 11)]
    assert percentile(values, 50) == 5.0 and percentile(values, 95) == 10.0 and percentile(values, 100) == 10.0
    assert percentile(values, 10) == 1.0 and percentile(values, 0) == 1.0 and percentile([7.0], 50) == 7.0
    assert percentile(list(reversed(values)), 50) == 5.0  # order does not matter
    assert math.isnan(percentile([], 50))


def test_the_standard_ladder_isolates_one_change_per_step():
    v = {x.name: x.config for x in standard_variants()}
    assert list(v) == ["dense", "hybrid-naive", "hybrid", "hybrid+speaker", "hybrid+rerank", "full", "full-replace"]
    assert (v["dense"].dense, v["dense"].sparse, v["dense"].speaker, v["dense"].rerank) == (True, False, False, "off")
    assert v["hybrid-naive"].sparse_coverage == 0.0 and v["hybrid"].sparse_coverage == SearchConfig().sparse_coverage
    assert v["hybrid"].speaker is False and v["hybrid+speaker"].speaker is True and v["hybrid+speaker"].rerank == "off"
    assert v["hybrid+rerank"].rerank == "blend" and v["hybrid+rerank"].speaker is False
    assert v["full"] == SearchConfig() and v["full-replace"].rerank == "replace"


def test_an_outcome_is_built_from_a_search_result_with_its_best_rerank_score():
    result = SimpleNamespace(hits=[hit("a", rerank=-2.0), hit("b", rerank=3.5), hit("c")], warnings=["w"], diagnostics={"timings_ms": {"total": 42.0}})
    q = SimpleNamespace(id="g1", category="paraphrase", relevant=[SimpleNamespace(speech_id="b")], expected_abstain=False)
    o = outcome_from_result(q, result)
    assert o.ranked == ["a", "b", "c"] and o.relevant == {"b"} and o.top_rerank == 3.5 and o.timings == {"total": 42.0} and o.warnings == ["w"]


def test_run_variant_searches_every_query_with_the_variants_config_and_keeps_going_after_a_failure():
    calls = []

    class FakeSearcher:
        async def search(self, question, top_k, config):
            calls.append((question, top_k, config))
            if question == "boom":
                raise RuntimeError("database down")
            return SimpleNamespace(hits=[hit("s1")], warnings=[], diagnostics={"timings_ms": {}})

    queries = [
        SimpleNamespace(id="g1", question="one", category="c", relevant=[SimpleNamespace(speech_id="s1")], expected_abstain=False),
        SimpleNamespace(id="g2", question="boom", category="c", relevant=[SimpleNamespace(speech_id="s1")], expected_abstain=False),
    ]
    variant = standard_variants()[0]
    outcomes = asyncio.run(run_variant(FakeSearcher(), queries, variant, top_k=30))
    assert [o.qid for o in outcomes] == ["g1", "g2"] and outcomes[0].ranked == ["s1"]
    assert outcomes[1].ranked == [] and outcomes[1].warnings and "database down" in outcomes[1].warnings[0]
    assert calls[0][1] == 30 and calls[0][2] == variant.config
