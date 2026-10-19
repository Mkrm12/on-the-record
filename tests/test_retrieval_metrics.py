"""The evaluation metrics, checked against hand-worked values."""

import math

import pytest

from retrieval.metrics import bootstrap_paired, dedupe_speeches, found_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


def test_dedupe_keeps_the_first_occurrence_in_order():
    assert dedupe_speeches(["a", "b", "a", "c", "b"]) == ["a", "b", "c"] and dedupe_speeches([]) == []


def test_found_counts_relevant_speeches_inside_the_cut_only_once_each():
    assert found_at_k({"a", "b"}, ["x", "a", "y", "b"], 3) == 1
    assert found_at_k({"a", "b"}, ["x", "a", "y", "b"], 4) == 2
    assert found_at_k({"a"}, ["a", "a", "a"], 3) == 1  # a repeated speech is not two finds
    assert found_at_k({"a"}, [], 5) == 0 and found_at_k(set(), ["a"], 5) == 0


def test_recall_is_the_share_of_relevant_speeches_found_and_undefined_without_any():
    assert recall_at_k({"a", "b", "c", "d"}, ["a", "x", "b"], 10) == pytest.approx(0.5)
    assert recall_at_k({"a"}, ["x"], 10) == 0.0
    assert recall_at_k(set(), ["x"], 10) is None  # an abstain question has nothing to find


def test_the_cut_is_exact():
    assert recall_at_k({"a"}, ["x", "y", "a"], 2) == 0.0 and recall_at_k({"a"}, ["x", "y", "a"], 3) == 1.0


def test_reciprocal_rank_is_one_over_the_first_relevant_position_within_the_cut():
    assert reciprocal_rank({"a"}, ["a"], 10) == 1.0
    assert reciprocal_rank({"a", "b"}, ["x", "y", "b", "a"], 10) == pytest.approx(1 / 3)
    assert reciprocal_rank({"a"}, ["x"] * 10 + ["a"], 10) == 0.0  # eleventh: outside the cut
    assert reciprocal_rank({"a"}, ["x"] * 9 + ["a"], 10) == pytest.approx(0.1)
    assert reciprocal_rank(set(), ["a"], 10) == 0.0


def test_ndcg_rewards_earlier_finds_and_is_one_for_a_perfect_list():
    assert ndcg_at_k({"a", "b"}, ["a", "b", "x"], 10) == pytest.approx(1.0)
    worse = ndcg_at_k({"a", "b"}, ["x", "a", "b"], 10)
    assert worse == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert 0 < worse < 1
    assert ndcg_at_k({"a"}, ["x"], 10) == 0.0 and ndcg_at_k(set(), ["a"], 10) is None


def test_ndcg_counts_a_speech_once_however_often_it_appears():
    once = ndcg_at_k({"a", "b"}, ["a", "x", "b"], 10)
    repeated = ndcg_at_k({"a", "b"}, ["a", "a", "a", "b"], 10)
    assert repeated == pytest.approx((1 + 1 / math.log2(5)) / (1 + 1 / math.log2(3)))  # the repeats add nothing
    assert repeated < 1.0 and once == pytest.approx((1 + 1 / 2) / (1 + 1 / math.log2(3)))


def test_ndcg_is_perfect_when_the_cut_is_smaller_than_the_number_of_relevant_speeches():
    assert ndcg_at_k({"a", "b", "c"}, ["a", "b", "c"], 2) == pytest.approx(1.0)  # only two can fit, and both are there


def test_the_bootstrap_is_reproducible_for_a_seed_and_reports_wins_losses_and_ties():
    diffs = [0.5, -0.2, 0.0, 0.3, 0.0, -0.1]
    a, b = bootstrap_paired(diffs, iterations=2000, seed=3), bootstrap_paired(diffs, iterations=2000, seed=3)
    assert a == b
    assert (a.wins, a.losses, a.ties) == (2, 2, 2)
    assert a.mean == pytest.approx(sum(diffs) / len(diffs)) and a.low <= a.mean <= a.high


def test_a_uniformly_positive_difference_has_an_interval_above_zero_and_a_zero_difference_is_pinned_at_zero():
    up = bootstrap_paired([0.2] * 12 + [0.3] * 6, iterations=1000, seed=1)
    assert up.low > 0 and up.wins == 18 and up.losses == 0
    flat = bootstrap_paired([0.0] * 10, iterations=1000, seed=1)
    assert (flat.mean, flat.low, flat.high) == (0.0, 0.0, 0.0) and flat.ties == 10


def test_a_wider_confidence_level_gives_a_wider_interval():
    diffs = [0.4, -0.3, 0.1, 0.0, -0.2, 0.5, 0.2, -0.1]
    narrow, wide = bootstrap_paired(diffs, confidence=0.5, seed=2), bootstrap_paired(diffs, confidence=0.99, seed=2)
    assert wide.high - wide.low > narrow.high - narrow.low


def test_the_bootstrap_of_nothing_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        bootstrap_paired([])
