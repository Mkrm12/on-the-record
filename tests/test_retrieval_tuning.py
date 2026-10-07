"""These are the tests that stop a tuning run from manufacturing findings. Every one of them corresponds to a way a
sweep over two dozen knob values produces a number that looks like a result and is not.
"""

from __future__ import annotations

import pytest
from retrieval import tuning
from retrieval.evaluation import QueryOutcome


def outcome(qid: str, relevant: set[str], ranked: list[str]) -> QueryOutcome:
    return QueryOutcome(qid=qid, category="exact_term", relevant=frozenset(relevant), ranked=ranked,
                        abstain=False, top_rerank=None, timings={})  # fmt: skip


class Item:
    """The parts of a v2 gold item the tuning code reads."""

    def __init__(self, ident, graded=None, evidence=(), window="W0", split="dev",
                 behaviour="answer", testable=True, category="exact_term"):  # fmt: skip
        self.id = ident
        self.question = f"question for {ident}?"
        self.category = category
        self.graded_relevance = graded or {}
        self.relevant_speech_ids = tuple(evidence)
        self.window = window
        self.split = split
        self.expected_behaviour = behaviour
        self.retrieval_testable = testable


# --- held-out protection -----------------------------------------------------------------------------


def test_the_test_split_refuses_without_confirm():
    with pytest.raises(tuning.HeldOutViolation, match="confirms the final configuration once"):
        tuning.guard_split("test")
    tuning.guard_split("test", confirm=True)


def test_w3_refuses_without_the_final_exam_flag():
    """A warning would not do: the value of a held-out set is that nobody looked, and that is lost by a harness that
    mentions it and carries on.
    """
    with pytest.raises(tuning.HeldOutViolation, match="scored ONCE"):
        tuning.guard_split("heldout")
    with pytest.raises(tuning.HeldOutViolation):
        tuning.guard_split("heldout", confirm=True)
    tuning.guard_split("heldout", final_exam=True)


def test_dev_needs_no_unlocking_and_w3_is_not_a_dev_window():
    tuning.guard_split("dev")
    assert tuning.HELDOUT_WINDOW not in tuning.DEV_WINDOWS


# --- relevance from the pool --------------------------------------------------------------------------


def test_relevance_is_binarised_at_two_because_one_is_not_an_answer():
    item = Item("a", graded={"s1": 3, "s2": 2, "s3": 1, "s4": 0})
    assert tuning.relevant_speeches(item) == {"s1", "s2"}


def test_the_threshold_is_configurable_for_a_sensitivity_check():
    item = Item("a", graded={"s1": 3, "s2": 2, "s3": 1, "s4": 0})
    assert tuning.relevant_speeches(item, threshold=1) == {"s1", "s2", "s3"}
    assert tuning.relevant_speeches(item, threshold=3) == {"s1"}


def test_an_items_own_evidence_stays_relevant_even_if_the_judge_disagreed():
    """The speech was chosen because it answers the question. A judge disagreeing is measured in the calibration
    sample; it is not a licence to unlabel ground truth in the middle of an experiment.
    """
    item = Item("a", graded={"s1": 0}, evidence=("s1", "s9"))
    assert tuning.relevant_speeches(item) == {"s1", "s9"}


def test_pooled_relevance_fixes_the_i39_undercount():
    """The v1 labels carried one quote per question, so finding a different speech that also answered was scored as a
    miss. The pool is what makes that no longer true.
    """
    item = Item("a", graded={"other": 3}, evidence=("original",))
    assert tuning.relevant_speeches(item) == {"original", "other"}


def test_only_items_with_relevance_and_retrieval_testability_are_scored():
    items = [
        Item("has", graded={"s1": 3}),
        Item("none", graded={"s1": 1}),  # nothing at or above the threshold
        Item("structured", graded={"s1": 3}, testable=False),
    ]
    assert [i.id for i in tuning.scorable(items)] == ["has"]


def test_a_gold_query_carries_what_the_searcher_harness_needs():
    query = tuning.GoldQuery.of(Item("a", graded={"s2": 2}, evidence=("s1",), behaviour="answer"))
    assert query.id == "a" and query.window == "W0" and query.split == "dev"
    assert {r.speech_id for r in query.relevant} == {"s1", "s2"}
    assert query.expected_abstain is False


def test_an_abstain_item_is_marked_as_such():
    assert tuning.GoldQuery.of(Item("a", evidence=("s1",), behaviour="abstain_not_in_record")).expected_abstain


# --- the family correction ---------------------------------------------------------------------------


def test_holm_bonferroni_tightens_the_bar_for_the_smallest_p_value():
    family = [
        tuning.Comparison("a", "recall", 100, [0.1], _interval(0.1), p=0.001),
        tuning.Comparison("b", "recall", 100, [0.1], _interval(0.1), p=0.02),
        tuning.Comparison("c", "recall", 100, [0.1], _interval(0.1), p=0.30),
    ]
    tuning.holm_bonferroni(family, alpha=0.05)
    a, b, c = family
    assert a.adjusted_alpha == pytest.approx(0.05 / 3)
    assert b.adjusted_alpha == pytest.approx(0.05 / 2)
    assert c.adjusted_alpha == pytest.approx(0.05 / 1)
    assert a.established and b.established and not c.established


def test_holm_is_a_step_down_so_a_failure_stops_everything_larger():
    """Without the step-down, a big p-value could pass on its own generous alpha while a smaller one failed -- which
    would let 'we tried twenty knobs and one worked' back in through the side door.
    """
    family = [
        tuning.Comparison("small", "recall", 100, [0.1], _interval(0.1), p=0.04),
        tuning.Comparison("large", "recall", 100, [0.1], _interval(0.1), p=0.049),
    ]
    tuning.holm_bonferroni(family, alpha=0.05)
    assert not family[0].established, "0.04 > 0.05/2, so it fails"
    assert not family[1].established, "and everything above it fails with it"


def test_a_single_comparison_is_not_penalised():
    [only] = tuning.holm_bonferroni(
        [tuning.Comparison("a", "recall", 100, [0.1], _interval(0.1), p=0.04)], alpha=0.05
    )
    assert only.adjusted_alpha == pytest.approx(0.05) and only.established


def test_a_bootstrap_p_value_is_small_for_a_consistent_difference_and_large_for_noise():
    consistent = tuning.bootstrap_p([0.2] * 30, seed=1)
    noisy = tuning.bootstrap_p([0.2, -0.2] * 15, seed=1)
    assert consistent < 0.01
    assert noisy > 0.5


def test_a_bootstrap_p_value_of_an_empty_comparison_is_one():
    assert tuning.bootstrap_p([]) == 1.0


# --- the adoption rule -------------------------------------------------------------------------------


def test_a_variant_is_not_adopted_on_a_mean_alone():
    base = [outcome(f"q{i}", {"gold"}, ["gold"] if i % 2 else ["miss"]) for i in range(20)]
    other = [outcome(f"q{i}", {"gold"}, ["gold"] if i % 2 or i == 0 else ["miss"]) for i in range(20)]
    c = tuning.compare(base, other, "one lucky query", metric="recall", k=10)
    assert c.mean > 0, "the mean improved"
    tuning.holm_bonferroni([c])
    assert not tuning.adopt(c), "but one query out of twenty is not an established gain"


def test_a_consistent_gain_is_adopted():
    base = [outcome(f"q{i}", {"gold"}, ["miss", "gold"]) for i in range(30)]
    other = [outcome(f"q{i}", {"gold"}, ["gold"]) for i in range(30)]
    c = tuning.compare(base, other, "always better", metric="mrr", k=10)
    tuning.holm_bonferroni([c])
    c.holds_across_windows = True
    assert tuning.adopt(c)


def test_a_gain_carried_by_one_window_is_refused():
    base = [outcome(f"q{i}", {"gold"}, ["miss"]) for i in range(20)]
    other = [outcome(f"q{i}", {"gold"}, ["gold"] if i < 10 else ["miss", "miss2"]) for i in range(20)]
    window_of = {f"q{i}": ("W0" if i < 10 else "W1") for i in range(20)}
    c = tuning.compare(base, other, "W0 only", metric="recall", k=10)
    tuning.holm_bonferroni([c])
    tuning.check_across_windows(c, base, other, window_of)
    assert c.per_window["W0"] > 0 and c.per_window["W1"] == 0
    # A window that is flat is not a sign flip, so this one holds; make W1 actively worse and it must not.
    worse = [outcome(f"q{i}", {"gold"}, ["gold"] if i < 10 else []) for i in range(20)]
    base_with_hits = [outcome(f"q{i}", {"gold"}, ["gold"] if i >= 10 else ["miss"]) for i in range(20)]
    d = tuning.compare(base_with_hits, worse, "flips", metric="recall", k=10)
    tuning.check_across_windows(d, base_with_hits, worse, window_of)
    assert d.holds_across_windows is False
    tuning.holm_bonferroni([d])
    assert not tuning.adopt(d)


def test_a_negative_established_difference_is_never_adopted():
    base = [outcome(f"q{i}", {"gold"}, ["gold"]) for i in range(30)]
    other = [outcome(f"q{i}", {"gold"}, ["miss"]) for i in range(30)]
    c = tuning.compare(base, other, "worse", metric="recall", k=10)
    tuning.holm_bonferroni([c])
    c.holds_across_windows = True
    assert c.established and c.mean < 0
    assert not tuning.adopt(c)


def test_one_window_of_data_cannot_claim_to_hold_across_windows():
    base = [outcome(f"q{i}", {"gold"}, ["miss"]) for i in range(10)]
    other = [outcome(f"q{i}", {"gold"}, ["gold"]) for i in range(10)]
    c = tuning.compare(base, other, "single window", metric="recall", k=10)
    tuning.check_across_windows(c, base, other, {f"q{i}": "W0" for i in range(10)})
    assert c.holds_across_windows is None, "unknown, not true"


# --- the knob chooser --------------------------------------------------------------------------------


def test_a_plateau_is_preferred_to_a_peak():
    """A peak on a 237-query set is mostly sampling noise, and the cheapest value on the plateau is the one that
    survives a new window.
    """
    scores = {10: 0.700, 20: 0.752, 30: 0.754, 50: 0.753}
    assert tuning.plateau(scores, tolerance=0.005) == 20


def test_a_real_difference_is_not_flattened_by_the_tolerance():
    assert tuning.plateau({10: 0.60, 20: 0.75}, tolerance=0.005) == 20


def test_the_plateau_handles_tuple_valued_knobs_like_leg_weights():
    scores = {(1.0, 1.0, 1.0): 0.75, (2.0, 1.0, 1.0): 0.752, (1.0, 0.5, 0.5): 0.60}
    assert tuning.plateau(scores, tolerance=0.01) == (1.0, 1.0, 1.0)


def test_leave_one_window_out_rotates_over_the_dev_windows_only():
    folds = tuning.lowo_folds()
    assert [f.held for f in folds] == list(tuning.DEV_WINDOWS)
    assert all(tuning.HELDOUT_WINDOW not in f.tune_on for f in folds)
    assert all(f.held not in f.tune_on for f in folds)


def test_a_knob_every_fold_agrees_on_is_reported_stable():
    scores = {
        20: {"W0": 0.70, "W1": 0.70, "W2": 0.70, "W4": 0.70},
        50: {"W0": 0.60, "W1": 0.60, "W2": 0.60, "W4": 0.60},
    }
    result = tuning.tune_knob("pool", scores)
    assert result.chosen == 20 and result.stable and not result.note


def test_a_knob_the_folds_disagree_on_is_flagged_weakly_supported():
    """This is the case that matters: the folds disagreeing IS the finding, and a harness that quietly picked one
    fold's favourite would hide it.
    """
    # Two windows favour 20 and two favour 50, hard enough that which value wins depends on which window is held out.
    scores = {
        20: {"W0": 0.90, "W1": 0.90, "W2": 0.50, "W4": 0.50},
        50: {"W0": 0.50, "W1": 0.50, "W2": 0.90, "W4": 0.90},
    }
    result = tuning.tune_knob("pool", scores)
    assert not result.stable
    assert "weakly supported" in result.note
    assert len({repr(v) for v in result.per_fold_choice.values()}) > 1


def test_the_held_out_score_is_read_at_the_value_chosen_without_it():
    scores = {
        20: {"W0": 0.10, "W1": 0.80, "W2": 0.80, "W4": 0.80},
        50: {"W0": 0.90, "W1": 0.10, "W2": 0.10, "W4": 0.10},
    }
    result = tuning.tune_knob("pool", scores)
    # Tuning on W1/W2/W4 picks 20, and the honest estimate is 20's score on W0 (0.10), not 50's (0.90).
    assert result.per_fold_choice["W0"] == 20
    assert result.held_out_scores["W0"] == pytest.approx(0.10)


def test_the_primary_metric_is_the_candidate_ceiling_not_the_presentation_depth():
    """The reranker cannot recover what the first stage never retrieved."""
    assert tuning.PRIMARY_K == 100
    assert tuning.RELEVANT_AT == 2


def test_paired_differences_only_use_queries_both_runs_scored():
    base = [outcome("a", {"g"}, ["g"]), outcome("b", {"g"}, ["x"])]
    other = [outcome("a", {"g"}, ["x"])]
    assert tuning.paired(base, other, "recall", 10) == [-1.0]


def test_a_query_with_no_relevant_speeches_is_left_out_of_every_metric():
    assert tuning.metric_of(outcome("a", set(), ["x"]), "recall", 10) is None
    assert tuning.mean_metric([outcome("a", set(), ["x"])], "recall", 10) is None


def _interval(mean: float):
    from retrieval.metrics import Bootstrap

    return Bootstrap(mean=mean, low=mean / 2, high=mean * 2, wins=1, losses=0, ties=0)

# `pool=50` beat the baseline by +0.025 recall@100 with an established interval, p = 0.0000 and the same sign in all
# four dev windows -- and improved nothing.


def test_the_depth_a_truncation_knob_can_be_compared_at_is_the_shortest_list_any_variant_returned():
    assert tuning.comparable_depth({10: 10, 20: 20, 50: 48}) == 10
    assert tuning.comparable_depth({}) == 0


def test_a_truncation_knob_compared_deeper_than_it_can_fill_is_flagged():
    why = tuning.truncation_confound("pool", 100, {10: 10, 20: 20, 50: 48})
    assert "controls how many results come back" in why
    assert "depth 10 or shallower" in why, "the report must say the depth it WOULD be valid at"


def test_the_same_knob_at_a_depth_every_variant_can_fill_is_not_flagged():
    assert tuning.truncation_confound("pool", 10, {10: 10, 20: 20, 50: 48}) == ""


def test_a_knob_that_does_not_change_the_result_length_is_never_flagged():
    assert tuning.truncation_confound("rrf_k", 100, {10: 10, 60: 10}) == ""
    assert tuning.truncation_confound("sparse_coverage", 100, {0.3: 20, 0.4: 20}) == ""


def test_every_knob_that_caps_the_output_is_in_the_list():
    """Missing one would let the same fake finding back in under a different name."""
    assert {"pool", "fused_limit", "dense_limit", "sparse_limit", "speaker_limit"} <= tuning.TRUNCATION_KNOBS


def test_a_confounded_comparison_is_refused_however_good_its_statistics():
    """The real numbers from the sweep: established, p = 0.0000, sign holds in all four windows."""
    base = [outcome(f"q{i}", {"gold"}, ["miss"] * 20) for i in range(40)]
    better = [outcome(f"q{i}", {"gold"}, ["miss"] * 47 + ["gold"]) for i in range(40)]
    c = tuning.compare(base, better, "pool=50", metric="recall", k=100)
    tuning.holm_bonferroni([c])
    c.holds_across_windows = True
    assert c.established and c.mean > 0, "the statistics say adopt"
    assert tuning.adopt(c), "and without the confound it would be adopted"

    c.confound = tuning.truncation_confound("pool", 100, {20: 20, 50: 48})
    assert c.confound
    assert not tuning.adopt(c), "with the confound it must be refused"


def test_the_confound_appears_in_the_reported_line_so_it_cannot_be_missed():
    c = tuning.compare(
        [outcome("q", {"g"}, ["x"])], [outcome("q", {"g"}, ["g"])], "pool=50", metric="recall", k=100
    )
    c.confound = "pool controls how many results come back"
    assert "CONFOUNDED" in c.line()
