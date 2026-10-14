"""The pre-registered adoption rule, against hand-worked values."""

from __future__ import annotations

import pytest

from retrieval import uplift


# --- Stage A tier 1: the free screen --------------------------------------------------------------------


def test_tier1_drops_an_arm_that_moves_nothing_before_anything_is_paid_for():
    """The optimisation the whole screen rests on: condensed-list metrics need no judgments, so an arm that reorders
    nothing among judged documents is dropped for free.
    """
    baseline = {"recall@10": 0.60, "mrr@10": 0.50}
    survivors, dropped = uplift.tier1_survivors(
        {
            "S1": {"recall@10": 0.6002, "mrr@10": 0.5001},  # noise
            "P50": {"recall@10": 0.6400, "mrr@10": 0.5300},  # real
        },
        baseline,
    )
    assert survivors == ["P50"]
    assert [d.arm for d in dropped] == ["S1"]
    assert "subsequence" in dropped[0].reason, "the reason has to say why the free screen is sound"


def test_tier1_keeps_an_arm_that_moves_one_metric_even_if_the_others_are_flat():
    survivors, dropped = uplift.tier1_survivors(
        {"D1": {"recall@10": 0.6001, "mrr@10": 0.5400}}, {"recall@10": 0.60, "mrr@10": 0.50})
    assert survivors == ["D1"] and dropped == []


def test_tier1_keeps_an_arm_it_cannot_score_rather_than_dropping_it_silently():
    """A missing number is not evidence of no effect. Dropping on absent data would retire an arm for free and log it
    as a measured negative.
    """
    survivors, dropped = uplift.tier1_survivors({"B1": {"recall@10": None}}, {"recall@10": 0.60})
    assert survivors == ["B1"] and dropped == []


def test_tier1_keeps_an_arm_that_moves_backwards_so_stage_c_can_see_it():
    """A regression is a finding. Only *no movement* is a no-op."""
    survivors, _ = uplift.tier1_survivors({"R1": {"recall@10": 0.50}}, {"recall@10": 0.60})
    assert survivors == ["R1"]


# --- the budget ----------------------------------------------------------------------------------------


def test_the_bill_is_priced_from_the_measured_rate():
    """3,959 judgments for $0.49. Not a guess, and not a round number."""
    assert uplift.price(3959).cost_usd == pytest.approx(0.49, abs=0.001)
    assert uplift.price(2298).cost_usd == pytest.approx(0.2845, abs=0.001)


def test_a_run_over_the_cap_is_refused_with_the_arithmetic_shown():
    budget = uplift.price(50_000, cap_usd=4.00)
    assert budget.within_cap is False and budget.ok is False
    assert "over this run's $4.00 cap" in budget.refusal()


def test_a_run_the_balance_cannot_cover_is_refused_even_under_the_cap():
    """The two questions fail for different reasons: a run can be inside its cap and still impossible."""
    budget = uplift.price(20_000, cap_usd=4.00, balance_usd=0.50)
    assert budget.within_cap is True and budget.affordable is False
    assert "Top up" in budget.refusal()


def test_a_run_inside_both_may_go_ahead_quietly():
    budget = uplift.price(6_900, cap_usd=4.00, balance_usd=3.51)
    assert budget.ok is True and budget.refusal() == ""


def test_an_unknown_balance_does_not_block_a_run_inside_the_cap():
    """The cap is the control we own; the balance is the provider's word for it."""
    assert uplift.price(1000, cap_usd=4.00, balance_usd=None).ok is True


def test_negative_pairs_are_a_bug_not_a_free_run():
    with pytest.raises(ValueError, match="cannot be negative"):
        uplift.price(-1)


def test_the_two_tier_screen_is_about_margin_not_impossibility():
    """Worth being exact, because an earlier draft of this claimed fourteen arms "do not fit" $3.51. They do: 2,298
    baseline pairs plus fourteen arms at 50% union overlap is about 18,400 pairs = $2.28, inside both the $4 cap and
    the balance.
    """
    fourteen = uplift.price(18_384, balance_usd=3.51)
    six = uplift.price(6_894, balance_usd=3.51)
    assert fourteen.ok is True and six.ok is True
    assert fourteen.cost_usd == pytest.approx(2.28, abs=0.01)
    assert six.cost_usd == pytest.approx(0.85, abs=0.01)
    assert 3.51 - fourteen.cost_usd < 1.30, "fourteen arms leave almost nothing for the rest of the endgame"


# --- Stage A tier 2: the four conditions ---------------------------------------------------------------


FOUR_WINDOWS = {"W0": 0.03, "W1": 0.02, "W2": 0.04, "W4": -0.01}


def test_an_arm_passes_only_when_all_four_conditions_hold():
    verdict = uplift.screen_arm("P50", {"recall@10": 0.03}, FOUR_WINDOWS, {"multi_speech": (0.01, 45)},
                                judged_at_10=0.97, p95_ms=900, baseline_p95_ms=800)  # fmt: skip
    assert verdict.passed is True and verdict.failures == []
    assert "PASS" in verdict.line()


def test_a_gain_under_the_floor_fails():
    verdict = uplift.screen_arm("S1", {"recall@10": 0.004}, FOUR_WINDOWS, {})
    assert verdict.passed is False
    assert "under the +0.01 floor" in verdict.failures[0]


def test_a_category_of_consequence_going_backwards_fails():
    """The protected-category rule that refused qwen3-0.6b in the bake-off, applied to the arms."""
    verdict = uplift.screen_arm("B1", {"recall@10": 0.05}, FOUR_WINDOWS,
                                {"multi_speech": (-0.05, 45), "temporal": (0.01, 6)})  # fmt: skip
    assert verdict.passed is False
    assert verdict.category_losses == {"multi_speech": -0.05}
    assert "multi_speech loses" in " ".join(verdict.failures)


def test_a_small_category_cannot_veto_an_arm():
    """Three items cannot support a veto."""
    verdict = uplift.screen_arm("B1", {"recall@10": 0.05}, FOUR_WINDOWS, {"comparison": (-0.40, 3)})
    assert verdict.category_losses == {} and verdict.passed is True


def test_a_sign_that_flips_across_windows_fails():
    flipping = {"W0": 0.05, "W1": -0.04, "W2": 0.06, "W4": -0.03}
    verdict = uplift.screen_arm("Q1", {"recall@10": 0.05}, flipping, {})
    assert verdict.passed is False
    assert "sign agrees in only 2 of 4" in " ".join(verdict.failures)


def test_sign_agreement_counts_a_consistent_negative_as_agreement():
    """It is a consistency check, not a direction check: Stage A's gain floor handles direction."""
    holds, agreeing, _ = uplift.sign_agreement({"W0": -0.05, "W1": -0.04, "W2": -0.06, "W4": 0.01})
    assert holds is True and agreeing == 3


def test_sign_agreement_ignores_windows_that_are_not_dev():
    """W3 is burned and W5 is the exam. Neither may influence a screen."""
    holds, agreeing, _ = uplift.sign_agreement({"W0": 0.05, "W1": 0.04, "W2": 0.06, "W3": -0.9, "W5": -0.9})
    assert holds is True and agreeing == 3


def test_no_per_window_numbers_fails_rather_than_passes():
    holds, _, why = uplift.sign_agreement({})
    assert holds is False and "cannot be checked" in why


def test_an_arm_over_the_latency_budget_fails():
    verdict = uplift.screen_arm("R2", {"recall@10": 0.05}, FOUR_WINDOWS, {}, p95_ms=1500)
    assert verdict.passed is False and "over the 1200 ms budget" in " ".join(verdict.failures)


def test_an_arm_that_adds_too_much_latency_fails_even_inside_the_budget():
    verdict = uplift.screen_arm("R1", {"recall@10": 0.05}, FOUR_WINDOWS, {}, p95_ms=1100, baseline_p95_ms=700)
    assert verdict.passed is False and "adds 400 ms" in " ".join(verdict.failures)


def test_an_arm_below_the_judged_gate_is_unscorable_not_failed():
    """A different verdict from failing: the arm may be fine, and we have not earned the right to say."""
    verdict = uplift.screen_arm("D2", {"recall@10": 0.05}, FOUR_WINDOWS, {}, judged_at_10=0.54)
    assert verdict.passed is False and verdict.unscorable
    assert "curriculum it never saw" in verdict.unscorable
    assert verdict.failures == [], "it failed no condition; it was never eligible"
    assert "NOT SCORABLE" in verdict.line()


def test_an_arm_with_no_computable_metric_is_unscorable():
    verdict = uplift.screen_arm("X", {}, FOUR_WINDOWS, {}, judged_at_10=1.0)
    assert verdict.unscorable and verdict.passed is False


# --- Stage B: the bundle -------------------------------------------------------------------------------


def test_one_arm_per_exclusive_group_the_best_on_its_target():
    bundle, aside = uplift.pick_within_groups(
        ["S1", "P30", "P50", "P50g", "D1"],
        {"pool": ["P30", "P50", "P50g"], "debate": ["D1", "D2"]},
        {"P30": 0.02, "P50": 0.05, "P50g": 0.03, "D1": 0.04},
    )
    assert bundle == ["S1", "P50", "D1"]
    assert aside == {"pool": ["P30", "P50g"]}


def test_a_group_tie_breaks_on_the_declared_order_so_the_bundle_is_reproducible():
    bundle, _ = uplift.pick_within_groups(["P30", "P50"], {"pool": ["P30", "P50"]},
                                          {"P30": 0.02, "P50": 0.02})  # fmt: skip
    assert bundle == ["P30"], "equal scores keep the first, not whichever the dict yielded last"


def test_an_arm_in_no_group_always_survives():
    bundle, aside = uplift.pick_within_groups(["S1", "R1"], {"pool": ["P30"]}, {})
    assert bundle == ["S1", "R1"] and aside == {}


def test_drop_one_retires_an_arm_the_bundle_does_not_need():
    """A bundle carrying a component that does nothing gives a false account of what earned the gain."""
    droppable, moved = uplift.drop_one(
        {"recall@10": 0.64, "mrr@10": 0.53},
        {
            "R1": {"recall@10": 0.6402, "mrr@10": 0.5299},  # removing it changes nothing
            "P50": {"recall@10": 0.6000, "mrr@10": 0.5000},  # removing it costs
        },
    )
    assert droppable == ["R1"]
    assert moved["P50"]["recall@10"] == pytest.approx(-0.04)


def test_drop_one_keeps_an_arm_whose_effect_cannot_be_measured():
    droppable, _ = uplift.drop_one({"recall@10": 0.64}, {"B1": {"recall@10": None}})
    assert droppable == []


# --- Stage C: the decision -----------------------------------------------------------------------------


def test_the_bundle_is_adopted_when_something_is_established_and_nothing_is_worse():
    got = uplift.decide(["S1", "P50"], {"recall@10": True, "mrr@10": False}, {"recall@10": 0.03, "mrr@10": 0.01})
    assert got.adopt == ["S1", "P50"] and got.established == ["recall@10"]
    assert "adopted and retrieval freezes" in got.reason


def test_anything_significantly_worse_adopts_nothing():
    got = uplift.decide(["S1", "P50"], {"recall@10": True}, {"recall@10": 0.03}, significantly_worse=["mrr@10"])
    assert got.adopt == [] and got.worse == ["mrr@10"]
    assert "nothing is adopted" in got.reason


def test_nothing_established_but_no_harm_adopts_s1_alone():
    """S1 is named in the rule, not chosen here: merging duplicate sentences of one speech cannot lower a metric
    measured at speech level.
    """
    got = uplift.decide(["S1", "P50", "R1"], {"recall@10": False}, {"recall@10": 0.004, "mrr@10": 0.0})
    assert got.adopt == ["S1"]
    assert "cannot lower a speech-level metric" in got.reason
    assert "weak evidence" in got.reason, "the decision has to declare how thin its evidence is"


def test_a_negative_point_estimate_adopts_nothing_even_without_significance():
    got = uplift.decide(["S1"], {"recall@10": False}, {"recall@10": -0.01})
    assert got.adopt == [] and "point downwards" in got.reason


def test_nothing_is_adopted_when_s1_was_not_in_the_bundle():
    got = uplift.decide(["P50"], {"recall@10": False}, {"recall@10": 0.0})
    assert got.adopt == [] and "S1 was not in the bundle" in got.reason


def test_retrieval_freezes_whatever_is_adopted():
    """The freeze is not conditional on a win. E1's job is to stop changing retrieval."""
    for adoption in (uplift.decide([], {}, {}), uplift.decide(["S1"], {"recall@10": True}, {"recall@10": 0.1})):
        assert adoption.freezes is True


# --- reporting a p-value honestly ----------------------------------------------------------------------


def test_a_p_below_the_bootstrap_floor_is_reported_as_an_inequality():
    """With 10,000 resamples the smallest resolvable p is 0.0001."""
    assert uplift.floor_p(0.0, 10_000) == "p < 0.0001"
    assert uplift.floor_p(0.0, 5_000) == "p < 0.0002"


def test_a_resolvable_p_is_printed_as_itself():
    assert uplift.floor_p(0.0384, 10_000) == "p = 0.0384"


def test_zero_iterations_is_a_bug():
    with pytest.raises(ValueError, match="must be positive"):
        uplift.floor_p(0.5, 0)


# --- the constants are the pre-registration ------------------------------------------------------------


def test_the_thresholds_are_the_ones_written_down_before_any_arm_ran():
    """If one of these changes after an arm has run, the rule was tuned on the results and the whole screen is
    worthless. They are asserted so a change has to be deliberate and visible in a diff.
    """
    assert uplift.MIN_GAIN == 0.01
    assert (uplift.CATEGORY_TOLERANCE, uplift.CATEGORY_MIN_ITEMS) == (0.03, 10)
    assert (uplift.WINDOWS_REQUIRED, uplift.DEV_WINDOWS) == (3, ("W0", "W1", "W2", "W4"))
    assert uplift.NOOP_EPSILON == 0.005
    assert (uplift.LATENCY_P95_MS, uplift.LATENCY_HEADROOM_MS) == (1200, 250)
    assert uplift.SPEND_CAP_USD == 4.00
    assert uplift.CONFIRMATORY == ("recall@10", "mrr@10", "normalised_recall@10:multi_speech",
                                   "debate_recall@3:who_spoke")  # fmt: skip
