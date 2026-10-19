"""The measurement repairs (E1 section 4.1), against hand-worked values."""

from __future__ import annotations

import pytest

from retrieval.audit import (
    JUDGED_GATE,
    attainable_recall,
    audit_arm,
    condense,
    condensed_pair,
    final_list_lengths,
    judged_at_k,
    normalised_recall,
    r_precision,
    scorable_at_gate,
    wasted_share,
    wasted_slots,
)


class Outcome:
    """The shape `retrieval.evaluation` produces, reduced to what the audit reads."""

    def __init__(self, qid, relevant, ranked, category=""):
        self.qid, self.relevant, self.ranked, self.category = qid, frozenset(relevant), list(ranked), category


# --- the metric's own ceiling ---------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "k", "expected"), [
    (1, 10, 1.0),
    (10, 10, 1.0),
    (14, 10, 10 / 14),  # a perfect ranking scores 0.714 and looks like a failure
    (20, 10, 0.5),
    (3, 100, 1.0),
])
def test_attainable_recall_is_the_ceiling_a_perfect_ranking_would_hit(count, k, expected):
    assert attainable_recall(count, k) == pytest.approx(expected)


def test_a_query_with_no_relevant_speeches_has_no_attainable_recall():
    """Better to raise than to return 1.0 and have an abstain question silently count as a perfect score."""
    with pytest.raises(ValueError, match="no relevant speeches"):
        attainable_recall(0, 10)


def test_normalised_recall_separates_a_ranking_failure_from_a_list_that_was_too_short():
    """Twelve relevant speeches, ten slots, and the ten best are all relevant. Raw recall says 0.833 and looks like a
    miss; normalised says 1.0, which is the truth about the ranking.
    """
    relevant = [f"s{i}" for i in range(12)]
    perfect = [f"s{i}" for i in range(10)]
    assert pytest.approx(10 / 12) == normalised_recall(relevant, perfect, 10) * attainable_recall(12, 10)
    assert normalised_recall(relevant, perfect, 10) == pytest.approx(1.0)


def test_normalised_recall_is_no_kinder_when_the_answer_fits():
    """With |R| <= k there is no ceiling to normalise away, so it must equal the raw number. Otherwise normalising
    would inflate every ordinary query.
    """
    relevant, ranked = ["a", "b"], ["a", "x", "b"]
    assert normalised_recall(relevant, ranked, 10) == pytest.approx(1.0)
    assert normalised_recall(relevant, ["a", "x"], 10) == pytest.approx(0.5)


def test_normalised_recall_is_none_for_an_abstain_question():
    assert normalised_recall([], ["a"], 10) is None


def test_r_precision_uses_each_querys_own_depth():
    """Two relevant speeches: precision is taken at depth 2, not 10, so a query with a small answer is not punished for
    the eight empty slots underneath it.
    """
    assert r_precision(["a", "b"], ["a", "b", "c", "d"]) == pytest.approx(1.0)
    assert r_precision(["a", "b"], ["a", "c", "b"]) == pytest.approx(0.5)
    assert r_precision(["a"], ["x", "a"]) == pytest.approx(0.0)
    assert r_precision([], ["a"]) is None


# --- is the depth real? --------------------------------------------------------------------------


def test_the_list_length_decides_whether_a_depth_may_be_claimed():
    """Measured on the real harness: the final list is about 19 speeches because pool=20 caps it, so the column called
    recall@100 is R@final. Claiming 100 measures how long the list is.
    """
    lengths = final_list_lengths([["a"] * 19, ["a"] * 20, ["a"] * 18])
    assert (lengths.minimum, lengths.median, lengths.maximum) == (18, 19, 20)
    assert lengths.honest_depth(100) == 18
    assert lengths.label(100) == "R@final"
    assert "really at depth 18" in lengths.note(100)


def test_a_list_that_reaches_the_claimed_depth_keeps_its_honest_name():
    lengths = final_list_lengths([["a"] * 100, ["a"] * 120])
    assert lengths.label(100) == "recall@100" and lengths.note(100) == ""
    assert lengths.honest_depth(100) == 100


def test_no_rankings_gives_no_lengths_rather_than_a_zero():
    assert final_list_lengths([]) is None


# --- the pooling blind spot -----------------------------------------------------------------------------


def test_judged_at_k_counts_a_grade_of_zero_as_judged():
    """A speech graded 0 has been looked at. Counting it as unjudged would make a well-covered arm look uncovered and
    block it at the gate for the wrong reason.
    """
    assert judged_at_k({"a", "b", "c"}, ["a", "b", "c", "d"], 4) == pytest.approx(0.75)
    assert judged_at_k({"a"}, ["a"], 10) == pytest.approx(1.0)
    assert judged_at_k(set(), [], 10) is None


def test_an_arm_below_the_gate_is_refused_with_the_queries_named():
    """The failure this prevents: comparing an arm whose results nobody judged, where every new document it surfaces is
    scored as a miss. A genuinely better arm would measure worse.
    """
    ok, why = scorable_at_gate({"q1": 0.5, "q2": 0.6})
    assert ok is False
    assert "below the 0.95 gate" in why and "q1" in why


def test_an_arm_at_the_gate_is_scorable():
    ok, why = scorable_at_gate({"q1": 1.0, "q2": 0.95})
    assert ok is True and why == ""
    assert JUDGED_GATE == 0.95


def test_an_arm_that_returned_nothing_is_refused_rather_than_scored_as_zero():
    ok, why = scorable_at_gate({"q1": None})
    assert ok is False and "nothing to score" in why


def test_the_condensed_list_drops_only_what_was_never_judged():
    assert condense({"a", "c"}, ["a", "b", "c", "d"]) == ["a", "c"]
    assert condense(set(), ["a"]) == []


def test_condensing_lifts_a_score_that_the_blind_spot_was_holding_down():
    """The relevant speech is third, behind two documents nobody ever judged. Raw MRR says 0.333 because it assumes the
    two unknowns are irrelevant; condensed says 1.0 because among what we know, it is first.
    """
    judged, relevant, ranked = {"rel"}, {"rel"}, ["unjudged1", "unjudged2", "rel"]
    assert condensed_pair(judged, relevant, ranked, 10) == (pytest.approx(1.0), pytest.approx(1.0))
    from retrieval.metrics import reciprocal_rank

    assert reciprocal_rank(relevant, ranked, 10) == pytest.approx(1 / 3)


def test_a_relevant_speech_counts_as_judged_even_if_the_pool_map_forgot_it():
    """`relevant_speeches` unions the judged pool with the item's evidence, so an evidence speech can be relevant
    without appearing in `graded_relevance`. Condensing must not delete it.
    """
    assert condense({"other"} | {"rel"}, ["rel"]) == ["rel"]
    got = condensed_pair(judged={"other"}, relevant={"rel"}, ranked=["rel"], k=10)
    assert got == (pytest.approx(1.0), pytest.approx(1.0))


# --- wasted slots ---------------------------------------------------------------------------------------


def test_wasted_slots_counts_repeats_of_a_speech_not_distinct_speeches():
    assert wasted_slots(["a", "a", "b", "a"], 10) == 2
    assert wasted_slots(["a", "b", "c"], 10) == 0
    assert wasted_slots(["a", "a"], 1) == 0, "a repeat below the cut is not a wasted slot"


def test_wasted_share_reports_the_rate_and_the_raw_counts():
    share, wasted, total = wasted_share([["a", "a", "b"], ["c", "d", "e"]], 3)
    assert (wasted, total) == (1, 6) and share == pytest.approx(1 / 6)


def test_wasted_share_of_nothing_is_zero_not_a_division_error():
    assert wasted_share([], 10) == (0.0, 0, 0)


# --- the whole audit ------------------------------------------------------------------------------------


def test_audit_arm_refuses_to_score_an_arm_the_pool_does_not_cover():
    """Judged@10 is 0.54, so the baseline itself is below the gate until the incremental judging in section 4.7 has
    run. The audit says so instead of printing a number.
    """
    outcomes = [Outcome("q1", {"rel"}, ["rel", "u1", "u2", "u3"])]
    audit = audit_arm("baseline", "dev", outcomes, {"q1": {"rel"}})
    assert audit.scorable is False and "below the 0.95 gate" in audit.refusal
    assert audit.recall_at_10 == pytest.approx(1.0), "the number is still computed, just not trusted"
    assert "NOT SCORABLE" in audit.line()


def test_audit_arm_reports_both_the_raw_and_the_condensed_number():
    outcomes = [Outcome("q1", {"rel"}, ["u1", "rel"])]
    audit = audit_arm("arm", "dev", outcomes, {"q1": {"rel", "u1"}})
    assert audit.scorable is True
    assert audit.mrr_at_10 == pytest.approx(0.5)
    assert audit.condensed_mrr_at_10 == pytest.approx(0.5), "u1 is judged, so condensing changes nothing"


def test_audit_arm_carries_the_ceiling_so_a_reader_can_see_it_was_reachable():
    """Answers the question E1's report has to answer: how much of the weakness was the metric? With |R| under 10 the
    attainable recall is 1.0 and the answer is 'none of it'.
    """
    outcomes = [Outcome("q1", {"a", "b"}, ["a", "b"]), Outcome("q2", {"c"}, ["x", "c"])]
    audit = audit_arm("arm", "dev", outcomes, {"q1": {"a", "b"}, "q2": {"c", "x"}})
    assert audit.attainable_recall_at_10 == pytest.approx(1.0)
    assert audit.mean_relevant == pytest.approx(1.5)
    assert audit.normalised_recall_at_10 == pytest.approx(audit.recall_at_10)


def test_audit_arm_skips_abstain_questions_without_counting_them_as_failures():
    outcomes = [Outcome("q1", set(), ["a", "b"]), Outcome("q2", {"c"}, ["c"])]
    audit = audit_arm("arm", "dev", outcomes, {"q1": {"a", "b"}, "q2": {"c"}})
    assert audit.queries == 1, "only the scorable query counts towards the metric"
    assert audit.recall_at_10 == pytest.approx(1.0)
    assert audit.lengths.queries == 2, "but both lists count towards the length audit"


def test_audit_arm_survives_an_arm_that_returned_nothing_at_all():
    audit = audit_arm("broken", "dev", [Outcome("q1", {"rel"}, [])], {"q1": {"rel"}})
    assert audit.scorable is False
    assert audit.recall_at_10 == pytest.approx(0.0) and audit.mrr_at_10 == pytest.approx(0.0)
