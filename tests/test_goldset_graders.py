"""Multi-grader comparison: the anonymisation that makes a second blind run blind, and the boundary logic."""

from __future__ import annotations

import json

import pytest
from goldset import judge as judging
from goldset.blind import anonymise, assert_blind, crosses_boundary, load_grader_file


# --- anonymisation -----------------------------------------------------------------------------------


def worksheet(n: int = 4) -> dict:
    return {
        "gold_hash": "h",
        "calibration_pairs": {
            f"k:adv{i:04d}:SPEECH{i}": {
                "item_id": f"adv{i:04d}", "speech_id": f"SPEECH{i}",
                "question": f"question {i}?", "passage": f"passage {i}", "speaker": "A Member",
            }  # fmt: skip
            for i in range(n)
        },
        "items": {f"adv{i:04d}": {"id": f"adv{i:04d}", "question": f"question {i}?"} for i in range(n)},
    }


def test_the_real_ids_do_not_travel_with_the_passages():
    """`item_id` and `speech_id` would hand the second grader the mapping to the first run's file."""
    anonymous, _key = anonymise(worksheet(), seed=1)
    for pair in anonymous["calibration_pairs"].values():
        assert "item_id" not in pair and "speech_id" not in pair
    for item in anonymous["items"].values():
        assert "id" not in item
    assert all(k.startswith("p") for k in anonymous["calibration_pairs"])


def test_the_key_maps_every_opaque_id_back_and_nothing_else():
    anonymous, key = anonymise(worksheet(), seed=1)
    assert set(key) == set(anonymous["calibration_pairs"]) | set(anonymous["items"])
    assert set(key.values()) == set(worksheet()["calibration_pairs"]) | set(worksheet()["items"])


def test_the_order_is_shuffled_so_position_does_not_leak_the_original_order():
    anonymous, key = anonymise(worksheet(12), seed=3)
    positions = [key[f"p{n:04d}"] for n in range(1, 13)]
    original = list(worksheet(12)["calibration_pairs"])
    assert positions != original, "an unshuffled worksheet lines up with the first run by position alone"
    assert sorted(positions) == sorted(original), "shuffling must not lose or duplicate a pair"


def test_the_shuffle_is_reproducible_from_its_seed():
    a, ka = anonymise(worksheet(8), seed=5)
    b, kb = anonymise(worksheet(8), seed=5)
    assert ka == kb and list(a["calibration_pairs"]) == list(b["calibration_pairs"])
    _c, kc = anonymise(worksheet(8), seed=6)
    assert kc != ka, "a different seed must give a different mapping"


def test_the_anonymous_worksheet_still_refuses_to_carry_a_grade():
    """The allow-list is the first guard and `assert_blind` is the second; anonymising must not skip it."""
    anonymous, _ = anonymise(worksheet(), seed=1)
    assert_blind(anonymous)
    with pytest.raises(ValueError, match="must not carry judge output"):
        assert_blind({"calibration_pairs": {"p0001": {"passage": "x", "graded_relevance": {"s": 3}}}})


# --- which disagreements can move a metric -----------------------------------------------------------


crossing = crosses_boundary


@pytest.mark.parametrize("a, b", [(0, 1), (2, 3), (1, 0), (3, 2)])
def test_a_disagreement_inside_a_side_of_the_boundary_cannot_move_any_metric(a, b):
    assert not crossing(a, b)


@pytest.mark.parametrize("a, b", [(1, 2), (0, 2), (1, 3), (0, 3)])
def test_a_disagreement_across_the_boundary_can_move_every_metric(a, b):
    assert crossing(a, b)


def test_the_boundary_is_two_not_one():
    """UMBRELA 1 is "on topic, does not answer". Counting it as relevant would inflate every recall number with
    passages that answer nothing.
    """
    assert judging.RELEVANT_AT == 2
    assert crossing(1, 2) and not crossing(0, 1)


# --- agreement between graders -----------------------------------------------------------------------


def test_two_graders_arguing_only_about_zero_versus_one_agree_perfectly_on_relevance():
    """The four-level number can look poor while the graders make the identical call on everything the metrics read.
    """
    a = {("i", f"s{n}"): g for n, g in enumerate([0, 1, 0, 1, 2, 3])}
    b = {("i", f"s{n}"): g for n, g in enumerate([1, 0, 1, 0, 2, 3])}
    assert judging.cohen_kappa(a, b)["kappa"] < 0.6
    assert judging.binary_agreement(a, b)["kappa"] == 1.0


def test_a_grader_that_misses_answers_fails_the_relevance_gate_however_close_its_digits_look():
    a = {("i", f"s{n}"): g for n, g in enumerate([3, 3, 2, 2, 0, 0, 0, 0])}
    b = {("i", f"s{n}"): g for n, g in enumerate([1, 1, 1, 1, 0, 0, 0, 0])}
    binary = judging.binary_agreement(a, b)
    assert binary["only_first_relevant"] == 4 and not binary["passes_gate"]


def test_agreement_is_computed_only_over_pairs_both_graders_scored():
    a = {("i", "s1"): 3, ("i", "s2"): 0}
    b = {("i", "s1"): 3}
    assert judging.cohen_kappa(a, b)["n"] == 1
    assert judging.binary_agreement(a, b)["n"] == 1


def test_an_ungraded_pair_is_not_silently_a_zero():
    """Unjudged and irrelevant are different claims, and conflating them would invent agreement."""
    assert judging.cohen_kappa({("i", "s1"): 0}, {})["n"] == 0
    assert judging.binary_agreement({("i", "s1"): 0}, {})["n"] == 0


# --- de-anonymising a grader's file ------------------------------------------------------------------


def test_a_graders_file_is_resolved_through_the_key_and_junk_is_dropped(tmp_path):
    """A grader that invents an id must not silently contribute a grade to the comparison."""
    key = {"p0001": "k:adv0001:S1", "p0002": "k:adv0002:S2"}
    grades = {"grader": "x", "grades": {"p0001": {"grade": 3, "why": "w"},
                                        "p0002": {"grade": 9, "why": "out of range"},
                                        "p9999": {"grade": 1, "why": "not in the key"}}}  # fmt: skip
    key_path, grades_path = tmp_path / "key.json", tmp_path / "g.json"
    key_path.write_text(json.dumps(key), encoding="utf-8")
    grades_path.write_text(json.dumps(grades), encoding="utf-8")

    resolved, why, rejected = load_grader_file(grades_path, key_path)
    assert len(rejected) == 2, rejected
    assert resolved == {"k:adv0001:S1": 3}, "an out-of-range grade and an unknown id are both dropped"
    assert why["k:adv0001:S1"] == "w"
