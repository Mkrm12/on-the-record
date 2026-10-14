"""The gold set schema, its validation, and the scoring it defines."""

from __future__ import annotations

import json

import pytest
from goldset.schema import (
    CATEGORIES,
    CRITICALITY_POINTS,
    SCHEMA_VERSION,
    Evidence,
    GoldItem,
    Nugget,
    census,
    content_hash,
    dump,
    from_dict,
    jaccard,
    load,
    longest_common_run,
    normalise_question,
    to_dict,
    validate,
)
from goldset.scoring import Graded, ScoringConfig, score, wilson_upper


def item(**kwargs) -> GoldItem:
    base = dict(
        id="hum0001",
        question="What did the Minister say about the levy?",
        category="exact_term",
        criticality="major",
        expected_behaviour="answer",
        provenance="passage_llm",
        evidence=(Evidence("SPEECH-1", "The Government published their response to the consultation."),),
    )
    return GoldItem(**(base | kwargs))


def wrap(*items: GoldItem) -> dict:
    return {"schema_version": SCHEMA_VERSION, "items": [to_dict(i) for i in items]}


# --- the vocabularies hold together ------------------------------------------------------------------


def test_every_category_maps_to_a_known_behaviour_or_to_none():
    from goldset.schema import EXPECTED_BEHAVIOURS

    for category, behaviour in CATEGORIES.items():
        assert behaviour is None or behaviour in EXPECTED_BEHAVIOURS, category


def test_a_clean_item_validates():
    assert validate(wrap(item())) == []


def test_the_schema_version_is_checked():
    data = wrap(item())
    data["schema_version"] = 1
    assert any("schema_version" in e for e in validate(data))


# --- the rules that stop a meaningless item ----------------------------------------------------------


def test_a_category_with_a_fixed_behaviour_refuses_any_other():
    bad = item(id="adv0001", category="false_premise", expected_behaviour="answer",
               forbidden_claims=("something",))  # fmt: skip
    assert any("must expect correct_false_premise" in e for e in validate(wrap(bad)))


def test_an_abstain_item_may_not_carry_evidence():
    bad = item(id="adv0002", category="out_of_window", expected_behaviour="abstain_out_of_window")
    errors = validate(wrap(bad))
    assert any("must have no evidence" in e for e in errors)


def test_an_answering_item_needs_evidence():
    assert any("needs at least one evidence speech" in e for e in validate(wrap(item(evidence=()))))


def test_a_structured_fact_item_needs_its_facts_and_not_evidence():
    ok = item(id="db0001", category="division_result", evidence=(),
              expected_structured_facts={"ayes": 300, "noes": 200})  # fmt: skip
    assert validate(wrap(ok)) == []
    bad = item(id="db0002", category="division_result", evidence=())
    assert any("needs expected_structured_facts" in e for e in validate(wrap(bad)))


def test_an_adversarial_item_must_name_what_may_not_be_asserted():
    # Without a forbidden claim there is nothing for a system to fail, so the item tests nothing.
    bad = item(id="adv0003", category="false_premise", expected_behaviour="correct_false_premise")
    assert any("needs at least one forbidden_claim" in e for e in validate(wrap(bad)))
    good = item(id="adv0004", category="false_premise", expected_behaviour="correct_false_premise",
                forbidden_claims=("That the Government cancelled it.",))  # fmt: skip
    assert validate(wrap(good)) == []


def test_a_w3_item_may_not_be_dev_or_test():
    bad = item(window="W3", split="dev")
    assert any("must be split 'heldout'" in e for e in validate(wrap(bad)))
    assert validate(wrap(item(window="W3", split="heldout"))) == []


def test_a_verified_item_may_not_still_carry_a_placeholder():
    bad = item(status="verified", nuggets=(Nugget("TO_VERIFY: the tally", True),))
    assert any("TO_VERIFY" in e for e in validate(wrap(bad)))


def test_duplicate_ids_and_duplicate_questions_are_both_caught():
    errors = validate(wrap(item(), item()))
    assert any("duplicate id" in e for e in errors)
    assert any("duplicate question" in e for e in errors)


def test_grades_must_be_integers_from_zero_to_three():
    assert any("must be an integer 0 to 3" in e for e in validate(wrap(item(graded_relevance={"S": 4}))))
    assert validate(wrap(item(graded_relevance={"S": 3}))) == []


def test_a_short_quote_is_rejected():
    bad = item(evidence=(Evidence("S", "too short"),))
    assert any("at least" in e for e in validate(wrap(bad)))


# --- the text measures the overlap filter is built on -------------------------------------------------


def test_longest_common_run_counts_consecutive_tokens_only():
    assert longest_common_run("the cat sat on the mat", "a cat sat on a rug") == 3
    assert longest_common_run("cat rug mat", "mat rug cat") == 1
    assert longest_common_run("", "anything") == 0


def test_longest_common_run_ignores_case_and_punctuation():
    assert longest_common_run("If we TAX something, we get less!", "if we tax something we get less") == 7


def test_jaccard_is_zero_when_either_side_has_no_content_words():
    assert jaccard("the and of", "a bill about housing") == 0.0
    assert 0 < jaccard("housing policy debate", "the housing policy") <= 1


def test_normalise_question_ignores_word_order_and_punctuation():
    assert normalise_question("Who voted against the Bill?") == normalise_question("the bill: voted, who against")


# --- census and hashing -------------------------------------------------------------------------------


def test_content_hash_ignores_notes_but_not_the_question_or_the_split():
    a = item()
    assert content_hash([a]) == content_hash([item(notes="anything", reviewer="me")])
    assert content_hash([a]) != content_hash([item(question="A different question entirely?")])
    assert content_hash([a]) != content_hash([item(split="test")])


def test_retrieval_testable_excludes_structured_facts_and_abstentions():
    assert item().retrieval_testable
    assert not item(category="division_result", evidence=(),
                    expected_structured_facts={"ayes": 1}).retrieval_testable  # fmt: skip
    assert not item(category="out_of_window", expected_behaviour="abstain_out_of_window",
                    evidence=()).retrieval_testable  # fmt: skip


def test_census_counts_points_by_tier():
    items = [item(), item(id="hum0002", criticality="critical", question="A second question here?")]
    assert census(items)["points_available"] == CRITICALITY_POINTS["major"] + CRITICALITY_POINTS["critical"]


def test_round_trip_through_dict_keeps_evidence_and_nuggets():
    original = item(nuggets=(Nugget("A fact.", True, ("S1",)),), forbidden_claims=("Not this.",))
    restored = from_dict(to_dict(original))
    assert restored == original


def test_dump_refuses_to_write_an_invalid_set(tmp_path):
    with pytest.raises(ValueError, match="refusing to write"):
        dump([item(window="W3", split="dev")], tmp_path / "gold.json")
    assert not (tmp_path / "gold.json").exists()


def test_dump_then_load_round_trips(tmp_path):
    path = tmp_path / "gold.json"
    dump([item()], path, created="2026-09-24")
    assert json.loads(path.read_text(encoding="utf-8"))["created"] == "2026-09-24"
    assert load(path)[0].question == item().question


def test_load_skips_dropped_items(tmp_path):
    path = tmp_path / "gold.json"
    dump([item(), item(id="hum0002", question="Dropped question here?", status="dropped")], path)
    assert [i.id for i in load(path)] == ["hum0001"]


# --- scoring -------------------------------------------------------------------------------------------


def test_the_briefs_worked_example_reproduces_exactly():
    """50 minor items, 49 correct and 1 harmful -> (49 - 8) / 50 = 82%, and the gate FAILS."""
    graded = [Graded(f"q{i}", "minor", "correct") for i in range(49)]
    graded.append(Graded("q49", "minor", "harmful"))
    result = score(graded)
    assert result["quality"]["points_earned"] == 41.0
    assert result["quality"]["weighted"] == 0.82
    assert result["gate"]["passed"] is False


def test_a_harmful_answer_fails_the_gate_whatever_the_score():
    graded = [Graded(f"q{i}", "critical", "correct") for i in range(999)]
    graded.append(Graded("bad", "minor", "harmful"))
    result = score(graded)
    assert result["quality"]["weighted"] > 0.99
    assert result["gate"]["passed"] is False


def test_a_clean_run_passes_the_gate_and_reports_the_rule_of_three():
    result = score([Graded(f"q{i}", "major", "correct") for i in range(300)])
    assert result["gate"]["passed"] is True
    assert result["harmful"]["rule_of_three"] == pytest.approx(0.01)
    assert result["harmful"]["upper95"] < 0.0127  # 300 error-free trials support "about 1%", not zero


def test_over_abstention_scores_nothing_and_is_counted_separately():
    result = score([Graded("a", "major", "correct_abstention"), Graded("b", "major", "over_abstention")])
    assert result["quality"]["points_earned"] == 3.0
    assert result["coverage"]["over_abstention"] == 1
    assert result["coverage"]["not_attempted"] == 2


def test_severity_weighting_makes_a_critical_item_worth_five_minor_ones():
    critical = score([Graded("a", "critical", "correct")])["quality"]["points_earned"]
    minor = score([Graded("b", "minor", "correct")])["quality"]["points_earned"]
    assert critical == 5 * minor


def test_enough_harmful_answers_drive_the_score_negative():
    # It must not bottom out at zero: "very bad" and "catastrophic" have to be distinguishable.
    assert score([Graded("a", "minor", "harmful")])["quality"]["weighted"] == -8.0


def test_an_unknown_outcome_is_refused_rather_than_scored_as_zero():
    with pytest.raises(ValueError, match="unknown outcome"):
        score([Graded("a", "minor", "mostly_fine")])


def test_wilson_upper_bound_is_sane_at_the_edges():
    assert wilson_upper(0, 0) == 1.0
    assert wilson_upper(0, 100) < 0.04
    assert wilson_upper(100, 100) == 1.0
    assert wilson_upper(1, 300) > wilson_upper(0, 300)


def test_the_scoring_config_can_be_overridden():
    config = ScoringConfig(points={"minor": 10, "major": 3, "critical": 5}, harmful_penalty=-1)
    result = score([Graded("a", "minor", "correct"), Graded("b", "minor", "harmful")], config)
    assert result["quality"]["points_available"] == 20
    assert result["quality"]["points_earned"] == 9.0
