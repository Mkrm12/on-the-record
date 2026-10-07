"""Turning a question's words into a structured fact's parameters, and the ambiguity rule."""

from __future__ import annotations

import pytest

from answer import resolve as RS


# --- the title scorer -----------------------------------------------------------------------------------


def test_a_title_the_question_names_scores_well():
    assert RS.title_score("was the motion on the Football Governance Bill carried?",
                          "Football Governance Bill") > 0.9  # fmt: skip


def test_a_title_the_question_does_not_name_scores_badly():
    assert RS.title_score("what did the Minister say about housing?", "Football Governance Bill") < 0.3


def test_the_score_is_normalised_on_the_title_not_the_question():
    """A long question must not out-score a short one against the same title, or verbose phrasing would win."""
    short = RS.title_score("Football Governance Bill?", "Football Governance Bill")
    long = RS.title_score(
        "I would like to know whether the motion relating to the Football Governance Bill was carried "
        "in the House of Commons on that particular day, please", "Football Governance Bill")  # fmt: skip
    assert short == pytest.approx(long)


def test_a_matching_clause_number_is_rewarded():
    """"New Clause 6" is how two divisions on one Bill on one day actually differ."""
    with_clause = RS.title_score("Health Bill: Report Stage: New Clause 6", "Health Bill: Report Stage: New Clause 6")
    without = RS.title_score("Health Bill: Report Stage", "Health Bill: Report Stage: New Clause 6")
    assert with_clause > without


def test_a_clause_number_in_the_title_but_not_the_question_is_penalised():
    """The case that matters: a question naming only the Bill must not match a specific amendment as well as it matches
    the Bill's own division.
    """
    specific = RS.title_score("was the Health Bill motion carried?", "Health Bill: Report Stage: New Clause 6")
    general = RS.title_score("was the Health Bill motion carried?", "Health Bill")
    assert general > specific


def test_procedural_noise_does_not_carry_signal():
    """"motion", "division", "carried", "the Lords" appear in almost every question of this kind."""
    assert RS.title_score("was the motion carried in the Lords?", "motion division carried lords") == 0.0


def test_an_empty_title_scores_zero_rather_than_raising():
    assert RS.title_score("anything", None) == 0.0
    assert RS.title_score("anything", "") == 0.0


# --- extractors -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(("question", "expected"), [
    ("what was the result of division 421?", 421),
    ("division no. 75 in the Commons", 75),
    ("division number 3", 3),
    ("which member spoke in the third division", 3),
    ("the first division of the day", 1),
    ("was the motion carried?", None),
])
def test_a_division_number_is_read_when_the_question_gives_one(question, expected):
    assert RS.division_number(question) == expected


@pytest.mark.parametrize(("question", "expected"), [
    ("when was the second reading?", "Second Reading"),
    ("the date of the Order of Commitment discharged", "Order Of Commitment Discharged"),
    ("what happened at report stage", "Report Stage"),
    ("when did it get royal assent", "Royal Assent"),
    ("what did the Minister say", None),
])
def test_the_offline_stage_regex_still_reads_a_named_stage(question, expected):
    """`stage_name_offline` is the fallback for a caller with no connection, and these are its outputs."""
    assert RS.stage_name_offline(question) == expected


@pytest.mark.parametrize("question", [
    "when was the second reading of the Bill?",
    "when was the 2nd reading of the Bill?",
])
def test_the_ordinal_a_person_writes_and_the_one_parliament_records_agree(question):
    """The commonest form of the mismatch, not an edge case: the three readings are the three most-asked stages."""
    assert "2nd" in RS._stage_terms(question)


def test_stage_terms_keep_the_house_because_two_stages_differ_only_by_it():
    """`_NOISE` drops "commons" and "lords" — right for a division title, fatal for a stage name. "Commons Examiners"
    and "Lords Examiners" both collapsed to {examiners} and tied, so a question naming one exactly was refused as
    ambiguous.
    """
    assert "commons" in RS._stage_terms("what was the date of the Commons Examiners of the Bill?")
    assert RS._stage_terms("Commons Examiners") != RS._stage_terms("Lords Examiners")


def test_stage_question_filler_is_dropped():
    """"what was the date of" appears in nearly every one of these questions and carries no signal."""
    terms = RS._stage_terms("What was the date of the Royal Assent of the Bill?")
    assert "royal" in terms and "assent" in terms
    assert "date" not in terms and "bill" not in terms


# --- the thresholds ------------------------------------------------------------------------------------


def test_the_margin_and_floor_are_the_measured_ones():
    """A resolution must clear a floor on its own AND beat the runner-up. Either alone is not enough: a high score with
    a tie means the question did not say which, and a clear win below the floor means it named none of them.
    """
    assert RS.MIN_SCORE == 0.30
    assert RS.MIN_MARGIN == 0.15


def test_the_noise_list_covers_the_words_every_such_question_uses():
    for word in ("motion", "division", "carried", "lords", "commons", "was", "the"):
        assert word in RS._NOISE
