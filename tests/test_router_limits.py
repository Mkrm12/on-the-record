"""The router's cost on hostile input."""

from __future__ import annotations

import time

import pytest

from answer import resolve as RS
from answer import router as RO


@pytest.mark.parametrize("hostile", ["vote " * 20000, "did " * 8000 + "?", "division " * 4000,
                                     "which " * 3000 + "vote", "x" * 200_000],
                         ids=["vote", "did", "division", "which", "one-word"])  # fmt: skip
def test_hostile_input_is_routed_quickly(hostile):
    started = time.perf_counter()
    decision = RO.route(hostile)
    assert time.perf_counter() - started < 0.5
    assert decision.route == "clarify" and "shorter" in decision.clarify[0]


def test_a_question_at_the_limit_is_still_read():
    question = "Which debate was division 149 in the Commons on 8 May 2024 taken during?"
    padded = question + " " * (RO.MAX_QUESTION_CHARS - len(question) - 1) + "?"
    assert len(padded) <= RO.MAX_QUESTION_CHARS
    assert RO.route(padded).category == "division_debate"


@pytest.mark.parametrize("n", [200, 999])
def test_the_debate_shape_is_linear_inside_the_limit(n):
    text = ("division " * n)[: RO.MAX_QUESTION_CHARS]
    started = time.perf_counter()
    for _ in range(20):
        RO._asks_for_debate_behind_division(text)
        RS.vote_subject(text)
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize("question, expect", [
    ("Which debate was division 149 in the Commons on 8 May 2024 taken during?", True),
    ("What was being debated when the Commons voted on 8 May 2024?", True),
    ("What was the result of the Opposition day debate division on 8 May 2024?", False),
    ("What was debated on 8 May 2024?", False),
])  # fmt: skip
def test_the_debate_shape_still_needs_both_halves(question, expect):
    assert RO._asks_for_debate_behind_division(question) is expect


# --- QA pass 4 -----------------------------------------------------------------------------------------


def test_the_length_cap_is_on_the_raw_question_not_on_what_survives_whitespace():
    """3,600 spaces, a short question and 400,000 characters were measured on the first 4,000 characters after
    whitespace collapsed, routed, and handed whole to the resolvers (29.75 s of database work).
    """
    padded = " " * 3600 + "Which party was Keir Starmer in on 13 May 2020? " + "y" * 400_000
    decision = RO.route(padded)
    assert decision.route == "clarify" and "shorter" in decision.clarify[0]
    assert RO.route(" " * 3600 + "Which party was Keir Starmer in on 13 May 2020?").route == "structured"


@pytest.mark.parametrize("question", [
    "How many Conservative MPs voted against the Agriculture Bill at Third Reading on 13 May 2020?",
    "How many Labour MPs voted for New Clause 2 to the Agriculture Bill on 13 May 2020?",
    "How many Lib Dem peers voted for amendment 1 on 24 February 2026?",
    "How many women voted against the Agriculture Bill on 13 May 2020?",
])  # fmt: skip
def test_a_count_of_part_of_the_house_is_not_the_house_s_tally(question):
    assert RO.route(question).category != "division_tally"


@pytest.mark.parametrize("question", [
    "How many MPs voted against the Agriculture Bill at Third Reading on 13 May 2020?",
    "How many members voted for New Clause 2 on 13 May 2020?",
    "How many peers voted against amendment 1 on 24 February 2026?",
    "How many voted for the Agriculture Bill at Third Reading on 13 May 2020?",
])  # fmt: skip
def test_a_count_of_the_whole_house_is_still_its_tally(question):
    assert RO.route(question).category == "division_tally"


def test_a_division_taken_during_a_debate_is_asked_about_not_its_debate():
    vote = RO.route("How did Keir Starmer vote in the division taken during the Agriculture Bill Third Reading debate "
                    "on 13 May 2020?")  # fmt: skip
    assert vote.category == "division_vote"
    assert RO.route("What was the result of the vote taken during the Agriculture Bill debate on 13 May 2020 on New "
                    "Clause 7?").category == "division_result"  # fmt: skip
    for asking in ("Which debate was the vote on New Clause 7 of the Agriculture Bill on 13 May 2020 taken during?",
                   "During which debate was the vote on New Clause 7 of the Agriculture Bill on 13 May 2020 taken?"):
        assert RO.route(asking).category == "division_debate", asking
