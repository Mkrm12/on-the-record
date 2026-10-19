"""A reporter's note standing alone in a member's place is judged by what it says (QA pass 8, Low 5; S06)."""

from __future__ import annotations

import pytest

from eval.oracles import openers as OO
from procedure import opener as OP

GAPS = ["<em>[Inaudible.]</em>", "[Inaudible.]", "(Inaudible.)", "［Inaudible.］", "<em>(Inaudible)</em>",
        "[Indistinct.]", "[Connection lost.]", "[No audio.]", "[Words not recorded.]", "[Technical difficulties.]"]
NOT_THE_MEMBER = ["<em>[Laughter.]</em>", "[Laughter.]", "[Hon. Members: \"Hear, hear!\"]", "[Interruption.]",
                  "(Interruption.)", "［Laughter.］", "[Proceedings suspended.]", "[An hon. Member: \"Inaudible!\"]",
                  "<em>[Applause.]</em>"]
SPEECHES = ["I beg to move. [Interruption.]", "We must act (inaudible) now.", "I thank the Minister.",
            "<em>Moved by</em> Lord Smith"]
NOTHING = ["<em>rose—</em>", "<em>rose— [Interruption.]</em>", "", "<em>indicated assent.</em>"]


@pytest.mark.parametrize("text", GAPS)
def test_a_gap_in_the_transcript_is_the_floor_with_words_unrecorded(text):
    assert OP.contribution_state(text) == OP.UNRECORDED
    assert not OP.non_speech(text), "the member had the floor: it counts as speaking"
    assert not OO.says_nothing(text), "the oracle reads it the same way"


@pytest.mark.parametrize("text", NOT_THE_MEMBER)
def test_a_note_about_anything_else_is_not_the_member_speaking(text):
    assert OP.contribution_state(text) == OP.NOTHING
    assert OP.non_speech(text) and OO.says_nothing(text)


@pytest.mark.parametrize("text", SPEECHES)
def test_words_beside_a_note_are_a_speech(text):
    assert OP.contribution_state(text) == OP.SPEECH
    assert not OP.non_speech(text) and not OO.says_nothing(text)


@pytest.mark.parametrize("text", NOTHING)
def test_stage_directions_alone_are_still_nothing(text):
    assert OP.contribution_state(text) == OP.NOTHING
    assert OP.non_speech(text) and OO.says_nothing(text)


def test_the_rule_and_the_oracle_agree_on_every_form_here():
    for text in GAPS + NOT_THE_MEMBER + SPEECHES + NOTHING:
        assert OP.non_speech(text) == OO.says_nothing(text), text


# --- markup that splits a word (QA pass 9, Med 5)
# ----------------------------------------------------------------------

# Adam Afriyie, 4 May 2020 (speech 0725BD5D-…): the reporter's "[Inaudible.]" set as five italic runs.
SPLIT_GAP = "<em>[</em><em>In</em><em>audible</em><em>.</em><em>]</em>"


@pytest.mark.parametrize("text", [SPLIT_GAP, "<em>[In</em>audible.]", "<i>[In</i><i>distinct.]</i>"])
def test_a_gap_note_split_by_markup_is_still_a_gap(text):
    assert OP.contribution_state(text) == OP.UNRECORDED
    assert OO.says_nothing(text) is False


def test_runs_that_are_two_words_still_read_as_two():
    """QA pass 7: "<em>Moved</em><em>by</em>" glued lost the words "Moved by"; tags stay word boundaries for everything
    but the gap test, which reads the notes both ways.
    """
    assert OP.contribution_state("<em>Moved</em><em>by</em> <em>Lord Smith</em>") == OP.SPEECH
    assert OP.contribution_state("<em>[</em><em>Laughter</em><em>.]</em>") == OP.NOTHING
