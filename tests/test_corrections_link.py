"""Linking a written correction to the speech it corrects."""

from __future__ import annotations

from datetime import date

import pytest

from ingest import corrections as K


# --- recovering the two texts --------------------------------------------------------------------------------


def test_the_original_drops_inserted_words_and_keeps_struck_ones():
    body = "The figure was <s>18</s><strong>1.8</strong> billion pounds."
    assert K.original_text(body).split() == ["The", "figure", "was", "18", "billion", "pounds."]


def test_the_corrected_text_keeps_inserted_words_and_drops_struck_ones():
    body = "The figure was <s>18</s><strong>1.8</strong> billion pounds."
    assert K.corrected_text(body).split() == ["The", "figure", "was", "1.8", "billion", "pounds."]


def test_markup_is_case_insensitive_and_may_span_lines():
    body = "Words <STRONG>inserted\nacross lines</STRONG> and <S>struck</S> here."
    assert "inserted" not in K.original_text(body) and "struck" in K.original_text(body)
    assert "inserted" in K.corrected_text(body) and "struck" not in K.corrected_text(body)


def test_other_tags_are_removed_but_their_text_is_kept():
    assert K.original_text("<p>Kept <em>emphasis</em> text</p>").split() == ["Kept", "emphasis", "text"]


def test_an_empty_or_missing_body_gives_no_shingles():
    assert K.shingles(K.original_text("")) == set()
    assert K.shingles(K.original_text(None)) == set()


def test_shingles_ignore_case_and_punctuation():
    assert K.shingles("The Minister, said: yes!", n=3) == K.shingles("the minister said yes", n=3)


# --- choosing the speech -------------------------------------------------------------------------------------

ORIGINAL = ("Earlier research found that lower working class speakers were judged to be fourteen per cent less "
            "intelligent and five per cent less trustworthy than upper middle class speakers in the survey")
CORRECTION = ("<strong>Dr Amanda Cole and </strong>Earlier research found that lower working class speakers were judged "
              "to be fourteen per cent less intelligent and five per cent less trustworthy than upper middle class "
              "speakers in the survey")


def candidate(speech_id: str, text: str, day: date = date(2026, 8, 20)):
    return K.Candidate(speech_id=speech_id, date=day, text=text)


def test_the_speech_holding_the_original_wording_is_chosen():
    found = K.link_correction(CORRECTION, [candidate("other", "We must fund the NHS properly this year and next"),
                                           candidate("orig", "I will say this. " + ORIGINAL + ". Thank you.")])
    assert found.speech_id == "orig" and found.basis == "shingle"
    assert found.score >= K.MIN_SCORE


def test_two_speeches_holding_the_same_passage_are_refused():
    """A member who repeated a passage in two speeches has not said which one the correction is for."""
    found = K.link_correction(CORRECTION, [candidate("a", ORIGINAL), candidate("b", ORIGINAL)])
    assert found.speech_id is None and found.basis == "ambiguous"


def test_a_weak_best_match_is_not_linked():
    found = K.link_correction(CORRECTION, [candidate("a", "Earlier research found that some things were true")])
    assert found.speech_id is None and found.basis == "weak"


def test_no_candidates_says_so():
    assert K.link_correction(CORRECTION, []).basis == "no_candidate"


def test_a_body_with_no_words_says_so():
    assert K.link_correction("<strong></strong>", [candidate("a", ORIGINAL)]).basis == "empty_body"


def test_the_thresholds_are_the_measured_ones():
    assert K.MIN_SCORE == 0.5 and K.MIN_MARGIN == 0.2 and K.LOOKBACK_DAYS == 120


@pytest.mark.parametrize("edit", ["<s>fourteen</s><strong>four</strong>", "<strong>not </strong>", "<s>less </s>"])
def test_a_small_edit_does_not_break_the_match(edit):
    """Most corrections change a figure or a word; the rest of the passage carries the match."""
    body = ORIGINAL.replace("fourteen", edit, 1)
    assert K.link_correction(body, [candidate("orig", ORIGINAL)]).speech_id == "orig"
