"""The answer contract."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from answer import contract as C

PROV = dict(
    route="lookup",
    resolved_filters={"house": "Commons"},
    corpus_version="v2",
    retrieval_config_hash="e57c6cdfd8793d76",
    recipe="bge-small-header-v1",
    prompt_version="p1",
    model_id="deepseek-chat",
)


def citation(quote="the House will rise at six o'clock", speech_id="s1", chunk_ids=(10,)):
    return C.Citation(speech_id=speech_id, chunk_ids=tuple(chunk_ids), quote=quote)


def passing() -> C.Verification:
    return C.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True,
                          window_ok=True, checked_by="test")  # fmt: skip


def claim(text="the House will rise at six o'clock", **kw):
    kw.setdefault("citations", (citation(),))
    kw.setdefault("verification", passing())
    return C.Claim(id=kw.pop("id", "c1"), kind=kw.pop("kind", "said"), text=text, **kw)


def answer(**kw):
    kw.setdefault("question", "when does the House rise?")
    kw.setdefault("provenance", C.Provenance(**PROV))
    return C.Answer(**kw)


# --- normalisation: the one definition of "the same text" ------------------------------------------------


@pytest.mark.parametrize(("hansard", "copied"), [
    ("the Minister’s statement", "the Minister's statement"),  # curly apostrophe
    ("“my honourable friend”", '"my honourable friend"'),  # curly double quotes
    ("a long—and costly—delay", "a long-and costly-delay"),  # em dash
    ("2024–25 spending", "2024-25 spending"),  # en dash in a year range
    ("a non-breaking space", "a non-breaking space"),  # NBSP
    ("co­operation", "cooperation"),  # soft hyphen
    ("THE HOUSE ROSE", "the house rose"),  # case
    ("double  spaced   text", "double spaced text"),  # collapsed whitespace
])
def test_typographic_variants_are_the_same_quote(hansard, copied):
    """Hansard's typography must not be able to fail a quote that is word-for-word correct."""
    assert C.normalise(hansard) == C.normalise(copied)
    assert C.quote_matches(copied, f"He said {hansard} to the House.")


def test_a_changed_word_is_a_different_quote():
    """The flip side, and the one that matters: normalisation folds punctuation, never meaning."""
    assert not C.quote_matches("the House will rise at seven", "the House will rise at six o'clock")


def test_normalisation_does_not_fold_a_negation():
    assert C.normalise("did not support") != C.normalise("did support")


@pytest.mark.parametrize("ellipsis", ["…", "..."])
def test_an_ellipsis_quote_matches_its_segments_in_order(ellipsis):
    source = "I beg to move that the Bill be now read a second time, and I thank my honourable friend."
    assert C.quote_matches(f"I beg to move{ellipsis}read a second time", source)


def test_an_ellipsis_quote_cannot_reorder_the_source():
    """A quote that reverses two stretches of a speech changes its meaning while keeping every word."""
    source = "we rejected the amendment because the Minister refused the funding"
    assert not C.quote_matches("the Minister refused the funding…we rejected the amendment", source)


def test_an_empty_source_never_matches():
    assert not C.quote_matches("anything", "")


# --- numbers --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [
    ("£35 billion", "35000000000"),
    ("£35bn", "£35 billion"),
    ("1,250 members", "1250 members"),
    ("2.5 million", "2500000"),
    ("twelve members", "12 members"),
    ("a dozen", "12"),
])
def test_equivalent_magnitudes_normalise_together(a, b):
    assert C.numbers_in(a) & C.numbers_in(b), f"{a!r} and {b!r} should share a normalised number"


def test_a_percentage_is_not_the_bare_number():
    """40% and 40 are different claims, so a quote saying one must not license the other."""
    assert C.missing_numbers("it rose 40%", "it rose by 40 places")


@pytest.mark.parametrize("phrasing", ["40%", "40 per cent", "40 percent"])
def test_percentage_spellings_agree(phrasing):
    assert "40%" in C.numbers_in(phrasing)


def test_a_year_is_tagged_as_a_year_not_a_magnitude():
    """So that a quote about the year 2024 cannot support a claim about £2,024."""
    assert C.numbers_in("in 2024") == {"year:2024"}
    assert C.missing_numbers("the Bill cost £2,024", "it happened in 2024") == {"2024"}


def test_a_bare_four_digit_count_is_indistinguishable_from_a_year():
    """The documented limit of this check, asserted rather than left to be discovered."""
    assert C.missing_numbers("2024 members voted", "it happened in 2024") == set()


def test_a_large_integer_keeps_its_exact_value():
    """%g switches to exponent form past ten significant digits, which collapsed two different eleven-digit figures
    onto one token and made them compare equal.
    """
    assert C.numbers_in("12345678901") == {"12345678901"}
    assert C.numbers_in("12345678901") != C.numbers_in("12345678902")


def test_a_claim_whose_numbers_are_all_in_the_quote_has_nothing_missing():
    assert C.missing_numbers("the vote was 316 to 295", "the Ayes were 316, the Noes 295") == set()


def test_a_scale_suffix_does_not_eat_the_first_letter_of_a_word():
    """The bug this pins: the optional `m` suffix matched the M of "members" and read "12 members" as twelve million,
    so a claim quoting "12 members" looked as though it had invented 12,000,000.
    """
    assert C.numbers_in("12 members") == {"12"}
    assert C.numbers_in("12 thousand members") == {"12000"}
    assert C.missing_numbers("12 members voted", "12 members voted") == set()


def test_a_scaled_magnitude_counts_once_not_twice():
    """The second bug: "£35 billion" matched as 35e9 AND "billion" matched separately as 1e9, so the claim carried a
    figure no quote could ever contain.
    """
    assert C.numbers_in("£35 billion") == {"35000000000"}
    assert C.missing_numbers("the Bill cost £35 billion", "the Bill cost £35bn") == set()


@pytest.mark.parametrize(("phrase", "expected"), [
    ("one hundred and fifty", "150"),
    ("twenty-five", "25"),
    ("two thousand", "2000"),
    ("fifteen", "15"),
])
def test_a_written_compound_is_one_number_not_its_parts(phrase, expected):
    """"one hundred and fifty" must not emit 1, 100 and 50: three figures the evidence would appear to lack."""
    assert C.numbers_in(phrase) == {expected}


def test_a_word_boundary_keeps_sixteen_out_of_six():
    assert C.numbers_in("sixteen") == {"16"}


@pytest.mark.parametrize("phrase", ["one of my constituents", "on the one hand", "two or three occasions"])
def test_small_written_numbers_are_not_treated_as_figures(phrase):
    """In Hansard prose these are not quantities at all. Emitting them as figures made the check refuse sound claims —
    and construction refusal is a hard failure, not a warning.
    """
    assert C.numbers_in(phrase) == set()
    assert C.WORD_NUMBER_FLOOR == 10


def test_a_currency_amount_is_never_read_as_a_year():
    assert C.numbers_in("£2024") == {"2024"}
    assert C.numbers_in("in 2024") == {"year:2024"}


# --- claims: atomic, cited, numerate ---------------------------------------------------------------------


def test_a_two_sentence_claim_cannot_be_built():
    """A claim is the unit the verifier passes or fails. Two sentences can be half true, and half true is what gets
    shown.
    """
    with pytest.raises(ValidationError, match="ONE sentence"):
        claim("The House rose at six. The Minister was absent.")


def test_a_single_sentence_ending_in_a_full_stop_is_fine():
    assert claim("the House rose at six o'clock.").text.endswith(".")


@pytest.mark.parametrize("text", [
    "the figure was unchanged since No. 10 published it",
    "the Hon. Members opposite voted against it",
    "the Rt. Hon. Gentleman gave way",
    "the report by A. B. Smith was laid before the House",
    "the Minister, i.e. The Secretary of State, replied",
])
def test_an_abbreviation_does_not_read_as_two_sentences(text):
    """Hansard writes "No. 10", "Hon. Members" and "Rt."""
    assert C.is_one_sentence(text), f"{text!r} is one sentence"


@pytest.mark.parametrize("text", [
    "The House rose at six. The Minister was absent.",
    "He moved the amendment! The House divided.",
    "Did the Minister agree? The record says nothing.",
    'He said "no". The House disagreed.',
])
def test_genuinely_separate_sentences_are_still_caught(text):
    """The other half of the abbreviation fix: loosening the rule must not blind it."""
    assert not C.is_one_sentence(text)


def test_a_claim_with_no_citation_cannot_be_built():
    with pytest.raises(ValidationError):
        C.Claim(id="c1", kind="said", text="the House rose", citations=())


def test_a_claim_cannot_state_a_number_absent_from_its_own_quote():
    """The single most likely fabrication: a correct-sounding figure attached to a real quote."""
    with pytest.raises(ValidationError, match="number"):
        claim("the Bill cost £35 billion", citations=(citation(quote="the Bill was expensive"),))


def test_a_claim_may_state_a_number_written_differently_in_the_quote():
    """£35bn in the record and £35 billion in the claim is the same figure, and refusing it would push the generator
    towards quoting digits it cannot see.
    """
    assert claim("the Bill cost £35 billion", citations=(citation(quote="the Bill cost £35bn"),))


def test_duplicate_chunk_ids_in_one_citation_are_refused():
    with pytest.raises(ValidationError, match="distinct"):
        C.Citation(speech_id="s1", chunk_ids=(10, 10), quote="q")


def test_a_citation_is_frozen():
    """Evidence that can be edited after verification is not evidence."""
    with pytest.raises(ValidationError):
        citation().quote = "something else"


def test_an_unknown_field_is_refused_rather_than_ignored():
    """extra='forbid' everywhere: a typo'd field name is a silently missing check."""
    with pytest.raises(ValidationError):
        C.Claim(id="c1", kind="said", text="t", citations=(citation(),), speaker="Nobody")


# --- verification: a check that did not run is not a pass ------------------------------------------------


def test_an_empty_verification_does_not_pass():
    """The failure mode this exists for: the checker fell over, nothing said no, and the claim sailed through. None is
    not True.
    """
    assert C.Verification().deterministic_passed is False
    assert C.Verification().showable is False


def test_one_failed_check_fails_the_conjunction():
    v = passing()
    v.quote_ok = False
    assert v.deterministic_passed is False


def test_a_partly_run_verification_passes_only_on_what_ran():
    """Graceful degradation: a verbatim `said` claim may pass on deterministic checks alone when no entailment model is
    available.
    """
    v = C.Verification(quote_ok=True, citation_ok=True, checked_by="deterministic-only")
    assert v.deterministic_passed is True and v.showable is True


def test_an_unsupported_entailment_verdict_blocks_a_claim_that_passed_the_rest():
    v = passing()
    v.entailment_verdict = "unsupported"
    assert v.deterministic_passed is True
    assert v.showable is False, "entailment is a veto, not a vote"


def test_an_unavailable_entailment_verdict_does_not_block():
    v = passing()
    v.entailment_verdict = "unavailable"
    assert v.showable is True


def test_an_entailment_score_outside_zero_to_one_is_refused():
    with pytest.raises(ValidationError):
        C.Verification(entailment_score=1.5)


# --- answers: answer or abstain, never both, never neither -----------------------------------------------


def test_an_empty_answer_must_say_why():
    with pytest.raises(ValidationError, match="abstained"):
        answer()


def test_an_abstention_cannot_also_assert_a_claim():
    """'I could not find it, and here is what it says' is the shape of a hedge the user cannot act on."""
    with pytest.raises(ValidationError, match="must not also assert"):
        answer(abstention=C.Abstention.NOT_IN_RECORD, claims=(claim(),), retrieved_chunk_ids=frozenset({10}))


def test_each_abstention_reason_is_a_closed_set_member():
    assert {a.value for a in C.Abstention} == {
        "not_in_record", "outside_window", "ambiguous", "false_premise", "insufficient_evidence",
    }


def test_an_invented_abstention_reason_is_refused():
    with pytest.raises(ValidationError):
        answer(abstention="could_not_find_it")


def test_an_abstention_alone_is_a_valid_answer():
    a = answer(abstention=C.Abstention.OUTSIDE_WINDOW, abstention_detail="asked about 2019; corpus starts 2024")
    assert a.abstention is C.Abstention.OUTSIDE_WINDOW


def test_a_citation_outside_the_retrieved_set_cannot_be_built():
    """'s citation check, at construction. A cited chunk that was never retrieved is a fabricated citation — the model
    invented an id that happens to exist in the corpus.
    """
    with pytest.raises(ValidationError, match="never retrieved"):
        answer(claims=(claim(citations=(citation(chunk_ids=(999,)),)),), retrieved_chunk_ids=frozenset({10, 11}))


def test_a_citation_inside_the_retrieved_set_is_accepted():
    assert answer(claims=(claim(),), retrieved_chunk_ids=frozenset({10, 11})).claims


def test_rendered_text_cannot_be_set_while_a_claim_is_unverified():
    """Truth rule 3, as a property of the object rather than of the renderer, so a second renderer written later cannot
    bypass it.
    """
    with pytest.raises(ValidationError, match="not passed verification"):
        answer(claims=(claim(verification=C.Verification()),), rendered="The House rose at six.",
               retrieved_chunk_ids=frozenset({10}))  # fmt: skip


def test_showable_claims_excludes_the_unverified():
    a = answer(claims=(claim(id="c1"), claim(id="c2", verification=C.Verification())),
               retrieved_chunk_ids=frozenset({10}))  # fmt: skip
    assert [c.id for c in a.showable_claims] == ["c1"]


def test_render_is_built_only_from_verified_claims():
    a = answer(claims=(claim(id="c1", text="the House rose at six o'clock", speaker_name="Ms Speaker",
                             date=date(2026, 5, 12)),
                       claim(id="c2", text="the Minister was absent", verification=C.Verification())),
               retrieved_chunk_ids=frozenset({10}))  # fmt: skip
    rendered = a.render()
    assert "Ms Speaker" in rendered and "2026-05-12" in rendered
    assert "Minister was absent" not in rendered


def test_render_never_drops_a_structured_fact_caveat():
    a = answer(structured_facts=(C.StructuredFact(
        template="division_tally", value={"ayes": 77, "noes": 317},
        caveat="The published name lists do not reconcile (I-8)."),))  # fmt: skip
    assert "do not reconcile" in a.render()


def test_a_structured_fact_cannot_claim_a_model_as_its_source():
    """Truth rule 1: structured facts come from SQL. The literal makes an LLM-sourced fact unconstructible."""
    with pytest.raises(ValidationError):
        C.StructuredFact(template="division_tally", source="llm")


# --- provenance -----------------------------------------------------------------------------------------


def test_provenance_is_mandatory():
    with pytest.raises(ValidationError):
        C.Answer(question="q", abstention=C.Abstention.NOT_IN_RECORD)


@pytest.mark.parametrize("field", ["corpus_version", "retrieval_config_hash", "recipe", "prompt_version",
                                  "model_id"])  # fmt: skip
def test_no_provenance_field_may_be_blank(field):
    """A cached answer with a blank config hash is indistinguishable from one taken before the freeze, which is the
    entire reason the hash exists.
    """
    with pytest.raises(ValidationError):
        C.Provenance(**{**PROV, field: ""})


def test_the_frozen_hash_round_trips():
    assert C.Provenance(**PROV).retrieval_config_hash == "e57c6cdfd8793d76"
