"""The verifier: five deterministic checks, then entailment, then what to do without it."""

from __future__ import annotations

from datetime import date

import pytest

from answer import verify as V
from answer.contract import Answer, Citation, Claim, Provenance
from ingest.runtime import run as run_async


# `verify_claims` and `verified_answer` are coroutines: entailment is an HTTP call to the GPU service, so the scoring
# callable may be async.
def verify_claims(*args, **kwargs):
    return run_async(V.verify_claims(*args, **kwargs))


def verified_answer(*args, **kwargs):
    return run_async(V.verified_answer(*args, **kwargs))

SPEECH = (
    "I beg to move that the Bill be now read a second time. "
    "The funding will rise to £35 billion next year. "
    "I thank my honourable friend for her work on this."
)

PROV = Provenance(route="lookup", corpus_version="v2-w0w5", retrieval_config_hash="e57c6cdfd8793d76",
                  recipe="bge-small-header-v1", prompt_version="p1", model_id="deepseek-chat")  # fmt: skip


def evidence(**over):
    base = dict(speech_id="s1", text=SPEECH, speaker_name="Ms A Member", mnis_id=1001,
                date=date(2026, 5, 12), house="Commons", chunk_ids=frozenset({10, 11, 12}))  # fmt: skip
    base.update(over)
    return {base["speech_id"]: V.Evidence(**base)}


def claim(text="The funding will rise to £35 billion next year.", quote=None, **over):
    fields = dict(
        id="c1", kind="said", text=text,
        speaker_mnis_id=1001, speaker_name="Ms A Member",
        date=date(2026, 5, 12), house="Commons",
        citations=(Citation(speech_id="s1", chunk_ids=(10,), quote=quote or text),),
    )  # fmt: skip
    fields.update(over)
    return Claim(**fields)


def bad_claim(**over):
    """A claim the contract would refuse to build, assembled with `model_construct` to skip validation."""
    fields = dict(
        id="c1", kind="said", text="The funding will rise to £35 billion next year.",
        speaker_mnis_id=1001, speaker_name="Ms A Member",
        date=date(2026, 5, 12), house="Commons",
        citations=(Citation(speech_id="s1", chunk_ids=(10,), quote="The funding will rise to"),),
    )  # fmt: skip
    fields.update(over)
    return Claim.model_construct(**fields)


def cite(quote, speech_id="s1", chunk_ids=(10,)):
    return (Citation(speech_id=speech_id, chunk_ids=chunk_ids, quote=quote),)


def unverifiable(**over):
    """A claim the contract accepts and the verifier rejects: a paraphrase whose quote is not in the speech."""
    fields = dict(
        id="bad", kind="summary", text="The Minister thanked the department for its report.",
        speaker_mnis_id=1001, speaker_name="Ms A Member", date=date(2026, 5, 12), house="Commons",
        citations=cite("The Minister thanked the department for its report."),
    )  # fmt: skip
    fields.update(over)
    return Claim(**fields)


def stub_scores(*values):
    """An entailment model that returns fixed scores, and asserts it was handed matching batches."""
    def score_batch(evidence_texts, claim_texts):
        assert len(evidence_texts) == len(claim_texts)
        return list(values)[:len(claim_texts)]
    return score_batch


# --- citation membership --------------------------------------------------------------------------------


def test_a_cited_chunk_that_was_never_retrieved_fails():
    """A plausible integer that happens to exist in a 1.4M-chunk corpus and points at something nobody read."""
    ok, why = V.check_citations(claim(), retrieved=frozenset({11, 12}))
    assert ok is False
    assert why["cited_but_not_retrieved"] == [10]


def test_a_cited_chunk_inside_the_retrieved_set_passes():
    assert V.check_citations(claim(), retrieved=frozenset({10, 11}))[0] is True


def test_nothing_retrieved_cannot_confirm_a_citation():
    """Not "vacuously true". An empty retrieved set means we cannot check, and cannot check is not a pass."""
    ok, why = V.check_citations(claim(), retrieved=frozenset())
    assert ok is False and "nothing was retrieved" in why["why"]


# --- quotes ---------------------------------------------------------------------------------------------


def test_a_quote_present_in_the_speech_passes_through_hansard_typography():
    """The record prints a curly apostrophe; a copied quote has a straight one. Same quote."""
    source = "I thank my honourable friend’s department for the work."
    c = claim(text="The Minister thanked the department.", kind="summary",
              quote="I thank my honourable friend's department")  # fmt: skip
    assert V.check_quotes(c, evidence(text=source))[0] is True


def test_a_quote_that_is_not_in_the_speech_fails():
    ok, why = V.check_quotes(bad_claim(citations=cite("The funding will be cut in half.")), evidence())
    assert ok is False
    assert why["quotes_not_in_source"][0]["speech_id"] == "s1"


def test_a_paraphrase_is_not_a_quote():
    """The check is verbatim or nothing. A near-miss quote is the most convincing kind of fabrication."""
    ok, _ = V.check_quotes(bad_claim(citations=cite("Funding rises to £35 billion next year")), evidence())
    assert ok is False, "'The funding will rise to' is not 'Funding rises to'"


def test_missing_evidence_raises_rather_than_failing_the_claim():
    """A claim whose evidence was not supplied is a caller bug. Marking it unsupported would hide that bug behind a
    plausible verdict about the claim.
    """
    with pytest.raises(V.EvidenceMissing, match="no evidence"):
        V.check_quotes(claim(), evidence(speech_id="a-different-speech"))


# --- numbers --------------------------------------------------------------------------------------------


def test_a_figure_absent_from_the_evidence_fails():
    ok, why = V.check_numbers(bad_claim(text="The funding will rise to £50 billion next year."), evidence())
    assert ok is False
    assert why["numbers_not_in_evidence"] == ["50000000000"]


def test_a_figure_written_differently_from_the_record_still_passes():
    """£35bn in the record and £35 billion in the claim is one figure; refusing it would push the generator towards
    quoting digits it cannot see.
    """
    c = bad_claim(text="Funding rises to £35bn.", kind="summary")
    assert V.check_numbers(c, evidence())[0] is True


def test_numbers_may_come_from_the_sentence_beside_the_quote():
    """Checked against the whole cited speech, not only the quoted fragment: a summary may legitimately draw a figure
    from the next sentence, and the quote's presence in that speech is already established.
    """
    c = bad_claim(text="The Bill's funding reaches £35 billion.", kind="summary",
                  citations=cite("I beg to move"))  # fmt: skip
    assert V.check_numbers(c, evidence())[0] is True


# --- attribution ----------------------------------------------------------------------------------------


def test_attribution_is_never_delegated_to_entailment():
    """The mnis id is a fact in SQL. A model asked "did the Secretary of State say this", shown a speech by a different
    member of the same party, will agree — and will be wrong in the way that matters most.
    """
    ok, why = V.check_attribution(claim(speaker_mnis_id=2002, speaker_name="Mr B Member"), evidence())
    assert ok is False
    assert why["misattributed"][0] == {"speech_id": "s1", "claimed": 2002, "record": 1001}


def test_a_matching_mnis_id_passes_even_when_the_names_are_written_differently():
    """Names in Hansard vary with office and courtesy title; the id does not."""
    assert V.check_attribution(claim(speaker_name="The Minister for Housing"), evidence())[0] is True


def test_a_name_match_is_recorded_as_a_name_match_when_no_id_is_available():
    """An unmatched id is a harder fact than a matching name, and the two must not be reported as equivalent."""
    ok, why = V.check_attribution(claim(speaker_mnis_id=None, speaker_name="Ms A Member"),
                                  evidence(mnis_id=None))  # fmt: skip
    assert ok is True and why == {}
    ok, why = V.check_attribution(claim(speaker_mnis_id=None, speaker_name="Mr Someone Else"),
                                  evidence(mnis_id=None))  # fmt: skip
    assert ok is False and why["misattributed"][0]["matched_on"] == "name"


def test_a_claim_that_attributes_nothing_has_nothing_to_misattribute():
    ok, why = V.check_attribution(claim(speaker_mnis_id=None, speaker_name=None), evidence())
    assert ok is True and "nothing to misattribute" in why["note"]


# --- window and scope -----------------------------------------------------------------------------------


def test_a_date_disagreeing_with_the_record_fails():
    ok, why = V.check_window(claim(date=date(2026, 5, 13)), evidence(), None)
    assert ok is False
    assert why["date_disagrees_with_record"][0]["record"] == "2026-05-12"


def test_a_house_disagreeing_with_the_record_fails():
    ok, why = V.check_window(claim(house="Lords"), evidence(), None)
    assert ok is False and why["house_disagrees_with_record"]


def test_evidence_outside_the_answers_scope_fails_even_when_it_matches_the_record():
    """A claim can agree with the record perfectly and still be out of scope — that is what a window is."""
    scope = V.Scope(first_date=date(2026, 6, 1), last_date=date(2026, 7, 31))
    ok, why = V.check_window(claim(), evidence(), scope)
    assert ok is False and why["outside_the_answer_scope"]


def test_a_house_outside_the_scope_fails():
    scope = V.Scope(houses=frozenset({"Lords"}))
    assert V.check_window(claim(), evidence(), scope)[0] is False


def test_a_claim_inside_the_scope_passes():
    scope = V.Scope(first_date=date(2026, 1, 1), last_date=date(2026, 12, 31),
                    houses=frozenset({"Commons", "Lords"}))  # fmt: skip
    assert V.check_window(claim(), evidence(), scope)[0] is True


# --- all five together ----------------------------------------------------------------------------------


def test_all_five_checks_run_even_after_one_fails():
    """Knowing a claim fails three checks is worth more than knowing it failed the first."""
    c = bad_claim(text="The funding will rise to £50 billion next year.",
                  citations=cite("not in the speech at all"), speaker_mnis_id=2002,
                  date=date(2026, 5, 13))  # fmt: skip
    v = V.check_deterministically(c, evidence(), retrieved=frozenset({99}))
    assert [v.citation_ok, v.quote_ok, v.numbers_ok, v.attribution_ok, v.window_ok] == [False] * 5
    assert v.deterministic_passed is False


def test_a_sound_claim_passes_all_five():
    v = V.check_deterministically(claim(), evidence(), retrieved=frozenset({10, 11}))
    assert v.deterministic_passed is True
    assert v.checked_by == "deterministic"


# --- entailment -----------------------------------------------------------------------------------------


def test_a_paraphrase_the_evidence_does_not_support_is_dropped():
    c = claim(text="The Minister opposed the Bill.", kind="summary", quote="I beg to move")
    outcome = verify_claims([c], evidence(), frozenset({10}), score_batch=stub_scores(0.02))
    assert outcome.verified == ()
    assert "entailment 0.0200" in outcome.dropped[0][1]


def test_a_supported_paraphrase_is_kept_with_its_score():
    c = claim(text="The Minister moved the Bill.", kind="summary", quote="I beg to move")
    outcome = verify_claims([c], evidence(), frozenset({10}), score_batch=stub_scores(0.97))
    assert len(outcome.verified) == 1
    assert outcome.verified[0].verification.entailment_score == pytest.approx(0.97)
    assert outcome.verified[0].verification.entailment_verdict == "supported"


def test_entailment_is_a_veto_not_a_vote():
    """A claim that passes all five deterministic checks is still dropped if the evidence does not entail it."""
    c = claim(text="The Minister moved the Bill.", kind="summary", quote="I beg to move")
    outcome = verify_claims([c], evidence(), frozenset({10}), score_batch=stub_scores(0.1))
    assert outcome.verified == () and len(outcome.dropped) == 1


def test_entailment_is_not_asked_about_a_claim_that_already_failed():
    """Cheap and certain first: a quote that is not in the record does not need a 770M-parameter model."""
    asked: list[int] = []

    def score_batch(evidence_texts, claim_texts):
        asked.append(len(claim_texts))
        return [0.99] * len(claim_texts)

    bad = bad_claim(id="bad", citations=cite("not in the speech"))
    good = claim(id="good")
    outcome = verify_claims([bad, good], evidence(), frozenset({10}), score_batch=score_batch)
    assert asked == [1], "only the surviving claim reached the model"
    assert [c.id for c in outcome.verified] == ["good"]


def test_a_score_batch_of_the_wrong_length_raises_rather_than_mismatching_claims():
    """Off-by-one here attaches each score to the wrong claim, and every verdict afterwards is about the wrong sentence
    while looking entirely normal.
    """
    with pytest.raises(V.EvidenceMissing, match="mismatched batch"):
        verify_claims([claim(id="a"), claim(id="b")], evidence(), frozenset({10}),
                        score_batch=lambda e, c: [0.9])  # fmt: skip


# --- degradation ----------------------------------------------------------------------------------------


def test_a_said_claim_survives_without_a_model():
    """A verbatim quote is its own evidence: a substring comparison has checked it as thoroughly as a substring
    comparison can check anything. allows exactly this.
    """
    outcome = verify_claims([claim()], evidence(), frozenset({10}), score_batch=None)
    assert len(outcome.verified) == 1
    assert outcome.degraded is True
    assert outcome.verified[0].verification.entailment_verdict == "unavailable"
    assert "deterministic only" in outcome.verifier


def test_a_summary_claim_is_dropped_when_no_model_is_available():
    """The degradation must fall the safe way. The temptation is to keep the paraphrase because it reads right, which
    is the failure this module exists to prevent.
    """
    c = claim(text="The Minister supported the Bill.", kind="summary", quote="I beg to move")
    outcome = verify_claims([c], evidence(), frozenset({10}), score_batch=None)
    assert outcome.verified == ()
    assert "need entailment" in outcome.dropped[0][1]


def test_only_said_claims_are_degradable():
    assert V.DEGRADABLE_KINDS == frozenset({"said"})


def test_a_model_that_throws_degrades_rather_than_passing_everything():
    """The dangerous failure: an exception swallowed into "no verdict", and no verdict read as no objection."""
    def explode(evidence_texts, claim_texts):
        raise RuntimeError("GPU service unreachable")

    said, summary = claim(id="said"), claim(id="sum", kind="summary", text="The Minister agreed.",
                                           quote="I beg to move")  # fmt: skip
    outcome = verify_claims([said, summary], evidence(), frozenset({10}), score_batch=explode)
    assert [c.id for c in outcome.verified] == ["said"]
    assert [c.id for c, _ in outcome.dropped] == ["sum"]
    assert outcome.degraded is True


# --- the whole answer -----------------------------------------------------------------------------------


def answer(claims, **over):
    fields = dict(question="what did the Minister say?", provenance=PROV, claims=tuple(claims),
                  retrieved_chunk_ids=frozenset({10, 11, 12}))  # fmt: skip
    fields.update(over)
    return Answer(**fields)


def test_a_verified_answer_renders_only_what_passed():
    good, bad = claim(id="good"), unverifiable()
    result, outcome = verified_answer(answer([good, bad]), evidence(),
                                        score_batch=stub_scores(0.99), verifier_name="minicheck")  # fmt: skip
    assert [c.id for c in result.claims] == ["good"]
    assert result.rendered and "Ms A Member" in result.rendered
    assert outcome.counts == {"claims": 2, "verified": 1, "dropped": 1}


def test_an_answer_whose_every_claim_failed_becomes_an_abstention():
    """Not an answer with no claims, which renders as a confident empty statement. And the reason is
    `insufficient_evidence`, not `not_in_record`: the record may hold the answer; what failed was our ability to
    show it.
    """
    result, outcome = verified_answer(answer([unverifiable()]), evidence(),
                                        score_batch=stub_scores(0.99))  # fmt: skip
    assert result.abstention is not None and result.abstention.value == "insufficient_evidence"
    assert result.claims == () and result.rendered == ""
    assert "did not pass verification" in result.abstention_detail


def test_degradation_is_disclosed_in_the_coverage_note():
    """A reader is entitled to know the answer was checked without the entailment model."""
    result, outcome = verified_answer(answer([claim()]), evidence(), score_batch=None)
    assert outcome.degraded is True
    assert "without the entailment model" in result.coverage_note
    assert "paraphrased claims were dropped" in result.coverage_note


def test_rendered_text_is_never_set_while_a_claim_is_unverified():
    """The contract refuses to build such an object at all, so this asserts the verifier cannot produce one."""
    result, _ = verified_answer(answer([claim()]), evidence(), score_batch=stub_scores(0.99))
    assert all(c.verification.showable for c in result.claims)
    assert result.rendered


def test_the_threshold_matches_the_gpu_services_own():
    """Two thresholds that drift apart mean a claim accepted on the GPU and dropped in the caller."""
    assert V.SUPPORT_THRESHOLD == 0.5


# --- attribution is never entailment's job ---------------------------------------------------------------


def test_the_attribution_is_stripped_before_entailment():
    """: "SQL-metadata attribution never NLI". `check_attribution` owns who said it, from `speech.mnis_id`."""
    assert V.strip_attribution("Liam Byrne said that funding will rise.", "Liam Byrne") == "Funding will rise."
    assert V.strip_attribution("Liam Byrne argued the Bill was rushed.", "Liam Byrne") == "The Bill was rushed."
    assert V.strip_attribution("Liam Byrne said: the House rose.", "Liam Byrne") == "The House rose."


def test_a_said_in_the_middle_of_a_claim_is_part_of_the_claim():
    """Only a *leading* attribution goes. "said" elsewhere is what is being claimed."""
    text = "Liam Byrne said the Minister said nothing about funding."
    assert V.strip_attribution(text, "Liam Byrne") == "The Minister said nothing about funding."
    unchanged = "The Minister said nothing about funding."
    assert V.strip_attribution(unchanged, "Liam Byrne") == unchanged


def test_a_claim_naming_someone_else_is_left_alone():
    """Stripping on any name would remove a clause the claim actually asserts."""
    text = "Rachel Reeves said the funding will rise."
    assert V.strip_attribution(text, "Liam Byrne") == text


def test_a_surname_match_works_where_hansard_prints_a_role():
    """Hansard prints "Mr Speaker" where the roster says "Sir Lindsay Hoyle"."""
    assert V.strip_attribution("Hoyle said the House would rise.", "Sir Lindsay Hoyle") == "The House would rise."


def test_a_short_first_name_never_triggers_a_strip():
    """"May" is a surname and the first word of "May I begin"; matching on it would mutilate the claim."""
    text = "May I begin by apologising to passengers."
    assert V.strip_attribution(text, "Theresa May") == text


def test_stripping_never_empties_a_claim():
    """A claim that is nothing but an attribution keeps its text, or entailment would be handed an empty string."""
    assert V.strip_attribution("Liam Byrne said that.", "Liam Byrne") == "Liam Byrne said that."


def test_no_speaker_means_nothing_is_stripped():
    text = "Funding will rise next year."
    assert V.strip_attribution(text, None) == text


def test_the_pair_sent_to_the_model_carries_the_substance_not_the_attribution():
    c = claim(text="Ms A Member said that the funding will rise.", quote="The funding will rise to \u00a335 billion next year.")
    evidence_text, claim_text = V.entailment_inputs(c, evidence())
    assert "The funding will rise" in evidence_text
    assert not claim_text.lower().startswith("ms a member"), "the attribution must not reach the model"
