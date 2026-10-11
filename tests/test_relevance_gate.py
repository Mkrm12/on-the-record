"""An answer none of whose claims is about the question is a near miss, however well each quote verified. The verifier
proves a quote EXISTS and belongs to its speaker; nothing asked whether it ANSWERS.
"""

from __future__ import annotations

from types import SimpleNamespace

from answer import contract as K
from answer import pipeline as PL
from ingest.runtime import run

PROV = dict(route="retrieval", resolved_filters={}, corpus_version="v2", retrieval_config_hash="x", recipe="r",
            prompt_version="p", model_id="m")


def claim(i, text):
    verified = K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True, window_ok=True,
                              checked_by="test")
    return K.Claim(id=f"c{i}", kind="summary", text=text, speaker_name="A Member",
                   citations=(K.Citation(speech_id="s1", chunk_ids=(i,), quote="words"),), verification=verified)


def answer(*texts):
    return K.Answer(question="q", provenance=K.Provenance(**PROV), claims=tuple(claim(i, t) for i, t in enumerate(texts)),
                    rendered=" ".join(texts))


def runtime(scores=None, error=None):
    class _Gpu:
        async def rerank(self, query, candidates):
            if error:
                raise error
            return list(scores)[: len(candidates)]
    return SimpleNamespace(searcher=SimpleNamespace(gpu=_Gpu()))


def test_an_answer_whose_best_claim_is_off_the_question_declines():
    got = run(PL.relevance_gate(runtime([-9.6, -8.0]), "did that bloke compare the housing market to a castle",
                                answer("Baroness Seccombe said a man she knew made his home his castle.", "Other.")))
    assert got.abstention == K.Abstention.NOT_IN_RECORD and not got.claims
    assert got.abstention_detail == "The passages found in the parliamentary record do not answer this question."


def test_one_relevant_claim_keeps_the_answer_whole():
    original = answer("Relevant.", "Barely relevant.")
    assert run(PL.relevance_gate(runtime([-7.0, -2.7]), "q", original)) is original


def test_the_floor_is_the_measured_one():
    """Hum0091's CORRECT answer scored -7.21 and was declined at -6.0. The widest margin between the labelled points —
    correct answers >= -7.21, the four near misses <= -8.1 — is their midpoint.
    """
    assert PL.CLAIM_RELEVANCE_FLOOR == -7.65


def test_a_correct_answer_in_the_old_gap_is_kept_and_the_near_misses_still_go():
    """The measured points, both sides of the floor."""
    kept = answer("She extended the image to older generations locked inside the castle towers.", "Other.")
    assert run(PL.relevance_gate(runtime([-7.21, -8.96]), "did that bloke really compare the housing market to a "
                                 "castle lol", kept)) is kept  # fmt: skip
    for near_miss in (-8.1, -8.2, -9.0, -9.6):
        got = run(PL.relevance_gate(runtime([near_miss]), "q", answer("Off the question.")))
        assert got.abstention == K.Abstention.NOT_IN_RECORD, near_miss


def test_the_best_claims_relevance_is_recorded_whatever_the_gate_decides():
    """So the floor can be re-fitted on judged answers rather than on five points (as `Sufficiency.top_rerank` is).
    """
    record = SimpleNamespace(claim_relevance=None)
    run(PL.relevance_gate(runtime([-3.5, -1.25]), "q", answer("A.", "B."), record=record))
    assert record.claim_relevance == -1.25
    record = SimpleNamespace(claim_relevance=None)
    run(PL.relevance_gate(runtime([-9.0]), "q", answer("Off."), record=record))
    assert record.claim_relevance == -9.0


def test_an_unavailable_scorer_leaves_the_answer_standing():
    """Retrieval degrades without the GPU rather than refusing; this gate does the same, and says so in the log."""
    original = answer("Relevant.")
    assert run(PL.relevance_gate(runtime(error=RuntimeError("gpu down")), "q", original)) is original


def test_a_decline_or_a_claimless_answer_passes_untouched():
    declined = K.Answer(question="q", provenance=K.Provenance(**PROV), abstention=K.Abstention.NOT_IN_RECORD,
                        abstention_detail="x")
    assert run(PL.relevance_gate(runtime([-9.0]), "q", declined)) is declined
    assert run(PL.relevance_gate(runtime([-9.0]), "q", None)) is None
