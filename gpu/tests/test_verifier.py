"""MiniCheck's wrapper and the /verify endpoint, without a GPU."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

import api
import verifier as V


# --- precision ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["fp16", "float16", "half", "int8", ""])
def test_fp16_and_friends_are_refused_before_anything_loads(bad):
    """Flan-T5's activations overflow in fp16 and the model emits NaNs. A NaN softmax arrives downstream as 0.5 — a
    confident-looking "not sure" about a claim nobody checked.
    """
    with pytest.raises(V.VerifierUnavailable, match="fp32 or bf16"):
        V.Verifier(precision=bad)


def test_the_two_allowed_precisions_are_the_two_the_brief_names():
    """A third precision appearing here would need its own agreement benchmark, not a default."""
    pytest.importorskip("torch")  # the weights check comes after the import check; CI's no-GPU step has no torch
    for good in ("fp32", "bf16"):
        with pytest.raises(V.VerifierUnavailable, match="no MiniCheck weights"):
            V.Verifier(precision=good, path="/definitely/not/here")


def test_missing_weights_say_so_rather_than_failing_later():
    pytest.importorskip("torch")  # reachable only once torch imports
    with pytest.raises(V.VerifierUnavailable, match="no MiniCheck weights at"):
        V.Verifier(path="/nope")


# --- the label ids --------------------------------------------------------------------------------------


def test_the_label_token_ids_are_the_ones_the_model_card_uses():
    """3 and 209, and neither is obvious. In this SentencePiece vocabulary "1" is one token (209) but "0" is two — id
    3, a bare space marker, then id 632 — and only the FIRST decoder token is read.
    """
    assert (V.NO_SUPPORT_TOKEN, V.SUPPORT_TOKEN) == (3, 209)


def test_the_adopted_precision_is_the_one_the_benchmark_earned():
    """300/300 verdict agreement with fp32, 95% Wilson lower bound 98.74% against a 98% floor, and 100% agreement over
    the 24 borderline pairs. Pinned as a constant so that changing it is a visible decision requiring its own run,
    not a default someone edits.
    """
    assert V.ADOPTED_PRECISION == "bf16"
    assert V.ADOPTED_PRECISION in ("fp32", "bf16"), "fp16 can never be adopted; Flan-T5 emits NaNs in it"


def test_the_default_precision_is_the_adopted_one():
    """The default a caller gets must be the measured one, or the benchmark decided nothing."""
    import inspect

    assert inspect.signature(V.Verifier.__init__).parameters["precision"].default == V.ADOPTED_PRECISION


def test_the_threshold_is_the_softmaxs_own_midpoint():
    """names no threshold. 0.5 is the point at which a two-way softmax changes its mind, and any other number would be
    an invention until E4 tunes it on data.
    """
    assert V.SUPPORT_THRESHOLD == 0.5


# --- chunking: what the model is shown --------------------------------------------------------------------


def test_a_short_speech_is_one_chunk():
    """Evidence here is one speech, well under 500 words, so in practice nothing is cut at all."""
    assert V.chunk("The House rose at six. The Minister was absent.") == [
        "The House rose at six. The Minister was absent."]


def test_a_long_document_is_split_at_sentence_boundaries():
    long = " ".join(f"Sentence number {i} runs on for a little while to add words." for i in range(120))
    pieces = V.chunk(long, words=100)
    assert len(pieces) > 1
    assert all(p.strip() for p in pieces)
    # Nothing is lost: every word survives the split, which is what max-over-chunks depends on.
    assert sum(len(p.split()) for p in pieces) == len(long.split())


def test_an_empty_document_still_yields_one_chunk():
    """Zero chunks would make `max` raise on an empty sequence, turning "no evidence" into a crash."""
    assert V.chunk("") == [""]
    assert V.chunk("   \n  ") == [""]


def test_a_single_sentence_longer_than_the_chunk_size_is_not_dropped():
    """A 600-word sentence cannot be cut at a boundary that does not exist. It must survive whole rather than vanish,
    even though it exceeds the target.
    """
    giant = " ".join(["word"] * 600)
    pieces = V.chunk(giant, words=100)
    assert len(pieces) == 1 and len(pieces[0].split()) == 600


def test_sentence_splitting_keeps_paragraph_breaks_apart():
    assert V.sentences("First line.\nSecond line.") == ["First line.", "Second line."]


# --- the endpoint ---------------------------------------------------------------------------------------


class FakeVerifier:
    """Scores by position, so a reordered response is visible."""

    precision = "fp32"

    def __init__(self, scores=None):
        self.scores = scores
        self.closed = False
        self.calls = 0

    def score(self, evidence, claims):
        self.calls += 1
        if self.scores is not None:
            return list(self.scores)[:len(claims)]
        return [0.9 - 0.1 * i for i in range(len(claims))]

    def info(self):
        return {"name": V.VERIFIER_NAME, "precision": self.precision}

    def close(self):
        self.closed = True


class FakeEngine:
    def embed(self, texts, is_query):
        return []

    def score(self, query, texts):
        return []

    def info(self):
        return {"device": "fake", "models": {}, "vram": {"total_mib": 1.0, "used_mib": 0.0, "free_mib": 1.0,
                                                         "torch_allocated_mib": 0.0, "torch_peak_mib": 0.0}}  # fmt: skip

    def close(self):
        pass


# The service refuses a Host header it does not recognise, so the test client has to present a real one.
BASE = "http://127.0.0.1:8001"


@pytest.fixture
def client(monkeypatch):
    fake = FakeVerifier()
    monkeypatch.setattr(api.verifier_module, "Verifier", lambda *a, **k: fake)
    app = api.create_app(engine_factory=FakeEngine)
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        c.fake = fake
        yield c


def test_results_come_back_in_request_order_not_score_order(client):
    """The one that matters. /rerank sorts by score because ranking is the point there; /verify must not, because the
    caller matches each result to the claim it sent, and a sorted response would attach every verdict to the wrong
    sentence while looking completely normal.
    """
    client.fake.scores = [0.1, 0.99, 0.4]
    body = {"pairs": [{"evidence": "e one", "claim": "c one"},
                      {"evidence": "e two", "claim": "c two"},
                      {"evidence": "e three", "claim": "c three"}]}  # fmt: skip
    data = client.post("/verify", json=body).json()
    assert [r["index"] for r in data["results"]] == [0, 1, 2]
    assert [round(r["score"], 2) for r in data["results"]] == [0.1, 0.99, 0.4]
    assert [r["verdict"] for r in data["results"]] == ["unsupported", "supported", "unsupported"]


def test_the_threshold_may_be_overridden_per_request(client):
    client.fake.scores = [0.6]
    body = {"pairs": [{"evidence": "e", "claim": "c"}], "threshold": 0.9}
    data = client.post("/verify", json=body).json()
    assert data["threshold"] == 0.9
    assert data["results"][0]["verdict"] == "unsupported"


def test_a_non_finite_score_is_skipped_rather_than_reported_as_a_verdict(client):
    """NaN >= threshold is False in Python, so a NaN would arrive as a confident "unsupported" about a claim the model
    never actually scored.
    """
    client.fake.scores = [float("nan"), 0.95]
    body = {"pairs": [{"evidence": "e1", "claim": "c1"}, {"evidence": "e2", "claim": "c2"}]}
    data = client.post("/verify", json=body).json()
    assert [r["index"] for r in data["results"]] == [1]
    assert data["skipped"] == [0]


def test_a_pair_with_no_scorable_text_is_skipped_by_index(client):
    body = {"pairs": [{"evidence": "   ", "claim": "c1"}, {"evidence": "e2", "claim": "c2"}]}
    data = client.post("/verify", json=body).json()
    assert data["skipped"] == [0]
    assert [r["index"] for r in data["results"]] == [1], "indices stay those of the request"


def test_no_pairs_is_an_empty_answer_not_an_error(client):
    data = client.post("/verify", json={"pairs": []}).json()
    assert data["results"] == [] and data["skipped"] == []


def test_too_many_pairs_is_refused(client):
    body = {"pairs": [{"evidence": "e", "claim": "c"}] * (api.MAX_VERIFY_PAIRS + 1)}
    assert client.post("/verify", json=body).status_code == 422


def test_an_unknown_field_is_refused(client):
    body = {"pairs": [{"evidence": "e", "claim": "c", "speaker": "nobody"}]}
    assert client.post("/verify", json=body).status_code == 422


def test_the_model_is_loaded_once_and_only_on_first_use(client):
    """3.1 GB of weights against about 5.6 GB free: loading at startup leaves no headroom for an embedding batch, and
    loading twice fails with an out-of-memory error that says nothing about the real cause.
    """
    health = client.get("/health").json()
    assert health["verifier"] == {"loaded": False, "note": "loaded on first /verify call"}
    client.post("/verify", json={"pairs": [{"evidence": "e", "claim": "c"}]})
    client.post("/verify", json={"pairs": [{"evidence": "e", "claim": "c"}]})
    assert client.fake.calls == 2, "scored twice"
    assert client.get("/health").json()["verifier"]["name"] == V.VERIFIER_NAME


def test_health_never_loads_the_model_as_a_side_effect(client):
    client.get("/health")
    client.get("/health")
    assert client.fake.calls == 0


def test_an_unavailable_verifier_is_a_503_so_the_caller_can_degrade(monkeypatch):
    """'s degradation path starts here: the caller keeps its verbatim claims and drops its paraphrases."""
    def refuse(*a, **k):
        raise V.VerifierUnavailable("no weights on this machine")

    monkeypatch.setattr(api.verifier_module, "Verifier", refuse)
    app = api.create_app(engine_factory=FakeEngine)
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        response = c.post("/verify", json={"pairs": [{"evidence": "e", "claim": "c"}]})
        assert response.status_code == 503
        assert "no weights" in response.json()["detail"]


def test_the_verifier_is_released_on_shutdown(monkeypatch):
    fake = FakeVerifier()
    monkeypatch.setattr(api.verifier_module, "Verifier", lambda *a, **k: fake)
    app = api.create_app(engine_factory=FakeEngine)
    with TestClient(app, base_url=BASE, raise_server_exceptions=False) as c:
        c.post("/verify", json={"pairs": [{"evidence": "e", "claim": "c"}]})
    assert fake.closed is True, "3.1 GB must not outlive the process that loaded it"


def test_a_score_count_mismatch_is_an_error_not_a_misalignment(client):
    """Returning fewer scores than pairs would pair each verdict with the wrong claim from that point on."""
    client.fake.scores = [0.9]
    body = {"pairs": [{"evidence": "e1", "claim": "c1"}, {"evidence": "e2", "claim": "c2"}]}
    assert client.post("/verify", json=body).status_code == 500


# --- the self-check -------------------------------------------------------------------------------------


def test_the_self_check_pairs_are_unambiguous():
    """They have to be: a load-time check that can plausibly fail on a working model is a check that will be disabled
    the first time it fires.
    """
    supported, contradicted = V.Verifier.SANITY_PAIRS
    assert supported[2] is True and contradicted[2] is False
    assert "35 billion" in supported[1]
    assert "cut by half" in contradicted[1]


def test_the_self_check_catches_a_pipeline_wired_backwards():
    """The token ids can be right while the softmax column or the prompt order is wrong — every static check passes and
    every score is inverted.
    """
    class Backwards:
        SANITY_PAIRS = V.Verifier.SANITY_PAIRS
        self_check = V.Verifier.self_check

        def score(self, evidence, claims):
            return [0.01, 0.99]  # inverted

    with pytest.raises(V.VerifierUnavailable, match="wired the wrong way round"):
        Backwards().self_check()


def test_the_self_check_passes_when_the_scores_separate_correctly():
    class Forwards:
        SANITY_PAIRS = V.Verifier.SANITY_PAIRS
        self_check = V.Verifier.self_check

        def score(self, evidence, claims):
            return [0.91, 0.006]

    result = Forwards().self_check()
    assert result["separation"] == pytest.approx(0.904)
    assert math.isclose(result["supported"], 0.91)
