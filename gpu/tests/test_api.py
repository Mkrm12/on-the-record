"""The HTTP service."""

import _thread
import asyncio
import gc
import json
import signal
import threading
import time
import types
import weakref
import zlib

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
import check_cuda
from api import (
    MAX_BODY_BYTES,
    MAX_EMBED_TEXTS,
    MAX_QUERY_CHARS,
    MAX_RERANK_CANDIDATES,
    MAX_TEXT_CHARS,
    MIN_FREE_VRAM_MIB,
    GpuBusy,
    GpuEngine,
    GpuQueue,
    dedupe,
    is_loopback_host,
    sanitize,
    vector_ok,
)
from embedder import DIMENSIONS, MODEL_NAME, MODEL_REVISION, QUERY_INSTRUCTION, l2_normalize
from reranker import RERANKER_NAME, RERANKER_REVISION, GpuOutOfMemory

BASE = "http://127.0.0.1:8001"
MiB = 2**20

# --------------------------------------------------------------------------- a deterministic stand-in for the two
# models ---------------------------------------------------------------------------


def fake_vector(text: str) -> np.ndarray:
    return l2_normalize(np.random.default_rng(zlib.crc32(text.encode("utf-8", "replace"))).standard_normal((1, DIMENSIONS)))[0]


class FakeEngine:
    """Vectors are a pure function of the text and of is_query (as the real instruction makes them); a score is the
    number of words the query and passage share, so 'relevant' is knowable. Hooks force every failure mode.
    """

    def __init__(self):
        self.embed_calls, self.score_calls = [], []
        self.threads, self.active, self.max_active = set(), 0, 0
        self.closed = False
        self.close_thread = None
        self.free_mib = 7000.0
        self.raise_on = None  # an exception to raise from embed / score
        self.info_error = None
        self.nan_scores_for = set()  # passages that get a NaN score
        self.bad_vector_for = {}  # text -> the (wrong) vector to return
        self.block = None  # a threading.Event the next calls wait on
        self.delay = 0.0  # seconds each call takes, so concurrent calls would overlap
        self.wide = False  # return float64 vectors that are NOT exactly float32 values
        self._lock = threading.Lock()

    def _enter(self):
        with self._lock:
            self.threads.add(threading.current_thread().name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        if self.delay:
            time.sleep(self.delay)
        if self.block is not None:
            self.block.wait(timeout=10)
        if self.raise_on is not None:
            raise self.raise_on

    def _exit(self):
        with self._lock:
            self.active -= 1

    def embed(self, texts, is_query):
        self.embed_calls.append((list(texts), is_query))
        self._enter()
        try:
            vectors = [self.bad_vector_for.get(t, fake_vector(("Q:" if is_query else "D:") + t)) for t in texts]
            return [v.astype(np.float64) * (1 + 1e-7) for v in vectors] if self.wide else vectors
        finally:
            self._exit()

    def score(self, query, texts):
        self.score_calls.append((query, list(texts)))
        self._enter()
        try:
            words = set(query.lower().split())
            return np.array([np.nan if t in self.nan_scores_for else len(words & set(t.lower().split())) for t in texts], dtype=np.float32)
        finally:
            self._exit()

    def info(self):
        if self.info_error:
            raise self.info_error
        total = 8192.0
        return {
            "device": "Fake GPU",
            "models": {"embedder": {"name": MODEL_NAME}, "reranker": {"name": RERANKER_NAME}},
            "vram": {"total_mib": total, "used_mib": total - self.free_mib, "free_mib": self.free_mib},
        }

    def close(self):
        self.closed = True
        self.close_thread = threading.current_thread().name


_dumps = json.dumps


class WireClient(TestClient):
    r"""Sends `json=` the way requests and browsers do: ASCII escapes, so a lone surrogate reaches the server as
    "\ud800". httpx's own encoder writes raw UTF-8 and refuses to encode one at all.
    """

    def post(self, url, *, json=None, **kwargs):
        if json is not None:
            kwargs["content"] = _dumps(json)
            kwargs["headers"] = {**(kwargs.get("headers") or {}), "content-type": "application/json"}
        return super().post(url, **kwargs)


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(engine):
    with WireClient(api.create_app(lambda: engine), base_url=BASE, raise_server_exceptions=False) as c:
        yield c


def norms(embeddings):
    return [float(np.linalg.norm(np.asarray(e, dtype=np.float64))) for e in embeddings if e is not None]


# --------------------------------------------------------------------------- pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [None, "", "  \n\t ", "...", "!?—", "\x00\x00", "\ud800", "\U0001F600\U0001F600"])
def test_sanitize_gives_none_when_nothing_embeddable_is_left(text):
    assert sanitize(text) is None


def test_sanitize_strips_nuls_lone_surrogates_and_extra_whitespace():
    assert sanitize("  a\x00b \ud800 c \n\n d ") == "ab c d"


def test_sanitize_keeps_real_text_including_non_latin_scripts_and_astral_characters():
    for text in ["The scheme costs £60.5 billion.", "مرحبا بالعالم", "你好世界，议会辩论", "naïve café 😀"]:
        assert sanitize(text) == text


def test_sanitize_cuts_text_at_the_limit_and_only_beyond_it():
    assert sanitize("a" * 50, limit=50) == "a" * 50
    assert sanitize("a" * 51, limit=50) == "a" * 50
    assert len(sanitize("word " * 100_000)) <= MAX_TEXT_CHARS


def test_sanitize_checks_again_after_a_cut_that_leaves_only_punctuation():
    assert sanitize("." * 100 + "abc", limit=50) is None


def test_sanitize_does_not_leave_a_trailing_space_where_it_cuts():
    assert sanitize("aaaa bbbb", limit=5) == "aaaa"


def test_dedupe_lists_each_string_once_in_first_seen_order_and_maps_every_position_back():
    unique, slots = dedupe(["b", None, "a", "b", "a", "c"])
    assert unique == ["b", "a", "c"] and slots == [0, None, 1, 0, 1, 2]


def test_dedupe_of_nothing():
    assert dedupe([]) == ([], []) and dedupe([None, None]) == ([], [None, None])


def test_vector_ok_accepts_unit_vectors_and_refuses_everything_else():
    good = fake_vector("x")
    assert vector_ok(good)
    assert not vector_ok(good * 2)
    assert not vector_ok(np.full(DIMENSIONS, np.nan, dtype=np.float32))
    assert not vector_ok(np.full(DIMENSIONS, np.inf, dtype=np.float32))
    assert not vector_ok(good[:100])
    assert not vector_ok(np.zeros(DIMENSIONS, dtype=np.float32))
    assert not vector_ok(np.float32(1.0))  # a scalar, not a vector


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost"])
def test_loopback_hosts_are_accepted(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "::1", "192.168.1.20", "10.0.0.1", "example.com", ""])
def test_anything_else_is_refused(host):
    assert not is_loopback_host(host)


# --------------------------------------------------------------------------- GET /health
# ---------------------------------------------------------------------------


def test_health_reports_status_device_models_and_vram(client, engine):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["device"] == "Fake GPU"
    assert body["models"]["embedder"]["name"] == MODEL_NAME and body["models"]["reranker"]["name"] == RERANKER_NAME
    assert body["vram"]["free_mib"] == 7000 and body["vram"]["total_mib"] == 8192
    assert body["vram"]["used_mib"] + body["vram"]["free_mib"] == body["vram"]["total_mib"]
    assert body["queue"] == {"queued": 0, "max_queued": api.MAX_QUEUED}
    assert isinstance(body["uptime_s"], float) and body["uptime_s"] >= 0


def test_health_flags_low_vram_below_the_headroom_the_owner_asked_for(client, engine):
    engine.free_mib = MIN_FREE_VRAM_MIB - 1
    body = client.get("/health").json()
    assert body["status"] == "low_vram" and body["vram"]["headroom_ok"] is False and body["vram"]["min_free_mib"] == 4096


def test_health_is_fine_at_exactly_the_headroom(client, engine):
    engine.free_mib = MIN_FREE_VRAM_MIB
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["vram"]["headroom_ok"] is True


def test_health_says_503_when_the_gpu_cannot_be_probed_and_does_not_leak_the_reason(client, engine):
    engine.info_error = RuntimeError("CUDA error: /secret/path")
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["status"] == "error" and "secret" not in response.text


def test_health_only_accepts_get(client):
    assert client.post("/health").status_code == 405


# --------------------------------------------------------------------------- POST /embed
# ---------------------------------------------------------------------------


def test_embed_returns_one_unit_vector_per_text_in_order(client, engine):
    body = client.post("/embed", json={"texts": ["housing", "the NHS", "rail fares"]}).json()
    assert body["dim"] == DIMENSIONS == 384 and body["count"] == 3 and body["skipped"] == []
    assert all(len(e) == 384 for e in body["embeddings"])
    assert np.allclose(norms(body["embeddings"]), 1.0, atol=1e-5)
    assert body["embeddings"][1] == pytest.approx(fake_vector("Q:the NHS").tolist())  # the second text got the second vector


def test_embed_values_are_exactly_the_float32_numbers_so_they_round_trip(client):
    body = client.post("/embed", json={"texts": ["housing"]}).json()
    values = np.array(body["embeddings"][0], dtype=np.float64)
    assert np.array_equal(values, values.astype(np.float32).astype(np.float64))
    assert body["embeddings"][0] == fake_vector("Q:housing").tolist()


def test_embed_serialises_float32_values_even_if_the_engine_hands_back_wider_numbers(client, engine):
    engine.wide = True
    values = np.array(client.post("/embed", json={"texts": ["housing"]}).json()["embeddings"][0], dtype=np.float64)
    assert np.array_equal(values, values.astype(np.float32).astype(np.float64))


def test_embed_defaults_to_query_mode_which_carries_the_retrieval_instruction(client, engine):
    client.post("/embed", json={"texts": ["housing"]})
    assert engine.embed_calls == [(["housing"], True)]


def test_embed_with_is_query_false_uses_document_mode(client, engine):
    client.post("/embed", json={"texts": ["housing"], "is_query": False})
    assert engine.embed_calls == [(["housing"], False)]


def test_query_and_document_vectors_of_the_same_text_differ(client):
    q = client.post("/embed", json={"texts": ["housing"], "is_query": True}).json()["embeddings"][0]
    d = client.post("/embed", json={"texts": ["housing"], "is_query": False}).json()["embeddings"][0]
    assert q != d


def test_embed_of_an_empty_list_is_an_immediate_empty_answer_with_no_gpu_call(client, engine):
    body = client.post("/embed", json={"texts": []}).json()
    assert body == {"embeddings": [], "dim": 384, "count": 0, "skipped": []} and engine.embed_calls == []


def test_embed_puts_null_where_nothing_is_embeddable_and_reports_which(client, engine):
    texts = ["real text", None, "", "   ", "...", "\x00", "\ud800", "😀", "more text"]
    body = client.post("/embed", json={"texts": texts}).json()
    assert body["count"] == 9 and body["skipped"] == [1, 2, 3, 4, 5, 6, 7]
    assert [e is None for e in body["embeddings"]] == [False, True, True, True, True, True, True, True, False]
    assert engine.embed_calls == [(["real text", "more text"], True)]  # the model never saw the unusable ones


def test_embed_of_only_unusable_texts_is_a_200_with_no_gpu_call(client, engine):
    body = client.post("/embed", json={"texts": ["", None, "..."]}).json()
    assert body["embeddings"] == [None, None, None] and body["skipped"] == [0, 1, 2] and engine.embed_calls == []


def test_embed_cleans_nuls_and_whitespace_before_the_model_sees_the_text(client, engine):
    client.post("/embed", json={"texts": ["a\x00b   c\n\nd"]})
    assert engine.embed_calls[0][0] == ["ab c d"]


def test_embed_cuts_over_long_text_before_the_model_sees_it(client, engine):
    client.post("/embed", json={"texts": ["word " * 50_000]})
    assert len(engine.embed_calls[0][0][0]) <= MAX_TEXT_CHARS


def test_embed_scores_duplicates_once_and_gives_them_identical_vectors(client, engine):
    body = client.post("/embed", json={"texts": ["a b", "c d", "a b", "a b"]}).json()
    assert engine.embed_calls == [(["a b", "c d"], True)]
    assert body["embeddings"][0] == body["embeddings"][2] == body["embeddings"][3] != body["embeddings"][1]


def test_embed_accepts_the_maximum_number_of_texts_and_refuses_one_more(client):
    ok = client.post("/embed", json={"texts": [f"text {i}" for i in range(MAX_EMBED_TEXTS)]})
    assert ok.status_code == 200 and ok.json()["count"] == MAX_EMBED_TEXTS
    too_many = client.post("/embed", json={"texts": [f"text {i}" for i in range(MAX_EMBED_TEXTS + 1)]})
    assert too_many.status_code == 422


def test_embed_handles_varied_real_world_input(client):
    texts = ["مرحبا بالعالم", "你好世界，议会辩论", "naïve café 😀", "a" * 200_000, "x​y﻿z", "‮abc", "Line one\nline two", "£60.5 billion", "2026"]
    body = client.post("/embed", json={"texts": texts}).json()
    assert body["skipped"] == [] and len(body["embeddings"]) == len(texts)


def test_embed_never_sends_out_a_vector_that_is_not_finite_and_unit_length(client, engine):
    engine.bad_vector_for = {
        "nan": np.full(DIMENSIONS, np.nan, dtype=np.float32),
        "long": fake_vector("x") * 3,
        "short": fake_vector("x")[:10],
    }
    response = client.post("/embed", json={"texts": ["fine", "nan", "long", "short"]})
    body = response.json()  # would not parse if a NaN had been serialised
    assert response.status_code == 200 and body["skipped"] == [1, 2, 3] and body["embeddings"][0] is not None


@pytest.mark.parametrize(
    "payload",
    [
        {},  # texts missing
        {"texts": "one string, not a list"},
        {"texts": None},
        {"texts": [1, 2]},  # numbers are not text
        {"texts": [["nested"]]},
        {"texts": ["a"], "is_querry": False},  # a typo must not silently fall back to query mode
        {"texts": ["a"], "is_query": "sideways"},
    ],
)
def test_embed_rejects_malformed_requests_with_422(client, payload):
    assert client.post("/embed", json=payload).status_code == 422


def test_embed_rejects_invalid_json_with_422(client):
    response = client.post("/embed", content=b'{"texts": ["unterminated', headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_a_422_never_echoes_the_request_text_back(client):
    marker = "PRIVATE-QUESTION-TEXT"
    response = client.post("/embed", json={"texts": [marker * 100, 7], "extra": marker})
    assert response.status_code == 422 and marker not in response.text
    invalid = client.post("/embed", content=('{"texts": ["' + marker).encode(), headers={"content-type": "application/json"})
    assert invalid.status_code == 422 and marker not in invalid.text


def test_a_422_reports_at_most_20_problems(client):
    response = client.post("/embed", json={"texts": list(range(300))})
    assert response.status_code == 422 and 0 < len(response.json()["detail"]) <= 20


def test_embed_only_accepts_post(client):
    assert client.get("/embed").status_code == 405


# --------------------------------------------------------------------------- POST /rerank
# ---------------------------------------------------------------------------

QUERY = "waiting lists for NHS treatment"
RELEVANT = "NHS waiting lists for treatment are too long"
PARTLY = "treatment plans for patients"
IRRELEVANT = "the recipe needs two eggs"


def test_rerank_puts_the_relevant_candidate_first_with_the_highest_score(client):
    candidates = [IRRELEVANT, PARTLY, RELEVANT]
    body = client.post("/rerank", json={"query": QUERY, "candidates": candidates}).json()
    assert [r["index"] for r in body["results"]] == [2, 1, 0]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1]
    assert body["count"] == 3 and body["skipped"] == []


def test_rerank_index_points_back_at_the_request_position_and_text_is_the_candidate_as_sent(client):
    messy = "  NHS   waiting\nlists for treatment  "
    body = client.post("/rerank", json={"query": QUERY, "candidates": [IRRELEVANT, messy]}).json()
    assert body["results"][0]["index"] == 1 and body["results"][0]["text"] == messy


def test_rerank_top_k_keeps_the_best_k(client):
    body = client.post("/rerank", json={"query": QUERY, "candidates": [IRRELEVANT, RELEVANT, PARTLY], "top_k": 2}).json()
    assert [r["index"] for r in body["results"]] == [1, 2] and body["count"] == 2


def test_rerank_top_k_larger_than_the_list_is_not_an_error(client):
    body = client.post("/rerank", json={"query": QUERY, "candidates": [RELEVANT], "top_k": 50}).json()
    assert body["count"] == 1


def test_rerank_top_k_of_one_returns_exactly_the_winner(client):
    body = client.post("/rerank", json={"query": QUERY, "candidates": [PARTLY, RELEVANT, IRRELEVANT], "top_k": 1}).json()
    assert [r["index"] for r in body["results"]] == [1]


@pytest.mark.parametrize("top_k", [0, -1, "many", 1.5])
def test_rerank_refuses_a_useless_top_k(client, top_k):
    assert client.post("/rerank", json={"query": QUERY, "candidates": [RELEVANT], "top_k": top_k}).status_code == 422


def test_rerank_equal_scores_keep_the_request_order(client):
    body = client.post("/rerank", json={"query": QUERY, "candidates": ["alpha one", "beta two", "gamma three"]}).json()
    assert [r["index"] for r in body["results"]] == [0, 1, 2]


def test_rerank_of_no_candidates_is_an_immediate_empty_answer_with_no_gpu_call_even_for_a_blank_query(client, engine):
    for query in (QUERY, ""):
        assert client.post("/rerank", json={"query": query, "candidates": []}).json() == {"results": [], "count": 0, "skipped": []}
    assert engine.score_calls == []


@pytest.mark.parametrize("query", ["", "   ", "...", "\x00", "\ud800", "😀"])
def test_rerank_refuses_a_query_with_nothing_to_score_against_with_a_clear_422(client, engine, query):
    response = client.post("/rerank", json={"query": query, "candidates": [RELEVANT]})
    assert response.status_code == 422 and "query" in response.json()["detail"] and engine.score_calls == []


def test_rerank_leaves_out_candidates_that_cannot_be_scored_and_says_which(client, engine):
    candidates = [RELEVANT, "", None, "   ", "...", "\x00", PARTLY, "\ud800"]
    body = client.post("/rerank", json={"query": QUERY, "candidates": candidates}).json()
    assert [r["index"] for r in body["results"]] == [0, 6] and body["skipped"] == [1, 2, 3, 4, 5, 7]
    assert engine.score_calls == [(QUERY, [RELEVANT, PARTLY])]


def test_rerank_of_only_unusable_candidates_is_a_200_with_no_gpu_call(client, engine):
    body = client.post("/rerank", json={"query": QUERY, "candidates": ["", None, "..."]}).json()
    assert body == {"results": [], "count": 0, "skipped": [0, 1, 2]} and engine.score_calls == []


def test_rerank_echoes_a_candidate_that_holds_a_lone_surrogate_without_the_surrogate_instead_of_failing(client):
    # found by the real-model tests: pydantic cannot serialise a lone surrogate, so echoing it verbatim was a 500
    response = client.post("/rerank", json={"query": QUERY, "candidates": ["NHS waiting \ud800 lists", IRRELEVANT]})
    assert response.status_code == 200 and response.json()["results"][0]["text"] == "NHS waiting  lists"


def test_rerank_scores_duplicate_candidates_once_and_gives_them_the_same_score(client, engine):
    body = client.post("/rerank", json={"query": QUERY, "candidates": [RELEVANT, IRRELEVANT, RELEVANT]}).json()
    assert engine.score_calls == [(QUERY, [RELEVANT, IRRELEVANT])]
    assert [r["index"] for r in body["results"]] == [0, 2, 1]
    assert body["results"][0]["score"] == body["results"][1]["score"]


def test_rerank_cleans_query_and_candidates_before_the_model_sees_them(client, engine):
    client.post("/rerank", json={"query": "  NHS\x00  lists ", "candidates": ["waiting\x00  lists\n\n now"]})
    assert engine.score_calls == [("NHS lists", ["waiting lists now"])]


def test_rerank_cuts_a_huge_query_and_huge_candidates_before_the_model_sees_them(client, engine):
    client.post("/rerank", json={"query": "word " * 50_000, "candidates": ["word " * 50_000]})
    query, (candidate,) = engine.score_calls[0]
    assert len(query) <= MAX_QUERY_CHARS and len(candidate) <= MAX_TEXT_CHARS and MAX_QUERY_CHARS < MAX_TEXT_CHARS


def test_rerank_skips_a_candidate_whose_score_is_not_finite_rather_than_failing(client, engine):
    engine.nan_scores_for = {PARTLY}
    response = client.post("/rerank", json={"query": QUERY, "candidates": [PARTLY, RELEVANT]})
    body = response.json()  # would not parse (or would be a 500) if a NaN had been serialised
    assert response.status_code == 200 and [r["index"] for r in body["results"]] == [1] and body["skipped"] == [0]


def test_rerank_accepts_the_maximum_number_of_candidates_and_refuses_one_more(client):
    ok = client.post("/rerank", json={"query": QUERY, "candidates": [f"candidate {i}" for i in range(MAX_RERANK_CANDIDATES)]})
    assert ok.status_code == 200 and ok.json()["count"] == MAX_RERANK_CANDIDATES
    assert client.post("/rerank", json={"query": QUERY, "candidates": [f"c {i}" for i in range(MAX_RERANK_CANDIDATES + 1)]}).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"candidates": [RELEVANT]},  # no query
        {"query": None, "candidates": [RELEVANT]},
        {"query": 5, "candidates": [RELEVANT]},
        {"query": QUERY},  # no candidates
        {"query": QUERY, "candidates": RELEVANT},
        {"query": QUERY, "candidates": [1, 2]},
        {"query": QUERY, "candidates": [RELEVANT], "topk": 3},
    ],
)
def test_rerank_rejects_malformed_requests_with_422(client, payload):
    assert client.post("/rerank", json=payload).status_code == 422


def test_rerank_scores_are_plain_floats_in_the_response(client):
    body = client.post("/rerank", json={"query": QUERY, "candidates": [RELEVANT]}).json()
    assert isinstance(body["results"][0]["score"], float)


def test_rerank_only_accepts_post(client):
    assert client.get("/rerank").status_code == 405


# --------------------------------------------------------------------------- what can go wrong around the request
# ---------------------------------------------------------------------------


def body_of_exact_size(size: int) -> bytes:
    prefix, suffix = b'{"texts":["', b'"]}'
    return prefix + b"a" * (size - len(prefix) - len(suffix)) + suffix


def test_a_body_of_exactly_the_limit_is_accepted_and_one_byte_more_is_refused_with_413(client, engine):
    headers = {"content-type": "application/json"}
    ok = client.post("/embed", content=body_of_exact_size(MAX_BODY_BYTES), headers=headers)
    assert ok.status_code == 200 and len(engine.embed_calls[0][0][0]) == MAX_TEXT_CHARS  # cut, not rejected
    big = client.post("/embed", content=body_of_exact_size(MAX_BODY_BYTES + 1), headers=headers)
    assert big.status_code == 413 and len(engine.embed_calls) == 1  # refused before it was read


def test_the_same_limit_protects_rerank(client, engine):
    body = json.dumps({"query": "q", "candidates": ["a" * (MAX_BODY_BYTES + 1)]}).encode()
    assert client.post("/rerank", content=body, headers={"content-type": "application/json"}).status_code == 413
    assert engine.score_calls == []


def test_a_chunked_body_with_no_declared_length_is_refused_with_411(client, engine):
    parts = iter([b'{"texts": ', b'["hello"]}'])
    response = client.post("/embed", content=parts, headers={"content-type": "application/json"})
    assert response.status_code == 411 and engine.embed_calls == []


def test_a_request_for_another_host_name_is_refused_to_defeat_dns_rebinding(client):
    assert client.get("/health", headers={"host": "evil.example.com"}).status_code == 400
    assert client.get("/health", headers={"host": "evil.example.com:8001"}).status_code == 400


def test_the_loopback_names_and_ports_are_accepted(client):
    for host in ("127.0.0.1", "127.0.0.1:8001", "localhost", "localhost:8001"):
        assert client.get("/health", headers={"host": host}).status_code == 200


def test_an_unexpected_engine_error_is_a_generic_500_that_leaks_nothing_and_the_service_carries_on(client, engine):
    engine.raise_on = RuntimeError("secret internal detail at /home/user/x.py")
    response = client.post("/embed", json={"texts": ["housing"]})
    assert response.status_code == 500 and response.json() == {"detail": "internal error"}
    engine.raise_on = None
    assert client.post("/embed", json={"texts": ["housing"]}).status_code == 200


def test_out_of_memory_is_a_503_with_retry_after_and_the_service_recovers(client, engine):
    engine.raise_on = GpuOutOfMemory("out of memory embedding 512 texts")
    response = client.post("/embed", json={"texts": ["housing"]})
    assert response.status_code == 503 and response.headers["retry-after"] == "2" and "memory" in response.json()["detail"]
    engine.raise_on = None
    assert client.post("/embed", json={"texts": ["housing"]}).status_code == 200


@pytest.mark.parametrize("returned", [1, 3])  # too few and too many: the second would otherwise be silently cut
def test_an_engine_that_returns_the_wrong_number_of_results_is_a_500_not_a_silent_misalignment(client, engine, returned):
    engine.embed = lambda texts, is_query: [fake_vector("x")] * returned
    assert client.post("/embed", json={"texts": ["a", "b"]}).status_code == 500
    engine.score = lambda query, texts: np.zeros(returned, dtype=np.float32)
    assert client.post("/rerank", json={"query": QUERY, "candidates": ["a b", "c d"]}).status_code == 500


def test_unknown_paths_are_404(client):
    assert client.get("/nope").status_code == 404


# --------------------------------------------------------------------------- one GPU thread, a bounded queue
# ---------------------------------------------------------------------------


async def call_app(application, scenario):
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url=BASE) as http:
            return await scenario(http)


def test_concurrent_requests_never_overlap_on_the_gpu_and_all_run_on_the_one_gpu_thread(engine):
    engine.delay = 0.005  # long enough that two calls WOULD overlap if the queue ran them side by side

    async def scenario(http):
        calls = [http.post("/embed", json={"texts": [f"text {i}"]}) for i in range(20)]
        calls += [http.post("/rerank", json={"query": QUERY, "candidates": [f"cand {i}", RELEVANT]}) for i in range(20)]
        return [r.status_code for r in await asyncio.gather(*calls)]

    assert set(asyncio.run(call_app(api.create_app(lambda: engine, max_queued=64), scenario))) == {200}
    assert engine.max_active == 1
    assert len(engine.threads) == 1 and next(iter(engine.threads)).startswith("gpu")


def test_when_too_many_requests_are_waiting_the_extra_ones_get_503_and_health_still_answers_at_once(engine):
    engine.block = threading.Event()

    async def scenario(http):
        tasks = [asyncio.ensure_future(http.post("/embed", json={"texts": [f"text {i}"]})) for i in range(5)]
        await asyncio.sleep(0.5)
        started = time.perf_counter()
        health = await http.get("/health")
        health_seconds = time.perf_counter() - started
        rejected = [t.result() for t in tasks if t.done()]
        engine.block.set()
        finished = await asyncio.gather(*tasks)
        return health, health_seconds, rejected, finished

    health, health_seconds, rejected, finished = asyncio.run(call_app(api.create_app(lambda: engine, max_queued=2), scenario))
    assert health.status_code == 200 and health.json()["queue"] == {"queued": 2, "max_queued": 2} and health_seconds < 1.0
    assert len(rejected) == 3 and all(r.status_code == 503 and r.headers["retry-after"] == "1" for r in rejected)
    assert sorted(r.status_code for r in finished) == [200, 200, 503, 503, 503]


def test_the_queue_count_returns_to_zero_after_a_burst(engine):
    async def scenario(http):
        await asyncio.gather(*[http.post("/embed", json={"texts": [f"t {i}"]}) for i in range(6)])
        return (await http.get("/health")).json()["queue"]["queued"]

    assert asyncio.run(call_app(api.create_app(lambda: engine, max_queued=8), scenario)) == 0


def test_queue_refuses_beyond_its_limit_with_gpu_busy():
    async def scenario():
        queue, release = GpuQueue(max_queued=2), threading.Event()
        jobs = [asyncio.ensure_future(queue.run(release.wait, 10)) for _ in range(2)]
        await asyncio.sleep(0.05)
        with pytest.raises(GpuBusy):
            await queue.run(lambda: None)
        assert queue.queued == 2
        release.set()
        await asyncio.gather(*jobs)
        await asyncio.sleep(0.05)
        assert queue.queued == 0
        assert await queue.run(lambda: "still works") == "still works"
        queue.shutdown()

    asyncio.run(scenario())


def test_queue_does_not_leak_a_slot_when_a_waiting_request_is_cancelled():
    async def scenario():
        queue, release = GpuQueue(max_queued=3), threading.Event()
        first = asyncio.ensure_future(queue.run(release.wait, 10))
        await asyncio.sleep(0.05)
        second = asyncio.ensure_future(queue.run(lambda: "never runs"))
        await asyncio.sleep(0.05)
        assert queue.queued == 2
        second.cancel()
        release.set()
        await first
        await asyncio.sleep(0.05)
        assert queue.queued == 0
        queue.shutdown()

    asyncio.run(scenario())


def test_queue_releases_the_slot_when_a_job_raises():
    async def scenario():
        queue = GpuQueue(max_queued=1)

        def boom():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            await queue.run(boom)
        await asyncio.sleep(0.05)
        assert queue.queued == 0 and await queue.run(lambda: 1) == 1
        queue.shutdown()

    asyncio.run(scenario())


def test_queue_unbounded_work_is_never_refused():
    async def scenario():
        queue, release = GpuQueue(max_queued=1), threading.Event()
        blocker = asyncio.ensure_future(queue.run(release.wait, 10))
        await asyncio.sleep(0.05)
        startup = asyncio.ensure_future(queue.run_unbounded(lambda: "loaded"))
        release.set()
        assert await startup == "loaded"
        await blocker
        queue.shutdown()

    asyncio.run(scenario())


# --------------------------------------------------------------------------- lifespan: load once at startup, release at
# shutdown ---------------------------------------------------------------------------


def test_models_are_built_once_at_startup_on_the_gpu_thread_and_released_on_it_at_shutdown(engine):
    built = []

    def factory():
        built.append(threading.current_thread().name)
        return engine

    with TestClient(api.create_app(factory), base_url=BASE) as c:
        c.post("/embed", json={"texts": ["a"]})
        c.post("/embed", json={"texts": ["b"]})
        assert len(built) == 1 and built[0].startswith("gpu") and not engine.closed
    assert engine.closed and engine.close_thread.startswith("gpu")
    assert engine.threads == {engine.close_thread}  # startup, every call and cleanup shared one thread


def test_the_engine_is_released_even_if_the_service_saw_errors(engine):
    with TestClient(api.create_app(lambda: engine), base_url=BASE, raise_server_exceptions=False) as c:
        engine.raise_on = RuntimeError("boom")
        assert c.post("/embed", json={"texts": ["a"]}).status_code == 500
    assert engine.closed


def test_a_failed_model_load_stops_the_service_from_starting_and_leaks_no_thread():
    def factory():
        raise RuntimeError("weights missing")

    before = set(threading.enumerate())  # thread objects, not names: every pool thread is called gpu_0
    with pytest.raises(RuntimeError, match="weights missing"):
        with TestClient(api.create_app(factory), base_url=BASE):
            pass
    assert {t for t in threading.enumerate() if t.name.startswith("gpu")} - before == set()


# --------------------------------------------------------------------------- GpuEngine, with fake models and a fake
# torch ---------------------------------------------------------------------------


class FakeCuda:
    def __init__(self, available=True, calls=None):
        self.available, self.calls = available, calls if calls is not None else []

    def is_available(self):
        return self.available

    def synchronize(self):
        self.calls.append("synchronize")

    def empty_cache(self):
        self.calls.append("empty_cache")

    def mem_get_info(self, device):
        return 7000 * MiB, 8192 * MiB  # (free, total), as torch returns them

    def get_device_name(self, device):
        return "NVIDIA GeForce RTX 3070 Laptop GPU"

    def memory_allocated(self, device):
        return 612 * MiB

    def max_memory_allocated(self, device):
        return 700 * MiB


class FakeEmbedder:
    dtype = "float16"

    def __init__(self):
        self.calls = []

    def encode_queries(self, texts):
        self.calls.append(("queries", list(texts)))
        return [fake_vector(t) for t in texts]

    def encode_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        return [fake_vector(t) for t in texts]


class FakeReranker:
    dtype = "float16"

    def score(self, query, texts):
        return np.arange(len(texts), dtype=np.float32)


def make_engine(**kwargs):
    torch_module = types.SimpleNamespace(cuda=FakeCuda(**kwargs))
    return GpuEngine(FakeEmbedder(), FakeReranker(), "cuda:0", torch_module), torch_module.cuda


def test_engine_embeds_queries_with_encode_queries_and_documents_with_encode_documents():
    engine, _ = make_engine()
    engine.embed(["a"], True)
    engine.embed(["b"], False)
    assert engine.embedder.calls == [("queries", ["a"]), ("documents", ["b"])]


def test_engine_score_delegates_to_the_reranker():
    engine, _ = make_engine()
    assert engine.score("q", ["a", "b", "c"]).tolist() == [0.0, 1.0, 2.0]


def test_engine_reports_device_versions_and_vram_in_mib():
    engine, _ = make_engine()
    info = engine.info()
    assert info["device"] == "NVIDIA GeForce RTX 3070 Laptop GPU"
    assert info["vram"] == {"total_mib": 8192.0, "used_mib": 1192.0, "free_mib": 7000.0, "torch_allocated_mib": 612.0, "torch_peak_mib": 700.0}
    assert info["models"]["embedder"] == {"name": MODEL_NAME, "revision": MODEL_REVISION, "dim": 384, "dtype": "float16"}
    assert info["models"]["reranker"] == {"name": RERANKER_NAME, "revision": RERANKER_REVISION, "dtype": "float16"}


def test_engine_maps_a_cuda_out_of_memory_to_gpu_out_of_memory_and_gives_the_cache_back():
    engine, cuda = make_engine()
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})

    def oom(texts):
        raise OutOfMemoryError("CUDA out of memory")

    engine.embedder.encode_queries = oom
    with pytest.raises(GpuOutOfMemory):
        engine.embed(["a"], True)
    assert "empty_cache" in cuda.calls


def test_engine_lets_any_other_error_through_unchanged_and_does_not_touch_the_cache():
    engine, cuda = make_engine()

    def broken(texts):
        raise ValueError("a real bug")

    engine.embedder.encode_documents = broken
    with pytest.raises(ValueError, match="a real bug"):
        engine.embed(["a"], False)
    assert cuda.calls == []


def test_close_drops_the_models_then_collects_garbage_then_empties_the_cuda_cache_in_that_order(monkeypatch):
    engine, cuda = make_engine()
    order = cuda.calls
    monkeypatch.setattr(api.gc, "collect", lambda *a: order.append("gc"))
    engine.close()
    assert order == ["gc", "synchronize", "empty_cache"] and engine.embedder is None and engine.reranker is None


def test_close_really_lets_go_of_the_models():
    engine, _ = make_engine()
    probe = weakref.ref(engine.embedder)
    engine.close()
    gc.collect()
    assert probe() is None


def test_close_skips_the_cuda_calls_when_there_is_no_gpu():
    engine, cuda = make_engine(available=False)
    engine.close()
    assert cuda.calls == []


# --------------------------------------------------------------------------- a real server: loopback socket, Windows
# event loop, clean shutdown ---------------------------------------------------------------------------


def start_server(application, port=0):
    server = api.make_server(application, "127.0.0.1", port, "warning")
    thread = threading.Thread(target=api.run_server, args=(server,), daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "the server did not start"
    return server, thread, server.servers[0].sockets[0].getsockname()


def test_a_real_server_listens_on_loopback_only_answers_over_a_socket_and_shuts_down_cleanly(engine):
    application = api.create_app(lambda: engine)

    @application.get("/_loop")
    async def loop_name():
        return {"loop": type(asyncio.get_running_loop()).__name__}

    server, thread, (host, port) = start_server(application)
    try:
        assert host == "127.0.0.1"
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as http:
            assert http.get("/health").json()["status"] == "ok"
            assert http.post("/embed", json={"texts": ["housing"]}).json()["count"] == 1
            assert "Selector" in http.get("/_loop").json()["loop"]  # never the Proactor loop (project rule, section 7)
    finally:
        server.should_exit = True
        thread.join(15)
    assert not thread.is_alive() and engine.closed


def test_ctrl_c_shuts_the_service_down_cleanly_and_does_not_leave_a_traceback(engine):
    # uvicorn re-raises the signal that stopped it once shutdown is done (uvicorn/server.py, capture_signals).
    server = api.make_server(api.create_app(lambda: engine), "127.0.0.1", 0, "warning")

    def press_ctrl_c():
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        _thread.interrupt_main(signal.SIGINT)

    threading.Thread(target=press_ctrl_c, daemon=True).start()
    try:
        api.run_server(server)
    except KeyboardInterrupt:  # an escaping one would otherwise abort the whole pytest session
        pytest.fail("Ctrl+C escaped run_server after the shutdown: the owner would see a traceback")
    assert engine.closed


# --------------------------------------------------------------------------- the command line
# ---------------------------------------------------------------------------

GOOD_MACHINE = {
    "torch_version": "2.11.0+cu128",
    "cuda_version": "12.8",
    "cuda_available": True,
    "device_name": "NVIDIA GeForce RTX 3070 Laptop GPU",
    "compute_capability": (8, 6),
    "total_mib": 8192,
    "free_mib": 7400,
}


@pytest.fixture
def launched(monkeypatch):
    """Replace the parts of main that would load a GPU or bind a port, and record what it tried to do."""
    record = types.SimpleNamespace(served=[], machine=dict(GOOD_MACHINE), started=True)
    monkeypatch.setattr(check_cuda, "inspect_environment", lambda: record.machine)

    def fake_make_server(application, host, port, log_level):
        record.served.append((host, port, log_level))
        return types.SimpleNamespace(started=record.started)

    monkeypatch.setattr(api, "make_server", fake_make_server)
    monkeypatch.setattr(api, "run_server", lambda server: None)
    return record


def test_main_serves_on_loopback_port_8001_by_default(launched):
    assert api.main([]) == 0 and launched.served == [("127.0.0.1", 8001, "info")]


def test_main_passes_the_port_and_log_level_through(launched):
    assert api.main(["--port", "9100", "--log-level", "warning"]) == 0 and launched.served == [("127.0.0.1", 9100, "warning")]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "example.com"])
def test_main_refuses_to_listen_anywhere_but_loopback(launched, capsys, host):
    assert api.main(["--host", host]) == 2
    assert launched.served == [] and "refusing" in capsys.readouterr().err


def test_main_stops_with_the_preflight_message_when_cuda_is_unusable(launched, capsys):
    launched.machine = {**GOOD_MACHINE, "torch_version": "2.11.0+cpu", "cuda_version": None, "cuda_available": False, "device_name": None}
    assert api.main([]) == 2
    err = capsys.readouterr().err
    assert launched.served == [] and "CPU-only" in err and "cu128" in err


def test_main_warns_but_still_starts_on_a_different_gpu(launched, capsys):
    launched.machine = {**GOOD_MACHINE, "device_name": "NVIDIA GeForce RTX 4090"}
    assert api.main([]) == 0 and "3070" in capsys.readouterr().err and launched.served


def test_main_reports_a_startup_failure_with_exit_code_3(launched):
    launched.started = False
    assert api.main([]) == 3


# --------------------------------------------------------------------------- the real models on the GPU, through the
# same HTTP surface ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpu_client():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    with WireClient(api.create_app(), base_url=BASE, raise_server_exceptions=False) as c:  # the default factory: the real models
        yield c


HOUSING_Q = "What has the Government said about affordable housing?"
NHS_RELEVANT = "The Government are investing billions of pounds in the NHS to cut waiting lists for hospital treatment."
RECIPE = "The recipe calls for two eggs, a cup of flour and a pinch of salt."


def test_gpu_health_reports_the_real_card_models_and_headroom(gpu_client):
    body = gpu_client.get("/health").json()
    assert "3070" in body["device"] and body["status"] == "ok"
    vram = body["vram"]
    assert abs(vram["used_mib"] + vram["free_mib"] - vram["total_mib"]) < 1
    assert vram["torch_allocated_mib"] < 1500 and vram["free_mib"] >= MIN_FREE_VRAM_MIB and vram["headroom_ok"]
    assert body["models"]["embedder"]["revision"] == MODEL_REVISION and body["models"]["reranker"]["revision"] == RERANKER_REVISION
    assert body["models"]["embedder"]["dtype"] == body["models"]["reranker"]["dtype"] == "float16"


def test_gpu_embed_gives_384_finite_unit_vectors(gpu_client):
    body = gpu_client.post("/embed", json={"texts": [HOUSING_Q, NHS_RELEVANT, RECIPE]}).json()
    assert body["dim"] == 384 and body["count"] == 3 and body["skipped"] == []
    matrix = np.array(body["embeddings"], dtype=np.float64)
    assert matrix.shape == (3, 384) and np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(matrix, matrix.astype(np.float32).astype(np.float64))  # exactly float32 values


def test_gpu_embed_applies_the_bge_instruction_to_queries_and_not_to_documents(gpu_client):
    engine = gpu_client.app.state.engine

    def vec(**payload):
        return np.array(gpu_client.post("/embed", json={"texts": [HOUSING_Q], **payload}).json()["embeddings"][0])

    query, document = vec(is_query=True), vec(is_query=False)
    assert float(query @ document) < 0.999  # the instruction changes the vector
    assert float(query @ engine.embedder.encode_documents([QUERY_INSTRUCTION + HOUSING_Q])[0]) > 0.9999  # by exactly this prefix
    assert float(query @ engine.embedder.encode_queries([HOUSING_Q])[0]) > 0.9999  # the API path is Embedder.encode_queries
    assert float(document @ engine.embedder.encode_documents([HOUSING_Q])[0]) > 0.9999


def test_gpu_embed_related_text_is_closer_than_unrelated_text(gpu_client):
    q, related, unrelated = np.array(
        gpu_client.post("/embed", json={"texts": ["NHS waiting lists", NHS_RELEVANT, RECIPE], "is_query": False}).json()["embeddings"]
    )
    assert float(q @ related) > float(q @ unrelated) + 0.1


def test_gpu_embed_survives_dirty_and_extreme_input(gpu_client):
    texts = [
        "abc\x00def", "abc \ud800 def", "\U0001F600\U0001F600", "", None, "  spaced   out  ", "‮abc", "abc\x01\x1b[31m def",
        "مرحبا بالعالم", "你好世界，议会辩论", "a" * 1_000_000, "word " * 100_000,
    ]  # fmt: skip
    response = gpu_client.post("/embed", json={"texts": texts})
    body = response.json()
    assert response.status_code == 200 and body["skipped"] == [2, 3, 4] and body["count"] == len(texts)
    assert np.allclose(norms(body["embeddings"]), 1.0, atol=1e-5)


def test_gpu_embed_a_full_request_of_maximum_length_texts_stays_within_the_vram_budget(gpu_client):
    import torch

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    texts = [(f"parliament debate number {i} " * 800)[:MAX_TEXT_CHARS] for i in range(MAX_EMBED_TEXTS)]  # each far beyond 512 tokens
    response = gpu_client.post("/embed", json={"texts": texts})
    assert response.status_code == 200 and response.json()["skipped"] == []
    assert (torch.cuda.max_memory_allocated() - before) / MiB < 1200  # measured +579 MiB


def test_gpu_rerank_puts_the_relevant_passage_first_with_a_wide_margin(gpu_client):
    body = gpu_client.post("/rerank", json={"query": "What is being done about NHS waiting lists?", "candidates": [RECIPE, "Rail fares rose in March.", NHS_RELEVANT]}).json()
    assert [r["index"] for r in body["results"]][0] == 2
    assert body["results"][0]["score"] - body["results"][-1]["score"] > 5
    assert all(np.isfinite(r["score"]) for r in body["results"])


def test_gpu_rerank_survives_dirty_empty_and_over_length_candidates(gpu_client):
    candidates = [NHS_RELEVANT, "", None, "\x00", "abc \ud800 def", "😀", "waiting lists " * 5_000, "x" * 500_000, RECIPE, NHS_RELEVANT]
    response = gpu_client.post("/rerank", json={"query": "NHS waiting lists", "candidates": candidates})
    body = response.json()
    assert response.status_code == 200 and body["skipped"] == [1, 2, 3, 5]
    assert sorted(r["index"] for r in body["results"]) == [0, 4, 6, 7, 8, 9]
    assert all(np.isfinite(r["score"]) for r in body["results"])
    assert body["results"][0]["index"] in (0, 6, 9)  # a relevant passage wins


def test_gpu_rerank_survives_dirty_and_over_length_queries(gpu_client):
    for query in ("abc \ud800 NHS waiting lists", "NHS\x00 waiting\x00 lists", "NHS waiting lists " * 5_000, "x" * 500_000, "مرحبا NHS"):
        response = gpu_client.post("/rerank", json={"query": query, "candidates": [NHS_RELEVANT, RECIPE]})
        assert response.status_code == 200 and response.json()["count"] == 2, query[:30]


def test_gpu_rerank_a_full_request_of_maximum_length_candidates_stays_within_the_vram_budget(gpu_client):
    import torch

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    candidates = [(f"parliament debate number {i} " * 800)[:MAX_TEXT_CHARS] for i in range(MAX_RERANK_CANDIDATES)]
    response = gpu_client.post("/rerank", json={"query": "parliament debate", "candidates": candidates})
    assert response.status_code == 200 and response.json()["count"] == MAX_RERANK_CANDIDATES
    assert (torch.cuda.max_memory_allocated() - before) / MiB < 600  # 16 batches of 32 pairs, one at a time


def test_gpu_a_mixed_burst_from_many_threads_all_succeeds_and_stays_small(gpu_client):
    import torch
    from concurrent.futures import ThreadPoolExecutor

    def one(i):
        if i % 2:
            return gpu_client.post("/embed", json={"texts": [f"question number {i} about housing"]}).status_code
        return gpu_client.post("/rerank", json={"query": HOUSING_Q, "candidates": [f"answer {i}", NHS_RELEVANT, RECIPE]}).status_code

    torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(8) as pool:
        statuses = list(pool.map(one, range(40)))
    assert set(statuses) <= {200, 503} and statuses.count(200) >= 30  # a burst may be turned away, never broken
    assert gpu_client.get("/health").json()["queue"]["queued"] == 0
    assert torch.cuda.max_memory_allocated() / MiB < 2000


def test_gpu_shutdown_gives_the_vram_back():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    server, thread, (_, port) = start_server(api.create_app())  # loads both real models on the GPU thread
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as http:
            assert http.post("/embed", json={"texts": ["housing"]}).status_code == 200
            assert http.post("/rerank", json={"query": "housing", "candidates": ["homes"]}).status_code == 200
        loaded = torch.cuda.memory_allocated()
    finally:
        server.should_exit = True
        thread.join(30)
    assert not thread.is_alive()
    assert loaded - baseline > 300 * MiB  # the models really were resident...
    assert torch.cuda.memory_allocated() - baseline < 50 * MiB  # ...and are gone after shutdown
