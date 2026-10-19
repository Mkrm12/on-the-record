"""The async client for the local GPU service (gpu/api.py). Everything runs against httpx.MockTransport with a fake
clock and sleeper, so retries, back-off and the circuit breaker are tested instantly and exactly.
"""

import asyncio
import json

import httpx
import pytest
from pgvector import Vector

from retrieval.gpu_client import (
    CircuitBreaker,
    GpuBadRequest,
    GpuBadResponse,
    GpuBusy,
    GpuClient,
    GpuError,
    GpuServerError,
    GpuTimeout,
    GpuUnavailable,
)

DIM = 384


def unit(axis=0):
    v = [0.0] * DIM
    v[axis] = 1.0
    return v


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Sleeper:
    def __init__(self):
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)


def make(handler, **kwargs):
    sleeper, clock = Sleeper(), Clock()
    client = GpuClient("http://gpu.test", transport=httpx.MockTransport(handler), sleep=sleeper, clock=clock, **kwargs)
    return client, sleeper, clock


def run(coro):
    return asyncio.run(coro)


def json_response(body, status=200, headers=None):
    return httpx.Response(status, json=body, headers=headers)


def embed_ok(request):
    return json_response({"embeddings": [unit()], "dim": DIM, "count": 1, "skipped": []})


# --- embedding a query: the instruction must be applied ------------------------------------------------------------


def test_a_query_is_sent_with_is_query_true_so_bge_applies_its_retrieval_instruction():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        return embed_ok(request)

    client, _, _ = make(handler)
    run(client.embed_query("What did the minister say?"))
    assert seen == [("POST", "/embed", {"texts": ["What did the minister say?"], "is_query": True})]


def test_the_answer_is_a_vector_object_ready_for_a_pgvector_query():
    client, _, _ = make(embed_ok)
    v = run(client.embed_query("housing"))
    assert isinstance(v, Vector) and v.dimensions() == DIM and v.to_list()[0] == 1.0


@pytest.mark.parametrize("text", ["", "   ", None])
def test_a_blank_query_is_refused_locally_without_calling_the_service(text):
    calls = []
    client, _, _ = make(lambda r: calls.append(r) or embed_ok(r))
    with pytest.raises(ValueError):
        run(client.embed_query(text))
    assert calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [unit()[:100]], "dim": 100, "count": 1, "skipped": []},  # wrong size
        {"embeddings": [None], "dim": DIM, "count": 1, "skipped": [0]},  # the service found nothing embeddable
        {"embeddings": [], "dim": DIM, "count": 0, "skipped": []},
        {"embeddings": [unit(), unit()], "dim": DIM, "count": 2, "skipped": []},  # more than we asked for
        {"embeddings": [[2.0] + [0.0] * (DIM - 1)], "dim": DIM, "count": 1, "skipped": []},  # not unit length
        {"embeddings": [["x"] * DIM], "dim": DIM, "count": 1, "skipped": []},
        {"count": 1},
        [],
    ],
)
def test_a_malformed_embedding_answer_is_refused_rather_than_sent_to_the_database(body):
    client, _, _ = make(lambda r: json_response(body))
    with pytest.raises(GpuBadResponse):
        run(client.embed_query("housing"))


def test_a_non_finite_coordinate_is_refused_even_though_it_would_slip_past_a_length_check():
    # JSON has no NaN, but Python's parser accepts the token; NaN compares false with everything, so a norm test alone
    # passes it
    body = b'{"embeddings":[[NaN' + b",0.0" * (DIM - 1) + b']],"dim":384,"count":1,"skipped":[]}'
    client, _, _ = make(lambda r: httpx.Response(200, content=body, headers={"content-type": "application/json"}))
    with pytest.raises(GpuBadResponse, match="finite"):
        run(client.embed_query("housing"))


def test_a_non_json_answer_is_a_bad_response():
    client, _, _ = make(lambda r: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(GpuBadResponse):
        run(client.embed_query("housing"))


# --- reranking -----------------------------------------------------------------------------------------------------


def test_rerank_returns_scores_aligned_with_the_candidates_and_none_for_those_skipped():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/rerank" and body == {"query": "q", "candidates": ["a", "b", "c"]}
        return json_response({"results": [{"index": 2, "score": 3.5, "text": "c"}, {"index": 0, "score": -1.0, "text": "a"}], "count": 2, "skipped": [1]})

    client, _, _ = make(handler)
    assert run(client.rerank("q", ["a", "b", "c"])) == [-1.0, None, 3.5]


def test_rerank_of_nothing_makes_no_request():
    calls = []
    client, _, _ = make(lambda r: calls.append(r))
    assert run(client.rerank("q", [])) == [] and calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"results": [{"index": 5, "score": 1.0, "text": "x"}], "count": 1, "skipped": []},  # index out of range
        {"results": [{"index": 0, "score": 1.0}, {"index": 0, "score": 2.0}], "count": 2, "skipped": []},  # repeated index
        {"results": [{"index": 0, "score": "high"}], "count": 1, "skipped": []},
        {"results": [{"index": 0}], "count": 1, "skipped": []},
        {"results": "none"},
        {},
    ],
)
def test_a_malformed_rerank_answer_is_refused(body):
    client, _, _ = make(lambda r: json_response(body))
    with pytest.raises(GpuBadResponse):
        run(client.rerank("q", ["a", "b"]))


def test_a_non_finite_score_is_refused():
    client, _, _ = make(lambda r: httpx.Response(200, content=b'{"results":[{"index":0,"score":NaN,"text":"a"}],"count":1,"skipped":[]}',
                                                headers={"content-type": "application/json"}))
    with pytest.raises(GpuBadResponse):
        run(client.rerank("q", ["a"]))


# --- when the service is not there ----------------------------------------------------------------------------------


def test_connection_refused_is_gpu_unavailable_and_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("connection refused")

    client, sleeper, _ = make(handler)
    with pytest.raises(GpuUnavailable):
        run(client.embed_query("housing"))
    assert len(calls) == 1 and sleeper.calls == []


def test_a_connect_timeout_counts_as_unavailable_and_a_read_timeout_as_a_timeout():
    def connect_timeout(request):
        raise httpx.ConnectTimeout("no route")

    def read_timeout(request):
        raise httpx.ReadTimeout("gpu busy")

    with pytest.raises(GpuUnavailable):
        run(make(connect_timeout)[0].embed_query("x"))
    with pytest.raises(GpuTimeout):
        run(make(read_timeout)[0].embed_query("x"))
    assert issubclass(GpuTimeout, GpuUnavailable) and issubclass(GpuUnavailable, GpuError)


def test_a_dropped_connection_mid_request_is_unavailable_too():
    def handler(request):
        raise httpx.RemoteProtocolError("server disconnected")

    with pytest.raises(GpuUnavailable):
        run(make(handler)[0].embed_query("x"))


# --- when the service is busy ---------------------------------------------------------------------------------------


def test_a_503_is_retried_after_the_delay_the_service_asks_for_then_succeeds():
    answers = iter([json_response({"detail": "busy"}, 503, {"Retry-After": "1"}), embed_ok(None)])
    client, sleeper, _ = make(lambda r: next(answers))
    assert run(client.embed_query("housing")).dimensions() == DIM
    assert sleeper.calls == [1.0]


def test_a_503_without_retry_after_backs_off_exponentially_and_gives_up_as_busy():
    calls = []

    def handler(request):
        calls.append(request)
        return json_response({"detail": "busy"}, 503)

    client, sleeper, _ = make(handler, retries=2)
    with pytest.raises(GpuBusy):
        run(client.embed_query("housing"))
    assert len(calls) == 3 and sleeper.calls == [0.25, 0.5]


def test_the_wait_is_capped_so_a_huge_retry_after_cannot_stall_a_search():
    answers = iter([json_response({}, 503, {"Retry-After": "3600"}), embed_ok(None)])
    client, sleeper, _ = make(lambda r: next(answers), max_backoff=2.0)
    run(client.embed_query("housing"))
    assert sleeper.calls == [2.0]


def test_a_nonsense_retry_after_falls_back_to_the_default_backoff():
    answers = iter([json_response({}, 503, {"Retry-After": "tomorrow"}), embed_ok(None)])
    client, sleeper, _ = make(lambda r: next(answers))
    run(client.embed_query("housing"))
    assert sleeper.calls == [0.25]


def test_no_retries_configured_means_one_attempt():
    calls = []
    client, sleeper, _ = make(lambda r: calls.append(r) or json_response({}, 503), retries=0)
    with pytest.raises(GpuBusy):
        run(client.embed_query("x"))
    assert len(calls) == 1 and sleeper.calls == []


# --- other failures ------------------------------------------------------------------------------------------------


def test_a_422_is_a_bad_request_and_a_500_a_server_error_and_neither_is_retried():
    calls = []
    client, _, _ = make(lambda r: calls.append(r) or json_response({"detail": "nope"}, 422))
    with pytest.raises(GpuBadRequest):
        run(client.embed_query("x"))
    client, _, _ = make(lambda r: calls.append(r) or json_response({"detail": "boom"}, 500))
    with pytest.raises(GpuServerError):
        run(client.embed_query("x"))
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 404, 405, 411, 413])
def test_other_client_errors_are_bad_requests(status):
    client, _, _ = make(lambda r: json_response({}, status))
    with pytest.raises(GpuBadRequest):
        run(client.embed_query("x"))


def test_error_messages_never_contain_the_users_query():
    secret = "MY-PRIVATE-QUESTION-ABOUT-A-NAMED-PERSON"
    for handler in (
        lambda r: json_response({"detail": secret}, 422),
        lambda r: json_response({"detail": secret}, 500),
        lambda r: json_response({"embeddings": [None], "dim": DIM, "count": 1, "skipped": [0]}),
    ):
        with pytest.raises(GpuError) as info:
            run(make(handler)[0].embed_query(secret))
        assert secret not in str(info.value)


# --- the circuit breaker ---------------------------------------------------------------------------------------------


def test_after_repeated_failures_calls_fail_fast_without_touching_the_service_until_the_cooldown_passes():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("refused")

    sleeper, clock = Sleeper(), Clock()
    client = GpuClient("http://gpu.test", transport=httpx.MockTransport(handler), sleep=sleeper, clock=clock, breaker_threshold=3, breaker_cooldown=10.0)
    for _ in range(3):
        with pytest.raises(GpuUnavailable):
            run(client.embed_query("x"))
    assert len(calls) == 3
    with pytest.raises(GpuUnavailable, match="circuit"):
        run(client.embed_query("x"))
    assert len(calls) == 3  # the fourth never left the process
    clock.now += 9.9
    with pytest.raises(GpuUnavailable, match="circuit"):
        run(client.embed_query("x"))
    assert len(calls) == 3
    clock.now += 0.2  # cooldown over: one trial goes through
    with pytest.raises(GpuUnavailable):
        run(client.embed_query("x"))
    assert len(calls) == 4
    with pytest.raises(GpuUnavailable, match="circuit"):  # and it failed, so the breaker is open again
        run(client.embed_query("x"))
    assert len(calls) == 4


def test_a_successful_trial_closes_the_breaker():
    state = {"up": False}
    calls = []

    def handler(request):
        calls.append(request)
        if not state["up"]:
            raise httpx.ConnectError("refused")
        return embed_ok(request)

    sleeper, clock = Sleeper(), Clock()
    client = GpuClient("http://gpu.test", transport=httpx.MockTransport(handler), sleep=sleeper, clock=clock, breaker_threshold=2, breaker_cooldown=5.0)
    for _ in range(2):
        with pytest.raises(GpuUnavailable):
            run(client.embed_query("x"))
    state["up"] = True
    clock.now += 6
    assert run(client.embed_query("x")).dimensions() == DIM  # trial succeeds
    assert run(client.embed_query("x")).dimensions() == DIM and len(calls) == 4  # and the breaker is closed again


def test_a_busy_service_or_a_bad_request_does_not_trip_the_breaker():
    codes = iter([503] * 12 + [422] * 5)
    calls = []
    client, _, _ = make(lambda r: calls.append(r) or json_response({}, next(codes)), retries=0, breaker_threshold=3)
    for _ in range(12):
        with pytest.raises(GpuBusy):
            run(client.embed_query("x"))
    for _ in range(5):
        with pytest.raises(GpuBadRequest):
            run(client.embed_query("x"))
    assert len(calls) == 17


def test_a_success_resets_the_failure_count():
    outcomes = iter(["fail", "fail", "ok", "fail", "fail", "ok"])

    def handler(request):
        if next(outcomes) == "fail":
            raise httpx.ConnectError("refused")
        return embed_ok(request)

    client, _, _ = make(handler, breaker_threshold=3)
    for expect_ok in (False, False, True, False, False, True):
        if expect_ok:
            run(client.embed_query("x"))
        else:
            with pytest.raises(GpuUnavailable, match="^(?!.*circuit)"):
                run(client.embed_query("x"))


def test_the_breaker_on_its_own():
    clock = Clock()
    b = CircuitBreaker(threshold=2, cooldown=4.0, clock=clock)
    assert b.allow()
    b.failure()
    assert b.allow()
    b.failure()
    assert not b.allow()
    clock.now += 4.0
    assert b.allow()  # half-open
    b.success()
    assert b.allow() and b.allow()


# --- housekeeping ----------------------------------------------------------------------------------------------------


def test_health_returns_the_services_report():
    client, _, _ = make(lambda r: json_response({"status": "ok", "queue": {"queued": 0}}))
    assert run(client.health())["status"] == "ok"


def test_a_trailing_slash_in_the_base_url_is_harmless_and_the_client_closes():
    seen = []

    async def go():
        async with GpuClient("http://gpu.test/", transport=httpx.MockTransport(lambda r: seen.append(str(r.url)) or embed_ok(r))) as client:
            await client.embed_query("x")

    run(go())
    assert seen == ["http://gpu.test/embed"]


def test_sensible_default_timeouts_bound_every_call():
    client = GpuClient("http://gpu.test")
    t = client.timeout
    assert 0 < t.connect <= 5 and 0 < t.read <= 60 and 0 < t.write <= 10 and 0 < t.pool <= 10
    run(client.aclose())


def test_the_default_url_is_the_configured_loopback_service():
    from app.config import settings

    assert settings.gpu_url == "http://127.0.0.1:8001" and str(GpuClient().base_url).startswith("http://127.0.0.1:8001")
