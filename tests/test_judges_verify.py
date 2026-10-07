"""A judge's model id is checked against the provider's live /models list before a run spends anything."""

from __future__ import annotations

import httpx
import pytest

from goldset import judges as J

SECRET = "sk-test-not-a-real-key-7f3a"


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None):
        self.status_code, self._payload = status, payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid/models")
            raise httpx.HTTPStatusError(f"{self.status_code}", request=request,
                                        response=httpx.Response(self.status_code, request=request))

    def json(self):
        return self._payload


@pytest.fixture
def listed(monkeypatch):
    """Any OpenAI-compatible provider: the check is the same for all of them."""
    provider = J.BY_NAME["deepseek"]
    monkeypatch.setattr(type(provider), "key", lambda self: SECRET)
    return provider


def answer_with(monkeypatch, response=None, raises=None):
    seen = {}

    def fake_get(url, *, timeout, headers):
        seen.update(url=url, headers=headers)
        if raises:
            raise raises
        return response

    monkeypatch.setattr(httpx, "get", fake_get)
    return seen


def test_a_listed_model_passes(listed, monkeypatch):
    answer_with(monkeypatch, FakeResponse(200, {"data": [{"id": "other"}, {"id": listed.model}]}))
    assert J.verify_model(listed) is None


def test_a_retired_model_is_refused_with_the_reason(listed, monkeypatch):
    answer_with(monkeypatch, FakeResponse(200, {"data": [{"id": "something-else"}]}))
    reason = J.verify_model(listed)
    assert reason and listed.model in reason and "retired" in reason


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_a_list_that_cannot_be_read_is_a_refusal_not_a_pass(listed, monkeypatch, status):
    """A check that could not run has not passed."""
    answer_with(monkeypatch, FakeResponse(status))
    assert J.verify_model(listed) is not None


@pytest.mark.parametrize("error", [httpx.ConnectError("down"), httpx.ReadTimeout("slow"), ValueError("not json")])
def test_a_network_or_parse_failure_is_a_refusal(listed, monkeypatch, error):
    answer_with(monkeypatch, raises=error)
    assert "could not list" in (J.verify_model(listed) or "")


def test_an_empty_list_is_a_refusal(listed, monkeypatch):
    answer_with(monkeypatch, FakeResponse(200, {"data": []}))
    assert J.verify_model(listed) is not None


def test_the_gemini_models_prefix_is_ignored(monkeypatch):
    provider = J.BY_NAME["google"]
    monkeypatch.setattr(type(provider), "key", lambda self: SECRET)
    answer_with(monkeypatch, FakeResponse(200, {"data": [{"id": f"models/{provider.model}"}]}))
    assert J.verify_model(provider) is None


def test_the_key_travels_only_in_the_header_and_never_in_a_reason(listed, monkeypatch):
    seen = answer_with(monkeypatch, FakeResponse(200, {"data": [{"id": "x"}]}))
    reason = J.verify_model(listed)
    assert seen["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in (reason or "") and SECRET not in seen["url"]


def test_the_url_is_the_providers_models_endpoint(listed, monkeypatch):
    seen = answer_with(monkeypatch, FakeResponse(200, {"data": [{"id": listed.model}]}))
    J.verify_model(listed)
    assert seen["url"] == f"{listed.base.rstrip('/')}/models"


def test_the_evaluation_refuses_a_dead_judge_before_it_spends(monkeypatch):
    """The wiring: `eval_answers.run` checks the primary judge before building the runtime or asking anything."""
    import importlib.util
    from pathlib import Path

    from ingest.runtime import run

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_answers.py"
    spec = importlib.util.spec_from_file_location("eval_answers_d02", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.J, "panel", lambda **k: [J.BY_NAME["google"]])
    monkeypatch.setattr(module.J, "primary", lambda available, **k: J.BY_NAME["google"])
    monkeypatch.setattr(module.J, "verify_model", lambda provider: "'x' is not on google's live /models list")

    async def must_not_build(*a, **k):
        raise AssertionError("the runtime was built before the judge was checked")

    monkeypatch.setattr(module.PL, "build_runtime", must_not_build)
    with pytest.raises(SystemExit, match="Nothing was spent"):
        run(module.run("dev", 1, "hansard_v2", True, 0.01, "t"))


# --- F4: an item whose gold is a placeholder is asked but never judged -------------------------------------------


def _eval_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_answers.py"
    spec = importlib.util.spec_from_file_location("eval_answers_f4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_placeholder_gold_is_recognised():
    EA = _eval_module()

    class N:
        def __init__(self, text):
            self.text = text

    class Item:
        def __init__(self, *texts):
            self.nuggets = tuple(N(t) for t in texts)

    assert EA.gold_incomplete(Item("TO_VERIFY: nuggets were not recorded in v1."))
    assert not EA.gold_incomplete(Item("Wendy Morton said it."))
    assert not EA.gold_incomplete(Item())


def test_the_real_gold_has_exactly_the_placeholders_we_know_of():
    """22 dev items are placeholders. More appearing is a regression in the gold file; fewer means someone wrote
    nuggets, and this number should be lowered with them.
    """
    from app.gold import load  # noqa: F401 - the loader import proves the path, the count uses the v2 schema
    from pathlib import Path

    from goldset.schema import load as load_v2

    EA = _eval_module()
    items = load_v2(Path(__file__).resolve().parents[1] / "tests" / "data" / "gold_v2.json")
    assert sum(1 for i in items if EA.gold_incomplete(i)) <= 22


def test_the_judge_ceiling_fits_a_reasoning_model():
    assert _eval_module().JUDGE_MAX_TOKENS >= 2000
