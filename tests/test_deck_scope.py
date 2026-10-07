"""The deck scales with the question's scope, and no route is silently capped below what it asks for."""

from __future__ import annotations

import pytest

from answer import cards as C
from answer import pipeline as PL
from answer import routes as RT
from ingest.runtime import run
from retrieval.hybrid import SearchConfig, SearchResult


@pytest.mark.parametrize("route, dated, want", [
    ("retrieval", True, (10, C.DEFAULT_TOKEN_BUDGET)),
    ("retrieval", False, (PL.UNDATED_TOP_K, C.DEFAULT_TOKEN_BUDGET)),
    ("overview", True, (RT.OVERVIEW_POOL, PL.OVERVIEW_TOKEN_BUDGET)),
    ("overview", False, (RT.OVERVIEW_POOL, PL.OVERVIEW_TOKEN_BUDGET)),
])  # fmt: skip
def test_the_deck_size_follows_the_scope(route, dated, want):
    assert PL.deck_size(route, dated=dated) == want


def test_an_undated_question_never_gets_fewer_cards_than_the_caller_asked_for():
    assert PL.deck_size("retrieval", dated=False, top_k=30) == (30, C.DEFAULT_TOKEN_BUDGET)


def test_an_undated_deck_is_never_shallower_than_a_dated_one():
    assert PL.deck_size("retrieval", dated=False)[0] >= PL.deck_size("retrieval", dated=True)[0]


def test_the_undated_depth_is_the_frozen_depth_until_a_run_shows_a_gain():
    assert PL.UNDATED_TOP_K == 10


def test_the_overview_budget_holds_the_overview_pool():
    assert RT.OVERVIEW_POOL * 235 <= PL.OVERVIEW_TOKEN_BUDGET


class _Frozen:
    def __init__(self, pool):
        self._pool = pool

    def search_config(self):
        return SearchConfig(pool=self._pool)


class _Searcher:
    def __init__(self):
        self.configs = []

    async def search(self, question, *, top_k, config):
        self.configs.append((top_k, config))
        return SearchResult(query=question, searched=question)


class _Runtime:
    def __init__(self, pool=20):
        self.frozen, self.searcher, self.serving_timeout_ms = _Frozen(pool), _Searcher(), None


@pytest.mark.parametrize("top_k, want_pool", [(10, 20), (20, 20), (60, 60), (100, 100)])
def test_the_reranker_pool_is_raised_only_when_a_caller_asks_for_more(top_k, want_pool):
    runtime = _Runtime(pool=20)
    run(PL.search_deck(runtime, "q", top_k))
    (asked, config), = runtime.searcher.configs
    assert asked == top_k and config.pool == want_pool


def test_the_frozen_config_object_itself_is_never_changed():
    runtime = _Runtime(pool=20)
    run(PL.search_deck(runtime, "q", 60))
    assert runtime.frozen.search_config().pool == 20


def test_the_token_budget_reaches_the_deck():
    runtime = _Runtime()
    deck, _ = run(PL.search_deck(runtime, "q", 10, token_budget=1234))
    assert deck.budget_tokens == 1234
