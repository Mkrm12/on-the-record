"""The real corpus, the real HNSW index and the real GPU service."""

import asyncio
import time

import httpx
import pytest
from dbhelpers import connect, scalar

from app.config import settings
from app.db import make_pool, to_vector
from app.gold import load as load_gold
from ingest.runtime import run
from retrieval.gpu_client import GpuClient
from retrieval.hybrid import HybridSearcher

GOLD = {q.id: q for q in load_gold()}


@pytest.fixture(scope="module", autouse=True)
def live():
    conn = connect()
    if conn is None:
        pytest.skip("Postgres is not reachable")
    if scalar(conn, "SELECT count(*) FROM chunk WHERE embedding IS NOT NULL") < 1000:
        pytest.skip("the corpus is not embedded (run the embedder)")
    conn.close()
    try:
        httpx.get(f"{settings.gpu_url}/health", timeout=2).raise_for_status()
    except (httpx.HTTPError, OSError):
        pytest.skip("the GPU service is not running (cd gpu; uv run python api.py)")


def search_live(scenario, gpu_url=None, **pool_kwargs):
    async def main():
        pool = make_pool(min_size=1, max_size=4, **pool_kwargs)
        await pool.open(wait=True, timeout=10)
        try:
            async with GpuClient(gpu_url) as gpu:
                searcher = HybridSearcher(pool, gpu)
                await searcher.refresh()
                return await scenario(searcher, pool)
        finally:
            await pool.close()

    return run(main())


def relevant(qid):
    return {r.speech_id for r in GOLD[qid].relevant}


@pytest.mark.parametrize("qid", ["g001", "g002", "g013", "g017", "g018"])
def test_known_questions_are_answered_at_the_top_with_no_warnings_and_within_the_latency_budget(qid):
    async def scenario(searcher, pool):
        # Untimed, and a DIFFERENT question, so the timed one's own pages are not pre-cached: a fresh pool's first
        # search pays connection set-up, plan caching and cold index pages.
        await searcher.search("What was said about school funding?", top_k=10)
        started = time.perf_counter()
        result = await searcher.search(GOLD[qid].question, top_k=10)
        return result, time.perf_counter() - started

    result, seconds = search_live(scenario)
    assert result.warnings == [] and result.hits
    assert result.hits[0].speech_id in relevant(qid) or any(h.speech_id in relevant(qid) for h in result.hits[:3])
    assert seconds < 3.0


def test_a_named_speaker_question_resolves_the_member_and_uses_the_speaker_leg():
    result = search_live(lambda s, p: s.search(GOLD["g017"].question, top_k=10))
    assert result.diagnostics["speakers"]["mentions"] == ["Graham Stuart"]
    assert result.diagnostics["legs"]["speaker"] > 0 and any(h.legs["speaker"] for h in result.hits)
    assert any(h.speaker == "Graham Stuart" for h in result.hits[:3])


def test_every_hit_has_what_a_citation_needs_and_is_never_boilerplate():
    async def scenario(searcher, pool):
        result = await searcher.search(GOLD["g004"].question, top_k=10)
        async with pool.connection() as conn:
            flagged = await (await conn.execute("SELECT count(*) FROM chunk WHERE id = ANY(%s) AND is_boilerplate", ([h.chunk_id for h in result.hits],))).fetchone()
        return result, flagged[0]

    result, flagged = search_live(scenario)
    assert flagged == 0
    for h in result.hits:
        assert h.speech_id and h.debate_id and h.sentence and h.passage and h.heading and h.date and h.house
        assert h.context and h.sentence in h.context  # the expansion always contains the matched sentence itself


def test_the_dense_leg_really_uses_the_hnsw_index_and_the_settings_are_gone_afterwards():
    """The PRODUCTION dense leg (hybrid._DENSE_HNSW, in the form the frozen file names), not a hand-copied query: a
    copy can keep passing after the real ORDER BY stops matching the index — the failure is a silent sequential
    scan, 10x slower and blind to ef_search.
    """
    from answer.pipeline import Frozen
    from retrieval.hybrid import _DENSE_HNSW
    from retrieval.vector_index import order_expression

    form = Frozen.load().search_config().dense_index
    leg = _DENSE_HNSW.format(filter="", order=order_expression(form, alias="c")).rstrip().rstrip(",")

    async def scenario(searcher, pool):
        vector = to_vector(await searcher.gpu.embed_query("housing benefit rules"))
        async with pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL hnsw.ef_search = 100")
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                plan = await (await conn.execute(f"EXPLAIN WITH{leg} SELECT id FROM dense",
                                                 {"qvec": vector, "dense_limit": 50})).fetchall()
            after = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            await conn.rollback()
        return "\n".join(r[0] for r in plan), after

    plan, after = search_live(scenario)
    assert "Index Scan using chunk_embedding_idx" in plan, plan
    assert after == "40"


def test_a_search_leaves_no_setting_behind_on_a_reused_connection():
    async def scenario(searcher, pool):
        await searcher.search(GOLD["g001"].question)
        async with pool.connection() as conn:
            values = [(await (await conn.execute(f"SHOW {name}")).fetchone())[0] for name in ("hnsw.ef_search", "hnsw.iterative_scan")]
            await conn.rollback()
        return values

    assert search_live(scenario) == ["40", "off"]


def test_with_the_gpu_service_down_a_search_still_answers_from_keywords_quickly_and_says_so():
    started = time.perf_counter()
    result = search_live(lambda s, p: s.search("What did the Minister say about the Immigration Act 1971?", top_k=5), gpu_url="http://127.0.0.1:9")
    assert result.hits and any("embedded" in w or "unavailable" in w for w in result.warnings)
    assert result.diagnostics["dense"] == "unavailable" and all(h.legs["dense"] is None for h in result.hits)
    assert time.perf_counter() - started < 10


def test_every_gold_question_runs_without_error_and_the_answerable_ones_return_results():
    async def scenario(searcher, pool):
        return [(q, await searcher.search(q.question, top_k=10)) for q in GOLD.values()]

    for q, result in search_live(scenario):
        assert not any("could not" in w for w in result.warnings), (q.id, result.warnings)
        if not q.expected_abstain:
            assert result.hits, q.id


def test_concurrent_searches_on_the_live_corpus_do_not_interfere():
    async def scenario(searcher, pool):
        ids = ["g001", "g002", "g004", "g013", "g014", "g015", "g017", "g018"]
        return ids, await asyncio.gather(*[searcher.search(GOLD[i].question, top_k=5) for i in ids])

    ids, results = search_live(scenario)
    alone = search_live(lambda s, p: s.search(GOLD["g013"].question, top_k=5))
    assert [h.chunk_id for h in results[ids.index("g013")].hits] == [h.chunk_id for h in alone.hits]  # same answer with or without company
    assert all(r.hits for r in results)


def test_hostile_and_huge_queries_do_not_break_the_live_search():
    queries = ["'; DROP TABLE chunk; --", "a & b | !c", "😀😀😀", "x" * 5000, "housing " * 3000, "\x00\ud800 NHS waiting lists"]

    async def scenario(searcher, pool):
        return [await searcher.search(q, top_k=3) for q in queries]

    results = search_live(scenario)
    assert len(results) == len(queries) and results[2].hits == [] and results[5].hits  # emoji only: nothing; a dirty NHS query: answered
