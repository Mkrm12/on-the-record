"""What retrieval costs at 1.2M chunks: latency, HNSW recall, filters, storage."""

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from psycopg.conninfo import make_conninfo
from rich.console import Console
from rich.table import Table

from app.config import settings
from app.db import make_pool, to_vector
from app.gold import load as load_gold
from ingest.runtime import run
from retrieval.gpu_client import GpuClient
from retrieval.hybrid import HybridSearcher, SearchConfig, SearchFilters
from retrieval.lexicon import build_lexicon, load_lexicon
from retrieval.vector_index import frozen_form, order_expression

console = Console()

SIZES_SQL = """
    SELECT pg_total_relation_size('chunk'), pg_relation_size('chunk'),
           pg_relation_size('chunk_embedding_idx'), pg_relation_size('chunk_ts_idx'),
           pg_indexes_size('chunk'),
           pg_total_relation_size('speech'), pg_total_relation_size('debate'),
           pg_database_size(current_database())
"""

COUNTS_SQL = """
    SELECT count(*), count(*) FILTER (WHERE is_boilerplate),
           count(*) FILTER (WHERE embedding IS NOT NULL),
           count(DISTINCT window_id), count(DISTINCT speech_id)
    FROM chunk
"""

# An exact scan over every embedded chunk.
EXACT_SQL = """
    SELECT id FROM chunk
    WHERE embedding IS NOT NULL AND NOT is_boilerplate
    ORDER BY embedding <=> %(qvec)s::vector
    LIMIT %(k)s
"""

# The same statement through the index frozen retrieval queries.
HNSW_SQL = f"""
    SELECT id FROM chunk
    WHERE embedding IS NOT NULL AND NOT is_boilerplate
    ORDER BY {order_expression(frozen_form(), alias="", param="%(qvec)s")}
    LIMIT %(k)s
"""


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[index], 1)


@dataclass
class Measurements:
    database: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    sizes_mb: dict[str, float] = field(default_factory=dict)
    per_leg_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    hnsw_recall: dict[str, float] = field(default_factory=dict)
    exact_scan_ms: float = 0.0
    filtered: list[dict[str, Any]] = field(default_factory=list)
    pool_ceiling: dict[str, Any] = field(default_factory=dict)
    refresh_s: dict[str, float] = field(default_factory=dict)
    statement_timeout_hits: int = 0
    notes: list[str] = field(default_factory=list)


async def _sizes(conn) -> tuple[dict[str, int], dict[str, float]]:
    total, heap, hnsw, ts, indexes, speech, debate, database = \
        await (await conn.execute(SIZES_SQL)).fetchone()
    chunks, boilerplate, embedded, windows, speeches = \
        await (await conn.execute(COUNTS_SQL)).fetchone()
    mb = 1024 * 1024
    return ({"chunks": chunks, "boilerplate": boilerplate, "embedded": embedded,
             "windows": windows, "speeches": speeches},
            {"chunk_total": round(total / mb, 1), "chunk_heap": round(heap / mb, 1),
             "hnsw_index": round(hnsw / mb, 1), "ts_index": round(ts / mb, 1),
             "chunk_indexes": round(indexes / mb, 1), "speech_total": round(speech / mb, 1),
             "debate_total": round(debate / mb, 1), "database": round(database / mb, 1)})


async def _hnsw_recall(conn, vectors, k: int, ef_values: tuple[int, ...]) -> tuple[dict[str, float], float]:
    """Recall of the index against an exact scan, at several ef_search values."""
    exact: list[set[int]] = []
    exact_ms: list[float] = []
    for vector in vectors:
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL statement_timeout = 120000")
            started = time.perf_counter()
            rows = await (await conn.execute(EXACT_SQL, {"qvec": vector, "k": k})).fetchall()
            exact_ms.append((time.perf_counter() - started) * 1000)
        exact.append({r[0] for r in rows})

    recall: dict[str, float] = {}
    for ef in ef_values:
        hits = total = 0
        for vector, truth in zip(vectors, exact):
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef)}")
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                rows = await (await conn.execute(HNSW_SQL, {"qvec": vector, "k": k})).fetchall()
            hits += len({r[0] for r in rows} & truth)
            total += len(truth)
        recall[f"ef_search_{ef}"] = round(hits / total, 4) if total else 0.0
    return recall, round(statistics.median(exact_ms), 1)


async def _measure(args) -> Measurements:
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database) if args.database else settings.pg_dsn
    out = Measurements(database=args.database or "(PG_DSN)")
    gold = load_gold()
    questions = [item.question for item in gold][: args.queries]

    pool = make_pool(dsn, min_size=1, max_size=4)
    await pool.open(wait=True)
    gpu = GpuClient(settings.gpu_url)
    try:
        async with pool.connection() as conn:
            out.counts, out.sizes_mb = await _sizes(conn)

        searcher = HybridSearcher(pool, gpu)
        # --- per-leg latency, each leg alone and then the whole pipeline ---
        legs = {
            "dense_only": SearchConfig(sparse=False, speaker=False, rerank="off", expand_radius=0),
            "keyword_only": SearchConfig(dense=False, speaker=False, rerank="off", expand_radius=0),
            "fused_no_rerank": SearchConfig(rerank="off", expand_radius=0),
            "full": SearchConfig(),
        }
        for name, cfg in legs.items():
            timings: dict[str, list[float]] = {}
            for question in questions:
                result = await searcher.search(question, top_k=10, config=cfg)
                for stage, value in (result.diagnostics.get("timings_ms") or {}).items():
                    timings.setdefault(stage, []).append(value)
                if any("timeout" in w.lower() for w in result.warnings):
                    out.statement_timeout_hits += 1
            out.per_leg_ms[name] = {f"{stage}_p50": pct(v, 0.5) for stage, v in timings.items()}
            out.per_leg_ms[name].update({f"{stage}_p95": pct(v, 0.95) for stage, v in timings.items()})

        # --- HNSW recall against an exact scan ---
        vectors = []
        for question in questions[: args.recall_queries]:
            vectors.append(to_vector(await gpu.embed_query(question)))
        async with pool.connection() as conn:
            out.hnsw_recall, out.exact_scan_ms = await _hnsw_recall(
                conn, vectors, args.k, (40, 100, 200))

        # --- filtered queries, with and without the exact-scan plan ---
        async with pool.connection() as conn:
            windows = [r[0] for r in await (await conn.execute(
                "SELECT window_id FROM corpus_window WHERE loaded_at IS NOT NULL ORDER BY window_id")).fetchall()]
        for label, filters in _filter_cases(windows):
            for exact_max in (0, 500_000):
                cfg = SearchConfig(filters=filters, exact_scan_max_rows=exact_max,
                                   rerank="off", expand_radius=0)
                elapsed, found, scope = [], 0, None
                for question in questions[: args.filter_queries]:
                    started = time.perf_counter()
                    result = await searcher.search(question, top_k=10, config=cfg)
                    elapsed.append((time.perf_counter() - started) * 1000)
                    found += len(result.hits)
                    scope = result.diagnostics.get("scope_rows", scope)
                out.filtered.append({
                    "filter": label, "plan": "exact" if exact_max else "hnsw",
                    "p50_ms": pct(elapsed, 0.5), "p95_ms": pct(elapsed, 0.95),
                    "hits": found, "scope_rows": scope})

        # --- what the first stage never hands the reranker ---
        out.pool_ceiling = await _pool_ceiling(searcher, gold, args.k)

        # --- the cost of a refresh, both ways ---
        async with pool.connection() as conn:
            started = time.perf_counter()
            await build_lexicon(conn)
            out.refresh_s["ts_stat"] = round(time.perf_counter() - started, 2)
            started = time.perf_counter()
            stored = await load_lexicon(conn, stored=True)
            out.refresh_s["stored_table"] = round(time.perf_counter() - started, 2)
            out.refresh_s["lexemes"] = len(stored.df)
    finally:
        await gpu.aclose()
        await pool.close()
    return out


def _filter_cases(windows: list[str]) -> list[tuple[str, SearchFilters]]:
    cases: list[tuple[str, SearchFilters]] = [("none", SearchFilters())]
    if windows:
        cases.append((f"window={windows[0]}", SearchFilters(windows=(windows[0],))))
    cases.append(("house=Lords", SearchFilters(houses=("Lords",))))
    cases.append(("section=Westminster Hall", SearchFilters(sections=("Westminster Hall",))))
    return cases


async def _pool_ceiling(searcher: HybridSearcher, gold, k: int) -> dict[str, Any]:
    """How many labelled speeches ever reach the reranker. The reranker cannot recover what the first stage did not
    retrieve, so this is the ceiling every later experiment sits under.
    """
    cfg = SearchConfig(rerank="off", expand_radius=0, fused_limit=100)
    within = {5: 0, 10: 0, 20: 0, 100: 0}
    labelled = 0
    for item in gold:
        if item.expected_abstain:
            continue
        result = await searcher.search(item.question, top_k=min(k, 50), config=cfg)
        speeches: list[str] = []
        for hit in result.hits:
            if hit.speech_id not in speeches:
                speeches.append(hit.speech_id)
        for relevant in item.relevant:
            labelled += 1
            if relevant.speech_id in speeches:
                rank = speeches.index(relevant.speech_id) + 1
                for cut in within:
                    if rank <= cut:
                        within[cut] += 1
    return {"labelled": labelled, **{f"within_{c}": n for c, n in within.items()}}


def _print(out: Measurements) -> None:
    table = Table(title=f"Corpus and storage ({out.database})")
    table.add_column("")
    table.add_column("Value", justify="right")
    for key, value in out.counts.items():
        table.add_row(key, f"{value:,}")
    for key, value in out.sizes_mb.items():
        table.add_row(f"{key} (MB)", f"{value:,.1f}")
    console.print(table)

    latency = Table(title="Latency by configuration (ms)")
    latency.add_column("configuration")
    for stage in ("embed_and_tokenise", "fuse", "rerank", "expand", "total"):
        latency.add_column(stage, justify="right")
    for name, values in out.per_leg_ms.items():
        latency.add_row(name, *[f"{values.get(f'{s}_p50', 0):.0f}/{values.get(f'{s}_p95', 0):.0f}"
                                for s in ("embed_and_tokenise", "fuse", "rerank", "expand", "total")])
    console.print(latency)
    console.print("[dim](p50/p95)[/dim]")

    recall = Table(title=f"HNSW recall against an exact scan (exact scan p50 {out.exact_scan_ms} ms)")
    recall.add_column("ef_search")
    recall.add_column("recall", justify="right")
    for key, value in out.hnsw_recall.items():
        recall.add_row(key.replace("ef_search_", ""), f"{value:.1%}")
    console.print(recall)

    filters = Table(title="Filtered queries")
    for column in ("filter", "plan", "p50 ms", "p95 ms", "hits", "scope rows"):
        filters.add_column(column, justify="right" if column != "filter" else "left")
    for row in out.filtered:
        filters.add_row(row["filter"], row["plan"], f"{row['p50_ms']:.0f}", f"{row['p95_ms']:.0f}",
                        str(row["hits"]), str(row.get("scope_rows") or "-"))
    console.print(filters)

    console.print(f"Candidate-pool ceiling: {out.pool_ceiling}")
    console.print(f"Statistics refresh: {out.refresh_s}")
    if out.statement_timeout_hits:
        console.print(f"[yellow]{out.statement_timeout_hits} query(ies) hit the statement timeout[/yellow]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--database", help="override the database name in PG_DSN")
    p.add_argument("--json", help="write the measurements here")
    p.add_argument("--queries", type=int, default=25, help="how many gold questions to time")
    p.add_argument("--recall-queries", type=int, default=25,
                   help="how many queries to measure HNSW recall with (each needs an exact scan). "
                        "Each contributes k comparisons, so 10 queries is +/- 3 points and says "
                        "almost nothing; 25 is the whole gold set")
    p.add_argument("--filter-queries", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    out = run(_measure(args))
    _print(out)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(out), indent=2, default=str), encoding="utf-8")
        console.print(f"written to {path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
