"""The two HNSW forms side by side on the live corpus, before either is adopted."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psycopg.conninfo import make_conninfo  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from retrieval import vector_index as V  # noqa: E402
from retrieval.gpu_client import GpuClient  # noqa: E402
from retrieval.hybrid import _DENSE_HNSW, SearchConfig  # noqa: E402

EF_SEARCH = SearchConfig().ef_search
DENSE_LIMIT = SearchConfig().dense_limit
RULE = {"recall_margin": 0.005, "latency_ratio": 1.2}

EXACT_SQL = ("SELECT c.id FROM chunk c WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate "
             "ORDER BY c.embedding <=> %(qvec)s::vector, c.id LIMIT %(dense_limit)s")


def leg_sql(form: str) -> str:
    """The production dense leg, verbatim, as a statement of its own: the CTE text with its trailing comma dropped.
    """
    leg = _DENSE_HNSW.format(filter="", order=V.order_expression(form, alias="c")).rstrip().rstrip(",")
    return f"WITH{leg}\nSELECT id FROM dense ORDER BY rnk"


def recall(found: list[int], truth: list[int], k: int) -> float:
    return len(set(found[:k]) & set(truth[:k])) / max(len(truth[:k]), 1)


async def indexes(conn) -> dict[str, dict]:
    """Every HNSW index on chunk, by form: name, size, validity."""
    rows = await (await conn.execute(
        "SELECT c.relname, pg_get_indexdef(i.indexrelid), i.indisvalid, pg_relation_size(i.indexrelid) "
        "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE i.indrelid = 'chunk'::regclass")).fetchall()
    found = {}
    for name, definition, valid, size in rows:
        form = V.form_of(definition)
        if form:
            found[form] = {"name": name, "valid": bool(valid), "bytes": int(size)}
    return found


async def measure(limit: int | None) -> dict:
    items = [i for i in load_gold(ROOT / "tests" / "data" / "gold_v2.json") if i.split == "dev"]
    items = items[:limit] if limit else items
    pool = make_pool(make_conninfo(settings.pg_dsn, dbname="hansard_v2"), min_size=1, max_size=2, served=False)  # served=False: index recall is measured over the whole index
    await pool.open(wait=True)
    gpu = GpuClient(settings.gpu_url)
    out: dict = {"questions": len(items), "ef_search": EF_SEARCH, "dense_limit": DENSE_LIMIT, "rule": RULE}
    try:
        async with pool.connection() as conn:
            present = await indexes(conn)
            out["indexes"] = present
            out["vectors"] = (await (await conn.execute(
                "SELECT count(*) FROM chunk WHERE embedding IS NOT NULL")).fetchone())[0]
            missing = [f for f in V.FORMS if not (present.get(f) or {}).get("valid")]
            if missing:
                raise SystemExit(f"both forms must be present and valid; missing {missing}: {present}")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")
            for form in V.FORMS:
                blocks = (await (await conn.execute("SELECT pg_prewarm(%s, 'read')",
                                                    (present[form]["name"],))).fetchone())[0]
                print(f"pre-read {present[form]['name']} ({form}): {blocks:,} blocks")
            await conn.commit()

            plans = {}
            probe = await gpu.embed_query(items[0].question)
            for form in V.FORMS:
                async with conn.transaction():
                    await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
                    await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                    rows = await (await conn.execute("EXPLAIN " + leg_sql(form),
                                                     {"qvec": probe, "dense_limit": DENSE_LIMIT})).fetchall()
                plan = " ".join(r[0] for r in rows)
                plans[form] = [p["name"] for p in present.values() if p["name"] in plan]
            out["plans_use"] = plans
            print(f"plans: {plans}")
            if any(plans[f] != [present[f]["name"]] for f in V.FORMS):
                raise SystemExit(f"a form's query is not using its own index: {plans}")

            per_item = []
            for n, item in enumerate(items, start=1):
                qvec = await gpu.embed_query(item.question)
                params = {"qvec": qvec, "dense_limit": DENSE_LIMIT}
                async with conn.transaction():
                    await conn.execute("SET LOCAL enable_indexscan = off")
                    await conn.execute("SET LOCAL statement_timeout = 120000")
                    truth = [r[0] for r in await (await conn.execute(EXACT_SQL, params)).fetchall()]
                row = {"id": item.id, "truth": truth}
                for form in V.FORMS:
                    async with conn.transaction():
                        await conn.execute(f"SET LOCAL hnsw.ef_search = {EF_SEARCH}")
                        await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                        started = time.perf_counter()
                        found = [r[0] for r in await (await conn.execute(leg_sql(form), params)).fetchall()]
                        row[f"{form}_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    row[form] = found
                per_item.append(row)
                if n % 25 == 0:
                    print(f"  [{n}/{len(items)}]")
    finally:
        await gpu.aclose()
        await pool.close()

    summary = {}
    for form in V.FORMS:
        times = [r[f"{form}_ms"] for r in per_item]
        summary[form] = {
            "recall@10": round(statistics.mean(recall(r[form], r["truth"], 10) for r in per_item), 4),
            "recall@50": round(statistics.mean(recall(r[form], r["truth"], 50) for r in per_item), 4),
            "top10_exactly_the_truth": sum(r[form][:10] == r["truth"][:10] for r in per_item),
            "ms_p50": round(statistics.median(times), 2),
            "ms_p95": round(sorted(times)[int(0.95 * (len(times) - 1))], 2),
            "index_mib": round(out["indexes"][form]["bytes"] / 2**20, 1),
            "bytes_per_vector": round(out["indexes"][form]["bytes"] / max(out["vectors"], 1)),
        }
    out["summary"] = summary
    out["between_forms"] = {
        "same_50_candidates": sum(set(r["vector"]) == set(r["halfvec"]) for r in per_item),
        "same_ranked_top10": sum(r["vector"][:10] == r["halfvec"][:10] for r in per_item),
        "items": len(per_item),
    }
    v, h = summary["vector"], summary["halfvec"]
    out["verdict"] = {
        "recall@10_within_margin": h["recall@10"] >= v["recall@10"] - RULE["recall_margin"],
        "recall@50_within_margin": h["recall@50"] >= v["recall@50"] - RULE["recall_margin"],
        "latency_within_ratio": h["ms_p50"] <= v["ms_p50"] * RULE["latency_ratio"],
        "still_needed": "the production deck path: measure_cards.py --dense-index vector / halfvec, no item loses "
                        "its citable evidence",
    }
    out["changed_items"] = [{"id": r["id"], "vector_recall@10": recall(r["vector"], r["truth"], 10),
                             "halfvec_recall@10": recall(r["halfvec"], r["truth"], 10)}
                            for r in per_item if r["vector"][:10] != r["halfvec"][:10]]
    out["per_item"] = per_item
    return out


async def main(limit: int | None) -> int:
    out = await measure(limit)
    path = ROOT / "reports" / "t3_index_ab.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for form, s in out["summary"].items():
        print(f"{form:8s} recall@10 {s['recall@10']:.4f}  recall@50 {s['recall@50']:.4f}  "
              f"top10 == truth {s['top10_exactly_the_truth']}/{out['questions']}  p50 {s['ms_p50']} ms  "
              f"p95 {s['ms_p95']} ms  index {s['index_mib']:,} MiB")
    print(f"between forms: {out['between_forms']}")
    print(f"verdict: {out['verdict']}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sys.exit(run_async(main(args.limit)))
