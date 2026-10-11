"""Measure the graph tools against's p95 < 100 ms gate."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import graph as G  # noqa: E402
from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

REPORTS = ROOT / "reports"
SEED = 20260927


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Not interpolated: a latency that was never measured is not a latency."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * len(ordered) + 0.5)) - 1))
    return ordered[index]


def bootstrap_p95(values: list[float], iterations: int = 2000, rng: random.Random | None = None) -> tuple[float, float]:
    """A 95% interval on the p95 itself, so the gate is judged against an interval."""
    rng = rng or random.Random(SEED)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    draws = sorted(percentile([values[rng.randrange(n)] for _ in range(n)], 0.95) for _ in range(iterations))
    return (draws[int(0.025 * iterations)], draws[int(0.975 * iterations)])


async def sample(conn, tool: str, seeds: list, samples: int) -> dict:
    timings: list[float] = []
    sizes: list[int] = []
    errors: list[str] = []
    for i in range(samples):
        seed = seeds[i % len(seeds)]
        started = time.perf_counter()
        try:
            if tool == "bill_walk":
                result = await G.bill_walk(conn, seed)
                size = len(result.stages)
            elif tool == "spine_depth5":
                result = await G.spine(conn, seed, depth=5)
                size = len(result)
            elif tool == "spine_depth2":
                result = await G.spine(conn, seed, depth=2)
                size = len(result)
            elif tool == "debate_walk":
                result = await G.debate_walk(conn, seed)
                size = len(result.participants)
            elif tool == "member_walk":
                result = await G.member_walk(conn, seed)
                size = len(result.speeches)
            elif tool == "speech_to_vote":
                result = await G.speech_to_vote(conn, seed)
                size = len(result.get("votes", ()))
            else:  # pragma: no cover - argparse restricts this
                raise ValueError(tool)
            sizes.append(size)
        except Exception as exc:  # noqa: BLE001 - a failing tool is a result, not a crash
            errors.append(f"{type(exc).__name__}: {exc}")
            # Roll back, or the failed transaction poisons every later tool and the report shows five
            # InFailedSqlTransaction errors instead of the one real cause.
            await conn.rollback()
            continue
        timings.append((time.perf_counter() - started) * 1000.0)

    if not timings:
        return {"tool": tool, "samples": 0, "errors": errors[:3]}
    low, high = bootstrap_p95(timings)
    p95 = percentile(timings, 0.95)
    return {
        "tool": tool, "samples": len(timings), "seeds": len(seeds),
        "median_ms": round(statistics.median(timings), 2),
        "mean_ms": round(statistics.fmean(timings), 2),
        "p95_ms": round(p95, 2),
        "p95_ci95": [round(low, 2), round(high, 2)],
        "max_ms": round(max(timings), 2),
        "gate_ms": G.LATENCY_GATE_MS,
        "target_ms": G.LATENCY_TARGET_MS,
        # The gate is judged on the interval's upper bound, not the point estimate.
        "passes_gate": bool(high < G.LATENCY_GATE_MS),
        "meets_target": bool(high < G.LATENCY_TARGET_MS),
        "mean_result_size": round(statistics.fmean(sizes), 1) if sizes else 0,
        "errors": errors[:3],
    }


async def measure(samples: int, database: str, label: str) -> dict:
    dsn = make_conninfo(settings.pg_dsn, dbname=database)
    rng = random.Random(SEED)
    async with make_pool(dsn) as db, db.connection() as conn:
        async def ids(sql: str) -> list:
            cur = await conn.execute(sql)
            return [r[0] for r in await cur.fetchall()]

        # Bills with the most stages, plus a random spread: the worst case and the typical case in one set.
        busiest = await ids("""SELECT bill_id FROM bill_stage GROUP BY bill_id
                               ORDER BY count(*) DESC LIMIT 10""")
        any_bills = await ids("SELECT id FROM bill ORDER BY random() LIMIT 20")
        debates = await ids("""SELECT debate_id FROM speech WHERE NOT is_boilerplate GROUP BY debate_id
                               ORDER BY count(*) DESC LIMIT 15""")
        members = await ids("""SELECT mnis_id FROM speech WHERE mnis_id IS NOT NULL GROUP BY mnis_id
                               ORDER BY count(*) DESC LIMIT 15""")
        speeches = await ids("""SELECT sp.id FROM speech sp JOIN division v ON v.debate_id = sp.debate_id
                                WHERE sp.mnis_id IS NOT NULL LIMIT 20""")
        bills = busiest + any_bills
        rng.shuffle(bills)

        plan = [("bill_walk", bills), ("spine_depth5", bills), ("spine_depth2", bills),
                ("debate_walk", debates), ("member_walk", members), ("speech_to_vote", speeches)]  # fmt: skip
        results = []
        for tool, seeds in plan:
            if not seeds:
                results.append({"tool": tool, "samples": 0, "errors": ["no seed rows in the corpus"]})
                continue
            await sample(conn, tool, seeds, min(5, samples))  # warm the cache; not measured
            results.append(await sample(conn, tool, seeds, samples))

    return {
        "label": label, "database": database, "samples_per_tool": samples,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED,
        "gate_ms": G.LATENCY_GATE_MS, "target_ms": G.LATENCY_TARGET_MS,
        "tools": results,
        "all_pass": all(r.get("passes_gate") for r in results),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--database", default=safety.TARGET)
    p.add_argument("--label", default="graph tools")
    p.add_argument("--out", default="e2_graph_latency.json")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report = run_async(measure(args.samples, args.database, args.label))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / args.out
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"{report['label']} on {report['database']}, {args.samples} samples per tool")
        print(f"{'tool':18s} {'median':>8s} {'p95':>8s} {'p95 CI95':>18s} {'max':>8s}  gate<{int(G.LATENCY_GATE_MS)}ms")
        for row in report["tools"]:
            if not row.get("samples"):
                print(f"{row['tool']:18s}   NO SAMPLES  {row.get('errors', [''])[0][:40]}")
                continue
            ci = f"[{row['p95_ci95'][0]:.1f}, {row['p95_ci95'][1]:.1f}]"
            flag = "PASS" if row["passes_gate"] else "FAIL"
            star = " *target" if row["meets_target"] else ""
            print(f"{row['tool']:18s} {row['median_ms']:8.2f} {row['p95_ms']:8.2f} {ci:>18s} "
                  f"{row['max_ms']:8.2f}  {flag}{star}")
        print(f"\nall tools inside the gate: {report['all_pass']}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
