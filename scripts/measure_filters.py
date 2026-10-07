"""Does passing the question's own scope to retrieval put the answer in front of the generator more often?"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from answer import pipeline as PL  # noqa: E402
from answer import router as RO  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402

import eval_answers as EA  # noqa: E402

# Below this many hits the filtered search is treated as too narrow and widened to the whole corpus.
WIDEN_BELOW = 3


async def main(split: str, top_k: int) -> int:
    if split != "dev":
        raise SystemExit("dev only: filters are adopted or rejected on dev, never on a confirmation split")
    items = [i for i in load_gold(ROOT / "tests" / "data" / "gold_v2.json")
             if i.split == "dev" and EA.gold_speech_ids(i)]  # fmt: skip
    runtime, pool = await PL.build_runtime(with_verifier=False)
    rows = []
    try:
        for index, item in enumerate(items, start=1):
            wanted = EA.gold_speech_ids(item)
            scope = PL.to_search_filters(RO.extract_filters(item.question))
            row = {"id": item.id, "category": item.category, "filtered": bool(scope),
                   "scope": {k: str(v) for k, v in scope.as_dict().items()}}

            async def rank_of(filters):
                started = time.perf_counter()
                deck, result = await PL.search_deck(runtime, item.question, top_k, filters=filters)
                shown = [str(c.speech_id) for c in deck.cards]
                hit = next((n for n, sid in enumerate(shown, 1) if sid in wanted), None)
                return hit, len(result.hits), int((time.perf_counter() - started) * 1000)

            row["unfiltered_rank"], _, row["unfiltered_ms"] = await rank_of(None)
            if scope:
                row["filtered_rank"], hits, row["filtered_ms"] = await rank_of(scope)
                row["filtered_hits"] = hits
                row["widened_rank"] = row["filtered_rank"] if hits >= WIDEN_BELOW else row["unfiltered_rank"]
                row["widened"] = hits < WIDEN_BELOW
            else:
                row["filtered_rank"] = row["widened_rank"] = row["unfiltered_rank"]
                row["widened"] = False
            rows.append(row)
            if index % 20 == 0:
                print(f"  [{index}/{len(items)}]")
    finally:
        await pool.close()

    def recall(key):
        return sum(1 for r in rows if r[key] is not None) / len(rows)

    scoped = [r for r in rows if r["filtered"]]
    gained = [r for r in scoped if r["unfiltered_rank"] is None and r["widened_rank"] is not None]
    lost = [r for r in scoped if r["unfiltered_rank"] is not None and r["widened_rank"] is None]
    improved = [r for r in scoped if r["unfiltered_rank"] and r["widened_rank"] and r["widened_rank"] < r["unfiltered_rank"]]
    worsened = [r for r in scoped if r["unfiltered_rank"] and r["widened_rank"] and r["widened_rank"] > r["unfiltered_rank"]]
    print(f"\n{len(rows)} dev items with answer-bearing evidence; {len(scoped)} carry a scope the router extracted")
    print(f"  deck recall  unfiltered {recall('unfiltered_rank'):.1%}   filtered {recall('filtered_rank'):.1%}   "
          f"widened {recall('widened_rank'):.1%}")
    print(f"  among the {len(scoped)} scoped items: gained {len(gained)}, lost {len(lost)}, "
          f"ranked higher {len(improved)}, ranked lower {len(worsened)}, widened {sum(r['widened'] for r in scoped)}")
    for r in lost:
        print(f"    LOST {r['id']} [{r['category']}] scope={r['scope']} unfiltered rank {r['unfiltered_rank']}")
    for r in gained[:8]:
        print(f"    gained {r['id']} [{r['category']}] scope={r['scope']} -> rank {r['widened_rank']}")
    timed = [r for r in scoped if "filtered_ms" in r]
    if timed:
        med = sorted(r["filtered_ms"] for r in timed)[len(timed) // 2]
        base = sorted(r["unfiltered_ms"] for r in timed)[len(timed) // 2]
        print(f"  median search latency on scoped items: unfiltered {base} ms, filtered {med} ms")
    out = ROOT / "reports" / "a01_filters_dev.json"
    out.write_text(json.dumps({"widen_below": WIDEN_BELOW, "top_k": top_k, "rows": rows}, indent=1, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    sys.exit(run_async(main(args.split, args.top_k)))
