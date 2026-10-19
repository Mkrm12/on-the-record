"""How stable is the answer judge? Re-judge IDENTICAL stored answers and count the verdicts that flip."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from goldset import judges as J  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402

import eval_answers as EA  # noqa: E402

# A stored row shorter than this was stored whole (older runs kept 600 characters).
STORED_WHOLE_BELOW = 590


def choose(rows: list[dict], judge: str, limit: int, seed: int = 18) -> list[dict]:
    """Every answer the judge called harmful or wrong-but-benign, then a random sample of the rest, up to `limit`.
    """
    judged = [r for r in rows if r.get("by") == judge and r.get("outcome")
              and len(r.get("rendered") or "") < STORED_WHOLE_BELOW and (r.get("rendered") or "").strip()]
    rare = [r for r in judged if r["outcome"] in ("harmful", "wrong_but_benign")]
    rest = [r for r in judged if r not in rare]
    random.Random(seed).shuffle(rest)
    return (rare + rest)[:limit]


def summarise(results: dict[str, list[str]]) -> dict:
    flips = {i: v for i, v in results.items() if len(set(v)) > 1}
    harmful_seen = {i: v for i, v in results.items() if "harmful" in v}
    return {
        "items": len(results), "flipped": len(flips),
        "flip_rate": round(len(flips) / len(results), 4) if results else None,
        "harmful_ever": len(harmful_seen),
        "harmful_every_time": sum(1 for v in harmful_seen.values() if set(v) == {"harmful"}),
        "per_item": results,
    }


async def rejudge(report_path: Path, repeats: int, limit: int) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gold = {i.id: i for i in load_gold(ROOT / "tests" / "data" / "gold_v2.json")}
    available = J.panel(exclude_family=J.GENERATOR_FAMILY)
    provider = J.primary(available, exclude_family=J.GENERATOR_FAMILY)
    if provider is None:
        raise SystemExit("no API judge from another family is available (the owner's judge is Gemini, run offline: "
                         "re-judge a packet there instead)")
    from goldset.llm import Client as JudgeClient

    client = JudgeClient(key=provider.key(), base=provider.base, model=provider.model)
    limiter = J.RateLimiter(provider.min_interval_s, per_day=provider.per_day)
    chosen = choose(report["rows"], provider.name, limit)
    results: dict[str, list[str]] = {}
    for row in chosen:
        answer = SimpleNamespace(rendered=row["rendered"], abstention_detail="")
        verdicts = [row["outcome"]]
        for _ in range(repeats):
            try:
                verdicts.append((await EA.judge_one(client, limiter, gold[row["id"]], answer)).get("outcome"))
            except J.QuotaExhausted as exc:
                print(f"STOPPING: the judge's quota is exhausted ({exc})")
                return {"stopped": str(exc), **summarise(results)}
        results[row["id"]] = verdicts
        mark = "" if len(set(verdicts)) == 1 else "   <- FLIPPED"
        print(f"  {row['id']:8} {' / '.join(str(v) for v in verdicts)}{mark}")
    return {"report": report_path.name, "judge": f"{provider.name}/{provider.model}", "repeats": repeats,
            **summarise(results)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", default=str(ROOT / "reports" / "e3_eval_dev.json"))
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args(argv)
    out = run_async(rejudge(Path(args.report).resolve(), args.repeats, args.limit))
    (ROOT / "reports" / "judge_stability.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{out['items']} answers re-judged {args.repeats}x: {out['flipped']} flipped (rate {out['flip_rate']}); "
          f"harmful at least once {out['harmful_ever']}, harmful every time {out['harmful_every_time']}")
    print(f"per-verdict: {Counter(v[0] for v in out['per_item'].values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
