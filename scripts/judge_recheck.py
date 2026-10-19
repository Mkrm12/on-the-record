"""Re-check the relevance judge before any new judgments are added."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.config import settings  # noqa: E402
from goldset import judge as judging  # noqa: E402
from goldset.judge import JUDGMENTS_PATH, Pool, cohen_kappa, load_judgments  # noqa: E402
from goldset.llm import Client, LlmError  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run  # noqa: E402

REPORTS = ROOT / "reports"
OUT = REPORTS / "e1_judge_recheck.json"

# The pre-registered bar for mixing old and new judgments.
MIX_GATE = 0.80

# The sample the plan asks for.
DEFAULT_SAMPLE = 150


def stratified(pool: Pool, grades: dict[str, dict[str, int]], size: int, seed: int = 20260926,
               categories: dict[str, str] | None = None) -> list[judging.Pair]:  # fmt: skip
    """`size` pairs spread across grades and categories, best effort, deterministic for a seed."""
    if size <= 0:
        raise ValueError("size must be at least 1")
    by_stratum: dict[tuple[int, str], list[judging.Pair]] = defaultdict(list)
    for pair in pool.pairs:
        grade = (grades.get(pair.item_id) or {}).get(pair.speech_id)
        if grade is None:
            continue  # never judged, so there is nothing to agree or disagree with
        category = (categories or {}).get(pair.item_id, "")
        by_stratum[(grade, category)].append(pair)

    rng = random.Random(seed)
    for pairs in by_stratum.values():
        rng.shuffle(pairs)
    # Order the strata by grade first so the sample is balanced across grades even if it runs short.
    order = sorted(by_stratum, key=lambda key: (key[0], key[1]))
    chosen: list[judging.Pair] = []
    while len(chosen) < size and any(by_stratum[key] for key in order):
        for key in order:
            if by_stratum[key] and len(chosen) < size:
                chosen.append(by_stratum[key].pop())
    return chosen


def per_stratum_agreement(
    old: dict[tuple[str, str], int], new: dict[tuple[str, str], int], threshold: int = judging.RELEVANT_AT
) -> dict[str, dict[str, float | int]]:
    """Binary agreement broken down by the stored grade."""
    buckets: dict[int, list[bool]] = defaultdict(list)
    for key, stored in old.items():
        if key in new:
            buckets[stored].append((stored >= threshold) == (new[key] >= threshold))
    return {
        str(grade): {"pairs": len(hits), "agreement": round(sum(hits) / len(hits), 4)}
        for grade, hits in sorted(buckets.items())
        if hits
    }


def crossings(old: dict[tuple[str, str], int], new: dict[tuple[str, str], int],
              threshold: int = judging.RELEVANT_AT) -> dict[str, int]:
    """How many pairs crossed the relevance line, and in which direction."""
    became_relevant = sum(1 for k, v in new.items()
                          if k in old and old[k] < threshold <= v)  # fmt: skip
    became_irrelevant = sum(1 for k, v in new.items()
                            if k in old and v < threshold <= old[k])  # fmt: skip
    return {"became_relevant": became_relevant, "became_irrelevant": became_irrelevant}


def verdict(binarised: float | None, agreement: float | None, four_level: float | None = None,
            gate: float = MIX_GATE) -> tuple[str, str]:  # fmt: skip
    """(verdict, what follows), decided on the BINARISED kappa. Pre-registered, so the number decides."""
    if binarised is None:
        return "unknown", "no pair could be compared, so nothing can be concluded. Do not add judgments."

    four = "" if four_level is None else (
        f" Four-level kappa is {four_level:.4f}, which is lower and expected to be: the judge has never been "
        "able to tell a 2 from a 3 (I-77), which is why no graded metric is reported anywhere in this project."
    )
    if binarised >= gate:
        return "stable", (
            f"binarised kappa {binarised:.4f} is at or above the {gate:.2f} gate: the judge still draws the "
            f"relevance line where Session 2 validated it, and old and new judgments may be mixed.{four}"
        )
    return "drifted", (
        f"binarised kappa {binarised:.4f} is below the {gate:.2f} gate (raw agreement "
        f"{agreement if agreement is None else f'{agreement:.1%}'}). Old and new judgments must NOT be mixed. "
        f"Either re-judge the whole pool with today's configuration, or freeze the old judge and use it for "
        f"everything -- and record which was chosen.{four}"
    )


def _grade_census(sample: list[judging.Pair], grades: dict[str, dict[str, int]]) -> dict[str, int]:
    """stored grade -> how many of the sample carry it. Printed so a lopsided sample is visible."""
    counts = Counter((grades.get(p.item_id) or {}).get(p.speech_id) for p in sample)
    return {str(grade): n for grade, n in sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0]))
            if grade is not None}


async def recheck(args: argparse.Namespace) -> dict:
    pool, grades, meta = load_judgments(Path(args.judgments))
    categories = {item.id: item.category for item in load_gold()}
    sample = stratified(pool, grades, args.sample, seed=args.seed, categories=categories)
    if not sample:
        raise SystemExit("no already-judged pair could be sampled; there is nothing to re-check")

    # `estimate_judging` returns an `Estimate` dataclass, not a dict.
    estimate = judging.estimate_judging(
        Pool(pairs=sample, variants=pool.variants, retrieved={}),
        settings.bulk_model, args.reasoning, limit=None)  # fmt: skip
    cost = float(estimate.usd)

    report: dict = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stored": {"judged_at": meta.get("judged_at"), "model_recorded": meta.get("model"),
                   "reasoning_effort": meta.get("reasoning_effort"), "grades": sum(len(v) for v in grades.values()),
                   "note": "the alias only; no resolved model id or fingerprint was recorded (E1 §4.2)"},
        "sample": {"pairs": len(sample), "seed": args.seed,
                   "by_stored_grade": _grade_census(sample, grades)},
        "estimate_usd": round(cost, 4),
        "estimate_basis": estimate.line(),
        "gate": MIX_GATE,
    }

    if args.estimate_only:
        report["verdict"] = "not run"
        report["note"] = "--estimate-only: nothing was sent and nothing was spent"
        return report

    client = Client(key=settings.deepseek_key, base=settings.deepseek_base,
                    model=settings.bulk_model, reasoning_effort=args.reasoning)  # fmt: skip
    if not client.configured:
        raise SystemExit("DEEPSEEK_KEY is not set, so the judge cannot be re-checked")

    ok, why = await client.affordable(cost, floor_usd=None, confirm_gbp=settings.spend_confirm_gbp)
    report["spend_check"] = why or "inside the approval bar"
    if not ok:
        report["verdict"] = "refused"
        return report

    ready, detail = await judging.smoke_test(client)
    report["smoke_test"] = detail
    if not ready:
        raise SystemExit(f"the judge failed its smoke test, so the re-check would measure the harness: {detail}")

    started = time.perf_counter()
    try:
        fresh = await judging.judge_run(Pool(pairs=sample, variants=[], retrieved={}), client)
    except LlmError as exc:
        raise SystemExit(f"the re-check could not complete: {exc}") from exc
    report["run"] = {"offered": fresh.offered, "graded": fresh.graded, "no_reply": fresh.no_reply,
                     "unparsable": fresh.unparsable, "seconds": round(time.perf_counter() - started, 1)}
    # WHICH model answered, which is the whole point of item 4.
    report["provenance"] = client.provenance()

    old = {k: v for k, v in judging.flatten(grades).items() if k in judging.flatten(fresh.grades)}
    new = judging.flatten(fresh.grades)
    binary = judging.binary_agreement(old, new)
    kappa = cohen_kappa(old, new)
    report["agreement"] = binary
    report["kappa"] = kappa
    report["per_stored_grade"] = per_stratum_agreement(old, new)
    report["crossings"] = crossings(old, new)
    report["confusion"] = judging.confusion(old, new)

    # `binary_agreement` gives the binarised kappa, which is the gate.
    binarised = binary.get("kappa") if isinstance(binary, dict) else None
    four_level = kappa.get("kappa") if isinstance(kappa, dict) else kappa
    raw = binary.get("agreement") if isinstance(binary, dict) else None
    report["gate_metric"] = "binarised kappa at grade >= 2"
    report["binarised_kappa"] = binarised
    report["four_level_kappa"] = four_level
    report["verdict"], report["what_follows"] = verdict(binarised, raw, four_level)
    if report["verdict"] == "drifted":
        whole_pool = judging.estimate_judging(pool, settings.bulk_model, args.reasoning, limit=None)
        report["full_rejudge_usd"] = round(float(whole_pool.usd), 4)
        report["full_rejudge_basis"] = whole_pool.line()
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--reasoning", default="", help="reasoning_effort; empty means the provider default")
    p.add_argument("--judgments", default=str(JUDGMENTS_PATH))
    p.add_argument("--estimate-only", action="store_true", help="price it and stop, spending nothing")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.sample < 1:
        p.error("--sample must be at least 1")

    report = run(recheck(args))
    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"sampled {report['sample']['pairs']} already-judged pairs "
              f"(by stored grade: {report['sample']['by_stored_grade']})")
        print(f"estimate ${report['estimate_usd']:.4f}")
        if report.get("provenance"):
            prov = report["provenance"]
            print(f"requested {prov['requested_model']!r}; served {prov['served_models']} "
                  f"fingerprints {prov['system_fingerprints'] or '(none returned)'}")
            if not prov["one_model"]:
                print("  WARNING: more than one model answered this run, so it is not one system")
        if report.get("binarised_kappa") is not None:
            print(f"binarised kappa {report['binarised_kappa']} (THE GATE)   "
                  f"raw agreement {report.get('agreement', {}).get('agreement')}")
            print(f"four-level kappa {report['four_level_kappa']} (diagnostic only; I-77)")
            print(f"crossings: {report.get('crossings')}")
            print(f"per stored grade: {report.get('per_stored_grade')}")
        print(f"\n{report['verdict'].upper()}: {report.get('what_follows', report.get('note', ''))}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if report["verdict"] in ("stable", "not run") else 1


if __name__ == "__main__":
    sys.exit(main())
