"""Apply the adoption rule to the bake-off, mechanically, and say exactly why."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.bakeoff import INCUMBENT_KEY, INCUMBENT_RECIPE, TrialResult, regressions  # noqa: E402

REPORTS = ROOT / "reports"

# A screening win smaller than this is noise on 108 queries: one query is worth about 0.009 of recall, so anything under
# two queries' worth is not a result.
NOISE_FLOOR = 0.02

# How much a protected category may slip before the candidate is refused.
CATEGORY_TOLERANCE = 0.02


def load_results(paths: list[Path]) -> list[dict]:
    """Every screened trial from every report, de-duplicated by trial, latest file winning."""
    out: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("results", []):
            if row.get("recall_at_100") is not None:
                out[row["trial"]] = row
    return list(out.values())


def as_trial_result(row: dict) -> TrialResult:
    return TrialResult(
        label=row.get("label", row["trial"]), stem=row["trial"], model_key=row["model_key"],
        recipe_key=row["recipe_key"], recall_at_100=row.get("recall_at_100"),
        recall_at_10=row.get("recall_at_10"), mrr_at_10=row.get("mrr_at_10"),
        per_category=row.get("per_category") or {}, dimensions=row.get("dimensions", 0),
        header_overhead_ratio=row.get("header_overhead_ratio", 0.0), chunks=row.get("chunks", 0),
    )  # fmt: skip


def decide(rows: list[dict], *, noise_floor: float = NOISE_FLOOR,
           tolerance: float = CATEGORY_TOLERANCE) -> dict:  # fmt: skip
    """The decision, as data. Every rejection carries the reason it was rejected."""
    if not rows:
        return {"adopt": False, "winner": None, "reason": "nothing was screened"}

    incumbent_row = next(
        (r for r in rows if r["model_key"] == INCUMBENT_KEY and r["recipe_key"] == INCUMBENT_RECIPE.key), None
    )
    if incumbent_row is None:
        return {
            "adopt": False, "winner": None,
            "reason": (f"the incumbent ({INCUMBENT_KEY} + {INCUMBENT_RECIPE.key}) was not screened, so "
                       "nothing has a baseline to beat. Screen it and re-run."),
        }  # fmt: skip

    incumbent = as_trial_result(incumbent_row)
    base_recall = incumbent.recall_at_100 or 0.0
    considered = []
    for row in sorted(rows, key=lambda r: -(r["recall_at_100"] or 0)):
        candidate = as_trial_result(row)
        if candidate.stem == incumbent.stem:
            continue
        gain = (candidate.recall_at_100 or 0.0) - base_recall
        hits = regressions(incumbent, candidate, tolerance=tolerance)
        reasons = []
        if gain <= noise_floor:
            reasons.append(
                f"gain {gain:+.4f} is inside the {noise_floor} noise floor "
                f"({'no better than' if gain <= 0 else 'not clearly better than'} the incumbent)"
            )
        if hits:
            reasons.append("regresses protected categories: " + "; ".join(hits))
        considered.append({
            "trial": candidate.stem, "label": candidate.label, "recall_at_100": candidate.recall_at_100,
            "recall_at_10": candidate.recall_at_10, "mrr_at_10": candidate.mrr_at_10,
            "dimensions": candidate.dimensions, "gain_vs_incumbent": round(gain, 4),
            "header_overhead_ratio": candidate.header_overhead_ratio,
            "regressions": hits, "eligible": not reasons, "reasons": reasons,
        })  # fmt: skip

    eligible = [c for c in considered if c["eligible"]]
    winner = max(eligible, key=lambda c: c["recall_at_100"]) if eligible else None
    return {
        "adopt": winner is not None,
        "winner": winner["trial"] if winner else None,
        "winner_label": winner["label"] if winner else None,
        "gain_vs_incumbent": winner["gain_vs_incumbent"] if winner else 0.0,
        "incumbent": {
            "trial": incumbent.stem, "recall_at_100": incumbent.recall_at_100,
            "recall_at_10": incumbent.recall_at_10, "mrr_at_10": incumbent.mrr_at_10,
            "dimensions": incumbent.dimensions,
        },  # fmt: skip
        "noise_floor": noise_floor,
        "category_tolerance": tolerance,
        "considered": considered,
        "reason": (
            f"{winner['label']} beats the incumbent by {winner['gain_vs_incumbent']:+.4f} recall@100 with no "
            "protected-category regression" if winner else
            "no candidate cleared the noise floor without regressing a protected category, so the "
            "incumbent stands"
        ),
        "caveats": [
            "screening recall is optimistic: the screening corpus holds fewer distractors than production. "
            "The comparison is fair because every candidate faced the identical corpus.",
            "a screening win is dense-only. The winner must still survive the full pipeline and the test "
            "split before it is production.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="append", default=[], help="a screening report (repeatable)")
    p.add_argument("--json", default=str(REPORTS / "session3_bakeoff_decision.json"))
    p.add_argument("--noise-floor", type=float, default=NOISE_FLOOR)
    p.add_argument("--tolerance", type=float, default=CATEGORY_TOLERANCE)
    args = p.parse_args(argv)

    paths = [Path(x) for x in args.report] or [
        REPORTS / "session3_bakeoff_headers.json",
        REPORTS / "session3_bakeoff_models.json",
        REPORTS / "session3_bakeoff_screen.json",
    ]
    rows = load_results(paths)
    decision = decide(rows, noise_floor=args.noise_floor, tolerance=args.tolerance)

    print(f"screened trials: {len(rows)}")
    if inc := decision.get("incumbent"):
        print(f"incumbent {inc['trial']}: recall@100 {inc['recall_at_100']}  "
              f"recall@10 {inc['recall_at_10']}  MRR {inc['mrr_at_10']}")  # fmt: skip
    for c in decision.get("considered", []):
        mark = "ELIGIBLE" if c["eligible"] else "refused "
        print(f"  {mark}  {c['label']:28s} R@100 {c['recall_at_100']}  ({c['gain_vs_incumbent']:+.4f})")
        for reason in c["reasons"]:
            print(f"            {reason}")
    print(f"\nDECISION: {'ADOPT ' + str(decision['winner_label']) if decision['adopt'] else 'keep the incumbent'}")
    print(f"  {decision['reason']}")
    for caveat in decision["caveats"]:
        print(f"  caveat: {caveat}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    # The orchestrator reads the last JSON object on stdout.
    print(json.dumps({"winner": decision["winner"], "adopt": decision["adopt"],
                      "reason": decision["reason"]}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
