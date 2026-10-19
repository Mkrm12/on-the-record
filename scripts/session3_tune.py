"""Sweep the retrieval knobs on dev, choose by leave-one-window-out, then confirm and sit the exam."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from retrieval import tuning  # noqa: E402
from retrieval.evaluation import Variant, run_variant  # noqa: E402
from retrieval.gpu_client import GpuClient  # noqa: E402
from retrieval.hybrid import HybridSearcher, SearchConfig  # noqa: E402

REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "tuning"

# They are one policy, not knobs to tune.
W0_COMMON_DF_RATIO = 50 / 90_040


def baseline_config() -> SearchConfig:
    return SearchConfig(
        common_df_ratio=W0_COMMON_DF_RATIO,
        speaker_date_aware=True,
        stored_lexicon=True,
        ef_search=100,
    )


# knob -> the values to try.
GRID: dict[str, list] = {
    "sparse_coverage": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    "pool": [10, 20, 30, 50],
    "expand_radius": [2, 4, 6],
    "rrf_k": [10, 30, 60, 100],
    "weights": [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0), (2.0, 1.0, 0.5)],
    "ef_search": [40, 100, 250],
}


def variants_for(knob: str, values: list) -> list[Variant]:
    base = baseline_config()
    return [Variant(f"{knob}={value}", replace(base, **{knob: value}), f"{knob} set to {value}") for value in values]


def _say(message: str) -> None:
    print(message, flush=True)


# --- the sweep ----------------------------------------------------------------------------------------


async def sweep(args: argparse.Namespace) -> int:
    items = [i for i in load_gold() if i.split == "dev"]
    queries = [tuning.GoldQuery.of(i) for i in tuning.scorable(items)]
    if not queries:
        _say("No dev query has pooled relevance. Run scripts/build_gold.py --judge first.")
        return 1
    by_window = defaultdict(int)
    for q in queries:
        by_window[q.window or "none"] += 1
    _say(f"dev queries with pooled relevance: {len(queries)} ({dict(sorted(by_window.items()))})")
    _say(f"mean relevant speeches per query: {sum(len(q.relevant) for q in queries) / len(queries):.1f}")

    # With the reranker off, `fused_limit` governs and recall@100 means what it says. pool caps the OUTPUT, not just the
    # rerank depth: with the reranker off the list is still 20 long.
    ceiling = replace(baseline_config(), rerank="off", fused_limit=100, pool=100)
    todo = [
        ("baseline", [Variant("baseline", baseline_config(), "the Session 1 configuration")]),
        ("ceiling", [Variant("ceiling", ceiling, "first-stage candidate ceiling: reranker off, fused_limit 100")]),
    ]
    todo += [(knob, variants_for(knob, values)) for knob, values in GRID.items() if knob in args.knob or not args.knob]
    total = sum(len(v) for _, v in todo)
    _say(f"{total} configurations x {len(queries)} queries = {total * len(queries):,} searches")

    CACHE.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, list] = {}
    started = time.monotonic()
    async with make_pool(make_conninfo(settings.pg_dsn, dbname=args.database), served=False) as db:  # served=False: reproduces E3's tuning, which ran unmasked
        gpu = GpuClient()
        searcher = HybridSearcher(db, gpu)
        await searcher.refresh()
        done = 0
        for knob, variants in todo:
            for variant in variants:
                got = await run_variant(searcher, queries, variant, top_k=args.top_k)
                outcomes[variant.name] = [
                    {"qid": o.qid, "category": o.category, "relevant": sorted(o.relevant),
                     "ranked": o.ranked, "warnings": o.warnings}  # fmt: skip
                    for o in got
                ]
                done += 1
                failed = sum(1 for o in got if o.warnings)
                recall = tuning.mean_metric(got, "recall", tuning.PRIMARY_K)
                _say(f"  [{done}/{total}] {variant.name:28s} recall@{tuning.PRIMARY_K} "
                     f"{recall:.4f}" + (f"  ({failed} query failures)" if failed else ""))  # fmt: skip

    out = CACHE / "sweep_dev.json"
    payload = {
        "database": args.database,
        "top_k": args.top_k,
        "primary_k": tuning.PRIMARY_K,
        "relevant_at": tuning.RELEVANT_AT,
        "queries": [{"qid": q.id, "window": q.window, "category": q.category, "split": q.split} for q in queries],
        "baseline_config": asdict(baseline_config()),
        "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vals] for k, vals in GRID.items()},
        "outcomes": outcomes,
        "minutes_this_run": round((time.monotonic() - started) / 60, 2),
    }

    # Merge, never replace: a partial sweep must not destroy the configurations it did not re-run.
    if out.exists():
        previous = json.loads(out.read_text(encoding="utf-8"))
        same_queries = [q["qid"] for q in previous.get("queries", [])] == [q["qid"] for q in payload["queries"]]
        if same_queries:
            kept = {k: v for k, v in previous.get("outcomes", {}).items() if k not in outcomes}
            payload["outcomes"] = {**kept, **outcomes}
            payload["minutes_total"] = round(
                previous.get("minutes_total", previous.get("minutes_this_run", previous.get("minutes", 0)))
                + payload["minutes_this_run"], 2
            )
            _say(f"  merged with {len(kept)} configuration(s) already in the cache")
        else:
            backup = out.with_suffix(".superseded.json")
            out.replace(backup)
            _say(f"  ! the cached sweep was run over a different query set; it is kept at {backup.name} "
                 "and not merged")  # fmt: skip

    out.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    _say(f"\nwrote {out.relative_to(ROOT)} with {len(payload['outcomes'])} configurations "
         f"in {payload['minutes_this_run']} minutes "
         f"({payload.get('minutes_total', payload['minutes_this_run'])} cumulative)")  # fmt: skip
    return 0


# --- the analysis -------------------------------------------------------------------------------------


class Loaded:
    """Outcomes read back from the sweep, in the shape the metric functions expect."""

    def __init__(self, raw: dict):
        self.raw = raw
        self.window_of = {q["qid"]: q["window"] for q in raw["queries"]}
        self.runs = {
            name: [_Outcome(o["qid"], o["category"], frozenset(o["relevant"]), o["ranked"]) for o in rows]
            for name, rows in raw["outcomes"].items()
        }

    def slice_window(self, name: str, window: str) -> list:
        return [o for o in self.runs[name] if self.window_of.get(o.qid) == window]

    def median_returned(self, name: str) -> int:
        """How many speeches this configuration actually returned, typically."""
        lens = sorted(len(o.ranked) for o in self.runs.get(name, []))
        return lens[len(lens) // 2] if lens else 0

    def values_of(self, knob: str) -> list:
        raw = self.raw["grid"].get(knob, [])
        return [tuple(v) if isinstance(v, list) else v for v in raw]


class _Outcome:
    __slots__ = ("qid", "category", "relevant", "ranked")

    def __init__(self, qid, category, relevant, ranked):
        self.qid, self.category, self.relevant, self.ranked = qid, category, relevant, ranked


def analyse(args: argparse.Namespace) -> int:
    path = CACHE / "sweep_dev.json"
    if not path.exists():
        _say(f"No sweep at {path}. Run --sweep first.")
        return 1
    loaded = Loaded(json.loads(path.read_text(encoding="utf-8")))
    baseline = loaded.runs["baseline"]
    metric, k = args.metric, args.k or tuning.PRIMARY_K
    _say(f"Analysing {len(loaded.runs)} configurations on {len(baseline)} dev queries, "
         f"primary metric {metric}@{k}, relevance at grade >= {loaded.raw['relevant_at']}\n")  # fmt: skip

    windows = [w for w in tuning.DEV_WINDOWS if any(v == w for v in loaded.window_of.values())]
    _say(f"dev windows present: {windows}")
    base_recall = tuning.mean_metric(baseline, metric, k)
    _say(f"baseline {metric}@{k}: {base_recall:.4f}\n")

    # --- knob by knob, leave-one-window-out -----------------------------------------------------------
    knob_results = []
    for knob, values in loaded.raw["grid"].items():
        scores_by_window: dict = {}
        for value in values:
            value = tuple(value) if isinstance(value, list) else value
            name = f"{knob}={value}"
            if name not in loaded.runs:
                continue
            scores_by_window[value] = {
                w: score for w in windows
                if (score := tuning.mean_metric(loaded.slice_window(name, w), metric, k)) is not None
            }  # fmt: skip
        if not scores_by_window:
            continue
        result = tuning.tune_knob(knob, scores_by_window, tolerance=args.tolerance, windows=tuple(windows))
        knob_results.append(result)
        _say(result.line())
        lengths = {v: loaded.median_returned(f"{knob}={v}") for v in scores_by_window}
        if confound := tuning.truncation_confound(knob, k, lengths):
            _say(f"    ! {confound}")
            _say(f"    ! the values return {dict(sorted(lengths.items(), key=lambda kv: str(kv[0])))} "
                 "speeches respectively, so this knob's numbers at this depth are not comparable")  # fmt: skip
        for value, by_window in sorted(scores_by_window.items(), key=lambda kv: str(kv[0])):
            overall = sum(by_window.values()) / len(by_window) if by_window else float("nan")
            marker = "  <- chosen" if value == result.chosen else ""
            _say(f"    {knob}={str(value):16s} dev mean {overall:.4f}   " +
                 " ".join(f"{w} {by_window.get(w, float('nan')):.3f}" for w in windows) + marker)  # fmt: skip

    # --- the family comparison: each chosen value against the baseline --------------------------------
    _say("\nPaired comparisons against the baseline, one family (Holm-Bonferroni over the chosen knobs):")
    family = []
    for result in knob_results:
        name = f"{result.knob}={result.chosen}"
        if name not in loaded.runs:
            continue
        comparison = tuning.compare(baseline, loaded.runs[name], name, metric=metric, k=k, seed=args.seed)
        tuning.check_across_windows(comparison, baseline, loaded.runs[name], loaded.window_of, windows=tuple(windows))
        # A knob that caps the result list cannot be compared at a depth it controls.
        lengths = {
            value: loaded.median_returned(f"{result.knob}={value}")
            for value in loaded.values_of(result.knob)
            if f"{result.knob}={value}" in loaded.runs
        }
        comparison.confound = tuning.truncation_confound(result.knob, k, lengths)
        family.append(comparison)
    tuning.holm_bonferroni(family, alpha=args.alpha)
    for comparison in sorted(family, key=lambda c: c.p):
        _say("  " + comparison.line())
        if comparison.per_window:
            _say("      per window: " + ", ".join(f"{w} {m:+.4f}" for w, m in sorted(comparison.per_window.items())))

    adopted = {c.name.split("=")[0]: c for c in family if tuning.adopt(c)}
    _say(f"\nAdopted by the pre-registered rule: {sorted(adopted) or 'nothing'}")
    for knob, comparison in sorted(adopted.items()):
        _say(f"  {knob}: {comparison.line()}")
    rejected = [c for c in family if not tuning.adopt(c)]
    if rejected:
        _say("Not adopted (kept at the baseline value):")
        for comparison in sorted(rejected, key=lambda c: c.p):
            reason = ("CONFOUNDED -- " + comparison.confound if comparison.confound
                      else "the interval includes zero after the family correction" if not comparison.established
                      else "the mean is not positive" if comparison.mean <= 0
                      else "the sign flips across dev windows")  # fmt: skip
            _say(f"  {comparison.name}: {reason}")

    # --- the chosen configuration ---------------------------------------------------------------------
    chosen = baseline_config()
    changes = {}
    for knob, comparison in adopted.items():
        result = next(r for r in knob_results if r.knob == knob)
        chosen = replace(chosen, **{knob: result.chosen})
        changes[knob] = result.chosen
    _say(f"\nChosen configuration: baseline{' + ' + str(changes) if changes else ' unchanged'}")

    report = {
        "metric": metric, "k": k, "alpha": args.alpha, "tolerance": args.tolerance,
        "dev_windows": windows, "baseline": base_recall,
        "knobs": [
            {"knob": r.knob, "chosen": list(r.chosen) if isinstance(r.chosen, tuple) else r.chosen,
             "stable": r.stable, "per_fold": {k2: (list(v) if isinstance(v, tuple) else v)
                                              for k2, v in r.per_fold_choice.items()},
             "held_out_scores": r.held_out_scores, "note": r.note}  # fmt: skip
            for r in knob_results
        ],
        "comparisons": [
            {"name": c.name, "mean": c.mean, "low": c.interval.low, "high": c.interval.high, "p": c.p,
             "adjusted_alpha": c.adjusted_alpha, "established": c.established,
             "holds_across_windows": c.holds_across_windows, "per_window": c.per_window,
             "confound": c.confound, "adopted": tuning.adopt(c)}  # fmt: skip
            for c in family
        ],
        "adopted": {k2: (list(v) if isinstance(v, tuple) else v) for k2, v in changes.items()},
        "chosen_config": asdict(chosen),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "session3_tuning.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    (CACHE / "chosen_config.json").write_text(json.dumps(report["adopted"], indent=2) + "\n", encoding="utf-8", newline="\n")
    _say(f"wrote {(REPORTS / 'session3_tuning.json').relative_to(ROOT)}")
    return 0


# --- confirmation and the exam ------------------------------------------------------------------------


async def score_split(args: argparse.Namespace, split: str) -> int:
    """Score the chosen configuration on a locked split. Refuses without the matching flag."""
    tuning.guard_split(split, confirm=args.confirm, final_exam=args.final_exam)

    chosen_path = CACHE / "chosen_config.json"
    changes = json.loads(chosen_path.read_text(encoding="utf-8")) if chosen_path.exists() else {}
    config = baseline_config()
    for knob, value in changes.items():
        config = replace(config, **{knob: tuple(value) if isinstance(value, list) else value})
    _say(f"scoring split {split!r} with baseline + {changes or 'no changes'}")

    items = [i for i in load_gold() if i.split == split]
    queries = [tuning.GoldQuery.of(i) for i in tuning.scorable(items)]
    if not queries:
        _say(f"no scorable query in split {split!r}: it has no pooled relevance")
        return 1
    _say(f"{len(queries)} scorable queries")

    async with make_pool(make_conninfo(settings.pg_dsn, dbname=args.database), served=False) as db:
        searcher = HybridSearcher(db, GpuClient())
        await searcher.refresh()
        chosen = await run_variant(searcher, queries, Variant("chosen", config, "the adopted configuration"), top_k=args.top_k)
        base = await run_variant(searcher, queries, Variant("baseline", baseline_config(), "Session 1"), top_k=args.top_k)

    rows = {}
    for label, got in (("baseline", base), ("chosen", chosen)):
        rows[label] = {
            f"{m}@{kk}": tuning.mean_metric(got, m, kk)
            for m, kk in (("recall", 100), ("recall", 10), ("mrr", 10), ("ndcg", 10))
        }
        _say(f"  {label:9s} " + "  ".join(f"{name} {value:.4f}" for name, value in rows[label].items()))

    comparison = tuning.compare(base, chosen, f"chosen vs baseline on {split}", metric=args.metric, k=args.k or 100)
    tuning.holm_bonferroni([comparison], alpha=args.alpha)
    _say("  " + comparison.line())

    report = {
        "split": split,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unlocked_with": "--final-exam" if split == "heldout" else "--confirm",
        "changes": changes,
        "queries": len(queries),
        "metrics": rows,
        "comparison": {"mean": comparison.mean, "low": comparison.interval.low, "high": comparison.interval.high,
                       "p": comparison.p, "established": comparison.established},  # fmt: skip
        "note": ("W3 has now been scored. It is burned as a held-out set: re-tuning on it is not allowed, "
                 "and a second look is not a second independent result.") if split == "heldout" else "",
    }
    name = "session3_final_exam.json" if split == "heldout" else "session3_confirm.json"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    _say(f"wrote reports/{name}")
    if split == "heldout":
        _say("\n  W3 is now burned. Do not tune on it and do not score it again.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", default="hansard_v2")
    p.add_argument("--sweep", action="store_true", help="run the knob sweep on dev (the long step)")
    p.add_argument("--analyse", action="store_true", help="folds, adoption and the chosen config, from the cache")
    p.add_argument("--confirm", action="store_true", help="score the chosen config once on the TEST split")
    p.add_argument("--final-exam", action="store_true", help="score W3 ONCE, at the very end")
    p.add_argument("--knob", action="append", default=[], help="sweep only these knobs (repeatable)")
    p.add_argument("--metric", default="recall", choices=("recall", "mrr", "ndcg", "found"))
    p.add_argument("--k", type=int, default=None, help="metric depth (default: the primary 100)")
    p.add_argument("--top-k", type=int, default=100, help="how deep each search returns; must be >= the metric depth")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--tolerance", type=float, default=0.005, help="plateau width when choosing a knob value")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sweep:
        return run(sweep(args))
    if args.analyse:
        return analyse(args)
    if args.final_exam:
        return run(score_split(args, "heldout"))
    if args.confirm:
        return run(score_split(args, "test"))
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
