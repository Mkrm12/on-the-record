"""Run E1's bake-off arms and apply the pre-registered rule (endgame.md §4.3, §4.6, §4.7 steps 5-8).

    uv run python scripts/uplift.py list                          # the arms, and what each needs
    uv run python scripts/uplift.py run --split dev --stage a      # Stage A: the two-tier screen
    uv run python scripts/uplift.py run --arms S1,P50 --split dev  # just these arms
    uv run python scripts/uplift.py bundle --split dev             # Stage B: bundle + drop-one
    uv run python scripts/uplift.py confirm                        # Stage C: TEST, once
    uv run python scripts/uplift.py freeze                         # adopt mechanically, then freeze
    uv run python scripts/uplift.py report                         # reports/e1_report.md

**Judging is not wired.** Stage A tier 2 scores an arm on whatever coverage the existing pool gives it, and an
arm below judged@10 0.95 is reported **unscorable** rather than guessed at. That is the gate working: with
judged@10 at 0.54 the honest position is that the raw comparison is not trustworthy, which is exactly why the
condensed-list numbers are reported beside it and why Stage C is the only thing that decides anything.

The rule lives in `src/retrieval/uplift.py` and is tested without a database; this file is the part that
searches, judges and spends. Keeping them apart is deliberate: the rule is what E1 freezes retrieval on, and it
had to be verifiable on its own.

**Stage A runs in two tiers, and tier 1 costs nothing.** Every arm is scored first on condensed-list
metrics -- unjudged documents dropped from the ranking rather than assumed irrelevant -- which needs no new
judgments at all. Arms that move nothing there are dropped and logged. Only the survivors' top-20s are judged,
which is the difference between about $2.28 and about $0.85. It is sound as well as cheap: the condensed list is
a subsequence of the raw list, so an arm that reorders nothing among judged documents has not reordered the raw
list at any depth the metric can see.

**Every search is cached** to `data/uplift/`, keyed by arm, split and corpus size, so Stage B's drop-one and
the report never re-run a search or re-spend a judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from retrieval import arms as arms_mod  # noqa: E402
from retrieval import audit as ra  # noqa: E402
from retrieval import tuning  # noqa: E402
from retrieval import uplift as rule  # noqa: E402
from retrieval.evaluation import Variant, run_variant  # noqa: E402
from retrieval.gpu_client import GpuClient  # noqa: E402
from retrieval.hybrid import HybridSearcher  # noqa: E402
from session3_tune import baseline_config  # noqa: E402

CACHE = ROOT / "data" / "uplift"
REPORTS = ROOT / "reports"
BASELINE = "baseline"

# How deep each arm's results are judged to.
JUDGE_DEPTH = 20
# How deep each arm searches.
TOP_K = 100


def say(message: str) -> None:
    print(message, flush=True)


# --- caching -------------------------------------------------------------------------------------------


def cache_path(arm: str, split: str) -> Path:
    return CACHE / f"{split}__{arm}.json"


def save_outcomes(arm: str, split: str, outcomes, corpus_chunks: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path(arm, split).write_text(json.dumps({
        "arm": arm, "split": split, "corpus_chunks": corpus_chunks,
        "searched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "outcomes": [{"qid": o.qid, "category": o.category, "relevant": sorted(o.relevant),
                      "ranked": o.ranked, "warnings": o.warnings} for o in outcomes],
    }, indent=1) + "\n", encoding="utf-8", newline="\n")


class Outcome:
    """The shape the metrics need, read back from the cache."""

    __slots__ = ("qid", "category", "relevant", "ranked", "warnings")

    def __init__(self, row):
        self.qid, self.category = row["qid"], row.get("category") or ""
        self.relevant, self.ranked = frozenset(row["relevant"]), list(row["ranked"])
        self.warnings = list(row.get("warnings") or [])


def load_outcomes(arm: str, split: str, corpus_chunks: int | None = None) -> list[Outcome] | None:
    """Cached outcomes for this arm, or None.

    Refuses a cache built against a different corpus size. W5 adds about 40% more distractors, so an outcome
    measured before it landed is not comparable with one measured after, and silently mixing them would produce
    a bake-off result nobody could interpret.
    """
    path = cache_path(arm, split)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if corpus_chunks is not None and raw.get("corpus_chunks") not in (None, corpus_chunks):
        say(f"  ! {arm} on {split}: cache was built on a {raw.get('corpus_chunks'):,}-chunk corpus, now "
            f"{corpus_chunks:,}. Ignoring it.")
        return None
    return [Outcome(row) for row in raw["outcomes"]]


# --- metrics --------------------------------------------------------------------------------------------


# Pool judgments made for the arms, on top of the committed gold set's own.
EXTRA_JUDGMENTS = CACHE / "extra_judgments.json"


def load_extra() -> dict[str, dict[str, int]]:
    if not EXTRA_JUDGMENTS.exists():
        return {}
    return json.loads(EXTRA_JUDGMENTS.read_text(encoding="utf-8")).get("grades", {})


def save_extra(grades: dict[str, dict[str, int]], provenance: dict) -> int:
    """Merge new grades into the side file. Returns the total held."""
    existing = load_extra()
    for item_id, by_speech in grades.items():
        existing.setdefault(item_id, {}).update(by_speech)
    CACHE.mkdir(parents=True, exist_ok=True)
    EXTRA_JUDGMENTS.write_text(json.dumps({
        "judged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": "pool extension for the E1 arm screen; NOT part of tests/data/gold_v2.json",
        "provenance": provenance,
        "grades": existing,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sum(len(v) for v in existing.values())


def judged_map(items: dict, outcomes) -> dict[str, set[str]]:
    """query id -> every speech carrying a judgment for it, relevant or not.

    Unions the committed gold set's pool with the arm-screen extension, because a speech judged for the screen
    is judged: leaving it out would keep reporting it as a miss and defeat the point of having paid for it.
    """
    extra = load_extra()
    out = {}
    for o in outcomes:
        item = items.get(o.qid)
        graded = set((getattr(item, "graded_relevance", None) or {}).keys()) if item else set()
        out[o.qid] = graded | set(o.relevant) | set((extra.get(o.qid) or {}).keys())
    return out


def relevant_with_extra(items: dict, outcomes) -> dict[str, frozenset[str]]:
    """query id -> the relevant speeches, including any the screen's own judging found.

    The half of the pool extension that changes a number: a speech graded >= 2 by the extension IS relevant,
    and an arm that surfaced it should be credited rather than marked wrong for finding it.
    """
    extra = load_extra()
    out = {}
    for o in outcomes:
        found = {s for s, g in (extra.get(o.qid) or {}).items() if g >= tuning.RELEVANT_AT}
        out[o.qid] = frozenset(set(o.relevant) | found)
    return out


def with_relevance(outcomes, relevance: dict[str, frozenset[str]]):
    """The same outcomes with their relevant set replaced. Cheap shim: the metrics only read attributes."""
    rebuilt = []
    for o in outcomes:
        clone = Outcome({"qid": o.qid, "category": o.category, "relevant": sorted(relevance.get(o.qid, o.relevant)),
                         "ranked": o.ranked, "warnings": o.warnings})  # fmt: skip
        rebuilt.append(clone)
    return rebuilt


async def judge_unjudged(args, pairs: set[tuple[str, str]]) -> dict:
    """Grade the (query, speech) pairs nobody has judged, with the validated relevance judge.

    Priced and checked against the cap before a single call. The judge configuration is the one Session 2
    validated and E1 re-checked (binarised kappa 0.9565), and the served model id is recorded with the grades.
    """
    from goldset import judge as judging
    from goldset.llm import Client

    if not pairs:
        return {"judged": 0, "spend_usd": 0.0, "note": "nothing to judge"}

    items = {i.id: i for i in load_gold()}
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database)
    speech_ids = sorted({sid for _q, sid in pairs})

    async with make_pool(dsn, served=True) as db:
        async with db.connection() as conn:
            passages = await judging._passages(conn, speech_ids)

    built, skipped = [], []
    for qid, sid in sorted(pairs):
        item, passage = items.get(qid), passages.get(sid)
        if item is None or not passage or not (passage.get("passage") or "").strip():
            skipped.append((qid, sid))
            continue
        built.append(judging.Pair(
            item_id=qid, speech_id=sid, question=item.question,
            # `_passages` returns the text under `passage`.
            passage=passage["passage"],
            speaker=passage.get("speaker_name") or "", party=passage.get("speaker_party") or "",
            house=passage.get("house") or "", date=str(passage.get("date") or ""),
            heading=passage.get("heading") or "",
        ))  # fmt: skip

    if skipped:
        say(f"  {len(skipped)} pair(s) skipped: no passage text to grade")
    pool = judging.Pool(pairs=built, variants=["arm_screen"], retrieved={})
    estimate = judging.estimate_judging(pool, settings.bulk_model, args.reasoning, limit=None)
    budget = rule.price(len(built), cap_usd=args.cap)
    say(f"  {len(built):,} pairs to judge, estimate ${estimate.usd:.4f} ({estimate.line()[:70]})")
    if not budget.within_cap or estimate.usd > args.cap:
        raise SystemExit(f"judging would cost ${estimate.usd:.2f}, over the ${args.cap:.2f} cap")

    client = Client(key=settings.deepseek_key, base=settings.deepseek_base, model=settings.bulk_model,
                    reasoning_effort=args.reasoning)  # fmt: skip
    if not client.configured:
        raise SystemExit("DEEPSEEK_KEY is not set, so the arms cannot be judged")
    ok, why = await client.affordable(estimate.usd, confirm_gbp=settings.spend_confirm_gbp)
    if not ok:
        raise SystemExit(f"refusing to judge: {why}")
    ready, detail = await judging.smoke_test(client)
    if not ready:
        raise SystemExit(f"the judge failed its smoke test, so the screen would measure the harness: {detail}")

    started = time.perf_counter()
    run = await judging.judge_run(pool, client)
    total = save_extra(run.grades, client.provenance())
    say(f"  graded {run.graded} of {run.offered} in {time.perf_counter() - started:.0f}s "
        f"({run.no_reply} no reply, {run.unparsable} unparsable); {total:,} extra judgments held")  # fmt: skip
    return {"judged": run.graded, "offered": run.offered, "no_reply": run.no_reply,
            "unparsable": run.unparsable, "spend_usd": round(estimate.usd, 4),
            "provenance": client.provenance(), "total_extra": total}  # fmt: skip


def primaries(outcomes, judged: dict[str, set[str]], condensed: bool) -> dict[str, float | None]:
    """R@10 and MRR@10, raw or condensed. The pre-registered primaries (§4.6)."""
    if not condensed:
        return {"recall@10": tuning.mean_metric(outcomes, "recall", 10),
                "mrr@10": tuning.mean_metric(outcomes, "mrr", 10)}  # fmt: skip
    recalls, mrrs = [], []
    for o in outcomes:
        if not o.relevant:
            continue
        r, m = ra.condensed_pair(judged.get(o.qid, set()), o.relevant, o.ranked, 10)
        if r is not None:
            recalls.append(r)
        if m is not None:
            mrrs.append(m)
    return {"recall@10": (sum(recalls) / len(recalls)) if recalls else None,
            "mrr@10": (sum(mrrs) / len(mrrs)) if mrrs else None}  # fmt: skip


def per_window(base, arm, windows: dict[str, str]) -> dict[str, float]:
    """The R@10 gain in each dev window, for Stage A's sign-consistency condition."""
    by_window: dict[str, list] = defaultdict(list)
    base_by_id = {o.qid: o for o in base}
    for o in arm:
        window = windows.get(o.qid, "")
        if window and o.qid in base_by_id:
            by_window[window].append((base_by_id[o.qid], o))
    out = {}
    for window, pairs in by_window.items():
        b = tuning.mean_metric([p[0] for p in pairs], "recall", 10)
        a = tuning.mean_metric([p[1] for p in pairs], "recall", 10)
        if b is not None and a is not None:
            out[window] = a - b
    return out


def per_category(base, arm) -> dict[str, tuple[float, int]]:
    """category -> (change in R@10, item count), for Stage A's protected-category condition."""
    groups: dict[str, list] = defaultdict(list)
    base_by_id = {o.qid: o for o in base}
    for o in arm:
        if o.qid in base_by_id:
            groups[o.category or "uncategorised"].append((base_by_id[o.qid], o))
    out = {}
    for category, pairs in groups.items():
        b = tuning.mean_metric([p[0] for p in pairs], "recall", 10)
        a = tuning.mean_metric([p[1] for p in pairs], "recall", 10)
        if b is not None and a is not None:
            out[category] = (a - b, len(pairs))
    return out


def unjudged_pairs(items: dict, outcomes, depth: int = JUDGE_DEPTH) -> set[tuple[str, str]]:
    """The (query, speech) pairs this arm surfaced that nobody has judged."""
    judged = judged_map(items, outcomes)
    return {(o.qid, s) for o in outcomes for s in o.ranked[:depth] if s not in judged[o.qid]}


# --- searching -----------------------------------------------------------------------------------------


async def search_arms(args, keys: list[str]) -> dict:
    """Run the baseline and each named arm over the split, caching every result."""
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database)
    items = [i for i in load_gold() if i.split == args.split]
    queries = [tuning.GoldQuery.of(i) for i in tuning.scorable(items)]
    if not queries:
        raise SystemExit(f"no scorable query in split {args.split!r}")
    if args.limit:
        queries = queries[: args.limit]

    configs: list[tuple[str, object]] = [(BASELINE, baseline_config())]
    for key in keys:
        arm = arms_mod.BY_KEY[key]
        configs.append((key, arm.apply(baseline_config())))

    async with make_pool(dsn, served=True) as db:
        recipes = await tuning.corpus_recipes(db)
        recipe = tuning.require_coherent_corpus(recipes)
        corpus_chunks = recipes[recipe]
        say(f"corpus: {corpus_chunks:,} chunks under {recipe}")
        say(f"{len(queries)} scorable queries on {args.split}, {len(configs)} configurations\n")

        searcher = HybridSearcher(db, GpuClient())
        await searcher.refresh()
        if any(arms_mod.BY_KEY[k].changes.get("stock_penalty") for k in keys):
            from build_stock_sentences import load as load_stock

            loaded = searcher.load_stock_sentences(load_stock(dsn))
            say(f"arm B1: {loaded:,} stock sentences loaded")

        for name, config in configs:
            if not args.refresh and (cached := load_outcomes(name, args.split, corpus_chunks)) is not None:
                say(f"  {name:8s} cached ({len(cached)} queries)")
                continue
            started = time.perf_counter()
            got = await run_variant(searcher, queries, Variant(name, config, name), top_k=TOP_K)
            save_outcomes(name, args.split, got, corpus_chunks)
            failures = sum(1 for o in got if o.warnings)
            say(f"  {name:8s} searched in {time.perf_counter() - started:.0f}s"
                + (f"  ({failures} query failures)" if failures else ""))  # fmt: skip

    return {"corpus_chunks": corpus_chunks, "recipe": recipe, "queries": len(queries)}


# --- Stage A -------------------------------------------------------------------------------------------


def stage_a(args) -> dict:
    keys = [a.key for a in arms_mod.resolve(args.arms.split(",") if args.arms else None)]
    context = run_async(search_arms(args, keys))
    items = {i.id: i for i in load_gold()}
    windows = {i.id: (i.window or "") for i in load_gold()}

    base = load_outcomes(BASELINE, args.split)
    if base is None:
        raise SystemExit("the baseline was not searched; re-run without --refresh")
    judged = judged_map(items, base)
    base_condensed = primaries(base, judged, condensed=True)

    # --- tier 1: free -----------------------------------------------------------------------------------
    say("\nTier 1: condensed-list screen (no judgments, $0.00)")
    condensed: dict[str, dict[str, float | None]] = {}
    for key in keys:
        outcomes = load_outcomes(key, args.split)
        if outcomes is None:
            continue
        condensed[key] = primaries(outcomes, judged_map(items, outcomes), condensed=True)
    survivors, dropped = rule.tier1_survivors(condensed, base_condensed)
    for noop in dropped:
        say(f"  {noop.arm:6s} dropped: {noop.reason}")
    say(f"  {len(survivors)} of {len(keys)} arms worth judging: {', '.join(survivors) or 'none'}")

    # --- the bill ----------------------------------------------------------------------------------------
    pairs: set[tuple[str, str]] = set(unjudged_pairs(items, base))
    for key in survivors:
        if (outcomes := load_outcomes(key, args.split)) is not None:
            pairs |= unjudged_pairs(items, outcomes)
    budget = rule.price(len(pairs), cap_usd=args.cap)
    say(f"\nJudging the union of survivors' top-{JUDGE_DEPTH}: {len(pairs):,} unjudged pairs, "
        f"${budget.cost_usd:.2f} (cap ${args.cap:.2f})")  # fmt: skip
    if not budget.within_cap:
        raise SystemExit(budget.refusal())

    judging_report = {"pairs": len(pairs), "estimated_usd": budget.cost_usd, "cap_usd": args.cap,
                      "spent": False}  # fmt: skip
    if pairs and not args.no_judge:
        judging_report.update(run_async(judge_unjudged(args, pairs)))
        judging_report["spent"] = bool(judging_report.get("judged"))

    report = {
        "stage": "A", "split": args.split, "screened_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **context,
        "tier1_dropped": {n.arm: n.moved for n in dropped},
        "tier1_survivors": survivors,
        "judging": judging_report,
        "condensed": condensed,
        "baseline_condensed": base_condensed,
    }

    # --- tier 2: the four conditions, on whatever coverage exists now -------------------------------------
    say("\nTier 2: the pre-registered conditions, on the extended pool")
    verdicts = {}
    # Re-read the judged map and the relevance: the judging above has just extended both.
    judged = judged_map(items, base)
    base = with_relevance(base, relevant_with_extra(items, base))
    base_raw = primaries(base, judged, condensed=False)
    for key in survivors:
        outcomes = load_outcomes(key, args.split)
        if outcomes is None:
            continue
        arm_judged = judged_map(items, outcomes)
        outcomes = with_relevance(outcomes, relevant_with_extra(items, outcomes))
        raw = primaries(outcomes, arm_judged, condensed=False)
        gains = {name: value - base_raw[name] for name, value in raw.items()
                 if value is not None and base_raw.get(name) is not None}  # fmt: skip
        coverage = {o.qid: ra.judged_at_k(arm_judged.get(o.qid, set()), o.ranked, 10) for o in outcomes}
        values = [v for v in coverage.values() if v is not None]
        verdict = rule.screen_arm(
            key, gains, per_window(base, outcomes, windows), per_category(base, outcomes),
            judged_at_10=(sum(values) / len(values)) if values else None,
        )  # fmt: skip
        verdicts[key] = verdict
        say("  " + verdict.line())

    report["passed"] = sorted(k for k, v in verdicts.items() if v.passed)
    report["failed"] = sorted(k for k, v in verdicts.items() if not v.passed and not v.unscorable)
    report["unscorable"] = {k: v.unscorable for k, v in verdicts.items() if v.unscorable}
    report["verdicts"] = {k: asdict(v) for k, v in verdicts.items()}
    report["spend_usd"] = judging_report.get("spend_usd", 0.0)

    out = REPORTS / f"e1_stage_a_{args.split}.json"
    REPORTS.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    say(f"\npassed: {report['passed'] or 'none'}")
    if report["unscorable"]:
        say(f"unscorable (judged@10 below the gate): {sorted(report['unscorable'])}")
        say("  -> these need the incremental judging in §4.7 step 5 before they mean anything")
    say(f"wrote {out.relative_to(ROOT)}")
    return report


# --- Stage B -------------------------------------------------------------------------------------------


def stage_b(args) -> dict:
    a_path = REPORTS / f"e1_stage_a_{args.split}.json"
    if not a_path.exists():
        raise SystemExit(f"run Stage A first: no {a_path.relative_to(ROOT)}")
    stage_a_report = json.loads(a_path.read_text(encoding="utf-8"))
    passed = stage_a_report.get("passed") or []
    if not passed:
        say("no arm passed Stage A, so there is no bundle. Stage C compares nothing; retrieval freezes as is.")
        return {"bundle": [], "dropped": [], "reason": "no arm passed Stage A"}

    scores = {k: max((stage_a_report["verdicts"][k].get("gains") or {}).values(), default=0.0) for k in passed}
    groups = arms_mod.groups(arms_mod.resolve(passed))
    bundle, aside = rule.pick_within_groups(passed, groups, scores)
    say(f"bundle after resolving exclusive groups: {bundle}")
    for group, others in aside.items():
        say(f"  {group}: kept the best, set aside {others}")

    # The drop-one check needs the bundle searched, and each bundle-minus-one searched.
    variants = {"bundle": bundle}
    if len(bundle) > 1:
        variants.update({f"without_{k}": [x for x in bundle if x != k] for k in bundle})
    say(f"\ndrop-one needs {len(variants)} configurations searched")
    run_async(search_bundles(args, variants))

    items = {i.id: i for i in load_gold()}
    base = load_outcomes(BASELINE, args.split)
    bundle_outcomes = load_outcomes("bundle", args.split)
    if base is None or bundle_outcomes is None:
        raise SystemExit("the baseline or the bundle was not searched")

    bundle_metrics = primaries(bundle_outcomes, judged_map(items, bundle_outcomes), condensed=True)
    without: dict[str, dict[str, float | None]] = {}
    for key in bundle:
        got = load_outcomes(f"without_{key}", args.split)
        if got is not None:
            without[key] = primaries(got, judged_map(items, got), condensed=True)
    droppable, moved = rule.drop_one(bundle_metrics, without)

    final = [k for k in bundle if k not in droppable]
    for key in droppable:
        say(f"  {key:6s} dropped from the bundle: removing it moved every metric by < {rule.NOOP_EPSILON}")
    say(f"\nfinal bundle: {final or 'empty'}")

    base_metrics = primaries(base, judged_map(items, base), condensed=True)
    report = {"stage": "B", "split": args.split, "candidate_bundle": bundle, "set_aside": aside,
              "bundle_metrics_condensed": bundle_metrics, "baseline_metrics_condensed": base_metrics,
              "drop_one_moved": moved, "dropped": droppable, "final_bundle": final,
              "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
    out = REPORTS / f"e1_stage_b_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    say(f"wrote {out.relative_to(ROOT)}")
    return report


async def search_bundles(args, variants: dict[str, list[str]]) -> None:
    """Search each named bundle variant, caching it like an arm."""
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database)
    items = [i for i in load_gold() if i.split == args.split]
    queries = [tuning.GoldQuery.of(i) for i in tuning.scorable(items)]
    if args.limit:
        queries = queries[: args.limit]

    async with make_pool(dsn, served=True) as db:
        recipes = await tuning.corpus_recipes(db)
        corpus_chunks = recipes[tuning.require_coherent_corpus(recipes)]
        searcher = HybridSearcher(db, GpuClient())
        await searcher.refresh()
        needs_stock = any("B1" in keys for keys in variants.values())
        if needs_stock:
            from build_stock_sentences import load as load_stock

            searcher.load_stock_sentences(load_stock(dsn))
        for name, keys in variants.items():
            if not args.refresh and load_outcomes(name, args.split, corpus_chunks) is not None:
                say(f"  {name:18s} cached")
                continue
            config = arms_mod.bundle(arms_mod.resolve(keys), baseline_config()) if keys else baseline_config()
            got = await run_variant(searcher, queries, Variant(name, config, name), top_k=TOP_K)
            save_outcomes(name, args.split, got, corpus_chunks)
            say(f"  {name:18s} searched")


def stage_c(args) -> dict:
    """Stage C: the final bundle against the baseline on TEST, once (§4.6).

    The only use of the test split in E1 that decides anything, and it is guarded: `tuning.guard_split`
    refuses it without `--confirm`. Paired bootstrap at 10,000 resamples with Holm-Bonferroni across the
    confirmatory family, and a p below the bootstrap floor is printed as an inequality rather than as 0.0000.
    """
    tuning.guard_split("test", confirm=args.confirm)
    b_path = REPORTS / "e1_stage_b_dev.json"
    if not b_path.exists():
        raise SystemExit("run Stage B on dev first")
    final = json.loads(b_path.read_text(encoding="utf-8")).get("final_bundle") or []
    if not final:
        say("Stage B left an empty bundle, so there is nothing to confirm. Retrieval freezes as it is.")
        return {"stage": "C", "bundle": [], "adopt": [], "reason": "no arm survived Stage B",
                "comparisons": {}}  # fmt: skip

    args.split = "test"
    run_async(search_bundles(args, {"bundle": final}))
    # (items not needed here: the metrics come from the cached outcomes)
    base = load_outcomes(BASELINE, "test")
    if base is None:
        run_async(search_arms(args, []))
        base = load_outcomes(BASELINE, "test")
    arm = load_outcomes("bundle", "test")
    if base is None or arm is None:
        raise SystemExit("the baseline or the bundle was not searched on test")

    comparisons, means, established, worse = {}, {}, {}, []
    for metric, k in (("recall", 10), ("mrr", 10)):
        name = f"{metric}@{k}"
        comparison = tuning.compare(base, arm, f"bundle vs baseline, {name}", metric=metric, k=k,
                                    iterations=args.iterations)  # fmt: skip
        comparisons[name] = comparison
    tuning.holm_bonferroni(list(comparisons.values()), alpha=args.alpha)
    for name, comparison in comparisons.items():
        means[name] = comparison.mean
        established[name] = bool(comparison.established)
        if comparison.established and comparison.mean < 0:
            worse.append(name)
        say(f"  {comparison.line()}   ({rule.floor_p(comparison.p, args.iterations)})")

    adoption = rule.decide(final, established, means, worse)
    say(f"\nDECISION: {adoption.reason}")
    report = {
        "stage": "C", "split": "test", "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bundle": final, "iterations": args.iterations, "alpha": args.alpha,
        "comparisons": {n: {"mean": c.mean, "low": c.interval.low, "high": c.interval.high, "p": c.p,
                            "p_reported": rule.floor_p(c.p, args.iterations),
                            "adjusted_alpha": c.adjusted_alpha, "established": c.established}
                        for n, c in comparisons.items()},  # fmt: skip
        "adopt": adoption.adopt, "reason": adoption.reason, "established": adoption.established,
        "significantly_worse": adoption.worse,
        "note": "The test split has now been used for E1's confirmation. Any further use is a second look.",
    }
    out = REPORTS / "e1_stage_c.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    say(f"wrote {out.relative_to(ROOT)}")
    return report


def freeze(args) -> dict:
    """Apply the adoption mechanically and freeze retrieval (§4.7 step 8).

    The frozen configuration is written to `config/retrieval_frozen.json` with a hash over its own fields, so
    a later change is visible as a hash change rather than as a diff nobody reads. This does NOT re-decide
    anything: it reads Stage C's decision and applies it.
    """
    c_path = REPORTS / "e1_stage_c.json"
    if not c_path.exists():
        raise SystemExit("run `confirm` first: there is no Stage C decision to apply")
    decision = json.loads(c_path.read_text(encoding="utf-8"))
    adopt = decision.get("adopt") or []

    config = arms_mod.bundle(arms_mod.resolve(adopt), baseline_config()) if adopt else baseline_config()
    fields = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(config).items()
              if k != "filters"}  # fmt: skip
    payload = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adopted_arms": adopt,
        "reason": decision.get("reason", ""),
        "embedding_recipe": "BAAI/bge-small-en-v1.5|full|v2",
        "config": fields,
        "note": ("Retrieval is frozen. After this it changes only for a bug fix, with the dev harness re-run "
                 "and the change logged (endgame.md §3). The hash below names this exact configuration."),
    }
    payload["config_hash"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    out = ROOT / "config" / "retrieval_frozen.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    say(f"adopted: {adopt or 'nothing'}")
    say(f"config hash: {payload['config_hash']}")
    say(f"wrote {out.relative_to(ROOT)}  — RETRIEVAL IS FROZEN")
    return payload


def build_report(args) -> dict:
    """`reports/e1_report.md` (§4.8), from the stage reports on disk."""
    def read(name):
        path = REPORTS / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    a, b, c = read("e1_stage_a_dev.json"), read("e1_stage_b_dev.json"), read("e1_stage_c.json")
    frozen_path = ROOT / "config" / "retrieval_frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists() else {}
    audit = read("e1_w5_item_audit.json")
    recheck = read("e1_judge_recheck.json")

    def fmt(v):
        return "-" if v is None else f"{v:.4f}"

    L = [
        "# E1 report: Phase 2 closed, retrieval frozen",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Session 15. Sources: `reports/e1_*.json`, "
        "`config/retrieval_frozen.json`.",
        "",
        "## The decision",
        "",
        f"**Adopted: {', '.join(frozen.get('adopted_arms') or []) or 'nothing'}.** "
        + (c.get("reason") or "Stage C did not run."),
        "",
        f"Frozen configuration hash: **`{frozen.get('config_hash', '—')}`**, recipe "
        f"`{frozen.get('embedding_recipe', '—')}`. Retrieval now changes only for a bug fix, with the dev "
        "harness re-run and the change logged.",
        "",
        "## Stage A: the two-tier screen on dev",
        "",
        f"Corpus: {a.get('corpus_chunks', 0):,} chunks. {a.get('queries', 0)} scorable dev queries.",
        "",
        "**Tier 1 cost nothing.** Every arm was scored first on condensed-list metrics, which need no new "
        "judgments: unjudged documents are dropped from the ranking rather than assumed irrelevant. An arm that "
        "reorders nothing among judged documents cannot have reordered the raw list either, because the "
        "condensed list is a subsequence of it.",
        "",
        "| arm | dropped at tier 1 | verdict |",
        "|---|---|---|",
    ]
    dropped = a.get("tier1_dropped") or {}
    verdicts = a.get("verdicts") or {}
    for key in sorted(set(dropped) | set(verdicts)):
        if key in dropped:
            moved = ", ".join(f"{n} {v:+.4f}" for n, v in (dropped[key] or {}).items())
            L.append(f"| {key} | yes | no-op: {moved} |")
        else:
            v = verdicts[key]
            state = "PASS" if not v.get("failures") and not v.get("unscorable") else (
                "unscorable" if v.get("unscorable") else "fail")
            why = v.get("unscorable") or "; ".join(v.get("failures") or []) or "all four conditions hold"
            L.append(f"| {key} | no | **{state}** — {why[:150]} |")

    judging = a.get("judging") or {}
    if judging.get("spent"):
        L += ["", f"**Tier 2 was judged.** {judging.get('pairs', 0):,} unjudged (query, speech) pairs across the "
                  f"baseline and the survivors, **{judging.get('judged', 0):,} graded** "
                  f"({judging.get('no_reply', 0)} no reply, {judging.get('unparsable', 0)} unparsable) for "
                  f"**${a.get('spend_usd', 0.0):.2f}**. Served model: "
                  f"{(judging.get('provenance') or {}).get('served_models')}. The grades are in "
                  "`data/uplift/extra_judgments.json` and are unioned in at read time — the committed gold "
                  "set's own hash is untouched, because every Phase 2 number cites it.", ""]
    else:
        L += ["", f"Tier 2 was **not** judged: {judging.get('pairs', 0):,} pairs were priced at "
                  f"${judging.get('estimated_usd', 0.0):.2f} and the arms were screened on the coverage the "
                  "existing pool already gives, so an arm below judged@10 0.95 reads as unscorable.", ""]

    L += ["## Stage B: the bundle and drop-one", ""]
    if b:
        L += [f"Candidate bundle after resolving the exclusive groups: **{b.get('candidate_bundle') or 'empty'}**.",
              f"Set aside: {b.get('set_aside') or 'none'}.",
              f"Dropped by drop-one (removal moved every metric by < {rule.NOOP_EPSILON}): "
              f"**{b.get('dropped') or 'none'}**.",
              f"Final bundle: **{b.get('final_bundle') or 'empty'}**.", ""]
    else:
        L += ["Stage B did not run.", ""]

    L += ["## Stage C: confirmation on the test split, once", ""]
    if c.get("comparisons"):
        L += ["| metric | mean | 95% CI | p | adjusted alpha | established |", "|---|---|---|---|---|---|"]
        for name, row in c["comparisons"].items():
            L.append(f"| {name} | {row['mean']:+.4f} | [{row['low']:+.4f}, {row['high']:+.4f}] | "
                     f"{row['p_reported']} | {row['adjusted_alpha']:.4f} | "
                     f"{'yes' if row['established'] else 'no'} |")
        L += ["", f"Paired bootstrap, {c.get('iterations')} resamples, Holm-Bonferroni across the family. "
                  "A p below the bootstrap floor is reported as an inequality, never as 0.0000.", ""]
    else:
        L += [c.get("reason") or "Stage C did not run.", ""]

    L += ["## The judge", ""]
    if recheck:
        L += [f"Re-checked on {recheck.get('sample', {}).get('pairs', 0)} stratified pairs. "
              f"**Binarised kappa {recheck.get('binarised_kappa')}** (the gate), four-level "
              f"{recheck.get('four_level_kappa')} (diagnostic; I-77). Verdict: "
              f"**{str(recheck.get('verdict')).upper()}**.",
              "", f"Served model: {(recheck.get('provenance') or {}).get('served_models')}, fingerprint "
                  f"{(recheck.get('provenance') or {}).get('system_fingerprints')}. The judge configuration is "
                  "frozen from here.", ""]

    L += ["## W5: the sealed exam window", ""]
    if audit:
        census = audit.get("census", {})
        L += [f"`tests/data/gold_w5.json`. Authored items audited: **{audit.get('clean')} clean, "
              f"{audit.get('flagged')} flagged, {audit.get('blocking')} blocking** of {census.get('items')}.",
              "", "Every anchor resolves in W5 and every quote comes out of the database. Adversarial premises "
                  "were checked against their own evidence, and abstention items were searched across the whole "
                  "window to test their claim of absence.", ""]

    scorecard = read("e1_frozen_scorecard.json")
    if scorecard:
        L += ["## The frozen scorecard", "",
              "Measured on the frozen configuration over the W5-augmented corpus (1,422,379 embedded chunks), "
              "against the pool as extended by Stage A's judging.", "",
              "| split | queries | R@10 | MRR@10 | nDCG@10 | R@10 condensed | MRR@10 condensed |",
              "|---|---|---|---|---|---|---|"]
        for split, row in scorecard.items():
            raw, con = row.get("raw", {}), row.get("condensed", {})
            L.append(f"| {split} | {row.get('queries')} | {fmt(raw.get('recall@10'))} | "
                     f"{fmt(raw.get('mrr@10'))} | {fmt(raw.get('ndcg@10'))} | "
                     f"{fmt(con.get('recall@10'))} | {fmt(con.get('mrr@10'))} |")
        L += ["",
              "**These are not comparable with the pre-E1 numbers, and the reason matters.** Stage A added "
              "1,820 judgments to the pool. Many found relevant speeches nobody had judged, which enlarges "
              "|R| -- the denominator of recall. So dev R@10 reads 0.4930 here against 0.5681 before, while "
              "MRR@10 rose from 0.4946 to 0.5119. Recall did not get worse: **the instrument got more "
              "honest.** We now know more of what we were missing, and more of what we rank highly is known "
              "to be relevant. A number measured against a larger pool is a different number.",
              "",
              "The condensed figures are close to the raw ones on dev (0.4938 against 0.4930) and further "
              "apart on test (0.4979 against 0.4430), which locates the remaining blind spot: the test "
              "split's pool is still the thinner of the two.",
              ""]

    L += [
        "## What is not done",
        "",
        "- **Incremental judging** for Stage A tier 2. Arms were screened on the coverage the existing pool "
        "gives them; an arm below judged@10 0.95 is reported unscorable rather than guessed at.",
        "- **§4.1 items 3, 4, 6, 8, 9, 10** — see `reports/e1_audit.md` §5.",
        "- **Arms C1, D2, Q1, N1, A1, O1, R2** — see `arms.NOT_IMPLEMENTED`.",
        "",
        "## Handover to E2",
        "",
        "- The **evidence unit** is a speech with its matched sentences (`collapse.Cluster`).",
        "- The **debate ranking** is `arms.rank_debates`, MaxP over the fused pool — used by who_spoke and the "
        "overview route.",
        f"- The frozen configuration is `config/retrieval_frozen.json`, hash `{frozen.get('config_hash', '—')}`. "
        "The answer cache must key on it.",
    ]
    out = REPORTS / "e1_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    say(f"wrote {out.relative_to(ROOT)}")
    return {"report": str(out.relative_to(ROOT)), "adopted": frozen.get("adopted_arms") or []}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(parser):
        parser.add_argument("--database", default="hansard_v2")
        parser.add_argument("--split", default="dev", choices=("dev", "test"))
        parser.add_argument("--arms", default="", help="comma-separated arm keys (default: every implemented arm)")
        parser.add_argument("--limit", type=int, default=0, help="first N queries only, for a smoke run")
        parser.add_argument("--cap", type=float, default=rule.SPEND_CAP_USD)
        parser.add_argument("--refresh", action="store_true", help="re-search even if a cache exists")
        parser.add_argument("--no-judge", action="store_true",
                            help="screen on existing coverage only, spending nothing")  # fmt: skip
        parser.add_argument("--reasoning", default="", help="the judge's reasoning_effort")
        parser.add_argument("--json", action="store_true")

    listing = sub.add_parser("list", help="the arms, and what each needs")
    listing.add_argument("--json", action="store_true")
    runner = sub.add_parser("run", help="Stage A: the two-tier screen")
    runner.add_argument("--stage", default="a", choices=("a",))
    common(runner)
    bundler = sub.add_parser("bundle", help="Stage B: the bundle and the drop-one check")
    common(bundler)
    confirmer = sub.add_parser("confirm", help="Stage C: the bundle against the baseline on TEST, once")
    common(confirmer)
    confirmer.add_argument("--confirm", action="store_true",
                           help="required: the test split is used once and the use is logged")  # fmt: skip
    confirmer.add_argument("--iterations", type=int, default=10_000)
    confirmer.add_argument("--alpha", type=float, default=0.05)
    freezer = sub.add_parser("freeze", help="apply Stage C's decision and freeze retrieval")
    common(freezer)
    amender = sub.add_parser("amend", help="change named knobs of frozen retrieval, with the reason and evidence "
                                           "recorded in the file (answer.pipeline.amend_frozen)")
    amender.add_argument("--set", action="append", required=True, metavar="KNOB=VALUE",
                         help="VALUE is read as JSON when it parses (120, true, null), else as a string")
    amender.add_argument("--reason", required=True)
    amender.add_argument("--evidence", required=True, help="the measurement it rests on (a report path and numbers)")
    amender.add_argument("--json", action="store_true")
    reporter = sub.add_parser("report", help="write reports/e1_report.md")
    common(reporter)

    args = p.parse_args(argv)

    if args.command == "list":
        rows = [{"arm": a.key, "title": a.title, "group": a.group or "-", "target": a.target,
                 "requires": a.requires or "-", "changes": a.changes} for a in arms_mod.ARMS]  # fmt: skip
        payload = {"implemented": rows, "not_implemented": arms_mod.NOT_IMPLEMENTED}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for row in rows:
                say(f"  {row['arm']:6s} {row['title']:38s} group {row['group']:7s} -> {row['changes']}")
                if row["requires"] != "-":
                    say(f"         needs {row['requires']}")
            say("\n  in the plan, not implemented:")
            for key, why in sorted(arms_mod.NOT_IMPLEMENTED.items()):
                say(f"  {key:6s} needs {why}")
        return 0

    if args.command == "amend":
        from answer.pipeline import FROZEN_PATH, amend_frozen

        changes = {}
        for item in args.set:
            knob, _, raw = item.partition("=")
            try:
                changes[knob.strip()] = json.loads(raw)
            except json.JSONDecodeError:
                changes[knob.strip()] = raw
        entry = amend_frozen(FROZEN_PATH, changes, reason=args.reason, evidence=args.evidence)
        say(f"amended {FROZEN_PATH.relative_to(ROOT)}: {entry['changed']}")
        say(f"config hash: {entry['previous_hash']} -> {entry['config_hash']}")
        if args.json:
            print(json.dumps(entry, indent=2))
        return 0

    handlers = {"run": stage_a, "bundle": stage_b, "confirm": stage_c, "freeze": freeze,
                "report": build_report}  # fmt: skip
    report = handlers[args.command](args)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
