"""Phase 2 gate: run the gold questions through the ladder of search configurations and report what each component
adds.
"""

import argparse
import json
import sys
from collections import defaultdict

from app.db import make_pool
from app.gold import load as load_gold
from ingest.runtime import run
from retrieval.evaluation import QueryOutcome, Variant, aggregate, auc, paired_differences, percentile, run_variant, score_outcome, standard_variants
from retrieval.gpu_client import GpuClient, GpuError
from retrieval.hybrid import HybridSearcher, SearchConfig, SearchFilters
from retrieval.metrics import bootstrap_paired

# The document-frequency ratio that reproduces the old absolute threshold (df >= 50) exactly on W0's 90,040 chunks and
# scales with the corpus everywhere else.
W0_COMMON_DF_RATIO = 50 / 90_040


def base_config(args: argparse.Namespace) -> SearchConfig:
    """The configuration every variant is built from."""
    filters = SearchFilters(
        windows=tuple(args.window or ()),
        houses=tuple(args.house or ()),
        sections=tuple(args.section or ()),
    )
    if not args.scale_fixes:
        return SearchConfig(filters=filters, ef_search=args.ef_search)
    return SearchConfig(
        filters=filters,
        ef_search=args.ef_search,
        common_df_ratio=W0_COMMON_DF_RATIO,
        speaker_date_aware=True,
        stored_lexicon=True,
        exact_scan_max_rows=args.exact_scan_max_rows,
    )

CATEGORIES = ("exact_term", "paraphrase", "numeric_fact", "named_speaker", "multi_speech")
COMPARISONS = [  # (question it answers, base, other)
    ("Does hybrid beat dense-only? (the Phase 3 gate)", "dense", "hybrid"),
    ("Did the keyword coverage threshold help?", "hybrid-naive", "hybrid"),
    ("Does the speaker leg help (I-20)?", "hybrid", "hybrid+speaker"),
    ("Does the reranker add value over the fused order? (I-36, Variant B vs A)", "hybrid", "hybrid+rerank"),
    ("... with the speaker leg too", "hybrid+speaker", "full"),
    ("Is blending better than letting the reranker replace the order?", "full-replace", "full"),
    ("Full pipeline against the dense baseline", "dense", "full"),
]


def cell(found: int, n: int) -> str:
    return f"{found}/{n}" if n else "-"


def print_table(results: dict[str, list[QueryOutcome]], variants: list[Variant], k: int) -> None:
    print(f"\nRecall@{k} by category (relevant speeches found in the top {k} distinct speeches / labelled), then overall")
    header = f"{'variant':16s}" + "".join(f"{c[:11]:>12s}" for c in CATEGORIES) + f"{'R@5':>8s}{'R@' + str(k):>8s}{'MRR':>8s}{'nDCG':>8s}"
    print(header)
    print("-" * len(header))
    for v in variants:
        t = aggregate(results[v.name], k)
        cells = "".join(f"{cell(t.by_category[c].found, t.by_category[c].n) if c in t.by_category else '-':>12s}" for c in CATEGORIES)
        o = t.overall
        print(f"{v.name:16s}{cells}{o.found5 / o.n:>8.0%}{o.recall:>8.0%}{o.mrr:>8.3f}{o.ndcg:>8.3f}")
    print(f"\n({sum(1 for o in next(iter(results.values())) if not o.abstain)} answerable questions, "
          f"{next(iter(aggregate(next(iter(results.values())), k).by_category.values())).n and sum(s.n for s in aggregate(next(iter(results.values())), k).by_category.values())} labelled speeches)")


def print_comparisons(results: dict[str, list[QueryOutcome]], k: int) -> None:
    print(f"\nPaired comparisons over the answerable questions: change in reciprocal rank and in recall@{k}, with a 95% bootstrap interval")
    print(f"{'comparison':78s} {'dMRR':>7s} {'95% CI':>17s} {'win/loss':>9s} {'dRecall':>8s}")
    for label, base, other in COMPARISONS:
        if base not in results or other not in results:
            continue
        rr = bootstrap_paired(paired_differences(results[base], results[other], "rr", k), seed=7)
        rc = paired_differences(results[base], results[other], "recall", k)
        verdict = "" if rr.low <= 0 <= rr.high else ("  <- established" if rr.mean else "")
        print(f"{label:78s} {rr.mean:>+7.3f} [{rr.low:>+6.3f},{rr.high:>+6.3f}] {rr.wins:>4d}/{rr.losses:<4d} {sum(rc) / len(rc):>+8.3f}{verdict}")
    print("   (an interval that spans zero: the difference is not established at this sample size)")


def print_pool(results: dict[str, list[QueryOutcome]]) -> None:
    base = results.get("hybrid+speaker")
    if not base:
        return
    print("\nCandidate-pool ceiling: relevant speeches present among the clusters the reranker is given (hybrid+speaker, top 20 clusters)")
    for cutoff in (5, 10, 20):
        found = sum(score_outcome(o, cutoff).found for o in base if not o.abstain)
        n = sum(score_outcome(o, cutoff).n for o in base if not o.abstain)
        print(f"  within {cutoff:>2d} speeches: {found}/{n} = {found / n:.0%}")


def print_latency(outcomes: list[QueryOutcome]) -> None:
    stages = defaultdict(list)
    for o in outcomes:
        for stage, ms in o.timings.items():
            stages[stage].append(ms)
    print("\nLatency of the full pipeline, per stage (ms), over the gold questions")
    print(f"{'stage':22s}{'p50':>9s}{'p95':>9s}{'max':>9s}")
    for stage in ("embed_and_tokenise", "fuse", "rerank", "expand", "total"):
        if stages[stage]:
            v = stages[stage]
            print(f"{stage:22s}{percentile(v, 50):>9.1f}{percentile(v, 95):>9.1f}{max(v):>9.1f}")


def print_abstain(outcomes: list[QueryOutcome]) -> None:
    pos = [o.top_rerank for o in outcomes if not o.abstain and o.top_rerank is not None]
    neg = [o.top_rerank for o in outcomes if o.abstain and o.top_rerank is not None]
    if not pos or not neg:
        return
    print("\nAbstention signal (for Block 14): the best reranker score among the results")
    print(f"  answerable questions: median {percentile(pos, 50):+.2f}, range [{min(pos):+.2f}, {max(pos):+.2f}]  (n={len(pos)})")
    print(f"  abstain questions:    median {percentile(neg, 50):+.2f}, range [{min(neg):+.2f}, {max(neg):+.2f}]  (n={len(neg)})")
    print(f"  AUC of the top score for telling them apart: {auc(pos, neg):.2f}  (0.5 = no signal; only {len(neg)} abstain questions, so indicative only)")


def print_per_query(results: dict[str, list[QueryOutcome]], k: int) -> None:
    print(f"\nPer question: rank of the first relevant speech (- = not in the top {k})")
    names = list(results)
    print(f"{'id':6s}{'category':15s}" + "".join(f"{n[:12]:>13s}" for n in names))
    for i, o in enumerate(next(iter(results.values()))):
        if o.abstain:
            continue
        row = []
        for n in names:
            ranked = results[n][i].ranked[:k]
            pos = next((j for j, s in enumerate(ranked, 1) if s in o.relevant), None)
            row.append(f"{pos or '-':>13}")
        print(f"{o.qid:6s}{o.category:15s}" + "".join(row))


async def main_async(args: argparse.Namespace) -> int:
    queries = load_gold(statuses=("verified",) if args.verified_only else ("provisional", "verified"))
    base = base_config(args)
    variants = [v for v in standard_variants(base) if not args.variants or v.name in args.variants]
    pool = make_pool(min_size=1, max_size=4, served=True)
    await pool.open(wait=True, timeout=10)
    try:
        async with GpuClient() as gpu:
            try:
                health = await gpu.health()
            except GpuError as exc:
                print(f"The GPU service is not reachable ({exc}). Start it: cd gpu; uv run python api.py", file=sys.stderr)
                return 2
            searcher = HybridSearcher(pool, gpu)
            await searcher.refresh()
            print(f"gold: {len(queries)} questions ({sum(q.status == 'verified' for q in queries)} verified, "
                  f"{sum(q.status == 'provisional' for q in queries)} provisional) | GPU service {health['status']} on {health['device']}")
            print(f"config: {'scale fixes ON' if args.scale_fixes else 'Block 11 defaults (old code, old weights)'}"
                  + (f" | filters {base.filters.as_dict()}" if base.filters else "")
                  + f" | common_df {searcher._common_df}")
            results: dict[str, list[QueryOutcome]] = {}
            for v in variants:
                results[v.name] = await run_variant(searcher, queries, v)
                failed = sum(1 for o in results[v.name] if any(w.startswith("search failed") for w in o.warnings))
                print(f"  ran {v.name:16s} {v.description}" + (f"  [{failed} FAILED]" if failed else ""))
    finally:
        await pool.close()

    print_table(results, variants, args.k)
    print_pool(results)
    print_comparisons(results, args.k)
    if "full" in results:
        print_latency(results["full"])
        print_abstain(results["full"])
    if args.per_query:
        print_per_query(results, args.k)
    if args.json:
        headline = {}
        for v in variants:
            t = aggregate(results[v.name], args.k)
            headline[v.name] = {"recall_at_5": round(t.overall.found5 / t.overall.n, 4) if t.overall.n else 0.0,
                                f"recall_at_{args.k}": round(t.overall.recall, 4),
                                "mrr": round(t.overall.mrr, 4), "ndcg": round(t.overall.ndcg, 4),
                                "found": t.overall.found, "labelled": t.overall.n}
        payload = {
            "config": {"scale_fixes": bool(args.scale_fixes), "k": args.k,
                       "filters": {k: str(v) for k, v in base.filters.as_dict().items()},
                       "questions": len(queries)},
            "variants": headline,
            "per_query": {n: [{"id": o.qid, "category": o.category, "relevant": sorted(o.relevant),
                               "ranked": o.ranked[:30], "abstain": o.abstain, "top_rerank": o.top_rerank,
                               "timings_ms": o.timings, "warnings": o.warnings} for o in outs]
                          for n, outs in results.items()},
        }  # fmt: skip
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(f"\nresults written to {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the gold set.")
    parser.add_argument("--k", type=int, default=10, help="the cut for recall and MRR (default %(default)s)")
    parser.add_argument("--variants", nargs="*", help="run only these variants (default: all)")
    parser.add_argument("--verified-only", action="store_true", help="score only owner-verified gold entries (the Phase 3 gate should)")
    parser.add_argument("--per-query", action="store_true", help="also print the rank of the first relevant speech per question")
    parser.add_argument("--json", help="write every ranking to this file")
    parser.add_argument("--scale-fixes", action="store_true",
                        help="turn on the Corpus v2 scale fixes (relative speaker threshold, "
                             "date-aware resolution, the stored lexicon)")
    parser.add_argument("--window", action="append", help="restrict the search to this window (repeatable)")
    parser.add_argument("--house", action="append", help="restrict the search to this House (repeatable)")
    parser.add_argument("--section", action="append", help="restrict the search to this section (repeatable)")
    parser.add_argument("--ef-search", type=int, default=100,
                        help="HNSW ef_search (default %(default)s). Raising it trades latency for recall; "
                             "the index is approximate, so this moves results on its own")
    parser.add_argument("--exact-scan-max-rows", type=int, default=500_000,
                        help="with --scale-fixes, scan the filtered set exactly below this many chunks")
    return run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
