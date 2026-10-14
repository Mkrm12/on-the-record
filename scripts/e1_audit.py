"""E1 section 4.1: the measurement audit. Writes `reports/e1_audit.md`, plain-English findings first."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from goldset.schema import load as load_gold  # noqa: E402
from retrieval import audit as ra  # noqa: E402
from retrieval.metrics import recall_at_k  # noqa: E402

REPORTS = ROOT / "reports"
OUT = REPORTS / "e1_audit.md"
FINAL = REPORTS / "session3_final.json"

# Used to price the incremental judging, not guessed.
COST_PER_JUDGMENT = 0.49 / 3959
SPLITS = ("dev", "test", "heldout")
SPLIT_NAMES = {"dev": "dev", "test": "test", "heldout": "W3 (burned)"}


class Outcome:
    __slots__ = ("qid", "relevant", "ranked", "category")

    def __init__(self, row):
        self.qid = row["qid"]
        self.relevant = frozenset(row["relevant"])
        self.ranked = list(row["ranked"])
        self.category = row.get("category") or "uncategorised"


def fmt(value, places=4):
    return "-" if value is None else f"{value:.{places}f}"


def load() -> tuple[dict, dict, dict[str, list[Outcome]]]:
    items = {i.id: i for i in load_gold()}
    report = json.loads(FINAL.read_text(encoding="utf-8"))
    outcomes = {s: [Outcome(r) for r in (report["splits"].get(s, {}).get("outcomes") or [])] for s in SPLITS}
    return items, report, {s: o for s, o in outcomes.items() if o}


def judged_map(items: dict, outcomes: list[Outcome]) -> dict[str, set[str]]:
    """query id -> every speech carrying a judgment, relevant or not."""
    out = {}
    for o in outcomes:
        item = items.get(o.qid)
        graded = set((getattr(item, "graded_relevance", None) or {}).keys()) if item else set()
        out[o.qid] = graded | set(o.relevant)
    return out


def unjudged_pairs(items: dict, outcomes: list[Outcome], depth: int) -> int:
    judged = judged_map(items, outcomes)
    return sum(1 for o in outcomes for s in o.ranked[:depth] if s not in judged[o.qid])


async def live_lengths(limit: int) -> tuple[ra.ListLengths | None, tuple[float, int, int], int, tuple[int, str]]:
    """The length and wasted-slot audit of the list production actually returns."""
    from app.config import settings
    from app.db import make_pool
    from psycopg.conninfo import make_conninfo
    from retrieval import tuning
    from retrieval.gpu_client import GpuClient
    from retrieval.hybrid import HybridSearcher
    from session3_tune import baseline_config

    items = [i for i in load_gold() if i.split == "dev"]
    queries = [tuning.GoldQuery.of(i) for i in tuning.scorable(items)][:limit]
    lists: list[list[str]] = []
    async with make_pool(make_conninfo(settings.pg_dsn, dbname="hansard_v2"), served=False) as db:  # served=False: an audit of the whole record
        searcher = HybridSearcher(db, GpuClient())
        await searcher.refresh()
        for q in queries:
            result = await searcher.search(q.question, config=baseline_config())
            lists.append([h.speech_id for h in result.hits])
    worst = max(((ra.wasted_slots(sl, 10), q.id) for sl, q in zip(lists, queries)), default=(0, ""))
    return ra.final_list_lengths(lists), ra.wasted_share(lists, 10), len(lists), worst


def render(items: dict, report: dict, outcomes: dict[str, list[Outcome]], live) -> str:
    audits = {s: ra.audit_arm("baseline", s, o, judged_map(items, o)) for s, o in outcomes.items()}
    dev = audits.get("dev")
    bill = sum(unjudged_pairs(items, outcomes[s], 20) for s in ("dev", "test") if s in outcomes)

    L = [
        "# E1 measurement audit",
        "",
        f"Session 15, {report['scored_at'][:10]}. Source: `reports/session3_final.json` (the frozen Phase 2",
        "scorecard) and the shipped gold set. Produced by `scripts/e1_audit.py`; the diagnostics live in",
        "`src/retrieval/audit.py` and are unit tested in `tests/test_retrieval_audit.py`.",
        "",
        "## What this found, in plain words",
        "",
        "**1. The set-type weakness is real. It is not a metric artefact.** The endgame plan (§2) and the",
        "Session 3 report both suspected that `multi_speech` and `who_spoke` scored badly partly because",
        "recall@10 divides by the number of relevant speeches, and a query with more than ten answers cannot",
        "score 1.0. Measured: **attainable R@10 is 1.0000 for both**, because `multi_speech` averages 3.9",
        "relevant speeches and `who_spoke` averages 1.1. Across all 237 retrieval-testable items attainable",
        f"R@10 is {fmt(dev.attainable_recall_at_10 if dev else None)} on dev. **None of the weakness is the",
        "metric.** E1 should not spend anything on normalising it away, and Phase 3's routing work is aimed",
        "at a real gap.",
        "",
        "**2. `who_spoke` has 1.1 relevant speeches per question, so the gold set does not label a roster at",
        "all.** That breaks §4.1 item 3 as written: roster recall has no gold to measure against. Either the",
        "roster is derived (every speaker in the debate the relevant speech belongs to) and that derivation is",
        "declared as gold, or the metric cannot be computed. Logged as a plan defect, not silently skipped.",
        "",
        "**3. About half of what we return has never been judged, and the metric calls all of it wrong.** The",
        "pool was built by three configurations (`dense`, `hybrid`, `full`), all variants of today's",
        "architecture, with a median of 16 judged speeches per item. Anything outside that pool is unjudged,",
        "and an unjudged result scores as a miss. **This is a systematic bias against exactly what E1 exists",
        "to test:** a new arm that surfaces relevant material the pool never saw measures worse, not better.",
        "",
        "**4. The number we have been calling `recall@100` is recall of a ~19-speech list.** `pool=20` caps",
        "it. Renamed **R@final** in every new report, with the length printed beside it.",
        "",
        f"**5. The incremental judging in §4.7 costs about ${bill * COST_PER_JUDGMENT:.2f} for the baseline",
        f"alone** ({bill:,} unjudged pairs at depth 20 across dev and test, priced from Session 2's measured",
        "$0.49 for 3,959 judgments). Fourteen arms at 50% union overlap add roughly $2.00, so about $2.28 in",
        "total. **That fits both the $4 cap and the $3.51 balance** -- what it does not leave is margin: about",
        "$1.23 for the judge re-check, a full re-judge if that fails, and every LLM call in E2 to E4. Six arms",
        "cost about $0.85. The two-tier screen is about margin and waste, not impossibility.",
        "",
        "## 1. Judgment coverage and the condensed-list sensitivity check (§4.1 item 7)",
        "",
        "`judged@k` is the share of the top k carrying any grade. A condensed list drops unjudged documents",
        "and scores the rest, so the pair brackets the truth: the raw number assumes every unknown is",
        f"irrelevant, the condensed one assumes nothing. An arm is only scored at judged@10 >= {ra.JUDGED_GATE:.2f}.",
        "",
        "| split | queries | judged@10 | judged@20 | R@10 | R@10 condensed | MRR@10 | MRR@10 condensed | scorable |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        a = audits.get(s)
        if not a:
            continue
        L.append(f"| {SPLIT_NAMES[s]} | {a.queries} | {fmt(a.judged_at_10)} | {fmt(a.judged_at_20)} | "
                 f"{fmt(a.recall_at_10)} | {fmt(a.condensed_recall_at_10)} | {fmt(a.mrr_at_10)} | "
                 f"{fmt(a.condensed_mrr_at_10)} | {'yes' if a.scorable else '**NO**'} |")
    L += [
        "",
        "**The baseline itself is below the gate.** That is not a bug in the gate: it is the gate telling the",
        "truth about the pool. Nothing can be compared until §4.7's incremental judging has run, which is why",
        "the plan sequences judging before scoring.",
        "",
        "## 2. Relevant-set sizes, attainable and normalised recall, R-precision (§4.1 item 2)",
        "",
        "| split | mean \\|R\\| | attainable R@10 | R@10 | normalised R@10 | R-precision |",
        "|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        a = audits.get(s)
        if not a:
            continue
        L.append(f"| {SPLIT_NAMES[s]} | {a.mean_relevant:.1f} | {fmt(a.attainable_recall_at_10)} | "
                 f"{fmt(a.recall_at_10)} | {fmt(a.normalised_recall_at_10)} | {fmt(a.r_precision)} |")

    by_cat = defaultdict(list)
    for s, rows in outcomes.items():
        for o in rows:
            if o.relevant:
                got = recall_at_k(o.relevant, o.ranked, 10) or 0.0
                by_cat[o.category].append((got, ra.attainable_recall(len(o.relevant), 10), len(o.relevant)))
    L += [
        "",
        "### By category, pooled over all three splits",
        "",
        "`normalised` is what the system achieved as a share of what the metric could reach. Where it equals",
        "R@10, the ceiling was 1.0 and the score is entirely the system's.",
        "",
        "| category | n | mean \\|R\\| | R@10 | attainable | normalised |",
        "|---|---|---|---|---|---|",
    ]
    for cat, vals in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        n = len(vals)
        got, att = sum(v[0] for v in vals) / n, sum(v[1] for v in vals) / n
        rel = sum(v[2] for v in vals) / n
        L.append(f"| {cat} | {n} | {rel:.1f} | {fmt(got)} | {fmt(att)} | {fmt(got / att if att else None)} |")

    L += ["", "## 3. The final list: length and wasted slots (§4.1 items 1 and 5)", ""]
    if live:
        lengths, (share, wasted, total), n, (worst_n, worst_q) = live
        L += [
            f"Measured live on {n} dev queries with the adopted configuration, because the stored outcomes",
            "cannot answer this: `evaluation.speech_ranking` deduplicates to speech level before the metric",
            "sees the list, so a zero duplicate rate there proves the metric dedupes and nothing about the",
            "product.",
            "",
            f"* **length:** {lengths.minimum} to {lengths.maximum}, median {lengths.median} "
            f"(mean {lengths.mean:.1f}). This is a default `search()`. The evaluation harness raises `top_k`",
            "  to 100, and `pool=20` then caps it at about 19 distinct speeches -- two different lists, and",
            "  both get quoted as though they were one.",
            f"* **the honest label at depth 100 is `{lengths.label(100)}`**: {lengths.note(100)}.",
            f"* **wasted slots:** {wasted} of {total} top-10 slots repeat a speech already listed higher up "
            f"= **{share:.2%}**. Worst single query: {worst_n} of 10"
            + (f" (`{worst_q}`)." if worst_q else "."),
            "",
            ("**Arm S1 ('one slot per speech') is therefore NOT a no-op**: it has "
             f"{share:.1%} of top-10 slots to reclaim. §4.3 says to skip an arm production already does; "
             "this one it does not."
             if wasted else
             "**Arm S1 ('one slot per speech') is a NO-OP**: production already shows each speech once. "
             "Log it and do not build the arm (§4.3's own instruction for O1)."),
        ]
    else:
        L += ["Not measured in this run. Re-run with `--live` (the GPU service must be up): the stored",
              "outcomes are already speech-deduplicated and cannot answer it."]

    L += [
        "",
        "## 4. The incremental judging bill (§4.7 steps 4, 5 and 7)",
        "",
        "| scope | unjudged pairs | cost at Session 2's measured rate |",
        "|---|---|---|",
    ]
    for s in SPLITS:
        if s in outcomes:
            for depth in (10, 20):
                pairs = unjudged_pairs(items, outcomes[s], depth)
                L.append(f"| {SPLIT_NAMES[s]}, depth {depth} | {pairs:,} | ${pairs * COST_PER_JUDGMENT:.2f} |")
    L += [
        f"| **baseline, dev + test, depth 20** | **{bill:,}** | **${bill * COST_PER_JUDGMENT:.2f}** |",
        "",
        "Each additional arm re-ranks the same corpus, so most of its top-20 is already in the union. At a",
        "50% overlap assumption, fourteen arms add about "
        f"{int(bill * 14 * 0.5):,} judgments (${bill * 14 * 0.5 * COST_PER_JUDGMENT:.2f}); six arms add about "
        f"{int(bill * 6 * 0.5):,} (${bill * 6 * 0.5 * COST_PER_JUDGMENT:.2f}).",
        "",
        "## 5. What section 4.1 still owes",
        "",
        "Built and tested here: items 1, 2, 5 and 7. Not yet built, and honest about it:",
        "",
        "| item | what it needs |",
        "|---|---|",
        "| 3, debate-level metrics and roster recall | a debate ranking (MaxP over the fused pool) and a "
        "declared gold roster — see finding 2, the gold set has no roster to measure against |",
        "| 4, distinct-speech ceiling | a ceiling run with the reranker off, counted in speeches |",
        "| 6, stock sentences | a materialised normalised-text frequency table over 1.07M chunks; it must be "
        "built once, not recomputed per query |",
        "| 8, breakdown by provenance | present for category and window; provenance needs the gold field |",
        "| 9, where-lost ladder | Session 3's tooling, re-pointed at the frozen config |",
        "| 10, latency p50/p95 per stage | a timed run on mains power |",
        "",
    ]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="also audit the live final list (needs the GPU service)")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args(argv)

    items, report, outcomes = load()
    live = None
    if args.live:
        from ingest.runtime import run

        live = run(live_lengths(args.limit))
    REPORTS.mkdir(exist_ok=True)
    OUT.write_text(render(items, report, outcomes, live), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
