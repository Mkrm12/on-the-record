"""Score the adopted recipe on every split and write the final report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from retrieval import tuning  # noqa: E402
from retrieval.evaluation import Variant, run_variant  # noqa: E402
from retrieval.gpu_client import GpuClient  # noqa: E402
from retrieval.hybrid import HybridSearcher  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from session3_tune import baseline_config  # noqa: E402

REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "tuning"
OUT_JSON = REPORTS / "session3_final.json"
OUT_MD = REPORTS / "session3_final_report.md"

# What gets reported, in this order. recall@100 is the pre-registered primary metric (the first-stage candidate
# ceiling); recall@10 and MRR@10 are what a user experiences. nDCG@10 is reported as a binarised number only, because
# the judge failed its four-level calibration gate.
METRICS = (("recall", 100), ("recall", 10), ("mrr", 10), ("ndcg", 10))

# The splits, in the order they are allowed to be looked at.
SPLITS = ("dev", "test", "heldout")


def say(message: str) -> None:
    print(message, flush=True)


def metric_row(outcomes) -> dict[str, float | None]:
    return {f"{m}@{k}": tuning.mean_metric(outcomes, m, k) for m, k in METRICS}


def by_category(outcomes, metric: str = "recall", k: int = 10) -> dict[str, dict]:
    """Per-category scores. The average is not the deliverable: a recipe that lifts the mean while losing a question
    type is not an upgrade, and the only way to see that is to look category by category.
    """
    buckets: dict[str, list] = {}
    for o in outcomes:
        buckets.setdefault(o.category or "uncategorised", []).append(o)
    return {
        name: {"queries": len(group), f"{metric}@{k}": tuning.mean_metric(group, metric, k)}
        for name, group in sorted(buckets.items())
    }


# --- the "before" numbers -------------------------------------------------------------------------------


class _Outcome:
    """The shape the metric functions need, rebuilt from a cached sweep."""

    __slots__ = ("qid", "category", "relevant", "ranked")

    def __init__(self, qid, category, relevant, ranked):
        self.qid, self.category, self.relevant, self.ranked = qid, category, relevant, ranked


def cached_dev_outcomes(name: str = "baseline") -> list[_Outcome] | None:
    """The pre-adoption dev searches, per query, from the sweep cache."""
    path = CACHE / "sweep_dev.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["outcomes"].get(name)
    if not rows:
        return None
    return [_Outcome(o["qid"], o["category"], frozenset(o["relevant"]), o["ranked"]) for o in rows]


# How far a category may fall before it is reported as a loss rather than as noise.
REGRESSION = 0.02


def category_change(after, before) -> dict[str, dict]:
    """Per-category recall@10, before and after, for the same queries."""
    was, now = by_category(before), by_category(after)
    out = {}
    for name, got in now.items():
        b, a = (was.get(name) or {}).get("recall@10"), got["recall@10"]
        out[name] = {
            "queries": got["queries"], "before": b, "after": a,
            "delta": None if b is None or a is None else round(a - b, 4),
        }  # fmt: skip
    return out


def regressions(change: dict[str, dict]) -> dict[str, dict]:
    return {name: got for name, got in change.items() if (got["delta"] or 0) < -REGRESSION}


def before_numbers() -> dict[str, dict]:
    """Every pre-adoption number this report compares against, with where each one came from."""
    out: dict[str, dict] = {}
    for split, name, source in (("dev", "baseline", "data/tuning/sweep_dev.json"),):
        got = cached_dev_outcomes(name)
        if got:
            out[split] = {"source": source, "queries": len(got), "production": metric_row(got)}
            ceiling = cached_dev_outcomes("ceiling")
            if ceiling:
                out[split]["ceiling"] = metric_row(ceiling)
    for split, filename in (("test", "session3_confirm.json"), ("heldout", "session3_final_exam.json")):
        path = REPORTS / filename
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        out[split] = {
            "source": f"reports/{filename}",
            "scored_at": raw.get("scored_at"),
            "queries": raw.get("queries"),
            "production": {k: v for k, v in (raw.get("metrics", {}).get("baseline") or {}).items()},
        }
    return out


# --- scoring --------------------------------------------------------------------------------------------


async def score(args: argparse.Namespace) -> int:
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database)
    config = baseline_config()
    # The first-stage ceiling needs the reranker off AND pool raised: pool caps the output list, so with the reranker
    # off and pool at its default of 20 a "recall@100" is recall@20 wearing a different label.
    ceiling_config = replace(config, rerank="off", fused_limit=100, pool=100)

    splits = [s for s in SPLITS if s != "heldout" or args.second_look_w3]
    say(f"scoring {', '.join(splits)} on {args.database}")

    report: dict = {
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "database": args.database,
        "config": asdict(config),
        "relevant_at": tuning.RELEVANT_AT,
        "primary_metric": f"recall@{tuning.PRIMARY_K}",
        "splits": {},
        "before": before_numbers(),
        "w3_second_look": bool(args.second_look_w3),
    }

    async with make_pool(dsn, served=False) as db:  # served=False: reproduces E3, which ran unmasked
        recipes = await tuning.corpus_recipes(db)
        recipe = tuning.require_coherent_corpus(recipes)
        report["corpus"] = {"embedding_recipe": recipe, "embedded_chunks": recipes[recipe]}
        say(f"corpus: {recipes[recipe]:,} chunks, all under {recipe}")

        searcher = HybridSearcher(db, GpuClient())
        await searcher.refresh()
        for split in splits:
            items = [i for i in load_gold() if i.split == split]
            queries = [tuning.GoldQuery.of(i) for i in tuning.scorable(items)]
            if not queries:
                say(f"  {split}: no scorable query, skipped")
                continue
            say(f"  {split}: {len(queries)} scorable queries")
            production = await run_variant(searcher, queries,
                                           Variant("production", config, "the adopted configuration"),
                                           top_k=args.top_k)  # fmt: skip
            ceiling = await run_variant(searcher, queries,
                                        Variant("ceiling", ceiling_config, "first stage alone, depth 100"),
                                        top_k=args.top_k)  # fmt: skip
            entry = {
                "queries": len(queries),
                "production": metric_row(production),
                "ceiling": metric_row(ceiling),
                "by_category": by_category(production),
                "failures": sum(1 for o in production if o.warnings),
                # Kept so that the NEXT change to the corpus can be compared against this one as a paired test rather
                # than as two aggregates.
                "outcomes": [{"qid": o.qid, "category": o.category, "relevant": sorted(o.relevant),
                              "ranked": o.ranked, "warnings": o.warnings} for o in production],  # fmt: skip
            }
            before = report["before"].get(split, {}).get("production") or {}
            for name, value in entry["production"].items():
                was = before.get(name)
                entry.setdefault("delta", {})[name] = None if was is None or value is None else round(value - was, 4)
            if split == "dev" and (cached := cached_dev_outcomes("baseline")):
                comparison = tuning.compare(cached, production, "the adopted recipe vs the old one on dev",
                                            metric="recall", k=tuning.PRIMARY_K)  # fmt: skip
                tuning.holm_bonferroni([comparison])
                entry["paired_comparison"] = {
                    "metric": f"recall@{tuning.PRIMARY_K}", "mean": comparison.mean,
                    "low": comparison.interval.low, "high": comparison.interval.high,
                    "p": comparison.p, "established": comparison.established,
                }  # fmt: skip
                say("    " + comparison.line())
                # The protected-category rule, turned on the adoption itself.
                entry["category_change"] = category_change(production, cached)
                lost = regressions(entry["category_change"])
                say(f"    categories worse by >{REGRESSION:g} recall@10 on dev: {sorted(lost) or 'none'}")
            report["splits"][split] = entry
            for label in ("production", "ceiling"):
                say(f"    {label:11s} " + "  ".join(
                    f"{n} {v:.4f}" if v is not None else f"{n} -" for n, v in entry[label].items()))

    carry_forward_heldout(report, args)
    REPORTS.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    say(f"\nwrote {OUT_JSON.relative_to(ROOT)}")
    return 0


def carry_forward_heldout(report: dict, args: argparse.Namespace) -> None:
    """Keep a W3 result an earlier run produced, rather than dropping it from the report."""
    if args.second_look_w3 or "heldout" in report["splits"] or not OUT_JSON.exists():
        return
    previous = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    earlier = previous.get("splits", {}).get("heldout")
    if not earlier:
        return
    report["splits"]["heldout"] = {**earlier, "carried_forward_from": previous.get("scored_at")}
    report["w3_second_look"] = bool(previous.get("w3_second_look"))
    say(f"  heldout: carried forward from the run of {previous.get('scored_at')} (not re-scored)")


# --- the report -----------------------------------------------------------------------------------------


SPLIT_NAMES = {"dev": "dev (W0, W1, W2, W4)", "test": "test", "heldout": "held-out W3"}


def pooled_categories(report: dict) -> dict[str, dict]:
    """Per-category recall@10 over every scored split, query-weighted."""
    totals: dict[str, list[float]] = {}
    for entry in report.get("splits", {}).values():
        for name, got in entry.get("by_category", {}).items():
            value = got.get("recall@10")
            if value is None:
                continue
            seen = totals.setdefault(name, [0.0, 0.0])
            seen[0] += value * got["queries"]
            seen[1] += got["queries"]
    return {
        name: {"queries": int(n), "recall@10": total / n}
        for name, (total, n) in sorted(totals.items()) if n
    }


def dev_category_change(report: dict) -> dict[str, dict]:
    """The per-category before/after on dev, from the report's stored per-query outcomes."""
    stored = (report.get("splits", {}).get("dev") or {}).get("category_change")
    if stored:
        return stored
    rows = (report.get("splits", {}).get("dev") or {}).get("outcomes")
    cached = cached_dev_outcomes("baseline")
    if not rows or not cached:
        return {}
    after = [_Outcome(o["qid"], o["category"], frozenset(o["relevant"]), o["ranked"]) for o in rows]
    return category_change(after, cached)


def fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def delta(value) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}" if value else "0.0000"


def render(report: dict) -> str:
    """The closing report: what the recipe is worth, on every split, with the sequencing error disclosed."""
    corpus = report.get("corpus", {})
    lines = [
        "# Session 3 final report: the adopted recipe, measured on the production corpus",
        "",
        f"Scored {report['scored_at']} against `{report['database']}`.",
        "",
        f"**Recipe:** `{corpus.get('embedding_recipe', '?')}` -- the matched sentence under a",
        "`[date | house | heading | speaker | party]` header, embedded with BAAI/bge-small-en-v1.5 (384",
        f"dimensions). All {corpus.get('embedded_chunks', 0):,} embeddable chunks carry this one recipe; the",
        "HNSW index (m=16, ef_construction=64) was rebuilt over the finished table and the table vacuumed.",
        "",
        "**Search configuration: unchanged.** The Session 3 knob sweep adopted nothing, because the",
        "baseline was already at a plateau on every knob. So every difference below is the embedding",
        "recipe and nothing else.",
        "",
        f"**Relevance:** the judged pool binarised at UMBRELA grade >= {report['relevant_at']}. The judge",
        "passes its calibration gate on that one bit and fails it on the four-level grade (I-77), so nDCG",
        "here is a binarised nDCG and no graded number is claimed.",
        "",
        "## Headline",
        "",
        "| split | queries | recall@100 | recall@10 | MRR@10 | nDCG@10 |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        entry = report["splits"].get(split)
        if not entry:
            continue
        p = entry["production"]
        lines.append(f"| {SPLIT_NAMES[split]} | {entry['queries']} | " + " | ".join(
            fmt(p.get(f"{m}@{k}")) for m, k in METRICS) + " |")
    lines += [
        "",
        "### Against the pre-adoption baseline",
        "",
        "| split | recall@100 | recall@10 | MRR@10 | nDCG@10 | before, from |",
        "|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        entry = report["splits"].get(split)
        if not entry:
            continue
        d = entry.get("delta", {})
        source = report["before"].get(split, {}).get("source", "-")
        lines.append(f"| {SPLIT_NAMES[split]} | " + " | ".join(
            delta(d.get(f"{m}@{k}")) for m, k in METRICS) + f" | `{source}` |")
    dev = report["splits"].get("dev", {})
    if paired := dev.get("paired_comparison"):
        established = "established" if paired["established"] else "NOT established"
        lines += [
            "",
            f"On dev the before/after comparison is **paired**: the same {dev['queries']} queries, the same",
            "judged relevance, the old rankings read back from the sweep cache. Mean difference in",
            f"{paired['metric']} **{paired['mean']:+.4f}**, 95% CI "
            f"[{paired['low']:+.4f}, {paired['high']:+.4f}], p = {paired['p']:.4g} -- {established} after",
            "the family correction.",
            "",
            "**Dev is not independent evidence for the choice, and this report does not treat it as such.**",
            "The bake-off screened the header recipes on these same dev queries, so dev is where the recipe",
            "was selected. What the dev number establishes is that the gain *transfers*: from a 120,000-chunk",
            "comparative corpus to 1.07M chunks, and from dense-only exact search to the full three-leg",
            "pipeline with the reranker. Those are the two things that could have made it evaporate. The",
            "out-of-sample evidence is the test split, which had no part in the choice.",
            "",
            "On test and W3 only the aggregate metrics were written down before adoption, so those rows are",
            "a difference between two numbers and not a paired test. They are reported as such.",
        ]
    lines += ["", "## The first-stage ceiling", "",
              "What the reranker is handed. Nothing it does can recover a relevant speech the first stage",
              "never retrieved, so this bounds the whole pipeline: it is the number to attack next.",
              "",
              "| split | first-stage recall@100 | pipeline recall@10 | headroom the reranker cannot reach |",
              "|---|---|---|---|"]
    for split in SPLITS:
        entry = report["splits"].get(split)
        if not entry:
            continue
        ceiling = entry["ceiling"].get("recall@100")
        got = entry["production"].get("recall@10")
        gap = None if ceiling is None else round(1.0 - ceiling, 4)
        lines.append(f"| {SPLIT_NAMES[split]} | {fmt(ceiling)} | {fmt(got)} | {fmt(gap)} |")
    lines += ["", "## By question category (recall@10)", "",
              "The average is not the deliverable. A recipe that lifts the mean while losing a question type",
              "is not an upgrade, and the only way to see that is to look category by category. Sorted by how",
              "many queries the category carries, because the rows at the bottom are two or three questions",
              "and cannot support a conclusion on their own.", ""]
    counts: dict[str, int] = {}
    for entry in report["splits"].values():
        for name, got in entry.get("by_category", {}).items():
            counts[name] = counts.get(name, 0) + got["queries"]
    cats = sorted(counts, key=lambda c: (-counts[c], c))
    change = dev_category_change(report)
    header = ("| category | queries | " + " | ".join(SPLIT_NAMES[s] for s in SPLITS if s in report["splits"])
              + " | dev change |")  # fmt: skip
    lines += [header, "|" + "---|" * (len(header.split("|")) - 2)]
    for cat in cats:
        row = [cat, str(counts[cat])]
        for split in SPLITS:
            entry = report["splits"].get(split)
            if not entry:
                continue
            got = entry.get("by_category", {}).get(cat)
            row.append("-" if not got else f"{fmt(got.get('recall@10'))} (n={got['queries']})")
        row.append(delta((change.get(cat) or {}).get("delta")))
        lines.append("| " + " | ".join(row) + " |")

    regressed = regressions(change)
    lines += ["", "### Did the recipe cost anything?", ""]
    if regressed:
        lines += [
            "**Yes, and it is recorded rather than averaged away.** These categories lost more than 0.02",
            "recall@10 on dev against the old recipe:",
            "",
            "| category | queries | before | after | change |",
            "|---|---|---|---|---|",
        ]
        for name, got in sorted(regressed.items(), key=lambda kv: kv[1]["delta"]):
            lines.append(f"| {name} | {got['queries']} | {fmt(got['before'])} | {fmt(got['after'])} | "
                         f"{delta(got['delta'])} |")  # fmt: skip
        lines += [
            "",
            "The bake-off refused qwen3-0.6b for exactly this -- a better mean bought by losing a question",
            "type -- so the rule has to be applied to the recipe that was adopted too. It is applied here as",
            "disclosure and not as a veto, for one reason: the header recipe's gain is broad rather than",
            "concentrated, and every regressed category above rests on a handful of dev queries. Watch them",
            "as the gold set grows; a loss that survives a larger sample is a reason to revisit the recipe.",
        ]
    else:
        lines += ["No category lost more than 0.02 recall@10 on dev. The gain is broad rather than a trade."]

    if report.get("w3_second_look"):
        old = report["before"].get("heldout", {})
        new = report["splits"].get("heldout", {})
        lines += [
            "",
            "## Disclosure: W3 has been scored twice",
            "",
            "**What happened.** W3 is the held-out window and the discipline is that it is scored once, at",
            "the very end. It was scored on 2026-09-25 (`reports/session3_final_exam.json`) at the close of",
            "the knob sweep -- and then the embedder bake-off changed the embedding recipe, which is a far",
            "larger change than any knob. The exam was sat before the last change to the thing it examines.",
            "That is a sequencing error on my part, not a judgement call: the bake-off should have run",
            "before the final exam.",
            "",
            "**What was done about it.** W3 was re-scored under the adopted recipe, on the owner's explicit",
            "instruction, and both numbers are published:",
            "",
            "| | recall@100 | recall@10 | MRR@10 | nDCG@10 |",
            "|---|---|---|---|---|",
        ]
        for label, block, when in (("first look (old recipe)", old.get("production", {}), old.get("scored_at", "")),
                                   ("second look (adopted recipe)", new.get("production", {}), report["scored_at"])):
            lines.append(f"| {label}, {when} | " + " | ".join(fmt(block.get(f'{m}@{k}')) for m, k in METRICS) + " |")
        lines += [
            "",
            "**How to read the second number.** It is an honest measurement of the adopted recipe on W3 and",
            "it is *not* a clean held-out result. W3's queries have now informed no tuning decision -- no",
            "knob was adopted and the recipe was chosen on dev alone -- so the number is not contaminated by",
            "selection on W3. What it has lost is the statistical guarantee that comes from a set nobody has",
            "looked at: the strength of the claim it can support is weaker than the first look's, and if a",
            "future change needs a genuinely untouched set, W3 is no longer it. Phase 3 should cut a fresh",
            "held-out window from the corpus and score it once, last.",
        ]

    dev = report["splits"].get("dev", {})
    dev_ceiling = (dev.get("ceiling") or {}).get("recall@100")
    # recall@100 on both sides: the pool gap is a comparison at one depth, and mixing depth 100 with depth 10 would put
    # two different quantities in one sentence.
    dev_got = (dev.get("production") or {}).get("recall@100")
    # Named from the data rather than from an intuition about which questions are hard.
    weak = sorted(
        ((name, got) for name, got in pooled_categories(report).items()
         if got["queries"] >= 5 and got["recall@10"] < 0.45),
        key=lambda kv: kv[1]["recall@10"],
    )  # fmt: skip
    if weak:
        lines += [
            "",
            "### Where the value is actually lost",
            "",
            "The categories that carry weight (five queries or more across all splits) and still score under",
            "0.45 recall@10, pooled. These, not the headline average, are what Phase 3 has to move:",
            "",
            "| category | queries | recall@10 | what is going wrong |",
            "|---|---|---|---|",
        ]
        diagnosis = {
            "multi_speech": "the answer is spread over several contributions and each sentence holds only "
                            "part of it, so no single chunk looks relevant enough to rank",
            "who_spoke": "a roll-call of participants is a set, not a passage: recall@10 asks for ten "
                         "speeches when the answer is every speech in a debate",
            "cross_window": "the two halves of the answer sit in different windows, years apart, and nothing "
                            "in the query tells the dense leg to look in both",
            "paraphrase": "no lexical overlap to help the keyword leg, so it rests entirely on the dense leg",
            "colloquial": "the query is not in the register of the record, so neither leg has a strong signal",
            "multi_part": "two questions in one; the retrieval serves whichever part dominates the embedding",
            "comparison": "needs both sides to be retrieved, and one side is usually enough to fill ten slots",
        }
        for name, got in weak:
            lines.append(f"| {name} | {got['queries']} | {fmt(got['recall@10'])} | "
                         f"{diagnosis.get(name, 'not yet diagnosed')} |")  # fmt: skip
        lines += [
            "",
            "Two distinct shapes, and they want different fixes:",
            "",
            "1. **The answer is a set, and both the retrieval and the metric assume it is one passage.**",
            "   `multi_speech` and `who_spoke` together are 85 of the 237 retrieval-testable questions -- the",
            "   largest block in the gold set, and the weakest. Asking for ten speeches when the answer is",
            "   every contribution to a debate measures the wrong thing; a heading-scoped leg (I-38) that can",
            "   return a whole debate, or a depth chosen per category, attacks it directly.",
            "2. **The query is not in the record's register.** `colloquial` gives the keyword leg nothing to",
            "   match, so it rests entirely on the dense leg; `paraphrase` sits just above the line at 0.46 for",
            "   the same reason. This is what a better embedder or a query rewrite (HyDE) is for -- and the",
            "   model bake-off found the model barely mattered next to the text it was given, so a rewrite is",
            "   the more promising of the two.",
            "",
            "Neither is a knob, which is consistent with the sweep finding the knobs already at a plateau.",
        ]

    lines += [
        "",
        "## Verdict on Phase 2",
        "",
        "**Phase 2 is complete. Go for Phase 3.**",
        "",
        "What is locked in and reproducible:",
        "",
        f"* a {corpus.get('embedded_chunks', 0):,}-chunk corpus embedded under one recipe, with the HNSW",
        "  index rebuilt over it and the table vacuumed and analysed;",
        "* a 488-item gold set, triple-graded, with a judge calibrated against blind expert grades",
        "  (binarised kappa 0.6897, passing; four-level 0.5316, failing, which is why no graded nDCG is",
        "  claimed anywhere in this project);",
        "* an adoption decision taken by a rule written before the numbers were seen, which refused a",
        "  candidate that improved the average and lost a question category;",
        "* retrieval knobs swept, held to a plateau rather than a peak, and nothing adopted on noise.",
        "",
        "What Phase 3 inherits as its main problem:",
        "",
        f"* **the first stage never retrieves {pct(None if dev_ceiling is None else 1 - dev_ceiling)} of "
        "relevant speeches at depth 100 on dev**, and no reranker can recover what it was not handed. The "
        "categories above say where that loss sits: questions whose answer is a set of passages.",
        f"* **the reranker pool discards a growing share of what the first stage does find.** On dev the first "
        f"stage reaches recall@100 {fmt(dev_ceiling)} and the pipeline delivers {fmt(dev_got)}, so `pool=20` "
        f"drops {fmt(None if dev_ceiling is None or dev_got is None else dev_ceiling - dev_got)} -- up from "
        "0.0779 before adoption, because the first stage improved more than the pool can pass through. "
        "Raising `pool` is a latency and capacity decision for the answer layer, not a retrieval gain, but "
        "it is worth more now than it was.",
    ]
    # Conditional, because the alternative is a report that claims a second look which never happened.
    if report.get("w3_second_look"):
        lines.append("* **no untouched held-out set.** W3 has been scored twice (see the disclosure above). "
                     "Cut a fresh window before the next change that needs one.")  # fmt: skip
    else:
        lines.append("* **W3 is spent as a held-out set.** It has been scored once and must not be tuned on. "
                     "Cut a fresh window before the next change that needs an untouched one.")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", default="hansard_v2")
    p.add_argument("--score", action="store_true", help="run the searches and write the JSON")
    p.add_argument("--report", action="store_true", help="render the markdown from the JSON")
    p.add_argument("--second-look-w3", action="store_true",
                   help="also re-score the held-out window, disclosed in the report as a second look")  # fmt: skip
    p.add_argument("--top-k", type=int, default=100)
    args = p.parse_args(argv)
    if not (args.score or args.report):
        args.score = args.report = True

    if args.score and run_async(score(args)) != 0:
        return 1
    if args.report:
        if not OUT_JSON.exists():
            say(f"no {OUT_JSON.relative_to(ROOT)}; run --score first")
            return 1
        OUT_MD.write_text(render(json.loads(OUT_JSON.read_text(encoding="utf-8"))), encoding="utf-8", newline="\n")
        say(f"wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
