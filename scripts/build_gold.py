"""Build gold set v2: generate, validate, split, judge, and prepare the maintainer's review pack."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from goldset import audit25, authored, dbgen, judge as judging, llm, review, split, validate  # noqa: E402
from goldset.schema import GOLD_PATH, census, content_hash, dump  # noqa: E402
from ingest.runtime import run  # noqa: E402
from psycopg import AsyncConnection  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

REPORTS = ROOT / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", default=safety.TARGET, help="the corpus to generate from (default hansard_v2)")
    p.add_argument("--seed", type=int, default=20260924, help="makes generation and the split reproducible")
    p.add_argument("--judge", action="store_true", help="pool with the real searcher and grade with the LLM judge")
    p.add_argument(
        "--regrade",
        action="store_true",
        help="re-grade the CACHED pool with the current judge settings: no GPU, no re-pooling, and the "
        "same pairs the previous grades were measured on, so before and after are comparable",
    )
    p.add_argument(
        "--reuse-judgments",
        action="store_true",
        help="read the cached pool and grades from data/review/judgments.json instead of "
        "re-pooling and re-grading: no GPU, no API calls, no spend",
    )
    p.add_argument("--estimate-only", action="store_true", help="print the judging cost and stop")
    p.add_argument("--judge-limit", type=int, default=None, help="cap the number of pairs graded (a cheap dry run)")
    p.add_argument(
        "--reasoning",
        default="",
        help="reasoning_effort for the judge: '' (the default) leaves the model's reasoning ON, which is "
        "what Session 12 measured as the only setting that passes the relevance gate; 'none' turns it "
        "off, which is cheaper and does not pass",
    )
    p.add_argument(
        "--min-balance",
        type=float,
        default=None,
        help="override the balance floor in USD for this run only (default: settings.min_balance_usd, "
        "which is 0 = no floor)",
    )
    p.add_argument(
        "--confirm-spend",
        type=float,
        default=None,
        help="raise the approval bar in GBP for this run only. A run estimated at or above the bar "
        "refuses; the default bar is settings.spend_confirm_gbp. Use this once the owner has approved",
    )
    p.add_argument("--no-review", action="store_true", help="skip building the review pack")
    p.add_argument("--out", default=str(GOLD_PATH), help="where to write the gold set")
    return p.parse_args(argv)


def _say(message: str) -> None:
    print(message, flush=True)


def _reuse_judgments(items, summary):
    """Rebuild the pool and the grades from the cache. Nothing is retrieved and nothing is sent."""
    pool, grades, meta = judging.load_judgments()
    graded_pairs = sum(len(v) for v in grades.values())
    judging.apply_grades(items, grades, pool.variants)
    summary["pool"] = {"pairs": len(pool.pairs), "variants": pool.variants,
                       "items_pooled": len(pool.retrieved), "source": "cache"}  # fmt: skip
    summary["judging"] = {
        "ran": False, "reused": True, "model": meta.get("model", ""), "judged_at": meta.get("judged_at", ""),
        "pairs_graded": graded_pairs, "judged_at_10": judging.judged_at_k(pool, grades, 10),
    }  # fmt: skip
    _say(f"   reused {graded_pairs:,} cached grades from {judging.JUDGMENTS_PATH.name} "
         f"(model {meta.get('model', '?')}, judged {meta.get('judged_at', '?')}); nothing was sent")  # fmt: skip
    return pool, grades


async def _regrade(items, args, summary):
    """Grade the cached pool again. Retrieval is not touched, so the pairs are identical by construction."""
    pool, previous, meta = judging.load_judgments()
    _say(f"   loaded the cached pool: {len(pool.pairs):,} pairs over {pool.variants}, "
         f"last graded by {meta.get('model', '?')} ({sum(len(v) for v in previous.values()):,} grades)")  # fmt: skip
    summary["pool"] = {"pairs": len(pool.pairs), "variants": pool.variants,
                       "items_pooled": len(pool.retrieved), "source": "cache"}  # fmt: skip
    return await _grade_pool(pool, items, args, summary, previous_model=meta.get("model", ""))


async def _pool_and_judge(conn, items, args, summary):
    """Pooling needs the GPU service and judging needs a key."""
    from retrieval.evaluation import standard_variants
    from retrieval.gpu_client import GpuClient
    from retrieval.hybrid import HybridSearcher, SearchConfig

    from app.db import make_pool

    base = SearchConfig(
        common_df_ratio=50 / 90_040, speaker_date_aware=True, stored_lexicon=True, ef_search=100
    )
    variants = [v for v in standard_variants(base) if v.name in ("dense", "hybrid", "full")]

    pool = None
    async with make_pool(make_conninfo(settings.pg_dsn, dbname=args.database), served=False) as db:  # served=False: the generators skip the window themselves
        gpu = GpuClient()
        searcher = HybridSearcher(db, gpu)
        await searcher.refresh()
        started = time.monotonic()
        pool = await judging.build_pool(conn, searcher, items, variants)
        summary["pool"] = {
            "pairs": len(pool.pairs),
            "variants": pool.variants,
            "items_pooled": len(pool.retrieved),
            "seconds": round(time.monotonic() - started, 1),
        }
    _say(f"  pooled {len(pool.pairs):,} query-passage pairs from {len(pool.retrieved)} items "
         f"across {len(pool.variants)} variants")  # fmt: skip

    # Reasoning ON by default.
    return await _grade_pool(pool, items, args, summary)


async def _grade_pool(pool, items, args, summary, previous_model: str = ""):
    """Estimate, check the floor, smoke-test, grade, apply and cache. Shared by --judge and --regrade."""
    client = llm.Client(reasoning_effort=args.reasoning)
    plan = judging.estimate_judging(pool, client.model, client.reasoning_effort, args.judge_limit)
    summary["judging_estimate"] = {
        "calls": plan.calls, "input_tokens": plan.input_tokens, "output_tokens": plan.output_tokens,
        "usd": plan.usd, "gbp": plan.gbp, "model": plan.model, "basis": plan.basis,
    }  # fmt: skip
    _say(f"  judging estimate: {plan.line()}")

    if args.estimate_only:
        return pool, {}
    if not client.configured:
        _say("  no DEEPSEEK_KEY: the pool is built and saved, but nothing is graded. "
             "Set the key and re-run with --judge to grade it.")  # fmt: skip
        summary["judging"] = {"ran": False, "reason": "no API key configured"}
        return pool, {}

    try:
        available = await client.models()
        summary["judging_models_available"] = available
        if client.model not in available:
            _say(f"  the configured model {client.model!r} is not in /models (I-14); grading is skipped")
            summary["judging"] = {"ran": False, "reason": f"model {client.model!r} not offered by the account"}
            return pool, {}
        summary["judging_balance"] = await client.balance()
        # The maintainer's floor.
        affordable, why = await client.affordable(plan.usd, args.min_balance, args.confirm_spend)
        summary["judging_affordable"] = {"ok": affordable, "detail": why}
        _say(f"  {why}")
        if not affordable:
            _say("  not spending: pass --confirm-spend (once approved) or --min-balance")
            summary["judging"] = {"ran": False, "reason": f"balance floor: {why}"}
            return pool, {}
    except llm.LlmError as exc:
        summary["judging"] = {"ran": False, "reason": f"pre-flight failed: {exc}"}
        _say(f"  LLM pre-flight failed ({exc}); grading is skipped")
        return pool, {}

    # One real grading prompt before 3,960 of them.
    ok, detail = await judging.smoke_test(client)
    summary["judging_smoke_test"] = {"passed": ok, "detail": detail}
    if not ok:
        _say(f"  the judge failed its smoke test ({detail}); nothing was sent")
        summary["judging"] = {"ran": False, "reason": f"smoke test failed: {detail}"}
        return pool, {}
    _say(f"  smoke test: {detail}")

    started = time.monotonic()
    result = await judging.judge_run(pool, client, limit=args.judge_limit)
    grades, graded_pairs = result.grades, result.graded

    # A capped run is a dry run.
    if not judging.persists(args.judge_limit):
        summary["judging"] = {
            "ran": True, "dry_run": True, "model": client.model,
            "reasoning_effort": client.reasoning_effort or "on (provider default)",
            "pairs_offered": result.offered, "pairs_graded": graded_pairs,
            "no_reply": result.no_reply, "unparsable": result.unparsable,
            "yield_rate": result.yield_rate, "seconds": round(time.monotonic() - started, 1),
        }  # fmt: skip
        _say(f"  {result.line()}")
        _say("  --judge-limit was set, so this was a DRY RUN: neither the judgments cache nor the gold "
             "set was written. Re-run without --judge-limit to keep the grades.")  # fmt: skip
        return pool, {}

    judging.apply_grades(items, grades, pool.variants)
    summary["judging"] = {
        "ran": True,
        "model": client.model,
        "reasoning_effort": client.reasoning_effort or "on (provider default)",
        "max_tokens": judging.GRADE_MAX_TOKENS,
        "pairs_offered": result.offered,
        "pairs_graded": graded_pairs,
        "no_reply": result.no_reply,
        "unparsable": result.unparsable,
        "yield_rate": result.yield_rate,
        "judged_at_10": judging.judged_at_k(pool, grades, 10),
        "seconds": round(time.monotonic() - started, 1),
    }
    if result.yield_rate < 0.9:
        _say(f"  ! LOW YIELD: {result.line()} -- treat these grades as incomplete")
    cache = judging.save_judgments(
        pool, grades, model=client.model,
        extra={"corpus": args.database, "seed": args.seed,
               "reasoning_effort": client.reasoning_effort or "on (provider default)",
               "replaces_model": previous_model},  # fmt: skip
    )
    summary["judging"]["cache"] = str(cache.relative_to(ROOT))
    _say(f"  {result.line()}; judged@10 {summary['judging']['judged_at_10']}")
    _say(f"  cached the pool and every raw grade in {cache.relative_to(ROOT)} "
         f"(re-render with --reuse-judgments and spend nothing)")  # fmt: skip
    return pool, grades


async def main(args: argparse.Namespace) -> int:
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database)
    started = time.monotonic()
    summary: dict = {
        "session": "session2",
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "database": args.database,
        "seed": args.seed,
    }

    async with await AsyncConnection.connect(dsn) as conn:
        await conn.set_read_only(True)  # generation must never be able to change the corpus it measures

        _say("1. database-derived items from SQL templates")
        db_items = await dbgen.generate(conn, seed=args.seed)
        _say(f"   {len(db_items)} items")

        _say("2. authored human-meaningful and adversarial items")
        human_items, unresolved = await authored.build(conn)
        _say(f"   {len(human_items)} items, {len(unresolved)} unresolved anchor(s)")
        for u in unresolved:
            _say(f"     ! {u.seed_id} {u.speech_id}: {u.reason}")

        _say("3. audit of the original 25, carried in as dev")
        old_items, audit_findings = await audit25.migrate(conn)
        report = audit25.audit_report(audit_findings)
        _say(f"   {len(old_items)} items, {report['findings']} audit findings on "
             f"{len(report['items_flagged'])} items")  # fmt: skip
        summary["audit_of_the_25"] = report

        items = db_items + human_items + old_items

        _say("4. validation")
        validation = await validate.run_all(conn, items)
        validation["built_at"] = summary["started"]
        _say(f"   {validation['errors']} errors, {validation['warnings']} warnings "
             f"over {validation['items']} items")  # fmt: skip
        for finding in validation["findings"]:
            if finding["severity"] == "error":
                _say(f"     ERROR {finding['item']} {finding['check']}: {finding['message'][:130]}")
        summary["validation"] = {k: v for k, v in validation.items() if k not in ("findings", "absence_candidates")}
        summary["unresolved_anchors"] = [
            {"item": u.seed_id, "speech": u.speech_id, "reason": u.reason} for u in unresolved
        ]

        # An item that failed a hard check is marked dropped rather than deleted: the review tool still shows it, and
        # the record of what was rejected and why stays in the instrument.
        failed = set(validation["failed_items"])
        for item in items:
            if item.id in failed:
                item.status = "dropped"
                item.notes = (item.notes + " | " if item.notes else "") + "dropped: failed automatic validation"

        _say("5. dev / test / heldout split")
        split.assign(items, seed=args.seed)
        split_report = split.report(items)
        summary["split"] = split_report
        _say(f"   dev {split_report['dev']}, test {split_report['test']}, heldout {split_report['heldout']} "
             f"(hash {split_report['hash']})")  # fmt: skip
        if split_report["single_sided_categories"]:
            _say(f"   ! categories entirely on one side: {split_report['single_sided_categories']}")

        pool, raw_grades = (None, {})
        if args.regrade:
            _say("6. re-grading the cached pool (no retrieval)")
            try:
                pool, raw_grades = await _regrade(items, args, summary)
            except (FileNotFoundError, ValueError) as exc:
                _say(f"   no usable cache ({exc}); run --judge once to build one")
                summary["judging"] = {"ran": False, "reason": f"cache unusable: {exc}"}
        elif args.reuse_judgments:
            _say("6. cached pooled retrieval and judging")
            try:
                pool, raw_grades = _reuse_judgments(items, summary)
            except (FileNotFoundError, ValueError) as exc:
                _say(f"   no usable cache ({exc}); run --judge once to build it")
                summary["judging"] = {"ran": False, "reason": f"cache unusable: {exc}"}
        elif args.judge or args.estimate_only:
            _say("6. pooled retrieval and judging")
            pool, raw_grades = await _pool_and_judge(conn, items, args, summary)
        else:
            _say("6. pooling and judging skipped (pass --judge, or --reuse-judgments to reuse a graded pool)")

        summary["census"] = census(items)
        summary["gold_hash"] = content_hash(items)

        # A capped run is a diagnostic, and a diagnostic writes nothing.
        if not judging.persists(args.judge_limit):
            _say(f"7. DRY RUN (--judge-limit {args.judge_limit}): {args.out} and the review pack were NOT "
                 "written, so the existing grades are untouched. Re-run without --judge-limit to keep them.")  # fmt: skip
            summary["dry_run"] = True
            summary["census"] = census(items)
            summary["gold_hash"] = content_hash(items)
            summary["elapsed_minutes"] = round((time.monotonic() - started) / 60, 2)
            REPORTS.mkdir(exist_ok=True)
            (REPORTS / "session2_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8", newline="\n")
            return 1 if validation["errors"] else 0

        out = Path(args.out)
        dump(
            [i for i in items],
            out,
            description=(
                "Gold set v2 for Evaluation v2. ~300 database-derived items whose answers come straight "
                "from SQL, plus authored human-meaningful and adversarial items, plus the original 25 "
                "carried in as permanent dev data (I-45). W3 is held out and scored once."
            ),
            created=summary["started"],
            corpus=args.database,
            seed=args.seed,
            split_hash=split_report["hash"],
            gold_hash=summary["gold_hash"],
        )
        _say(f"7. wrote {out.relative_to(ROOT)} ({len(items)} items)")

        if not args.no_review:
            _say("8. review pack")
            pack = await review.build_pack(
                conn, items, pool=pool, validation=validation, seed=args.seed,
                judge_grades=raw_grades, judge_model=summary.get("judging", {}).get("model", ""),
            )  # fmt: skip
            if problems := pack["baseline"]["problems"]:
                _say(f"   ! the authored baseline does not match this pack ({len(problems)} problems); "
                     "it is not shown in the tool")  # fmt: skip
                for problem in problems[:3]:
                    _say(f"     {problem}")
            page = review.render(pack)
            summary["review"] = pack["plan"] | {
                "page": str(page.relative_to(ROOT)),
                "baseline_reviewer": pack["baseline"]["reviewer"],
                "baseline_grades": len(pack["baseline"]["kappa"]),
                "baseline_decisions": len(pack["baseline"]["items"]),
                "baseline_problems": pack["baseline"]["problems"],
                "judge_grades_shown": len(pack["judge"]["grades"]),
            }
            _say(f"   {pack['plan']['total']} things to review, about "
                 f"{pack['plan']['estimated_minutes']} minutes: {page.relative_to(ROOT)}")  # fmt: skip

    summary["elapsed_minutes"] = round((time.monotonic() - started) / 60, 2)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "session2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8", newline="\n")
    (REPORTS / "session2_summary.md").write_text(_markdown(summary), encoding="utf-8", newline="\n")
    _say(f"9. wrote reports/session2_summary.json and .md ({summary['elapsed_minutes']} min)")
    return 1 if validation["errors"] else 0


def _markdown(s: dict) -> str:
    c, sp = s["census"], s["split"]
    lines = [
        "# session2 summary: gold set v2",
        "",
        f"- Built: {s['started']}   Corpus: `{s['database']}`   Seed: {s['seed']}",
        f"- Gold hash: `{s['gold_hash']}`   Split hash: `{sp['hash']}`",
        f"- Elapsed: {s['elapsed_minutes']} minutes",
        "",
        "## Size",
        "",
        f"**{c['items']} items**, {c['points_available']} points available, "
        f"{c['structured_fact']} structured-fact and {c['retrieval_testable']} retrieval-testable.",
        "",
        "| split | items |", "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(sp["counts"].items())),
        "",
        "| provenance | items |", "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(c["by_provenance"].items())),
        "",
        "| expected behaviour | items |", "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(c["by_expected_behaviour"].items())),
        "",
        "| criticality | items |", "|---|---|",
        *(f"| {k} | {v} |" for k, v in sorted(c["by_criticality"].items())),
        "",
        "## Validation",
        "",
        f"- {s['validation']['errors']} errors, {s['validation']['warnings']} warnings",
        f"- measured token-Jaccard threshold for paraphrases: {s['validation']['jaccard_threshold']}",
        f"- checks run: {s['validation']['by_check']}",
        "",
        "## Audit of the original 25",
        "",
        f"{s['audit_of_the_25']['findings']} findings on {len(s['audit_of_the_25']['items_flagged'])} items "
        f"({', '.join(s['audit_of_the_25']['items_flagged'])}). Reported, not applied: all 25 stay dev.",
        "",
    ]
    for finding in s["audit_of_the_25"]["detail"]:
        lines.append(f"- **{finding['item']}** ({finding['kind']}): {finding['message']}")
    if "judging_estimate" in s or s.get("judging"):
        lines += ["", "## Judging", ""]
        if e := s.get("judging_estimate"):
            lines.append(
                f"- estimate: {e['calls']:,} calls, ~{e['input_tokens']:,} in / {e['output_tokens']:,} out "
                f"on `{e['model']}`, about ${e['usd']:.2f} (~GBP {e['gbp']:.2f})"
            )
        if affordable := s.get("judging_affordable"):
            lines.append(f"- balance floor: {affordable['detail']}"
                         + ("" if affordable["ok"] else " -- **refused to spend**"))  # fmt: skip
        if smoke := s.get("judging_smoke_test"):
            lines.append(f"- smoke test (one real grading prompt before the batch): "
                         f"{'passed' if smoke['passed'] else 'FAILED'} -- {smoke['detail']}")  # fmt: skip
        j = s.get("judging", {})
        if j.get("ran"):
            lines += [
                f"- model: `{j.get('model', '')}`",
                f"- graded **{j['pairs_graded']:,} of {j['pairs_offered']:,}** pairs offered "
                f"(yield {j.get('yield_rate', 0):.1%}); {j.get('no_reply', 0):,} with no reply, "
                f"{j.get('unparsable', 0):,} unparsable",
                f"- judged@10 {j['judged_at_10']}",
            ]
            if j.get("yield_rate", 1) < 0.9:
                lines.append("- **LOW YIELD: treat these grades as incomplete.**")
        elif j.get("reused"):
            lines.append(f"- reused {j['pairs_graded']:,} cached grades from `{j.get('model', '')}` "
                         f"(judged {j.get('judged_at', '')}); nothing was sent")  # fmt: skip
        else:
            lines.append(f"- not run: {j.get('reason', 'skipped')}")
    if "review" in s:
        r = s["review"]
        lines += ["", "## Review", "",
                  f"{r['total']} things to review, about **{r['estimated_minutes']} minutes**: "
                  f"{r['critical']} critical items, {r['kappa']} calibration pairs, "
                  f"{r['spot_check']} database spot-checks.", "",
                  "```powershell", "uv run python scripts/review_gold.py", "```"]  # fmt: skip
        if problems := r.get("baseline_problems"):
            lines += ["", f"The authored baseline did not match this pack ({len(problems)} problems), so it is "
                          "not shown in the tool:", ""] + [f"- {p}" for p in problems[:10]]  # fmt: skip
        elif r.get("baseline_reviewer"):
            lines += ["", f"The page carries the blind baseline by **{r['baseline_reviewer']}** "
                          f"({r.get('baseline_grades', 0)} calibration grades, "
                          f"{r.get('baseline_decisions', 0)} item decisions) beside "
                          f"{r.get('judge_grades_shown', 0)} judge grades, with an Approve all action."]  # fmt: skip
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(run(main(parse_args())))
