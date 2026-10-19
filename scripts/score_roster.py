"""Score the roster and debate identification against the gold set."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import roster as R  # noqa: E402
from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

REPORTS = ROOT / "reports"
CATEGORY = "who_spoke"


def rule_of_three(n: int) -> float:
    """The 95% upper bound on an error rate after n clean trials. 3/n."""
    return 3.0 / n if n > 0 else 1.0


def wilson_upper(errors: int, n: int) -> float:
    if n == 0:
        return 1.0
    z, p = 1.96, errors / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + spread) / (1 + z * z / n))


def norm(name: str | None) -> str:
    """Compare names on the part that identifies a person."""
    if not name:
        return ""
    head = name.split("(")[0]
    return " ".join(head.replace("’", "'").split()).casefold()


async def score(split: str, database: str, threshold: float) -> dict:
    items = [i for i in load_gold() if i.category == CATEGORY and i.split == split]
    if not items:
        raise SystemExit(f"no {CATEGORY} item in split {split!r}")
    dsn = make_conninfo(settings.pg_dsn, dbname=database)

    given_exact = 0
    given_misses: list[dict] = []
    e2e_exact = 0
    e2e_clarify = 0
    e2e_wrong_debate = 0
    e2e_misses: list[dict] = []
    roster_recall: list[float] = []
    roster_precision: list[float] = []
    started = time.perf_counter()

    async with make_pool(dsn, served=True) as db, db.connection() as conn:
        for item in items:
            want = item.expected_structured_facts or {}
            debate_id = want.get("debate_id")
            want_id, want_name = want.get("mnis_id"), want.get("opened_by")

            # --- given the gold debate: the SQL on its own -------------------------------------------
            opener = await R.first_speaker(conn, debate_id) if debate_id else {"missing": "no gold debate id"}
            if opener.get("missing"):
                given_misses.append({"id": item.id, "why": opener["missing"]})
            elif (want_id is not None and opener.get("mnis_id") == want_id) or \
                 (want_id is None and norm(opener.get("speaker_name")) == norm(want_name)):  # fmt: skip
                given_exact += 1
            else:
                given_misses.append({"id": item.id, "why": "wrong opener",
                                     "gold": {"mnis_id": want_id, "name": want_name},
                                     "engine": {"mnis_id": opener.get("mnis_id"),
                                                "name": opener.get("speaker_name")}})  # fmt: skip

            # The roster over the gold debate must contain the gold opener.
            if debate_id:
                built = await R.roster(conn, [debate_id])
                ids = {e.mnis_id for e in built.entries}
                names = {norm(e.speaker_name) for e in built.entries}
                found = (want_id in ids) if want_id is not None else (norm(want_name) in names)
                roster_recall.append(1.0 if found else 0.0)
                roster_precision.append(1.0 if found else 0.0)

            # --- end to end: find the debate first --------------------------------------------------- END TO END means
            # the question, as production passes it.
            candidates = await R.candidates_in_question(conn, item.question, item.house or None,
                                                        item.date or None)  # fmt: skip
            choice = R.pick_debate(candidates, threshold)
            if choice.needs_clarification:
                e2e_clarify += 1
                # Recorded, not counted as a miss: asking is a valid outcome and the gold set's own contract treats
                # clarify as distinct from both an answer and a failure.
                e2e_misses.append({"id": item.id, "outcome": "clarify",
                                   "confidence": round(choice.confidence, 3),
                                   "offered": [c.debate_id for c in choice.clarify]})  # fmt: skip
            elif not choice.ok:
                e2e_misses.append({"id": item.id, "outcome": "no_candidate", "why": choice.reason})
            elif choice.debate_id != debate_id:
                e2e_wrong_debate += 1
                e2e_misses.append({"id": item.id, "outcome": "wrong_debate",
                                   "picked": choice.debate_id, "gold": debate_id,
                                   "confidence": round(choice.confidence, 3)})  # fmt: skip
            else:
                found = await R.first_speaker(conn, choice.debate_id)
                if (want_id is not None and found.get("mnis_id") == want_id) or \
                   (want_id is None and norm(found.get("speaker_name")) == norm(want_name)):  # fmt: skip
                    e2e_exact += 1
                else:
                    e2e_misses.append({"id": item.id, "outcome": "right_debate_wrong_opener",
                                       "engine": found.get("speaker_name")})  # fmt: skip

    n = len(items)
    given_errors = n - given_exact
    answered = n - e2e_clarify
    return {
        "split": split, "database": database, "category": CATEGORY,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": n,
        "threshold": threshold,
        "seconds": round(time.perf_counter() - started, 1),

        "given_the_gold_debate": {
            "exact": given_exact, "errors": given_errors,
            "exact_rate": round(given_exact / n, 4),
            "error_upper_bound_95": round(rule_of_three(n) if given_errors == 0
                                          else wilson_upper(given_errors, n), 4),
            "bound_method": "rule of three (3/n)" if given_errors == 0 else "Wilson upper",
            "roster_recall": round(sum(roster_recall) / len(roster_recall), 4) if roster_recall else None,
            "roster_precision": (round(sum(roster_precision) / len(roster_precision), 4)
                                 if roster_precision else None),
            "misses": given_misses,
        },
        "end_to_end": {
            "exact": e2e_exact,
            "clarify": e2e_clarify,
            "wrong_debate": e2e_wrong_debate,
            "answered": answered,
            # Two rates, because they answer different questions: how often the system is right when it commits, and how
            # often it is right at all.
            "accuracy_when_answered": round(e2e_exact / answered, 4) if answered else None,
            "accuracy_over_all_items": round(e2e_exact / n, 4),
            "clarify_rate": round(e2e_clarify / n, 4),
            "misses": e2e_misses[:20],
        },
    }


async def calibrate(split: str, database: str) -> dict:
    """Sweep the clarify threshold on dev and report the whole curve, not just the winner."""
    rows = []
    for threshold in [0.0, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.80, 0.90]:
        report = await score(split, database, threshold)
        e2e = report["end_to_end"]
        rows.append({"threshold": threshold, "exact": e2e["exact"], "clarify": e2e["clarify"],
                     "wrong_debate": e2e["wrong_debate"],
                     "accuracy_when_answered": e2e["accuracy_when_answered"],
                     "accuracy_over_all_items": e2e["accuracy_over_all_items"],
                     "clarify_rate": e2e["clarify_rate"]})  # fmt: skip
    return {"split": split, "sweep": rows, "current_default": R.DEBATE_CONFIDENCE_THRESHOLD}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=("dev", "test", "heldout"))
    p.add_argument("--database", default=safety.TARGET)
    p.add_argument("--threshold", type=float, default=R.DEBATE_CONFIDENCE_THRESHOLD)
    p.add_argument("--calibrate", action="store_true", help="sweep the threshold instead of scoring once")
    p.add_argument("--confirm", action="store_true", help="required for the test split; the use is logged")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.split == "test" and not args.confirm:
        p.error("the test split confirms once: pass --confirm, and log that you used it")
    if args.split == "heldout":
        p.error("W5 and W3 are exam windows. The roster is not scored on them here.")
    if args.calibrate and args.split != "dev":
        p.error("calibrate on dev only: choosing a threshold on test is tuning on the confirmation split")

    REPORTS.mkdir(exist_ok=True)
    if args.calibrate:
        report = run_async(calibrate(args.split, args.database))
        out = REPORTS / "e2_roster_calibration.json"
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
        print(f"{'threshold':>10s} {'exact':>6s} {'clarify':>8s} {'wrong':>6s} "
              f"{'acc|answered':>13s} {'acc|all':>8s}")
        for row in report["sweep"]:
            acc = row["accuracy_when_answered"]
            print(f"{row['threshold']:10.2f} {row['exact']:6d} {row['clarify']:8d} {row['wrong_debate']:6d} "
                  f"{(f'{acc:.3f}' if acc is not None else 'n/a'):>13s} "
                  f"{row['accuracy_over_all_items']:8.3f}")
        print(f"\ncurrent default: {report['current_default']}")
        print(f"wrote {out.relative_to(ROOT)}")
        return 0

    report = run_async(score(args.split, args.database, args.threshold))
    out = REPORTS / f"e2_roster_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        given, e2e = report["given_the_gold_debate"], report["end_to_end"]
        print(f"{CATEGORY} on {args.split}: {report['items']} items in {report['seconds']}s "
              f"(clarify threshold {report['threshold']})")
        print(f"  given the gold debate : {given['exact']}/{report['items']} exact "
              f"({given['exact_rate']:.1%}); 95% error upper bound {given['error_upper_bound_95']:.1%} "
              f"({given['bound_method']})")
        print(f"  roster recall/precision: {given['roster_recall']:.1%} / {given['roster_precision']:.1%}")
        print(f"  end to end            : {e2e['exact']}/{report['items']} exact, "
              f"{e2e['clarify']} clarify, {e2e['wrong_debate']} wrong debate")
        acc = e2e["accuracy_when_answered"]
        print(f"                          accuracy when it answered "
              f"{(f'{acc:.1%}' if acc is not None else 'n/a')}, over all items "
              f"{e2e['accuracy_over_all_items']:.1%}, clarify rate {e2e['clarify_rate']:.1%}")
        for miss in given["misses"][:6]:
            print(f"  MISS (given) {miss['id']}: {miss.get('why')}")
            if "gold" in miss:
                print(f"       gold={miss['gold']} engine={miss['engine']}")
        for miss in e2e["misses"][:6]:
            print(f"  e2e {miss['id']}: {miss['outcome']} {miss.get('confidence', '')}")
    print(f"wrote {out.relative_to(ROOT)}")
    # Only the given-the-gold-debate gate is a pass/fail; the end-to-end number is reported, not gated, because asks for
    # it to be reported and retrieval is frozen.
    return 1 if report["given_the_gold_debate"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
