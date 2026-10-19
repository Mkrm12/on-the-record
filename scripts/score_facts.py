"""Score the structured-fact engine against the gold set."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import facts as F  # noqa: E402
from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from goldset.schema import STRUCTURED_FACT  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

REPORTS = ROOT / "reports"


def rule_of_three(n: int) -> float:
    """The 95% upper bound on an error rate after n trials with zero errors."""
    return 3.0 / n if n > 0 else 1.0


def wilson_upper(errors: int, n: int) -> float:
    """A 95% upper bound that still works when there IS an error, which rule-of-three does not."""
    if n == 0:
        return 1.0
    z, p = 1.96, errors / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + spread) / (1 + z * z / n))


def sealed_facts(want: dict) -> bool:
    """Whether a gold item states a fact dated inside the sealed window: the masked engine cannot answer it."""
    from answer.sealing import _dates_within
    from app import sealed

    return any(sealed.hidden(day) for _, day in _dates_within(want))


def breach(got) -> list[str]:
    """The sealed dates an answer states: a breach of the mask, whatever the gold says."""
    from answer.sealing import _dates_within
    from app import sealed

    return sorted({text for text, day in _dates_within(getattr(got, "value", None) or {}) if sealed.hidden(day)})


async def score(split: str, database: str) -> dict:
    items = [i for i in load_gold() if i.category in STRUCTURED_FACT and i.split == split]
    if not items:
        raise SystemExit(f"no structured-fact item in split {split!r}")
    dsn = make_conninfo(settings.pg_dsn, dbname=database)
    exact = 0
    misses: list[dict] = []
    withheld: list[str] = []  # Gold dated inside the sealed window, which the mask hides
    by_category: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [exact, total]
    started = time.perf_counter()

    async with make_pool(dsn, served=True) as db, db.connection() as conn:
        for item in items:
            want = dict(item.expected_structured_facts)
            try:
                got = await F.answer(conn, item.category, want)
            except Exception as exc:  # noqa: BLE001 - recorded as a miss, never swallowed
                by_category[item.category][1] += 1
                misses.append({"id": item.id, "category": item.category,
                               "why": f"{type(exc).__name__}: {exc}"})  # fmt: skip
                continue
            if sealed_facts(want):
                # the mask's own check, reversed: such an item must NOT be answered with the window's dates
                if got.ok and breach(got):
                    by_category[item.category][1] += 1
                    misses.append({"id": item.id, "category": item.category,
                                   "why": f"mask breach: stated {breach(got)}"})  # fmt: skip
                else:
                    withheld.append(item.id)
                continue
            by_category[item.category][1] += 1
            if not got.ok:
                misses.append({"id": item.id, "category": item.category,
                               "why": got.missing or f"clarify with {len(got.clarify)} candidates"})  # fmt: skip
                continue
            wrong = {k: {"gold": want[k], "engine": got.value.get(k)} for k in want
                     if k in got.value and str(got.value.get(k)) != str(want[k])}
            if wrong:
                misses.append({"id": item.id, "category": item.category, "why": "mismatch", "fields": wrong})
            else:
                exact += 1
                by_category[item.category][0] += 1

    n = len(items) - len(withheld)
    errors = n - exact
    return {
        "split": split, "database": database, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": n, "exact": exact, "errors": errors,
        # counted apart, never scored: gold dated inside the sealed window, held by the mask
        "sealed_window": {"count": len(withheld), "ids": withheld, "defect": "D-063"},
        "exact_rate": round(exact / n, 4),
        "error_upper_bound_95": round(rule_of_three(n) if errors == 0 else wilson_upper(errors, n), 4),
        "bound_method": "rule of three (3/n)" if errors == 0 else "Wilson upper",
        "seconds": round(time.perf_counter() - started, 1),
        "by_category": {c: {"exact": e, "total": t} for c, (e, t) in sorted(by_category.items())},
        "misses": misses,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=("dev", "test", "heldout"))
    p.add_argument("--database", default=safety.TARGET)
    p.add_argument("--confirm", action="store_true", help="required for the test split; the use is logged")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.split == "test" and not args.confirm:
        p.error("the test split confirms once: pass --confirm, and log that you used it")
    if args.split == "heldout":
        p.error("W5 and W3 are exam windows. The structured-fact engine is not scored on them here.")

    report = run_async(score(args.split, args.database))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"e2_facts_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"{report['exact']} of {report['items']} exact ({report['exact_rate']:.1%}) on {args.split} "
              f"in {report['seconds']}s")
        if report["sealed_window"]["count"]:
            print(f"  plus {report['sealed_window']['count']} dated inside the sealed window, held by the mask and not "
                  f"scored (D-063): {', '.join(report['sealed_window']['ids'])}")
        print(f"95% upper bound on the error rate: {report['error_upper_bound_95']:.1%} "
              f"({report['bound_method']})")
        for category, row in report["by_category"].items():
            flag = "" if row["exact"] == row["total"] else "   <-- MISS"
            print(f"  {category:22s} {row['exact']:3d}/{row['total']:3d}{flag}")
        for miss in report["misses"][:10]:
            print(f"  MISS {miss['id']} ({miss['category']}): {miss['why']}")
            for field, pair in list(miss.get("fields", {}).items())[:3]:
                print(f"       {field}: gold={pair['gold']!r} engine={pair['engine']!r}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
