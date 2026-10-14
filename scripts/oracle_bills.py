"""The bill-stage oracle against the database: every dated stage outside the sealed window, field by field."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from app import safety, sealed  # noqa: E402
from app.config import settings  # noqa: E402
from eval.oracles import bills as OB  # noqa: E402

EXAMPLES = 5


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "reports" / f"oracle_bills_{date.today().isoformat()}.json")
    args = p.parse_args(argv)
    found: list[OB.Disagreement] = []
    stages = 0
    with psycopg.connect(safety.working_dsn(settings.pg_dsn), connect_timeout=5, autocommit=True,
                         options="-c default_transaction_read_only=on") as conn:  # fmt: skip
        conn.execute(sealed.serving_sql())  # the masked record, as a reader is served it
        for truth in OB.truths():
            # the aggregation `answer.facts._STAGE_SQL` serves: sitting days, the first of them
            row = conn.execute("""
                SELECT s.id, s.bill_id, s.stage_name, s.house, min(t.date) AS first_sitting,
                       count(DISTINCT t.date) AS sittings
                  FROM bill_stage s JOIN bill b ON b.id = s.bill_id
                  -- the route's as-of (D-112): a sitting after the bill's own last update is not yet held
                  JOIN bill_stage_sitting t ON t.bill_stage_id = s.id
                                           AND t.date <= coalesce(b.last_update, b.fetched_at)::date
                 WHERE s.id = %s
                 GROUP BY s.id, s.bill_id, s.stage_name, s.house
            """, (truth.stage_id,)).fetchone()
            record = None if row is None else dict(zip(("id", "bill_id", "stage_name", "house", "first_sitting",
                                                        "sittings"), row))  # fmt: skip
            found += OB.compare(truth, record)
            stages += 1
    by_check = Counter(d.check for d in found)
    examples: dict[str, list[dict]] = {}
    for d in found:
        bucket = examples.setdefault(d.check, [])
        if len(bucket) < EXAMPLES:
            bucket.append({"stage_id": d.stage_id, "oracle": d.oracle, "database": d.database})
    report = {"report": "oracle_bills", "date": date.today().isoformat(), "stages_checked": stages,
              "source": "data/raw/corpus_v2/bills (Bills API payloads as fetched), stages outside the sealed window",
              "disagreements": len(found), "by_check": dict(sorted(by_check.items())), "examples": examples}  # fmt: skip
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"{stages} bill stages checked against the database: {len(found)} disagreement(s) {dict(by_check)}")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
