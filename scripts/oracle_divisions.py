"""The division oracle against the database: every division outside the sealed window, field by field and vote by vote.
"""

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

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from eval.oracles import divisions as O  # noqa: E402

# Examples kept per check in the report.
EXAMPLES = 5
# The database's division columns the oracle reads.
COLUMNS = ("id", "house", "division_date", "number", "result", "ayes", "noes", "ayes_listed", "noes_listed",
           "ayes_named", "noes_named", "tally_reconciled", "title", "is_deferred")  # fmt: skip


def summarise(found: list[O.Disagreement], divisions: int, votes: int,
              conflicts: list[O.SourceConflict] | None = None, printed: int = 0) -> dict:  # fmt: skip
    by_check = Counter(d.check for d in found)
    examples: dict[str, list[dict]] = {}
    for d in found:
        bucket = examples.setdefault(d.check, [])
        if len(bucket) < EXAMPLES:
            bucket.append({"division_id": d.division_id, "mnis_id": d.mnis_id, "oracle": d.oracle,
                           "database": d.database})
    conflicts = conflicts or []
    return {"divisions_checked": divisions, "votes_checked": votes, "disagreements": sum(by_check.values()),
            "divisions_disagreeing": len({d.division_id for d in found}), "by_check": dict(sorted(by_check.items())),
            "examples": examples, "divisions_with_hansard_figures": printed,
            "source_conflicts": [{"division_id": c.division_id, "votes_api": list(c.votes_api),
                                  "hansard": list(c.hansard)} for c in conflicts]}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--out", type=Path, default=ROOT / "reports" / f"oracle_divisions_{date.today().isoformat()}.json")
    args = p.parse_args(argv)
    found: list[O.Disagreement] = []
    conflicts: list[O.SourceConflict] = []
    divisions = votes = printed = 0
    with psycopg.connect(safety.working_dsn(settings.pg_dsn), connect_timeout=5, autocommit=True,
                         options="-c default_transaction_read_only=on") as conn:  # fmt: skip
        for truth in O.truths():
            row = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM division WHERE id = %s",
                               (truth.division_id,)).fetchone()
            record = None if row is None else dict(zip(COLUMNS, row))
            rows = conn.execute("SELECT mnis_id, lobby, is_teller, proxy_name, party FROM division_vote "
                                "WHERE division_id = %s", (truth.division_id,)).fetchall()
            recorded = {r[0]: {"lobby": r[1], "is_teller": r[2], "proxy_name": r[3], "party": r[4]} for r in rows}
            found += O.compare(truth, record, recorded)
            conflicts += O.source_conflicts(truth)
            printed += truth.printed is not None
            divisions += 1
            votes += len(truth.votes)
    report = {"report": "oracle_divisions", "date": date.today().isoformat(),
              "source": "data/raw/corpus_v2 (Votes API payloads as fetched), outside the sealed window",
              **summarise(found, divisions, votes, conflicts, printed)}
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"{divisions} divisions and {votes} votes checked against the database: {report['disagreements']} "
          f"disagreement(s) in {report['divisions_disagreeing']} division(s) {report['by_check']}; "
          f"Hansard's printed figures found for {printed}, and {len(conflicts)} conflict(s) between them and the "
          f"Votes API (D-069): {[c.division_id for c in conflicts]}")
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
