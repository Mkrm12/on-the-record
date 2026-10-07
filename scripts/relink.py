"""Recompute every bill link in an existing database, offline, and report what changed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from ingest import bills as B  # noqa: E402
from ingest import corrections as K  # noqa: E402
from ingest import divisions as D  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402

COUNTS = {
    "divisions linked to a bill stage": "SELECT count(*) FILTER (WHERE bill_stage_id IS NOT NULL) FROM division",
    "divisions naming a Bill": r"SELECT count(*) FROM division WHERE title ~* '\mbill\M'",
    "bill stages with a debate": "SELECT count(*) FILTER (WHERE debate_id IS NOT NULL) FROM bill_stage",
    "in-corpus sittings linked to a debate": """
        SELECT count(*) FILTER (WHERE t.debate_id IS NOT NULL) FROM bill_stage_sitting t
         WHERE EXISTS (SELECT 1 FROM corpus_window w WHERE t.date BETWEEN w.start_date AND w.end_date)""",
    "in-corpus sittings": """
        SELECT count(*) FROM bill_stage_sitting t
         WHERE EXISTS (SELECT 1 FROM corpus_window w WHERE t.date BETWEEN w.start_date AND w.end_date)""",
    "divisions reaching a bill via their debate": """
        SELECT count(DISTINCT v.id) FROM division v JOIN debate_bill db ON db.debate_id = v.debate_id""",
    "sittings labelled formal_no_debate": "SELECT count(*) FROM bill_stage_sitting WHERE debate_link = 'formal_no_debate'",
    "corrections linked to the speech corrected": "SELECT count(corrected_speech_id) FROM hansard_correction",
    "corrections": "SELECT count(*) FROM hansard_correction",
    "divisions with a debate": "SELECT count(debate_id) FROM division",
}

# A fingerprint of EXACTLY the columns the linker writes.
FINGERPRINT = """
    SELECT md5(coalesce((SELECT string_agg(id || ':' || coalesce(debate_id, '') || ':' || coalesce(debate_link, ''),
                                           ',' ORDER BY id) FROM bill_stage_sitting), '')
            || coalesce((SELECT string_agg(id || ':' || coalesce(bill_stage_id::text, '') || ':'
                                           || coalesce(bill_link, '') || ':' || coalesce(debate_id, '') || ':'
                                           || coalesce(debate_link, ''), ',' ORDER BY id) FROM division), '')
            || coalesce((SELECT string_agg(id || ':' || coalesce(debate_id, ''), ',' ORDER BY id) FROM bill_stage), '')
            || coalesce((SELECT string_agg(id || ':' || coalesce(corrected_speech_id, '') || ':'
                                           || coalesce(link_basis, ''), ',' ORDER BY id) FROM hansard_correction), ''))
"""


async def counts(conn) -> dict[str, int]:
    out = {}
    for label, sql in COUNTS.items():
        out[label] = (await (await conn.execute(sql)).fetchone())[0]
    out["_fingerprint"] = (await (await conn.execute(FINGERPRINT)).fetchone())[0]
    return out


async def committed_counts(dsn: str) -> dict[str, int]:
    """Counts as any OTHER session sees them — which is to say, only what has been committed."""
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as reader:
        return await counts(reader)


async def main(database: str, dry_run: bool) -> int:
    if database in safety.PROTECTED:
        raise SystemExit(f"refusing: {database!r} is the protected rollback corpus")
    dsn = safety.require(make_conninfo(settings.pg_dsn, dbname=database), database)
    before = await committed_counts(dsn)
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as writer:
        actual = (await (await writer.execute("SELECT current_database()")).fetchone())[0]
        print(f"database: {actual}")
        if dry_run:
            for label, n in before.items():
                print(f"  {label:44} {n:>6}")
            return 0
        report = await B.link_all(writer)
        # Deferred divisions first: placing one gives it a debate, which the bill-stage linker can then use.
        deferred = await D.link_deferred(writer)
        report["deferred"] = deferred
        report["deferred_changed"] = deferred["changed"]
        corrections = await K.link_all(writer)
        report["corrections"] = corrections
        report["corrections_changed"] = corrections["changed"]
    after = await committed_counts(dsn)
    print(f"\n  {'(read on a second connection)':44} {'before':>7} {'after':>7}")
    for label in COUNTS:
        print(f"  {label:44} {before[label]:>7} {after[label]:>7}")
    print(f"  {'fingerprint of every linked column':44} {before['_fingerprint'][:7]:>7} {after['_fingerprint'][:7]:>7}")
    print("\nlinker report:", json.dumps(report, default=str))
    # The linker says what it changed; a second connection says what was kept.
    reported = sum(report.get(k) or 0 for k in ("divisions_changed", "sittings_changed", "stages_debate_set",
                                                "corrections_changed", "deferred_changed"))
    moved = after["_fingerprint"] != before["_fingerprint"]
    if reported and not moved:
        print(f"ERROR: the linker reports {reported} change(s) but the committed state did not move")
        return 1
    if moved and not reported:
        print("ERROR: the committed state moved but the linker reports no change — something else wrote to these tables")
        return 1
    print(f"verified: linker reported {reported} change(s); committed state {'moved' if moved else 'unchanged'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", default=safety.TARGET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sys.exit(run_async(main(args.database, args.dry_run)))
