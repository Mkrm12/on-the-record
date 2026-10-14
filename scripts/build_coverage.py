"""Refresh the coverage table."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import coverage as C  # noqa: E402
from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import make_pool  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

REPORTS = ROOT / "reports"


async def run(window: str | None, database: str, check: bool, show: bool) -> dict:
    dsn = safety.require(make_conninfo(settings.pg_dsn, dbname=database), database)
    async with make_pool(dsn, served=False) as db, db.connection() as conn:  # served=False: coverage counts every window
        if check:
            pending = await C.stale(conn)
            rows = await C.load(conn)
            return {"mode": "check", "stale_windows": pending, "rows": len(rows),
                    "ok": not pending}  # fmt: skip
        started = time.perf_counter()
        written = await C.refresh(conn, window)
        await conn.commit()
        rows = await C.load(conn)
        report = {
            "mode": "refresh", "window": window or "all", "database": database,
            "rows_written": written, "rows_total": len(rows),
            "seconds": round(time.perf_counter() - started, 2),
            "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stale_after": await C.stale(conn),
            "windows": sorted({r.window_id for r in rows}),
            "totals": {
                "sitting_days": sum(r.sitting_days for r in rows),
                "debates": sum(r.debates for r in rows),
                "contributions": sum(r.contributions for r in rows),
                "chunks": sum(r.chunks for r in rows),
                "divisions": sum(r.divisions for r in rows),
                "unreconciled_divisions": sum(r.unreconciled_divisions for r in rows),
                "unattributed": sum(r.unattributed for r in rows),
            },
            "gaps": [{k: str(v) for k, v in g.items()} for g in (await C.gaps(conn))[:10]],
        }
        if show:
            report["table"] = [
                {"window": r.window_id, "house": r.house, "section": r.section, "role": r.role,
                 "first_sitting": str(r.first_sitting), "last_sitting": str(r.last_sitting),
                 "sitting_days": r.sitting_days, "debates": r.debates, "contributions": r.contributions,
                 "speakers": r.speakers, "unattributed": r.unattributed, "chunks": r.chunks,
                 "divisions": r.divisions, "unreconciled": r.unreconciled_divisions,
                 "max_gap_days": r.max_gap_days}  # fmt: skip
                for r in rows
            ]
        return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", default=None, help="refresh one window instead of all of them")
    p.add_argument("--database", default=safety.TARGET)
    p.add_argument("--check", action="store_true", help="report staleness only; non-zero exit if stale")
    p.add_argument("--show", action="store_true", help="include the whole table in the report")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    report = run_async(run(args.window, args.database, args.check, args.show))
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "e2_coverage.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    elif args.check:
        if report["ok"]:
            print(f"coverage is current across {report['rows']} rows")
        else:
            print(f"STALE: {', '.join(report['stale_windows'])} — run without --check to refresh")
    else:
        t = report["totals"]
        print(f"refreshed {report['rows_written']} row(s) for {report['window']} in {report['seconds']}s; "
              f"{report['rows_total']} rows over {len(report['windows'])} windows")
        print(f"  {t['sitting_days']:,} sitting days, {t['debates']:,} debates, "
              f"{t['contributions']:,} contributions, {t['chunks']:,} chunks")
        print(f"  {t['divisions']:,} divisions, of which {t['unreconciled_divisions']:,} do not reconcile (I-8)")
        print(f"  {t['unattributed']:,} contributions could not be matched to a member")
        if report["stale_after"]:
            print(f"  STILL STALE: {report['stale_after']}")
        for gap in report["gaps"][:5]:
            print(f"  gap {gap['days']:>3}d  {gap['window_id']} {gap['house']} {gap['section']}: "
                  f"{gap['start']} -> {gap['end']}")
        if args.show:
            print(f"\n{'window':7s} {'house':8s} {'section':24s} {'days':>5s} {'debates':>8s} "
                  f"{'contrib':>8s} {'chunks':>8s} {'gap':>5s}")
            for row in report["table"]:
                print(f"{row['window']:7s} {row['house']:8s} {row['section']:24s} {row['sitting_days']:5d} "
                      f"{row['debates']:8d} {row['contributions']:8d} {row['chunks']:8d} "
                      f"{str(row['max_gap_days'] or '-'):>5s}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
