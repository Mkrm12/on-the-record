"""Load Hansard's printed division figures beside the Votes API's."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from ingest import printed_figures as PF  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true", help="match and report; roll back")
    args = p.parse_args(argv)
    figures = PF.load()
    with psycopg.connect(safety.working_dsn(settings.pg_dsn)) as conn:
        report = PF.apply(conn, figures)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    print(f"{len(figures)} printed records outside the sealed window; {report['matched']} matched a division "
          f"({report['differ']} whose Votes API figures differ), {report['unmatched']} matched none; "
          f"{'would write' if args.dry_run else 'wrote'} {report['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
