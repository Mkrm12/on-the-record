"""Fold old ordinary `answer_trace` rows into a daily aggregate and delete them. Evaluation rows stay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402

# Ordinary rows older than this (whole days) are folded.
SQL_DUE = """
    SELECT count(*) FROM answer_trace
     WHERE NOT is_evaluation AND created_at < date_trunc('day', now()) - make_interval(days => %(days)s)
"""
SQL_FOLD = """
    INSERT INTO answer_trace_daily AS d (day, route, outcome, requests, degraded, cache_hits, failed, llm_calls,
                                         prompt_tokens, completion_tokens, cost_usd, total_ms_sum, total_ms_max)
    SELECT created_at::date, coalesce(route, 'unknown'), coalesce(outcome, 'unknown'), count(*),
           count(*) FILTER (WHERE degraded), count(*) FILTER (WHERE cache_hit), count(*) FILTER (WHERE failed_reason IS NOT NULL),
           coalesce(sum(llm_calls), 0), coalesce(sum(prompt_tokens), 0), coalesce(sum(completion_tokens), 0),
           coalesce(sum(cost_usd), 0), coalesce(sum(total_ms), 0), max(total_ms)
      FROM answer_trace
     WHERE NOT is_evaluation AND created_at < date_trunc('day', now()) - make_interval(days => %(days)s)
     GROUP BY 1, 2, 3
    ON CONFLICT (day, route, outcome) DO UPDATE SET
        requests = d.requests + excluded.requests, degraded = d.degraded + excluded.degraded,
        cache_hits = d.cache_hits + excluded.cache_hits, failed = d.failed + excluded.failed,
        llm_calls = d.llm_calls + excluded.llm_calls, prompt_tokens = d.prompt_tokens + excluded.prompt_tokens,
        completion_tokens = d.completion_tokens + excluded.completion_tokens, cost_usd = d.cost_usd + excluded.cost_usd,
        total_ms_sum = d.total_ms_sum + excluded.total_ms_sum,
        total_ms_max = greatest(d.total_ms_max, excluded.total_ms_max)
"""
SQL_DELETE = """
    DELETE FROM answer_trace
     WHERE NOT is_evaluation AND created_at < date_trunc('day', now()) - make_interval(days => %(days)s)
"""


def fold(dsn: str, days: int, dry_run: bool = False) -> dict[str, int]:
    if days < 1:
        raise SystemExit("--days must be at least 1: a retention of zero days would fold today's traffic")
    safety.refuse_protected(dsn)
    with psycopg.connect(dsn, autocommit=True) as conn:
        due = conn.execute(SQL_DUE, {"days": days}).fetchone()[0]
        if dry_run or not due:
            return {"due": due, "folded": 0, "deleted": 0}
        with conn.transaction():
            folded = conn.execute(SQL_FOLD, {"days": days}).rowcount
            deleted = conn.execute(SQL_DELETE, {"days": days}).rowcount
    if deleted != due:
        raise SystemExit(f"deleted {deleted} rows where {due} were due: rows arrived or vanished mid-run; re-run")
    with psycopg.connect(dsn) as check:  # a second connection: committed, not merely executed (gotcha 37)
        left = check.execute(SQL_DUE, {"days": days}).fetchone()[0]
    if left:
        raise SystemExit(f"{left} due rows are still present after the commit")
    return {"due": due, "folded": folded, "deleted": deleted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--database", default=safety.TARGET)
    args = parser.parse_args()
    dsn = safety.require(make_conninfo(settings.pg_dsn, dbname=args.database), args.database)
    result = fold(dsn, args.days, args.dry_run)
    print(f"{result['due']:,} ordinary row(s) older than {args.days} days"
          + (" (dry run: nothing changed)" if args.dry_run else
             f"; folded into {result['folded']:,} daily group(s) and deleted {result['deleted']:,}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
