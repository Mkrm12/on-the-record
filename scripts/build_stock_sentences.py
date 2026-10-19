"""Materialise the stock-sentence table that arm B1 needs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from app import safety  # noqa: E402
from app.config import settings  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402
from retrieval.arms import STOCK_OCCURRENCES, normalise_sentence  # noqa: E402

TABLE = "stock_sentence"

# Derived data, rebuilt from `chunk` whenever the corpus changes, so it is created here rather than in a migration: a
# migration is for schema the loader depends on, and nothing depends on this but arm B1.
CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    normalised  text PRIMARY KEY,
    occurrences integer NOT NULL,
    example     text,
    built_at    timestamptz NOT NULL DEFAULT now()
)
"""

# The normalisation has to happen in SQL to avoid pulling 1.07M rows into Python.
NORMALISE_SQL = r"btrim(regexp_replace(regexp_replace(lower(body), '[^[:alnum:][:space:]_]', ' ', 'g'), '\s+', ' ', 'g'))"

COUNT = f"""
SELECT {NORMALISE_SQL} AS normalised, count(*) AS occurrences, min(body) AS example
  FROM chunk
 WHERE body IS NOT NULL
 GROUP BY 1
HAVING count(*) >= %(min_occurrences)s
   AND {NORMALISE_SQL} <> ''
"""


def build(dsn: str, min_occurrences: int = STOCK_OCCURRENCES) -> dict:
    """Rebuild the table. Idempotent: the same corpus gives the same rows."""
    started = time.perf_counter()
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(CREATE)
        chunks = conn.execute("SELECT count(*) FROM chunk WHERE body IS NOT NULL").fetchone()[0]
        with conn.transaction():
            # Replaced wholesale rather than upserted: a sentence that has fallen below the threshold must leave the
            # table, or B1 would go on penalising it after the corpus changed.
            conn.execute(f"TRUNCATE {TABLE}")
            conn.execute(
                f"INSERT INTO {TABLE} (normalised, occurrences, example) {COUNT}",
                {"min_occurrences": min_occurrences},
            )
        rows, covered = conn.execute(
            f"SELECT count(*), coalesce(sum(occurrences), 0) FROM {TABLE}").fetchone()  # fmt: skip
        top = conn.execute(
            f"SELECT occurrences, example FROM {TABLE} ORDER BY occurrences DESC, normalised LIMIT 15"
        ).fetchall()
    return {
        "table": TABLE,
        "min_occurrences": min_occurrences,
        "stock_sentences": rows,
        "chunks_matched": covered,
        "chunks_total": chunks,
        "share_of_corpus": round(covered / chunks, 6) if chunks else 0.0,
        "seconds": round(time.perf_counter() - started, 1),
        "top": [{"occurrences": n, "example": (text or "")[:120]} for n, text in top],
    }


def load(dsn: str) -> list[str]:
    """Every stock sentence's example text, for `HybridSearcher.load_stock_sentences`."""
    with psycopg.connect(dsn) as conn:
        try:
            return [row[0] for row in conn.execute(f"SELECT example FROM {TABLE}").fetchall() if row[0]]
        except psycopg.errors.UndefinedTable:
            raise SystemExit(
                f"{TABLE} does not exist. Build it: uv run python scripts/build_stock_sentences.py"
            ) from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database", default=safety.TARGET)
    p.add_argument("--min-occurrences", type=int, default=STOCK_OCCURRENCES)
    p.add_argument("--sample", type=int, default=0, help="print this many of the most common, then stop")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    if args.min_occurrences < 2:
        p.error("--min-occurrences must be at least 2: a sentence occurring once is not boilerplate")

    dsn = safety.require(make_conninfo(settings.pg_dsn, dbname=args.database))
    report = build(dsn, args.min_occurrences)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"{report['stock_sentences']:,} stock sentences (>= {args.min_occurrences} occurrences) "
              f"covering {report['chunks_matched']:,} of {report['chunks_total']:,} chunks "
              f"= {report['share_of_corpus']:.2%} of the corpus, in {report['seconds']}s")  # fmt: skip
        for row in report["top"][: args.sample or 10]:
            print(f"  {row['occurrences']:6,d}  {row['example']}")
        # The sanity check that matters: the SQL folding and the Python one must agree, or B1 looks up normalised text
        # the table never stored.
        mismatched = [r for r in report["top"] if normalise_sentence(r["example"]) == ""]
        if mismatched:
            print(f"\nWARNING: {len(mismatched)} example(s) normalise to nothing in Python but grouped in "
                  "SQL. The two definitions have drifted.")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
