"""Read-only SQL against the record, always through the sealed-window mask: the one way a session or a review agent
reads `hansard_v2` by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from app import safety, sealed  # noqa: E402
from app.config import settings  # noqa: E402

# What reaches past the served views: the unmasked schema named outright, the session settings the mask rests on, or a
# Unicode-escaped name (`U&"\0070ublic"` is `public`), checked after comments, which Postgres reads as whitespace.
_PAST_THE_MASK = re.compile(r'"?\bpublic\b"?\s*\.|\bsearch_path\b|\bset_config\b|\botr\s*\.|\breset\b|\bdiscard\b'
                            r"|\bU&", re.I)
_COMMENT = re.compile(r"/\*.*?\*/|--[^\n]*", re.S)
# One read, and only one: a statement that starts SELECT, WITH or EXPLAIN (QA pass 8, Med 3: `SET ROLE NONE`, `SET
# SCHEMA`, `COMMIT; BEGIN READ WRITE` undid the reader role, the search path and the read-only transaction).
_ONE_READ = re.compile(r"^\s*(?:select|with|explain)\b", re.I)
# Functions that run SQL built from strings, or read past the statement, whose text no check can see.
_DYNAMIC = re.compile(r"\b(?:\w*_to_xml\w*|dblink\w*|pg_read_\w+|pg_ls_\w+|lo_\w+)\s*\(", re.I)


class Refused(RuntimeError):
    """SQL or a target this helper will not run."""


def query(sql: str, *, dsn: str) -> tuple[list[str], list[tuple]]:
    """(columns, rows) of one read through the mask, in a read-only transaction as the SELECT-only reader."""
    plain = _COMMENT.sub(" ", sql)
    if not _ONE_READ.match(plain) or ";" in plain.strip().rstrip(";") or _DYNAMIC.search(plain):
        raise Refused("one read only: a single SELECT, WITH or EXPLAIN, with no SQL built from strings")
    if _PAST_THE_MASK.search(plain):
        raise Refused("this SQL reaches past the sealed-window mask (public., the search path or the window settings)")
    if not sealed.masked():
        raise Refused("the sealed window is lifted (the W5 exam); this helper reads only through the mask")
    with psycopg.connect(dsn, options="-c default_transaction_read_only=on", connect_timeout=5) as conn:
        conn.execute(sealed.serving_sql())
        conn.execute("SET ROLE otr_reader")
        cur = conn.execute(sql, prepare=True)  # a prepared statement: the server itself refuses a second one
        if cur.description is None:
            return [], []
        return [d.name for d in cur.description], cur.fetchall()


def render(columns: list[str], found: list[tuple], *, rows: int = 40, width: int = 160) -> str:
    """A header, at most `rows` rows each cut to `width` characters, and the count."""
    lines = [" | ".join(columns)] + [" | ".join(str(v) for v in row)[:width] for row in found[:rows]]
    shown = min(rows, len(found))
    lines.append(f"({len(found)} rows{'' if shown == len(found) else f', {shown} shown'})")
    return "\n".join(lines)


def run(sql: str, *, rows: int = 40, width: int = 160) -> str:
    if safety.TARGET != "hansard_v2":
        raise Refused(f"the target is {safety.TARGET!r}; this helper reads hansard_v2 only")
    columns, found = query(sql, dsn=make_conninfo(settings.pg_dsn, dbname=safety.TARGET))
    return render(columns, found, rows=rows, width=width)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sql")
    p.add_argument("--rows", type=int, default=40)
    p.add_argument("--width", type=int, default=160)
    args = p.parse_args(argv)
    try:
        print(run(args.sql, rows=args.rows, width=args.width))
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
