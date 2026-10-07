"""Apply db/migrations/*.sql in filename order, recording each in schema_migrations."""

import argparse
import re
from pathlib import Path

import psycopg

from app import safety
from app.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"
_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def discover(directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Migration filenames in order. Raises on a bad name, a duplicate number or a gap."""
    names = sorted(p.name for p in directory.glob("*.sql"))
    numbers = []
    for name in names:
        m = _NAME.match(name)
        if not m:
            raise ValueError(f"bad migration filename {name!r}: expected NNN_snake_case.sql")
        numbers.append(int(m.group(1)))
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"migration numbers must be 1..N with no gaps or duplicates, got {numbers}")
    return names


def pending(files: list[str], applied: set[str]) -> list[str]:
    return [f for f in files if f not in applied]


def target_dsn(database: str, dsn: str | None = None) -> str:
    """The DSN this run is allowed to migrate, or a refusal."""
    if database in safety.PROTECTED:
        raise safety.WrongDatabase(
            f"refusing to migrate {database!r}: it is the rollback corpus, and a rollback is only a rollback "
            f"while its schema is the one it was taken against. Migrations belong on {safety.TARGET!r}.")
    return safety.require(safety.with_database(dsn or settings.pg_dsn, database), database)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply database migrations.")
    parser.add_argument("--status", action="store_true", help="show status and exit")
    parser.add_argument("--database", default=safety.TARGET,
                        help=f"which database to migrate (default {safety.TARGET})")  # fmt: skip
    args = parser.parse_args(argv)

    files = discover()
    dsn = target_dsn(args.database)
    with psycopg.connect(dsn, autocommit=True) as conn:
        # Stated out loud, every run.
        print(f"database: {conn.execute('SELECT current_database()').fetchone()[0]}")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        todo = pending(files, applied)
        if args.status:
            for f in files:
                print(f"{'applied' if f in applied else 'PENDING':8} {f}")
            return 0
        for name in todo:
            print(f"applying {name} ...")
            conn.execute((MIGRATIONS_DIR / name).read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (name,))
        print(f"done: {len(todo)} applied, {len(files) - len(todo)} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
