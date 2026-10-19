"""CI must build and test the database the code actually targets."""

from __future__ import annotations

import re
from pathlib import Path

from app import safety

CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _text() -> str:
    return CI.read_text(encoding="utf-8")


def _values(key: str) -> list[str]:
    """Every value given to `key:` in the workflow (one per line it appears on)."""
    return [m.group(1).strip() for m in re.finditer(rf"^\s*{key}:\s*(\S.*)$", _text(), re.M)]


def test_the_service_creates_the_working_database():
    assert _values("POSTGRES_DB") == [safety.TARGET], "the Postgres service must create the database app.migrate targets"


def test_the_health_check_probes_the_working_database():
    probes = re.findall(r"pg_isready[^\"\n]*-d\s+(\S+?)\"?\s*$", _text(), re.M)
    assert probes == [safety.TARGET]


def test_pg_dsn_names_the_working_database():
    dsns = _values("PG_DSN")
    assert dsns, "the test job must set PG_DSN"
    assert {safety.database_of(dsn) for dsn in dsns} == {safety.TARGET}


def test_no_protected_database_is_named_anywhere_in_ci():
    named = set(_values("POSTGRES_DB")) | {safety.database_of(d) for d in _values("PG_DSN")}
    named |= set(re.findall(r"-d\s+(\w+)", _text()))
    assert not named & safety.PROTECTED, f"CI names a protected database: {sorted(named & safety.PROTECTED)}"


def test_every_push_checks_that_a_release_archive_holds_no_gold():
    """Code reaches a server only as a release archive, so CI proves on every push that an archive of the pushed commit
    holds no gold, no W5 code and no working history.
    """
    lint = _text().split("\n  lint:", 1)[1].split("\n  test:", 1)[0]
    assert "scripts/release_archive.py --ref HEAD --check-only" in lint


def test_the_example_environment_names_the_working_database():
    """A fresh clone copies `.env.example` to `.env`: its PG_DSN named `hansard`, the protected rollback, which
    `safety.working_dsn` quietly redirects, so the example taught the wrong database.
    """
    example = CI.parents[2] / ".env.example"
    dsns = re.findall(r"^PG_DSN=(\S+)$", example.read_text(encoding="utf-8"), re.M)
    assert dsns and all(safety.database_of(d) == safety.TARGET for d in dsns), dsns
