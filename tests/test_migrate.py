"""The migration runner's pure parts: ordering, naming and the pending calculation."""

from pathlib import Path

import pytest

from app import safety
from app.migrate import MIGRATIONS_DIR, discover, pending, target_dsn


def make(tmp_path: Path, *names: str) -> Path:
    for n in names:
        (tmp_path / n).write_text("select 1;")
    return tmp_path


def test_real_migrations_directory_is_well_formed():
    files = discover(MIGRATIONS_DIR)
    assert files[0].startswith("001_") and len(files) >= 2


def test_discover_orders_by_number(tmp_path):
    assert discover(make(tmp_path, "002_b.sql", "001_a.sql", "003_c.sql")) == ["001_a.sql", "002_b.sql", "003_c.sql"]


@pytest.mark.parametrize(
    "names",
    [
        ("001_a.sql", "003_c.sql"),  # gap
        ("001_a.sql", "001_b.sql"),  # duplicate number
        ("002_a.sql",),  # does not start at 1
        ("001_A.sql",),  # not snake_case
        ("1_a.sql",),  # wrong width
        ("001_a.txt.sql",),  # bad name
    ],
)
def test_discover_rejects_bad_sets(tmp_path, names):
    with pytest.raises(ValueError):
        discover(make(tmp_path, *names))


def test_pending_preserves_order_and_skips_applied():
    files = ["001_a.sql", "002_b.sql", "003_c.sql"]
    assert pending(files, set()) == files
    assert pending(files, {"001_a.sql"}) == ["002_b.sql", "003_c.sql"]
    assert pending(files, set(files)) == []
    assert pending(files, {"002_b.sql"}) == ["001_a.sql", "003_c.sql"]  # a skipped one is still pending


# It connected to `settings.pg_dsn` raw, the environment happened to point at `hansard`, and 005 to 007 were applied
# there.


def test_the_default_database_is_the_target_not_whatever_pg_dsn_says():
    """The default must be a constant, not the environment. A forgotten PG_DSN from an earlier shell is how this went
    wrong.
    """
    dsn = target_dsn(safety.TARGET, "postgresql://u:p@localhost:5432/hansard")
    assert safety.database_of(dsn) == safety.TARGET == "hansard_v2"


def test_migrating_the_rollback_is_refused_even_when_asked_for_explicitly():
    """`safety.require` would allow it, on the grounds that the caller named it. A migration runner must not offer that
    choice: a rollback is only a rollback while its schema is the one it was taken against.
    """
    with pytest.raises(safety.WrongDatabase, match="rollback"):
        target_dsn("hansard", "postgresql://u:p@localhost:5432/hansard")


@pytest.mark.parametrize("name", sorted(safety.PROTECTED))
def test_every_protected_database_is_refused(name):
    with pytest.raises(safety.WrongDatabase):
        target_dsn(name, f"postgresql://u:p@localhost:5432/{name}")


def test_a_refusal_never_leaks_the_dsn():
    """The DSN carries a password; an error message is the easiest place to lose one."""
    with pytest.raises(safety.WrongDatabase) as exc:
        target_dsn("hansard", "postgresql://u:sekrit@localhost:5432/hansard")
    assert "sekrit" not in str(exc.value)


def test_a_scratch_database_is_allowed():
    """Migrating a throw-away database is the normal case for tests and must keep working."""
    dsn = target_dsn("hansard_scratch", "postgresql://u:p@localhost:5432/hansard")
    assert safety.database_of(dsn) == "hansard_scratch"


def test_the_real_migration_set_is_still_wellformed():
    """Guards the numbering of the actual directory, not a fixture."""
    names = discover(MIGRATIONS_DIR)
    assert {"007_answer_cache_provenance.sql", "008_graph_indexes.sql", "009_speech_covering_index.sql",
            "010_division_vote_covering_index.sql"} <= set(names), names  # fmt: skip
    assert len(names) == len(set(names)) >= 10
