"""The ten-year window plan — fill the gaps around the existing windows, never disturb them."""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from ingest import window_plan as WP
from ingest.corpus_fetch import ROLES, ConfigError, WindowSpec, load_config

REPO = Path(__file__).resolve().parents[1]
# The one [http] field a config cannot load without; without it a test fails on user_agent, not on what it meant to
# test.
HTTP = '[http]\nuser_agent = "on-the-record-tests/1.0"\n'


def w(wid, start, end, role="dev_test", active=True):
    return WindowSpec(id=wid, start=date.fromisoformat(start), end=date.fromisoformat(end), role=role, active=active)


EXISTING = [w("W3", "2020-03-01", "2020-05-31", "heldout"), w("W1", "2024-05-01", "2024-07-31"),
            w("W2", "2025-10-01", "2025-12-31"), w("W4", "2026-01-01", "2026-03-31"),
            w("W5", "2026-04-01", "2026-07-31", "heldout"), w("W0", "2026-09-01", "2026-09-17", "dev")]


def covered(windows):
    return {d for win in windows for d in win.days}


# --- the properties, on this project's layout --------------------------------------------------------------------


def test_the_gaps_plus_the_existing_windows_cover_the_span_exactly():
    span = (date(2016, 10, 1), date(2026, 9, 29))
    gaps = WP.gap_windows(EXISTING, *span)
    want = {span[0] + timedelta(days=i) for i in range((span[1] - span[0]).days + 1)}
    assert covered(gaps) | (covered(EXISTING) & want) == want


def test_no_gap_window_overlaps_an_existing_one_or_another_gap():
    gaps = WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29))
    everything = EXISTING + gaps
    for i, a in enumerate(everything):
        for b in everything[i + 1:]:
            assert not a.overlaps(b), f"{a.id} overlaps {b.id}"


def test_the_existing_windows_are_untouched():
    """W3 and W5 keep their identity and role: an extension that renamed or reshaped them would break evaluation."""
    before = [(x.id, x.start, x.end, x.role) for x in EXISTING]
    WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29))
    assert [(x.id, x.start, x.end, x.role) for x in EXISTING] == before


def test_every_gap_window_has_the_corpus_role_and_is_active():
    gaps = WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29))
    assert gaps and all(g.role == "corpus" and g.active for g in gaps)


def test_gap_windows_never_cross_a_calendar_quarter():
    """A quarter is the unit of fetch, load and resumption; a window crossing one is a unit that cannot be retried
    alone.
    """
    for g in WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29)):
        assert (g.start.month - 1) // 3 == (g.end.month - 1) // 3 and g.start.year == g.end.year, g.id


def test_ids_are_deterministic_and_unique():
    a = WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29))
    b = WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29))
    assert [g.id for g in a] == [g.id for g in b]
    assert len({g.id for g in a}) == len(a)


def test_a_partial_quarter_next_to_an_existing_window_gets_its_own_id():
    ids = [g.id for g in WP.gap_windows(EXISTING, date(2016, 10, 1), date(2026, 9, 29))]
    assert "C2020Q2" in ids and "C2020Q3" in ids


def test_no_gap_window_runs_into_the_future():
    today = date(2026, 9, 29)
    assert all(g.end <= today for g in WP.gap_windows(EXISTING, date(2016, 10, 1), today))


def test_inactive_windows_do_not_count_as_covered():
    """An inactive candidate is an alternative that was NOT fetched; its days are a gap like any other."""
    existing = [w("A", "2024-01-01", "2024-03-31"), w("CAND", "2024-04-01", "2024-06-30", "candidate", active=False)]
    gaps = WP.gap_windows(existing, date(2024, 1, 1), date(2024, 6, 30))
    assert covered(gaps) == set(w("x", "2024-04-01", "2024-06-30").days)


def test_a_span_already_fully_covered_needs_no_windows():
    assert WP.gap_windows([w("A", "2024-01-01", "2024-12-31")], date(2024, 1, 1), date(2024, 12, 31)) == []


def test_an_inverted_span_is_refused():
    with pytest.raises(ValueError):
        WP.gap_windows(EXISTING, date(2026, 1, 1), date(2025, 1, 1))


# --- the same properties on random layouts ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_the_properties_hold_on_random_layouts(seed):
    """Forty random layouts of existing windows: exact cover, no overlap, quarter-bounded, deterministic."""
    rng = random.Random(seed)
    span_start = date(2010, 1, 1) + timedelta(days=rng.randrange(0, 400))
    span_end = span_start + timedelta(days=rng.randrange(200, 3650))
    existing, cursor = [], span_start - timedelta(days=rng.randrange(0, 60))
    for i in range(rng.randrange(0, 8)):
        cursor += timedelta(days=rng.randrange(1, 300))
        length = rng.randrange(1, 200)
        existing.append(WindowSpec(f"E{i}", cursor, cursor + timedelta(days=length), "dev_test", True))
        cursor += timedelta(days=length + 1)
    gaps = WP.gap_windows(existing, span_start, span_end)
    want = {span_start + timedelta(days=i) for i in range((span_end - span_start).days + 1)}
    assert covered(gaps) | (covered(existing) & want) == want
    assert covered(gaps) <= want and not (covered(gaps) & covered(existing))
    assert all((g.start.month - 1) // 3 == (g.end.month - 1) // 3 and g.start.year == g.end.year for g in gaps)
    assert gaps == WP.gap_windows(existing, span_start, span_end)


# --- the config: roles fail fast, and a plan extends rather than copies ---------------------------------------------


def test_the_corpus_role_exists():
    assert "corpus" in ROLES and {"dev", "dev_test", "heldout", "candidate"} <= ROLES


def test_an_unknown_role_is_refused_when_the_config_loads(tmp_path):
    """It used to be accepted by the loader and refused by the database's CHECK at `register_window` — hours into a
    ten-year fetch. A typo must cost a second, not a night.
    """
    cfg = tmp_path / "c.toml"
    cfg.write_text(HTTP + '[corpus]\nversion = "v2"\n[[windows]]\nid = "X"\nstart = 2024-01-01\nend = 2024-01-31\n'
                   'role = "corpse"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="role"):
        load_config(cfg, repo_root=tmp_path, today=date(2026, 9, 29))


def test_a_plan_file_extends_the_base_config_without_copying_it(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(HTTP + '[corpus]\nversion = "v2"\n[[windows]]\nid = "W1"\nstart = 2024-05-01\nend = 2024-07-31\n'
                    'role = "dev_test"\n', encoding="utf-8")
    plan = tmp_path / "plan.toml"
    plan.write_text('extends = "base.toml"\n[[windows]]\nid = "C2024Q3"\nstart = 2024-08-01\nend = 2024-09-30\n'
                    'role = "corpus"\n', encoding="utf-8")
    cfg = load_config(plan, repo_root=tmp_path, today=date(2026, 9, 29))
    assert [x.id for x in cfg.windows] == ["W1", "C2024Q3"]


def test_an_extending_plan_that_overlaps_the_base_is_refused(tmp_path):
    base = tmp_path / "base.toml"
    base.write_text(HTTP + '[corpus]\nversion = "v2"\n[[windows]]\nid = "W1"\nstart = 2024-05-01\nend = 2024-07-31\n',
                    encoding="utf-8")
    plan = tmp_path / "plan.toml"
    plan.write_text('extends = "base.toml"\n[[windows]]\nid = "C1"\nstart = 2024-07-01\nend = 2024-09-30\n'
                    'role = "corpus"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="overlap"):
        load_config(plan, repo_root=tmp_path, today=date(2026, 9, 29))


def test_an_extending_plan_may_only_add_windows(tmp_path):
    """Anything else in it would silently shadow the base's HTTP policy or sources for the old windows too."""
    (tmp_path / "base.toml").write_text(HTTP + '[corpus]\nversion = "v2"\n[[windows]]\nid = "W1"\n'
                                        'start = 2024-05-01\nend = 2024-07-31\n', encoding="utf-8")
    plan = tmp_path / "plan.toml"
    plan.write_text('extends = "base.toml"\n[http]\nconcurrency = 4\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="only add"):
        load_config(plan, repo_root=tmp_path, today=date(2026, 9, 29))


def test_the_real_config_still_loads_and_its_windows_keep_their_roles():
    cfg = load_config(REPO / "config" / "corpus_v2.toml", repo_root=REPO, today=date(2026, 9, 29))
    roles = {x.id: x.role for x in cfg.windows if x.active}
    assert roles["W3"] == "heldout" and roles["W5"] == "heldout" and roles["W0"] == "dev"


def test_the_config_roles_and_the_database_roles_are_the_same_list(scratch):
    """Two definitions of one vocabulary drift. Migration 016's CHECK and `corpus_fetch.ROLES` are compared here."""
    definition = scratch.execute(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'corpus_window_role_check'").fetchone()[0]
    import re as _re

    in_db = set(_re.findall(r"'([a-z_]+)'", definition))
    assert in_db == set(ROLES)
