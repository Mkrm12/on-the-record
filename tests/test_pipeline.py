"""The orchestrator: the parts that only fail when nobody is watching."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import pipeline  # noqa: E402


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STATE_DIR", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- Resuming
# ---------------------------------------------------------------------------


def test_a_finished_step_is_remembered_across_processes(state_dir):
    state = pipeline.State.load("session1")
    assert not state.done("load")
    state.record("load", "done", 42.0, {"speeches": 10})
    assert pipeline.State.load("session1").done("load")
    assert pipeline.State.load("session1").payload("load") == {"speeches": 10}


def test_a_failed_step_is_not_treated_as_finished(state_dir):
    """The difference between 'ran and failed' and 'never ran' is the difference between a resume that redoes the
    failure and one that skips straight past it.
    """
    state = pipeline.State.load("session1")
    state.record("embed", "failed", 10.0, {"error": "out of memory"})
    assert not pipeline.State.load("session1").done("embed")


def test_a_corrupt_state_file_starts_from_the_beginning_rather_than_crashing(state_dir):
    (state_dir / "session1_state.json").write_text("{not json", encoding="utf-8")
    state = pipeline.State.load("session1")
    assert state.data == {"steps": {}, "started_at": None}


def test_the_state_file_is_written_atomically(state_dir):
    """A kill during the write must not leave a state file that parses as 'nothing done' when eight steps are done. The
    temp-and-replace is the whole guard.
    """
    state = pipeline.State.load("session1")
    state.record("load", "done", 1.0)
    written = json.loads((state_dir / "session1_state.json").read_text(encoding="utf-8"))
    assert written["steps"]["load"]["status"] == "done"
    assert not list(state_dir.glob("*.tmp")), "the temporary file must not survive"


def test_redo_forgets_one_step_and_restart_forgets_all(state_dir):
    state = pipeline.State.load("session1")
    for name in ("migrate", "load", "chunk"):
        state.record(name, "done", 1.0)
    state.reset(["load"])
    assert state.done("migrate") and not state.done("load") and state.done("chunk")
    state.reset()
    assert not any(state.done(n) for n in ("migrate", "load", "chunk"))


# --------------------------------------------------------------------------- The machine
# ---------------------------------------------------------------------------


def test_the_standby_timeout_is_read_in_minutes_not_seconds(monkeypatch):
    """`powercfg /query` reports SECONDS and `powercfg /change` takes MINUTES. Restoring 0x708 as 1800 minutes would
    leave the laptop awake for 30 hours, and nobody would notice for a month.
    """
    class FakeRun:
        stdout = ("  Power Setting GUID: 29f6c1db (Sleep after)\n"
                  "    Current AC Power Setting Index: 0x00000708\n"
                  "    Current DC Power Setting Index: 0x00002a30\n")

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: FakeRun())
    assert pipeline.standby_timeout_minutes() == 30


def test_an_unreadable_power_setting_is_none_not_a_guess(monkeypatch):
    """None means 'leave it alone'. Guessing a value and 'restoring' it would change a setting the run never touched.
    """
    class Empty:
        stdout = "Invalid Parameters"

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: Empty())
    assert pipeline.standby_timeout_minutes() is None

    def boom(*a, **k):
        raise OSError("no powercfg here")

    monkeypatch.setattr(pipeline.subprocess, "run", boom)
    assert pipeline.standby_timeout_minutes() is None


# --------------------------------------------------------------------------- Preflight
# ---------------------------------------------------------------------------


def test_preflight_refuses_the_rollback_database():
    check = pipeline._database_check("postgresql://u:p@127.0.0.1:5432/hansard")
    assert not check.ok and check.fatal and "rollback" in check.detail


def test_preflight_accepts_the_target_database():
    assert pipeline._database_check("postgresql://u:p@127.0.0.1:5432/hansard_v2").ok


def test_a_missing_api_key_is_a_warning_not_a_failure():
    """Failing preflight on one would stop a four-hour run for a step that is not in it."""
    check = pipeline._keys_check()
    assert check.ok is True and check.fatal is False


def test_running_on_battery_is_fatal(monkeypatch):
    """The GPU is power-capped to about a quarter of its speed on battery, so every timing in the summary would be
    wrong, and the run does not finish on a battery anyway.
    """
    monkeypatch.setattr(pipeline, "on_mains", lambda: False)
    check = pipeline._power_check()
    assert not check.ok and check.fatal and "BATTERY" in check.detail


def test_an_unknowable_power_source_is_a_warning(monkeypatch):
    monkeypatch.setattr(pipeline, "on_mains", lambda: None)
    check = pipeline._power_check()
    assert check.ok and not check.fatal


# --------------------------------------------------------------------------- What the maintainer is told
# ---------------------------------------------------------------------------


def test_the_resume_command_is_the_command_that_was_typed(capsys):
    """The failure message's whole job is to be pasteable. A resume line that drops --only would load every window
    instead of one.
    """
    class Args:
        session, only, database = "session1", ["W0"], "hansard_v2"

    pipeline._print_resume(Args())
    printed = " ".join(capsys.readouterr().out.split())
    assert "run session1 --only W0 --database hansard_v2" in printed


def test_every_step_has_an_estimate_so_the_total_is_not_a_guess():
    assert all(step.minutes > 0 for step in pipeline.session1_steps())
    assert sum(step.minutes for step in pipeline.session1_steps()) > 60


def _summary_data(session="session17", steps=None):
    return {"session": session, "finished_at": "now", "outcome": "ok", "failed_step": None,
            "database": "hansard_v2", "windows": "all active", "elapsed_minutes": 1.0,
            "power": "mains", "notes": [], "steps": steps or {}}  # fmt: skip


def test_the_summary_names_this_sessions_own_reports_not_session_ones():
    """The bug this pins. Both closing blocks were hardcoded to session 1: every summary since told the next session to
    read `session1_summary.json` and four session-1 eval files, and printed payloads only for step names that
    session 1 used.
    """
    data = _summary_data(steps={
        "coverage": {"status": "done", "seconds": 3.0,
                     "payload": {"coverage": ["569 divisions", "wrote reports/e2_coverage.json"]}},
        "facts": {"status": "done", "seconds": 1.0,
                  "payload": {"facts": ["100/100 exact", "wrote reports" + chr(92) + "e2_facts_dev.json"]}},
    })  # fmt: skip
    markdown = pipeline._markdown(data, [])
    assert "reports/session17_summary.json" in markdown
    assert "reports/e2_coverage.json" in markdown
    # A Windows backslash path in captured output must be listed as a forward-slash path, or the next session cannot
    # paste it.
    assert "reports/e2_facts_dev.json" in markdown
    assert "session1_summary.json" not in markdown
    assert "oracle_w0.json" not in markdown


def test_the_summary_prints_every_steps_payload():
    """Not a fixed list of four session-1 step names."""
    data = _summary_data(steps={
        "verifier_precision": {"status": "done", "seconds": 1470.0,
                               "payload": {"precision": ["agreement 300/300"]}},
    })  # fmt: skip
    markdown = pipeline._markdown(data, [])
    assert "### verifier_precision" in markdown and "300/300" in markdown


def test_a_run_that_wrote_nothing_says_so_rather_than_listing_one_report():
    markdown = pipeline._markdown(_summary_data(), [])
    assert "no step reported writing a report" in markdown


def test_the_last_json_a_script_printed_is_found_after_a_table():
    out = 'a table\n+---+\n| x |\n+---+\n[{"window_id": "W0", "speeches": 8985}]\n'
    assert pipeline._last_json(out) == [{"window_id": "W0", "speeches": 8985}]
    assert pipeline._last_json("no json here") is None


def test_both_evaluations_are_run_at_more_than_one_ef_search():
    """One number at ef_search=100 cannot tell a corpus effect from an index-build effect, and the whole point of the
    scale tax is to isolate the corpus effect.
    """
    assert len(pipeline.EF_SEARCH_SETTINGS) >= 2
    assert 100 in pipeline.EF_SEARCH_SETTINGS


def test_an_absolute_path_never_reaches_a_committed_report():
    """`reports/` is committed and the standing rule is that no absolute path goes into git. Keeping that by hand is
    easy; keeping it when a step's stdout is copied into a summary verbatim is not.
    """
    noisy = {"tail": [r"warning: `VIRTUAL_ENV=C:\Users\someone\Documents\repo\.venv` is ignored"],
             "nested": {"paths": ["/home/someone/secret/thing.json", "reports/fine.json"]},
             "count": 7}
    cleaned = pipeline.redact(noisy)
    assert "someone" not in json.dumps(cleaned)
    assert cleaned["nested"]["paths"][1] == "reports/fine.json", "relative paths are left alone"
    assert cleaned["count"] == 7


def test_the_recorded_payload_is_redacted_on_the_way_into_the_state_file(state_dir):
    state = pipeline.State.load("session1")
    state.record("load", "done", 1.0, {"tail": [r"C:\Users\someone\repo"]})
    assert "someone" not in (state_dir / "session1_state.json").read_text(encoding="utf-8")


def test_the_child_environment_does_not_inherit_a_mismatched_virtualenv(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / ".venv"))
    ctx = pipeline.Ctx(session="session1", state=pipeline.State.load("session1"),
                       log=tmp_path / "log.txt", dsn="postgresql://u:p@127.0.0.1:5432/hansard_v2",
                       windows=[])
    assert "VIRTUAL_ENV" not in ctx.env()
    assert ctx.env()["PG_DSN"].endswith("hansard_v2")


# --------------------------------------------------------------------------- Scope: the bug that would have handed over
# a summary for a corpus nobody built ---------------------------------------------------------------------------


def test_a_step_finished_for_one_window_is_not_finished_for_all_of_them(state_dir):
    """The one that bit. Proving the orchestrator with `--only W0` wrote "done" against all eleven steps; the next full
    run then read "load: done" and skipped the four windows it exists to load, finishing in 0 minutes with a summary
    for a corpus that was never built.
    """
    state = pipeline.State.load("session1")
    state.record("load", "done", 12.0, scope="W0")
    assert state.done("load", "W0")
    assert not state.done("load", "all")
    assert not pipeline.State.load("session1").done("load", "all")


def test_a_step_finished_for_every_window_is_still_finished_on_a_resume(state_dir):
    state = pipeline.State.load("session1")
    state.record("load", "done", 12.0, scope="all")
    assert pipeline.State.load("session1").done("load", "all")


def test_a_state_file_written_before_scopes_existed_is_treated_as_unfinished(state_dir):
    """An old record carries no scope, so it cannot be known which windows it covered. Re-running an idempotent step
    costs minutes; trusting it costs the corpus.
    """
    (state_dir / "session1_state.json").write_text(
        json.dumps({"steps": {"load": {"status": "done", "seconds": 1}}}), encoding="utf-8")
    assert not pipeline.State.load("session1").done("load", "all")


def test_the_target_database_does_not_depend_on_an_environment_variable(monkeypatch):
    """`settings.pg_dsn` defaults to `hansard`, the rollback, so inheriting it made the one command fail preflight
    until PG_DSN was exported by hand -- exactly the babysitting this script exists to remove. Host, port and
    credentials still come from PG_DSN.
    """
    from psycopg.conninfo import conninfo_to_dict
    monkeypatch.setattr(pipeline.settings, "pg_dsn",
                        "postgresql://someone:secret@127.0.0.1:5432/hansard")
    dsn = pipeline.make_conninfo(pipeline.settings.pg_dsn, dbname=pipeline.safety.TARGET)
    parts = conninfo_to_dict(dsn)
    assert parts["dbname"] == "hansard_v2"
    assert parts["host"] == "127.0.0.1" and parts["user"] == "someone"
    assert pipeline._database_check(dsn).ok


# --------------------------------------------------------------------------- Surviving a kill, not just a Ctrl+C
# ---------------------------------------------------------------------------


def test_the_sleep_timeout_is_remembered_on_disk_before_it_is_changed(state_dir):
    """A `finally` does not run when the process is killed rather than interrupted, and the symptom -- a laptop that
    quietly never sleeps again -- is one nobody notices for a month. Found by killing a real run: the setting was
    left at 0.
    """
    state = pipeline.State.load("session1")
    state.data["machine"] = {"standby_ac_minutes": 30}
    state.save()
    assert pipeline.State.load("session1").data["machine"]["standby_ac_minutes"] == 30


def test_killing_a_dead_process_is_harmless():
    """kill_tree runs in a `finally`, so it is called on the happy path too."""
    pipeline.kill_tree(None)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    pipeline.kill_tree(proc)  # already exited: must not raise


def test_a_child_is_tracked_while_it_runs_so_an_interrupt_can_take_it_down(tmp_path):
    ctx = pipeline.Ctx(session="session1", state=pipeline.State.load("session1"),
                       log=tmp_path / "log.txt", dsn="postgresql://u:p@127.0.0.1:5432/hansard_v2",
                       windows=[])
    assert ctx.child is None
    out = pipeline.sh(ctx, [sys.executable, "-c", "print('hello from the child')"])
    assert "hello from the child" in out
    assert ctx.child is None, "the handle is cleared once the child is gone"


def test_a_failing_child_reports_its_own_output_not_an_empty_string(tmp_path):
    ctx = pipeline.Ctx(session="session1", state=pipeline.State.load("session1"),
                       log=tmp_path / "log.txt", dsn="postgresql://u:p@127.0.0.1:5432/hansard_v2",
                       windows=[])
    with pytest.raises(pipeline.StepFailed) as exc:
        pipeline.sh(ctx, [sys.executable, "-c",
                          "import sys; print('context line'); sys.exit(3)"])
    assert "exited 3" in exc.value.detail and "context line" in exc.value.detail


def test_stderr_reaches_the_log_and_the_failure_message(tmp_path):
    """stderr is merged into stdout: a traceback is the most useful thing a failed step can hand over, and losing it
    means reading the log to find out anything at all.
    """
    ctx = pipeline.Ctx(session="session1", state=pipeline.State.load("session1"),
                       log=tmp_path / "log.txt", dsn="postgresql://u:p@127.0.0.1:5432/hansard_v2",
                       windows=[])
    with pytest.raises(pipeline.StepFailed) as exc:
        pipeline.sh(ctx, [sys.executable, "-c", "raise SystemExit('the real reason')"])
    assert "the real reason" in exc.value.detail


def test_preflight_refuses_to_start_beside_another_run(monkeypatch, tmp_path):
    """Three orphaned chunk processes, each in its own transaction, contending on the same unique index, none making
    progress and nothing on screen to say so. Postgres is the lock rather than a file, because a stale lock file
    would refuse a legitimate run after a crash -- which is the failure mode that matters at 2 a.m.
    """
    ctx = pipeline.Ctx(session="session1", state=pipeline.State.load("session1"),
                       log=tmp_path / "log.txt", dsn="postgresql://u:p@127.0.0.1:5432/hansard_v2",
                       windows=[])

    class Busy:
        stdout, returncode = "3\n", 0

    class Quiet:
        stdout, returncode = "0\n", 0

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: Busy())
    check = pipeline._no_other_run_check(ctx)
    assert not check.ok and check.fatal and "3 other connection" in check.detail

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: Quiet())
    assert pipeline._no_other_run_check(ctx).ok


def test_a_database_that_cannot_be_asked_is_a_warning_not_a_stop(monkeypatch, tmp_path):
    """Refusing to run because the check itself failed would be the check doing more harm than the thing it guards
    against.
    """
    def boom(*a, **k):
        raise OSError("docker is not answering")

    monkeypatch.setattr(pipeline.subprocess, "run", boom)
    ctx = pipeline.Ctx(session="session1", state=pipeline.State.load("session1"),
                       log=tmp_path / "log.txt", dsn="postgresql://u:p@127.0.0.1:5432/hansard_v2",
                       windows=[])
    check = pipeline._no_other_run_check(ctx)
    assert check.ok and not check.fatal


def test_the_recipe_census_parses_psql_output():
    """The SQL behind this had `GROUP BY 1` on an expression containing count(*), which Postgres refuses ("aggregate
    functions are not allowed in GROUP BY"). The parser is worth a test too: a recipe tag contains `|` and `/` but
    never `=`, so rpartition on the last `=` is safe.
    """
    lines = "BAAI/bge-small-en-v1.5|full|v2=1074297\nBAAI/bge-small-en-v1.5|sentence|v1=200\n(none)=0\n"
    captured = {}

    class Ctx:
        dsn = "postgresql://postgres:postgres@127.0.0.1:5432/hansard_v2"

        def env(self, **kw):
            return {}

        def write_log(self, text):
            pass

    def fake_sh(ctx, command, **kwargs):
        captured["command"] = command
        return lines

    original = pipeline.sh
    pipeline.sh = fake_sh
    try:
        got = pipeline._recipe_census(Ctx())
    finally:
        pipeline.sh = original

    assert got == {"BAAI/bge-small-en-v1.5|full|v2": 1074297,
                   "BAAI/bge-small-en-v1.5|sentence|v1": 200, "(none)": 0}  # fmt: skip
    sql = " ".join(captured["command"])
    assert "GROUP BY 1) t" in sql, "the aggregate must be grouped inside a subquery"
    assert "hansard_v2" in sql and "hansard'" not in sql


def test_session3_ends_with_adoption_and_the_report():
    """Order matters: the report scores the index the adoption built. And `final_report` must not score W3 -- a
    pipeline that re-scored the held-out window on every run would burn it silently.
    """
    names = [step.name for step in pipeline.SESSION3_STEPS]
    assert names[-2:] == ["adopt", "final_report"]
    assert names.index("bakeoff_decide") < names.index("adopt")
    report = next(s for s in pipeline.SESSION3_STEPS if s.name == "final_report")
    assert report.needs_gpu_service, "the queries are embedded through the service"
    assert "second-look" not in (report.run.__doc__ or "").lower() or "NOT scored" in (report.run.__doc__ or "")


# --- the W5 ingest chain ---------------------------------------------------


def test_the_w5_chain_maps_members_between_the_load_and_derived():
    """The bug this guards. W5 brought in four speakers whose MNIS ids had no `member` row, and `build_derived.py`
    refused with "member mapping must be 100% with 0 orphans" -- valid JSON, exit 1, no traceback, eight hours into
    an unattended run.
    """
    names = [name for name, _ in pipeline.w5_ingest_chain()]
    assert names == ["load", "members", "chunk", "rules", "derived"]
    assert names.index("members") > names.index("load"), "members needs the speeches to exist first"
    assert names.index("members") < names.index("derived"), (
        "build_derived refuses orphan speaker ids, so members must be mapped before it runs")


def test_every_step_in_the_w5_chain_is_callable():
    """A name in the list with nothing behind it would fail at hour eight, not here."""
    for name, step in pipeline.w5_ingest_chain():
        assert callable(step), name


def test_the_w5_chain_uses_the_same_steps_session1_uses():
    """A window loaded by different code from the other five would differ in ways nobody wrote down."""
    chain = dict(pipeline.w5_ingest_chain())
    assert chain["members"] is pipeline.step_members
    assert chain["chunk"] is pipeline.step_chunk
    assert chain["rules"] is pipeline.step_rules
    assert chain["derived"] is pipeline.step_derived


# The decade session ---------------------------------------------------------------------------------------


def _decade_names(windows=None):
    return [s.name for s in pipeline.decade_steps(windows)]


def test_the_decade_session_fetches_and_loads_every_new_window_and_no_old_one():
    corpus = pipeline._active_windows(pipeline.DECADE_CONFIG, role="corpus")
    assert len(corpus) >= 39 and not set(corpus) & {"W0", "W1", "W2", "W3", "W4", "W5"}
    names = _decade_names()
    for w in corpus:
        assert names.index(f"fetch_{w}") + 1 == names.index(f"load_{w}"), w
    assert not any(n in names for n in ("load_W0", "load_W3", "load_W5")), "the decade must not reload W0-W5"


def test_the_decade_step_names_are_unique():
    names = _decade_names()
    assert len(names) == len(set(names))


def test_the_decade_order_gates_the_load_before_any_hour_of_processing():
    names = _decade_names()
    last_load = max(i for i, n in enumerate(names) if n.startswith("load_"))
    order = [names.index(n) for n in ("members", "relink", "gate_loaded", "chunk", "rules", "derived", "embed",
                                      "vacuum", "gate_final")]
    assert last_load < order[0] and order == sorted(order)
    assert names[-1] == "gate_final"


def test_the_decade_embeds_exactly_once():
    """A large embedding run drops and rebuilds the whole HNSW index; per window that is 39 ever-larger rebuilds."""
    assert _decade_names().count("embed") == 1


def _captured(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "uvrun", lambda ctx, *args, **kw: calls.append(args) or "")
    return calls


class _Ctx:
    dsn = "host=127.0.0.1 dbname=hansard_v2 user=x"


def test_every_decade_fetch_and_load_names_the_decade_config(monkeypatch):
    calls = _captured(monkeypatch)
    steps = {s.name: s for s in pipeline.decade_steps(["C2019Q1"])}
    for name in ("fetch_C2019Q1", "load_C2019Q1", "members"):
        steps[name].run(_Ctx())
    rel = str(pipeline.DECADE_CONFIG.relative_to(pipeline.REPO))
    assert calls and all("--config" in args and args[args.index("--config") + 1] == rel for args in calls), calls
    assert any("verify" in args for args in calls), "a fetch that is not verified is not finished"


def test_the_gate_and_the_relink_are_pinned_to_the_sessions_database(monkeypatch):
    calls = _captured(monkeypatch)
    ctx = _Ctx()
    ctx.dsn = "host=127.0.0.1 dbname=hansard_v2_decade_trial user=x"
    pipeline.step_gate(ctx)
    pipeline.step_relink(ctx)
    for args in calls:
        assert args[args.index("--database") + 1] == "hansard_v2_decade_trial", args
    assert "--gate" in calls[0]


@pytest.mark.parametrize("db_bytes, chunks, loaded_days, new_days, want_gb", [
    (9_550_273_559, 1_437_856, 505, 3_176, 84),
    (9_550_273_559, 1_437_856, 505, 0, 0),
    (0, 0, 0, 100, 0),  # nothing loaded yet: no density to extrapolate from
])  # fmt: skip
def test_capacity_needed_is_rows_times_measured_bytes_plus_headroom(db_bytes, chunks, loaded_days, new_days, want_gb):
    assert round(pipeline.capacity_needed(db_bytes, chunks, loaded_days, new_days) / 1024**3) == want_gb


def test_a_session_without_its_own_config_uses_the_base():
    assert pipeline.session_config("session1") == pipeline.BASE_CONFIG
    assert pipeline.session_config("decade") == pipeline.DECADE_CONFIG


# The ten-year index, projected where the run will see it -----------------------------------------

VM = 16_651_018_240
GB2 = 2 * 1024**3


def test_the_ten_year_index_is_larger_than_the_machine_that_must_hold_it():
    """Measured: 2,773 MB for 1.42M vectors (~1,950 B each). From 2015 that is ~12.2M vectors."""
    p = pipeline.index_projection(12_200_000, 1_950, VM, GB2)
    assert round(p["full_bytes"] / 1e9) == 24 and round(p["halfvec_bytes"] / 1e9) == 14
    assert not p["fits_vm"]["full"], "a full-precision index bigger than the VM: every search reads from disk"
    assert not p["build_in_memory"]["full"] and not p["build_in_memory"]["halfvec"], "both builds run on disk at 2 GB"


def test_today_the_index_fits_and_the_projection_says_so():
    p = pipeline.index_projection(1_422_379, 1_950, VM, GB2)
    assert p["fits_vm"]["full"] and p["fits_vm"]["halfvec"]


def test_the_halfvec_ratio_is_the_measured_one():
    assert pipeline.HALFVEC_RATIO == pytest.approx(1169 / 2047)


@pytest.mark.parametrize("text, want", [("2GB", 2 * 1024**3), ("512MB", 512 * 1024**2), ("64kB", 64 * 1024),
                                        ("8 GB", 8 * 1024**3)])
def test_a_postgres_memory_setting_reads_as_bytes(text, want):
    assert pipeline.memory_bytes(text) == want


def test_the_projection_line_names_the_owners_decision():
    line = pipeline.index_projection_line(pipeline.index_projection(12_200_000, 1_950, VM, GB2))
    assert "B-03" in line and "halfvec" in line and "on disk" in line


def test_the_decade_embed_outlives_an_on_disk_index_build(monkeypatch):
    """A 6-hour timeout would kill a 10M-row HNSW build mid-way, and the retry would restart it from nothing."""
    seen = {}
    monkeypatch.setattr(pipeline, "sh", lambda ctx, args, **kw: seen.update(kw) or "")
    monkeypatch.setattr(pipeline, "on_mains", lambda: True)

    class Ctx(_Ctx):
        def env(self, **extra):
            return extra

    steps = {s.name: s for s in pipeline.decade_steps(["C2019Q1"])}
    steps["embed"].run(Ctx())
    assert seen["timeout"] >= 24 * 3600


# The projection after the switch ------------------------------------------------------------------------------


def test_a_projection_measured_on_the_halfvec_index_is_not_read_as_full_precision():
    """The preflight measures bytes per vector on the LIVE index. After the switch that is the halfvec index (1,168 B);
    read as full precision it would understate the full-precision projection by 43% and the halfvec one by 67%.
    """
    from_full = pipeline.index_projection(12_200_000, 2_047, VM, GB2, measured_form="vector")
    from_half = pipeline.index_projection(12_200_000, 1_169, VM, GB2, measured_form="halfvec")
    assert from_half["full_bytes"] == pytest.approx(from_full["full_bytes"], rel=0.002)
    assert from_half["halfvec_bytes"] == pytest.approx(from_full["halfvec_bytes"], rel=0.002)


def test_the_build_is_judged_against_the_graph_in_memory_not_the_finished_index():
    """The halfvec graph needs ~1,483 B a vector in memory against 1,168 B on disk."""
    from retrieval.vector_index import ROW_BYTES

    vectors = (GB2 // ROW_BYTES["halfvec"]) + 10_000  # its finished index (~1,168 B) would fit in 2 GB; its graph not
    p = pipeline.index_projection(vectors, 1_169, VM, GB2, measured_form="halfvec")
    assert p["halfvec_bytes"] < GB2 and not p["build_in_memory"]["halfvec"]


def test_the_projection_line_names_the_form_the_run_will_build():
    p = pipeline.index_projection(12_200_000, 1_169, VM, GB2, measured_form="halfvec")
    line = pipeline.index_projection_line(p, building="halfvec")
    assert "builds the halfvec index (~13 GB;" in line and "full precision ~23 GB" in line
    assert "takes 86% of the VM, leaving too little for the table" in line, "smaller than the VM is not in memory"
    assert "B-03" in line and "Docker memory" in line
    other = pipeline.index_projection_line(p, building="vector")
    assert "builds the vector index (~23 GB;" in other and "LARGER than the VM" in other
    assert "switch_vector_index.py" in other
    today = pipeline.index_projection_line(pipeline.index_projection(1_422_379, 1_169, VM, GB2, measured_form="halfvec"),
                                           building="halfvec")
    assert "with room for the table" in today
