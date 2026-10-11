"""The one command per session: preflight, then every long step, resumable."""

import argparse
import json
import os
import re
import shutil

import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable

from psycopg.conninfo import make_conninfo
from rich.console import Console
from rich.table import Table

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from app import safety  # noqa: E402
from app.config import settings  # noqa: E402

console = Console()

STATE_DIR = REPO / "data" / "pipeline"
LOG_DIR = REPO / "data" / "logs"
REPORTS = REPO / "reports"
GPU_DIR = REPO / "gpu"
GPU_URL = settings.gpu_url

# Free space the whole run needs.
MIN_FREE_GB = 20.0

# For the HNSW build at 1.2M rows.
MAINTENANCE_WORK_MEM = "2GB"


class StepFailed(RuntimeError):
    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step, self.detail = step, detail


# --------------------------------------------------------------------------- State
# ---------------------------------------------------------------------------


@dataclass
class State:
    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, session: str) -> "State":
        path = STATE_DIR / f"{session}_state.json"
        data = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        return cls(path=path, data=data or {"steps": {}, "started_at": None})

    def done(self, name: str, scope: str = "") -> bool:
        """Did this step finish FOR THIS SCOPE?"""
        record = self.data.get("steps", {}).get(name, {})
        return record.get("status") == "done" and record.get("scope", "") == scope

    def record(self, name: str, status: str, seconds: float, payload: dict | None = None,
               scope: str = "") -> None:
        self.data.setdefault("steps", {})[name] = {
            "status": status, "scope": scope, "seconds": round(seconds, 1),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "payload": redact(payload or {}),
        }  # fmt: skip
        self.save()

    def payload(self, name: str) -> dict:
        return self.data.get("steps", {}).get(name, {}).get("payload", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def reset(self, names: list[str] | None = None) -> None:
        if names is None:
            self.data["steps"] = {}
        else:
            for name in names:
                self.data.get("steps", {}).pop(name, None)
        self.save()


# --------------------------------------------------------------------------- Running things
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    session: str
    state: State
    log: Path
    dsn: str
    windows: list[str]
    dry_run: bool = False
    started: float = field(default_factory=time.monotonic)
    gpu_process: subprocess.Popen | None = None
    # The step's child, so an interrupt can take it down with us.
    child: subprocess.Popen | None = None
    notes: list[str] = field(default_factory=list)

    def env(self, **extra: str) -> dict[str, str]:
        env = {**os.environ, "PG_DSN": self.dsn, "PYTHONIOENCODING": "utf-8", **extra}
        # `uv run` warns, on stderr, that an inherited VIRTUAL_ENV does not match the project's -- printing an absolute
        # path that then lands in a committed report.
        env.pop("VIRTUAL_ENV", None)
        return env

    def write_log(self, text: str) -> None:
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(text)


def sh(ctx: Ctx, args: list[str], *, cwd: Path = REPO, timeout: float = 14_400,
       env: dict[str, str] | None = None) -> str:
    """Run a command, stream nothing, log everything, and raise with the tail on failure."""
    printable = " ".join(args)
    ctx.write_log(f"\n{'=' * 78}\n$ {printable}\n  (cwd {cwd.relative_to(REPO) if cwd != REPO else '.'})\n")
    started = time.monotonic()
    proc = subprocess.Popen(args, cwd=str(cwd), env=env or ctx.env(), text=True,
                            encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ctx.child = proc
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        kill_tree(proc)
        proc.communicate()
        raise StepFailed(printable, f"timed out after {timeout / 60:.0f} minutes") from exc
    except BaseException:  # Ctrl+C reaches the parent first
        kill_tree(proc)
        raise
    finally:
        ctx.child = None
    output = stdout or ""
    ctx.write_log(output)
    ctx.write_log(f"\n[exit {proc.returncode} after {time.monotonic() - started:.1f}s]\n")
    if proc.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-25:])
        raise StepFailed(printable, f"exited {proc.returncode}\n{tail}")
    return output


def uvrun(ctx: Ctx, *args: str, cwd: Path = REPO, **kw) -> str:
    return sh(ctx, ["uv", "run", "python", *args], cwd=cwd, **kw)


def kill_tree(proc: subprocess.Popen | None) -> None:
    """Kill a child and everything it started."""
    if proc is None or proc.poll() is not None:
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


# --------------------------------------------------------------------------- Machine management
# ---------------------------------------------------------------------------

_AC_INDEX = re.compile(r"Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)")


def standby_timeout_minutes() -> int | None:
    """The AC sleep timeout in minutes, or None if it cannot be read."""
    try:
        out = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _AC_INDEX.search(out or "")
    return int(match.group(1), 16) // 60 if match else None


def set_standby_timeout(minutes: int) -> bool:
    try:
        return subprocess.run(["powercfg", "/change", "standby-timeout-ac", str(minutes)],
                              capture_output=True, text=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def on_mains() -> bool | None:
    """True on mains, False on battery, None if it cannot be told."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; "
             "if ($null -eq $b) { 'mains' } elseif ($b.BatteryStatus -eq 1) { 'battery' } else { 'mains' }"],
            capture_output=True, text=True, timeout=25)
    except (OSError, subprocess.SubprocessError):
        return None
    answer = (out.stdout or "").strip().splitlines()
    if not answer:
        return None
    return answer[-1].strip() == "mains"


def gpu_healthy(timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"{GPU_URL}/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def start_gpu(ctx: Ctx, wait_s: float = 180.0) -> bool:
    """Start `gpu/api.py` and wait for /health. Returns True if this call started it."""
    if gpu_healthy():
        note = "the GPU service was already running; this run will not stop it"
        if note not in ctx.notes:
            ctx.notes.append(note)
        return False
    log = LOG_DIR / f"{ctx.session}_gpu_service.log"
    handle = log.open("a", encoding="utf-8")
    ctx.gpu_process = subprocess.Popen(["uv", "run", "python", "api.py"], cwd=str(GPU_DIR),
                                       stdout=handle, stderr=subprocess.STDOUT, env=ctx.env())
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if gpu_healthy():
            return True
        if ctx.gpu_process.poll() is not None:
            raise StepFailed("gpu service", f"it exited with code {ctx.gpu_process.returncode}; see {log.name}")
        time.sleep(2.0)
    raise StepFailed("gpu service", f"no /health answer within {wait_s:.0f}s; see {log.name}")


def stop_gpu(ctx: Ctx) -> None:
    proc, ctx.gpu_process = ctx.gpu_process, None
    if proc is None or proc.poll() is not None:
        return
    # CTRL_BREAK would need a new process group; terminate is what uvicorn handles cleanly here, and the service
    # releases its VRAM in the lifespan shutdown either way.
    proc.terminate()
    try:
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------- Preflight
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def _docker_check() -> Check:
    if shutil.which("docker") is None:
        return Check("docker", False, "the docker command is not on PATH")
    try:
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", "hansard-db"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("docker", False, f"could not ask Docker about hansard-db ({type(exc).__name__})")
    status = (out.stdout or "").strip()
    if out.returncode != 0:
        return Check("docker", False, "the hansard-db container is not running (docker compose up -d)")
    return Check("docker", status == "healthy", f"hansard-db is {status or 'in an unknown state'}")


def _disk_check() -> Check:
    usage = shutil.disk_usage(REPO)
    free_gb = usage.free / 1024**3
    return Check("disk", free_gb >= MIN_FREE_GB,
                 f"{free_gb:.1f} GB free (need {MIN_FREE_GB:.0f} GB)")


# `_capacity_check` re-measures it on the live database; this is only the fallback.
BYTES_PER_CHUNK = 6_650
# WAL, the HNSW rebuild's working space and VACUUM, on top of the rows themselves.
CAPACITY_HEADROOM = 1.5


def capacity_needed(db_bytes: int, chunks: int, loaded_days: int, new_days: int) -> int:
    """Bytes the new windows will need: their days at the loaded corpus's chunks per day, at its bytes per chunk, plus
    headroom. Pure, so the arithmetic is tested without a database.
    """
    per_chunk = db_bytes / chunks if chunks else BYTES_PER_CHUNK
    per_day = chunks / loaded_days if loaded_days else 0
    return int(new_days * per_day * per_chunk * CAPACITY_HEADROOM)


def _capacity_check(ctx: Ctx, windows: Sequence[str]) -> Check:
    """Room for the plan, measured: a fixed MIN_FREE_GB was sized for five windows, and the ten-year corpus is ~7x.
    Free space is read on the repository's drive, where Docker Desktop keeps the database's virtual disk here.
    """
    try:
        import psycopg

        from ingest.corpus_fetch import load_config

        cfg = load_config(DECADE_CONFIG, today=datetime.now(UTC).date())
        plan = {w.id: w for w in cfg.windows}
        new_days = sum(len(plan[w].days) for w in windows if w in plan)
        with psycopg.connect(ctx.dsn, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
            db_bytes, chunks = conn.execute(
                "SELECT pg_database_size(current_database()), (SELECT count(*) FROM chunk)").fetchone()
            loaded_days = conn.execute(
                "SELECT coalesce(sum(end_date - start_date + 1), 0) FROM corpus_window "
                "WHERE window_id IN (SELECT DISTINCT window_id FROM chunk)").fetchone()[0]
    except Exception as exc:  # noqa: BLE001 - reported as a failed check, never swallowed
        return Check("capacity", False, f"could not measure: {type(exc).__name__}: {str(exc)[:120]}")
    need = capacity_needed(db_bytes, chunks, int(loaded_days), new_days)
    free = shutil.disk_usage(REPO).free
    return Check("capacity", free >= need,
                 f"{len(windows)} window(s), {new_days:,} days: need ~{need / 1024**3:.0f} GB, "
                 f"{free / 1024**3:.0f} GB free (measured {db_bytes / max(chunks, 1):,.0f} B/chunk)")


# A halfvec HNSW index's bytes per vector over a full-precision one's, measured on 400k real vectors at equal recall@10
# (reports/b03_index_options.json: 1,169 vs 2,047 B).
HALFVEC_RATIO = 1169 / 2047


def memory_bytes(setting: str) -> int:
    """A Postgres-style memory setting ("2GB", "512MB", "64kB") in bytes."""
    number, unit = re.fullmatch(r"\s*(\d+)\s*([kMGT]?B)\s*", setting).groups()
    return int(number) * {"B": 1, "kB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]


def index_projection(vectors: int, bytes_per_vector: float, vm_bytes: int, build_bytes: int,
                     measured_form: str = "vector") -> dict:
    """The HNSW index a corpus of `vectors` needs, full precision and halfvec, against the memory that must hold it:
    the Docker VM (for searches) and the build's maintenance_work_mem (for an in-memory build). Pure.
    """
    from retrieval.vector_index import ROW_BYTES

    full = int(vectors * (bytes_per_vector if measured_form == "vector" else bytes_per_vector / HALFVEC_RATIO))
    sizes = {"full": full, "halfvec": int(full * HALFVEC_RATIO)}
    graph = {"full": vectors * ROW_BYTES["vector"], "halfvec": vectors * ROW_BYTES["halfvec"]}
    return {"vectors": vectors, "full_bytes": sizes["full"], "halfvec_bytes": sizes["halfvec"],
            "vm_bytes": vm_bytes, "build_bytes": build_bytes,
            "fits_vm": {k: v < vm_bytes for k, v in sizes.items()},
            "vm_share": {k: v / vm_bytes for k, v in sizes.items()},
            "build_in_memory": {k: v <= build_bytes for k, v in graph.items()}}


def index_projection_line(p: dict, building: str = "vector") -> str:
    """The preflight's sentence, about the index the embed step will BUILD (the form frozen retrieval queries)."""
    gb = 1024**3
    key = "halfvec" if building == "halfvec" else "full"
    size = p["halfvec_bytes"] if key == "halfvec" else p["full_bytes"]
    build = "in memory" if p["build_in_memory"][key] else "on disk (hours; the 48 h embed step allows for it)"
    share = p["vm_share"][key]
    if not p["fits_vm"][key]:
        search = "is LARGER than the VM, so searches read it from disk"
    elif share > 0.5:
        # The index is not alone in the VM: the table's heap, shared_buffers and Postgres itself want room too, so
        # "smaller than the VM" is not "held in memory".
        search = f"takes {share:.0%} of the VM, leaving too little for the table: searches will read from disk"
    else:
        search = f"takes {share:.0%} of the VM, with room for the table"
    decision = ("halfvec adopted (T3); Docker memory (~24 GB, so the index stays in memory) is the owner's"
                if key == "halfvec" else
                "Docker memory and the halfvec switch (scripts/switch_vector_index.py), measured on dev first")
    return (f"~{p['vectors'] / 1e6:.1f}M vectors: the embed step builds the {building} index (~{size / gb:.0f} GB; "
            f"full precision ~{p['full_bytes'] / gb:.0f} GB, halfvec ~{p['halfvec_bytes'] / gb:.0f} GB); VM "
            f"{p['vm_bytes'] / gb:.1f} GB, build memory {p['build_bytes'] / gb:.0f} GB -> built {build}; the index "
            f"{search}. B-03: {decision}")


def _index_memory_check(ctx: Ctx, windows: Sequence[str]) -> Check:
    """The ten-year index against the memory that has to hold it. Never fatal — the run works either way, and how fast
    it builds and searches is the maintainer's decision, not the preflight's.
    """
    try:
        import psycopg

        from ingest.corpus_fetch import load_config

        cfg = load_config(DECADE_CONFIG, today=datetime.now(UTC).date())
        plan = {w.id: w for w in cfg.windows}
        new_days = sum(len(plan[w].days) for w in windows if w in plan)
        with psycopg.connect(ctx.dsn, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
            index_bytes, embedded, chunks, definition = conn.execute(
                "SELECT coalesce((SELECT pg_relation_size('chunk_embedding_idx'::regclass)), 0), "
                "(SELECT count(*) FROM chunk WHERE embedding IS NOT NULL), (SELECT count(*) FROM chunk), "
                "pg_get_indexdef(to_regclass('chunk_embedding_idx'))").fetchone()
            loaded_days = conn.execute(
                "SELECT coalesce(sum(end_date - start_date + 1), 0) FROM corpus_window "
                "WHERE window_id IN (SELECT DISTINCT window_id FROM chunk)").fetchone()[0]
        vm = int(subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True, text=True,
                                timeout=30).stdout.strip())
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed; not fatal (see the docstring)
        return Check("index memory", True, f"could not project: {type(exc).__name__}: {str(exc)[:100]}", fatal=False)
    from retrieval.vector_index import form_of, frozen_form

    measured = form_of(definition) or "vector"
    per_vector = index_bytes / embedded if embedded and index_bytes else 2047
    if not (embedded and index_bytes):
        measured = "vector"  # the fallback figure is the full-precision measurement
    vectors = embedded + int(new_days * (chunks / loaded_days if loaded_days else 0))
    projection = index_projection(vectors, per_vector, vm, memory_bytes(MAINTENANCE_WORK_MEM), measured_form=measured)
    return Check("index memory", True, index_projection_line(projection, building=frozen_form()), fatal=False)


def _database_check(dsn: str) -> Check:
    try:
        name = safety.require(dsn, safety.TARGET)
    except safety.WrongDatabase as exc:
        return Check("target database", False, str(exc))
    return Check("target database", True, f"{safety.database_of(name)} (never {', '.join(safety.PROTECTED)})")


def _keys_check() -> Check:
    """Present or absent, never printed."""
    present = [name for name, value in (("DEEPSEEK_KEY", settings.deepseek_key),) if value]
    return Check("api keys", True,
                 f"{len(present)} of 1 present ({', '.join(present) or 'none'}); not needed this session",
                 fatal=False)


def _gpu_check() -> Check:
    try:
        out = subprocess.run(["uv", "run", "python", "check_cuda.py"], cwd=str(GPU_DIR),
                             capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("gpu", False, f"check_cuda.py could not be run ({type(exc).__name__})")
    first = next((line for line in (out.stdout or "").splitlines() if line.strip()), "")
    return Check("gpu", out.returncode == 0, first[:120] or f"check_cuda.py exited {out.returncode}")


def _power_check() -> Check:
    mains = on_mains()
    if mains is None:
        return Check("power", True, "could not tell; record the power source by hand", fatal=False)
    return Check("power", mains, "on mains" if mains else
                 "ON BATTERY: the GPU is power-capped to about a quarter of its speed (I-33)")


def _shm_check() -> Check:
    """Docker's /dev/shm and the memory Postgres can use for the index build."""
    try:
        out = subprocess.run(["docker", "exec", "hansard-db", "df", "-B1", "/dev/shm"],
                             capture_output=True, text=True, timeout=30)
        shm_mb = int((out.stdout or "").splitlines()[-1].split()[1]) / 1024**2
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return Check("container memory", True, "could not read /dev/shm; index builds stay serial", fatal=False)
    return Check("container memory", True,
                 f"/dev/shm {shm_mb:.0f} MB, so the HNSW build stays serial (I-25); "
                 f"maintenance_work_mem is set to {MAINTENANCE_WORK_MEM} for it",
                 fatal=False)


def _no_other_run_check(ctx: Ctx) -> Check:
    """Is another run already writing to this database?"""
    name = safety.database_of(ctx.dsn)
    try:
        out = subprocess.run(
            ["docker", "exec", "hansard-db", "psql", "-U", "postgres", "-tAc",
             "SELECT count(*) FROM pg_stat_activity WHERE datname = "
             f"{_sql_literal(name)} AND state <> 'idle' AND pid <> pg_backend_pid() "
             # `client backend` only: an autovacuum worker tidying up after the last step is normal, and a check that
             # stopped the run for it would cry wolf on every start.
             "AND backend_type = 'client backend' AND application_name <> 'psql'"],
            capture_output=True, text=True, timeout=60)
        busy = int((out.stdout or "0").strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return Check("no other run", True, "could not ask Postgres; check by hand", fatal=False)
    if busy:
        return Check("no other run", False,
                     f"{busy} other connection(s) are working in {name}. Another pipeline, "
                     "load or embed is already running, or one was killed and left a "
                     "process behind. Stop it before starting this run.")
    return Check("no other run", True, f"nothing else is writing to {name}")


def _cache_check(ctx: Ctx, fetching: Sequence[str] = (), config: Path | None = None) -> Check:
    """`verify` over every window, offline. It is the gate on the data the load reads: if the cache disagrees with
    itself or with the second source, nothing after this matters.
    """
    config = config or BASE_CONFIG
    command = ["uv", "run", "python", "scripts/fetch_corpus_v2.py", "--config", str(config.relative_to(REPO)), "verify"]
    if fetching:
        # Verify only the windows that are supposed to be there already.
        present = [w for w in _active_windows(config) if w not in set(fetching)]
        for window in present:
            command += ["--window", window]
    try:
        out = sh(ctx, command, timeout=1800)
    except StepFailed as exc:
        return Check("fetch cache", False, exc.detail.splitlines()[-1][:200])
    clean = "problem" not in out.lower() or "0 problem" in out.lower()
    summary = next((line for line in reversed(out.splitlines()) if line.strip()), "")
    note = summary[:160]
    if fetching:
        note = f"{note}  (not checking {', '.join(fetching)}: this session fetches them)"
    return Check("fetch cache", clean, note)


def _scripts_check(steps: "Sequence[Step]") -> Check:
    """Every script the chosen steps shell out to must exist before the first row moves."""
    absent = missing_scripts(steps)
    if not absent:
        return Check("step scripts", True, f"all {sum(len(s.needs_scripts) for s in steps)} present")
    detail = "; ".join(f"{step} needs {', '.join(names)}" for step, names in absent.items())
    return Check("step scripts", False, detail[:300])


# session -> the windows it fetches itself, so preflight does not fail on their absence from the cache.
SESSION_FETCHES: dict[str, tuple[str, ...]] = {"session15": ("W5",)}


def preflight(ctx: Ctx, steps: "Sequence[Step]" = (), session: str = "") -> list[Check]:
    config = session_config(session)
    fetching = SESSION_FETCHES.get(session, ())
    if session == "decade":
        fetching = tuple(ctx.windows or _active_windows(DECADE_CONFIG, role="corpus"))
    checks = [_docker_check(), _disk_check(), _database_check(ctx.dsn),
              _no_other_run_check(ctx), _power_check(),
              _gpu_check(), _keys_check(), _shm_check(),
              _cache_check(ctx, fetching, config)]
    if session == "decade":
        checks.append(_capacity_check(ctx, fetching))
        checks.append(_index_memory_check(ctx, fetching))
    if steps:
        checks.append(_scripts_check(steps))
    table = Table(title="Preflight")
    table.add_column("check")
    table.add_column("")
    table.add_column("detail")
    for check in checks:
        mark = "[green]ok[/green]" if check.ok else (
            "[bold red]FAIL[/bold red]" if check.fatal else "[yellow]warn[/yellow]")
        table.add_row(check.name, mark, check.detail)
    console.print(table)
    return checks


# --------------------------------------------------------------------------- The steps
# ---------------------------------------------------------------------------


@dataclass
class Step:
    name: str
    title: str
    minutes: float
    run: Callable[[Ctx], dict]
    needs_gpu_service: bool = False
    # Repo-relative scripts this step shells out to.
    needs_scripts: tuple[str, ...] = ()
    # False while the step's work is not actually implemented yet.
    ready: bool = True
    blocked: str = ""


def step_migrate(ctx: Ctx) -> dict:
    """Create `hansard_v2` if it does not exist, apply the baseline, then migrate."""
    name = safety.database_of(ctx.dsn)
    if name in safety.PROTECTED:
        raise StepFailed("migrate", f"refusing to build {name!r}: it is the rollback corpus")
    existing = sh(ctx, ["docker", "exec", "hansard-db", "psql", "-U", "postgres", "-tAc",
                        "SELECT 1 FROM pg_database WHERE datname = %s" % _sql_literal(name)],
                  timeout=120).strip()
    created = False
    if existing != "1":
        sh(ctx, ["docker", "exec", "hansard-db", "psql", "-U", "postgres", "-c",
                 f'CREATE DATABASE "{name}"'], timeout=600)
        created = True
    tables = sh(ctx, ["docker", "exec", "hansard-db", "psql", "-U", "postgres", "-d", name, "-tAc",
                      "SELECT to_regclass('public.chunk') IS NOT NULL"], timeout=120).strip()
    if tables != "t":
        # The baseline.
        schema = (REPO / "db" / "schema.sql").read_text(encoding="utf-8")
        proc = subprocess.run(["docker", "exec", "-i", "hansard-db", "psql", "-U", "postgres",
                               "-d", name, "-v", "ON_ERROR_STOP=1", "-q"],
                              input=schema, capture_output=True, text=True, timeout=1800)
        ctx.write_log((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0:
            raise StepFailed("migrate", f"db/schema.sql failed: {(proc.stderr or '')[-500:]}")
    out = uvrun(ctx, "-m", "app.migrate", timeout=1800)
    status = uvrun(ctx, "-m", "app.migrate", "--status", timeout=600)
    pending = [line for line in status.splitlines() if line.startswith("PENDING")]
    if pending:
        raise StepFailed("migrate", f"still pending after migrating: {pending}")
    return {"database": name, "created": created,
            "applied": out.strip().splitlines()[-1] if out.strip() else "", "pending": 0}


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def step_load_window(window: str, config: Path | None = None):
    """One window's load, as its own step."""
    extra = ["--config", str(config.relative_to(REPO))] if config else []

    def run_it(ctx: Ctx) -> dict:
        out = uvrun(ctx, "scripts/load_corpus_v2.py", *extra, "--json", "--no-sync-filters",
                    "--no-members", "--window", window, timeout=14_400)
        return {"window": window, "load": _last_json(out)}
    return run_it


def step_members(ctx: Ctx) -> dict:
    """Members are shared across windows, so they load once, after every window's speeches (the fallback name for a
    member the API does not know comes from `speech.speaker_name`).
    """
    out = uvrun(ctx, "scripts/load_corpus_v2.py", "--members-only", "--no-sync-filters",
                timeout=7200)
    return {"tail": _tail(out, 3)}


def step_chunk(ctx: Ctx) -> dict:
    out = uvrun(ctx, "-m", "ingest.chunk", timeout=14_400)
    return {"tail": _tail(out, 12)}


def step_rules(ctx: Ctx) -> dict:
    out = uvrun(ctx, "-m", "ingest.rules", timeout=14_400)
    return {"tail": _tail(out, 6)}


def step_derived(ctx: Ctx) -> dict:
    """Everything computed from the loaded rows: chunk filter columns, the lexicon, the office index, and the Wikidata
    birth dates.
    """
    out = uvrun(ctx, "scripts/load_corpus_v2.py", "--sync-only", timeout=3600)
    derived = uvrun(ctx, "scripts/build_derived.py", "--json", timeout=7200)
    dob = uvrun(ctx, "scripts/backfill_dob.py", timeout=3600)
    return {"filters": _tail(out, 2), "derived": _last_json(derived), "dob": _tail(dob, 4)}


def step_embed(ctx: Ctx) -> dict:
    out = sh(ctx, ["uv", "run", "python", "embedder.py"], cwd=GPU_DIR, timeout=21_600,
             env=ctx.env(PGOPTIONS=f"-c maintenance_work_mem={MAINTENANCE_WORK_MEM}"))
    return {"tail": _tail(out, 18), "power": "mains" if on_mains() else "battery"}


def step_vacuum(ctx: Ctx) -> dict:
    name = safety.database_of(ctx.dsn)
    sh(ctx, ["docker", "exec", "hansard-db", "psql", "-U", "postgres", "-d", name,
             "-c", "VACUUM ANALYZE chunk", "-c", "VACUUM ANALYZE speech", "-c", "VACUUM ANALYZE debate"],
       timeout=7200)
    return {"vacuumed": ["chunk", "speech", "debate"]}


def step_oracle(ctx: Ctx) -> dict:
    """The W0 rebuild oracle. It is allowed to report differences; it is not allowed to report one nobody has looked
    at.
    """
    path = REPORTS / "oracle_w0.json"
    try:
        uvrun(ctx, "scripts/oracle_w0.py", "--json", str(path), timeout=3600)
    except StepFailed as exc:
        raise StepFailed("oracle", "the W0 rebuild differs from `hansard` in a way nobody has "
                                   f"explained yet; see {path.name}\n{exc.detail}") from exc
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# Both evaluations are run at the production ef_search AND at a much higher one.
EF_SEARCH_SETTINGS = (100, 500)


def _evaluate(ctx: Ctx, stem: str, extra: list[str]) -> dict:
    out: dict[str, Any] = {}
    for ef in EF_SEARCH_SETTINGS:
        path = REPORTS / f"{stem}_ef{ef}.json"
        uvrun(ctx, "scripts/eval_retrieval.py", "--json", str(path), "--ef-search", str(ef),
              *extra, timeout=7200)
        out[f"ef{ef}"] = {"report": path.name, **_headline(path)}
    return out


def step_eval_old(ctx: Ctx) -> dict:
    """The scale tax: the same 25 questions, the same code, the same weights, 13x the corpus."""
    return _evaluate(ctx, "session1_eval_old_weights", [])


def step_eval_fixed(ctx: Ctx) -> dict:
    """The same questions with the scale fixes switched on, so the two numbers differ by exactly one thing."""
    return _evaluate(ctx, "session1_eval_scale_fixes", ["--scale-fixes"])


def step_measure(ctx: Ctx) -> dict:
    path = REPORTS / "session1_measurements.json"
    uvrun(ctx, "scripts/measure_retrieval.py", "--json", str(path), timeout=14_400)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


BASE_CONFIG = REPO / "config" / "corpus_v2.toml"
DECADE_CONFIG = REPO / "config" / "corpus_decade.toml"
# A session's windows come from ITS config, as its database does: `decade` is the ten-year plan.
SESSION_CONFIG: dict[str, Path] = {"decade": DECADE_CONFIG}


def session_config(session: str) -> Path:
    return SESSION_CONFIG.get(session, BASE_CONFIG)


# Windows in load order, taken from the same config the fetch used, so adding a window to a config adds a step here
# without touching this file.
def _active_windows(config: Path = BASE_CONFIG, role: str | None = None) -> list[str]:
    try:
        from ingest.corpus_fetch import load_config
        from datetime import date as _date
        return [w.id for w in load_config(config, today=_date.today()).active_windows
                if role is None or w.role == role]
    except Exception:  # noqa: BLE001 - a bad config is preflight's problem, not import time's
        return ["W0", "W4", "W2", "W1", "W3"] if config == BASE_CONFIG and role is None else []


# Rounded up to the nearest minute so the estimate is a ceiling; the load is dominated by re-reading the gzip cache, not
# by the writes.
_LOAD_MINUTES = {"W0": 1, "W4": 1, "W2": 1, "W1": 1, "W3": 1}


def session1_steps(windows: list[str] | None = None) -> list[Step]:
    chosen = windows or _active_windows()
    return [
        Step("migrate", "Build and migrate hansard_v2", 1, step_migrate),
        *[Step(f"load_{w}", f"Load {w} from the cache", _LOAD_MINUTES.get(w, 20),
               step_load_window(w)) for w in chosen],
        Step("members", "Load every member seen in any window", 3, step_members),
        *SESSION1_TAIL,
    ]


SESSION1_TAIL: list[Step] = [
    Step("chunk", "Chunk every speech", 10, step_chunk),
    Step("rules", "Procedure rules and boilerplate", 5, step_rules),
    Step("derived", "Filter columns, lexicon, offices, birth dates", 10, step_derived),
    Step("embed", "Embed the corpus and rebuild the HNSW index", 30, step_embed),
    Step("vacuum", "VACUUM ANALYZE", 10, step_vacuum),
    Step("oracle", "The W0 rebuild oracle", 2, step_oracle),
    Step("eval_old", "Evaluate at the OLD weights (the scale tax)", 5, step_eval_old, needs_gpu_service=True),
    Step("eval_fixed", "Evaluate with the scale fixes on", 5, step_eval_fixed, needs_gpu_service=True),
    Step("measure", "Latency, HNSW recall, filters, storage", 25, step_measure, needs_gpu_service=True),
]

# Everything here runs unattended for hours, so every step is resumable, every artifact is written before the next step
# starts, and nothing touches the production vectors until a winner has been established on dev.


def _bakeoff(ctx: Ctx, *args: str, timeout: float = 14_400) -> str:
    """Run the GPU worker from gpu/, which is a separate uv project with torch."""
    return sh(ctx, ["uv", "run", "python", "bakeoff.py", *args], cwd=REPO / "gpu", timeout=timeout)


def step_bakeoff_corpus(ctx: Ctx) -> dict:
    """Choose the screening corpus once and write it down."""
    out = _bakeoff(ctx, "--build-screen-corpus", "--json", str(REPORTS / "session3_screen_corpus.json"),
                   timeout=1_800)  # fmt: skip
    return _last_json(out) or {"note": _tail(out, 4)}


def step_bakeoff_smoke(ctx: Ctx) -> dict:
    """Every candidate, a tiny slice of real chunks, before a single hour is spent."""
    report = REPORTS / "session3_bakeoff_smoke.json"
    _bakeoff(ctx, "--smoke", "--json", str(report), timeout=3_600)
    data = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
    available = data.get("available") or []
    unavailable = data.get("unavailable") or {}
    if not data.get("ok"):
        raise StepFailed("bakeoff_smoke", data.get("blocker") or "no usable candidate survived the smoke test")
    for key, reason in sorted(unavailable.items()):
        ctx.notes.append(f"{key} excluded from the bake-off: {str(reason)[:160]}")
    # The measured throughput is the honest estimate for the rest of the run, so it goes in the summary.
    projected = {
        k: v.get("projected_minutes_per_screen_corpus")
        for k, v in (data.get("candidates") or {}).items() if v.get("ok")
    }  # fmt: skip
    return {"available": available, "unavailable": unavailable, "vram_total_mb": data.get("vram_total_mb"),
            "corpus_chunks": data.get("corpus_chunks"),
            "projected_minutes_per_candidate": projected,
            "projected_total_minutes": round(sum(v for v in projected.values() if v), 1)}  # fmt: skip


def _available_models(ctx: Ctx) -> list[str]:
    """The candidates that passed smoke, from the step state. Embedding a model that cannot load is a guaranteed
    failure twenty minutes into the next step.
    """
    recorded = ctx.state.payload("bakeoff_smoke").get("available")
    if recorded:
        return list(recorded)
    report = REPORTS / "session3_bakeoff_smoke.json"
    if report.is_file():
        return list(json.loads(report.read_text(encoding="utf-8")).get("available") or [])
    return []


def step_bakeoff_headers(ctx: Ctx) -> dict:
    """Stage one: every header recipe on the incumbent model."""
    from retrieval.bakeoff import HEADER_RECIPES, INCUMBENT_KEY

    trials = [f"{INCUMBENT_KEY}__{r.key}" for r in HEADER_RECIPES]
    args = ["--embed"]
    for trial in trials:
        args += ["--trial", trial]
    _bakeoff(ctx, *args)
    out = _bakeoff(ctx, "--screen", "--json", str(REPORTS / "session3_bakeoff_headers.json"), timeout=7_200)
    return {"trials": trials, "screen_tail": _tail(out, 12)}


def step_bakeoff_models(ctx: Ctx) -> dict:
    """Stage two: every model candidate at the header that won stage one."""
    from retrieval.bakeoff import INCUMBENT_KEY, INCUMBENT_RECIPE

    usable = set(_available_models(ctx))
    headers = REPORTS / "session3_bakeoff_headers.json"
    winner = INCUMBENT_RECIPE.key
    if headers.is_file():
        data = json.loads(headers.read_text(encoding="utf-8"))
        scored = [r for r in data.get("results", []) if r.get("recall_at_100") is not None]
        if scored:
            best = max(scored, key=lambda r: r["recall_at_100"])
            incumbent = next((r for r in scored if r["recipe_key"] == INCUMBENT_RECIPE.key), None)
            # Only move off the incumbent recipe if it actually won; a tie stays where production is.
            if incumbent and best["recall_at_100"] > incumbent["recall_at_100"]:
                winner = best["recipe_key"]
    trials = [f"{key}__{winner}" for key in sorted(usable) if key != INCUMBENT_KEY]
    if not trials:
        raise StepFailed("bakeoff_models", "no challenger survived the smoke test, so there is nothing to compare")
    args = ["--embed"]
    for trial in trials:
        args += ["--trial", trial]
    _bakeoff(ctx, *args)
    out = _bakeoff(ctx, "--screen", "--json", str(REPORTS / "session3_bakeoff_models.json"), timeout=7_200)
    return {"header_winner": winner, "trials": trials, "screen_tail": _tail(out, 14)}


def step_bakeoff_decide(ctx: Ctx) -> dict:
    """Apply the adoption rule to the screened trials, mechanically."""
    out = uvrun(ctx, "scripts/bakeoff_decide.py", "--json", str(REPORTS / "session3_bakeoff_decision.json"),
                timeout=1_800)  # fmt: skip
    decision = _last_json(out) or {}
    return {"winner": decision.get("winner"), "adopt": bool(decision.get("adopt")),
            "reason": decision.get("reason", "")[:400]}  # fmt: skip


# MEASURED on this machine on mains, not guessed.
def step_adopt(ctx: Ctx) -> dict:
    """Re-embed the production corpus under the adopted recipe, then rebuild the index and vacuum."""
    before = _recipe_census(ctx)
    result = step_embed(ctx)
    return {**result, "recipes_before": before, "recipes_after": _recipe_census(ctx)}


def _recipe_census(ctx: Ctx) -> dict:
    name = safety.database_of(ctx.dsn)
    out = sh(ctx, ["docker", "exec", "hansard-db", "psql", "-U", "postgres", "-d", name, "-tAc",
                   # The concatenation has to sit OUTSIDE the grouping: `GROUP BY 1` on an expression containing
                   # count(*) is "aggregate functions are not allowed in GROUP BY".
                   "SELECT recipe || '=' || n FROM (SELECT coalesce(embedding_recipe, '(none)') AS recipe,"
                   " count(*) AS n FROM chunk WHERE embedding IS NOT NULL GROUP BY 1) t"
                   " ORDER BY n DESC"], timeout=600)  # fmt: skip
    census = {}
    for line in (out or "").splitlines():
        recipe, _, count = line.strip().rpartition("=")
        if recipe and count.isdigit():
            census[recipe] = int(count)
    return census


def step_final_report(ctx: Ctx) -> dict:
    """Score dev and test with the new index and write reports/session3_final_report.md."""
    out = uvrun(ctx, "scripts/session3_final.py", "--score", "--report", timeout=7_200)
    return {"tail": _tail(out, 12), "report": "reports/session3_final_report.md"}


SESSION3_STEPS: list[Step] = [
    Step("bakeoff_corpus", "Choose and record the screening corpus", 2, step_bakeoff_corpus),
    Step("bakeoff_smoke", "Smoke-test and time every candidate", 12, step_bakeoff_smoke),
    Step("bakeoff_headers", "Stage 1: header recipes on the incumbent", 8, step_bakeoff_headers),
    Step("bakeoff_models", "Stage 2: model candidates at the winning header", 40, step_bakeoff_models),
    Step("bakeoff_decide", "Apply the adoption rule", 2, step_bakeoff_decide),
    # 6.2 min to embed 1,074,097 chunks at 2,877/s, 10.1 min for the HNSW build, then VACUUM ANALYZE.
    Step("adopt", "Re-embed production under the adopted recipe, rebuild HNSW, vacuum", 25, step_adopt),
    Step("final_report", "Score dev and test on the new index; write the final report", 15,
         step_final_report, needs_gpu_service=True),  # fmt: skip
]


def session3_steps(windows: list[str] | None = None) -> list[Step]:
    return list(SESSION3_STEPS)


# W5's fetch is the long pole and it talks to Parliament's API, so it gets its own generous budget.
W5_FETCH_MINUTES = 300
W5 = "W5"


def step_w5_fetch(ctx: Ctx) -> dict:
    """Fetch W5 from the Hansard API into the read-only cache."""
    out = sh(ctx, ["uv", "run", "python", "scripts/fetch_corpus_v2.py", "fetch", "--window", W5],
             timeout=W5_FETCH_MINUTES * 60 + 1800)  # fmt: skip
    return {"tail": _tail(out, 20), "power": "mains" if on_mains() else "battery"}


def step_w5_verify(ctx: Ctx) -> dict:
    """Verify the cache against Parliament's own counts, including division reconciliation."""
    out = sh(ctx, ["uv", "run", "python", "scripts/fetch_corpus_v2.py", "verify", "--window", W5],
             timeout=7200)  # fmt: skip
    return {"tail": _tail(out, 25)}


# `members` is not optional and it belongs here, in the middle.
def w5_ingest_chain() -> list[tuple[str, Callable[[Ctx], dict]]]:
    return [
        ("load", step_load_window(W5)),
        ("members", step_members),
        ("chunk", step_chunk),
        ("rules", step_rules),
        ("derived", step_derived),
    ]


def step_w5_load(ctx: Ctx) -> dict:
    """Load W5, map its members, chunk, apply the rules, and rebuild the derived columns."""
    out: dict = {}
    for name, step in w5_ingest_chain():
        result = step(ctx)
        # Keep the interesting field per step; the full payloads are already in the log.
        out[name] = result.get("derived") or result.get("tail") or result
    return out


def step_w5_embed(ctx: Ctx) -> dict:
    """Embed W5 under the adopted recipe and rebuild the HNSW index over the whole corpus."""
    before = _recipe_census(ctx)
    result = step_embed(ctx)
    after = _recipe_census(ctx)
    return {**result, "recipes_before": before, "recipes_after": after}


def step_stock_sentences(ctx: Ctx) -> dict:
    """Materialise the stock-sentence table arm B1 needs, once, over the whole corpus."""
    out = uvrun(ctx, "scripts/build_stock_sentences.py", "--json", timeout=7200)
    return _last_json(out) or {"tail": _tail(out, 6)}


def step_judge_recheck(ctx: Ctx) -> dict:
    """Re-judge 150 already-judged pairs with today's judge configuration."""
    out = uvrun(ctx, "scripts/judge_recheck.py", "--sample", "150", "--json", timeout=5400)
    payload = _last_json(out) or {}
    return {"kappa": payload.get("kappa"), "agreement": payload.get("agreement"),
            "model": payload.get("model"), "spend_usd": payload.get("spend_usd"),
            "verdict": payload.get("verdict"), "tail": _tail(out, 8)}  # fmt: skip


def step_uplift_screen(ctx: Ctx) -> dict:
    """Stage A: every arm on dev, two tiers."""
    out = uvrun(ctx, "scripts/uplift.py", "run", "--split", "dev", "--stage", "a", "--json", timeout=14_400)
    payload = _last_json(out) or {}
    return {"tier1_dropped": payload.get("tier1_dropped"), "passed": payload.get("passed"),
            "failed": payload.get("failed"), "spend_usd": payload.get("spend_usd"),
            "tail": _tail(out, 20)}  # fmt: skip


def step_uplift_bundle(ctx: Ctx) -> dict:
    """Stage B: combine the passing arms, resolve the mutually exclusive groups, then drop-one."""
    out = uvrun(ctx, "scripts/uplift.py", "bundle", "--split", "dev", "--json", timeout=14_400)
    payload = _last_json(out) or {}
    return {"bundle": payload.get("bundle"), "dropped": payload.get("dropped"), "tail": _tail(out, 20)}


def step_uplift_confirm(ctx: Ctx) -> dict:
    """Stage C: the bundle against the baseline on TEST, once, with --confirm."""
    out = uvrun(ctx, "scripts/uplift.py", "confirm", "--json", timeout=14_400)
    payload = _last_json(out) or {}
    return {"adopt": payload.get("adopt"), "reason": payload.get("reason"),
            "comparisons": payload.get("comparisons"), "tail": _tail(out, 25)}  # fmt: skip


def step_freeze(ctx: Ctx) -> dict:
    """Apply the adoption mechanically, record the config hash, and freeze retrieval."""
    out = uvrun(ctx, "scripts/uplift.py", "freeze", "--json", timeout=14_400)
    payload = _last_json(out) or {}
    return {"config_hash": payload.get("config_hash"), "frozen": payload.get("frozen"),
            "scorecard": payload.get("scorecard"), "tail": _tail(out, 20)}  # fmt: skip


def step_w5_gold(ctx: Ctx) -> dict:
    """Generate W5's gold items, sealed, and the review pack for its critical questions."""
    out = uvrun(ctx, "scripts/build_gold.py", "--window", W5, "--sealed", "--json", timeout=7200)
    payload = _last_json(out) or {}
    return {"items": payload.get("items"), "gold_hash": payload.get("gold_hash"),
            "by_category": payload.get("by_category"), "tail": _tail(out, 12)}  # fmt: skip


def step_e1_report(ctx: Ctx) -> dict:
    """The audit, then the E1 report."""
    audit = uvrun(ctx, "scripts/e1_audit.py", "--live", "--limit", "108", timeout=3600)
    out = uvrun(ctx, "scripts/uplift.py", "report", timeout=1800)
    return {"audit": _tail(audit, 3), "report": "reports/e1_report.md", "tail": _tail(out, 8)}


# The W5 build.
SESSION15_STEPS: list[Step] = [
    Step("w5_fetch", "Fetch W5 (2026-04-01 to 2026-07-31) from the Hansard API", W5_FETCH_MINUTES,
         step_w5_fetch, needs_scripts=("scripts/fetch_corpus_v2.py",)),  # fmt: skip
    Step("w5_verify", "Verify W5's cache against Parliament's own counts", 20, step_w5_verify,
         needs_scripts=("scripts/fetch_corpus_v2.py",)),  # fmt: skip
    Step("w5_load", "Load, chunk, rule and derive W5", 90, step_w5_load,
         needs_scripts=("scripts/load_corpus_v2.py", "scripts/build_derived.py")),  # fmt: skip
    Step("w5_embed", "Embed W5 and rebuild the HNSW index over the whole corpus", 45, step_w5_embed),
    Step("stock_sentences", "Materialise the stock-sentence table (arm B1)", 15, step_stock_sentences,
         needs_scripts=("scripts/build_stock_sentences.py",)),  # fmt: skip
    Step("e1_audit", "Re-run the measurement audit on the W5-augmented corpus", 15,
         lambda ctx: {"audit": _tail(uvrun(ctx, "scripts/e1_audit.py", "--live", "--limit", "108",
                                          timeout=3600), 6)},
         needs_gpu_service=True, needs_scripts=("scripts/e1_audit.py",)),  # fmt: skip
]

# The bake-off half.
SESSION15_BAKEOFF_STEPS: list[Step] = [
    # `build_gold.py` has no --window or --sealed yet: it generates over a whole database and writes the one gold set.
    Step("w5_gold", "Generate W5's gold items, sealed, and its review pack", 30, step_w5_gold,
         needs_scripts=("scripts/build_gold_w5.py",), ready=False,
         blocked="scripts/build_gold_w5.py is not written yet: build_gold.py generates over a whole database "
                 "and writes the one gold set, and W5 needs its own sealed set with its own content hash"),  # fmt: skip
    Step("judge_recheck", "Re-judge 150 pairs with today's judge configuration", 30, step_judge_recheck,
         needs_scripts=("scripts/judge_recheck.py",)),  # fmt: skip
    Step("uplift_screen", "Stage A: screen every arm on dev (condensed first, then judged)", 120,
         step_uplift_screen, needs_gpu_service=True, needs_scripts=("scripts/uplift.py",), ready=False,
         blocked="scripts/uplift.py implements the two-tier screen, but the incremental judging in endgame.md 4.7 step 5 is not wired yet, so an arm below the judged@10 gate reports as unscorable instead of being judged"),  # fmt: skip
    Step("uplift_bundle", "Stage B: the bundle and the drop-one check", 60, step_uplift_bundle,
         needs_gpu_service=True, needs_scripts=("scripts/uplift.py",), ready=False,
         blocked="uplift.py bundle resolves the exclusive groups but does not yet search the bundle-minus-one variants the drop-one check needs"),  # fmt: skip
    Step("uplift_confirm", "Stage C: confirm the bundle on TEST, once", 45, step_uplift_confirm,
         needs_gpu_service=True, needs_scripts=("scripts/uplift.py",), ready=False,
         blocked="uplift.py has no `confirm` subcommand yet (Stage C, the one use of the test split)"),  # fmt: skip
    Step("freeze", "Adopt mechanically, record the config hash, freeze retrieval", 45, step_freeze,
         needs_gpu_service=True, needs_scripts=("scripts/uplift.py",), ready=False,
         blocked="uplift.py has no `freeze` subcommand yet"),  # fmt: skip
    Step("e1_report", "Write reports/e1_report.md", 15, step_e1_report,
         needs_gpu_service=True, needs_scripts=("scripts/uplift.py",), ready=False,
         blocked="uplift.py has no `report` subcommand yet"),  # fmt: skip
]


def session15_steps(windows: list[str] | None = None) -> list[Step]:
    """The W5 build, plus the bake-off steps whose scripts exist."""
    joinable = [s for s in SESSION15_BAKEOFF_STEPS
                if s.ready and all((REPO / name).is_file() for name in s.needs_scripts)]  # fmt: skip
    return list(SESSION15_STEPS) + joinable


# Registered so the session is one command rather than a list in a report.
SESSION16_STEPS: list[Step] = [
    Step("migrate_truth_layer", "Apply migrations 007-011 (cache provenance, graph indexes, coverage)", 5,
         lambda ctx: {"migrate": _tail(uvrun(ctx, "-m", "app.migrate", timeout=1800), 4)}),  # fmt: skip
    Step("coverage", "Build the coverage table (window x house x section) and check it is current", 5,
         lambda ctx: {"coverage": _tail(uvrun(ctx, "scripts/build_coverage.py", timeout=1800), 8)},
         needs_scripts=("scripts/build_coverage.py",)),  # fmt: skip
    Step("facts_dev", "Score the structured-fact engine on dev (must be 100% exact)", 5,
         lambda ctx: {"facts": _tail(uvrun(ctx, "scripts/score_facts.py", timeout=1800), 8)},
         needs_scripts=("scripts/score_facts.py",)),  # fmt: skip
    Step("roster_dev", "Score the roster on dev (100% exact given the gold debate) and report end to end", 5,
         lambda ctx: {"roster": _tail(uvrun(ctx, "scripts/score_roster.py", timeout=1800), 8)},
         needs_scripts=("scripts/score_roster.py",)),  # fmt: skip
    Step("graph_latency", "Measure the relationship tools against the p95 < 100 ms gate", 10,
         lambda ctx: {"latency": _tail(uvrun(ctx, "scripts/graph_latency.py", "--samples", "60",
                                            timeout=3600), 12)},
         needs_scripts=("scripts/graph_latency.py",)),  # fmt: skip
    Step("verifier_precision", "fp32 vs bf16 agreement on 300 claim/evidence pairs", 25,
         lambda ctx: {"precision": _tail(uvrun(ctx, "verify_precision.py", "--pairs", "300",
                                               cwd=REPO / "gpu", timeout=5400), 14)},
         needs_scripts=("gpu/verify_precision.py", "gpu/verifier.py")),  # fmt: skip
]


def session16_steps(windows: list[str] | None = None) -> list[Step]:
    """E2's steps. `windows` is accepted and ignored: E2 adds no corpus, it adds a layer over the one E1 froze."""
    return list(SESSION16_STEPS)


# E3's steps.
SESSION17_STEPS: list[Step] = [
    Step("truth_layer_ready", "Confirm E2's layer is current: migrations 001-012, coverage, frozen retrieval hash", 3,
         lambda ctx: {"migrate": _tail(uvrun(ctx, "-m", "app.migrate", timeout=1800), 3),
                      "coverage": _tail(uvrun(ctx, "scripts/build_coverage.py", "--check", timeout=900), 3)}),
    # Every answer-layer suite, not a subset.
    Step("answer_offline", "Run the answer layer's offline suites (no provider, no GPU)", 5,
         lambda ctx: {"tests": _tail(uvrun(ctx, "-m", "pytest", "tests/test_answer_cache_trace.py",
                                           "tests/test_answer_contract.py", "tests/test_answer_facts.py",
                                           "tests/test_answer_generate.py", "tests/test_answer_injection.py",
                                           "tests/test_answer_resolution_defects.py",
                                           "tests/test_answer_resolve.py", "tests/test_answer_router.py",
                                           "tests/test_answer_routes.py", "tests/test_answer_verify.py",
                                           "-q", timeout=1800), 4)}),
    Step("prompt_bakeoff", "Answer 6 dev questions with all three prompts and build the comparison page", 8,
         lambda ctx: {"bakeoff": _tail(uvrun(ctx, "scripts/prompt_bakeoff.py", "--items", "6",
                                             timeout=3600), 10)},
         needs_gpu_service=True, needs_scripts=("scripts/prompt_bakeoff.py",)),  # fmt: skip
    Step("eval_dev", "Grade every dev answer: deterministic evidence, then the cross-family judge", 30,
         lambda ctx: {"eval": _tail(uvrun(ctx, "scripts/eval_answers.py", timeout=5400), 14)},
         needs_gpu_service=True, needs_scripts=("scripts/eval_answers.py",)),  # fmt: skip
    Step("eval_test", "Confirm once on the test split", 30,
         lambda ctx: {"eval_test": _tail(uvrun(ctx, "scripts/eval_answers.py", "--split", "test",
                                               "--confirm", timeout=5400), 14)},
         needs_gpu_service=True, needs_scripts=("scripts/eval_answers.py",)),  # fmt: skip
]


def session17_steps(windows: list[str] | None = None) -> list[Step]:
    """E3's steps. `windows` is accepted and ignored: E3 adds no corpus."""
    return list(SESSION17_STEPS)


def pending_steps() -> list[Step]:
    """The bake-off steps being held back, for the report and for `--list-pending`."""
    return [s for s in SESSION15_BAKEOFF_STEPS
            if not s.ready or not all((REPO / name).is_file() for name in s.needs_scripts)]  # fmt: skip


def missing_scripts(steps: Sequence[Step]) -> dict[str, list[str]]:
    """step name -> the scripts it needs that do not exist. Empty means the session can run end to end."""
    out: dict[str, list[str]] = {}
    for step in steps:
        absent = [name for name in step.needs_scripts if not (REPO / name).is_file()]
        if absent:
            out[step.name] = absent
    return out


# The step list is built per run, because it depends on which windows are being loaded.
# --------------------------------------------------------------------------- The decade: ten years of Hansard
# --------------------------------------------------------------------------- The plan is `config/corpus_decade.toml`
# (scripts/plan_decade.py): the gaps around W0-W5 cut into calendar quarters, role `corpus`.

# A calendar quarter's fetch: an estimate for the progress line, never a timeout (`fetch_corpus_v2.py plan` measures the
# real latency per window).
DECADE_FETCH_MINUTES = 150


def step_fetch_window(window: str, config: Path):
    """Fetch one window into the cache, then verify it. The fetch resumes where it stopped; `verify` exits non-zero on
    any unit missing, failed or hash-mismatched, so a window counts as fetched only when its cache is whole.
    """
    rel = str(config.relative_to(REPO))

    def run_it(ctx: Ctx) -> dict:
        uvrun(ctx, "scripts/fetch_corpus_v2.py", "--config", rel, "fetch", "--window", window, timeout=86_400)
        out = uvrun(ctx, "scripts/fetch_corpus_v2.py", "--config", rel, "verify", "--window", window, timeout=7200)
        return {"window": window, "verify": _tail(out, 3)}
    return run_it


def step_members_from(config: Path):
    rel = str(config.relative_to(REPO))

    def run_it(ctx: Ctx) -> dict:
        out = uvrun(ctx, "scripts/load_corpus_v2.py", "--config", rel, "--members-only", "--no-sync-filters",
                    timeout=14_400)
        return {"tail": _tail(out, 3)}
    return run_it


def step_relink(ctx: Ctx) -> dict:
    out = uvrun(ctx, "scripts/relink.py", "--database", safety.database_of(ctx.dsn), timeout=14_400)
    return {"tail": _tail(out, 4)}


def step_gate(ctx: Ctx) -> dict:
    """The integrity gate: exits 1 on any hard invariant failing, or any check erroring (an audit that errored has
    checked nothing). Pinned to the session's database, never the default.
    """
    out = uvrun(ctx, "scripts/audit_invariants.py", "--gate", "--database", safety.database_of(ctx.dsn),
                timeout=14_400)
    return {"tail": _tail(out, 2)}


# The decade's embed: ~10.8M new vectors, then ONE HNSW build over ~12.2M (~24 GB at the measured ~1,950 B/vector).
DECADE_EMBED_TIMEOUT_S = 48 * 3600


def step_embed_decade(ctx: Ctx) -> dict:
    out = sh(ctx, ["uv", "run", "python", "embedder.py"], cwd=GPU_DIR, timeout=DECADE_EMBED_TIMEOUT_S,
             env=ctx.env(PGOPTIONS=f"-c maintenance_work_mem={MAINTENANCE_WORK_MEM}"))
    return {"tail": _tail(out, 18), "power": "mains" if on_mains() else "battery"}


def decade_steps(windows: list[str] | None = None) -> list[Step]:
    chosen = windows or _active_windows(DECADE_CONFIG, role="corpus")
    per_window: list[Step] = []
    for w in chosen:
        per_window += [
            Step(f"fetch_{w}", f"Fetch and verify {w}", DECADE_FETCH_MINUTES, step_fetch_window(w, DECADE_CONFIG),
                 needs_scripts=("scripts/fetch_corpus_v2.py",)),
            Step(f"load_{w}", f"Load {w} from the cache", 3, step_load_window(w, DECADE_CONFIG),
                 needs_scripts=("scripts/load_corpus_v2.py",)),
        ]
    gate = ("scripts/audit_invariants.py",)
    return [
        Step("migrate", "Build and migrate hansard_v2", 1, step_migrate),
        *per_window,
        Step("members", "Load every member seen in any window", 20, step_members_from(DECADE_CONFIG)),
        Step("relink", "Relink bills, divisions and corrections across every window", 10, step_relink,
             needs_scripts=("scripts/relink.py",)),
        Step("gate_loaded", "Integrity gate on the loaded corpus", 15, step_gate, needs_scripts=gate),
        Step("chunk", "Chunk every new speech", 60, step_chunk),
        Step("rules", "Procedure rules and boilerplate, one window at a time", 30, step_rules),
        Step("derived", "Filter columns, lexicon, offices, birth dates", 60, step_derived),
        Step("embed", "Embed every new chunk, then rebuild the HNSW index once", 480, step_embed_decade),
        Step("vacuum", "VACUUM ANALYZE", 30, step_vacuum),
        Step("gate_final", "Integrity gate on the finished corpus", 20, step_gate, needs_scripts=gate),
    ]


SESSIONS = {"session1": session1_steps, "session3": session3_steps, "session15": session15_steps,
            "session16": session16_steps, "session17": session17_steps, "decade": decade_steps}  # fmt: skip


# Anything that looks like a path into a home directory.
_ABSOLUTE = re.compile(r"(?i)(?:[A-Z]:[\\/]|/)(?:Users|home)[\\/][^\s\"']+")


def redact(value: Any) -> Any:
    """Replace absolute paths with a placeholder, anywhere in a nested structure."""
    if isinstance(value, str):
        return _ABSOLUTE.sub("<path>", value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def _remaining(steps: list["Step"], after_index: int) -> float:
    """Estimated minutes still to go once the step at `after_index` (1-based) finishes."""
    return sum(s.minutes for s in steps[after_index:])


# "wrote reports/x.json" in a step's captured output, so a summary can list what the run produced.
_WROTE = re.compile(r"wrote\s+(\S+\.(?:json|md|csv|html))")


def _tail(text: str, lines: int) -> list[str]:
    return [line for line in text.strip().splitlines()[-lines:] if line.strip()]


def _last_json(text: str) -> Any:
    """The last JSON document a script printed, or None. Scripts print a table first."""
    for start in range(len(text)):
        if text[start] in "[{":
            try:
                return json.loads(text[start:])
            except ValueError:
                continue
    return None


def _headline(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    variants = data.get("variants") or data.get("results") or {}
    full = variants.get("full") if isinstance(variants, dict) else None
    return {"full": full} if full else {"keys": sorted(variants)[:8] if isinstance(variants, dict) else []}


# --------------------------------------------------------------------------- The run
# ---------------------------------------------------------------------------


def run_session(args) -> int:
    session = args.session
    steps = SESSIONS[session](args.only or None)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    state = State.load(session)
    if args.restart:
        state.reset()
    if args.redo:
        state.reset(args.redo)

    # The target database is a property of the session, not of whatever PG_DSN happens to say.
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database or safety.TARGET)
    ctx = Ctx(session=session, state=state, log=LOG_DIR / f"{session}.log", dsn=dsn,
              windows=args.only or [], dry_run=args.dry_run)
    scope = ",".join(ctx.windows) if ctx.windows else "all"
    ctx.write_log(f"\n\n########## {session} started {datetime.now(UTC).isoformat(timespec='seconds')} ##########\n")

    todo = [s for s in steps if not state.done(s.name, scope)]
    estimate = sum(s.minutes for s in todo)
    console.rule(f"[bold]{session}")
    console.print(f"{len(todo)} step(s) to run of {len(steps)}; "
                  f"estimated [bold]{estimate:.0f} minutes[/bold] "
                  f"({estimate / 60:.1f} hours). Log: {ctx.log.relative_to(REPO)}")
    console.print(f"Database: [bold]{safety.database_of(dsn)}[/bold]   "
                  f"Windows: [bold]{scope}[/bold]"
                  + ("" if ctx.windows else f" (every active window in {session_config(session).relative_to(REPO)})"))
    stale = [s.name for s in steps
             if s.name in state.data.get("steps", {}) and not state.done(s.name, scope)
             and state.data["steps"][s.name].get("status") == "done"]
    if stale:
        console.print(f"[yellow]{len(stale)} step(s) finished under a different scope "
                      f"({', '.join(sorted({state.data['steps'][n].get('scope', '?') for n in stale}))}) "
                      f"and will run again: {', '.join(stale)}[/yellow]")

    checks = preflight(ctx, steps, args.session)
    failed = [c for c in checks if not c.ok and c.fatal]
    if failed:
        console.print(f"[bold red]Preflight failed: {', '.join(c.name for c in failed)}."
                      " Nothing was changed.[/bold red]")
        _print_resume(args)
        return 2
    if args.dry_run:
        plan = Table(title="Steps that would run")
        plan.add_column("#")
        plan.add_column("step")
        plan.add_column("estimate", justify="right")
        for i, step in enumerate(todo, start=1):
            plan.add_row(str(i), step.title, f"{step.minutes:.0f} min")
        console.print(plan)
        console.print("[green]Dry run: preflight passed and nothing was changed.[/green]")
        return 0

    # Remember the machine setting in the STATE FILE before changing it, not only in a local variable.
    original_standby = state.data.get("machine", {}).get("standby_ac_minutes")
    current = standby_timeout_minutes()
    if original_standby is not None and current == 0:
        set_standby_timeout(original_standby)
        console.print(f"[yellow]A previous run left the AC sleep timeout at never; "
                      f"restored to {original_standby} min.[/yellow]")
        current = original_standby
    if current is not None and current != 0:
        original_standby = current
        state.data["machine"] = {"standby_ac_minutes": current}
        state.save()
        set_standby_timeout(0)
        console.print(f"AC sleep timeout was {current} min; set to never for this run.")

    failure: StepFailed | None = None
    try:
        for index, step in enumerate(steps, start=1):
            if state.done(step.name, scope):
                console.print(f"[dim]{index}/{len(steps)} {step.title}: already done[/dim]")
                continue
            if step.needs_gpu_service:
                start_gpu(ctx)
            console.print(f"[bold]{index}/{len(steps)}[/bold] {step.title} "
                          f"[dim](about {step.minutes:.0f} min; "
                          f"{_remaining(steps, index):.0f} min left after it)[/dim] ...")
            started = time.monotonic()
            try:
                payload = step.run(ctx)
            except StepFailed as exc:
                state.record(step.name, "failed", time.monotonic() - started,
                             {"error": exc.detail[:2000]}, scope=scope)
                failure = exc
                break
            seconds = time.monotonic() - started
            state.record(step.name, "done", seconds, payload, scope=scope)
            console.print(f"    [green]done[/green] in {seconds / 60:.1f} min")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Every finished step is recorded; "
                      "re-run the same command to continue.[/yellow]")
        failure = StepFailed("interrupted", "Ctrl+C")
    finally:
        kill_tree(ctx.child)
        stop_gpu(ctx)
        if original_standby is not None and original_standby != 0:
            restored = set_standby_timeout(original_standby)
            console.print(f"AC sleep timeout restored to {original_standby} min"
                          f"{'' if restored else ' [red](FAILED: set it by hand)[/red]'}")
            if restored:
                state.data.pop("machine", None)
                state.save()

    write_summary(ctx, steps, failure)
    if failure:
        console.print(f"\n[bold red]{failure.step} failed.[/bold red]\n{failure.detail}")
        console.print(f"\nFull log: {ctx.log.relative_to(REPO)}")
        _print_resume(args)
        return 1
    console.print(f"\n[bold green]{session} finished.[/bold green] "
                  f"Total {(time.monotonic() - ctx.started) / 60:.0f} minutes.")
    console.print(f"Read: reports/{session}_summary.md")
    return 0


def _print_resume(args) -> None:
    parts = ["uv run python scripts/pipeline.py run", args.session]
    for window in args.only or []:
        parts += ["--only", window]
    if args.database:
        parts += ["--database", args.database]
    console.print("\n[bold]Resume with exactly this:[/bold]\n  " + " ".join(parts))


# --------------------------------------------------------------------------- The summary the next session reads first
# ---------------------------------------------------------------------------


def write_summary(ctx: Ctx, steps: list[Step], failure: StepFailed | None) -> None:
    data = {
        "session": ctx.session,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_minutes": round((time.monotonic() - ctx.started) / 60, 1),
        "database": safety.database_of(ctx.dsn),
        "windows": ctx.windows or f"all active windows in {session_config(ctx.session).relative_to(REPO)}",
        "power": "mains" if on_mains() else "battery",
        "outcome": "failed" if failure else "ok",
        "failed_step": failure.step if failure else None,
        "notes": ctx.notes,
        "steps": ctx.state.data.get("steps", {}),
    }  # fmt: skip
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{ctx.session}_summary.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    (REPORTS / f"{ctx.session}_summary.md").write_text(_markdown(data, steps), encoding="utf-8")


def _markdown(data: dict, steps: list[Step]) -> str:
    lines = [f"# {data['session']} summary", "",
             f"- Finished: {data['finished_at']}",
             f"- Outcome: **{data['outcome']}**" + (f" at step `{data['failed_step']}`" if data["failed_step"] else ""),
             f"- Database: `{data['database']}`   Windows: {data['windows']}",
             f"- Elapsed: {data['elapsed_minutes']} minutes, on {data['power']}", ""]
    if data["notes"]:
        lines += ["## Notes", *[f"- {n}" for n in data["notes"]], ""]
    lines += ["## Steps", "", "| step | status | minutes |", "|---|---|---|"]
    for step in steps:
        record = data["steps"].get(step.name, {})
        lines.append(f"| {step.title} | {record.get('status', 'not run')} | "
                     f"{record.get('seconds', 0) / 60:.1f} |")
    # Derived from this run, not hardcoded.
    written: list[str] = []
    for record in data["steps"].values():
        for chunk in (record.get("payload") or {}).values():
            for line in (chunk if isinstance(chunk, list) else [chunk]):
                match = _WROTE.search(str(line))
                if match:
                    path = match.group(1).replace("\\", "/")
                    if path not in written:
                        written.append(path)
    lines += ["", "## What the next session should read", "",
              f"1. `reports/{data['session']}_summary.json` (this file's data, with every step's payload)"]
    lines += ([f"{i}. `{path}`" for i, path in enumerate(written, start=2)]
              or ["2. (no step reported writing a report)"])
    lines += [""]
    for name, record in data["steps"].items():
        payload = record.get("payload")
        if payload:
            lines += [f"### {name}", "", "```json",
                      json.dumps(payload, indent=2, default=str)[:4000], "```", ""]
    return "\n".join(lines)


def show_status(args) -> int:
    state = State.load(args.session)
    steps = SESSIONS[args.session](None)
    table = Table(title=f"{args.session}")
    table.add_column("step")
    table.add_column("status")
    table.add_column("windows")
    table.add_column("minutes", justify="right")
    table.add_column("finished")
    for step in steps:
        record = state.data.get("steps", {}).get(step.name, {})
        table.add_row(step.title, record.get("status", "[dim]not run[/dim]"),
                      record.get("scope", "") or "[dim]-[/dim]",
                      f"{record.get('seconds', 0) / 60:.1f}", record.get("finished_at", ""))
    console.print(table)
    console.print("[dim]A step counts as done only for the windows it ran over: a step "
                  "finished under `--only W0` runs again on a full run.[/dim]")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a session end to end, resuming where it stopped")
    run_p.add_argument("session", choices=sorted(SESSIONS))
    run_p.add_argument("--only", action="append", help="restrict the load to this window (repeatable)")
    run_p.add_argument("--dry-run", action="store_true", help="preflight and print the plan; change nothing")
    run_p.add_argument("--restart", action="store_true", help="forget every finished step first")
    run_p.add_argument("--redo", action="append", help="forget one step so it runs again (repeatable)")
    run_p.add_argument("--database", help="override the database name in PG_DSN")

    status_p = sub.add_parser("status", help="what has finished so far")
    status_p.add_argument("session", choices=sorted(SESSIONS))

    args = parser.parse_args()
    if args.command == "status":
        sys.exit(show_status(args))
    sys.exit(run_session(args))


if __name__ == "__main__":
    main()
