"""Corpus v2: download every API response a window needs, into a local cache."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import sqlite3
import time
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from ingest.http import PoliteClient

log = logging.getLogger("corpus_fetch")

# --------------------------------------------------------------------------- Endpoints
# ---------------------------------------------------------------------------

HANSARD = "https://hansard-api.parliament.uk"
SPOKEN_URL = f"{HANSARD}/search/contributions/Spoken.json"
CORRECTIONS_URL = f"{HANSARD}/search/contributions/Corrections.json"
CALENDAR_URL = f"{HANSARD}/overview/calendar.json"
SECTIONS_URL = f"{HANSARD}/overview/sectionsforday.json"
HANSARD_DIVISIONS_URL = f"{HANSARD}/search/divisions.json"
COMMONS_SEARCH_URL = "https://commonsvotes-api.parliament.uk/data/divisions.json/search"
COMMONS_DETAIL_URL = "https://commonsvotes-api.parliament.uk/data/division/{id}.json"
LORDS_SEARCH_URL = "https://lordsvotes-api.parliament.uk/data/Divisions/search"
BILLS_BASE = "https://bills-api.parliament.uk/api/v1"
MEMBERS_BASE = "https://members-api.parliament.uk/api/Members"

HOUSES = ("Commons", "Lords")
ORDERS = ("SittingDateAsc", "SittingDateDesc")

# Sources in the order they must run: each one may need the previous one's cache.
SOURCE_ORDER = (
    "calendar", "probe", "sections", "contributions", "corrections",
    "hansard_divisions", "commons_divisions", "lords_divisions", "bills", "members",
)  # fmt: skip

# Units that describe an object, not a window: fetched once, reused by every window.
SHARED = "_shared"

# Statuses.
DONE, PENDING, FAILED, NOT_SITTING, KNOWN_GAP = "done", "pending", "failed", "not_sitting", "known_gap"

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_FAILURES, EXIT_BREAKER, EXIT_DISK, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 5, 130


class Interrupted(Exception):
    """Ctrl+C was pressed. Unwinds to the orchestrator, which saves state and exits 130."""


class ConfigError(Exception):
    """The window config is not usable. Never guessed around."""


# --------------------------------------------------------------------------- 1.


# Every role a window may have, mirrored by `corpus_window`'s CHECK.
ROLES = frozenset({"dev", "dev_test", "heldout", "candidate", "corpus"})


@dataclass(frozen=True)
class WindowSpec:
    id: str
    start: date
    end: date
    role: str
    active: bool
    note: str = ""

    @property
    def days(self) -> list[date]:
        """Every calendar day in the window, oldest first, inclusive at both ends."""
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]

    @property
    def months(self) -> list[tuple[int, int]]:
        out, y, m = [], self.start.year, self.start.month
        while (y, m) <= (self.end.year, self.end.month):
            out.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out

    def overlaps(self, other: WindowSpec) -> bool:
        return self.start <= other.end and other.start <= self.end


@dataclass(frozen=True)
class HttpPolicy:
    user_agent: str
    concurrency: int
    max_concurrency: int
    min_interval: float
    connect_timeout: float
    read_timeout: float
    write_timeout: float
    pool_timeout: float
    max_attempts: int
    backoff_base: float
    backoff_cap: float
    retry_after_cap: float
    page_size: int
    slowdown_factor: float
    slowdown_decay: float


@dataclass(frozen=True)
class BreakerPolicy:
    consecutive_failures: int
    pause_seconds: float
    pause_cap: float
    max_pauses: int


@dataclass(frozen=True)
class DiskPolicy:
    min_free_gb: float
    check_every_units: int


@dataclass(frozen=True)
class EstimatePolicy:
    chunks_per_contribution: float
    served_fraction: float
    bytes_per_contribution: int


@dataclass(frozen=True)
class FetchConfig:
    version: str
    cache_root: Path
    log_dir: Path
    http: HttpPolicy
    breaker: BreakerPolicy
    disk: DiskPolicy
    estimate: EstimatePolicy
    sources: tuple[str, ...]
    member_endpoints: tuple[str, ...]
    known_ids_file: Path | None
    fetch_known_members: bool
    windows: tuple[WindowSpec, ...]
    source_path: Path | None = None

    def window(self, window_id: str) -> WindowSpec:
        for w in self.windows:
            if w.id.casefold() == window_id.casefold():
                return w
        known = ", ".join(w.id for w in self.windows)
        raise ConfigError(f"unknown window {window_id!r}; config has {known}")

    @property
    def active_windows(self) -> tuple[WindowSpec, ...]:
        return tuple(w for w in self.windows if w.active)

    @property
    def manifest_path(self) -> Path:
        return self.cache_root / "manifest.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.cache_root / "reports"


def _require(table: dict, key: str, where: str) -> Any:
    if key not in table:
        raise ConfigError(f"[{where}] is missing {key!r}")
    return table[key]


def load_config(path: str | Path, *, repo_root: Path | None = None, today: date | None = None) -> FetchConfig:
    """Read and VALIDATE the TOML. Every failure here is cheaper than a failure four hours in."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no config at {path}")
    root = repo_root or path.resolve().parent.parent
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    # A plan file (the ten-year extension) EXTENDS the base config and may only add windows.
    if "extends" in raw:
        extra = sorted(set(raw) - {"extends", "windows"})
        if extra:
            raise ConfigError(f"{path.name} extends another config and may only add [[windows]]; it also sets {extra}")
        base_path = (path.parent / str(raw["extends"])).resolve()
        if not base_path.is_file():
            raise ConfigError(f"{path.name} extends {raw['extends']!r}, which does not exist")
        with base_path.open("rb") as fh:
            base = tomllib.load(fh)
        if "extends" in base:
            raise ConfigError("a config may extend one base, not a chain of them")
        raw = {**base, "windows": [*base.get("windows", []), *raw.get("windows", [])]}

    corpus = raw.get("corpus", {})
    http_raw = raw.get("http", {})
    user_agent = str(_require(http_raw, "user_agent", "http"))
    # gotcha 14: the maintainer's address must never leave this machine, not even in a header.
    if "@" in user_agent:
        raise ConfigError("[http] user_agent must not contain '@' (no e-mail address in a User-Agent)")
    if not user_agent.strip():
        raise ConfigError("[http] user_agent is empty")

    max_concurrency = int(http_raw.get("max_concurrency", 4))
    concurrency = int(http_raw.get("concurrency", 2))
    if concurrency < 1:
        raise ConfigError("[http] concurrency must be at least 1")
    if concurrency > max_concurrency:
        raise ConfigError(f"[http] concurrency {concurrency} exceeds max_concurrency {max_concurrency}")
    if max_concurrency > 4:
        raise ConfigError("[http] max_concurrency is capped at 4: this is someone else's public API")
    page_size = int(http_raw.get("page_size", 100))
    if not 1 <= page_size <= 100:
        raise ConfigError(f"[http] page_size {page_size} is outside 1..100 (the API returns 500 above 100)")

    http = HttpPolicy(
        user_agent=user_agent, concurrency=concurrency, max_concurrency=max_concurrency,
        min_interval=float(http_raw.get("min_interval", 0.12)),
        connect_timeout=float(http_raw.get("connect_timeout", 10.0)),
        read_timeout=float(http_raw.get("read_timeout", 60.0)),
        write_timeout=float(http_raw.get("write_timeout", 30.0)),
        pool_timeout=float(http_raw.get("pool_timeout", 20.0)),
        max_attempts=int(http_raw.get("max_attempts", 6)),
        backoff_base=float(http_raw.get("backoff_base", 1.0)),
        backoff_cap=float(http_raw.get("backoff_cap", 60.0)),
        retry_after_cap=float(http_raw.get("retry_after_cap", 120.0)),
        page_size=page_size,
        slowdown_factor=float(http_raw.get("slowdown_factor", 2.0)),
        slowdown_decay=float(http_raw.get("slowdown_decay", 120.0)),
    )  # fmt: skip
    if http.max_attempts < 1:
        raise ConfigError("[http] max_attempts must be at least 1")

    br = raw.get("breaker", {})
    breaker = BreakerPolicy(
        consecutive_failures=int(br.get("consecutive_failures", 10)),
        pause_seconds=float(br.get("pause_seconds", 300.0)),
        pause_cap=float(br.get("pause_cap", 1800.0)),
        max_pauses=int(br.get("max_pauses", 4)),
    )  # fmt: skip
    if breaker.consecutive_failures < 1:
        raise ConfigError("[breaker] consecutive_failures must be at least 1")

    dk = raw.get("disk", {})
    disk = DiskPolicy(min_free_gb=float(dk.get("min_free_gb", 2.0)),
                      check_every_units=int(dk.get("check_every_units", 25)))
    es = raw.get("estimate", {})
    estimate = EstimatePolicy(
        chunks_per_contribution=float(es.get("chunks_per_contribution", 10.07)),
        served_fraction=float(es.get("served_fraction", 0.9961)),
        bytes_per_contribution=int(es.get("bytes_per_contribution", 2600)),
    )  # fmt: skip

    sources = tuple(raw.get("sources", {}).get("enabled", SOURCE_ORDER))
    unknown = [s for s in sources if s not in SOURCE_ORDER]
    if unknown:
        raise ConfigError(f"[sources] unknown source(s) {unknown}; known: {list(SOURCE_ORDER)}")

    members_raw = raw.get("members", {})
    endpoints = tuple(members_raw.get("endpoints", ("",)))
    known_ids = members_raw.get("known_ids_file")
    known_path = (root / known_ids) if known_ids else None
    # `known_ids_file` was an optimisation: don't re-fetch a member the `hansard` database already holds.
    fetch_known = bool(members_raw.get("fetch_known", True))

    today = today or datetime.now(UTC).date()
    windows: list[WindowSpec] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw.get("windows", [])):
        wid = str(_require(item, "id", f"windows[{i}]"))
        if wid in seen_ids:
            raise ConfigError(f"duplicate window id {wid!r}")
        seen_ids.add(wid)
        start, end = _as_date(item, "start", wid), _as_date(item, "end", wid)
        if start > end:
            raise ConfigError(f"window {wid}: start {start} is after end {end}")
        if end > today:
            raise ConfigError(f"window {wid}: end {end} is in the future (today is {today})")
        role = str(item.get("role", "dev_test"))
        if role not in ROLES:
            raise ConfigError(f"window {wid}: role {role!r} is not one of {sorted(ROLES)}")
        windows.append(WindowSpec(
            id=wid, start=start, end=end, role=role,
            active=bool(item.get("active", True)), note=str(item.get("note", "")),
        ))  # fmt: skip
    if not windows:
        raise ConfigError("the config defines no windows")

    # Overlapping ACTIVE windows would double-fetch days and confound every per-window comparison.
    active = [w for w in windows if w.active]
    for a, b in ((a, b) for i, a in enumerate(active) for b in active[i + 1:]):
        if a.overlaps(b):
            raise ConfigError(f"active windows {a.id} ({a.start}..{a.end}) and {b.id} ({b.start}..{b.end}) overlap")

    return FetchConfig(
        version=str(corpus.get("version", "v2")),
        cache_root=root / corpus.get("cache_root", "data/raw/corpus_v2"),
        log_dir=root / corpus.get("log_dir", "data/logs"),
        http=http, breaker=breaker, disk=disk, estimate=estimate,
        sources=sources, member_endpoints=endpoints, known_ids_file=known_path,
        fetch_known_members=fetch_known, windows=tuple(windows), source_path=path,
    )  # fmt: skip


def _as_date(item: dict, key: str, wid: str) -> date:
    value = _require(item, key, f"windows.{wid}")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"window {wid}: {key} {value!r} is not a date") from exc


def load_known_member_ids(path: Path | None) -> set[int]:
    """MNIS ids we already hold, one per line. Missing file means 'we hold none'."""
    if not path or not path.is_file():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                ids.add(int(line))
            except ValueError:
                continue
    return ids


# --------------------------------------------------------------------------- 2.


class Written(NamedTuple):
    sha256: str  # of the RAW bytes, so it is independent of compression
    raw_bytes: int
    disk_bytes: int


class Cache:
    """Append-only gzip cache. Every write is atomic: a crash leaves the old file or none."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def full(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self.full(rel).is_file()

    def write(self, rel: str, raw: bytes) -> Written:
        target = self.full(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{random.getrandbits(32):08x}.part")
        try:
            with open(tmp, "wb") as fh:
                # filename="" and mtime=0: without both, gzip stamps the temp file's random name and the clock into the
                # header, so the same response would produce different bytes every time and `verify` could not compare
                # them.
                with gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:
                    gz.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)  # never leave a half-written file that looks complete
            raise
        return Written(hashlib.sha256(raw).hexdigest(), len(raw), target.stat().st_size)

    def read(self, rel: str) -> bytes:
        with gzip.open(self.full(rel), "rb") as fh:
            return fh.read()

    def read_json(self, rel: str) -> Any:
        return json.loads(self.read(rel).decode("utf-8"))

    def sha256(self, rel: str) -> str | None:
        """None when the file is missing or unreadable (truncated, not gzip,...)."""
        try:
            return hashlib.sha256(self.read(rel)).hexdigest()
        except (OSError, EOFError, gzip.BadGzipFile, zlib_error()):
            return None

    def sweep_partials(self) -> int:
        """Remove `.part` files left by a kill -9. Never touches a completed file."""
        n = 0
        if not self.root.is_dir():
            return 0
        for p in self.root.rglob("*.part"):
            p.unlink(missing_ok=True)
            n += 1
        return n

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*.json.gz"))


def zlib_error() -> type[BaseException]:
    import zlib

    return zlib.error


# Cache paths.


def contributions_path(window_id: str, day: date, order: str, skip: int) -> str:
    short = "asc" if order.endswith("Asc") else "desc"
    return f"contributions/{window_id}/{day.isoformat()}/{short}_{skip:06d}.json.gz"


def probe_path(window_id: str, day: date) -> str:
    return f"probe/{window_id}/{day.isoformat()}.json.gz"


def corrections_path(window_id: str, day: date) -> str:
    return f"corrections/{window_id}/{day.isoformat()}.json.gz"


def calendar_path(window_id: str, house: str, year: int, month: int) -> str:
    return f"calendar/{window_id}/{house}-{year:04d}-{month:02d}.json.gz"


def sections_path(window_id: str, house: str, day: date) -> str:
    return f"sections/{window_id}/{house}/{day.isoformat()}.json.gz"


def hansard_divisions_path(window_id: str, house: str, skip: int) -> str:
    # The house is a DIRECTORY, not a filename prefix: the unit records hansard_divisions/<window>/<house> as its path,
    # and `verify` checks that path exists.
    return f"hansard_divisions/{window_id}/{house}/index_{skip:06d}.json.gz"


def commons_index_path(window_id: str, skip: int) -> str:
    return f"commons_divisions/{window_id}/index_{skip:06d}.json.gz"


def commons_detail_path(division_id: int) -> str:
    return f"commons_divisions/detail/{division_id}.json.gz"


def lords_index_path(window_id: str, skip: int) -> str:
    return f"lords_divisions/{window_id}/index_{skip:06d}.json.gz"


def bill_sittings_path(window_id: str, skip: int) -> str:
    return f"bills/{window_id}/sittings_{skip:06d}.json.gz"


def bill_path(bill_id: int, what: str) -> str:
    return f"bills/bill/{bill_id}/{what}.json.gz"


def member_path(mnis_id: int, endpoint: str) -> str:
    return f"members/{mnis_id}/{endpoint or 'profile'}.json.gz"


# --------------------------------------------------------------------------- 3.

SCHEMA = """
CREATE TABLE IF NOT EXISTS unit (
    source      TEXT NOT NULL,
    window_id   TEXT NOT NULL,
    key         TEXT NOT NULL,
    url         TEXT,
    params      TEXT,
    path        TEXT,
    status      TEXT NOT NULL,
    http_status INTEGER,
    attempts    INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT,
    bytes       INTEGER NOT NULL DEFAULT 0,
    items       INTEGER,
    fingerprint TEXT,
    fetched_at  TEXT,
    last_error  TEXT,
    PRIMARY KEY (source, window_id, key)
);
CREATE INDEX IF NOT EXISTS unit_status_idx ON unit(status);
CREATE INDEX IF NOT EXISTS unit_window_idx ON unit(window_id, source);

CREATE TABLE IF NOT EXISTS day (
    window_id        TEXT NOT NULL,
    date             TEXT NOT NULL,
    house            TEXT NOT NULL,
    sitting          INTEGER,
    calendar_sitting INTEGER,
    listed           INTEGER,
    debates          INTEGER,
    asc_items        INTEGER,
    desc_items       INTEGER,
    merged           INTEGER,
    sections         TEXT,
    anomalies        TEXT,
    PRIMARY KEY (window_id, date, house)
);

CREATE TABLE IF NOT EXISTS run (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id  TEXT,
    command    TEXT,
    started_at TEXT,
    ended_at   TEXT,
    requests   INTEGER NOT NULL DEFAULT 0,
    retries    INTEGER NOT NULL DEFAULT 0,
    failures   INTEGER NOT NULL DEFAULT 0,
    bytes      INTEGER NOT NULL DEFAULT 0,
    outcome    TEXT
);
"""

_UNIT_COLUMNS = ("source", "window_id", "key", "url", "params", "path", "status", "http_status",
                 "attempts", "sha256", "bytes", "items", "fingerprint", "fetched_at", "last_error")
UPSERT_UNIT = f"""
INSERT INTO unit ({", ".join(_UNIT_COLUMNS)})
VALUES ({", ".join("?" * len(_UNIT_COLUMNS))})
ON CONFLICT(source, window_id, key) DO UPDATE SET
  url=excluded.url, params=excluded.params, path=excluded.path, status=excluded.status,
  http_status=excluded.http_status, attempts=unit.attempts + excluded.attempts,
  sha256=excluded.sha256, bytes=excluded.bytes, items=excluded.items,
  fingerprint=excluded.fingerprint, fetched_at=excluded.fetched_at, last_error=excluded.last_error
"""
_DAY_COLUMNS = ("window_id", "date", "house", "sitting", "calendar_sitting", "listed", "debates",
                "asc_items", "desc_items", "merged", "sections", "anomalies")
UPSERT_DAY = f"""
INSERT INTO day ({", ".join(_DAY_COLUMNS)}) VALUES ({", ".join("?" * len(_DAY_COLUMNS))})
ON CONFLICT(window_id, date, house) DO UPDATE SET
  {", ".join(f"{c}=excluded.{c}" for c in _DAY_COLUMNS[3:])}
"""


@dataclass
class DayRow:
    window_id: str
    date: date
    house: str
    sitting: bool | None = None
    calendar_sitting: bool | None = None
    listed: int | None = None
    debates: int | None = None
    asc_items: int | None = None
    desc_items: int | None = None
    merged: int | None = None
    sections: dict[str, int] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)

    def as_params(self) -> tuple:
        return (self.window_id, self.date.isoformat(), self.house,
                _bit(self.sitting), _bit(self.calendar_sitting), self.listed, self.debates,
                self.asc_items, self.desc_items, self.merged,
                json.dumps(self.sections, sort_keys=True), json.dumps(self.anomalies))  # fmt: skip


def _bit(value: bool | None) -> int | None:
    return None if value is None else int(value)


class Manifest:
    """One SQLite file, WAL mode, opened by ONE writer. Readers open it separately."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        if not read_only:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    # -- reads -------------------------------------------------------------

    def unit(self, source: str, window_id: str, key: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM unit WHERE source=? AND window_id=? AND key=?", (source, window_id, key)
        ).fetchone()

    def units(self, window_id: str | None = None, source: str | None = None,
              status: str | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM unit WHERE 1=1", []
        if window_id:
            sql += " AND window_id IN (?, ?)"
            args += [window_id, SHARED]
        if source:
            sql += " AND source=?"
            args.append(source)
        if status:
            sql += " AND status=?"
            args.append(status)
        return self.conn.execute(sql + " ORDER BY source, key", args).fetchall()

    def status_counts(self, window_id: str | None = None) -> dict[tuple[str, str], int]:
        sql = "SELECT source, status, count(*) n FROM unit"
        args: list[Any] = []
        if window_id:
            sql += " WHERE window_id IN (?, ?)"
            args += [window_id, SHARED]
        sql += " GROUP BY source, status"
        return {(r["source"], r["status"]): r["n"] for r in self.conn.execute(sql, args)}

    def days(self, window_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM day WHERE window_id=? ORDER BY date, house", (window_id,)
        ).fetchall()

    def totals(self, window_id: str | None = None) -> dict[str, int]:
        sql = "SELECT count(*) units, coalesce(sum(bytes),0) bytes, coalesce(sum(items),0) items FROM unit"
        args: list[Any] = []
        if window_id:
            sql += " WHERE window_id IN (?, ?)"
            args += [window_id, SHARED]
        row = self.conn.execute(sql, args).fetchone()
        return {"units": row["units"], "bytes": row["bytes"], "items": row["items"]}

    # -- writes (only from ManifestWriter, or from a test) ------------------

    def apply(self, ops: Sequence[tuple[str, tuple]]) -> None:
        for sql, params in ops:
            self.conn.execute(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def start_run(self, window_id: str | None, command: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO run (window_id, command, started_at) VALUES (?, ?, ?)",
            (window_id, command, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def finish_run(self, run_id: int, *, requests: int, retries: int, failures: int,
                   nbytes: int, outcome: str) -> None:
        self.conn.execute(
            "UPDATE run SET ended_at=?, requests=?, retries=?, failures=?, bytes=?, outcome=? WHERE id=?",
            (_now(), requests, retries, failures, nbytes, outcome, run_id),
        )
        self.conn.commit()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class _Flush(NamedTuple):
    event: asyncio.Event


class ManifestWriter:
    """The single writer. Every manifest change is queued here and applied by one task."""

    def __init__(self, manifest: Manifest, *, batch: int = 64, interval: float = 2.0) -> None:
        self.manifest = manifest
        self.batch = batch
        self.interval = interval
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._pending: list[tuple[str, tuple]] = []
        self._last_commit = time.monotonic()
        self.applied = 0

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="manifest-writer")

    def record_unit(self, source: str, window_id: str, key: str, result: UnitResult) -> None:
        self.queue.put_nowait((UPSERT_UNIT, (
            source, window_id, key, result.url, result.params, result.path, result.status,
            result.http_status, result.attempts, result.sha256, result.bytes, result.items,
            result.fingerprint, _now(), result.error,
        )))  # fmt: skip

    def record_day(self, row: DayRow) -> None:
        self.queue.put_nowait((UPSERT_DAY, row.as_params()))

    async def flush(self) -> None:
        """Return once everything queued so far is committed."""
        if self._task is None or self._task.done():
            self._drain_sync()
            return
        event = asyncio.Event()
        self.queue.put_nowait(_Flush(event))
        await event.wait()

    async def close(self) -> None:
        await self.flush()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._commit()

    def _drain_sync(self) -> None:
        """Last-resort drain when the writer task is gone (cancelled, or crashed)."""
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if isinstance(item, _Flush):
                item.event.set()
            else:
                self._pending.append(item)
        self._commit()

    def _commit(self) -> None:
        if self._pending:
            self.manifest.apply(self._pending)
            self.applied += len(self._pending)
            self._pending.clear()
        self.manifest.commit()
        self._last_commit = time.monotonic()

    async def _run(self) -> None:
        while True:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=self.interval)
            except TimeoutError:
                if self._pending:
                    self._commit()
                continue
            except asyncio.CancelledError:
                self._drain_sync()
                raise
            if isinstance(item, _Flush):
                self._commit()
                item.event.set()
                continue
            self._pending.append(item)
            if len(self._pending) >= self.batch or time.monotonic() - self._last_commit >= self.interval:
                self._commit()


# --------------------------------------------------------------------------- 4.


class StopFlag:
    """First Ctrl+C asks for a clean stop; a second one is taken as 'I meant it'."""

    def __init__(self) -> None:
        self.requested = False
        self.presses = 0
        self._previous: Any = None
        self._installed = False

    def install(self) -> None:
        try:
            self._previous = signal.signal(signal.SIGINT, self._handle)
            self._installed = True
        except ValueError:
            self._installed = False  # not the main thread (a test, or an embedded run)

    def restore(self) -> None:
        if self._installed and self._previous is not None:
            try:
                signal.signal(signal.SIGINT, self._previous)
            except ValueError:
                pass
            self._installed = False

    def request(self) -> None:
        self.requested = True
        self.presses += 1

    def _handle(self, signum, frame) -> None:  # pragma: no cover - exercised by hand, not by CI
        self.request()
        if self.presses == 1:
            print("\n[stopping] finishing the unit in flight and saving state. "
                  "Press Ctrl+C again to abandon it.", flush=True)
        else:
            self.restore()
            raise KeyboardInterrupt

    def check(self) -> None:
        if self.requested:
            raise Interrupted()


class Breaker:
    """Consecutive unit failures mean the network or the API is down, not that one day is bad."""

    def __init__(self, policy: BreakerPolicy) -> None:
        self.policy = policy
        self.consecutive = 0
        self.pauses = 0
        self.next_pause = policy.pause_seconds

    def record(self, ok: bool) -> None:
        self.consecutive = 0 if ok else self.consecutive + 1

    @property
    def tripped(self) -> bool:
        return self.consecutive >= self.policy.consecutive_failures

    @property
    def exhausted(self) -> bool:
        return self.pauses >= self.policy.max_pauses

    def take_pause(self) -> float:
        """Consume one pause and return how long to sleep. Resets the failure run."""
        seconds = min(self.next_pause, self.policy.pause_cap)
        self.pauses += 1
        self.next_pause = min(self.next_pause * 2, self.policy.pause_cap)
        self.consecutive = 0
        return seconds


class DiskGuard:
    """Stop cleanly while there is still room to write the manifest, not after."""

    def __init__(self, path: Path, policy: DiskPolicy) -> None:
        self.path = Path(path)
        self.policy = policy
        self._counter = 0

    def free_gb(self) -> float:
        probe = self.path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        return shutil.disk_usage(probe).free / 1e9

    def low(self, *, force: bool = False) -> float | None:
        self._counter += 1
        if not force and self.policy.check_every_units > 0 and self._counter % self.policy.check_every_units:
            return None
        free = self.free_gb()
        return free if free < self.policy.min_free_gb else None


# --------------------------------------------------------------------------- 5.

# Worth another go.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# httpx raises these when the connection drops, stalls or is reset (gotcha 21).
RETRYABLE_EXC = (httpx.TransportError,)


class RawResult(NamedTuple):
    ok: bool
    status: int | None
    content: bytes | None
    attempts: int
    retries: int
    error: str | None

    @property
    def is_404(self) -> bool:
        return self.status == 404


class RawClient(PoliteClient):
    """PoliteClient's gate and semaphore, but returning the bytes the API actually sent."""

    def __init__(self, http: httpx.AsyncClient, policy: HttpPolicy, stop: StopFlag) -> None:
        super().__init__(http, concurrency=policy.concurrency, min_interval=policy.min_interval,
                         max_attempts=policy.max_attempts, backoff_base=policy.backoff_base)
        self.policy = policy
        self.stop = stop
        self.requests = 0
        self.retries = 0
        self.rate_limited = 0
        self.latencies: list[float] = []  # for the plan's honest time interval, not a mean
        self._slow_until = 0.0
        self._extra_interval = 0.0

    def latency_quantiles(self) -> tuple[float, float]:
        """(p50, p95) seconds per request. (0, 0) before anything has been measured."""
        if not self.latencies:
            return (0.0, 0.0)
        ordered = sorted(self.latencies)
        return (ordered[len(ordered) // 2], ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))])

    async def get_raw(self, url: str, params: dict[str, Any] | None = None, *,
                      allow_404: bool = False) -> RawResult:
        last_status: int | None = None
        last_error: str | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            self.stop.check()
            retry_after: float | None = None
            async with self._sem:
                await self._gate.wait()
                if time.monotonic() < self._slow_until and self._extra_interval > 0:
                    await self.sleep(self._extra_interval)
                started = time.monotonic()
                try:
                    self.requests += 1
                    response = await self._http.get(url, params=params)
                    self.latencies.append(time.monotonic() - started)
                    last_status = response.status_code
                    if response.status_code < 400 or (allow_404 and response.status_code == 404):
                        return RawResult(True, response.status_code, response.content, attempt,
                                         attempt - 1, None)
                    if response.status_code not in RETRYABLE_STATUS:
                        snippet = response.text[:200].replace("\n", " ")
                        return RawResult(False, response.status_code, None, attempt, attempt - 1,
                                         f"HTTP {response.status_code}: {snippet}")
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code == 429:
                        self.rate_limited += 1
                        retry_after = self._retry_after_seconds(response)
                        self._slow_down()
                        self._gate.penalise(retry_after if retry_after is not None else self._jitter(attempt))
                except RETRYABLE_EXC as exc:
                    last_error = repr(exc)
                except Interrupted:
                    raise
            self.retries += 1
            if attempt == self.policy.max_attempts:
                return RawResult(False, last_status, None, attempt, attempt - 1,
                                 f"{last_error} after {attempt} attempts")
            await self.sleep(retry_after if retry_after is not None else self._jitter(attempt))
        raise AssertionError("unreachable")

    def _retry_after_seconds(self, response: httpx.Response) -> float | None:
        try:
            return min(float(response.headers["Retry-After"]), self.policy.retry_after_cap)
        except (KeyError, ValueError):
            return None  # absent, or an HTTP-date: fall back to exponential backoff

    def _jitter(self, attempt: int) -> float:
        """Exponential backoff with FULL jitter: uniform(0, cap). Equal-jitter and plain exponential both keep retries
        synchronised across tasks, which is how a wobble becomes a thundering herd.
        """
        ceiling = min(self.policy.backoff_cap, self.policy.backoff_base * 2 ** (attempt - 1))
        return random.uniform(0.0, ceiling)

    def _slow_down(self) -> None:
        self._extra_interval = self.policy.min_interval * max(self.policy.slowdown_factor - 1.0, 0.0)
        self._slow_until = time.monotonic() + self.policy.slowdown_decay

    async def sleep(self, seconds: float) -> None:
        """Sleep in slices so Ctrl+C is noticed inside a long backoff."""
        end = time.monotonic() + seconds
        while True:
            self.stop.check()
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))


def build_client(policy: HttpPolicy, stop: StopFlag, *,
                 transport: httpx.AsyncBaseTransport | None = None) -> tuple[httpx.AsyncClient, RawClient]:
    timeout = httpx.Timeout(connect=policy.connect_timeout, read=policy.read_timeout,
                            write=policy.write_timeout, pool=policy.pool_timeout)
    http = httpx.AsyncClient(
        timeout=timeout, headers={"User-Agent": policy.user_agent},
        limits=httpx.Limits(max_connections=policy.concurrency * 2,
                            max_keepalive_connections=policy.concurrency),
        transport=transport, follow_redirects=True,
    )  # fmt: skip
    return http, RawClient(http, policy, stop)


# --------------------------------------------------------------------------- 6.


def fingerprint(payload: Any) -> str:
    """The sorted top-level keys. A change between 2020 and 2026 becomes a visible diff in the report rather than a
    mystery in the loader.
    """
    if isinstance(payload, dict):
        return "obj{" + ",".join(sorted(payload)) + "}"
    if isinstance(payload, list):
        if not payload:
            return "list[]"
        head = payload[0]
        inner = "obj{" + ",".join(sorted(head)) + "}" if isinstance(head, dict) else type(head).__name__
        return f"list[{inner}]"
    return type(payload).__name__


# source -> (expected container, keys that must be present).
EXPECTED_SHAPE: dict[str, tuple[type, tuple[str, ...]]] = {
    "contributions": (dict, ("Results", "TotalResultCount")),
    "probe": (dict, ("TotalResultCount",)),
    "corrections": (dict, ("TotalResultCount",)),
    "calendar": (list, ()),
    "sections": (list, ()),
    "hansard_divisions": (dict, ()),
    "commons_divisions": (list, ()),
    "commons_detail": (dict, ("DivisionId",)),
    "lords_divisions": (list, ()),
    "bills": (dict, ("items",)),
    "bill_detail": (dict, ()),
    "members": (dict, ()),
}


def parse_and_check(source: str, raw: bytes) -> tuple[Any, list[str]]:
    """(payload, anomalies). An unexpected shape is RECORDED, never raised: one odd day must not end a four-hour run,
    and the raw bytes are on disk either way.
    """
    anomalies: list[str] = []
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"unparseable:{exc.__class__.__name__}"]
    expected = EXPECTED_SHAPE.get(source)
    if expected is not None:
        container, keys = expected
        if not isinstance(payload, container):
            anomalies.append(f"shape:expected_{container.__name__}_got_{type(payload).__name__}")
        elif isinstance(payload, dict):
            missing = [k for k in keys if k not in payload]
            if missing:
                anomalies.append(f"missing_keys:{','.join(missing)}")
    return payload, anomalies


# --------------------------------------------------------------------------- 7.


@dataclass(frozen=True)
class Unit:
    source: str
    window_id: str
    key: str

    def __str__(self) -> str:
        return f"{self.source}/{self.window_id}/{self.key}"


@dataclass
class UnitResult:
    status: str
    http_status: int | None = None
    attempts: int = 0
    retries: int = 0
    sha256: str | None = None
    bytes: int = 0
    items: int | None = None
    fingerprint: str | None = None
    path: str | None = None
    url: str | None = None
    params: str | None = None
    error: str | None = None
    days: list[DayRow] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (DONE, NOT_SITTING, KNOWN_GAP)


_SECRETISH = ("key", "token", "secret", "password", "auth")


def safe_params(params: dict[str, Any] | None) -> str | None:
    """Params as JSON for the manifest, with anything that smells like a credential removed. These APIs need none, but
    a manifest is a file that gets shared.
    """
    if not params:
        return None
    clean = {k: v for k, v in params.items() if not any(s in k.casefold() for s in _SECRETISH)}
    return json.dumps(clean, sort_keys=True, default=str)


# --------------------------------------------------------------------------- 8.


@dataclass
class Context:
    cfg: FetchConfig
    window: WindowSpec
    cache: Cache
    client: RawClient
    writer: ManifestWriter
    manifest: Manifest
    stop: StopFlag
    breaker: Breaker
    disk: DiskGuard
    force: bool = False
    bytes_written: int = 0
    failures: int = 0

    def page_size(self) -> int:
        return self.cfg.http.page_size


async def _store(ctx: Context, source: str, rel: str, raw: bytes) -> tuple[Any, Written, list[str]]:
    payload, anomalies = parse_and_check(source, raw)
    written = ctx.cache.write(rel, raw)
    ctx.bytes_written += written.disk_bytes
    return payload, written, anomalies


def _failure(result: RawResult, url: str, params: dict | None) -> UnitResult:
    return UnitResult(status=FAILED, http_status=result.status, attempts=result.attempts,
                      retries=result.retries, error=result.error, url=url, params=safe_params(params))


# -- calendar ---------------------------------------------------------------


def _day_params(day: date, order: str, skip: int, take: int) -> dict[str, Any]:
    return {
        "queryParameters.startDate": day.isoformat(), "queryParameters.endDate": day.isoformat(),
        "queryParameters.orderBy": order, "queryParameters.take": take, "queryParameters.skip": skip,
    }  # fmt: skip


async def fetch_calendar(ctx: Context, unit: Unit) -> UnitResult:
    """One (house, year-month) of /overview/calendar.json: the sitting days the House records."""
    house, ym = unit.key.split("/")
    year, month = (int(x) for x in ym.split("-"))
    params = {"year": year, "month": month, "house": house}
    result = await ctx.client.get_raw(CALENDAR_URL, params)
    if not result.ok:
        return _failure(result, CALENDAR_URL, params)
    rel = calendar_path(ctx.window.id, house, year, month)
    payload, written, anomalies = await _store(ctx, "calendar", rel, result.content or b"")
    days = _calendar_days(payload)
    return UnitResult(status=DONE, http_status=result.status, attempts=result.attempts,
                      retries=result.retries, sha256=written.sha256, bytes=written.disk_bytes,
                      items=len(days), fingerprint=fingerprint(payload), path=rel,
                      url=CALENDAR_URL, params=safe_params(params),
                      error=";".join(anomalies) or None, extra={"days": days})


def _calendar_days(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    out = set()
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("ItemDate"), str):
            out.add(item["ItemDate"][:10])
    return sorted(out)


def calendar_sitting_days(ctx_cache: Cache, window: WindowSpec, house: str) -> set[date]:
    """Sitting days for one house, read back out of the cache."""
    days: set[date] = set()
    for year, month in window.months:
        rel = calendar_path(window.id, house, year, month)
        if not ctx_cache.exists(rel):
            continue
        try:
            for ds in _calendar_days(ctx_cache.read_json(rel)):
                d = date.fromisoformat(ds)
                if window.start <= d <= window.end:
                    days.add(d)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return days


# -- probe ------------------------------------------------------------------


async def fetch_probe(ctx: Context, unit: Unit) -> UnitResult:
    """One take=1 request per CALENDAR day, not per sitting day."""
    day = date.fromisoformat(unit.key)
    params = _day_params(day, ORDERS[0], 0, 1)
    result = await ctx.client.get_raw(SPOKEN_URL, params)
    if not result.ok:
        return _failure(result, SPOKEN_URL, params)
    rel = probe_path(ctx.window.id, day)
    payload, written, anomalies = await _store(ctx, "probe", rel, result.content or b"")
    listed = int((payload or {}).get("TotalResultCount") or 0) if isinstance(payload, dict) else 0
    return UnitResult(status=DONE if listed else NOT_SITTING, http_status=result.status,
                      attempts=result.attempts, retries=result.retries, sha256=written.sha256,
                      bytes=written.disk_bytes, items=listed, fingerprint=fingerprint(payload),
                      path=rel, url=SPOKEN_URL, params=safe_params(params),
                      error=";".join(anomalies) or None, extra={"listed": listed})


def cached_listed(cache: Cache, window: WindowSpec, day: date) -> int | None:
    rel = probe_path(window.id, day)
    if not cache.exists(rel):
        return None
    try:
        payload = cache.read_json(rel)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return int(payload.get("TotalResultCount") or 0) if isinstance(payload, dict) else None


# -- sections ---------------------------------------------------------------


async def fetch_sections(ctx: Context, unit: Unit) -> UnitResult:
    """/overview/sectionsforday.json: which Hansard sections a House produced that day."""
    house, day_s = unit.key.split("/")
    day = date.fromisoformat(day_s)
    params = {"date": day.isoformat(), "house": house}
    result = await ctx.client.get_raw(SECTIONS_URL, params, allow_404=True)
    if not result.ok:
        return _failure(result, SECTIONS_URL, params)
    rel = sections_path(ctx.window.id, house, day)
    payload, written, anomalies = await _store(ctx, "sections", rel, result.content or b"")
    n = len(payload) if isinstance(payload, list) else 0
    return UnitResult(status=DONE, http_status=result.status, attempts=result.attempts,
                      retries=result.retries, sha256=written.sha256, bytes=written.disk_bytes,
                      items=n, fingerprint=fingerprint(payload), path=rel, url=SECTIONS_URL,
                      params=safe_params(params), error=";".join(anomalies) or None)


# -- contributions ----------------------------------------------------------


async def _fetch_day_order(ctx: Context, day: date, order: str, take: int,
                           known_total: int | None) -> tuple[list[dict], int, int, list[str]]:
    """One sort order of one day, paged. Returns (items, listed total, disk bytes, anomalies)."""
    items: list[dict] = []
    anomalies: list[str] = []
    skip, total, disk, pages = 0, known_total, 0, 0
    while True:
        ctx.stop.check()
        rel = contributions_path(ctx.window.id, day, order, skip)
        payload = None
        # Page 0 is ALWAYS fetched fresh, never read from the cache.
        if not ctx.force and skip > 0 and ctx.cache.exists(rel):
            try:
                cached = ctx.cache.read_json(rel)
                if isinstance(cached, dict) and (total is None or cached.get("TotalResultCount") == total):
                    payload = cached
            except (OSError, ValueError, json.JSONDecodeError):
                payload = None  # corrupt: fetch it again
        if payload is None:
            params = _day_params(day, order, skip, take)
            result = await ctx.client.get_raw(SPOKEN_URL, params)
            if not result.ok:
                anomalies.append(f"page_failed:{order}:{skip}:{result.error}")
                return items, total or 0, disk, anomalies
            payload, written, page_anomalies = await _store(ctx, "contributions", rel, result.content or b"")
            disk += written.disk_bytes
            anomalies += [f"{a}@{order}:{skip}" for a in page_anomalies]
        if not isinstance(payload, dict):
            anomalies.append(f"page_shape:{order}:{skip}")
            return items, total or 0, disk, anomalies
        if total is None:
            total = int(payload.get("TotalResultCount") or 0)
        batch = payload.get("Results") or []
        items.extend(i for i in batch if isinstance(i, dict))
        pages += 1
        if not batch:
            break
        # By what the server returned, not by what was asked for.
        skip += len(batch)
        if skip >= (total or 0):
            break
        if pages > 200:  # 20,000 contributions in a day is impossible; something is looping
            anomalies.append(f"page_runaway:{order}")
            break
    return items, total or 0, disk, anomalies


async def fetch_contributions(ctx: Context, unit: Unit) -> UnitResult:
    """One sitting day, read in BOTH sort orders and merged by ContributionExtId."""
    day = date.fromisoformat(unit.key)
    take = ctx.page_size()
    # The probe's total is a hint from `plan`, not the truth: page 0 is re-read below and its total wins.
    probed = cached_listed(ctx.cache, ctx.window, day)
    listed = None
    per_order: dict[str, list[dict]] = {}
    anomalies: list[str] = []
    disk = 0
    for order in ORDERS:
        items, total, order_disk, order_anomalies = await _fetch_day_order(ctx, day, order, take, listed)
        per_order[order] = items
        anomalies += order_anomalies
        disk += order_disk
        listed = total if listed is None else listed
        if total and listed and total != listed:
            anomalies.append(f"listed_total_changed:{listed}->{total}")
            listed = max(listed, total)
    if probed is not None and listed is not None and probed != listed:
        anomalies.append(f"grew_since_probe:{probed}->{listed}")

    merged: dict[str, dict] = {}
    for order in ORDERS:
        for item in per_order[order]:
            cid = item.get("ContributionExtId")
            if cid:
                merged.setdefault(cid, item)
    ids = {o: {i.get("ContributionExtId") for i in per_order[o] if i.get("ContributionExtId")} for o in ORDERS}
    only_asc = len(ids[ORDERS[0]] - ids[ORDERS[1]])
    only_desc = len(ids[ORDERS[1]] - ids[ORDERS[0]])
    if only_asc or only_desc:
        anomalies.append(f"sort_order_disagreement:asc_only={only_asc},desc_only={only_desc}")
    listed = listed or 0
    # The API lists a few contributions it will not serve.
    if listed and len(merged) < listed:
        anomalies.append(f"unserved:{listed - len(merged)}")
    if listed and len(merged) > listed:
        anomalies.append(f"more_than_listed:{len(merged) - listed}")

    days = _day_rows(ctx.window, day, merged, per_order, listed, anomalies)
    failed_pages = [a for a in anomalies if a.startswith("page_failed")]
    status = FAILED if failed_pages else (KNOWN_GAP if any(a.startswith("unserved") for a in anomalies) else DONE)
    if not merged and not listed:
        status = NOT_SITTING
    return UnitResult(status=status, attempts=1, sha256=None, bytes=disk, items=len(merged),
                      fingerprint=fingerprint(per_order[ORDERS[0]][0]) if per_order[ORDERS[0]] else None,
                      path=f"contributions/{ctx.window.id}/{day.isoformat()}", url=SPOKEN_URL,
                      params=safe_params({"date": day.isoformat(), "orders": ",".join(ORDERS), "take": take}),
                      error=";".join(anomalies)[:800] or None, days=days,
                      extra={"member_ids": sorted({i["MemberId"] for i in merged.values()
                                                   if isinstance(i.get("MemberId"), int)})})


def _day_rows(window: WindowSpec, day: date, merged: dict[str, dict],
              per_order: dict[str, list[dict]], listed: int, anomalies: list[str]) -> list[DayRow]:
    """One row per (day, house), for BOTH houses, even when a house did not sit: a missing row and a row saying 'did
    not sit' are different claims, and only one of them is a fact.
    """
    rows = []
    for house in HOUSES:
        items = [i for i in merged.values() if i.get("House") == house]
        sections = Counter(str(i.get("Section")) for i in items)
        rows.append(DayRow(
            window_id=window.id, date=day, house=house,
            sitting=bool(items),
            listed=listed if house == HOUSES[0] else None,
            debates=len({i.get("DebateSectionExtId") for i in items if i.get("DebateSectionExtId")}),
            asc_items=sum(1 for i in per_order[ORDERS[0]] if i.get("House") == house),
            desc_items=sum(1 for i in per_order[ORDERS[1]] if i.get("House") == house),
            merged=len(items), sections=dict(sections),
            anomalies=[a for a in anomalies],
        ))  # fmt: skip
    return rows


# -- corrections ------------------------------------------------------------


async def fetch_corrections(ctx: Context, unit: Unit) -> UnitResult:
    """Written and Ministerial corrections for a day. Usually zero; occasionally one."""
    day = date.fromisoformat(unit.key)
    take = ctx.page_size()
    params = _day_params(day, ORDERS[0], 0, take)
    result = await ctx.client.get_raw(CORRECTIONS_URL, params)
    if not result.ok:
        return _failure(result, CORRECTIONS_URL, params)
    rel = corrections_path(ctx.window.id, day)
    payload, written, anomalies = await _store(ctx, "corrections", rel, result.content or b"")
    total = int((payload or {}).get("TotalResultCount") or 0) if isinstance(payload, dict) else 0
    if total > take:
        anomalies.append(f"corrections_truncated:{total}>{take}")
    return UnitResult(status=DONE if total else NOT_SITTING, http_status=result.status,
                      attempts=result.attempts, retries=result.retries, sha256=written.sha256,
                      bytes=written.disk_bytes, items=total, fingerprint=fingerprint(payload),
                      path=rel, url=CORRECTIONS_URL, params=safe_params(params),
                      error=";".join(anomalies) or None)


# -- divisions --------------------------------------------------------------


class Paged(NamedTuple):
    items: list[Any]
    disk: int
    anomalies: list[str]
    pages: int


async def _paged(ctx: Context, source: str, url: str, params_for, path_for, take: int,
                 *, container: str | None = None, id_key: str | None = None,
                 total_key: str | None = None, max_pages: int = 400) -> Paged:
    """Page a list endpoint until it stops returning rows."""
    items: list[Any] = []
    seen: set[Any] = set()
    anomalies: list[str] = []
    disk = skip = pages = 0
    stated_total: int | None = None
    short_page: int | None = None
    capped = False
    while pages < max_pages:
        ctx.stop.check()
        params = params_for(skip)
        result = await ctx.client.get_raw(url, params)
        if not result.ok:
            anomalies.append(f"page_failed:{skip}:{result.error}")
            break
        payload, written, page_anomalies = await _store(ctx, source, path_for(skip), result.content or b"")
        disk += written.disk_bytes
        anomalies += page_anomalies
        pages += 1
        if total_key and isinstance(payload, dict) and isinstance(payload.get(total_key), int):
            stated_total = payload[total_key]
        batch = payload.get(container) if (container and isinstance(payload, dict)) else payload
        if not isinstance(batch, list):
            anomalies.append(f"page_shape:{skip}")
            break
        if not batch:
            break
        if short_page is not None:
            capped = True  # a short page was NOT the last one: `take` is not being honoured
        elif len(batch) < take:
            short_page = len(batch)
        fresh = 0
        for row in batch:
            if id_key is not None and isinstance(row, dict):
                key = row.get(id_key)
                if key is None:
                    anomalies.append(f"row_without_{id_key}:{skip}")
                elif key in seen:
                    continue
                else:
                    seen.add(key)
            items.append(row)
            fresh += 1
        skip += len(batch)
        if id_key is not None and fresh == 0:
            anomalies.append(f"page_repeated:{skip - len(batch)}")  # the server ignored `skip`
            break
        if stated_total is not None and skip >= stated_total:
            break
    else:
        anomalies.append("page_runaway")
    if capped:
        # Worth saying out loud: it means `take` is not being honoured, so every other caller of this endpoint that
        # trusts a short page is losing rows.
        anomalies.append(f"server_page_cap:{short_page}<{take}")
    if stated_total is not None and len(items) != stated_total:
        anomalies.append(f"total_disagrees:{len(items)}!={stated_total}")
    return Paged(items, disk, anomalies, pages)


async def fetch_hansard_divisions(ctx: Context, unit: Unit) -> UnitResult:
    """Hansard's own division list for one House."""
    house = unit.key
    take = ctx.page_size()
    w = ctx.window

    def params_for(skip: int) -> dict[str, Any]:
        return {"queryParameters.house": house, "queryParameters.startDate": w.start.isoformat(),
                "queryParameters.endDate": w.end.isoformat(), "queryParameters.take": take,
                "queryParameters.skip": skip}

    items, disk, anomalies, _ = await _paged(
        ctx, "hansard_divisions", HANSARD_DIVISIONS_URL, params_for,
        lambda skip: hansard_divisions_path(w.id, house, skip), take, container="Results",
        id_key="Id", total_key="TotalResultCount")
    return UnitResult(status=FAILED if any(a.startswith("page_failed") for a in anomalies) else DONE,
                      attempts=1, bytes=disk, items=len(items),
                      path=f"hansard_divisions/{w.id}/{house}", url=HANSARD_DIVISIONS_URL,
                      params=safe_params(params_for(0)), error=";".join(anomalies)[:500] or None)


async def fetch_commons_divisions(ctx: Context, unit: Unit) -> UnitResult:
    """The Commons division index for the window. Details are separate, shared units."""
    take = ctx.page_size()
    w = ctx.window

    def params_for(skip: int) -> dict[str, Any]:
        return {"queryParameters.startDate": w.start.isoformat(), "queryParameters.endDate": w.end.isoformat(),
                "queryParameters.take": take, "queryParameters.skip": skip}

    items, disk, anomalies, _ = await _paged(
        ctx, "commons_divisions", COMMONS_SEARCH_URL, params_for,
        lambda skip: commons_index_path(w.id, skip), take, id_key="DivisionId")
    ids = sorted({i["DivisionId"] for i in items if isinstance(i, dict) and isinstance(i.get("DivisionId"), int)})
    return UnitResult(status=FAILED if any(a.startswith("page_failed") for a in anomalies) else DONE,
                      attempts=1, bytes=disk, items=len(ids), path=f"commons_divisions/{w.id}",
                      url=COMMONS_SEARCH_URL, params=safe_params(params_for(0)),
                      error=";".join(anomalies)[:500] or None, extra={"division_ids": ids})


async def fetch_commons_detail(ctx: Context, unit: Unit) -> UnitResult:
    """One Commons division in full: both lobbies, tellers, NoVoteRecorded, proxies."""
    division_id = int(unit.key)
    url = COMMONS_DETAIL_URL.format(id=division_id)
    result = await ctx.client.get_raw(url)
    if not result.ok:
        return _failure(result, url, None)
    rel = commons_detail_path(division_id)
    payload, written, anomalies = await _store(ctx, "commons_detail", rel, result.content or b"")
    voters = sum(len(payload.get(k) or []) for k in ("Ayes", "Noes", "NoVoteRecorded")) if isinstance(payload, dict) else 0
    return UnitResult(status=DONE, http_status=result.status, attempts=result.attempts,
                      retries=result.retries, sha256=written.sha256, bytes=written.disk_bytes,
                      items=voters, fingerprint=fingerprint(payload), path=rel, url=url,
                      error=";".join(anomalies) or None,
                      extra={"member_ids": _member_ids_from_commons(payload)})


def _member_ids_from_commons(payload: Any) -> list[int]:
    if not isinstance(payload, dict):
        return []
    ids = set()
    for key in ("Ayes", "Noes", "NoVoteRecorded", "AyeTellers", "NoTellers"):
        for v in payload.get(key) or []:
            if isinstance(v, dict) and isinstance(v.get("MemberId"), int):
                ids.add(v["MemberId"])
    return sorted(ids)


async def fetch_lords_divisions(ctx: Context, unit: Unit) -> UnitResult:
    """Lords divisions. The search response already carries the voter lists, so there is no detail pass."""
    take = ctx.page_size()
    w = ctx.window

    def params_for(skip: int) -> dict[str, Any]:
        return {"startDate": w.start.isoformat(), "endDate": w.end.isoformat(), "take": take, "skip": skip}

    items, disk, anomalies, _ = await _paged(
        ctx, "lords_divisions", LORDS_SEARCH_URL, params_for,
        lambda skip: lords_index_path(w.id, skip), take, id_key="divisionId")
    if not items:
        anomalies.append("no_lords_divisions_in_window")
    return UnitResult(status=FAILED if any(a.startswith("page_failed") for a in anomalies) else DONE,
                      attempts=1, bytes=disk, items=len(items), path=f"lords_divisions/{w.id}",
                      url=LORDS_SEARCH_URL, params=safe_params(params_for(0)),
                      error=";".join(anomalies)[:500] or None,
                      extra={"member_ids": _member_ids_from_lords(items)})


def _member_ids_from_lords(items: list[Any]) -> list[int]:
    ids = set()
    for div in items:
        if not isinstance(div, dict):
            continue
        for key in ("contents", "notContents", "contentTellers", "notContentTellers"):
            for v in div.get(key) or []:
                if isinstance(v, dict) and isinstance(v.get("memberId"), int):
                    ids.add(v["memberId"])
    return sorted(ids)


# -- bills ------------------------------------------------------------------


async def fetch_bill_sittings(ctx: Context, unit: Unit) -> UnitResult:
    """Every bill SITTING inside the window, from bills-api /Sittings?DateFrom&DateTo."""
    take = ctx.page_size()
    w = ctx.window

    def params_for(skip: int) -> dict[str, Any]:
        return {"DateFrom": w.start.isoformat(), "DateTo": w.end.isoformat(), "Take": take, "Skip": skip}

    items, disk, anomalies, _ = await _paged(
        ctx, "bills", f"{BILLS_BASE}/Sittings", params_for,
        lambda skip: bill_sittings_path(w.id, skip), take, container="items",
        id_key="id", total_key="totalResults")
    bill_ids = sorted({i["billId"] for i in items if isinstance(i, dict) and isinstance(i.get("billId"), int)})
    return UnitResult(status=FAILED if any(a.startswith("page_failed") for a in anomalies) else DONE,
                      attempts=1, bytes=disk, items=len(items), path=f"bills/{w.id}",
                      url=f"{BILLS_BASE}/Sittings", params=safe_params(params_for(0)),
                      error=";".join(anomalies)[:500] or None, extra={"bill_ids": bill_ids})


async def fetch_bill(ctx: Context, unit: Unit) -> UnitResult:
    """One bill: its record and its stages. Shared across windows."""
    bill_id = int(unit.key)
    disk, anomalies, items = 0, [], 0
    for what, url in (("detail", f"{BILLS_BASE}/Bills/{bill_id}"),
                      ("stages", f"{BILLS_BASE}/Bills/{bill_id}/Stages")):
        result = await ctx.client.get_raw(url, {"Take": 100} if what == "stages" else None, allow_404=True)
        if not result.ok:
            return _failure(result, url, None)
        if result.is_404:
            anomalies.append(f"{what}:404")
            continue
        payload, written, part_anomalies = await _store(ctx, "bill_detail", bill_path(bill_id, what),
                                                        result.content or b"")
        disk += written.disk_bytes
        anomalies += [f"{what}:{a}" for a in part_anomalies]
        if what == "stages" and isinstance(payload, dict):
            items = len(payload.get("items") or [])
    return UnitResult(status=DONE, attempts=1, bytes=disk, items=items,
                      path=f"bills/bill/{bill_id}", url=f"{BILLS_BASE}/Bills/{bill_id}",
                      error=";".join(anomalies)[:300] or None)


# -- members ----------------------------------------------------------------


async def fetch_member(ctx: Context, unit: Unit) -> UnitResult:
    """One member, across every endpoint ingest_members.py stores. Shared across windows."""
    mnis_id = int(unit.key)
    disk, anomalies, fetched = 0, [], 0
    profile_missing = False
    for endpoint in ctx.cfg.member_endpoints:
        ctx.stop.check()
        url = f"{MEMBERS_BASE}/{mnis_id}" + (f"/{endpoint}" if endpoint else "")
        result = await ctx.client.get_raw(url, allow_404=True)
        if not result.ok:
            if not endpoint:
                return _failure(result, url, None)
            anomalies.append(f"{endpoint or 'profile'}:failed")
            continue
        if result.is_404:
            if not endpoint:
                profile_missing = True
            anomalies.append(f"{endpoint or 'profile'}:404")
            continue
        _, written, part = await _store(ctx, "members", member_path(mnis_id, endpoint), result.content or b"")
        disk += written.disk_bytes
        fetched += 1
        anomalies += [f"{endpoint or 'profile'}:{a}" for a in part]
    return UnitResult(status=KNOWN_GAP if profile_missing else DONE, attempts=1, bytes=disk,
                      items=fetched, path=f"members/{mnis_id}", url=f"{MEMBERS_BASE}/{mnis_id}",
                      error=";".join(anomalies)[:300] or None)


FETCHERS = {
    "calendar": fetch_calendar,
    "probe": fetch_probe,
    "sections": fetch_sections,
    "contributions": fetch_contributions,
    "corrections": fetch_corrections,
    "hansard_divisions": fetch_hansard_divisions,
    "commons_divisions": fetch_commons_divisions,
    "commons_detail": fetch_commons_detail,
    "lords_divisions": fetch_lords_divisions,
    "bills": fetch_bill_sittings,
    "bill": fetch_bill,
    "members": fetch_member,
}

# Sources whose units describe an object rather than a window.
SHARED_SOURCES = frozenset({"commons_detail", "bill", "members"})


def unit_window(source: str, window_id: str) -> str:
    return SHARED if source in SHARED_SOURCES else window_id


# --------------------------------------------------------------------------- 9.


@dataclass
class WindowScan:
    """What the cache says about a window, recomputed from the files themselves."""

    window_id: str
    pages: int = 0
    contributions: int = 0
    member_ids: set[int] = field(default_factory=set)
    debate_ids: set[str] = field(default_factory=set)
    sections: Counter = field(default_factory=Counter)
    houses: Counter = field(default_factory=Counter)
    headings: Counter = field(default_factory=Counter)
    per_day: dict[str, int] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id, "pages": self.pages, "contributions": self.contributions,
            "members": len(self.member_ids), "debates": len(self.debate_ids),
            "sections": dict(self.sections), "houses": dict(self.houses),
            "top_headings": dict(self.headings.most_common(25)),
            "per_day": self.per_day, "unreadable": self.unreadable,
        }  # fmt: skip


def _rel(cache: Cache, path: Path) -> str:
    return str(path.relative_to(cache.root)).replace(os.sep, "/")


def scan_window(cache: Cache, window: WindowSpec) -> WindowScan:
    """Read every cached contributions page for a window once, merging by id."""
    scan = WindowScan(window_id=window.id)
    base = cache.root / "contributions" / window.id
    if not base.is_dir():
        return scan
    for day_dir in sorted(base.iterdir()):
        if not day_dir.is_dir():
            continue
        merged: dict[str, dict] = {}
        for page in sorted(day_dir.glob("*.json.gz")):
            try:
                payload = cache.read_json(_rel(cache, page))
            except (OSError, ValueError, EOFError, json.JSONDecodeError):
                scan.unreadable.append(_rel(cache, page))
                continue
            scan.pages += 1
            results = payload.get("Results") if isinstance(payload, dict) else None
            for item in results or []:
                cid = item.get("ContributionExtId") if isinstance(item, dict) else None
                if cid:
                    merged.setdefault(cid, item)
        scan.per_day[day_dir.name] = len(merged)
        scan.contributions += len(merged)
        for item in merged.values():
            if isinstance(item.get("MemberId"), int):
                scan.member_ids.add(item["MemberId"])
            if item.get("DebateSectionExtId"):
                scan.debate_ids.add(item["DebateSectionExtId"])
            scan.sections[f"{item.get('House')} / {item.get('Section')}"] += 1
            scan.houses[str(item.get("House"))] += 1
            heading = (item.get("DebateSection") or "").strip()
            if heading:
                scan.headings[heading] += 1
    return scan


def _cached_ids(cache: Cache, directory: str, key: str, *, container: str | None = None) -> list[int]:
    """Collect one integer field out of every cached page under a directory."""
    base = cache.root / directory
    ids: set[int] = set()
    if not base.is_dir():
        return []
    for page in sorted(base.glob("*.json.gz")):
        try:
            payload = cache.read_json(_rel(cache, page))
        except (OSError, ValueError, EOFError, json.JSONDecodeError):
            continue
        rows = payload.get(container) if (container and isinstance(payload, dict)) else payload
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get(key), int):
                ids.add(row[key])
    return sorted(ids)


def _probe_sitting_days(cache: Cache, window: WindowSpec) -> set[date]:
    """Days whose take=1 probe reported at least one contribution."""
    return {d for d in window.days if cached_listed(cache, window, d)}


# -- cross-source division reconciliation -----------------------------------


@dataclass
class DivisionTally:
    """One House's divisions in one window, counted from two independent sources."""

    house: str
    hansard: set[tuple[str, int]] = field(default_factory=set)  # (day, Hansard division number)
    votes: set[tuple[str, int]] = field(default_factory=set)  # the same key from the Votes API
    hansard_committee: int = 0  # Hansard rows flagged IsCommitteeDivision
    votes_deferred: int = 0  # votes rows flagged IsDeferred
    deferred: set[tuple[str, int]] = field(default_factory=set)
    hansard_unnumbered: int = 0  # rows with no usable (day, number)
    votes_unnumbered: int = 0

    # Tolerance, in days, for pairing one division that the two sources date differently.
    DATE_TOLERANCE_DAYS = 2

    def _paired(self) -> tuple[list[tuple[str, str, int]], set, set]:
        """(date-shifted pairs, Hansard-only, Votes-only) after pairing on the number."""
        only_h, only_v = self.hansard - self.votes, self.votes - self.hansard
        by_number: dict[int, list[str]] = {}
        for day, number in sorted(only_v):
            by_number.setdefault(number, []).append(day)
        shifted: list[tuple[str, str, int]] = []
        for h_day, number in sorted(only_h):
            for v_day in by_number.get(number, []):
                if abs((date.fromisoformat(h_day) - date.fromisoformat(v_day)).days) <= self.DATE_TOLERANCE_DAYS:
                    shifted.append((h_day, v_day, number))
                    by_number[number].remove(v_day)
                    only_h = only_h - {(h_day, number)}
                    only_v = only_v - {(v_day, number)}
                    break
        return shifted, only_h, only_v

    @property
    def date_shifted(self) -> list[tuple[str, str, int]]:
        return self._paired()[0]

    @property
    def only_hansard(self) -> list[tuple[str, int]]:
        """Divisions Hansard minutes that the Votes API has no record of."""
        return sorted(self._paired()[1])

    @property
    def only_votes(self) -> list[tuple[str, int]]:
        """Divisions the Votes API records that Hansard's own division list omits."""
        return sorted(self._paired()[2])

    def tolerated_votes_only(self) -> int:
        """How many one-sided Votes rows count as source noise rather than a bug."""
        return max(3, round(0.02 * max(len(self.hansard), len(self.votes))))

    @property
    def matched(self) -> bool:
        shifted, only_h, only_v = self._paired()
        return not only_h and len(only_v) <= self.tolerated_votes_only()

    def as_dict(self) -> dict[str, Any]:
        shifted, only_h, only_v = self._paired()
        return {"house": self.house, "hansard": len(self.hansard), "votes": len(self.votes),
                "hansard_committee": self.hansard_committee, "votes_deferred": self.votes_deferred,
                "hansard_unnumbered": self.hansard_unnumbered, "votes_unnumbered": self.votes_unnumbered,
                "date_shifted": [f"{h}/{v}#{n}" for h, v, n in shifted],
                "only_hansard": [f"{d}#{n}" for d, n in sorted(only_h)[:20]],
                "only_votes": [f"{d}#{n}" for d, n in sorted(only_v)[:20]],
                "tolerated_votes_only": self.tolerated_votes_only(),
                "matched": self.matched}


def _rows_under(cache: Cache, directory: str, *, container: str | None = None) -> list[dict]:
    base = cache.root / directory
    out: list[dict] = []
    if not base.is_dir():
        return out
    for page in sorted(base.glob("*.json.gz")):
        try:
            payload = cache.read_json(_rel(cache, page))
        except (OSError, ValueError, EOFError, json.JSONDecodeError):
            continue
        rows = payload.get(container) if (container and isinstance(payload, dict)) else payload
        out += [r for r in (rows or []) if isinstance(r, dict)]
    return out


def _day_of(value: Any) -> str | None:
    return value[:10] if isinstance(value, str) and len(value) >= 10 else None


def reconcile_divisions(cache: Cache, window: WindowSpec) -> dict[str, DivisionTally]:
    """Hansard's own division list against the Commons and Lords Votes APIs, per House."""
    tallies = {h: DivisionTally(house=h) for h in HOUSES}

    def add(target: set, tally: DivisionTally, day: str | None, number: int | None, *, votes: bool) -> None:
        if day and number is not None:
            target.add((day, number))
        elif votes:
            tally.votes_unnumbered += 1
        else:
            tally.hansard_unnumbered += 1

    for house in HOUSES:
        tally = tallies[house]
        for row in _rows_under(cache, f"hansard_divisions/{window.id}/{house}", container="Results"):
            if row.get("IsCommitteeDivision"):
                tally.hansard_committee += 1  # minuted in Hansard, absent from the Votes API
                continue
            add(tally.hansard, tally, _day_of(row.get("Date")), _number(row.get("Number")), votes=False)
    commons = tallies["Commons"]
    for row in _rows_under(cache, f"commons_divisions/{window.id}"):
        day, number = _day_of(row.get("Date")), _number(row.get("Number"))
        if row.get("IsDeferred"):
            commons.votes_deferred += 1
            if day and number is not None:
                commons.deferred.add((day, number))
        add(commons.votes, commons, day, number, votes=True)
    lords = tallies["Lords"]
    for row in _rows_under(cache, f"lords_divisions/{window.id}"):
        add(lords.votes, lords, _day_of(row.get("date")), _number(row.get("number")), votes=True)
    return tallies


def _number(value: Any) -> int | None:
    """Hansard serves the division number as a string, the votes APIs as an int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _member_ids_from_division_cache(cache: Cache, window: WindowSpec) -> set[int]:
    ids: set[int] = set()
    detail_dir = cache.root / "commons_divisions" / "detail"
    if detail_dir.is_dir():
        for page in detail_dir.glob("*.json.gz"):
            try:
                ids.update(_member_ids_from_commons(cache.read_json(_rel(cache, page))))
            except (OSError, ValueError, EOFError, json.JSONDecodeError):
                continue
    lords_dir = cache.root / "lords_divisions" / window.id
    if lords_dir.is_dir():
        for page in lords_dir.glob("*.json.gz"):
            try:
                payload = cache.read_json(_rel(cache, page))
            except (OSError, ValueError, EOFError, json.JSONDecodeError):
                continue
            ids.update(_member_ids_from_lords(payload if isinstance(payload, list) else []))
    return ids


def plan_units(ctx: Context, source: str) -> list[Unit]:
    """The units for one source, from the window and from what earlier sources cached."""
    w, cache = ctx.window, ctx.cache
    if source == "calendar":
        return [Unit(source, w.id, f"{h}/{y:04d}-{m:02d}") for h in HOUSES for y, m in w.months]
    if source == "probe":
        return [Unit(source, w.id, d.isoformat()) for d in w.days]
    if source == "sections":
        probed = _probe_sitting_days(cache, w)
        out = []
        for house in HOUSES:
            for d in sorted(calendar_sitting_days(cache, w, house) | probed):
                out.append(Unit(source, w.id, f"{house}/{d.isoformat()}"))
        return out
    if source in ("contributions", "corrections"):
        days = _probe_sitting_days(cache, w)
        days |= calendar_sitting_days(cache, w, "Commons") | calendar_sitting_days(cache, w, "Lords")
        return [Unit(source, w.id, d.isoformat()) for d in sorted(days)]
    if source == "hansard_divisions":
        return [Unit(source, w.id, h) for h in HOUSES]
    if source in ("commons_divisions", "lords_divisions", "bills"):
        return [Unit(source, w.id, "index")]
    if source == "commons_detail":
        return [Unit(source, SHARED, str(i))
                for i in _cached_ids(cache, f"commons_divisions/{w.id}", "DivisionId")]
    if source == "bill":
        return [Unit(source, SHARED, str(i))
                for i in _cached_ids(cache, f"bills/{w.id}", "billId", container="items")]
    if source == "members":
        seen = set(scan_window(cache, w).member_ids) | _member_ids_from_division_cache(cache, w)
        if not ctx.cfg.fetch_known_members:
            seen -= load_known_member_ids(ctx.cfg.known_ids_file)
        return [Unit(source, SHARED, str(i)) for i in sorted(seen)]
    raise ConfigError(f"no unit plan for source {source!r}")


# The passes in dependency order, each with the follow-on pass it unlocks.
PASSES: tuple[tuple[str, str | None], ...] = (
    ("calendar", None), ("probe", None), ("sections", None),
    ("contributions", None), ("corrections", None),
    ("hansard_divisions", None),
    ("commons_divisions", "commons_detail"),
    ("lords_divisions", None),
    ("bills", "bill"),
    ("members", None),
)  # fmt: skip

# Passes that `plan` may run: cheap indexes and probes, never a day of text.
PLAN_PASSES = ("calendar", "probe")


def passes_for(sources: Iterable[str]) -> list[str]:
    """Expand enabled source names into the actual passes, in dependency order."""
    enabled = set(sources)
    out: list[str] = []
    for name, follow in PASSES:
        if name in enabled:
            out.append(name)
            if follow:
                out.append(follow)
    return out


# --------------------------------------------------------------------------- 10.


class DiskLow(Exception):
    def __init__(self, free_gb: float) -> None:
        super().__init__(f"only {free_gb:.1f} GB free")
        self.free_gb = free_gb


class BreakerExhausted(Exception):
    def __init__(self, pauses: int) -> None:
        super().__init__(f"stopped cleanly after {pauses} circuit-breaker pause(s)")
        self.pauses = pauses


@dataclass
class Progress:
    window_id: str = ""
    source: str = ""
    done: int = 0
    total: int = 0
    failures: int = 0
    skipped: int = 0
    requests: int = 0
    retries: int = 0
    started: float = field(default_factory=time.monotonic)

    def eta_seconds(self) -> float | None:
        fresh = self.done - self.skipped
        if fresh < 2 or self.done >= self.total:
            return None
        rate = fresh / max(time.monotonic() - self.started, 1e-6)
        return (self.total - self.done) / rate if rate > 0 else None


@dataclass
class RunOutcome:
    window_id: str
    exit_code: int = EXIT_OK
    interrupted: bool = False
    breaker_stopped: bool = False
    disk_stopped: bool = False
    units_done: int = 0
    units_skipped: int = 0
    units_failed: int = 0
    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    bytes_written: int = 0
    elapsed: float = 0.0
    message: str = ""


def already_complete(ctx: Context, unit: Unit) -> bool:
    """Resume rule: skip a unit only when the manifest says it finished AND the bytes are still there and still hash to
    what was recorded. A deleted or corrupted file is work.
    """
    if ctx.force:
        return False
    row = ctx.manifest.unit(unit.source, unit.window_id, unit.key)
    if row is None or row["status"] not in (DONE, NOT_SITTING, KNOWN_GAP):
        return False
    path, expected = row["path"], row["sha256"]
    if not path:
        return True
    if expected:  # a single-file unit: the hash is checkable, so check it
        return ctx.cache.sha256(path) == expected
    return (ctx.cache.root / path).exists()  # a multi-file unit: its directory must survive


async def run_source(ctx: Context, source: str, on_progress=None,
                     limit: int | None = None) -> tuple[int, int, int]:
    """Run one pass. Returns (done, skipped, failed)."""
    units = plan_units(ctx, source)
    if limit is not None:
        units = units[:limit]
    progress = Progress(window_id=ctx.window.id, source=source, total=len(units))
    todo = [u for u in units if not already_complete(ctx, u)]
    progress.skipped = len(units) - len(todo)
    progress.done = progress.skipped
    if on_progress:
        on_progress(progress)
    if not todo:
        return 0, progress.skipped, 0

    fetcher = FETCHERS[source]
    semaphore = asyncio.Semaphore(ctx.cfg.http.concurrency)
    done = failed = 0
    lock = asyncio.Lock()

    async def one(unit: Unit) -> None:
        nonlocal done, failed
        async with semaphore:
            ctx.stop.check()
            try:
                result = await fetcher(ctx, unit)
            except Interrupted:
                raise
            except Exception as exc:  # a bug in one fetcher must not cost the other 3,000 units
                log.exception("unit %s raised", unit)
                result = UnitResult(status=FAILED, error=f"{type(exc).__name__}: {exc}"[:300])
            ctx.writer.record_unit(unit.source, unit.window_id, unit.key, result)
            for row in result.days:
                ctx.writer.record_day(row)
            async with lock:
                ctx.breaker.record(result.ok)
                if result.ok:
                    done += 1
                else:
                    failed += 1
                    ctx.failures += 1
                    log.warning("%s FAILED: %s", unit, result.error)
                progress.done += 1
                progress.failures = failed
                progress.requests = ctx.client.requests
                progress.retries = ctx.client.retries
                if on_progress:
                    on_progress(progress)

    index = 0
    while index < len(todo):
        ctx.stop.check()
        if (free := ctx.disk.low()) is not None:
            raise DiskLow(free)
        if ctx.breaker.tripped:
            if ctx.breaker.exhausted:
                raise BreakerExhausted(ctx.breaker.pauses)
            seconds = ctx.breaker.take_pause()
            log.error("circuit breaker: %d consecutive unit failures; pausing %.0fs (pause %d of %d)",
                      ctx.cfg.breaker.consecutive_failures, seconds, ctx.breaker.pauses,
                      ctx.cfg.breaker.max_pauses)
            await ctx.client.sleep(seconds)
        # Small batches on purpose: the breaker and the disk guard are checked between batches, so a large one would let
        # a dead network burn through 40 units first.
        batch = todo[index:index + max(2, ctx.cfg.http.concurrency * 2)]
        index += len(batch)
        await asyncio.gather(*(one(u) for u in batch))
        await ctx.writer.flush()  # checkpoint: a kill here loses nothing already fetched
    return done, progress.skipped, failed


def open_run(cfg: FetchConfig, window: WindowSpec, stop: StopFlag, *, force: bool = False,
             transport: httpx.AsyncBaseTransport | None = None
             ) -> tuple[Context, httpx.AsyncClient, Manifest, ManifestWriter]:
    """Everything a run needs, wired together. Split out so tests can drive a Context directly."""
    cfg.cache_root.mkdir(parents=True, exist_ok=True)
    cache = Cache(cfg.cache_root)
    swept = cache.sweep_partials()
    if swept:
        log.info("removed %d partial file(s) left behind by an earlier kill", swept)
    manifest = Manifest(cfg.manifest_path)
    writer = ManifestWriter(manifest)
    http, client = build_client(cfg.http, stop, transport=transport)
    ctx = Context(cfg=cfg, window=window, cache=cache, client=client, writer=writer,
                  manifest=manifest, stop=stop, breaker=Breaker(cfg.breaker),
                  disk=DiskGuard(cfg.cache_root, cfg.disk), force=force)
    return ctx, http, manifest, writer


async def fetch_window(cfg: FetchConfig, window: WindowSpec, *, sources: Sequence[str] | None = None,
                       force: bool = False, on_progress=None, limit: int | None = None,
                       transport: httpx.AsyncBaseTransport | None = None,
                       stop: StopFlag | None = None) -> RunOutcome:
    """Fetch one window. Resumable, interruptible, and never fatal because of one bad day."""
    started = time.monotonic()
    stop = stop or StopFlag()
    outcome = RunOutcome(window_id=window.id)
    ctx, http, manifest, writer = open_run(cfg, window, stop, force=force, transport=transport)
    run_id = manifest.start_run(window.id, "fetch")

    if (free := ctx.disk.low(force=True)) is not None:
        outcome.exit_code, outcome.disk_stopped = EXIT_DISK, True
        outcome.message = (f"refusing to start: {free:.1f} GB free, "
                           f"below the {cfg.disk.min_free_gb} GB floor")
        manifest.finish_run(run_id, requests=0, retries=0, failures=0, nbytes=0, outcome="disk")
        await http.aclose()
        manifest.close()
        return outcome

    writer.start()
    try:
        for source in passes_for(sources or cfg.sources):
            log.info("[%s] pass: %s", window.id, source)
            done, skipped, failed = await run_source(ctx, source, on_progress, limit)
            outcome.units_done += done
            outcome.units_skipped += skipped
            outcome.units_failed += failed
    except Interrupted:
        outcome.interrupted, outcome.exit_code = True, EXIT_INTERRUPTED
        outcome.message = "interrupted by Ctrl+C; state saved"
    except DiskLow as exc:
        outcome.disk_stopped, outcome.exit_code, outcome.message = True, EXIT_DISK, str(exc)
    except BreakerExhausted as exc:
        outcome.breaker_stopped, outcome.exit_code, outcome.message = True, EXIT_BREAKER, str(exc)
    finally:
        # Whatever happened, the manifest is flushed and the socket closed before we leave.
        try:
            await writer.close()
        except Exception:  # noqa: BLE001 - a failure here must not mask the real one
            log.exception("manifest flush failed")
        await http.aclose()
        outcome.requests = ctx.client.requests
        outcome.retries = ctx.client.retries
        outcome.rate_limited = ctx.client.rate_limited
        outcome.bytes_written = ctx.bytes_written
        outcome.elapsed = time.monotonic() - started
        manifest.finish_run(
            run_id, requests=outcome.requests, retries=outcome.retries,
            failures=outcome.units_failed, nbytes=outcome.bytes_written,
            outcome=("interrupted" if outcome.interrupted else "disk" if outcome.disk_stopped
                     else "breaker" if outcome.breaker_stopped
                     else "failures" if outcome.units_failed else "ok"))
        manifest.close()
    if outcome.exit_code == EXIT_OK and outcome.units_failed:
        outcome.exit_code = EXIT_FAILURES
        outcome.message = f"{outcome.units_failed} unit(s) failed; re-run the same command to retry them"
    return outcome


# --------------------------------------------------------------------------- 11. plan
# ---------------------------------------------------------------------------


@dataclass
class WindowPlan:
    window: WindowSpec
    commons_days: int = 0
    lords_days: int = 0
    calendar_days: int = 0
    probed_days: int = 0
    sitting_days: int = 0
    listed: int = 0
    per_day: dict[str, int] = field(default_factory=dict)
    disagreements: list[str] = field(default_factory=list)

    @property
    def house_days(self) -> int:
        return self.commons_days + self.lords_days

    def requests(self, cfg: FetchConfig) -> dict[str, int]:
        take = cfg.http.page_size
        pages = sum(-(-n // take) for n in self.per_day.values())
        return {
            "calendar": 2 * len(self.window.months),
            "probe": self.calendar_days,
            "sections": self.commons_days + self.lords_days,
            "contributions": 2 * pages,
            "corrections": self.sitting_days,
            "divisions_and_bills": 40,  # indexes plus per-division and per-bill detail: measured small
            "members": 0,  # counted only after the contributions pass names the ids
        }  # fmt: skip

    def estimate(self, cfg: FetchConfig) -> dict[str, float]:
        e = cfg.estimate
        reqs = self.requests(cfg)
        return {
            "requests": sum(reqs.values()),
            "chunks": self.listed * e.chunks_per_contribution,
            "served": self.listed * e.served_fraction,
            "cache_bytes": self.listed * e.bytes_per_contribution,
        }  # fmt: skip


async def plan_windows(cfg: FetchConfig, windows: Sequence[WindowSpec], *, on_progress=None,
                       transport: httpx.AsyncBaseTransport | None = None,
                       stop: StopFlag | None = None) -> tuple[list[WindowPlan], RawClient]:
    """Cheap: the calendar for each month, and one take=1 probe per calendar day."""
    stop = stop or StopFlag()
    plans: list[WindowPlan] = []
    client: RawClient | None = None
    for window in windows:
        ctx, http, manifest, writer = open_run(cfg, window, stop, transport=transport)
        client = ctx.client
        writer.start()
        try:
            for source in PLAN_PASSES:
                await run_source(ctx, source, on_progress)
        finally:
            await writer.close()
            await http.aclose()
        plan = WindowPlan(window=window, calendar_days=len(window.days))
        cache = ctx.cache
        commons = calendar_sitting_days(cache, window, "Commons")
        lords = calendar_sitting_days(cache, window, "Lords")
        plan.commons_days, plan.lords_days = len(commons), len(lords)
        for day in window.days:
            listed = cached_listed(cache, window, day)
            if listed:
                plan.per_day[day.isoformat()] = listed
                plan.listed += listed
            in_calendar = day in commons or day in lords
            if bool(listed) != in_calendar:
                plan.disagreements.append(
                    f"{day.isoformat()}: calendar={'sitting' if in_calendar else 'not sitting'}, "
                    f"contributions={listed or 0}")
        plan.sitting_days = len(plan.per_day)
        plan.probed_days = sum(1 for d in window.days if cached_listed(cache, window, d) is not None)
        manifest.close()
        plans.append(plan)
    return plans, client  # type: ignore[return-value]


def estimate_runtime(plans: Sequence[WindowPlan], cfg: FetchConfig,
                     p50: float, p95: float) -> tuple[float, float]:
    """(optimistic, pessimistic) seconds for the fetch, from MEASURED request latency."""
    requests = sum(p.estimate(cfg)["requests"] for p in plans)
    effective = max(1.0, cfg.http.concurrency * 0.7)
    return (requests * max(p50, 0.05) / effective, requests * max(p95, 0.1) / effective)


# --------------------------------------------------------------------------- 12. verify: did we actually get the whole
# window? ---------------------------------------------------------------------------


@dataclass
class Problem:
    kind: str
    detail: str
    unit: str | None = None

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}" + (f" [{self.unit}]" if self.unit else "")


@dataclass
class VerifyReport:
    window_id: str
    checked_units: int = 0
    problems: list[Problem] = field(default_factory=list)
    known_gaps: list[Problem] = field(default_factory=list)
    repaired: int = 0
    scan: WindowScan | None = None
    divisions: dict[str, DivisionTally] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.problems

    def add(self, kind: str, detail: str, unit: str | None = None) -> None:
        self.problems.append(Problem(kind, detail, unit))

    def note_gap(self, kind: str, detail: str, unit: str | None = None) -> None:
        self.known_gaps.append(Problem(kind, detail, unit))


def verify_window(cfg: FetchConfig, window: WindowSpec) -> VerifyReport:
    """Check the manifest against the cache, and the cache against itself."""
    report = VerifyReport(window_id=window.id)
    cache = Cache(cfg.cache_root)
    if not cfg.manifest_path.exists():
        report.add("no_manifest", "nothing has been fetched yet")
        return report
    # Read-only: verify must never take a write lock on a manifest a fetch may be using.
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        rows = manifest.units(window_id=window.id)
        report.checked_units = len(rows)
        for row in rows:
            unit_name = f"{row['source']}/{row['window_id']}/{row['key']}"
            if row["status"] == FAILED:
                report.add("failed_unit", row["last_error"] or "no error recorded", unit_name)
                continue
            if row["status"] == KNOWN_GAP:
                report.note_gap("known_gap", row["last_error"] or "", unit_name)
            if row["status"] == PENDING:
                report.add("pending_unit", "never finished", unit_name)
                continue
            path = row["path"]
            if not path or row["status"] == NOT_SITTING and not row["sha256"]:
                continue
            target = cache.root / path
            if row["sha256"]:
                if not target.is_file():
                    report.add("missing_file", path, unit_name)
                elif cache.sha256(path) != row["sha256"]:
                    report.add("hash_mismatch", path, unit_name)
            elif not target.exists():
                report.add("missing_path", path, unit_name)

        # Every calendar day accounted for, per house, by two independent signals.
        day_rows = {(r["date"], r["house"]): r for r in manifest.days(window.id)}
        for day in window.days:
            listed = cached_listed(cache, window, day)
            if listed is None:
                report.add("unprobed_day", day.isoformat())
                continue
            for house in HOUSES:
                in_calendar = day in calendar_sitting_days(cache, window, house)
                row = day_rows.get((day.isoformat(), house))
                if listed and row is None:
                    report.add("day_row_missing", f"{day} {house}")
                    continue
                if row is None:
                    continue
                sat = bool(row["sitting"])
                if in_calendar and not sat:
                    report.add("calendar_says_sitting_cache_empty", f"{day} {house}")
                if sat and not in_calendar:
                    report.note_gap("sitting_not_in_calendar", f"{day} {house}")
                if sat and not row["debates"]:
                    report.add("sitting_day_without_debates", f"{day} {house}")

        scan = scan_window(cache, window)
        report.scan = scan
        for rel in scan.unreadable:
            report.add("unreadable_page", rel)

        # The merged total per day must match what the manifest recorded, and the shortfall against the API's own
        # listing must be a known gap, never a silent loss.
        for day_s, merged in scan.per_day.items():
            listed = cached_listed(cache, window, date.fromisoformat(day_s)) or 0
            recorded = sum((day_rows.get((day_s, h)) or {"merged": 0})["merged"] or 0 for h in HOUSES)
            if recorded and recorded != merged:
                report.add("manifest_disagrees_with_cache", f"{day_s}: manifest {recorded}, cache {merged}")
            if listed and merged < listed:
                report.note_gap("unserved_contributions", f"{day_s}: listed {listed}, served {merged}")
            if listed and merged > listed:
                report.add("more_served_than_listed", f"{day_s}: listed {listed}, served {merged}")

        # Follow-on units: every id an index named must have been fetched.
        enabled = set(cfg.sources)
        for parent, source, directory, key, container in (
            ("commons_divisions", "commons_detail", f"commons_divisions/{window.id}", "DivisionId", None),
            ("bills", "bill", f"bills/{window.id}", "billId", "items"),
        ):
            if parent not in enabled:
                continue
            for ident in _cached_ids(cache, directory, key, container=container):
                row = manifest.unit(source, SHARED, str(ident))
                if row is None or row["status"] not in (DONE, KNOWN_GAP):
                    report.add(f"{source}_missing", str(ident))

        if "members" in enabled:
            known = set() if cfg.fetch_known_members else load_known_member_ids(cfg.known_ids_file)
            for mnis in sorted(scan.member_ids - known):
                row = manifest.unit("members", SHARED, str(mnis))
                if row is None or row["status"] not in (DONE, KNOWN_GAP):
                    report.add("member_missing", str(mnis))

        # Two official sources, counted against each other.
        if {"hansard_divisions", "commons_divisions", "lords_divisions"} <= enabled:
            report.divisions = reconcile_divisions(cache, window)
            for house, tally in report.divisions.items():
                if tally.hansard and not tally.votes:
                    # A SOURCE COVERAGE BOUNDARY, not a paging bug: the Votes API's units for this window completed (a
                    # failed or missing unit is already a problem above) and returned no division at all for the House,
                    # while Hansard lists some.
                    report.note_gap("votes_api_no_coverage",
                                    f"{house}: Hansard lists {len(tally.hansard)} division(s) in {window.id} and the "
                                    f"Votes API returned none for the whole window")
                    continue
                # No voter list exists for these at all: always a failure.
                for day, number in tally.only_hansard:
                    report.add("division_missing_from_votes_api", f"{house} {day} #{number}")
                # The division is complete; only Hansard's free debate link is lost.
                only_votes = tally.only_votes
                if len(only_votes) > tally.tolerated_votes_only():
                    report.add("divisions_missing_from_hansard_list",
                               f"{house}: {len(only_votes)} of {len(tally.votes)} are absent from Hansard's "
                               f"own list, above the {tally.tolerated_votes_only()} tolerated; "
                               f"first {only_votes[0][0]} #{only_votes[0][1]}")
                else:
                    for day, number in only_votes:
                        report.note_gap("division_not_in_hansard_list",
                                        f"{house} {day} #{number}: debate link must be inferred (I-8)")
                for h_day, v_day, number in tally.date_shifted:
                    report.note_gap("division_dated_differently",
                                    f"{house} #{number}: Hansard {h_day}, Votes API {v_day}")
                if tally.hansard_committee:
                    report.note_gap("committee_divisions",
                                    f"{house}: {tally.hansard_committee} committee division(s) are in "
                                    "Hansard and not in the Votes API, by design")
                if tally.hansard_unnumbered or tally.votes_unnumbered:
                    report.add("division_without_a_number",
                               f"{house}: hansard {tally.hansard_unnumbered}, votes {tally.votes_unnumbered}")
    finally:
        manifest.close()
    return report


def failed_units(cfg: FetchConfig, window_id: str | None = None) -> list[tuple[str, str, str]]:
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        rows = manifest.conn.execute(
            "SELECT source, window_id, key FROM unit WHERE status IN (?, ?)"
            + (" AND window_id IN (?, ?)" if window_id else ""),
            (FAILED, PENDING) + ((window_id, SHARED) if window_id else ()),
        ).fetchall()
        return [(r["source"], r["window_id"], r["key"]) for r in rows]
    finally:
        manifest.close()


def clear_failed(cfg: FetchConfig, window_id: str | None = None) -> int:
    """Delete failed and pending rows so the next `fetch` re-plans them. The cache is untouched: anything already on
    disk and hashed is still skipped.
    """
    manifest = Manifest(cfg.manifest_path)
    try:
        cur = manifest.conn.execute(
            "DELETE FROM unit WHERE status IN (?, ?)" + (" AND window_id IN (?, ?)" if window_id else ""),
            (FAILED, PENDING) + ((window_id, SHARED) if window_id else ()),
        )
        manifest.commit()
        return cur.rowcount
    finally:
        manifest.close()


# --------------------------------------------------------------------------- 13. report
# ---------------------------------------------------------------------------


def build_report(cfg: FetchConfig, window: WindowSpec) -> dict[str, Any]:
    """Everything the maintainer and the next session need to judge whether a window is usable."""
    cache = Cache(cfg.cache_root)
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        scan = scan_window(cache, window)
        counts = manifest.status_counts(window.id)
        days = manifest.days(window.id)
        sitting = {(r["date"], r["house"]) for r in days if r["sitting"]}
        fingerprints: dict[str, set[str]] = {}
        for row in manifest.units(window_id=window.id):
            if row["fingerprint"]:
                fingerprints.setdefault(row["source"], set()).add(row["fingerprint"])
        totals = manifest.totals(window.id)
        anomalies = Counter()
        for row in days:
            for a in json.loads(row["anomalies"] or "[]"):
                anomalies[a.split(":")[0]] += 1
        divisions = {
            "commons_listed": len(_cached_ids(cache, f"commons_divisions/{window.id}", "DivisionId")),
            "commons_detail_cached": len(list((cache.root / "commons_divisions" / "detail").glob("*.json.gz")))
            if (cache.root / "commons_divisions" / "detail").is_dir() else 0,
            "lords_listed": _count_list_items(cache, f"lords_divisions/{window.id}"),
            "hansard_listed": sum(_count_list_items(cache, f"hansard_divisions/{window.id}/{h}",
                                                    container="Results") for h in HOUSES),
            "reconciliation": {h: t.as_dict() for h, t in reconcile_divisions(cache, window).items()},
        }
        bills = _cached_ids(cache, f"bills/{window.id}", "billId", container="items")
        known = load_known_member_ids(cfg.known_ids_file)
        return {
            "window": {"id": window.id, "start": window.start.isoformat(), "end": window.end.isoformat(),
                       "role": window.role, "note": window.note},
            "generated_at": _now(),
            "corpus_version": cfg.version,
            "sitting_days": {h: sum(1 for (_, house) in sitting if house == h) for h in HOUSES},
            "calendar_days": {h: len(calendar_sitting_days(cache, window, h)) for h in HOUSES},
            "days_with_contributions": len(scan.per_day),
            "contributions": scan.contributions,
            "debates": len(scan.debate_ids),
            "section_census": dict(scan.sections.most_common()),
            "house_census": dict(scan.houses),
            "top_debate_headings": dict(scan.headings.most_common(25)),
            "divisions": divisions,
            "bills": {"sittings": _count_list_items(cache, f"bills/{window.id}", container="items"),
                      "distinct_bills": len(bills)},
            "members": {"seen": len(scan.member_ids), "new": len(scan.member_ids - known),
                        "already_held": len(scan.member_ids & known)},
            "estimated_chunks": round(scan.contributions * cfg.estimate.chunks_per_contribution),
            "schema_fingerprints": {k: sorted(v) for k, v in sorted(fingerprints.items())},
            "unit_status": {f"{s}/{st}": n for (s, st), n in sorted(counts.items())},
            "day_anomalies": dict(anomalies.most_common()),
            "cache_bytes": totals["bytes"],
            "unreadable_pages": scan.unreadable,
        }  # fmt: skip
    finally:
        manifest.close()


def _count_list_items(cache: Cache, directory: str, *, container: str | None = None) -> int:
    base = cache.root / directory
    if not base.is_dir():
        return 0
    n = 0
    for page in base.glob("*.json.gz"):
        try:
            payload = cache.read_json(_rel(cache, page))
        except (OSError, ValueError, EOFError, json.JSONDecodeError):
            continue
        rows = payload.get(container) if (container and isinstance(payload, dict)) else payload
        n += len(rows) if isinstance(rows, list) else 0
    return n


def write_report(cfg: FetchConfig, window: WindowSpec) -> tuple[Path, dict[str, Any]]:
    """Write the report to the gitignored reports directory and return where it went."""
    data = build_report(cfg, window)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = cfg.reports_dir / f"{window.id}-{stamp}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    return path, data


# --------------------------------------------------------------------------- 14.


def setup_logging(cfg: FetchConfig, *, console_level: int = logging.WARNING) -> Path:
    """A rotating file log plus a quiet console. Paths in the log stay relative to the repo."""
    from logging.handlers import RotatingFileHandler

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.log_dir / "corpus_fetch.log"
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    file_handler = RotatingFileHandler(path, maxBytes=8_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    file_handler.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    console.setLevel(console_level)
    log.addHandler(file_handler)
    log.addHandler(console)
    log.propagate = False
    return path
