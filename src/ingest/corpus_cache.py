"""Read-only access to the Corpus v2 fetch cache."""

from __future__ import annotations

import gzip
import json
import pathlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from ingest.corpus_fetch import (
    ORDERS,
    WindowSpec,
    bill_path,
    calendar_path,
    contributions_path,
    corrections_path,
    member_path,
    probe_path,
    sections_path,
)

# Member endpoint key -> the URL suffix the fetch tool saved it under.
MEMBER_ENDPOINT_KEYS = {
    "": "profile", "Biography": "biography", "Contact": "contact", "Synopsis": "synopsis",
    "Experience": "experience", "Focus": "focus", "Staff": "staff",
    "RegisteredInterests": "interests", "LatestElectionResult": "election",
}  # fmt: skip

# A cache path is a plain relative name: letters, digits, dots, dashes, underscores and forward slashes, with no segment
# that is "..".
_SAFE = re.compile(r"^[A-Za-z0-9_.\-/]+$")


def _inside_root(rel: str) -> bool:
    return bool(_SAFE.match(rel)) and ".." not in rel.split("/") and not rel.startswith("/")


def _member_absences(manifest: Path) -> dict[int, frozenset[str]]:
    """mnis_id -> the endpoint suffixes the fetch recorded as a 404, read-only from the fetch manifest."""
    import sqlite3

    if not manifest.is_file():
        return {}
    con = sqlite3.connect(f"{manifest.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT key, last_error FROM unit WHERE source = 'members'").fetchall()
    finally:
        con.close()
    out: dict[int, frozenset[str]] = {}
    for key, recorded in rows:
        if not recorded or not str(key).isdigit():
            continue
        absent = {token[: -len(":404")] for token in str(recorded).split(";") if token.endswith(":404")}
        if absent:
            out[int(key)] = frozenset("" if suffix == "profile" else suffix for suffix in absent)
    return out


class CacheError(Exception):
    """A cached file is missing or unreadable. Never raised for 'this window has none'."""


class CacheMiss(CacheError):
    """The file is not in the cache at all. The only failure `_maybe` treats as absence: an unreadable file (truncated,
    not gzip, not JSON) is damage, and a load that read it as "none" would drop real data without a word.
    """


@dataclass(frozen=True)
class DayItems:
    """One sitting day, merged across both sort orders exactly as the fetch merged it."""

    day: date
    items: dict[str, dict]  # ContributionExtId -> the API item
    listed: int | None = None  # TotalResultCount at fetch time
    pages: int = 0
    # Contributions the API served more than once on the same day under DIFFERENT debate ids: id -> the debate ids seen,
    # first (the one kept) first.
    debate_conflicts: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def unserved(self) -> int:
        """Contributions the API listed but would not serve. A property of the source."""
        return max((self.listed or 0) - len(self.items), 0)


@dataclass
class CorpusCache:
    """Everything the loader needs, addressed by window. Offline by construction."""

    root: Path
    _missing: list[str] = field(default_factory=list, repr=False)
    _absent: dict[int, frozenset[str]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # -- primitives ---------------------------------------------------------

    def _read(self, rel: str) -> Any:
        if not _inside_root(rel):
            raise CacheError(f"refusing a cache path that is not a plain relative name: {rel!r}")
        path = self.root / rel
        try:
            with gzip.open(path, "rb") as fh:
                return json.loads(fh.read().decode("utf-8"))
        except FileNotFoundError as exc:
            raise CacheMiss(f"not in the cache: {rel}") from exc
        except (OSError, ValueError, EOFError) as exc:
            raise CacheError(f"unreadable cache file {rel}: {type(exc).__name__}") from exc

    def _maybe(self, rel: str) -> Any | None:
        """The payload, or None when the file is not in the cache. An UNREADABLE file raises: it used to come back as
        None too, so a corrupt calendar month read as "no sitting days" and a corrupt bill as "no such bill".
        """
        try:
            return self._read(rel)
        except CacheMiss:
            return None

    def _pages(self, directory: str, *, container: str | None = None) -> Iterator[dict]:
        base = self.root / directory
        if not base.is_dir():
            return
        for page in sorted(base.glob("*.json.gz")):
            payload = self._read(self._rel(page))
            rows = payload.get(container) if (container and isinstance(payload, dict)) else payload
            for row in rows or []:
                if isinstance(row, dict):
                    yield row

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.root)).replace("\\", "/")

    def exists(self, rel: str) -> bool:
        return (self.root / rel).is_file()

    # -- contributions ------------------------------------------------------

    def days(self, window: WindowSpec) -> list[date]:
        """Days this window actually has contribution pages for, in order."""
        base = self.root / "contributions" / window.id
        if not base.is_dir():
            return []
        out = []
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            try:
                day = date.fromisoformat(child.name)
            except ValueError:
                continue
            if window.start <= day <= window.end:
                out.append(day)
        return out

    def day_items(self, window: WindowSpec, day: date) -> DayItems:
        """One day's contributions, merged by ContributionExtId with the FIRST sort order winning -- the same rule the
        fetch used, so the merge is not re-decided here.
        """
        base = self.root / "contributions" / window.id / day.isoformat()
        merged: dict[str, dict] = {}
        listed: int | None = None
        pages = 0
        seen_debates: dict[str, list[str]] = {}
        for order in ORDERS:
            prefix = pathlib.PurePosixPath(contributions_path(window.id, day, order, 0)).name.split("_")[0]
            for page in sorted(base.glob(f"{prefix}_*.json.gz")) if base.is_dir() else []:
                payload = self._read(self._rel(page))
                if not isinstance(payload, dict):
                    continue
                pages += 1
                if listed is None and isinstance(payload.get("TotalResultCount"), int):
                    listed = payload["TotalResultCount"]
                for item in payload.get("Results") or []:
                    cid = item.get("ContributionExtId") if isinstance(item, dict) else None
                    if not cid:
                        continue
                    merged.setdefault(cid, item)
                    debate_id = item.get("DebateSectionExtId")
                    if debate_id and debate_id not in seen_debates.setdefault(cid, []):
                        seen_debates[cid].append(debate_id)
        conflicts = {cid: tuple(ids) for cid, ids in seen_debates.items() if len(ids) > 1}
        return DayItems(day=day, items=merged, listed=listed, pages=pages, debate_conflicts=conflicts)

    def contributions(self, window: WindowSpec) -> Iterator[DayItems]:
        for day in self.days(window):
            yield self.day_items(window, day)

    def corrections(self, window: WindowSpec) -> Iterator[tuple[date, dict]]:
        base = (self.root / corrections_path(window.id, window.start)).parent
        if not base.is_dir():
            return
        for page in sorted(base.glob("*.json.gz")):
            try:
                day = date.fromisoformat(page.name.split(".")[0])
            except ValueError:
                continue
            payload = self._read(self._rel(page))
            for item in (payload.get("Results") or []) if isinstance(payload, dict) else []:
                if isinstance(item, dict):
                    yield day, item

    # -- divisions ----------------------------------------------------------

    def commons_divisions(self, window: WindowSpec) -> list[dict]:
        return list(self._pages(f"commons_divisions/{window.id}"))

    def commons_detail(self, division_id: int) -> dict | None:
        payload = self._maybe(f"commons_divisions/detail/{division_id}.json.gz")
        return payload if isinstance(payload, dict) else None

    def lords_divisions(self, window: WindowSpec) -> list[dict]:
        return list(self._pages(f"lords_divisions/{window.id}"))

    def hansard_divisions(self, window: WindowSpec, house: str | None = None) -> list[dict]:
        houses = [house] if house else ["Commons", "Lords"]
        return [r for h in houses
                for r in self._pages(f"hansard_divisions/{window.id}/{h}", container="Results")]

    # -- bills --------------------------------------------------------------

    def bill_sittings(self, window: WindowSpec) -> list[dict]:
        return list(self._pages(f"bills/{window.id}", container="items"))

    def bill(self, bill_id: int) -> tuple[dict | None, dict | None]:
        """(detail, stages) for one bill; either may be None where the API served a 404."""
        detail = self._maybe(bill_path(bill_id, "detail"))
        stages = self._maybe(bill_path(bill_id, "stages"))
        return (detail if isinstance(detail, dict) else None,
                stages if isinstance(stages, dict) else None)

    # -- members ------------------------------------------------------------

    def member_ids(self) -> list[int]:
        base = self.root / "members"
        if not base.is_dir():
            return []
        out = []
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name.isdigit():
                out.append(int(child.name))
        return sorted(out)

    def member(self, mnis_id: int) -> tuple[dict[str, Any], list[str]]:
        """(payloads keyed as `members.parse_member` wants them, endpoints that FAILED or were never fetched)."""
        absent = self.member_absent(mnis_id)
        payloads: dict[str, Any] = {}
        missing: list[str] = []
        for suffix, key in MEMBER_ENDPOINT_KEYS.items():
            payload = self._maybe(member_path(mnis_id, suffix))
            if payload is not None:
                payloads[key] = payload
            elif suffix not in absent:
                missing.append(key)
        return payloads, missing

    def member_absent(self, mnis_id: int) -> frozenset[str]:
        """The endpoint suffixes ('' for the profile) that the fetch found ABSENT -- HTTP 404 -- for this member."""
        if self._absent is None:
            self._absent = _member_absences(self.root / "manifest.sqlite")
        return self._absent.get(int(mnis_id), frozenset())

    # -- sections and calendar (reporting only) -----------------------------

    def sections_for_day(self, window: WindowSpec, house: str, day: date) -> list[str] | None:
        payload = self._maybe(sections_path(window.id, house, day))
        return payload if isinstance(payload, list) else None

    def calendar_days(self, window: WindowSpec, house: str) -> set[date]:
        days: set[date] = set()
        for year, month in window.months:
            payload = self._maybe(calendar_path(window.id, house, year, month))
            for item in payload or []:
                if isinstance(item, dict) and isinstance(item.get("ItemDate"), str):
                    try:
                        day = date.fromisoformat(item["ItemDate"][:10])
                    except ValueError:
                        continue
                    if window.start <= day <= window.end:
                        days.add(day)
        return days

    def probe_listed(self, window: WindowSpec, day: date) -> int | None:
        payload = self._maybe(probe_path(window.id, day))
        if isinstance(payload, dict) and isinstance(payload.get("TotalResultCount"), int):
            return payload["TotalResultCount"]
        return None
