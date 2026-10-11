"""The sealed W5 window, one definition for every guard."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG = Path(__file__).resolve().parents[2] / "config" / "sealed.toml"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SealedWindow:
    start: date
    end: date
    window_id: str
    statement: str

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


@lru_cache(maxsize=1)
def window() -> SealedWindow:
    """The window, read once from `config/sealed.toml`."""
    data = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["sealed"]
    start, end = data["from"], data["to"]
    if not (isinstance(start, date) and isinstance(end, date)) or isinstance(start, datetime) or start > end:
        raise ValueError(f"{CONFIG}: `from` and `to` must be dates, from <= to")
    return SealedWindow(start, end, str(data["window_id"]), str(data["statement"]))


# What a reader is told about the window: an honest scope answer, never a pretence that the record is silent.
SCOPE_STATEMENT = window().statement

_LIFT: contextvars.ContextVar[str | None] = contextvars.ContextVar("sealed_window_lift", default=None)


def masked() -> bool:
    """Whether the mask is on. It is, everywhere, except inside the W5 exam's own run."""
    return _LIFT.get() is None


@contextlib.contextmanager
def lifted_for_exam(reason: str) -> Iterator[None]:
    """Lift the mask for the current task only, for the W5 exam command. Logged, and always restored."""
    if not reason.strip():
        raise ValueError("a lift of the sealed window names its reason (the W5 exam run)")
    log.warning("the sealed window (%s, %s..%s) is LIFTED for: %s", window().window_id, window().start, window().end,
                reason)
    token = _LIFT.set(reason)
    try:
        yield
    finally:
        _LIFT.reset(token)


_ISO_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_PREFIX.match(value):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def is_sealed(day: Any) -> bool:
    """Whether `day` (a date, a datetime, or an ISO string) falls inside the window. Anything else is not a date."""
    d = _day(day)
    return d is not None and window().contains(d)


def hidden(day: Any) -> bool:
    """Whether a record dated `day` must be withheld right now: inside the window, with the mask on."""
    return masked() and is_sealed(day)


def overlaps(start: Any, end: Any) -> bool:
    """Whether the range [start, end] meets the window; an open end runs on forever. No dates at all is no range."""
    s, e = _day(start), _day(end)
    if s is None and e is None:
        return False
    w = window()
    return (s is None or s <= w.end) and (e is None or e >= w.start)


def inside(start: Any, end: Any) -> bool:
    """Whether the whole range [start, end] lies inside the window (both ends given)."""
    s, e = _day(start), _day(end)
    return s is not None and e is not None and window().contains(s) and window().contains(e)


def sql_exclusion(column: str) -> tuple[str, dict[str, date]]:
    """An SQL fragment to AND into a WHERE clause, and its parameters; nothing at all while the exam has lifted it.
    """
    if not masked():
        return "", {}
    w = window()
    return (f" AND ({column} < %(sealed_from)s OR {column} > %(sealed_to)s)",
            {"sealed_from": w.start, "sealed_to": w.end})


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")  # fmt: skip
_LONG = re.compile(r"\b(\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def sealed_days_in(text: str) -> list[date]:
    """Every day-level date written in `text` that falls inside the window, in order and without repeats. Months alone
    ("April to July 2026") are never a day, so the scope statement never trips it.
    """
    found: list[tuple[int, date]] = []
    for m in _LONG.finditer(text):
        found.append((m.start(), (int(m.group(3)), _MONTHS.index(m.group(2)) + 1, int(m.group(1)))))  # type: ignore[arg-type]
    for m in _ISO.finditer(text):
        found.append((m.start(), (int(m.group(1)), int(m.group(2)), int(m.group(3)))))  # type: ignore[arg-type]
    out: list[date] = []
    for _, ymd in sorted(found, key=lambda pair: pair[0]):
        try:
            day = date(*ymd)  # type: ignore[misc]
        except ValueError:
            continue
        if window().contains(day) and day not in out:
            out.append(day)
    return out


def served_label(corpus_version: str) -> str:
    """The corpus as served: masked answers and unmasked ones must never share a cache entry or a trace label."""
    if not masked():
        return corpus_version
    w = window()
    return f"{corpus_version}+sealed:{w.start.isoformat()}..{w.end.isoformat()}"
