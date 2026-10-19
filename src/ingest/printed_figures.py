"""Hansard's printed division figures, loaded beside the Votes API's."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from app import sealed

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "corpus_v2" / "hansard_divisions"

Key = tuple[str, date, str]


def parse(results: Iterable[Mapping[str, Any]]) -> dict[Key, tuple[int, int]]:
    """Each record's printed figures by (House, sitting day, division number)."""
    seen: dict[Key, set[tuple[int, int]]] = {}
    for r in results:
        when = str(r.get("Date") or "")[:10]
        if not when or sealed.hidden(when) or r.get("AyesCount") is None or r.get("NoesCount") is None:
            continue
        key = (str(r.get("House") or ""), date.fromisoformat(when), str(r.get("Number") or "").strip())
        seen.setdefault(key, set()).add((int(r["AyesCount"]), int(r["NoesCount"])))
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def load(raw: Path = RAW) -> dict[Key, tuple[int, int]]:
    """Every window's records but the sealed one's."""
    if not raw.exists():
        return {}
    sealed_id = sealed.window().window_id if sealed.masked() else None
    results: list[Mapping[str, Any]] = []
    for window in sorted(p for p in raw.iterdir() if p.is_dir() and p.name != sealed_id):
        for index in sorted(window.glob("*/index_*.json.gz")):
            with gzip.open(index, "rt", encoding="utf-8") as fh:
                results += json.load(fh).get("Results") or []
    return parse(results)


def apply(conn, figures: Mapping[Key, tuple[int, int]]) -> dict[str, int]:
    """Write the figures to the divisions they belong to (a synchronous connection). Returns how many keys matched a
    division, how many did not, how many matched one whose Votes API figures differ, and how many rows changed.
    """
    keys = list(figures)
    if not keys:
        return {"matched": 0, "unmatched": 0, "differ": 0, "written": 0}
    rows = conn.execute(
        "SELECT u.house, u.day, u.num, u.a, u.n, d.id, d.ayes, d.noes "
        "FROM unnest(%s::text[], %s::date[], %s::text[], %s::int[], %s::int[]) AS u(house, day, num, a, n) "
        "JOIN division d ON d.house = u.house AND d.division_date = u.day AND d.number::text = u.num",
        ([k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys], [figures[k][0] for k in keys],
         [figures[k][1] for k in keys])).fetchall()  # fmt: skip
    matched = {(h, day, num) for h, day, num, *_ in rows}
    differ = sum(1 for *_, a, n, _id, ayes, noes in rows if (a, n) != (ayes, noes))
    written = conn.execute(
        "UPDATE division d SET printed_ayes = u.a, printed_noes = u.n "
        "FROM unnest(%s::int[], %s::int[], %s::int[]) AS u(id, a, n) "
        "WHERE d.id = u.id AND (d.printed_ayes, d.printed_noes) IS DISTINCT FROM (u.a, u.n)",
        ([r[5] for r in rows], [r[3] for r in rows], [r[4] for r in rows])).rowcount  # fmt: skip
    return {"matched": len(matched), "unmatched": len(keys) - len(matched), "differ": differ, "written": written}
