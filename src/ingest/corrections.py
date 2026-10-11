"""Link a written correction to the speech it corrects."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from typing import Any, NamedTuple

# The share of the correction's shingles the chosen speech must contain.
MIN_SCORE = 0.5
# How far the best must beat the runner-up.
MIN_MARGIN = 0.2
# How far back to look.
LOOKBACK_DAYS = 120
# Shingle length in words.
SHINGLE = 5

_TAG = re.compile(r"<[^>]+>")
_INSERTED = re.compile(r"<strong\b[^>]*>.*?</strong>", re.I | re.S)
_STRUCK = re.compile(r"<s\b[^>]*>.*?</s>", re.I | re.S)
_STRUCK_TAGS = re.compile(r"</?s\b[^>]*>", re.I)
_INSERTED_TAGS = re.compile(r"</?strong\b[^>]*>", re.I)
_WORD = re.compile(r"[a-z0-9]+")


class Candidate(NamedTuple):
    speech_id: str
    date: date
    text: str


class Link(NamedTuple):
    speech_id: str | None
    basis: str  # shingle | ambiguous | weak | no_candidate | empty_body
    score: float = 0.0
    runner_up: float = 0.0


def original_text(body: str | None) -> str:
    """The wording as the member first said it: inserted words dropped, struck words kept."""
    text = _INSERTED.sub(" ", body or "")
    text = _STRUCK_TAGS.sub(" ", text)
    return " ".join(_TAG.sub(" ", text).split())


def corrected_text(body: str | None) -> str:
    """The wording as corrected: struck words dropped, inserted words kept."""
    text = _STRUCK.sub(" ", body or "")
    text = _INSERTED_TAGS.sub(" ", text)
    return " ".join(_TAG.sub(" ", text).split())


def shingles(text: str | None, n: int = SHINGLE) -> set[tuple[str, ...]]:
    words = _WORD.findall((text or "").lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def link_correction(body: str | None, candidates: list[Candidate]) -> Link:
    """The speech this correction corrects, or None and why."""
    target = shingles(original_text(body)) | shingles(corrected_text(body))
    if not target:
        return Link(None, "empty_body")
    if not candidates:
        return Link(None, "no_candidate")
    scored = sorted(((len(target & shingles(c.text)) / len(target), c.speech_id) for c in candidates), reverse=True)
    best, speech_id = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if best < MIN_SCORE:
        return Link(None, "weak", best, runner)
    if best - runner < MIN_MARGIN:
        return Link(None, "ambiguous", best, runner)
    return Link(speech_id, "shingle", best, runner)


SQL_CORRECTIONS = "SELECT id, mnis_id, date, house, body, corrected_speech_id, link_basis FROM hansard_correction"
SQL_CANDIDATES = """
    SELECT s.id, d.date, s.text FROM speech s JOIN debate d ON d.id = s.debate_id
     WHERE s.mnis_id = %s AND d.house = %s AND d.date BETWEEN %s::date - %s AND %s::date AND NOT s.is_boilerplate
"""
SQL_WRITE = """
    UPDATE hansard_correction h SET corrected_speech_id = u.speech_id, link_basis = u.basis
      FROM unnest(%s::text[], %s::text[], %s::text[]) AS u(id, speech_id, basis)
     WHERE h.id = u.id AND (h.corrected_speech_id, h.link_basis) IS DISTINCT FROM (u.speech_id, u.basis)
"""


async def link_all(conn: Any) -> dict[str, Any]:
    """Every correction, relinked from the database. Offline, idempotent; the caller owns the commit (see
    `ingest.bills.link_all` for why that contract is spelled out).
    """
    rows = await (await conn.execute(SQL_CORRECTIONS)).fetchall()
    ids, speeches, bases, counts = [], [], [], Counter()
    for cid, mnis, day, house, body, _old_speech, _old_basis in rows:
        candidates = []
        if mnis is not None and day is not None and house:
            found = await (await conn.execute(SQL_CANDIDATES, (mnis, house, day, LOOKBACK_DAYS, day))).fetchall()
            candidates = [Candidate(sid, d, text or "") for sid, d, text in found]
        link = link_correction(body, candidates)
        ids.append(str(cid))
        speeches.append(link.speech_id)
        # The score goes into the basis so a reviewer can see how close a call it was without re-running anything.
        bases.append(f"{link.basis}:{link.score:.2f}" if link.basis in ("shingle", "weak", "ambiguous") else link.basis)
        counts[link.basis] += 1
    async with conn.transaction():
        changed = (await conn.execute(SQL_WRITE, (ids, speeches, bases))).rowcount
    return {"corrections": len(rows), "outcomes": dict(counts), "changed": changed}
