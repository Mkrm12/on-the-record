"""Turn the authored seeds into gold items, resolving every quote against the database."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from goldset.adversarial_items import ADVERSARIAL
from goldset.human_items import HUMAN
from goldset.schema import CATEGORIES, Evidence, GoldItem, Nugget

# See `dbgen.HELDOUT_WINDOWS`: W3 is burned and W5 is the sealed exam.
HELDOUT_WINDOWS = frozenset({"W3", "W5"})
HELDOUT_WINDOW = "W3"  # kept as an alias; prefer HELDOUT_WINDOWS


class SeedError(ValueError):
    """A seed that cannot be turned into an item. Always a fault in the seed, never in the data."""


@dataclass
class Unresolved:
    seed_id: str
    speech_id: str
    anchor: str
    reason: str


def _fold(text: str) -> str:
    """Compare on the fold, not the glyphs: NFKC, ASCII quotes and dashes, collapsed space, lower case."""
    text = unicodedata.normalize("NFKC", text)
    for fancy, plain in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                         ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")):  # fmt: skip
        text = text.replace(fancy, plain)
    return " ".join(text.split()).lower()


async def _speech_context(conn: AsyncConnection, speech_ids: list[str]) -> dict[str, dict]:
    if not speech_ids:
        return {}
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT s.id, s.window_id, d.house, d.section, d.date, s.speaker_name, s.mnis_id
              FROM speech s LEFT JOIN debate d ON d.id = s.debate_id
             WHERE s.id = ANY(%(ids)s)
            """,
            {"ids": speech_ids},
        )
        return {r["id"]: r for r in await cur.fetchall()}


async def _sentences(conn: AsyncConnection, speech_ids: list[str]) -> dict[str, list[str]]:
    """Every chunk body of each speech, in order. These are the exact strings the index holds."""
    if not speech_ids:
        return {}
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT speech_id, sent_idx, body FROM chunk WHERE speech_id = ANY(%(ids)s) ORDER BY speech_id, sent_idx",
            {"ids": speech_ids},
        )
        out: dict[str, list[str]] = {}
        for row in await cur.fetchall():
            out.setdefault(row["speech_id"], []).append(row["body"])
        return out


def _resolve(anchor: str, bodies: list[str]) -> str | None:
    """The sentence containing the anchor, or two consecutive ones when it straddles a boundary."""
    needle = _fold(anchor)
    for body in bodies:
        if needle in _fold(body):
            return body
    for first, second in zip(bodies, bodies[1:]):
        if needle in _fold(f"{first} {second}"):
            return f"{first} {second}"
    return None


ADVERSARIAL_BEHAVIOURS = frozenset({
    "correct_false_premise", "abstain_not_in_record", "abstain_out_of_window",
    "clarify_ambiguous", "neutralise_loaded_framing",
})  # fmt: skip


def _prefix(behaviour: str) -> str:
    """`hum` for the human-meaningful half, `adv` for the adversarial one, so an id says what it is."""
    return "adv" if behaviour in ADVERSARIAL_BEHAVIOURS else "hum"


async def build(conn: AsyncConnection, seeds: list[dict[str, Any]] | None = None) -> tuple[list[GoldItem], list[Unresolved]]:
    """Every authored item, with quotes and metadata taken from the database."""
    seeds = HUMAN + ADVERSARIAL if seeds is None else seeds
    speech_ids = sorted({sid for seed in seeds for sid, _ in seed.get("ev", [])})
    context = await _speech_context(conn, speech_ids)
    bodies = await _sentences(conn, speech_ids)

    counters: dict[str, int] = {}
    items: list[GoldItem] = []
    unresolved: list[Unresolved] = []

    for seed in seeds:
        category = seed["cat"]
        if category not in CATEGORIES:
            raise SeedError(f"unknown category {category!r} in seed {seed['q'][:40]!r}")
        behaviour = seed.get("behaviour") or CATEGORIES[category]
        if behaviour is None:
            raise SeedError(f"category {category} needs an explicit behaviour: {seed['q'][:40]!r}")

        prefix = _prefix(behaviour)
        counters[prefix] = counters.get(prefix, 0) + 1
        item_id = f"{prefix}{counters[prefix]:04d}"

        evidence: list[Evidence] = []
        for speech_id, anchor in seed.get("ev", []):
            if speech_id not in context:
                unresolved.append(Unresolved(item_id, speech_id, anchor, "speech not in the corpus"))
                continue
            quote = _resolve(anchor, bodies.get(speech_id, []))
            if quote is None:
                unresolved.append(Unresolved(item_id, speech_id, anchor, "anchor not found in the speech"))
                continue
            evidence.append(Evidence(speech_id, quote))

        # The window and house are read off the evidence, never asserted by the seed.
        windows = {context[e.speech_id]["window_id"] for e in evidence if context.get(e.speech_id)}
        if len(windows) > 1 and windows & HELDOUT_WINDOWS:
            raise SeedError(
                f"{item_id}: evidence mixes a held-out window with {sorted(windows - HELDOUT_WINDOWS)}"
            )
        first = context.get(evidence[0].speech_id) if evidence else None

        window = (first or {}).get("window_id") or "" if len(windows) <= 1 else ""
        spans = sorted(windows) if len(windows) > 1 else []
        items.append(
            GoldItem(
                id=item_id,
                question=seed["q"],
                category=category,
                criticality=seed["crit"],
                expected_behaviour=behaviour,
                provenance=seed.get("prov", "passage_llm"),
                window=window,
                split="heldout" if window in HELDOUT_WINDOWS else "",
                house=(first or {}).get("house") or "" if not spans else "",
                section=(first or {}).get("section") or "" if not spans else "",
                date=str((first or {}).get("date") or "") if not spans else "",
                subcategory=seed.get("sub") or ("spans_" + "_".join(spans) if spans else ""),
                evidence=tuple(evidence),
                nuggets=tuple(Nugget(text, vital) for text, vital in seed.get("nug", [])),
                forbidden_claims=tuple(seed.get("forbid", [])),
                notes=seed.get("note", ""),
            )
        )
    return items, unresolved
