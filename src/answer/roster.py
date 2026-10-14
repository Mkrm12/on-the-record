"""Who spoke: rosters, first speakers, and identifying the debate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

from answer.contract import Roster, RosterEntry

# Below this, `pick_debate` returns clarify instead of an answer.
DEBATE_CONFIDENCE_THRESHOLD = 0.55

# How many candidates a clarify offers. says three.
CLARIFY_CANDIDATES = 3

# Boilerplate is kept when asking who spoke first, and the measurement says this choice does not change the score rather
# than that it rescues it.
FIRST_SPEAKER_INCLUDES_BOILERPLATE = True

# Who presides, as the record shows it: the Chair's roles, and its printed forms for the presiding Chairmen the role
# classifier tags `office_holder`.
PRESIDING_ROLES = ("speaker", "deputy_speaker", "chair", "lord_speaker", "lords_deputy")
CHAIR_PRINTED = r"^The (?:(?:First |Second )?Deputy )?Chairman(?: of Ways and Means)?(?: \(|$)|^The (?:Temporary )?Chair(?: \(|$)"


@dataclass
class DebateCandidate:
    """One debate retrieval thinks the question might mean."""

    debate_id: str
    score: float
    title: str | None = None
    date: Date | None = None
    house: str | None = None


@dataclass
class Choice:
    """Either one debate, or a request to clarify between several. Never a silent best guess."""

    debate_id: str | None = None
    confidence: float = 0.0
    clarify: tuple[DebateCandidate, ...] = ()
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.debate_id is not None

    @property
    def needs_clarification(self) -> bool:
        return not self.ok and bool(self.clarify)


def confidence(candidates: list[DebateCandidate]) -> float:
    """How sure we are of the top debate: its share of the top two scores."""
    if not candidates:
        return 0.0
    ordered = sorted(candidates, key=lambda c: -c.score)
    if len(ordered) == 1:
        return 1.0
    top, second = ordered[0].score, ordered[1].score
    total = top + second
    if total <= 0:
        return 0.0
    return top / total


def pick_debate(candidates: list[DebateCandidate],
                threshold: float = DEBATE_CONFIDENCE_THRESHOLD) -> Choice:  # fmt: skip
    """One debate, or clarify with the top three. Never the best guess when the margin is thin."""
    if not candidates:
        return Choice(reason="retrieval returned no debate for this question")
    ordered = sorted(candidates, key=lambda c: (-c.score, c.debate_id))
    margin = confidence(ordered)
    if margin >= threshold:
        return Choice(debate_id=ordered[0].debate_id, confidence=margin)
    return Choice(
        confidence=margin,
        clarify=tuple(ordered[:CLARIFY_CANDIDATES]),
        reason=(f"the best match leads the runner-up by too little ({margin:.2f} < {threshold:.2f}) to name one "
                f"debate; {min(len(ordered), CLARIFY_CANDIDATES)} candidates are offered instead"),
    )  # fmt: skip


# --- SQL ------------------------------------------------------------------------------------------------


async def _all(conn, sql: str, params: tuple | dict = ()) -> list[dict]:
    cur = await conn.execute(sql, params)
    columns = [c.name for c in cur.description or []]
    return [dict(zip(columns, row, strict=True)) for row in await cur.fetchall()]


# Party and office as at the debate's own date, from the date-aware tables.
_AT_DATE = """
    LEFT JOIN LATERAL (
        SELECT h.party FROM member_party_history h
        WHERE h.mnis_id = sp.mnis_id
          AND h.start_date <= d.date AND (h.end_date IS NULL OR h.end_date >= d.date)
        ORDER BY h.start_date DESC LIMIT 1
    ) party_then ON TRUE
    LEFT JOIN LATERAL (
        SELECT o.office FROM member_office o
        WHERE o.mnis_id = sp.mnis_id
          AND o.start_date <= d.date AND (o.end_date IS NULL OR o.end_date >= d.date)
        ORDER BY o.start_date DESC LIMIT 1
    ) office_then ON TRUE
"""


async def first_speaker(conn, debate_id: str,
                        include_boilerplate: bool = FIRST_SPEAKER_INCLUDES_BOILERPLATE) -> dict[str, Any]:
    """Who opened a debate, with their party and office as at that date."""
    rows = await _all(conn, f"""
        SELECT sp.id AS speech_id, sp.seq, sp.speaker_name, sp.mnis_id, sp.is_boilerplate,
               -- the members who spoke, the Chair not among them (D-001: "N members spoke")
               (SELECT count(DISTINCT s2.mnis_id) FROM speech s2
                 WHERE s2.debate_id = sp.debate_id AND s2.mnis_id IS NOT NULL AND NOT s2.is_boilerplate
                   AND NOT (coalesce(s2.speaker_role, '') = ANY(%s) OR coalesce(s2.speaker_name, '') ~ %s)) AS members,
               sp.speaker_party AS party_on_the_day, party_then.party AS party_at_date,
               office_then.office AS office_at_date, d.date, d.house, d.heading, d.section,
               -- The speaker as a card names them (N-01, migration 021): the member's name, or the name printed at
               -- the time for a member in another House now; Hansard's label ("Mr Speaker", "Lincoln Jopp
               -- (Spelthorne) (Con)") where the speech resolves to no member.
               speaker_as_printed(sp.id) AS display_name
        FROM speech sp
        JOIN debate d ON d.id = sp.debate_id
        {_AT_DATE}
        WHERE sp.debate_id = %s AND (%s OR NOT sp.is_boilerplate)
          AND NOT (coalesce(sp.speaker_role, '') = ANY(%s) OR coalesce(sp.speaker_name, '') ~ %s)
        ORDER BY sp.seq
        LIMIT 1
    """, (list(PRESIDING_ROLES), CHAIR_PRINTED, debate_id, include_boilerplate, list(PRESIDING_ROLES), CHAIR_PRINTED))
    if not rows:
        return {"missing": f"no member's speech is recorded for debate {debate_id!r} (only the Chair's, or none)"}
    row = rows[0]
    # Whatever Hansard printed on the day wins over a biography interval: the printed party is evidence about this
    # sitting, and the interval is evidence about a period that may not cover it cleanly.
    row["party"] = row.get("party_on_the_day") or row.get("party_at_date")
    return row


async def roster(conn, debate_ids: list[str] | tuple[str, ...], confidence_score: float = 1.0,
                 query: str | None = None, limit: int = 200) -> Roster:  # fmt: skip
    """Everyone who spoke in these debates, in speaking order, with party and office at the date."""
    if not debate_ids:
        raise ValueError("a roster needs at least one debate id")
    ids = list(debate_ids)

    people = await _all(conn, f"""
        SELECT sp.mnis_id,
               -- The mapped name where the id resolves, the printed one otherwise. Hansard prints
               -- "Nick Thomas-Symonds (Torfaen) (Lab)" — constituency and party inside the name string — and a
               -- roster that shows that has put three facts in the field reserved for one, with the party then
               -- appearing twice and possibly disagreeing with `party_at_date`. A member in another House now is
               -- named as printed then (N-01, migration 021), never by the later title.
               speaker_as_printed(min(sp.id)) AS speaker_name,
               min(sp.speaker_name) AS name_as_printed,
               max(sp.speaker_party) AS party_on_the_day,
               max(party_then.party) AS party_at_date,
               max(office_then.office) AS office_at_date,
               count(*) AS contributions,
               min(sp.seq) AS first_seq
        FROM speech sp
        JOIN debate d ON d.id = sp.debate_id
        {_AT_DATE}
        WHERE sp.debate_id = ANY(%s) AND NOT sp.is_boilerplate AND sp.speaker_name IS NOT NULL
        GROUP BY sp.mnis_id, CASE WHEN sp.mnis_id IS NULL THEN sp.speaker_name ELSE '' END
        ORDER BY min(sp.seq)
        LIMIT %s
    """, (ids, limit))

    best: dict[Any, int] = {}
    if query:
        # `websearch_to_tsquery` rather than `to_tsquery`: it never raises on ordinary prose, and a roster that threw on
        # an apostrophe would be a roster nobody could use.
        for row in await _all(conn, """
            SELECT DISTINCT ON (c.mnis_id) c.mnis_id, c.id AS chunk_id
            FROM chunk c
            WHERE c.debate_id = ANY(%s) AND NOT c.is_boilerplate
              AND c.ts @@ websearch_to_tsquery('english', %s)
            ORDER BY c.mnis_id, ts_rank(c.ts, websearch_to_tsquery('english', %s)) DESC, c.id
        """, (ids, query, query)):
            best[row["mnis_id"]] = row["chunk_id"]
    # Fill the gaps per speaker, not only when nobody matched.
    for row in await _all(conn, """
        SELECT DISTINCT ON (c.mnis_id) c.mnis_id, c.id AS chunk_id
        FROM chunk c
        WHERE c.debate_id = ANY(%s) AND NOT c.is_boilerplate
        ORDER BY c.mnis_id, c.seq, c.sent_idx, c.id
    """, (ids,)):
        best.setdefault(row["mnis_id"], row["chunk_id"])

    entries = tuple(
        RosterEntry(
            mnis_id=row["mnis_id"],
            speaker_name=row["speaker_name"],
            party_at_date=row.get("party_on_the_day") or row.get("party_at_date"),
            office_at_date=row.get("office_at_date"),
            contributions=row["contributions"],
            best_chunk_id=best.get(row["mnis_id"]),
        )
        for row in people
    )
    return Roster(debate_ids=tuple(ids), confidence=confidence_score, entries=entries)


async def roster_with_gaps(conn, debate_ids: list[str] | tuple[str, ...], **kwargs) -> dict[str, Any]:
    """A roster plus what it could not resolve."""
    built = await roster(conn, debate_ids, **kwargs)
    rows = await _all(conn, """
        SELECT count(*) AS unattributed, count(DISTINCT speaker_name) AS unattributed_names
        FROM speech
        WHERE debate_id = ANY(%s) AND NOT is_boilerplate
          AND mnis_id IS NULL AND speaker_name IS NOT NULL
    """, (list(debate_ids),))
    gap = rows[0] if rows else {"unattributed": 0, "unattributed_names": 0}
    return {
        "roster": built,
        "unattributed": gap["unattributed"],
        "unattributed_names": gap["unattributed_names"],
        "complete": gap["unattributed"] == 0,
        "caveat": ("" if gap["unattributed"] == 0 else
                   f"{gap['unattributed_names']} speaker(s) in this debate could not be matched to a member "
                   f"record, so the list may be incomplete."),  # fmt: skip
    }


# Curly and straight apostrophes are one character to a reader and two to a text parser; both sides are folded.
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'"})


async def candidates_in_question(conn, question: str, house: str | None = None, when: Date | None = None,
                                 limit: int = 10) -> list[DebateCandidate]:  # fmt: skip
    """Debates whose heading the QUESTION contains — every lexeme of the heading present in the question — the most
    specific heading first.
    """
    # The apostrophe characters travel as parameters, not SQL literals: nothing to quote, so nothing to get wrong.
    rows = await _all(conn, """
        SELECT d.id AS debate_id, d.heading, d.date, d.house,
               numnode(plainto_tsquery('english', translate(d.heading, %(curly)s, %(straight)s))) AS score
        FROM debate d
        WHERE (%(house)s::text IS NULL OR d.house = %(house)s)
          AND (%(when)s::date IS NULL OR d.date = %(when)s)
          AND d.heading IS NOT NULL
          AND numnode(plainto_tsquery('english', translate(d.heading, %(curly)s, %(straight)s))) > 0
          AND to_tsvector('english', %(question)s)
              @@ plainto_tsquery('english', translate(d.heading, %(curly)s, %(straight)s))
        ORDER BY score DESC, d.date DESC, d.id
        LIMIT %(limit)s
    """, {"question": question.translate(_APOSTROPHES), "house": house, "when": when, "limit": limit,
          "curly": "\u2019\u2018", "straight": "''"})
    # A heading the question contains VERBATIM (order and all) is stronger evidence than one whose words it contains:
    # "Local Government Reorganisation in Devon" and "\u2026: Devon" have the same lexemes, and the question named the
    # first.
    asked = f" {_heading_text(question)} "
    found = []
    for r in rows:
        score = float(r["score"] or 0.0)
        if (title := _heading_text(r["heading"])) and f" {title} " in asked:
            score *= VERBATIM_WEIGHT
        found.append(DebateCandidate(debate_id=r["debate_id"], score=score, title=r["heading"], date=r["date"],
                                     house=r["house"]))  # fmt: skip
    return sorted(found, key=lambda c: (-c.score, c.debate_id))


# How much a verbatim heading outweighs a merely lexeme-contained one (`candidates_in_question`).
VERBATIM_WEIGHT = 2.0


def _heading_text(text: str | None) -> str:
    """Casefolded, apostrophes straightened, punctuation to spaces, whitespace collapsed."""
    folded = (text or "").translate(_APOSTROPHES).casefold()
    return " ".join(re.sub(r"[^\w\s']", " ", folded).split())


async def candidates_by_heading(conn, heading: str, house: str | None = None, when: Date | None = None,
                                limit: int = 10) -> list[DebateCandidate]:  # fmt: skip
    """Debates whose heading matches, ranked by full-text similarity."""
    rows = await _all(conn, """
        SELECT d.id AS debate_id, d.heading, d.date, d.house,
               ts_rank(to_tsvector('english', d.heading), websearch_to_tsquery('english', %s)) AS score
        FROM debate d
        WHERE to_tsvector('english', d.heading) @@ websearch_to_tsquery('english', %s)
          AND (%s::text IS NULL OR d.house = %s)
          AND (%s::date IS NULL OR d.date = %s)
        ORDER BY score DESC, d.date DESC
        LIMIT %s
    """, (heading, heading, house, house, when, when, limit))
    return [DebateCandidate(debate_id=r["debate_id"], score=float(r["score"] or 0.0),
                            title=r["heading"], date=r["date"], house=r["house"]) for r in rows]  # fmt: skip
