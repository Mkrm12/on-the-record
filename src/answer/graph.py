"""Relationship tools: walking the record's structure in SQL."""

from __future__ import annotations

from datetime import date as Date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from answer import tally as T

# Per-query ceiling.
STATEMENT_TIMEOUT_MS = 2_000

# How deep the spine walk may go.
MAX_DEPTH = 5
DEFAULT_DEPTH = 5

# The p95 gates on, and the target it asks for.
LATENCY_GATE_MS = 100.0
LATENCY_TARGET_MS = 25.0

NodeKind = Literal["bill", "bill_stage", "bill_stage_sitting", "debate", "division"]


class GraphError(RuntimeError):
    """A walk was asked for something it cannot do — an unknown kind, or a depth beyond the cap."""


# --- typed returns --------------------------------------------------------------------------------------


class Node(BaseModel):
    """One node of a walk. `kind` and `id` together are the identity; `label` is for display only."""

    model_config = ConfigDict(extra="forbid")

    kind: NodeKind
    id: str
    label: str = ""
    depth: int = Field(ge=0)
    parent_kind: str | None = None
    parent_id: str | None = None
    house: str | None = None
    date: Date | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class DivisionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    house: str | None = None
    number: int | None = None
    date: Date | None = None
    title: str | None = None
    ayes: int | None = None
    noes: int | None = None
    result: str | None = None
    tally_reconciled: bool | None = None
    debate_id: str | None = None
    caveat: str = ""


class StageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    stage_name: str | None = None
    house: str | None = None
    date: Date | None = None
    sort_order: int | None = None
    first_sitting: Date | None = None
    last_sitting: Date | None = None
    sitting_count: int = 0
    debate_ids: tuple[str, ...] = ()
    divisions: tuple[DivisionSummary, ...] = ()


class BillWalk(BaseModel):
    """A bill and everything the record hangs off it."""

    model_config = ConfigDict(extra="forbid")

    bill_id: int
    title: str | None = None
    originating_house: str | None = None
    current_house: str | None = None
    is_act: bool | None = None
    is_defeated: bool | None = None
    is_withdrawn: bool | None = None
    stages: tuple[StageSummary, ...] = ()
    speech_count: int = 0
    speaker_count: int = 0
    first_date: Date | None = None
    last_date: Date | None = None
    # Said when the served stages lie on both sides of the sealed window: a stage inside it is not served (QA pass 3).
    coverage_note: str = ""
    missing: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing


def _window_note(stages: tuple[StageSummary, ...]) -> str:
    """The scope statement, when a bill's served stages fall both before and after the sealed window: any stage it held
    there is withheld, and a timeline without it must not read as the whole bill. Nothing about what is withheld is
    said, only that the period is not yet available.
    """
    from app import sealed

    if not sealed.masked():
        return ""
    window = sealed.window()
    days = [d for s in stages for d in (s.date, s.first_sitting, s.last_sitting) if d is not None]
    if days and min(days) < window.start and max(days) > window.end:
        return (f"Stages between {window.start.day} {window.start:%B %Y} and {window.end.day} {window.end:%B %Y} may be "
                f"missing. {sealed.SCOPE_STATEMENT}")
    return ""


class SpeechRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speech_id: str
    debate_id: str | None = None
    date: Date | None = None
    house: str | None = None
    heading: str | None = None
    seq: int | None = None
    speaker_name: str | None = None
    mnis_id: int | None = None


class VoteRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    division_id: int
    date: Date | None = None
    house: str | None = None
    title: str | None = None
    lobby: str | None = None
    is_teller: bool | None = None
    proxy_name: str | None = None


class MemberWalk(BaseModel):
    """What one member did in a window: what they said, and how they voted."""

    model_config = ConfigDict(extra="forbid")

    mnis_id: int
    name: str | None = None
    party_at_start: str | None = None
    speeches: tuple[SpeechRef, ...] = ()
    votes: tuple[VoteRef, ...] = ()
    debate_count: int = 0
    teller_count: int = 0
    missing: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing


class Participant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mnis_id: int | None = None
    speaker_name: str
    party: str | None = None
    contributions: int = Field(ge=1)
    first_seq: int | None = None


class DebateWalk(BaseModel):
    """A debate, who took part, what it was about, and what was decided."""

    model_config = ConfigDict(extra="forbid")

    debate_id: str
    date: Date | None = None
    house: str | None = None
    heading: str | None = None
    section: str | None = None
    participants: tuple[Participant, ...] = ()
    bill_ids: tuple[int, ...] = ()
    divisions: tuple[DivisionSummary, ...] = ()
    speech_count: int = 0
    missing: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing


def _division_caveat(row: dict) -> str:
    """The note a tally may not be quoted without: `answer.tally`'s, the one wording every route shares."""
    return T.caveat(row)


def _division(row: dict) -> DivisionSummary:
    return DivisionSummary(
        id=row["id"], house=row.get("house"), number=row.get("number"), date=row.get("division_date"),
        title=row.get("title"), ayes=row.get("ayes"), noes=row.get("noes"), result=row.get("result"),
        tally_reconciled=row.get("tally_reconciled"), debate_id=row.get("debate_id"),
        caveat=_division_caveat(row),
    )  # fmt: skip


# --- plumbing -------------------------------------------------------------------------------------------


async def _timeout(conn) -> None:
    """Bound this transaction's queries. LOCAL, so it dies with the transaction rather than leaking."""
    await conn.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")


async def _all(conn, sql: str, params: tuple | dict = ()) -> list[dict]:
    cur = await conn.execute(sql, params)
    columns = [c.name for c in cur.description or []]
    return [dict(zip(columns, row, strict=True)) for row in await cur.fetchall()]


async def _one(conn, sql: str, params: tuple | dict = ()) -> dict | None:
    rows = await _all(conn, sql, params)
    return rows[0] if rows else None


# --- the recursive spine ---------------------------------------------------------------------------------

# Heterogeneous nodes need one identity type, so an id is carried as text and cast back on use.
_SPINE = """
WITH RECURSIVE edges (pk, pid, ck, cid, label, house, date) AS (
        SELECT 'bill', s.bill_id::text, 'bill_stage', s.id::text, s.stage_name, s.house, s.date
        FROM bill_stage s WHERE s.bill_id = %(bill)s
    UNION ALL
        SELECT 'bill_stage', t.bill_stage_id::text, 'bill_stage_sitting', t.id::text,
               NULL, t.house, t.date
        FROM bill_stage_sitting t WHERE t.bill_id = %(bill)s
    UNION ALL
        SELECT 'bill_stage_sitting', t.id::text, 'debate', d.id, d.heading, d.house, d.date
        FROM bill_stage_sitting t JOIN debate d ON d.id = t.debate_id WHERE t.bill_id = %(bill)s
    UNION ALL
        SELECT 'debate', v.debate_id, 'division', v.id::text, v.title, v.house, v.division_date
        FROM division v
        WHERE v.debate_id IN (SELECT debate_id FROM bill_stage_sitting WHERE bill_id = %(bill)s)
),
walk (kind, id, label, depth, parent_kind, parent_id, house, date, path) AS (
        SELECT 'bill', b.id::text, b.title, 0, NULL::text, NULL::text,
               b.originating_house, NULL::date, ARRAY['bill:' || b.id::text]
        FROM bill b WHERE b.id = %(bill)s
    UNION ALL
        SELECT e.ck, e.cid, e.label, w.depth + 1, w.kind, w.id, e.house, e.date,
               w.path || (e.ck || ':' || e.cid)
        FROM walk w
        JOIN edges e ON e.pk = w.kind AND e.pid = w.id
        WHERE w.depth < %(depth)s
          AND NOT (e.ck || ':' || e.cid) = ANY(w.path)
)
SELECT kind, id, label, depth, parent_kind, parent_id, house, date, cardinality(path) AS path_length
FROM walk
ORDER BY depth, kind, id
"""


async def spine(conn, bill_id: int, depth: int = DEFAULT_DEPTH) -> list[Node]:
    """Everything structurally under a bill, to `depth` levels, as flat typed nodes."""
    if depth < 0 or depth > MAX_DEPTH:
        raise GraphError(f"depth must be between 0 and {MAX_DEPTH}; the spine is only {MAX_DEPTH} kinds deep")
    await _timeout(conn)
    rows = await _all(conn, _SPINE, {"bill": bill_id, "depth": depth})
    return [
        Node(kind=r["kind"], id=r["id"], label=r["label"] or "", depth=r["depth"],
             parent_kind=r["parent_kind"], parent_id=r["parent_id"], house=r["house"], date=r["date"],
             detail={"path_length": r["path_length"]})  # fmt: skip
        for r in rows
    ]


# --- the fixed-depth tools -------------------------------------------------------------------------------


async def bill_walk(conn, bill_id: int) -> BillWalk:
    """Bill → stages → sittings → debates → speeches → divisions, aggregated."""
    await _timeout(conn)
    head = await _one(conn, """
        SELECT id, title, originating_house, current_house, is_act, is_defeated, is_withdrawn
        FROM bill WHERE id = %s
    """, (bill_id,))
    if head is None:
        return BillWalk(bill_id=bill_id, missing=f"no bill with id {bill_id}")

    stages = await _all(conn, """
        SELECT s.id, s.stage_name, s.house, s.date, s.sort_order, s.first_sitting, s.last_sitting,
               count(DISTINCT t.id) AS sitting_count,
               array_remove(array_agg(DISTINCT t.debate_id), NULL) AS debate_ids
        FROM bill_stage s
        LEFT JOIN bill_stage_sitting t ON t.bill_stage_id = s.id
        WHERE s.bill_id = %s
        GROUP BY s.id, s.stage_name, s.house, s.date, s.sort_order, s.first_sitting, s.last_sitting
        ORDER BY s.sort_order NULLS LAST, s.date NULLS LAST, s.id
    """, (bill_id,))

    # A division reaches a bill two ways — by its stage, or by the debate a sitting of that stage was held in — and a
    # division recorded against the stage but not the debate would be invisible through either alone.
    divisions = await _all(conn, """
        SELECT DISTINCT v.id, v.house, v.number, v.division_date, v.title, v.ayes, v.noes, v.result,
               v.ayes_listed, v.noes_listed, v.ayes_named, v.noes_named, v.tally_reconciled, v.debate_id,
               v.bill_stage_id
        FROM division v
        WHERE v.bill_stage_id IN (SELECT id FROM bill_stage WHERE bill_id = %s)
           OR v.debate_id IN (SELECT debate_id FROM bill_stage_sitting WHERE bill_id = %s)
        ORDER BY v.division_date NULLS LAST, v.id
    """, (bill_id, bill_id))
    by_stage: dict[int | None, list[DivisionSummary]] = {}
    for row in divisions:
        by_stage.setdefault(row.get("bill_stage_id"), []).append(_division(row))

    counts = await _one(conn, """
        SELECT count(*) AS speech_count, count(DISTINCT sp.mnis_id) AS speaker_count,
               min(d.date) AS first_date, max(d.date) AS last_date
        FROM speech sp
        JOIN debate d ON d.id = sp.debate_id
        WHERE sp.debate_id IN (
            SELECT debate_id FROM bill_stage_sitting WHERE bill_id = %s AND debate_id IS NOT NULL
        ) AND NOT sp.is_boilerplate
    """, (bill_id,)) or {}

    summaries = tuple(
        StageSummary(
            id=s["id"], stage_name=s.get("stage_name"), house=s.get("house"), date=s.get("date"),
            sort_order=s.get("sort_order"), first_sitting=s.get("first_sitting"),
            last_sitting=s.get("last_sitting"), sitting_count=s.get("sitting_count") or 0,
            debate_ids=tuple(s.get("debate_ids") or ()),
            divisions=tuple(by_stage.get(s["id"], ())),
        ) for s in stages
    )  # fmt: skip
    return BillWalk(
        bill_id=bill_id, title=head.get("title"), originating_house=head.get("originating_house"),
        current_house=head.get("current_house"), is_act=head.get("is_act"),
        is_defeated=head.get("is_defeated"), is_withdrawn=head.get("is_withdrawn"),
        stages=summaries, coverage_note=_window_note(summaries),
        speech_count=counts.get("speech_count") or 0,
        speaker_count=counts.get("speaker_count") or 0,
        first_date=counts.get("first_date"), last_date=counts.get("last_date"),
    )  # fmt: skip


async def member_walk(conn, mnis_id: int, from_date: Date | None = None, to_date: Date | None = None,
                      limit: int = 200) -> MemberWalk:  # fmt: skip
    """One member's contributions and votes, optionally inside a date range."""
    await _timeout(conn)
    # `party_at_start` is the party ON the walk's first date, from the affiliation history — not `member_id_map.party`,
    # which is today's party and was wrong for every one of the 245 members whose party changed.
    head = await _one(conn, """
        SELECT m.mnis_id, m.name,
               member_party_on(m.mnis_id, coalesce(%s::date, (
                   SELECT least(
                       (SELECT min(d.date) FROM speech sp JOIN debate d ON d.id = sp.debate_id
                         WHERE sp.mnis_id = m.mnis_id),
                       (SELECT min(v.division_date) FROM division_vote dv JOIN division v ON v.id = dv.division_id
                         WHERE dv.mnis_id = m.mnis_id))))) AS party_at_start
          FROM member_id_map m WHERE m.mnis_id = %s
    """, (from_date, mnis_id))
    if head is None:
        return MemberWalk(mnis_id=mnis_id, missing=f"no member with mnis id {mnis_id}")

    speeches = await _all(conn, """
        SELECT sp.id AS speech_id, sp.debate_id, d.date, d.house, d.heading, sp.seq,
               sp.speaker_name, sp.mnis_id
        FROM speech sp
        JOIN debate d ON d.id = sp.debate_id
        WHERE sp.mnis_id = %s AND NOT sp.is_boilerplate
          AND (%s::date IS NULL OR d.date >= %s) AND (%s::date IS NULL OR d.date <= %s)
        ORDER BY d.date, sp.seq
        LIMIT %s
    """, (mnis_id, from_date, from_date, to_date, to_date, limit))

    votes = await _all(conn, """
        SELECT dv.division_id, v.division_date AS date, v.house, v.title, dv.lobby, dv.is_teller, dv.proxy_name
        FROM division_vote dv
        JOIN division v ON v.id = dv.division_id
        WHERE dv.mnis_id = %s
          AND (%s::date IS NULL OR v.division_date >= %s) AND (%s::date IS NULL OR v.division_date <= %s)
        ORDER BY v.division_date, dv.division_id
    """, (mnis_id, from_date, from_date, to_date, to_date))

    totals = await _one(conn, """
        SELECT count(DISTINCT sp.debate_id) AS debate_count
        FROM speech sp JOIN debate d ON d.id = sp.debate_id
        WHERE sp.mnis_id = %s AND NOT sp.is_boilerplate
          AND (%s::date IS NULL OR d.date >= %s) AND (%s::date IS NULL OR d.date <= %s)
    """, (mnis_id, from_date, from_date, to_date, to_date)) or {}

    return MemberWalk(
        mnis_id=mnis_id, name=head.get("name"), party_at_start=head.get("party_at_start"),
        speeches=tuple(SpeechRef(**s) for s in speeches),
        votes=tuple(VoteRef(**v) for v in votes),
        debate_count=totals.get("debate_count") or 0,
        teller_count=sum(1 for v in votes if v.get("is_teller")),
    )  # fmt: skip


async def debate_walk(conn, debate_id: str) -> DebateWalk:
    """A debate's participants, the bills it concerned, and its divisions."""
    await _timeout(conn)
    head = await _one(conn, "SELECT id, date, house, heading, section FROM debate WHERE id = %s", (debate_id,))
    if head is None:
        return DebateWalk(debate_id=debate_id, missing=f"no debate with id {debate_id!r}")

    participants = await _all(conn, """
        SELECT sp.mnis_id, sp.speaker_name, max(sp.speaker_party) AS party,
               count(*) AS contributions, min(sp.seq) AS first_seq
        FROM speech sp
        WHERE sp.debate_id = %s AND NOT sp.is_boilerplate AND sp.speaker_name IS NOT NULL
        GROUP BY sp.mnis_id, sp.speaker_name
        ORDER BY min(sp.seq)
    """, (debate_id,))

    bills = await _all(conn, """
        SELECT DISTINCT bill_id FROM bill_stage_sitting
        WHERE debate_id = %s AND bill_id IS NOT NULL ORDER BY bill_id
    """, (debate_id,))

    divisions = await _all(conn, """
        SELECT id, house, number, division_date, title, ayes, noes, result,
               ayes_listed, noes_listed, ayes_named, noes_named, tally_reconciled, debate_id
        FROM division WHERE debate_id = %s ORDER BY number NULLS LAST, id
    """, (debate_id,))

    return DebateWalk(
        debate_id=debate_id, date=head.get("date"), house=head.get("house"),
        heading=head.get("heading"), section=head.get("section"),
        participants=tuple(Participant(**p) for p in participants),
        bill_ids=tuple(b["bill_id"] for b in bills),
        divisions=tuple(_division(d) for d in divisions),
        speech_count=sum(p["contributions"] for p in participants),
    )  # fmt: skip


async def speech_to_vote(conn, speech_id: str) -> dict[str, Any]:
    """From a speech to how its speaker voted in the same debate's divisions."""
    await _timeout(conn)
    row = await _one(conn, """
        SELECT sp.id AS speech_id, sp.debate_id, sp.mnis_id, sp.speaker_name, sp.speaker_party,
               d.date, d.house, d.heading
        FROM speech sp JOIN debate d ON d.id = sp.debate_id
        WHERE sp.id = %s
    """, (speech_id,))
    if row is None:
        return {"missing": f"no speech with id {speech_id!r}"}
    votes = await _all(conn, """
        SELECT dv.division_id, v.title, v.division_date, dv.lobby, dv.is_teller, v.result
        FROM division v
        JOIN division_vote dv ON dv.division_id = v.id AND dv.mnis_id = %s
        WHERE v.debate_id = %s
        ORDER BY v.number NULLS LAST, v.id
    """, (row["mnis_id"], row["debate_id"]))
    return {**row, "votes": votes}
