"""The record's exact answers as MCP tools, over stdio."""

from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Any

import psycopg
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from app import safety, sealed
from app.config import settings

DSN = safety.working_dsn(settings.pg_dsn)
# The most divisions `find_divisions` returns at once (the largest day in the record holds 22).
LIMIT = 50
# The longest member name or title fragment read; no member's name is longer than 120 characters.
MAX_ARGUMENT = 200
HOUSES = {"commons": "Commons", "lords": "Lords"}


def _unavailable(division_id: int) -> str:
    return (f"No division with id {division_id} is available: the record does not hold one, or it falls in a period "
            f"that is not yet available. {sealed.SCOPE_STATEMENT}")
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

server = MCPServer(
    name="on-the-record",
    instructions=("Exact answers from the UK Parliament record (Hansard divisions and votes). Every answer comes from "
                  "the database, never from a model. April to July 2026 is not yet available."),
)  # fmt: skip


async def connect() -> psycopg.AsyncConnection:
    """A read-only connection that reads the masked record."""
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True, connect_timeout=5,
                                                 options="-c default_transaction_read_only=on")  # fmt: skip
    await conn.execute(sealed.serving_sql())
    await conn.execute("SET ROLE otr_reader")  # SELECT only (migration 026); fails closed if the role is missing
    return conn


async def _served_division(conn, division_id: int) -> bool:
    """Whether the masked record holds this division."""
    return await (await conn.execute("SELECT 1 FROM division WHERE id = %s", (int(division_id),))).fetchone() is not None


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


@server.tool(name="find_divisions", annotations=READ_ONLY,
             description="The divisions (recorded votes) held on a day, optionally in one House or by title words.")
async def find_divisions(day: str, house: str | None = None, title_contains: str | None = None) -> dict:
    """`day` as YYYY-MM-DD; `house` Commons or Lords, in any case; `title_contains` matched literally."""
    try:
        when = date.fromisoformat(day)
    except ValueError:
        return {"error": f"{day!r} is not a date as YYYY-MM-DD"}
    if house is not None and house.strip().lower() not in HOUSES:
        return {"error": f"{house!r} is not a House; Commons or Lords"}
    house = HOUSES[house.strip().lower()] if house is not None else None
    if title_contains is not None and len(title_contains) > MAX_ARGUMENT:
        return {"error": f"title words are read up to {MAX_ARGUMENT} characters"}
    if sealed.hidden(when):
        return {"day": day, "divisions": [], "note": sealed.SCOPE_STATEMENT}
    # `%` and `_` are characters to a person typing title words, not wildcards
    literal = None if title_contains is None else re.sub(r"([\\%_])", r"\\\1", title_contains)
    async with await connect() as conn:
        cur = await conn.execute("""
            SELECT id, house, number, division_date, title, result, ayes, noes, tally_reconciled FROM division
             WHERE division_date = %s AND (%s::text IS NULL OR house = %s)
               AND (%s::text IS NULL OR title ILIKE '%%' || %s || '%%' ESCAPE '\\')
             ORDER BY house, number NULLS LAST, id LIMIT %s
        """, (when, house, house, literal, literal, LIMIT))  # fmt: skip
        cols = ("division_id", "house", "number", "date", "title", "result", "ayes", "noes", "tally_reconciled")
        found = [{k: _iso(v) for k, v in zip(cols, row)} for row in await cur.fetchall()]
    if found:
        return {"day": day, "divisions": found}
    where = f" in the {house}" if house else ""
    words = f" with {title_contains!r} in its title" if title_contains else ""
    return {"day": day, "divisions": [], "note": f"No division is recorded{where} on {day}{words}."}


@server.tool(name="division_tally", annotations=READ_ONLY,
             description="One division's Ayes and Noes, as the tellers counted them, with any caveat the record needs.")
async def division_tally(division_id: int) -> dict:
    from answer import facts as F

    async with await connect() as conn:
        if not await _served_division(conn, division_id):
            return {"error": _unavailable(division_id)}
        fact = await F.division_tally(conn, int(division_id))
    if not fact.ok:
        return {"error": fact.missing or "no such division in the record"}
    v = fact.value or {}
    return {"division_id": v.get("division_id"), "ayes": v.get("ayes"), "noes": v.get("noes"),
            "lists_reconcile": v.get("tally_reconciled"), "sentence": fact.sentence(), "caveat": fact.caveat}  # fmt: skip


@server.tool(name="member_vote", annotations=READ_ONLY,
             description="How a named member voted in one division, or that the division's list does not record them.")
async def member_vote(division_id: int, member: str) -> dict:
    from answer import facts as F
    from answer import pipeline as P
    from answer import resolve as RS

    if not member.strip() or len(member) > MAX_ARGUMENT:
        return {"error": f"a member's name, up to {MAX_ARGUMENT} characters"}
    async with await connect() as conn:
        if not await _served_division(conn, division_id):
            return {"error": _unavailable(division_id)}
        # the argument is the name: read as given, against the division's list and then the member record, and refused
        # when it names more than one member.
        mnis, why = await RS.mnis_id(conn, member, division_ids=(int(division_id),))
        if mnis is None:
            return {"error": why}
        fact = await F.division_vote(conn, int(division_id), mnis)
        if fact.ok:
            v = fact.value or {}
            return {"recorded": True, "member": v.get("member"), "lobby": v.get("lobby"),
                    "teller": v.get("is_teller"), "proxy": v.get("proxy_name"), "sentence": fact.sentence()}  # fmt: skip
        return {"recorded": False, "statement": await P._not_in_list(conn, int(division_id), mnis)}


@server.tool(name="ask", annotations=READ_ONLY,
             description=("A question with an exact answer in the record (a division's result or tally, a member's "
                          "vote, a party or office on a date, a bill stage's date, a count). Free-text questions are "
                          "declined: they need the model."))  # fmt: skip
async def ask(question: str) -> dict:
    from answer import pipeline as P
    from answer import router as RO
    from answer import sealing as SG
    from answer.contract import Provenance

    decision = RO.route(question)
    prov = Provenance(route=decision.route, corpus_version=sealed.served_label("mcp"), retrieval_config_hash="mcp",
                      recipe="mcp", prompt_version="mcp", model_id="none")  # fmt: skip
    if SG.question_is_sealed(decision.filters):
        return {"answered": False, "text": P.reader_ready(SG.scope_answer(question, prov)).rendered}
    if decision.route != "structured":
        return {"answered": False, "why": f"this question has no exact answer in the record's tables ({decision.route})"}
    async with await connect() as conn:
        answer, notes = await P.structured_answer(conn, question, decision, prov)
        if answer is None:
            return {"answered": False, "why": "; ".join(notes)[:400] or "the record holds no exact answer to that"}
        answer, _ = await SG.guard(conn, P.reader_ready(answer))
    text = answer.rendered or answer.abstention_detail or answer.coverage_note
    return {"answered": answer.abstention is None and not (answer.facets or {}).get("clarify"), "text": text,
            "facts": [{"template": sf.template, "value": {k: _iso(x) for k, x in (sf.value or {}).items()}}
                      for sf in answer.structured_facts]}  # fmt: skip


def main() -> None:
    # A selector loop, not asyncio's default: on Windows the default Proactor loop cannot run async psycopg, and every
    # database tool failed over stdio while the in-process tests passed.
    asyncio.run(server.run_stdio_async(), loop_factory=asyncio.SelectorEventLoop)


if __name__ == "__main__":
    main()
