"""The MCP server: the record's exact answers as tools an agent can call."""

from __future__ import annotations

import psycopg
import pytest

from app import mcp_server as MS
from ingest.runtime import run as run_async

TITLE = "Health Bill: Report Stage: New Clause 6"


@pytest.fixture
def record(scratch, scratch_dsn, monkeypatch):
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")
    for did, day in ((2418, "2026-09-07"), (2150, "2026-05-20")):  # the second is inside the sealed window
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, number, title, ayes, noes, result, "
                        "ayes_listed, noes_listed, ayes_named, noes_named, tally_reconciled) VALUES "
                        "(%s, %s, 'Commons', %s, 6, %s, 2, 1, 'agreed', 2, 1, 2, 1, true)",
                        (did, did, day, TITLE))  # fmt: skip
        for mid, name, lobby in ((11, "Alan Mak", "aye"), (12, "Ann Member", "aye"), (13, "Neil Other", "no")):
            scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby) VALUES (%s, %s, %s, %s)",
                            (did, mid, name, lobby))  # fmt: skip
    for mid, name in ((11, "Alan Mak"), (12, "Ann Member"), (13, "Neil Other"), (14, "Kemi Absent")):
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (%s, %s, 'Commons')", (mid, name))
    monkeypatch.setattr(MS, "DSN", scratch_dsn)
    yield scratch
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")


def call(fn, *args, **kwargs):
    return run_async(fn(*args, **kwargs))


def test_the_server_offers_exactly_its_read_only_tools():
    tools = run_async(MS.server.list_tools())
    assert {t.name for t in tools} == {"find_divisions", "division_tally", "member_vote", "ask"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools), "every tool declares it changes nothing"


def test_find_divisions_reads_the_masked_record(record):
    out = call(MS.find_divisions, "2026-09-07")
    assert [d["division_id"] for d in out["divisions"]] == [2418]
    sealed_day = call(MS.find_divisions, "2026-05-20")
    assert sealed_day["divisions"] == [] and "not yet available" in sealed_day["note"],         "a sealed-window day is said to be unavailable, not shown as a day without divisions (QA pass 3)"


def test_find_divisions_says_when_a_day_holds_none(record):
    out = call(MS.find_divisions, "2026-09-08")
    assert out["divisions"] == [] and "No division is recorded" in out["note"]


def test_find_divisions_reads_the_house_in_any_case_and_refuses_an_unknown_one(record):
    assert [d["division_id"] for d in call(MS.find_divisions, "2026-09-07", house="commons")["divisions"]] == [2418]
    assert "error" in call(MS.find_divisions, "2026-09-07", house="Senate")


def test_title_words_are_matched_literally(record):
    assert call(MS.find_divisions, "2026-09-07", title_contains="Clause 6")["divisions"]
    assert call(MS.find_divisions, "2026-09-07", title_contains="%")["divisions"] == [], "a wildcard is a character"
    assert call(MS.find_divisions, "2026-09-07", title_contains="Health_Bill")["divisions"] == []


@pytest.mark.parametrize("division_id", [2150, 99999999], ids=["sealed", "nonexistent"])
def test_a_division_the_record_does_not_serve_is_said_to_be_unavailable(record, division_id):
    """QA pass 3: a sealed id answered "no division with id..." and a nonexistent one "not recorded in the published
    list". Neither is true of what the reader asked; one statement covers both, and says nothing of a sealed one.
    """
    for out in (call(MS.division_tally, division_id), call(MS.member_vote, division_id, "Alan Mak")):
        assert "error" in out and "not yet available" in out["error"] and "published list" not in str(out)


def test_a_tally_comes_with_its_sentence_and_caveat(record):
    out = call(MS.division_tally, 2418)
    assert (out["ayes"], out["noes"]) == (2, 1) and "2 voted Aye and 1 voted No" in out["sentence"]


def test_a_member_s_vote_and_a_member_absent_from_the_list(record):
    assert call(MS.member_vote, 2418, "Alan Mak")["lobby"] == "aye"
    absent = call(MS.member_vote, 2418, "Kemi Absent")
    assert absent["recorded"] is False and "not recorded in the published list" in absent["statement"]


@pytest.mark.parametrize("member", ["Neil Other's whip, and how did Alan Mak", "Alan Mak and Neil Other"])
def test_the_member_argument_names_one_member_or_is_refused(record, member):
    """It was pasted into "How did {member} vote?" and re-parsed, so a crafted name answered for someone else."""
    out = call(MS.member_vote, 2418, member)
    assert "error" in out and "more than one member" in out["error"]


def test_an_overlong_member_argument_is_refused(record):
    assert "error" in call(MS.member_vote, 2418, "Alan Mak " * 100)


def test_ask_answers_an_exact_question_and_refuses_what_needs_the_model(record):
    out = call(MS.ask, f"What were the Ayes and Noes in the Commons division on {TITLE} on 7 September 2026?")
    assert out["answered"] and "2 voted Aye and 1 voted No" in out["text"]
    free = call(MS.ask, "What did MPs say about the NHS?")
    assert free["answered"] is False and "exact answer" in free["why"]
    sealed = call(MS.ask, "What was the result of the Commons division on 20 May 2026?")
    assert "not yet available" in sealed["text"]


def test_the_server_cannot_write(record):
    async def go():
        async with await MS.connect() as conn:
            await conn.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (99, 'x')")

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        run_async(go())


def test_the_server_reads_as_a_role_that_can_only_select(record):
    """Read-only was one setting a superuser session could turn off. The server now reads as `otr_reader`, which holds
    SELECT and nothing else (migration 026), so a write fails on privilege even with that setting off.
    """
    async def go():
        async with await MS.connect() as conn:
            who = await (await conn.execute("SELECT current_user")).fetchone()
            await conn.execute("SET default_transaction_read_only = off")
            await conn.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (99, 'x')")
            return who

    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        run_async(go())
    async def user():
        async with await MS.connect() as conn:
            return (await (await conn.execute("SELECT current_user")).fetchone())[0]

    assert run_async(user()) == "otr_reader"


def test_a_real_client_session_over_stdio_reaches_the_database(record, scratch_dsn):
    """The handshake an MCP client performs, against the server as a child process. The server ran on Windows' default
    Proactor loop, which async psycopg cannot use, so every database tool failed over stdio while the in-process
    tests (on the project's selector loop) passed.
    """
    import asyncio
    import os
    import sys

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"],
                                   env={**os.environ, "PG_DSN": scratch_dsn})  # fmt: skip

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            res = await client.call_tool("find_divisions", {"day": "2026-09-07"})
            return res

    res = asyncio.run(asyncio.wait_for(session(), timeout=90))
    assert not res.is_error and "2418" in str(res.content), res.content
