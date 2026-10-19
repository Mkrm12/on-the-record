"""The deterministic probes on every push, with no corpus, no model and no GPU (v2.1 S02)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import psycopg
import pytest

from app import sealed
from eval.oracles import divisions as O
from eval.probes import core
from eval.probes import divisions as PD
from ingest.divisions import parse_commons, parse_lords, store
from ingest.runtime import run as run_async

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "divisions.json").read_text(encoding="utf-8"))


def payloads():
    return ([("Commons", copy.deepcopy(p)) for p in FIXTURES["commons"].values()]
            + [("Lords", copy.deepcopy(p)) for p in FIXTURES["lords"].values()])  # fmt: skip


@pytest.fixture
def loaded(scratch, scratch_dsn):
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")

    async def load():
        async with await psycopg.AsyncConnection.connect(scratch_dsn, autocommit=True) as conn:
            for house, payload in payloads():
                await store(conn, parse_commons(payload) if house == "Commons" else parse_lords(payload))

    run_async(load())
    # The member record, as production has it: every member the payloads name, so a member absent from one division's
    # list can still be named and told they are not in it.
    members = {}
    for house, payload in payloads():
        lists = ("Ayes", "Noes", "AyeTellers", "NoTellers", "NoVoteRecorded") if house == "Commons" else (
            "contents", "notContents", "contentTellers", "notContentTellers")
        for key in lists:
            for m in payload.get(key) or []:
                mid, name = m.get("MemberId") or m.get("memberId"), m.get("Name") or m.get("name")
                if mid and name:
                    members[int(mid)] = (name, house)
    for mid, (name, house) in members.items():
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                        (mid, name, house))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE division, division_vote, member_id_map CASCADE")


def probe(dsn):
    truths = [O.parse_commons(p) if house == "Commons" else O.parse_lords(p) for house, p in payloads()]
    trials = PD.st_trials(truths)

    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            await conn.execute(sealed.serving_sql())  # production's serving connection
            await conn.commit()
            return await core.run(conn, trials, PD.verdict_for(truths))

    return truths, run_async(go())


def test_the_probes_find_no_harm_and_answer_every_canonical_question(loaded):
    truths, outcomes = probe(loaded)
    assert len(truths) == 5 and len(outcomes) >= 40, "the suite must not pass by running nothing"
    summary = core.summarise(outcomes)
    assert summary["all"]["harm"] == 0, summary["examples"].get("harm")
    assert "error" not in summary["all"]["outcomes"], summary["examples"].get("error")
    weak = [(o.trial.trial_id, o.outcome, o.why) for o in outcomes if o.trial.form == "canonical" and not o.acceptable]
    assert not weak, weak


def test_every_class_is_exercised(loaded):
    _, outcomes = probe(loaded)
    classes = {o.trial.cls for o in outcomes}
    assert classes == {"ST-division_result", "ST-division_tally", "ST-division_vote"}
    roles = {o.trial.subject.get("role") for o in outcomes if o.trial.cls == "ST-division_vote"}
    assert {"aye", "no", "teller"} <= roles, roles


def test_a_day_s_divisions_asked_about_without_a_title_are_never_answered_as_one(loaded):
    """MR-06 on the fixtures' one two-division day (2 September 2026): one division's answer must never be given as the
    day's, and, since S05 built the day's table, every trial, the noun singular or plural, is answered with every
    division, each value right.
    """
    truths = [O.parse_commons(p) if house == "Commons" else O.parse_lords(p) for house, p in payloads()]
    trials = PD.mr06_trials(truths)
    assert trials, "the fixtures hold a day with two divisions"

    async def go():
        async with await psycopg.AsyncConnection.connect(loaded) as conn:
            return await core.run(conn, trials, PD.verdict_for(truths))

    outcomes = run_async(go())
    assert not [o for o in outcomes if o.outcome in ("harm", "error")], [(o.trial.trial_id, o.why) for o in outcomes]
    # the plural trials are a class of their own, so MR-06's pre-registered bar is exactly its canonical trials
    assert {(o.trial.cls, o.trial.form) for o in outcomes} == {("MR-06", "canonical"), ("MR-06-plural", "variant")}
    assert [o.outcome for o in outcomes] == ["aggregate"] * len(outcomes), [(o.trial.trial_id, o.outcome, o.why)
                                                                             for o in outcomes]


@pytest.mark.parametrize("make", ["mr02_trials", "mr10_trials", "mr11_trials"])
def test_the_mutation_probes_find_no_harm_on_the_fixtures(loaded, make):
    """MR-02 (a division the record lacks: a wrong number, bill, stage or day), MR-10 (a misspelt member) and MR-11
    (the other House): the shapes the rebuilt probes found 23 harms in, on every push.
    """
    truths = [O.parse_commons(p) if house == "Commons" else O.parse_lords(p) for house, p in payloads()]
    trials = getattr(PD, make)(truths)
    assert trials, f"{make} must produce trials from the fixtures"

    async def go():
        async with await psycopg.AsyncConnection.connect(loaded) as conn:
            await conn.execute(sealed.serving_sql())
            await conn.commit()
            return await core.run(conn, trials, PD.verdict_for(truths))

    outcomes = run_async(go())
    assert not [o for o in outcomes if o.outcome in ("harm", "error")], [(o.trial.trial_id, o.why) for o in outcomes
                                                                         if o.outcome in ("harm", "error")]
