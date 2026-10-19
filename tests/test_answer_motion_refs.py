"""A Lords motion in ping-pong is a reference a question can name."""

from __future__ import annotations

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import router as R
from app import safety
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e", recipe="r",
            prompt_version="p", model_id="m")
BILL = "Bus Services (No. 2) Bill [HL]"
DIVISIONS = [
    (1_003_387, 1, "Lord Moylan moved Motion 8A, as an amendment to the motion that this House do agree with the "
                   "Commons in their Amendment 8, to leave out from “House” to end and insert “do "
                   "disagree”. The House divided:", 150, 200, "negatived"),
    (1_003_388, 2, "Baroness Pidgeon moved Motion 8B, as an amendment to the motion that this House do agree with the "
                   "Commons in their Amendment 8, at end to insert “and do propose Amendment 8C”. The House "
                   "divided:", 160, 190, "negatived"),
    (1_003_389, 3, "Lord Moynihan moved Motion 31A, as an amendment to the motion that this House do agree with the "
                   "Commons in their Amendment 31, to leave out from “House” to end. The House divided:", 170,
     180, "negatived"),
    (1_003_390, 4, "Baroness Jones of Whitchurch moved Motion C2, as an amendment to Motion C, at end to insert "
                   "“and do propose Amendment 12B”. The House divided:", 210, 140, "agreed"),
]


@pytest.fixture
def pingpong(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    for vid, number, motion, ayes, noes, result in DIVISIONS:
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, motion_text, ayes, "
                        "noes, result, ayes_listed, noes_listed) VALUES (%s, %s, 'Lords', %s, '2025-10-13', %s, %s, %s, "
                        "%s, %s, %s, %s)", (vid, vid - 1_000_000, number, BILL, motion, ayes, noes, result, ayes, noes))
    return scratch_dsn


def ask(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()

    return run(go())


@pytest.mark.parametrize(("motion", "division"), [("Motion 31A", 1_003_389), ("Motion 8B", 1_003_388),
                                                  ("motion 8A", 1_003_387), ("Motion C2", 1_003_390)])
def test_the_motion_named_is_the_division_answered(pingpong, motion, division):
    answer, notes = ask(pingpong, f"What was the result of the Lords division on {motion} on the {BILL} on 13 October 2025?")
    assert answer is not None and answer.abstention is None, notes
    assert [sf.value["division_id"] for sf in answer.structured_facts] == [division], answer.rendered


@pytest.mark.parametrize("motion", ["Motion 31", "Motion C", "Motion 40A"])
def test_a_motion_no_division_was_on_is_never_answered_with_another(pingpong, motion):
    """"Motion 31" and "Motion C" were amended, and only their amendments divided on; "Motion 40A" is not in the
    record.
    """
    answer, notes = ask(pingpong, f"What was the result of the Lords division on {motion} on the {BILL} on 13 October 2025?")
    assert answer is None or not answer.structured_facts, answer.rendered
