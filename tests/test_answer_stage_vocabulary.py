"""The stage a division was taken at, from its title and from the Bills API's stage link."""

from __future__ import annotations

import pytest

from answer import contract as K
from answer import pipeline as PL
from answer import resolve as RS
from answer import router as R
from app import safety
from ingest.runtime import run

PROV = dict(route="structured", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e", recipe="r",
            prompt_version="p", model_id="m")
BILL = "Employment Rights Bill"


@pytest.fixture
def staged(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO bill (id, title) VALUES (3737, 'Employment Rights Act 2025')")
    for sid, name in ((901, "Report stage"), (902, "Committee of the whole House")):
        scratch.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house, abbreviation) "
                        "VALUES (%s, 3737, %s, %s, 'X')", (sid, name, "Lords" if sid == 901 else "Commons"))
    rows = [  # (id, house, day, number, title, motion, stage)
        (1_003_420, "Lords", "2025-07-14", 1, BILL, "Lord Sharpe of Epsom moved amendment 5, in clause 1, page 1, line "
                                                     "3, to leave out “may”. The House divided:", 901),
        (1_003_421, "Lords", "2025-07-14", 2, BILL, "Lord Fox moved amendment 9, after clause 2, to insert the new "
                                                     "clause. The House divided:", 901),
        (1_003_422, "Lords", "2025-07-16", 1, BILL, "Lord Burns moved amendment 12. The House divided:", None),
        (2_101, "Commons", "2025-03-11", 101, f"{BILL}: New Clause 7", None, 902),
    ]
    for vid, house, day, number, title, motion, stage in rows:
        source = vid - 1_000_000 if house == "Lords" else vid
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, motion_text, ayes, "
                        "noes, result, ayes_listed, noes_listed, bill_stage_id) VALUES (%s, %s, %s, %s, %s, %s, %s, 200, "
                        "150, 'agreed', 200, 150, %s)", (vid, source, house, number, day, title, motion, stage))
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


@pytest.mark.parametrize("question", [
    f"What was the result of the Lords report stage division on amendment 5 to the {BILL} on 14 July 2025?",
    f"What was the result of the division on amendment 5 on Report of the {BILL} in the Lords on 14 July 2025?",
    f"What was the result of the Commons division on New Clause 7 at committee stage of the {BILL} on 11 March 2025?",
    f"What was the result of the division on New Clause 7 in Committee of the whole House on the {BILL} on 11 March 2025?",
])
def test_the_stage_the_bills_api_links_confirms_the_stage_a_question_names(staged, question):
    answer, notes = ask(staged, question)
    assert answer is not None and answer.structured_facts, notes
    assert len(answer.structured_facts) == 1


@pytest.mark.parametrize("question", [
    f"What was the result of the Lords third reading division on amendment 5 to the {BILL} on 14 July 2025?",
    f"What was the result of the Commons division on New Clause 7 at report stage of the {BILL} on 11 March 2025?",
    # no title stage and no stage link: nothing confirms the stage named, so it is not assumed
    f"What was the result of the Lords report stage division on amendment 12 to the {BILL} on 16 July 2025?",
])
def test_a_stage_the_division_was_not_taken_at_or_cannot_be_shown_is_never_answered(staged, question):
    answer, notes = ask(staged, question)
    assert answer is None or not answer.structured_facts, answer.rendered


@pytest.mark.parametrize(("text", "stages"), [
    ("the report stage division", {"report stage"}), ("on Report", {"report stage"}), ("at report", {"report stage"}),
    ("at committee stage", {"committee stage"}), ("in Committee of the whole House", {"committee stage"}),
    ("in Grand Committee", {"committee stage"}), ("on consideration of Commons amendments", {"consideration"}),
    ("on consideration of Lords amendments", {"consideration"}),
    ("the report of the Standards Committee", set()), ("a report on housing", set()), ("Lords Amendment 5", set()),
    ("Second Reading", {"2nd reading"}),
])
def test_the_stage_vocabulary_reads_the_houses_own_names_and_nothing_else(text, stages):
    assert RS._stages(text) == stages


@pytest.mark.parametrize(("linked", "stage"), [
    ("Report stage", "report stage"), ("Committee of the whole House", "committee stage"),
    ("Committee stage", "committee stage"), ("Grand Committee", "committee stage"), ("2nd reading", "2nd reading"),
    ("3rd reading", "3rd reading"), ("Consideration of Commons amendments and / or reasons", "consideration"),
    ("Consideration of Lords amendments", "consideration"), ("Consideration of Lords message", "consideration"),
    ("Royal Assent", None), (None, None),
])
def test_a_linked_stage_name_is_read_into_the_same_vocabulary(linked, stage):
    assert RS.linked_stage(linked) == stage
