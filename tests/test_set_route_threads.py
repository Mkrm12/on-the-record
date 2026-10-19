"""The set route over debate threads."""

from __future__ import annotations

import pytest

from answer import contract as K
from answer import router as RO
from answer import routes as RT
from ingest.runtime import run

PROV = {"route": "set", "corpus_version": "t", "retrieval_config_hash": "t", "recipe": "t", "prompt_version": "t",
        "model_id": "none"}

MEMBERS = {1: "Ann Minister", 2: "Bob Minister", 3: "Cat Member", 4: "Lord Xavier", 5: "Lord Young", 6: "Dan Raiser",
           7: "Eve Mover", 8: "Luke Murphy", 40: "Caroline Nokes", 50: "Lord McFall of Alcluith"}


@pytest.fixture
def record(scratch, scratch_dsn):
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WT', 't', '2020-01-01', '2026-12-31', 'dev') ON CONFLICT DO NOTHING")
    for mnis, name in MEMBERS.items():
        scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING", (mnis, name))
    debates = [
        # (id, date, house, section, order, heading)
        ("SR", "2026-09-14", "Commons", "Commons Chamber", 27, "Sovereign Grant Bill"),
        ("TR", "2026-09-14", "Commons", "Commons Chamber", 28, "Sovereign Grant Bill"),
        ("NX", "2026-09-14", "Commons", "Commons Chamber", 29, "Next Business"),
        ("A1", "2020-04-30", "Lords", "Lords Chamber", 3, "Arrangement of Business"),
        ("OB", "2020-04-30", "Lords", "Lords Chamber", 4, "Other Business"),
        ("A2", "2020-04-30", "Lords", "Lords Chamber", 8, "Arrangement of Business"),
        ("ST", "2020-04-30", "Lords", "Lords Chamber", 9, "A Statement"),
        ("A3", "2020-04-30", "Lords", "Lords Chamber", 10, "Arrangement of Business"),
        ("LC", "2020-03-19", "Commons", "Commons Chamber", 12, "Loan Charge 2019"),
        ("CC", "2026-03-19", "Commons", "Commons Chamber", 9, "Climate Change"),
        ("H1", "2026-09-01", "Commons", "Commons Chamber", 5, "Housing Supply"),
        ("H2", "2026-09-08", "Commons", "Commons Chamber", 5, "Housing Supply"),
    ]
    for did, day, house, section, order, heading in debates:
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, section_order, debate_type, window_id, "
                        "corpus_version) VALUES (%s, %s, %s, %s, %s, %s, 'other', 'WT', 't')",
                        (did, day, house, heading, section, order))  # fmt: skip
    speeches = [
        # (debate, seq, mnis, printed, role, text)
        ("SR", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "Order. I call the Minister."),
        ("SR", 1, 1, "The Parliamentary Secretary (Ann Minister)", "minister", "I beg to move, That the Bill be now read a Second time."),
        ("SR", 2, 3, "Cat Member (Somewhere) (Lab)", "member", "I support the Bill."),
        ("TR", 0, 2, "The Exchequer Secretary (Bob Minister)", "minister", "I beg to move, That the Bill be now read the Third time."),
        ("TR", 1, 3, "Cat Member (Somewhere) (Lab)", "member", "Briefly, I welcome it."),
        ("NX", 0, 3, "Cat Member (Somewhere) (Lab)", "member", "Something else."),
        ("A1", 0, 50, "The Lord Speaker (Lord McFall of Alcluith)", "lord_speaker", "My Lords, an announcement."),
        ("A1", 1, 4, "Lord Xavier (Con)", "member", "My Lords, I beg to move."),
        ("OB", 0, 4, "Lord Xavier (Con)", "member", "Other business."),
        ("A2", 0, 50, "The Lord Speaker (Lord McFall of Alcluith)", "lord_speaker", "My Lords, a further announcement."),
        ("ST", 0, 4, "Lord Xavier (Con)", "member", "A statement."),
        ("A3", 0, 50, "The Lord Speaker (Lord McFall of Alcluith)", "lord_speaker", "My Lords."),
        ("A3", 1, 5, "Lord Young (Lab)", "member", "My Lords, I beg to move."),
        ("LC", 0, 40, "Mr Deputy Speaker (Mr Nigel Evans)", "deputy_speaker", "Order."),
        ("LC", 1, 6, "Dan Raiser (Somewhere) (Con)", "member", "On a point of order, Mr Deputy Speaker. I am sorry to have delayed it."),
        ("LC", 2, 40, "Mr Deputy Speaker (Mr Nigel Evans)", "deputy_speaker", "I thank him."),
        ("LC", 3, 7, "Eve Mover (Somewhere) (Con)", "member", "I beg to move, That this House has considered the loan charge."),
        ("CC", 0, 40, "Madam Deput y Speaker (Caroline Nokes)", "member", "I call Luke Murphy, who will speak for up to 15 minutes."),
        ("CC", 1, 8, "Luke Murphy (Basingstoke) (Lab)", "member", "I beg to move, That this House has considered climate change."),
        ("H1", 0, 1, "The Minister (Ann Minister)", "minister", "I beg to move on housing supply."),
        ("H2", 0, 2, "The Minister (Bob Minister)", "minister", "I beg to move on housing supply again."),
    ]
    for n, (did, seq, mnis, printed, role, text) in enumerate(speeches):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_role, mnis_id, text, window_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, 'WT')", (f"sp{n}", did, seq, printed, role, mnis, text))
    return scratch_dsn


def ask(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await RT.set_route(conn, question, K.Provenance(**PROV), filters=RO.extract_filters(question))
        finally:
            await conn.close()

    return run(go()).answer


def test_one_debate_split_across_adjacent_records_is_answered_as_one(record):
    answer = ask(record, "Who opened the debate on Sovereign Grant Bill in the Commons on 14 September 2026?")
    assert answer.facets["opener"]["mnis_id"] == 1 and answer.facets["opener"]["debate_id"] == "SR"
    assert answer.facets["thread"] == ["SR", "TR"]
    assert answer.rendered.startswith("Ann Minister opened the debate on Sovereign Grant Bill in the Commons on "
                                      "2026-09-14"), answer.rendered
    assert "3 members spoke in all" in answer.rendered, "the Chair is never counted; both records' members are"
    assert "2 parts" in answer.rendered and "Bob Minister" in answer.rendered, "the later part's own opener is said"


def test_the_roster_of_a_thread_runs_across_its_records_in_the_days_order_and_never_lists_the_chair(record):
    answer = ask(record, "Who spoke in the debate on Sovereign Grant Bill in the Commons on 14 September 2026?")
    assert [e.mnis_id for e in answer.roster.entries] == [1, 3, 2]
    assert "Nokes" not in answer.rendered and "Deputy Speaker" not in answer.rendered


def test_recurring_items_under_one_heading_are_each_answered_with_their_opener(record):
    """Three Lords items headed "Arrangement of Business" that day, with other business between: one answer that names
    every item's opener (or says only the Chair spoke), never a guess and never a bare question back.
    """
    answer = ask(record, "Who opened the debate on Arrangement of Business in the Lords on 30 April 2020?")
    assert answer.abstention is None and "clarify" not in (answer.facets or {})
    assert [(o["debate_id"], o.get("mnis_id")) for o in answer.facets["openers"]] == [("A1", 4), ("A2", None), ("A3", 5)]
    assert answer.facets["openers"][1] == {"debate_id": "A2", "no_opener": "only_chair"}, "every item typed (QA pass 7)"
    text = answer.rendered
    assert "3 items headed 'Arrangement of Business'" in text and "Lord Xavier" in text and "Lord Young" in text
    assert "only the Chair spoke" in text and "McFall" not in text


def test_a_point_of_order_before_the_business_does_not_open_it(record):
    answer = ask(record, "Who opened the debate on Loan Charge 2019 in the Commons on 19 March 2020?")
    assert answer.facets["opener"]["mnis_id"] == 7, answer.rendered
    assert "2 members spoke in all" in answer.rendered, "the raiser spoke; the Chair did not count"


def test_the_chair_misprinted_and_stored_as_a_member_still_never_opens(record):
    answer = ask(record, "Who opened the debate on Climate Change in the Commons on 19 March 2026?")
    assert answer.facets["opener"]["mnis_id"] == 8 and "Nokes" not in answer.rendered


def test_a_question_back_lists_every_option_with_its_date_house_heading_and_opener(record):
    """Asked with no date, the heading recurs on two days."""
    answer = ask(record, "Who opened the debate on Housing Supply?")
    offered = answer.facets["clarify"]
    assert {o["debate_id"] for o in offered} == {"H1", "H2"}
    assert {o["opener"] for o in offered} == {"Ann Minister", "Bob Minister"}
    text = answer.rendered
    for needed in ("Housing Supply", "Commons", "2026-09-01", "2026-09-08", "Ann Minister", "Bob Minister"):
        assert needed in text, needed


# Divisions taken in one debate's records give one answer -----------------------------------------------------


@pytest.fixture
def resumed(scratch, scratch_dsn):
    """7 September 2026: the Financial Services and Markets Bill [HL] at records 18 and 20, a statement at 19; four
    divisions taken in record 18 and one in record 20. "The answers differ" was printed over five identical answers.
    """
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WT', 't', '2020-01-01', '2026-12-31', 'dev') ON CONFLICT DO NOTHING")
    for did, order, heading in (("F18", 18, "Financial Services and Markets Bill [HL]"), ("LG", 19, "Local Government"),
                                ("F20", 20, "Financial Services and Markets Bill [HL]")):
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, section_order, debate_type, window_id, "
                        "corpus_version) VALUES (%s, '2026-09-07', 'Lords', %s, 'Lords Chamber', %s, 'bill', 'WT', 't')",
                        (did, heading, order))  # fmt: skip
        for n in range(3):
            scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, window_id) "
                            "VALUES (%s, %s, %s, 'Lord A', 1, 'My Lords', 'WT')", (f"{did}-{n}", did, n))
    for n, debate in ((1, "F18"), (2, "F18"), (3, "F20")):
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, number, title, motion_text, ayes, "
                        "noes, result, debate_id, debate_link) VALUES (%s, %s, 'Lords', '2026-09-07', %s, "
                        "'Financial Services and Markets Bill [HL]', %s, 200, 100, 'agreed', %s, 'same_day')",
                        (1_003_700 + n, 3_700 + n, n, f"Lord X moved amendment {n}.", debate))  # fmt: skip
    return scratch_dsn


def test_divisions_taken_in_one_resumed_debate_give_one_answer(resumed):
    import psycopg

    from answer import pipeline as P

    question = "Find the debate that led to the Lords division on Financial Services and Markets Bill [HL] on 7 September 2026."

    async def go():
        conn = await psycopg.AsyncConnection.connect(resumed)
        try:
            return await P.structured_answer(conn, question, RO.route(question), K.Provenance(**PROV))
        finally:
            await conn.close()

    answer, notes = run(go())
    assert answer is not None, notes
    assert "differ" not in answer.rendered and "give the same answer" in answer.rendered, answer.rendered
    assert "a debate of 6 contributions" in answer.rendered, "the debate is both records, not the first alone"


# --- QA pass 6 (S05): contributions that say nothing, and the Chair's own statement ----------------------------------


@pytest.fixture
def qa6(record, scratch):
    """Shapes QA pass 6 found: a Speaker's Statement followed by a point of order; a member who "rose—" and was not
    called; a minister's "King's consent signified." before the member in charge moves Third Reading; an empty
    contribution first; an item whose only member contribution is a stage direction; and a heading of the Chair's
    statement recurring that day.
    """
    debates = [
        ("SS", "2026-09-15", "Speaker’s Statement", "speaker_statement", 2),
        ("RO", "2026-09-15", "Rail Fares", "general_debate", 3),
        ("KC", "2026-09-15", "Example Bill", "third_reading", 4),
        ("EM", "2026-09-15", "Empty Opening", "general_debate", 5),
        ("SO", "2026-09-15", "Stage Only", "other", 6),
        ("D1", "2026-09-16", "Deputy Speaker’s Statement", "other", 2),
        ("NB", "2026-09-16", "Next Business", "other", 3),
        ("D2", "2026-09-16", "Deputy Speaker’s Statement", "other", 4),
    ]
    for did, day, heading, kind, order in debates:
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, section_order, debate_type, window_id, "
                        "corpus_version) VALUES (%s, %s, 'Commons', %s, 'Commons Chamber', %s, %s, 'WT', 't')",
                        (did, day, heading, order, kind))  # fmt: skip
    scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (467, 'Sir Lindsay Hoyle') ON CONFLICT DO NOTHING")
    speeches = [
        ("SS", 0, 467, "Mr Speaker", "speaker", "I wish to make a short statement about the conduct of questions."),
        ("SS", 1, 3, "Cat Member (Somewhere) (Lab)", "member", "On a point of order, Mr Speaker. Further to your statement."),
        ("SS", 2, 467, "Mr Speaker", "speaker", "I thank the hon. Lady."),
        ("RO", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "Order."),
        ("RO", 1, 3, "Cat Member (Somewhere) (Lab)", "member", "<em>rose—</em>"),
        ("RO", 2, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "I call the Minister."),
        ("RO", 3, 1, "The Minister (Ann Minister)", "minister", "I beg to move, That this House has considered rail fares."),
        ("KC", 0, 2, "The Solicitor General (Bob Minister)", "minister", "<em>King’s consent signified.</em>"),
        ("KC", 1, 7, "Eve Mover (Somewhere) (Con)", "member", "I beg to move, That the Bill be now read the Third time."),
        ("EM", 0, 6, "Dan Raiser (Somewhere) (Con)", "member", ""),
        ("EM", 1, 1, "The Minister (Ann Minister)", "minister", "I beg to move."),
        ("SO", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "Order."),
        ("SO", 1, 3, "Cat Member (Somewhere) (Lab)", "member", "<em>indicated dissent.</em>"),
        ("D1", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "I have a short statement."),
        ("D1", 1, 6, "Dan Raiser (Somewhere) (Con)", "member", "On a point of order, Madam Deputy Speaker."),
        ("NB", 0, 1, "The Minister (Ann Minister)", "minister", "Other business."),
        ("D2", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "A further statement."),
        ("D2", 1, 7, "Eve Mover (Somewhere) (Con)", "member", "On a point of order, Madam Deputy Speaker."),
    ]
    from ingest.rules import boilerplate_reason

    for n, (did, seq, mnis, printed, role, text) in enumerate(speeches):
        # flagged as ingest flags them: "rose—" and "indicated dissent." are boilerplate, so the roster leaves them out
        reason = boilerplate_reason(text, printed)
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_role, mnis_id, text, window_id, "
                        "is_boilerplate, boilerplate_reason) VALUES (%s, %s, %s, %s, %s, %s, %s, 'WT', %s, %s)",
                        (f"q{n}", did, seq, printed, role, mnis, text, reason is not None, reason))  # fmt: skip
    return record


def test_the_speakers_statement_is_answered_as_the_chairs_never_as_opened_by_the_point_of_order_raiser(qa6):
    answer = ask(qa6, "Who opened the debate on Speaker’s Statement in the Commons on 15 September 2026?")
    assert "opener" not in answer.facets, answer.rendered
    assert answer.facets["chair_statement"]["debate_id"] == "SS"
    assert answer.facets["chair_statement"]["mnis_id"] == 467
    text = answer.rendered
    assert "opened the debate" not in text and "Cat Member opened" not in text, text
    assert text.startswith("The Commons item headed 'Speaker’s Statement' on 2026-09-15 is a statement by the "
                           "Chair (Mr Speaker, Sir Lindsay Hoyle), not a debate a member opened."), text
    assert "1 member also spoke." in text, text


def test_a_member_who_rose_and_was_not_called_never_opens(qa6):
    answer = ask(qa6, "Who opened the debate on Rail Fares in the Commons on 15 September 2026?")
    assert answer.facets["opener"]["mnis_id"] == 1, answer.rendered
    assert answer.rendered.startswith("Ann Minister opened the debate on Rail Fares"), answer.rendered


def test_the_kings_consent_signified_never_opens_the_member_in_charges_third_reading(qa6):
    answer = ask(qa6, "Who opened the debate on Example Bill in the Commons on 15 September 2026?")
    assert answer.facets["opener"]["mnis_id"] == 7, answer.rendered


def test_an_empty_contribution_never_opens(qa6):
    answer = ask(qa6, "Who opened the debate on Empty Opening in the Commons on 15 September 2026?")
    assert answer.facets["opener"]["mnis_id"] == 1, answer.rendered


@pytest.mark.parametrize("question", ["Who opened the debate on Stage Only in the Commons on 15 September 2026?",
                                      "Who spoke in the debate on Stage Only in the Commons on 15 September 2026?"])
def test_an_item_whose_members_only_rose_or_nodded_names_no_opener_and_says_why(qa6, question):
    """Its one member contribution is a stage direction, which ingest flags as boilerplate and the roster leaves out:
    the answer says what the record shows, never "records no contributions" (it records two).
    """
    answer = ask(qa6, question)
    assert answer.abstention is None and "opener" not in (answer.facets or {}), answer
    assert "Cat Member opened" not in answer.rendered
    assert answer.rendered.startswith("No member's speech opens the Commons item headed 'Stage Only' on 2026-09-15"), \
        answer.rendered
    assert answer.facets["no_opener"] == {"reason": "said_nothing", "debate_id": "SO"}, "the claim is typed"


def test_recurring_chairs_statements_are_each_called_the_chairs(qa6):
    answer = ask(qa6, "Who opened the debate on Deputy Speaker’s Statement in the Commons on 16 September 2026?")
    entries = answer.facets["openers"]
    assert [(e["debate_id"], e.get("chair_statement")) for e in entries] == [("D1", True), ("D2", True)], entries
    assert "Dan Raiser" not in answer.rendered and "Eve Mover" not in answer.rendered, answer.rendered
    assert answer.rendered.count("is a statement by the Chair") == 2, answer.rendered


def test_who_spoke_at_the_speakers_statement_still_lists_the_members_who_spoke(qa6):
    answer = ask(qa6, "Who spoke in the debate on Speaker’s Statement in the Commons on 15 September 2026?")
    assert [e.mnis_id for e in answer.roster.entries] == [3]


# --- QA pass 7: committee chairs, misprinted headings, what is said where no member made a speech ---------------------


@pytest.fixture
def qa7(record, scratch):
    """QA pass 7's shapes: a committee chair's statement; the Chair's statement misprinted; members who only shook
    their heads, rose or signified consent after it; recurring Lords items moved formally or only risen to; an item
    whose only member contribution is a formal motion; an item where only the Chair speaks.
    """
    from ingest.rules import boilerplate_reason

    for mnis, name in ((9, "Bernard Jenkin"), (467, "Sir Lindsay Hoyle")):
        scratch.execute("INSERT INTO member_id_map (mnis_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING", (mnis, name))
    debates = [
        ("QLC", "2025-11-12", "Commons", "Commons Chamber", 7, "Statement by the Chair of the Liaison Committee", "other"),
        ("QSP", "2025-11-13", "Commons", "Commons Chamber", 2, "Speakers Statement", "other"),
        ("QSS", "2025-11-14", "Commons", "Commons Chamber", 2, "Speaker\u2019s Statement", "speaker_statement"),
        ("QSR", "2025-11-17", "Commons", "Commons Chamber", 2, "Speaker\u2019s Statement", "speaker_statement"),
        ("QB1", "2025-11-18", "Lords", "Lords Chamber", 3, "Arrangement of Business", "business_statement"),
        ("QOB", "2025-11-18", "Lords", "Lords Chamber", 4, "Other Business", "other"),
        ("QB2", "2025-11-18", "Lords", "Lords Chamber", 8, "Arrangement of Business", "business_statement"),
        ("QSB", "2025-11-18", "Lords", "Lords Chamber", 9, "A Statement", "other"),
        ("QB3", "2025-11-18", "Lords", "Lords Chamber", 10, "Arrangement of Business", "business_statement"),
        ("QFM", "2025-11-19", "Commons", "Commons Chamber", 4, "Formal Motion Example", "other"),
        ("QOC", "2025-11-19", "Commons", "Commons Chamber", 5, "Only Chair Item", "other"),
        ("QIN", "2025-11-20", "Commons", "Commons Chamber", 3, "Virtual Floor Example", "other"),
    ]
    for did, day, house, section, order, heading, kind in debates:
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, section_order, debate_type, window_id, "
                        "corpus_version) VALUES (%s, %s, %s, %s, %s, %s, %s, 'WT', 't')",
                        (did, day, house, heading, section, order, kind))  # fmt: skip
    speeches = [
        ("QLC", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker",
         "Order. I call the Chair of the Liaison Committee."),
        ("QLC", 1, 9, "The Chair of the Liaison Committee (Sir Bernard Jenkin)", "member",
         "I am grateful for the opportunity to make this statement on the Committee's report."),
        ("QLC", 2, 3, "Cat Member (Somewhere) (Lab)", "member", "I thank the Chair of the Committee for his statement."),
        ("QSP", 0, 467, "Mr Speaker", "speaker", "I wish to make a short statement about the conduct of questions."),
        ("QSP", 1, 6, "Dan Raiser (Somewhere) (Con)", "member", "On a point of order, Mr Speaker."),
        ("QSS", 0, 467, "Mr Speaker", "speaker", "I wish to make a statement about security."),
        ("QSS", 1, 3, "Cat Member (Somewhere) (Lab)", "member", "<em>shook her head.</em>"),
        ("QSS", 2, 6, "Dan Raiser (Somewhere) (Con)", "member", "<em>rose\u2014</em>"),
        ("QSS", 3, 1, "The Minister (Ann Minister)", "minister", "<em>King\u2019s consent signified.</em>"),
        ("QSR", 0, 467, "Mr Speaker", "speaker", "A short statement."),
        ("QSR", 1, 6, "Dan Raiser (Somewhere) (Con)", "member", "<em>rose\u2014</em>"),
        ("QB1", 0, 50, "The Lord Speaker (Lord McFall of Alcluith)", "lord_speaker", "My Lords, an announcement."),
        ("QB1", 1, 4, "Lord Xavier (Con)", "member", "My Lords, I should like to say a word about today's business."),
        ("QOB", 0, 4, "Lord Xavier (Con)", "member", "Other business."),
        ("QB2", 0, 50, "The Lord Speaker (Lord McFall of Alcluith)", "lord_speaker", "My Lords, a further announcement."),
        ("QB2", 1, 5, "Lord Young (Lab)", "member", "My Lords, I beg to move."),
        ("QSB", 0, 4, "Lord Xavier (Con)", "member", "A statement."),
        ("QB3", 0, 50, "The Lord Speaker (Lord McFall of Alcluith)", "lord_speaker", "My Lords."),
        ("QB3", 1, 6, "Dan Raiser (Somewhere) (Con)", "member", "<em>rose\u2014</em>"),
        ("QFM", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "Order."),
        ("QFM", 1, 1, "The Minister (Ann Minister)", "minister", "I beg to move."),
        ("QOC", 0, 40, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", "Order. A short announcement."),
        ("QIN", 0, 1, "The Minister (Ann Minister)", "minister", "I beg to move that the House do now adjourn."),
        ("QIN", 1, 7, "Richard Drax (South Dorset) (Con) [V]", "member", "<em>[Inaudible.]</em>"),
        ("QIN", 2, 6, "Dan Raiser (Somewhere) (Con)", "member", "<em>rose—</em>"),
        ("QIN", 3, 3, "Cat Member (Somewhere) (Lab)", "member", "I thank the Minister."),
    ]
    for n, (did, seq, mnis, printed, role, text) in enumerate(speeches):
        reason = boilerplate_reason(text, printed)
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_role, mnis_id, text, window_id, "
                        "is_boilerplate, boilerplate_reason) VALUES (%s, %s, %s, %s, %s, %s, %s, 'WT', %s, %s)",
                        (f"w{n}", did, seq, printed, role, mnis, text, reason is not None, reason))  # fmt: skip
    return record


def test_a_committee_chairs_statement_is_opened_by_the_committee_chair(qa7):
    answer = ask(qa7, "Who opened the debate on Statement by the Chair of the Liaison Committee in the Commons on "
                      "12 November 2025?")
    assert answer.facets["opener"]["mnis_id"] == 9 and "chair_statement" not in answer.facets, answer.rendered


def test_the_chairs_statement_misprinted_without_its_apostrophe_is_still_the_chairs(qa7):
    answer = ask(qa7, "Who opened the debate on Speakers Statement in the Commons on 13 November 2025?")
    assert answer.facets["chair_statement"]["mnis_id"] == 467 and "opener" not in answer.facets, answer.rendered


@pytest.mark.parametrize("day", ["14 November 2025", "17 November 2025"])
def test_members_who_only_shook_their_heads_rose_or_signified_consent_made_no_speech(qa7, day):
    answer = ask(qa7, f"Who opened the debate on Speaker\u2019s Statement in the Commons on {day}?")
    text = answer.rendered
    assert "also spoke" not in text and "No member's contribution is recorded" not in text, text
    assert text.endswith("not a debate a member opened. No member made a speech in it."), text


def test_a_who_spoke_listing_says_what_each_item_records_never_only_the_chair_where_a_member_contributed(qa7):
    answer = ask(qa7, "Who spoke in the debate on Arrangement of Business in the Lords on 18 November 2025?")
    text = answer.rendered
    assert "only the Chair spoke" not in text, text
    assert "the second: only procedural contributions are recorded, by Lord Young (Lab)" in text, text
    assert "the third: no member's speech is recorded (only stage directions or empty contributions)" in text, text


def test_a_who_opened_listing_types_the_item_no_member_opened(qa7):
    answer = ask(qa7, "Who opened the debate on Arrangement of Business in the Lords on 18 November 2025?")
    entries = {e["debate_id"]: e for e in answer.facets["openers"]}
    assert entries["QB2"]["mnis_id"] == 5 and entries["QB3"] == {"debate_id": "QB3", "no_opener": "said_nothing"}, entries
    assert "the third: no member's speech opens it" in answer.rendered, answer.rendered


def test_an_item_whose_only_member_contribution_is_a_formal_motion_is_never_an_empty_list(qa7):
    spoke = ask(qa7, "Who spoke in the debate on Formal Motion Example in the Commons on 19 November 2025?")
    assert spoke.rendered == ("Only procedural contributions are recorded in the debate on Formal Motion Example "
                              "(Commons, 2025-11-19), by The Minister (Ann Minister)."), spoke.rendered
    opened = ask(qa7, "Who opened the debate on Formal Motion Example in the Commons on 19 November 2025?")
    assert opened.facets["opener"]["mnis_id"] == 1 and "0 members" not in opened.rendered, opened.rendered
    assert opened.rendered.endswith("No member's contribution beyond procedure is recorded."), opened.rendered


def test_an_item_only_the_chair_spoke_in_says_so_typed(qa7):
    answer = ask(qa7, "Who opened the debate on Only Chair Item in the Commons on 19 November 2025?")
    assert answer.rendered.startswith("Only the Chair is recorded speaking"), answer.rendered
    assert answer.facets["no_opener"] == {"reason": "only_chair", "debate_id": "QOC"}


def test_the_chairs_statement_names_its_maker_as_printed_then_never_by_a_later_title(qa7, scratch):
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (4019, 'Baroness Laing of Elderslie', "
                    "'Lords')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, section_order, debate_type, window_id, "
                    "corpus_version) VALUES ('QDS', '2021-03-10', 'Commons', 'Deputy Speaker\u2019s Statement', "
                    "'Commons Chamber', 2, 'speaker_statement', 'WT', 't')")
    for n, (seq, mnis, printed, text) in enumerate(((0, 4019, "Madam Deputy Speaker", "I have a short statement."),
                                                    (1, 3, "Cat Member (Somewhere) (Lab)", "On a point of order."))):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_person, mnis_id, text, window_id) "
                        "VALUES (%s, 'QDS', %s, %s, %s, %s, %s, 'WT')",
                        (f"ds{n}", seq, printed, "Dame Eleanor Laing" if mnis == 4019 else None, mnis, text))
    answer = ask(qa7, "Who opened the debate on Deputy Speaker\u2019s Statement in the Commons on 10 March 2021?")
    assert "a statement by the Chair (Madam Deputy Speaker, Dame Eleanor Laing (now Baroness Laing of Elderslie))" \
        in answer.rendered, answer.rendered


def test_a_member_who_spoke_inaudibly_on_the_virtual_floor_is_on_the_roster(qa7):
    """S05's real-data check: Richard Drax's only contribution on 29 April 2020 is "[Inaudible.]". He had the floor and
    spoke, so he is listed; the member who only rose is not.
    """
    answer = ask(qa7, "Who spoke in the debate on Virtual Floor Example in the Commons on 20 November 2025?")
    assert [e.mnis_id for e in answer.roster.entries] == [1, 7, 3], answer.rendered


def test_a_member_whose_words_are_not_recorded_is_listed_with_that_said(qa7):
    """The floor with the words unrecorded is a state of its own: listed, and the reader told."""
    answer = ask(qa7, "Who spoke in the debate on Virtual Floor Example in the Commons on 20 November 2025?")
    entries = {e.mnis_id: e for e in answer.roster.entries}
    assert entries[7].words_recorded is False and entries[3].words_recorded is True
    assert f"{entries[7].speaker_name} had the floor, but their words are not recorded" in answer.rendered
