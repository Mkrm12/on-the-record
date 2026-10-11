"""The routes that answered "who opened the debate?" wrongly or not at all."""

from __future__ import annotations

from datetime import date

import pytest

from answer import contract as K
from answer import roster as R
from answer import router as RO
from answer import routes as RT
from app import safety
from ingest.runtime import run

PROV = dict(route="set", resolved_filters={}, corpus_version="v2", retrieval_config_hash="c",
            recipe="r", prompt_version="p", model_id="m")  # fmt: skip


# --- the router: every everyday form of "who opened" ----------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "Who led off on New Homes in the Commons on 12 January 2026?",
    "Who kicked off the debate on Vicars: Rural Parishes?",
    "Which member spoke first in the Lords debate on Carer's Allowance: Repayments on 29 July 2024?",
    "Who was the first speaker in the debate on Court Backlogs?",
    "Who opened the debate on Public Office (Accountability) Bill on 1 September 2026?",
    "Who began the debate on D1 Licence Training?",
    "Who wound up the debate on Ukraine and Russia?",
])  # fmt: skip
def test_every_form_of_who_opened_routes_to_the_set_route(question):
    assert RO.route(question).route == "set"


@pytest.mark.parametrize("question", [
    "What did MPs say about the scheme that started in 2024?",
    "What happened after the negotiations began?",
    "Why did the Government open a consultation on fracking?",
])  # fmt: skip
def test_a_question_merely_using_the_words_is_not_an_opener_question(question):
    assert RO.route(question).route != "set"


@pytest.mark.parametrize("question, want", [
    ("Who led off on New Homes?", True), ("Which member spoke first in the debate?", True),
    ("Who was the first speaker?", True), ("Who kicked off the debate?", True), ("Who opened it?", True),
    ("Who spoke in the debate on New Homes?", False), ("List the members who took part", False),
])  # fmt: skip
def test_the_set_route_knows_when_the_opener_was_asked_for(question, want):
    assert RT.asks_for_opener(question) is want


# --- the lookup: the heading inside the question -------------------------------------------------------------------


@pytest.fixture
def debates(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WK', 'test', '2024-01-01', '2026-12-31', 'dev_test')")
    rows = [
        ("dCA", "Carer’s Allowance: Repayments", "2024-07-29", "Lords"),
        ("dLGD", "Local Government Reorganisation in Devon", "2026-09-07", "Commons"),
        ("dLGC", "Local Government Reorganisation: Devon", "2026-09-07", "Commons"),
        ("dLG", "Local Government Reorganisation", "2026-09-07", "Commons"),
        ("dAB1", "Arrangement of Business", "2025-11-05", "Lords"),
        ("dAB2", "Arrangement of Business", "2025-11-06", "Lords"),
        ("dThe", "The", "2025-11-06", "Lords"),
    ]
    for did, heading, day, house in rows:
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                        "VALUES (%s, %s, %s, %s, 'Chamber', 'WK', 'test')", (did, day, house, heading))
    return scratch_dsn


def lookup(dsn, question, house=None, when=None):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await R.candidates_in_question(conn, question, house, when)
        finally:
            await conn.close()
    return run(go())


@pytest.mark.parametrize("question", [
    "Which member spoke first in the Lords debate on Carer’s Allowance: Repayments on 29 July 2024?",
    "Which member spoke first in the Lords debate on Carer's Allowance: Repayments on 29 July 2024?",
])  # fmt: skip
def test_a_normally_worded_question_finds_its_debate_with_either_apostrophe(debates, question):
    found = lookup(debates, question, "Lords", date(2024, 7, 29))
    assert R.pick_debate(found).debate_id == "dCA"


def test_the_old_lookup_given_the_question_found_nothing(debates):
    """The defect, pinned: a heading lookup given a whole question requires the heading to hold every question word.
    """
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(debates)
        try:
            return await R.candidates_by_heading(conn, "Who spoke first in the debate on Carer's Allowance: "
                                                       "Repayments?", "Lords", date(2024, 7, 29))
        finally:
            await conn.close()
    assert run(go()) == []


def test_a_verbatim_heading_beats_one_with_the_same_words(debates):
    found = lookup(debates, "Which member spoke first in the Commons debate on Local Government Reorganisation in "
                            "Devon on 7 September 2026?", "Commons", date(2026, 9, 7))
    assert R.pick_debate(found).debate_id == "dLGD"


def test_the_question_naming_the_generic_heading_gets_the_generic_debate(debates):
    found = lookup(debates, "Who opened the debate on Local Government Reorganisation?", "Commons", date(2026, 9, 7))
    assert R.pick_debate(found).debate_id == "dLG"


def test_a_recurring_heading_without_a_date_is_asked_about_not_guessed(debates):
    found = lookup(debates, "Who opened Arrangement of Business?", "Lords")
    choice = R.pick_debate(found)
    assert choice.needs_clarification and {c.debate_id for c in choice.clarify} == {"dAB1", "dAB2"}


def test_a_question_naming_no_heading_finds_nothing(debates):
    assert lookup(debates, "Who spoke about potholes?", "Lords") == []


def test_a_stopword_heading_never_matches_everything(debates):
    assert all(c.debate_id != "dThe" for c in lookup(debates, "Who spoke in the debate on the matter?"))


# --- the set route states its answer --------------------------------------------------------------------------------


@pytest.fixture
def sitting(debates, scratch):
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, party) VALUES (1, 'Baroness Altmann', 'Conservative'), "
                    "(2, 'Baroness Blake of Leeds', 'Labour')")
    for sid, seq, mnis, label, party, text in (
            ("s1", 0, 1, "Baroness Altmann (Con)", "Con", "To ask His Majesty's Government..."),
            ("s2", 1, 2, "Baroness Blake of Leeds (Lab)", "Lab", "My Lords, the Government...")):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_party, text, window_id, mnis_id) "
                        "VALUES (%s, 'dCA', %s, %s, %s, %s, 'WK', %s)", (sid, seq, label, party, text, mnis))
    return debates


def ask_set(dsn, question):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await RT.set_route(conn, question, K.Provenance(**PROV),
                                      filters=RO.extract_filters(question))  # fmt: skip
        finally:
            await conn.close()
    return run(go())


def test_the_opener_question_is_answered_in_words_with_the_members_name(sitting):
    answer = ask_set(sitting, "Which member spoke first in the Lords debate on Carer’s Allowance: Repayments on "
                              "29 July 2024?").answer
    assert answer.abstention is None
    assert answer.rendered.startswith("Baroness Altmann (Con) opened"), "the member's name, and the party printed that day"
    assert "opened the debate on Carer’s Allowance: Repayments in the Lords on 2024-07-29" in answer.rendered
    assert "2 members spoke in all" in answer.rendered and "Baroness Blake" not in answer.rendered,         "an opener question is answered with the opener, not the roster"


def test_a_plain_who_spoke_question_is_answered_in_words_too(sitting):
    """`rendered` is what a reader sees and what the evaluation grades: a roster must never reach them as nothing.
    """
    answer = ask_set(sitting, "Who spoke in the Lords debate on Carer's Allowance: Repayments on 29 July 2024?").answer
    assert answer.rendered.startswith("Speakers in the debate on Carer’s Allowance: Repayments (Lords, 2024-07-29)")
    assert "opened the debate" not in answer.rendered


# --- the overview: the combined answer accepts what the per-debate decks offered --------------------------------------


class _Hit:
    def __init__(self, chunk_id, debate_id):
        self.chunk_id, self.debate_id, self.rank, self.score = chunk_id, debate_id, 1, 1.0
        self.speech_id, self.sentence, self.passage = "s1", "Anchor sentence.", "Anchor sentence."
        self.seq, self.sent_idx = 1, 0
        self.date, self.house, self.party, self.office, self.heading = date(2026, 9, 15), "Commons", "Lab", None, "Fracking"
        self.speaker, self.url, self.column_ref, self.section, self.rerank_score = "A Member", None, None, "Chamber", 4.0
        self.sibling_chunk_ids = ()
        # The neighbour (seq 2) carries its own id, 102
        self.context_sentences = ((1, 0, "Anchor sentence.", None, 101), (2, 0, "The neighbour holds the fact.", None, 102))


def test_an_overview_keeps_a_claim_that_cites_a_neighbour_the_deck_offered(monkeypatch):
    citing_neighbour = K.Claim(
        id="c1", kind="said", text="A Member said the neighbour holds the fact.", speaker_name="A Member",
        citations=(K.Citation(speech_id="s1", chunk_ids=(102,), quote="The neighbour holds the fact."),),
        verification=K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True,
                                    window_ok=True, checked_by="test"))  # fmt: skip

    class _Generated:
        def __init__(self):
            self.answer = type("A", (), {"claims": (citing_neighbour,)})()

    async def fake_answer_question(question, deck, *a, **k):
        assert 102 in deck.chunk_ids, "the per-debate deck must offer the neighbour's id"
        return _Generated()

    async def fake_facets(conn, chunk_ids):
        return {"label": "passages found by search", "total": len(chunk_ids)}

    monkeypatch.setattr(RT.G, "answer_question", fake_answer_question)
    monkeypatch.setattr(RT, "facets", fake_facets)

    class _Client:
        async def chat(self, *a, **k):
            raise RuntimeError("no reduction in this test")

    overview = run(RT.overview_route(None, "What did both sides say about fracking?", [_Hit(101, "dF")], _Client(),
                                     K.Provenance(**{**PROV, "route": "overview"})))  # fmt: skip
    assert [c.id for c in overview.answer.claims] == ["c1"]
    assert {101, 102} <= set(overview.answer.retrieved_chunk_ids)


def test_an_overview_that_verified_nothing_declines_rather_than_showing_nothing(monkeypatch):
    """An overview in which no per-debate claim verified returned facet counts and a note — and no claims and no
    abstention, so the reader saw an empty answer. It declines, in the system's own words.
    """
    class _Generated:
        def __init__(self):
            self.answer = type("A", (), {"claims": ()})()

    async def fake_answer_question(question, deck, *a, **k):
        return _Generated()

    async def fake_facets(conn, chunk_ids):
        return {"label": "passages found by search", "total": len(chunk_ids)}

    monkeypatch.setattr(RT.G, "answer_question", fake_answer_question)
    monkeypatch.setattr(RT, "facets", fake_facets)

    class _Client:
        async def chat(self, *a, **k):
            raise AssertionError("nothing to reduce")

    overview = run(RT.overview_route(None, "What did MPs say about the Grenfell Tower fire in the week it happened?",
                                     [_Hit(101, "dF")], _Client(), K.Provenance(**{**PROV, "route": "overview"})))
    assert overview.answer.abstention == K.Abstention.INSUFFICIENT_EVIDENCE
    assert "none produced a claim that could be verified" in overview.answer.abstention_detail
    assert not overview.answer.facets, "an answer that abstains asserts nothing"
