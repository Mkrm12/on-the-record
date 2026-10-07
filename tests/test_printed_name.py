"""A speaker is named as the record printed them, not by a title they took later."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from answer import cards as C
from answer import names as N
from answer import pipeline as PL
from answer import roster as RO
from app import safety
from ingest.runtime import run
from retrieval.hybrid import Hit, SearchConfig, SearchResult

ROOT = Path(__file__).resolve().parents[1]

# --- what counts as a full printed name --------------------------------------------------------------------------


@pytest.mark.parametrize("name, full", [
    ("Michael Gove", True), ("Dame Eleanor Laing", True), ("Mr Kevan Jones", True), ("Dr Alan Whitehead", True),
    ("Sir Graham Brady", True), ("Mrs Theresa May", True), ("Iain Duncan Smith", True), ("Dr Thérèse Coffey", True),
    ("Mr Jones", False), ("Dr Whitehead", False), ("Mrs May", False), ("Dr. Whitehead", False), ("Sir Edward", False),
    ("Gove", False), ("", False), ("   ", False),
])  # fmt: skip
def test_a_full_printed_name_has_a_forename_after_any_courtesy_title(scratch, name, full):
    assert scratch.execute("SELECT printed_full_name(%s)", (name,)).fetchone()[0] is full


def test_no_name_is_not_a_full_name(scratch):
    assert scratch.execute("SELECT coalesce(printed_full_name(NULL), false)").fetchone()[0] is False


# --- the scratch record ---------------------------------------------------------------------------------------------

MEMBERS = [  # mnis_id, name now, House now
    (1571, "Lord Gove", "Lords"),
    (390, "Baroness Laing of Elderslie", "Lords"),
    (360, "Lord Beamish", "Lords"),
    (62, "Lord Whitehead", "Lords"),
    (4619, "Lord Jack of Courance", "Lords"),
    (77, "Lord Other", "Lords"),
    (88, "Lord Redwood", "Lords"),
    (99, "Lord Tie", "Lords"),
    (10, "Baroness Altmann", "Lords"),
    (11, "Sir Lindsay Hoyle", "Commons"),
    (12, "A Member", None),
]

SPEECHES = [  # id, debate, mnis_id, label as printed, person parsed from it
    ("gove1", "dC", 1571, "Michael Gove", "Michael Gove"),
    ("gove2", "dC", 1571, "The Secretary of State for Levelling Up (Michael Gove)", "Michael Gove"),
    ("laing1", "dC", 390, "Madam Deputy Speaker (Dame Eleanor Laing)", "Dame Eleanor Laing"),
    ("laing2", "dC", 390, "Madam Deputy Speaker", None),
    ("jones1", "dC", 360, "Mr Kevan Jones (North Durham) (Lab)", "Mr Kevan Jones"),
    ("jones2", "dC", 360, "Mr Jones", "Mr Jones"),
    ("jones3", "dC2", 360, "Mr Jones", "Mr Jones"),
    ("white1", "dC", 62, "Dr Whitehead", "Dr Whitehead"),
    ("jack1", "dC", 4619, "Mr Jack", "Mr Jack"),
    ("other1", "dC", 77, "Mr Other", "Mr Other"),
    ("other2", "dL", 77, "Lord Other of Somewhere (Con)", "Lord Other of Somewhere"),
    ("red1", "dC", 88, "Sir John Redwood", "Sir John Redwood"),
    ("red2", "dC", 88, "John Redwood", "John Redwood"),
    ("red3", "dC2", 88, "John Redwood", "John Redwood"),
    ("red4", "dC2", 88, "Mr Redwood", "Mr Redwood"),
    ("tie1", "dC", 99, "Bob Tie", "Bob Tie"),
    ("tie2", "dC", 99, "Alan Tie", "Alan Tie"),
    ("tie3", "dC2", 99, "Mr Tie", "Mr Tie"),
    ("alt1", "dL", 10, "Baroness Altmann (Con)", "Baroness Altmann"),
    ("hoyle1", "dC", 11, "Mr Speaker", None),
    ("anon1", "dC", 12, "Mr Member", "Mr Member"),
    ("chair1", "dC", None, "The Deputy Speaker", None),
    ("ghost1", "dC", 999, "Some Label (Somewhere) (Lab)", "Some Label"),
]


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE division CASCADE")
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WN', 'test', '2021-01-01', '2024-12-31', 'dev_test')")
    for did, house, day in (("dC", "Commons", "2022-03-01"), ("dC2", "Commons", "2023-05-02"),
                            ("dL", "Lords", "2024-10-01")):
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                        "VALUES (%s, %s, %s, 'A Debate', 'Chamber', 'WN', 'test')", (did, day, house))
    for mnis_id, name, house in MEMBERS:
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (%s, %s, %s)", (mnis_id, name, house))
    for seq, (sid, did, mnis_id, label, person) in enumerate(SPEECHES):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_person, text, window_id, mnis_id)"
                        " VALUES (%s, %s, %s, %s, %s, 'Words.', 'WN', %s)", (sid, did, seq, label, person, mnis_id))
    # The Votes API names a member as they were in THAT House's division lists.
    scratch.execute("INSERT INTO division (id, source_id, house, division_date) VALUES (5, 5, 'Commons', '2022-03-01'), "
                    "(1000006, 6, 'Lords', '2024-10-01')")
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name) VALUES "
                    "(5, 62, 'aye', 'Alan Whitehead'), (1000006, 77, 'no', 'Lord Other of Elsewhere'), "
                    "(5, 390, 'no', 'Eleanor Laing')")
    return scratch_dsn


def printed(scratch, speech_id):
    return scratch.execute("SELECT speaker_as_printed(%s)", (speech_id,)).fetchone()[0]


@pytest.mark.parametrize("speech_id, want", [
    ("gove1", "Michael Gove (now Lord Gove)"),
    ("gove2", "Michael Gove (now Lord Gove)"),
    ("laing1", "Dame Eleanor Laing (now Baroness Laing of Elderslie)"),
    ("jones1", "Mr Kevan Jones (now Lord Beamish)"),
])  # fmt: skip
def test_a_member_who_has_since_changed_house_is_named_as_printed_then(record, scratch, speech_id, want):
    assert printed(scratch, speech_id) == want


def test_a_label_with_no_name_is_completed_from_the_members_other_labels_in_that_house(record, scratch):
    """Hansard's own label outranks the Votes API's "Eleanor Laing": the record's words first, the division list only
    where Hansard never printed a full name.
    """
    assert printed(scratch, "laing2") == "Dame Eleanor Laing (now Baroness Laing of Elderslie)"


def test_a_terse_label_is_completed_from_the_members_fullest_printed_form(record, scratch):
    assert printed(scratch, "jones2") == printed(scratch, "jones3") == "Mr Kevan Jones (now Lord Beamish)"


def test_the_speechs_own_full_label_beats_a_more_frequent_one(record, scratch):
    """"Sir John Redwood" is what was printed on THIS speech (after his knighthood); the commoner form is not."""
    assert printed(scratch, "red1") == "Sir John Redwood (now Lord Redwood)"
    assert printed(scratch, "red4") == "John Redwood (now Lord Redwood)", "the most frequent full form completes a terse label"


def test_a_tie_between_full_forms_is_broken_the_same_way_every_time(record, scratch):
    assert printed(scratch, "tie3") == "Alan Tie (now Lord Tie)"


def test_the_division_lists_of_that_house_complete_a_member_hansard_only_printed_tersely(record, scratch):
    assert printed(scratch, "white1") == "Alan Whitehead (now Lord Whitehead)"


def test_a_name_from_the_other_house_is_never_used(record, scratch):
    """His Lords label and his Lords division name are both full — and both are the wrong House for this speech."""
    assert printed(scratch, "other1") == "Mr Other (now Lord Other)"


def test_with_nothing_fuller_the_terse_label_is_kept_and_the_identity_still_given(record, scratch):
    assert printed(scratch, "jack1") == "Mr Jack (now Lord Jack of Courance)"


@pytest.mark.parametrize("speech_id, want", [
    ("alt1", "Baroness Altmann"),  # still in the House she spoke in: the name she has
    ("hoyle1", "Sir Lindsay Hoyle"),  # the chair, resolved to a member still in the Commons
    ("anon1", "A Member"),  # no House recorded: nothing to contradict, nothing added
    ("chair1", "The Deputy Speaker"),  # no member: the label, as before
    ("ghost1", "Some Label (Somewhere) (Lab)"),  # an id with no member row: the label, as before
])  # fmt: skip
def test_everyone_else_is_named_exactly_as_before(record, scratch, speech_id, want):
    assert printed(scratch, speech_id) == want


def test_an_unknown_speech_has_no_name(record, scratch):
    assert printed(scratch, "no-such-speech") is None


# --- the deck -------------------------------------------------------------------------------------------------------


def hit(speech_id, speaker, rank=1):
    return Hit(rank=rank, score=1.0, chunk_id=rank, speech_id=speech_id, debate_id="dC", seq=0, sent_idx=0,
               sentence="Words.", passage="Words.", speaker=speaker, party=None, office=None, heading="A Debate",
               house="Commons", date=date(2022, 3, 1), column_ref=None, url=None, rrf=0.1, rerank_score=None, legs={},
               sibling_chunk_ids=(), also_in_speeches=())  # fmt: skip


def test_the_deck_shows_the_printed_name_and_keeps_retrievals_where_none_is_given():
    deck = C.build_deck([hit("gove1", "Lord Gove", 1), hit("x", "Retrieval's Name", 2)],
                        speakers={"gove1": "Michael Gove (now Lord Gove)"})  # fmt: skip
    assert [c.speaker for c in deck.cards] == ["Michael Gove (now Lord Gove)", "Retrieval's Name"]
    assert "Michael Gove (now Lord Gove)" in deck.render() and "| Lord Gove |" not in deck.render()


def as_printed(dsn, ids):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await N.as_printed(conn, ids)
        finally:
            await conn.close()
    return run(go())


def test_the_lookup_returns_every_speech_it_can_name(record):
    got = as_printed(record, ["gove1", "alt1", "chair1", "no-such-speech"])
    assert got == {"gove1": "Michael Gove (now Lord Gove)", "alt1": "Baroness Altmann", "chair1": "The Deputy Speaker"}


def test_no_speeches_asks_the_database_nothing():
    class _Refuses:
        async def execute(self, *a, **k):
            raise AssertionError("an empty deck must not query")
    assert run(N.as_printed(_Refuses(), [])) == {}


class _Pool:
    def __init__(self, dsn):
        self.dsn = dsn

    def connection(self):
        import contextlib

        import psycopg

        @contextlib.asynccontextmanager
        async def cm():
            conn = await psycopg.AsyncConnection.connect(self.dsn)
            try:
                yield conn
            finally:
                await conn.close()
        return cm()


class _Searcher:
    def __init__(self, dsn, hits):
        self.pool, self._hits = _Pool(dsn), hits

    async def search(self, question, *, top_k, config):
        return SearchResult(query=question, searched=question, hits=list(self._hits))


class _Frozen:
    def search_config(self):
        return SearchConfig()


class _Runtime:
    def __init__(self, dsn, hits):
        self.frozen, self.searcher, self.serving_timeout_ms = _Frozen(), _Searcher(dsn, hits), None


def test_every_deck_the_pipeline_builds_names_speakers_as_printed(record):
    deck, _ = run(PL.search_deck(_Runtime(record, [hit("gove1", "Lord Gove"), hit("alt1", "Baroness Altmann", 2)]),
                                 "What did Michael Gove say?"))  # fmt: skip
    assert [c.speaker for c in deck.cards] == ["Michael Gove (now Lord Gove)", "Baroness Altmann"]


# --- the roster -----------------------------------------------------------------------------------------------------


def roster_of(dsn, fn, *args):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await fn(conn, *args)
        finally:
            await conn.close()
    return run(go())


def test_the_opener_is_named_as_printed(record):
    assert roster_of(record, RO.first_speaker, "dC")["display_name"] == "Michael Gove (now Lord Gove)"


def test_the_roster_names_every_speaker_as_printed(record):
    names = [e.speaker_name for e in roster_of(record, RO.roster, ["dC"]).entries]
    assert "Michael Gove (now Lord Gove)" in names and "Mr Kevan Jones (now Lord Beamish)" in names
    assert not any(re.match(r"(Lord|Baroness) ", n) for n in names if "(now" not in n and n != "Baroness Altmann")


# --- the answer text: one attribution, whichever of the names the claim leads with --------------------------------

PROV = dict(route="retrieval", resolved_filters={}, corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76",
            recipe="r", prompt_version="p", model_id="m")


def rendered(text, speaker):
    from answer import contract as K

    verified = K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True, window_ok=True,
                              checked_by="test")
    claim = K.Claim(id="c1", kind="summary", text=text, speaker_name=speaker, date=date(2022, 3, 1),
                    citations=(K.Citation(speech_id="gove1", chunk_ids=(1,), quote="the scheme will close"),),
                    verification=verified)  # fmt: skip
    return K.Answer(question="q", provenance=K.Provenance(**PROV), claims=(claim,)).render()


@pytest.mark.parametrize("text", [
    "Michael Gove told the Commons that the scheme will close.",
    "Lord Gove told the Commons that the scheme will close.",
    "Michael Gove (now Lord Gove) told the Commons that the scheme will close.",
])  # fmt: skip
def test_a_claim_naming_the_speaker_either_way_is_attributed_once(text):
    assert rendered(text, "Michael Gove (now Lord Gove)") == f"{text} [2022-03-01]"


def test_a_claim_that_names_nobody_is_attributed_with_the_printed_name():
    assert rendered("The scheme will close.", "Michael Gove (now Lord Gove)") == \
        "Michael Gove (now Lord Gove) said: The scheme will close. [2022-03-01]"


def test_a_label_carrying_seat_and_party_is_not_attributed_twice():
    """A pre-existing case of the same defect: an unmapped speaker's card shows Hansard's whole label."""
    text = "Lincoln Jopp said that the scheme will close."
    assert rendered(text, "Lincoln Jopp (Spelthorne) (Con)") == f"{text} [2022-03-01]"


@pytest.mark.parametrize("text", ["Michael Gove said that the scheme will close.",
                                  "Lord Gove said that the scheme will close.",
                                  "Gove said that the scheme will close."])
def test_entailment_is_asked_about_the_substance_whichever_name_leads(text):
    from answer import verify as V

    assert V.strip_attribution(text, "Michael Gove (now Lord Gove)") == "The scheme will close."


def test_a_now_name_is_never_read_from_a_name_that_merely_contains_now():
    from answer import contract as K

    assert K.attribution_names("Mr Snow (Con)") == ("Mr Snow (Con)", "Mr Snow")


# --- no reader-facing path names a speaker by the current name alone ------------------------------------------------


@pytest.mark.parametrize("pattern", [r"coalesce\(\s*named\.name", r"coalesce\(\s*min\(\s*m\.name", r"coalesce\(\s*cc\.member_name"])
def test_the_answer_layer_never_names_a_speaker_by_the_current_name_alone(pattern):
    """The retrieval SQL still does (the reranker reads it, and retrieval is frozen); nothing a reader sees may."""
    offenders = [p.name for p in (ROOT / "src" / "answer").glob("*.py")
                 if re.search(pattern, p.read_text(encoding="utf-8"), re.IGNORECASE)]
    assert offenders == []
