"""Who opens a debate: never the Chair, never a point of order raised before the business, whatever the record's
printing of the Chair's name.
"""

from __future__ import annotations

import pytest

from procedure import opener as OP

PRESIDING = [
    "Mr Speaker", "Madam Speaker", "MR SPEAKER", "Mr. Speaker", "Mr Deputy Speaker (Mr Nigel Evans)",
    "Madam Deputy Speaker (Dame Rosie Winterton)", "Madam Deputy Speaker(Dame Rosie Winterton)",
    "Madam Deput y Speaker (Caroline Nokes)", "Madam  Deputy  Speaker", "The Speaker", "The Deputy Speaker (Lord Haskel)",
    "The Senior Deputy Speaker (Lord Gardiner of Kimble)", "The Lord Speaker (Lord McFall of Alcluith)",
    "The Lord Speaker", "The Chairman of Committees (Lord Laming)", "The Deputy Chairman of Committees (Baroness Garden)",
    "The Principal Deputy Chairman of Committees (Lord Haskel)", "The Chairman of Ways and Means (Dame Eleanor Laing)",
    "The First Deputy Chairman of Ways and Means (Judith Cummins)", "The Second Deputy Chairman of Ways and Means",
    "The Temporary Chair of Ways and Means (Sir Roger Gale)", "Chairman of Ways and Means", "The Chairman",
    "The Chair", "The Chair (Sir Christopher Chope)", "The First Deputy Chairman", "The Temporary Chair",
    "Mr Peter Dowd (in the Chair)", "Sir Roger Gale (in the Chair)", "Mr Speaker\u00a0", "Madam Deputy\u00a0Speaker",
    "Mr Speaker-Elect", "Madam Speaker-Elect",  # elected, presiding before the Royal approbation
]
NOT_PRESIDING = [
    "The Chair of the Treasury Committee (Dame Meg Hillier)", "The Chairman of the Liaison Committee (Sir Bernard Jenkin)",
    "The Speaker's Chaplain", "Mr Speaker\u2019s Secretary", "The Lord Speaker's Secretary", "Mr Speakman (Con)",
    "Sir Lindsay Hoyle (Chorley) (Ind)", "The Deputy Prime Minister (David Lammy)", "The Deputy Leader of the House",
    "Captain of the Honourable Corps of Gentlemen-at-Arms (Lord Kennedy of Southwark)", "Lord Speaker-Smith",
    "The Chairman of the Public Accounts Committee", "Caroline Nokes (Romsey and Southampton North) (Con)", "", None,
]


@pytest.mark.parametrize("printed", PRESIDING)
def test_every_printing_of_the_chair_is_presiding(printed):
    assert OP.presiding(printed, None)


@pytest.mark.parametrize("printed", NOT_PRESIDING)
def test_a_member_speaking_is_not_presiding(printed):
    assert not OP.presiding(printed, "member")


@pytest.mark.parametrize("role", OP.PRESIDING_ROLES)
def test_a_presiding_role_presides_whatever_the_name(role):
    assert OP.presiding("Caroline Nokes", role)


@pytest.mark.parametrize("text", [
    "On a point of order, Mr Speaker.", "  On a point of order, Madam Deputy Speaker. I seek your advice.",
    "Further to that point of order, Mr Speaker.", "Further to the point of order raised by my hon. Friend…",
    "On a further point of order, Mr Speaker.", "I rise on a point of order, Mr Dowd.",
    "Further to the hon. Gentleman\u2019s point of order, Madam Deputy Speaker",
])
def test_a_point_of_order_is_recognised_by_its_opening(text):
    assert OP.point_of_order(text)


@pytest.mark.parametrize("text", [
    "I beg to move, That this House has considered climate change.", "The point of order raised earlier was answered.",
    "My Lords, I beg to move that the Bill be now read a second time.", "", None,
])
def test_a_speech_that_mentions_a_point_of_order_is_not_one(text):
    assert not OP.point_of_order(text)


def _s(seq, name, *, mnis=1, role="member", text="I beg to move.", boiler=False, whole=True):
    """A contribution; `whole=False` is one too long to be read whole, of which only the opening is known."""
    return OP.Speech(debate_id="d", seq=seq, speech_id=f"s{seq}", mnis_id=mnis, speaker_name=name, speaker_role=role,
                     is_boilerplate=boiler, opening=text, whole=text if whole else None)


def test_the_chair_calls_the_debate_and_the_member_opens_it():
    found = OP.choose([_s(1, "Madam Deput y Speaker (Caroline Nokes)", mnis=4048, text="I call Luke Murphy."),
                       _s(2, "Luke Murphy (Basingstoke) (Lab)", mnis=5000)], business_is_point_of_order=False)
    assert found.speech.speaker_name.startswith("Luke Murphy") and found.chair_first


def test_a_point_of_order_before_the_business_does_not_open_it():
    found = OP.choose([_s(1, "Mr Deputy Speaker (Mr Nigel Evans)", role="deputy_speaker", text="Order."),
                       _s(2, "Sir Desmond Swayne (New Forest West) (Con)", mnis=7, text="On a point of order, Mr Deputy "
                          "Speaker. I am sorry to have delayed my point of order."),
                       _s(3, "Mr Deputy Speaker", role="deputy_speaker", text="I thank the right hon. Gentleman."),
                       _s(4, "Mr David Davis (Haltemprice and Howden) (Con)", mnis=8, text="I beg to move…")],
                      business_is_point_of_order=False)
    assert found.speech.mnis_id == 8 and found.points_of_order_before == 1


def test_where_the_business_is_a_point_of_order_its_raiser_opens_it():
    speeches = [_s(1, "Cat Eccles (Stourbridge) (Lab)", mnis=9, text="On a point of order, Madam Deputy Speaker."),
                _s(2, "Madam Deputy Speaker", role="deputy_speaker", text="I thank the hon. Member."),
                _s(3, "The Minister (Someone)", mnis=10, role="minister", text="I am grateful to the hon. Member.")]
    assert OP.choose(speeches, business_is_point_of_order=True).speech.mnis_id == 9


def test_a_record_holding_only_points_of_order_is_opened_by_the_first():
    speeches = [_s(1, "Alicia Kearns (Rutland and Stamford) (Con)", mnis=11, text="On a point of order, Madam Deputy "
                   "Speaker. Yesterday, I was left with no choice"),
                _s(2, "Madam Deputy Speaker", role="deputy_speaker", text="I thank the hon. Lady.")]
    assert OP.choose(speeches, business_is_point_of_order=False).speech.mnis_id == 11


def test_a_record_where_only_the_chair_speaks_has_no_opener():
    found = OP.choose([_s(1, "The Deputy Speaker (Lord Haskel)", role="lords_deputy", text="My Lords, …")],
                      business_is_point_of_order=False)
    assert found.speech is None and found.chair_first


def test_a_collective_attribution_never_opens_and_an_unresolved_individual_can():
    found = OP.choose([_s(1, "Hon. Members", mnis=None, text="Hear, hear!"), _s(2, "Noble Lords", mnis=None),
                       _s(3, "Mr J. Smith (Somewhere) (Con)", mnis=None, text="I beg to move.")],
                      business_is_point_of_order=False)
    assert found.speech.speaker_name.startswith("Mr J. Smith")


def test_a_formal_motion_by_a_member_opens_the_debate():
    """Boilerplate is a contribution's form, not its maker: a member moving formally is someone speaking."""
    found = OP.choose([_s(1, "Lord Lemos", mnis=5382, boiler=True, text="Moved by Lord Lemos"),
                       _s(2, "Lord Lemos", mnis=5382, text="My Lords, …")], business_is_point_of_order=False)
    assert found.speech.seq == 1


@pytest.mark.parametrize(("heading", "kind", "expected"), [
    ("Point of Order: Rectification Procedure", "other", True), ("Points of Order", "other", True),
    ("Anything", "point_of_order", True), ("Loan Charge 2019: Sir Amyas Morse Review", "general_debate", False),
    ("Business without Debate", "other", False), (None, None, False),
])
def test_the_business_is_a_point_of_order_by_its_heading_or_its_type(heading, kind, expected):
    assert OP.business_is_point_of_order(heading, kind) is expected


def test_members_who_spoke_never_count_the_chair_or_boilerplate():
    speeches = [_s(1, "Mr Speaker", mnis=467, role="speaker", text="Order."), _s(2, "A", mnis=1), _s(3, "B", mnis=2),
                _s(4, "A", mnis=1), _s(5, "C", mnis=3, boiler=True), _s(6, "Madam Deput y Speaker", mnis=4048)]
    assert OP.members_who_spoke(speeches) == 2


# The classifier and the stored roles agree with the one definition -----------------------------


@pytest.mark.parametrize("printed", [p for p in PRESIDING if p.strip()])
def test_the_ingest_classifier_tags_every_printing_of_the_chair_as_presiding(printed):
    from ingest.rules import CHAIR_ROLES, parse_speaker

    info = parse_speaker(printed)
    assert info.role in CHAIR_ROLES and info.is_chair, (printed, info)


@pytest.mark.parametrize("printed", [p for p in NOT_PRESIDING if p])
def test_the_ingest_classifier_never_tags_a_member_speaking_as_presiding(printed):
    from ingest.rules import CHAIR_ROLES, parse_speaker

    assert parse_speaker(printed).role not in CHAIR_ROLES, printed


def test_the_backfill_gives_every_stored_printing_of_the_chair_the_classifiers_presiding_role(scratch):
    """Migration 029 re-tags speeches stored before the classifier knew these forms, in SQL, so nothing is read out of
    the database; its patterns must say what `presiding` and the classifier say, form for form.
    """
    from pathlib import Path

    from corpus import TEST_WINDOW, ensure_window
    from ingest.rules import parse_speaker

    ensure_window(scratch)
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
                    "VALUES ('d', '2026-09-01', 'Commons', 'h', 'Commons Chamber', 'other', %s, 't')", (TEST_WINDOW,))
    names = [p for p in PRESIDING + NOT_PRESIDING if p]
    for n, name in enumerate(names):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_role, text, window_id) "
                        "VALUES (%s, 'd', %s, %s, %s, 'x', %s)", (f"s{n}", n, name, "office_holder", TEST_WINDOW))
    migration = next((Path(__file__).resolve().parents[1] / "db" / "migrations").glob("029_*.sql"))
    scratch.execute(migration.read_text(encoding="utf-8"))
    stored = dict(scratch.execute("SELECT speaker_name, speaker_role FROM speech").fetchall())
    for name in names:
        expected = parse_speaker(name).role if OP.presiding(name, None) else "office_holder"
        assert stored[name] == expected, (name, stored[name], expected)
    scratch.execute(migration.read_text(encoding="utf-8"))  # idempotent: a second run changes nothing
    assert dict(scratch.execute("SELECT speaker_name, speaker_role FROM speech").fetchall()) == stored


# --- QA pass 6 (S05): a contribution that says nothing does not open the business ------------------------------------
# Hansard prints what the reporter describes rather than records in italics: a member "rose—" and was not called, a
# minister "indicated assent", "King's consent signified.


@pytest.mark.parametrize("text", [
    "", "   ", None, "<p></p>", "&nbsp;", "<em>indicated assent.</em>", "<em>indicated dissent.</em>", "<em>rose—</em>",
    "<em>King\u2019s consent signified.</em>", "<em>King's consent signified.</em>",
    "<p><em>Prince of Wales\u2019s consent signified.</em></p>",  # a form the corpus does not hold
    "<i>shook his head.</i>", '<span class="column-number" data-column-number="1341"></span><em>rose\u2014</em>',
    "<em>indicated assent.</em> <em>[Laughter.]</em>", "<EM>rose\u2014</EM>",
    # a reporter's note beside a stage direction is noise around the act; a note with no word in it is no note
    "<em>rose\u2014</em> <em>[Interruption.]</em>", "<em>[Laughter.]</em> <em>indicated dissent.</em>", "<em>[]</em>",
    "<em>[ . ]</em>",
    # A note alone is judged by what it says; one about anything but a gap in the transcript is the reporter describing
    # something other than the member's words, however its markup is split or nested
    "<em>[Interruption.]</em> ", '<span class="column-number" data-column-number="1341"></span><em>[Interruption.]</em>',
    "<em>[a [nested] note]</em>", "<em>[Laughter.]</em>", "[Hon. Members: “Hear, hear!”]",
    # QA pass 7 (L5): nesting, mismatched pairs, a trailing open tag with nothing in it, a comment holding ">"
    "<em>Hear, <em>hear</em>.</em>", "<i>rose\u2014</em>", "<em>rose\u2014</em><em>", "<em>", "<!-- a > b -->",
    "<em>3 < 4</em>",  # a bare "<" is text, as HTML reads it, not the start of a tag
])  # fmt: skip
def test_an_empty_contribution_or_one_of_stage_directions_alone_says_nothing(text):
    assert OP.non_speech(text)


@pytest.mark.parametrize("text", [
    "I beg to move.", "Mr Speaker, the <em>Daily Mail</em> reports otherwise.",
    "<em>rose—</em> I am grateful, Mr Speaker.",
    # moving the business is the act that opens it, and a point of order is the member's own words, in italics or not
    "<em>Moved by Lord Lemos</em>", "<em>Moved formally.</em>", "<em>On a point of order, Mr Speaker.</em>",
    "<em>unclosed italics", "Thank you.",
    # asking and presenting open business as moving does (a Lords question, a bill or petition), however printed
    "<em>Asked by Baroness Example</em>", "<em>To ask His Majesty\u2019s Government what plans they have.</em>",
    "<em>Bill presented by the Minister.</em>",
    # a stage direction is a phrase, never a passage: a long run of italics is a speech (a quotation, Welsh)
    "<em>Diolch yn fawr iawn, Llywydd, am y cyfle i siarad heddiw am y Bil pwysig hwn i bobl Cymru.</em>",
    # QA pass 7 (Med-3): every tag is a word boundary, so neither an act nor a word count is lost across runs
    "<p><em>Amendment 2</em></p><p><em>Moved by Lord Xavier</em></p>", "<em>Clause 1</em><br><em>Asked by Lord X</em>",
    "<em>one</em> <em>two three four five six seven eight nine ten eleven twelve thirteen</em>",
    "<!-- a > b -->I beg to move.", "<em>an unclosed run of words",
    "<em>rose\u2014</ em>",  # "</ em>" is no end tag in HTML, so the italics are never closed
    # S05's real-data check: a note of a gap in the transcript standing alone in a member's place stands where their
    # words were, so the member had the floor (members on the virtual floor in 2020, "[Inaudible.]"), however the record
    # splits its markup, closes it or nests it.
    "<em>[Inaudible.]</em>", "<em>[</em><em>Inaudible</em><em>.</em><em>]</em>", "<em>[Inaudible</em><em>.</em><em>]</em>",
    "<em>[Inaudible].</em>", "<em>[Inaudible.</em>",
    '<span class="column-number" data-column-number="1341"></span><em>[Inaudible.]</em>',
    "<em>[Technical difficulties: the Member\u2019s connection was lost.]</em>", "<em>[an [inaudible] passage]</em>",
    "<p><em>[Inaudible.]</em></p><p><em>[Inaudible.]</em></p>",
])  # fmt: skip
def test_words_spoken_or_an_act_of_moving_are_a_speech(text):
    assert not OP.non_speech(text)


def test_a_member_who_rose_and_was_not_called_does_not_open_the_debate():
    found = OP.choose([_s(1, "Mr Speaker", mnis=467, role="speaker", text="Order."),
                       _s(2, "Sir Desmond Swayne (New Forest West) (Con)", mnis=7, text="<em>rose\u2014</em>"),
                       _s(3, "Mr Speaker", mnis=467, role="speaker", text="I call the Minister."),
                       _s(4, "The Minister (Ann Minister)", mnis=8, text="I beg to move…")],
                      business_is_point_of_order=False)
    assert found.speech.mnis_id == 8, found


def test_a_stage_direction_then_the_same_members_speech_opens_with_the_speech():
    found = OP.choose([_s(1, "Mr Speaker", mnis=467, role="speaker", text="Order."),
                       _s(2, "The Minister (Ann Minister)", mnis=8, text="<em>indicated assent.</em>"),
                       _s(3, "The Minister (Ann Minister)", mnis=8, text="I beg to move…")],
                      business_is_point_of_order=False)
    assert (found.speech.mnis_id, found.speech.seq) == (8, 3)


def test_a_member_who_spoke_inaudibly_spoke_and_opens_where_first():
    """S05's real-data check: Richard Drax's only contribution on 29 April 2020 is "[Inaudible.]". He had the floor, so
    he spoke: on the roster's count, and the opener where his is the first member's contribution.
    """
    speeches = [_s(1, "Mr Speaker", mnis=467, role="speaker", text="Order."),
                _s(2, "Richard Drax (South Dorset) (Con) [V]", mnis=1, text="<em>[Inaudible.]</em>"),
                _s(3, "B", mnis=2, text="<em>rose—</em>"), _s(4, "C", mnis=3, text="I thank the Minister.")]
    assert OP.members_who_spoke(speeches) == 2
    assert OP.choose(speeches, business_is_point_of_order=False).speech.mnis_id == 1


def test_the_kings_consent_signified_by_a_minister_does_not_open_a_members_third_reading():
    found = OP.choose([_s(1, "The Solicitor General (Ann Minister)", mnis=8, text="<em>King\u2019s consent signified.</em>"),
                       _s(2, "Mr Member in Charge (Somewhere) (Lab)", mnis=9,
                          text="I beg to move, That the Bill be now read the Third time.")],
                      business_is_point_of_order=False)
    assert found.speech.mnis_id == 9


def test_an_empty_contribution_does_not_open_the_debate():
    found = OP.choose([_s(1, "Ann Tabler (Somewhere) (Con)", mnis=10, text=""),
                       _s(2, "Bob Mover (Somewhere) (Lab)", mnis=11, text="I beg to move…")],
                      business_is_point_of_order=False)
    assert found.speech.mnis_id == 11


def test_a_record_whose_members_only_rose_or_nodded_has_no_opener():
    found = OP.choose([_s(1, "Mr Speaker", mnis=467, role="speaker", text="Order."),
                       _s(2, "A Member (Somewhere) (Con)", mnis=7, text="<em>rose\u2014</em>")],
                      business_is_point_of_order=False)
    assert found.speech is None and found.chair_first


def test_a_contribution_too_long_to_read_whole_is_never_taken_for_a_stage_direction():
    """Only a contribution read whole can be judged to say nothing: its opening alone may be all italics."""
    long_ = _s(1, "Ann Member (Somewhere) (Con)", mnis=12, text="<em>Diolch yn fawr", whole=False)
    assert not long_.says_nothing
    assert OP.choose([long_, _s(2, "Bob (Somewhere) (Lab)", mnis=13)], business_is_point_of_order=False).speech.mnis_id \
        == 12


# --- QA pass 6 (S05): the Chair's own statement is the business, never a debate a member opened
# ------------------------


@pytest.mark.parametrize(("heading", "kind", "expected"), [
    ("Speaker\u2019s Statement", "other", True), ("Speaker's Statement", None, True),
    ("Deputy Speaker\u2019s Statement", "other", True), ("Lord Speaker\u2019s Statement", "other", True),
    ("Mr Speaker's Statement", "other", True), ("Speaker's Statements", "other", True),
    ("Statement by the Speaker", "other", True), ("Anything", "speaker_statement", True),
    ("Speaker Statement", "other", True), ("Speakers\u2019 Statement", "other", True),  # however the possessive prints
    ("Speaker's Conference", "other", False), ("Statement on the Speaker's Conference", "other", False),
    # QA pass 7: a committee's chair is a member speaking (Med-1); the presiding offices in every printing (L3)
    ("Statement by the Chair of the Liaison Committee", "other", False),
    ("Chair\u2019s Statement: Public Accounts Committee Report", "other", False),
    ("Statement by the Chairman of the Conservative Party", "other", False),
    ("Statement from the Speaker of the Ukrainian Parliament", "other", False),
    ("Statements by the Speaker's Committee on the Electoral Commission", "other", False),
    ("Speakers Statement", "other", True), ("Speaker: Statement", "other", True), ("Lord Speaker: Statement", "other", True),
    ("Chairman of Ways and Means\u2019 Statement", "other", True), ("Chairman of Committees\u2019 Statement", "other", True),
    ("First Deputy Chairman\u2019s Statement", "other", True), ("Chair's Statement", "other", True),
    ("Speaker\u2019s Statement: Hybrid Proceedings", "other", True),
    ("Business Statement", "business_statement", False), ("Points of Order", "point_of_order", False),
    (None, None, False),
])  # fmt: skip
def test_the_business_is_the_chairs_own_statement_by_its_type_or_its_heading(heading, kind, expected):
    assert OP.business_of_the_chair(heading, kind) is expected


def test_a_point_of_order_after_the_speakers_statement_never_opens_it():
    speeches = [_s(1, "Mr Speaker", mnis=467, role="speaker", text="I wish to make a short statement about…"),
                _s(2, "Sir Desmond Swayne (New Forest West) (Con)", mnis=7, text="On a point of order, Mr Speaker."),
                _s(3, "Mr Speaker", mnis=467, role="speaker", text="I thank him."),
                _s(4, "Ann Member (Somewhere) (Lab)", mnis=8, text="I welcome your statement, Mr Speaker.")]
    found = OP.choose(speeches, business_is_point_of_order=False, business_of_the_chair=True)
    assert found.speech is None and found.chair_business and found.statement.seq == 1, found
    assert OP.Record("d", "Commons", None, "Speaker\u2019s Statement", "other", "Commons Chamber", 1,
                     tuple(speeches)).found.chair_business
