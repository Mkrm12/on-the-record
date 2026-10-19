"""Who opened each debate, re-derived from the Hansard contributions as fetched."""

from __future__ import annotations

import ast
import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from eval.oracles import openers as OO

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("printed", [
    "Mr Speaker", "Madam Speaker", "Madam Deputy Speaker (Caroline Nokes)", "Mr Deputy Speaker (Mr Nigel Evans)",
    "The Deputy Speaker (Lord Faulkner of Worcester)", "The Senior Deputy Speaker", "The Lord Speaker (Lord Forsyth of Drumlean)",
    "The Deputy Chairman of Committees (Baroness Fookes)", "The Chairman of Ways and Means (Dame Eleanor Laing)",
    "The First Deputy Chairman of Ways and Means", "The Chair", "The Chairman", "The Temporary Chair (Sir Mark Hendrick)",
    "Sir Roger Gale (in the Chair)",
    # the short repeat forms, printed after the first mention
    "The First Deputy Chairman", "The Second Deputy Chairman", "The Deputy Chairman",
    # the Lords' pre-2016 title, inside the ten-year span, and the temporary Ways and Means chair (QA pass 3)
    "The Principal Deputy Chairman of Committees (Lord Brabazon of Tara)", "The Temporary Chair of Ways and Means",
])  # fmt: skip
def test_the_chair_as_hansard_prints_it_presides(printed):
    assert OO.presiding(printed)


@pytest.mark.parametrize("printed", [
    "The Chair of the Treasury Committee (Dame Meg Hillier)",  # a member speaking, not presiding
    "Captain of the Honourable Corps of Gentlemen-at-Arms (Lord Kennedy of Southwark)",  # a government whip
    "The Leader of the House of Commons (Sir Alan Campbell)", "Baroness Smith of Basildon", "Jim Shannon (Strangford) (DUP)",
    "The Parliamentary Under-Secretary of State for Health and Social Care (Ashley Dalton)", "",
    "The Speaker's Chaplain", "Mr Speaker's Secretary",  # the Speaker's staff, who do not preside (QA pass 3)
])  # fmt: skip
def test_a_member_or_minister_does_not_preside(printed):
    assert not OO.presiding(printed)


def contribution(order, mid, printed, text="Words.", ext=None, day="2026-09-08", debate="D1", heading="Housing",
                 section_order=None):
    return {"OrderInDebateSection": order, "MemberId": mid, "AttributedTo": printed, "MemberName": printed,
            "ContributionTextFull": text, "ContributionExtId": ext or f"{debate}-{order}", "DebateSectionExtId": debate,
            "DebateSection": f" {heading}", "SittingDate": f"{day}T00:00:00", "House": "Commons",
            "Section": "Commons Chamber",
            "DebateSectionOrder": None if section_order is None else str(section_order)}  # fmt: skip


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"Results": rows, "TotalResultCount": len(rows)}, fh)


@pytest.fixture
def raw(tmp_path):
    base = tmp_path / "contributions"
    _write(base / "W0" / "2026-09-08" / "asc_000000.json.gz", [
        contribution(2, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order. I call the Minister."),
        contribution(1, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order."),
        contribution(3, 5001, "The Minister for Housing (Ann Minister)", "I beg to move..."),
        contribution(4, 5002, "Jim Shannon (Strangford) (DUP)"),
    ])
    # the same contribution on a second page: counted once
    _write(base / "W0" / "2026-09-08" / "asc_000100.json.gz", [contribution(4, 5002, "Jim Shannon (Strangford) (DUP)")])
    _write(base / "W5" / "2026-05-20" / "asc_000000.json.gz", [contribution(1, 1, "Mr Speaker", debate="S1", day="2026-05-20")])
    _write(base / "W0" / "2026-06-10" / "asc_000000.json.gz", [contribution(1, 1, "Mr Speaker", debate="S2", day="2026-06-10")])
    return base


def test_a_debate_opened_by_the_chair_has_its_first_member_as_opener(raw):
    (t,) = list(OO.truths(raw))
    assert (t.debate_id, t.house, t.date, t.heading) == ("D1", "Commons", date(2026, 9, 8), "Housing")
    assert [c.order for c in t.contributions] == [1, 2, 3, 4], "in the record's order, each contribution once"
    assert t.presiding_first and t.opener.mnis_id == 5001 and t.presiding_ids == {4048}


def test_the_oracle_never_opens_the_sealed_window(raw, monkeypatch):
    opened = []
    real = gzip.open

    def spy(path, *a, **k):
        opened.append(Path(path))
        return real(path, *a, **k)

    monkeypatch.setattr(OO.gzip, "open", spy)
    assert {t.debate_id for t in OO.truths(raw)} == {"D1"}  # S1 in W5, S2 dated inside it though filed in W0
    assert not any("W5" in p.parts for p in opened)


def test_the_opener_oracle_shares_no_code_with_ingest_or_the_answer_layer():
    tree = ast.parse((ROOT / "src" / "eval" / "oracles" / "openers.py").read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m.startswith(("ingest", "answer", "retrieval", "goldset")) for m in modules), modules


# --- S05: the record's misprints, points of order, and threads ------------------------------------------


@pytest.mark.parametrize("printed", ["Madam Deput y Speaker (Caroline Nokes)", "Madam Deputy Speaker(Dame Rosie Winterton)",
                                     "Mr. Speaker", "MR SPEAKER", "The  Chairman", "Mr Speaker\u00a0"])
def test_the_chair_presides_however_the_record_spaces_or_cases_it(printed):
    """19 March 2026's "Madam Deput y Speaker" (a space inside the word) evaded this regex and the route's filter
    alike, so MR-08 could not see the Deputy Speaker named as an opener.
    """
    assert OO.presiding(printed)


def _truth(*contributions, heading="Loan Charge 2019", did="D1", order=None):
    return OO.DebateTruth(did, "Commons", date(2020, 3, 19), heading, "Commons Chamber", tuple(contributions),
                          section_order=order)


def test_a_point_of_order_before_the_business_does_not_open_it():
    t = _truth(OO.Contribution(1, 1, "Mr Deputy Speaker (Mr Nigel Evans)", "Order."),
               OO.Contribution(2, 7, "Sir Desmond Swayne (New Forest West) (Con)", "On a point of order, Mr Deputy "
                               "Speaker. I am sorry to have delayed my point of order."),
               OO.Contribution(3, 1, "Mr Deputy Speaker (Mr Nigel Evans)", "I thank him."),
               OO.Contribution(4, 8, "Mr David Davis (Haltemprice and Howden) (Con)", "I beg to move…"))
    assert t.opener.mnis_id == 8 and t.presiding_first


def test_where_the_business_is_a_point_of_order_or_nothing_else_is_recorded_its_raiser_opens_it():
    raised = OO.Contribution(1, 9, "Cat Eccles (Stourbridge) (Lab)", "On a point of order, Madam Deputy Speaker.")
    reply = OO.Contribution(3, 10, "The Minister (Someone)", "I am grateful.")
    assert _truth(raised, reply, heading="Point of Order: Rectification Procedure").opener.mnis_id == 9
    assert _truth(raised, heading="Business without Debate").opener.mnis_id == 9


def test_threads_are_contiguous_records_under_one_heading_in_the_days_order(tmp_path):
    base = tmp_path / "contributions"
    _write(base / "W0" / "2026-09-14" / "asc_000000.json.gz", [
        contribution(1, 4048, "Madam Deputy Speaker (Caroline Nokes)", "Order.", debate="SR", heading="Student Loans",
                     day="2026-09-14", section_order=27),
        contribution(2, 5001, "The Parliamentary Secretary (A Minister)", "I beg to move...", debate="SR",
                     heading="Student Loans", day="2026-09-14", section_order=27),
        contribution(1, 5003, "The Exchequer Secretary (B Minister)", "I beg to move...", debate="TR",
                     heading="Student Loans", day="2026-09-14", section_order=28),
        contribution(1, 4048, "Madam Deputy Speaker", "Statement.", debate="ST", heading="Something Else",
                     day="2026-09-14", section_order=29),
        contribution(1, 4048, "Madam Deputy Speaker", "Order.", debate="SR2", heading="Student Loans",
                     day="2026-09-14", section_order=30),
        contribution(2, 5004, "Jim Shannon (Strangford) (DUP)", "Words.", debate="SR2", heading="Student Loans",
                     day="2026-09-14", section_order=30),
    ])
    threads = OO.threads(OO.truths(base))
    by = {t.debate_ids: t for t in threads}
    assert set(by) == {("SR", "TR"), ("ST",), ("SR2",)}
    joined = by[("SR", "TR")]
    assert joined.opener.mnis_id == 5001 and joined.presiding_first and joined.presiding_ids == {4048}
    assert joined.members == {5001, 5003}
    assert by[("SR2",)].opener.mnis_id == 5004


def test_a_record_with_no_place_in_the_days_order_is_never_joined(raw):
    assert [t.debate_ids for t in OO.threads(OO.truths(raw))] == [("D1",)]


def test_a_bills_proceedings_resumed_after_a_statement_are_one_thread_to_the_oracle_too():
    def rec(did, order, heading):
        return OO.DebateTruth(did, "Lords", date(2026, 9, 7), heading, "Lords Chamber",
                              (OO.Contribution(1, 1, "Lord A", "My Lords"),), section_order=order)

    found = OO.threads([rec("r18", 18, "Financial Services and Markets Bill [HL]"), rec("st", 19, "Local Government"),
                        rec("r20", 20, "Financial Services and Markets Bill [HL]"), rec("b1", 21, "Business of the House"),
                        rec("x", 22, "A Statement"), rec("b2", 23, "Business of the House")])
    assert sorted(t.debate_ids for t in found) == [("b1",), ("b2",), ("r18", "r20"), ("st",), ("x",)]


# --- QA pass 6 (S05): contributions that say nothing, and the Chair's own statement ----------------------------------


@pytest.mark.parametrize("text", [
    "", "<em>indicated assent.</em>", "<em>indicated dissent.</em>", "<em>rose—</em>",
    "<em>King’s consent signified.</em>", "<p><i>shook his head.</i></p>", "<em>rose—</em> <em>[Interruption.]</em> ",
    "<em>[]</em>",
    # A note alone about anything but a gap in the transcript is not the member speaking
    "<em>[Interruption.]</em> ", "<em>[Laughter.]</em>", "[Hon. Members: “Hear, hear!”]",
])  # fmt: skip
def test_the_oracle_reads_an_empty_or_stage_direction_only_contribution_as_no_speech(text):
    assert OO.says_nothing(text)


@pytest.mark.parametrize("text", ["<em>[Inaudible.]</em>", "<em>[</em><em>Inaudible</em><em>.</em><em>]</em>",
                                  "<em>[Inaudible].</em>", "(Inaudible.)", "<em>[Inaudible.</em>"])
def test_the_oracle_reads_a_note_of_a_gap_alone_in_a_members_place_as_their_speech(text):
    """S05's real-data check: "[Inaudible.]" stands where a member's words were; they had the floor. Only a note of a
    gap in the transcript; "[Interruption.]" alone is not (the list above).
    """
    assert not OO.says_nothing(text)


@pytest.mark.parametrize("text", ["I beg to move.", "<em>Moved by Lord Lemos</em>", "<em>rose—</em> Thank you.",
                                  "<em>On a point of order, Mr Speaker.</em>", "The <em>Guardian</em> said so.",
                                  "<em>Asked by Baroness Example</em>", "<i>To ask His Majesty’s Government.</i>",
                                  "<em>Bill presented.</em>",
                                  "<em>Diolch yn fawr iawn, Llywydd, am y cyfle i siarad heddiw am y Bil pwysig hwn i "
                                  "bobl Cymru.</em>"])
def test_the_oracle_reads_words_spoken_or_an_act_of_moving_as_a_speech(text):
    assert not OO.says_nothing(text)


def test_a_member_who_rose_or_signified_consent_does_not_open_the_oracles_debate():
    t = _truth(OO.Contribution(1, 1, "Mr Speaker", "Order."),
               OO.Contribution(2, 7, "Sir Desmond Swayne (New Forest West) (Con)", "<em>rose—</em>"),
               OO.Contribution(3, 8, "The Solicitor General (Ann Minister)", "<em>King’s consent signified.</em>"),
               OO.Contribution(4, 9, "Mr Member in Charge (Somewhere) (Lab)", "I beg to move, That the Bill be now read "
                               "the Third time."))
    assert t.opener.mnis_id == 9
    same = _truth(OO.Contribution(1, 8, "The Minister (Ann Minister)", "<em>indicated assent.</em>"),
                  OO.Contribution(2, 8, "The Minister (Ann Minister)", "I beg to move."))
    assert (same.opener.mnis_id, same.opener.order) == (8, 2)
    assert _truth(OO.Contribution(1, 7, "A Member (Somewhere) (Con)", "<em>rose—</em>")).opener is None


def test_the_speakers_statement_is_the_chairs_and_the_point_of_order_after_it_opens_nothing():
    t = _truth(OO.Contribution(1, 467, "Mr Speaker", "I wish to make a short statement."),
               OO.Contribution(2, 7, "Sir Desmond Swayne (New Forest West) (Con)", "On a point of order, Mr Speaker."),
               heading="Speaker’s Statement")
    assert t.chair_business and t.opener is None and t.chair_statement.mnis_id == 467
    for heading in ("Deputy Speaker’s Statement", "Lord Speaker's Statement", "Statement by the Speaker",
                    "Speaker Statement"):
        assert _truth(t.contributions[0], heading=heading).chair_business, heading
    assert not _truth(t.contributions[0], heading="Statement on the Speaker's Conference").chair_business
    thread = OO.ThreadTruth((t,))
    assert thread.chair_business and thread.opener is None and thread.chair_statement.mnis_id == 467


# --- QA pass 7: the oracle's definitions, and their agreement with the route's on every form both pin -----------------


@pytest.mark.parametrize(("heading", "expected"), [
    ("Speaker\u2019s Statement", True), ("Speakers Statement", True), ("Speaker: Statement", True),
    ("Lord Speaker: Statement", True), ("Deputy Speaker\u2019s Statement", True), ("Statement by the Speaker", True),
    ("Chairman of Ways and Means\u2019 Statement", True), ("Chairman of Committees\u2019 Statement", True),
    ("First Deputy Chairman\u2019s Statement", True), ("Chair's Statement", True),
    ("Speaker\u2019s Statement: Hybrid Proceedings", True),
    ("Statement by the Chair of the Liaison Committee", False), ("Chair\u2019s Statement: Public Accounts Committee Report", False),
    ("Statement by the Chairman of the Conservative Party", False),
    ("Statement from the Speaker of the Ukrainian Parliament", False),
    ("Statements by the Speaker's Committee on the Electoral Commission", False),
    ("Statement on the Speaker's Conference", False), ("Speaker's Conference", False), ("Business Statement", False),
])  # fmt: skip
def test_the_oracle_reads_the_chairs_own_statement_from_the_heading(heading, expected):
    assert OO.chairs_statement(heading) is expected


def test_the_route_and_the_oracle_read_every_pinned_contribution_alike():
    """Separate code, one definition: on every form either side pins, they agree (QA pass 7 found five where they did
    not: text runs glued together, nesting, mismatched pairs, a trailing open tag, a comment holding ">").
    """
    from importlib import import_module

    battery = [p for name in ("test_an_empty_contribution_or_one_of_stage_directions_alone_says_nothing",
                              "test_words_spoken_or_an_act_of_moving_are_a_speech")
               for p in _params(import_module("test_procedure_opener"), name)]
    battery += ["<em>indicated assent.</em>", "<p><i>shook his head.</i></p>", "The <em>Guardian</em> said so."]
    from procedure import opener as OP

    differ = [t for t in battery if OP.non_speech(t) != OO.says_nothing(t)]
    assert not differ and len(battery) > 30, differ


def _params(module, name):
    marks = getattr(getattr(module, name), "pytestmark", [])
    return [v for m in marks if m.name == "parametrize" for v in m.args[1]]


@pytest.mark.parametrize("text", ["<em>Hear, <em>hear</em>.</em>", "<i>rose\u2014</em>", "<em>rose\u2014</em><em>", "<em>",
                                  "<!-- a > b -->"])
def test_the_oracle_reads_nesting_mismatched_and_empty_markup_as_saying_nothing(text):
    assert OO.says_nothing(text)


@pytest.mark.parametrize("text", ["<p><em>Amendment 2</em></p><p><em>Moved by Lord Xavier</em></p>",
                                  "<em>Clause 1</em><br><em>Asked by Lord X</em>",
                                  "<em>one</em> <em>two three four five six seven eight nine ten eleven twelve thirteen</em>",
                                  "<!-- a > b -->I beg to move."])
def test_the_oracle_never_glues_text_runs_together(text):
    assert not OO.says_nothing(text)
