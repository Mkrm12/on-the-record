"""The objective-correction audit."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_gold_objective_t", ROOT / "scripts" / "audit_gold_objective.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


class FakeSources:
    """The primary sources, as the audit asks for them, with the answers each test needs. Asking for anything else
    raises, so a test also proves what the audit did NOT read.
    """

    def __init__(self, **answers):
        self.answers = answers

    def __getattr__(self, name):
        if name in self.answers:
            value = self.answers[name]
            return value if callable(value) else (lambda *a, **k: value)
        raise AttributeError(name)


def gold(item_id="db0006", split="dev", category="division_result", facts=None, nuggets=(), evidence=(),
         forbidden=(), question="Did the Commons agree the regulations in the division on 28 January 2026?", day=""):
    return SimpleNamespace(id=item_id, split=split, category=category, question=question, date=day,
                           expected_structured_facts=facts or {}, nuggets=tuple(nuggets), evidence=tuple(evidence),
                           forbidden_claims=tuple(forbidden))


def nugget(text, vital=True, speech_ids=()):
    return SimpleNamespace(text=text, vital=vital, speech_ids=tuple(speech_ids))


def evidence(speech_id="S1", quote="A quotation from the record."):
    return SimpleNamespace(speech_id=speech_id, quote=quote)


DIVISION = {"division_id": 421, "house": "Commons", "date": "2026-01-28", "result": "agreed", "ayes": 294, "noes": 108}
RECORD = {"result": "negatived", "ayes": 294, "noes": 108, "date": date(2026, 1, 28)}
OPENERS = [{"mnis_id": 1, "role": "deputy_speaker", "boilerplate": True},
           {"mnis_id": 2, "role": "member", "boilerplate": False}]
DEBATE = {"date": date(2024, 7, 29), "heading": "Carer’s Allowance: Repayments", "house": "Lords"}


# --- divisions -----------------------------------------------------------------------------------------------------


def test_a_division_result_the_record_contradicts_is_found():
    findings = A.audit_item(gold(facts=DIVISION), FakeSources(division=RECORD))
    assert [(f.check, f.expected, f.found) for f in findings] == [("division.result", "agreed", "negatived")]
    assert findings[0].source == "division.id=421"


def test_a_division_the_record_agrees_with_is_clean():
    src = FakeSources(division={**RECORD, "result": "agreed"})
    assert A.audit_item(gold(facts=DIVISION), src) == []


def test_a_divisions_house_number_and_date_are_checked_too():
    src = FakeSources(division={**RECORD, "result": "agreed", "date": date(2026, 1, 29), "house": "Lords", "number": 7})
    findings = A.audit_item(gold(facts={**DIVISION, "division_number": 7}), src)
    assert sorted(f.check for f in findings) == ["division.date", "division.house"]


def test_a_division_the_record_does_not_hold_is_found():
    assert [f.check for f in A.audit_item(gold(facts=DIVISION), FakeSources(division=None))] == ["division.missing"]


def test_a_members_vote_the_record_contradicts_is_found():
    src = FakeSources(vote={"lobby": "no", "party": "Labour", "is_teller": False, "proxy_name": None,
                            "date": date(2025, 3, 4)})
    item = gold("db0072", category="division_vote", facts={"division_id": 1813, "mnis_id": 4755, "lobby": "aye",
                                                           "party_at_vote": "Labour", "is_teller": False})
    assert [(f.check, f.expected, f.found) for f in A.audit_item(item, src)] == [("division_vote.lobby", "aye", "no")]


# --- members and offices -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("parties,checks", [({"Labour"}, []), ({"Conservative"}, ["party_at_date"]),
                                            (set(), ["party_at_date.no_division_list"])])
def test_a_party_on_a_date_is_the_party_on_that_days_division_list(parties, checks):
    item = gold("db0117", category="party_at_date", facts={"mnis_id": 5143, "date": "2024-07-22", "party": "Labour"})
    assert [f.check for f in A.audit_item(item, FakeSources(party_on=parties))] == checks


@pytest.mark.parametrize("holders,checks", [({1547}, []), ({99}, ["office_held_at_date"]),
                                            (set(), ["office_held_at_date.no_holder"])])
def test_an_office_holder_on_a_date_is_checked(holders, checks):
    item = gold("db0144", category="office_held_at_date",
                facts={"office_key": "lord chamberlain (hm household)", "date": "2026-09-01", "mnis_id": 1547})
    assert [f.check for f in A.audit_item(item, FakeSources(office_holders=holders))] == checks


# --- bills and counts ----------------------------------------------------------------------------------------------


def test_a_bill_stages_first_sitting_and_its_sittings_are_checked():
    item = gold("db0175", category="bill_stage_date", facts={"bill_id": 3746, "stage": "Order of Commitment discharged",
                                                             "house": "Lords", "first_sitting": "2024-11-04",
                                                             "sittings": 1})
    assert A.audit_item(item, FakeSources(stage_sittings=(date(2024, 11, 4), 1))) == []
    findings = A.audit_item(item, FakeSources(stage_sittings=(date(2024, 11, 5), 2)))
    assert sorted(f.check for f in findings) == ["bill_stage.first_sitting", "bill_stage.sittings"]
    assert [f.check for f in A.audit_item(item, FakeSources(stage_sittings=None))] == ["bill_stage.missing"]


@pytest.mark.parametrize("facts,answer,check", [
    ({"house": "Lords", "month": "2026-09-01", "divisions": 13}, {"division_count": 14}, "count.divisions"),
    ({"house": "Commons", "date": "2026-09-08", "debates": 22}, {"debate_count": 21}, "count.debates"),
    ({"mnis_id": 1940, "from": "2024-05-01", "to": "2024-07-31", "contributions": 5, "excludes": "boilerplate"},
     {"contributions": 6}, "count.contributions"),
])
def test_a_count_the_record_contradicts_is_found(facts, answer, check):
    item = gold("db0202", category="count", facts=facts)
    assert [f.check for f in A.audit_item(item, FakeSources(**answer))] == [check]
    right = {name: facts[check.split(".")[1]] for name in answer}
    assert A.audit_item(item, FakeSources(**right)) == []


# --- debates and openers -------------------------------------------------------------------------------------------


def test_a_gold_opener_who_was_presiding_is_flagged():
    item = gold("db0227", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Madam Deputy Speaker",
                                                       "mnis_id": 1})
    checks = [f.check for f in A.audit_item(item, FakeSources(debate=DEBATE, first_speeches=OPENERS))]
    assert checks == ["who_spoke.opener"]


def test_a_chairman_in_committee_is_presiding_whatever_the_role_data_says():
    """Speaker_role tags a presiding Chairman of Ways and Means `office_holder`."""
    first = [{"mnis_id": 36, "role": "office_holder", "boilerplate": False,
              "speaker_name": "The Chairman of Ways and Means (Dame Eleanor Laing)"},
             {"mnis_id": 2, "role": "member", "boilerplate": False, "speaker_name": "Ann Member (Somewhere) (Lab)"}]
    item = gold("db0234", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Baroness Laing of Elderslie",
                                                       "mnis_id": 36})
    assert [f.check for f in A.audit_item(item, FakeSources(debate=DEBATE, first_speeches=first))] == [
        "who_spoke.opener"]


def test_a_who_spoke_question_that_cannot_identify_its_debate_is_flagged():
    """The question names a debate by heading, House and date, and two debates share them (a Bill's second reading and
    its Committee stage), so it has two right answers.
    """
    item = gold("db0234", category="who_spoke", day="2020-03-23",
                facts={"debate_id": "D1", "heading": "Coronavirus Bill", "opened_by": "Ann Member", "mnis_id": 2})
    item.house = "Commons"
    src = FakeSources(debate={**DEBATE, "date": date(2020, 3, 23), "heading": "Coronavirus Bill"},
                      first_speeches=OPENERS, debates_like=2)
    assert [f.check for f in A.audit_item(item, src)] == ["who_spoke.ambiguous_debate"]
    assert A.audit_item(item, FakeSources(debate={**DEBATE, "date": date(2020, 3, 23), "heading": "Coronavirus Bill"},
                                          first_speeches=OPENERS, debates_like=1)) == []


def test_a_gold_opener_who_spoke_first_substantively_is_clean():
    item = gold("db0227", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Ann Member", "mnis_id": 2,
                                                       "heading": "Carer's Allowance: Repayments"})
    assert A.audit_item(item, FakeSources(debate=DEBATE, first_speeches=OPENERS)) == []  # curly or straight: same


def test_a_who_spoke_opener_named_by_a_later_name_is_flagged():
    """The generator wrote member_id_map's CURRENT name ("Lord Gove") for a speech the record printed under the name
    the member held then ("Michael Gove"). Same member, so the opener check passes; the name is still wrong.
    """
    first = [{"mnis_id": 9, "role": "member", "boilerplate": False, "speaker_name": "Michael Gove (Surrey Heath) (Con)"}]
    item = gold("db0299", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Lord Gove", "mnis_id": 9})
    src = FakeSources(debate=DEBATE, first_speeches=first, member_names=lambda mnis: {"Lord Gove", "Michael Gove"})
    assert [f.check for f in A.audit_item(item, src)] == ["who_spoke.printed_name"]
    same = gold("db0299", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Michael Gove", "mnis_id": 9})
    assert A.audit_item(same, src) == []


def test_a_heading_the_record_contradicts_is_flagged():
    item = gold("db0268", category="multi_speech", facts={"debate_id": "D1", "heading": "Finance (No. 2) Bill"})
    assert [f.check for f in A.audit_item(item, FakeSources(debate=DEBATE))] == ["debate.heading"]


def test_a_debate_inside_the_sealed_window_is_never_read():
    """Only its date is asked for; its speeches are never requested (FakeSources would raise if they were)."""
    item = gold("db0227", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Ann Member", "mnis_id": 2})
    findings = A.audit_item(item, FakeSources(debate={"date": date(2026, 5, 1)}))
    assert [f.check for f in findings] == ["sealed_window"]


def test_an_item_dated_inside_the_sealed_window_is_never_checked():
    item = gold(facts=DIVISION, day="2026-06-01")
    assert [f.check for f in A.audit_item(item, FakeSources())] == ["sealed_window"]  # nothing was asked for


def test_a_fact_dated_inside_the_sealed_window_is_counted_and_never_checked():
    src = FakeSources(division={"result": "negatived", "ayes": 1, "noes": 2, "date": date(2026, 5, 20)})
    findings = A.audit_item(gold(facts={**DIVISION, "date": "2026-05-20"}), src)
    assert [f.check for f in findings] == ["sealed_window"] and findings[0].found == ""


def test_a_record_dated_inside_the_sealed_window_stops_the_item():
    """The gold's own date may be wrong; the record's date is what keeps W5 sealed."""
    findings = A.audit_item(gold(facts=DIVISION), FakeSources(division={"date": date(2026, 4, 2)}))
    assert [f.check for f in findings] == ["sealed_window"]


# --- attribution ---------------------------------------------------------------------------------------------------


def _speech(name, mnis=7, day=date(2026, 9, 15), api=None):
    return {"speaker_name": name, "mnis_id": mnis, "date": day, "api_member_name": api}


def test_a_member_named_in_a_vital_point_who_did_not_speak_the_evidence_is_flagged():
    src = FakeSources(speech=_speech("Rebecca Paul"), member_names=lambda mnis: {"Rebecca Paul"})
    item = gold("hum0079", category="quote_verification", question="Did anyone say it?",
                nuggets=[nugget("Nigel Huddleston said it.")], evidence=[evidence()])
    assert [f.check for f in A.audit_item(item, src)] == ["attribution.speaker"]


def test_a_member_named_by_a_name_the_record_did_not_print_is_flagged():
    """Gold written with a member's later name; the record printed the name the member held then."""
    src = FakeSources(speech=_speech("Michael Gove", mnis=9, day=date(2020, 5, 1)),
                      member_names=lambda mnis: {"Michael Gove", "Lord Gove"})
    item = gold("old0099", category="named_speaker", question="What did he say?",
                nuggets=[nugget("Lord Gove said the plan would work.")], evidence=[evidence()])
    assert [f.check for f in A.audit_item(item, src)] == ["attribution.printed_name"]


def test_a_member_said_not_to_have_spoken_is_not_an_attribution():
    src = FakeSources(speech=_speech("Rebecca Paul"), member_names=lambda mnis: {"Rebecca Paul"})
    item = gold("mvs0001", category="mention_vs_speaker", question="Did Keir Starmer speak?",
                nuggets=[nugget("Keir Starmer did not speak; he was mentioned.")], evidence=[evidence()])
    assert A.audit_item(item, src) == []


def test_a_lords_territorial_designation_is_the_same_member():
    src = FakeSources(speech=_speech("Lord Hanson of Flint"), member_names=lambda mnis: {"Lord Hanson of Flint"})
    item = gold("hum0100", category="paraphrase", question="What did the Government say?",
                nuggets=[nugget("Lord Hanson said the review would report in spring.")], evidence=[evidence()])
    assert A.audit_item(item, src) == []


def test_a_minister_printed_by_office_is_named_from_the_brackets():
    src = FakeSources(speech=_speech("The Minister for Housing (Matthew Pennycook)", api="Matthew Pennycook"),
                      member_names=lambda mnis: {"Matthew Pennycook"})
    item = gold("hum0101", category="paraphrase", question="What did the Government say about planning?",
                nuggets=[nugget("Matthew Pennycook said the targets would stand.")], evidence=[evidence()])
    assert A.audit_item(item, src) == []
    assert A.printed_name("Nigel Huddleston (Droitwich and Evesham) (Con)") == "Nigel Huddleston"
    assert A.printed_name("The Minister for Housing (Matthew Pennycook)") == "Matthew Pennycook"


def test_a_lords_office_printed_before_the_name_is_read_from_the_brackets():
    """The Lords print some office-holders office first, without "The"."""
    printed = "Captain of the Honourable Corps of Gentlemen-at-Arms and Chief Whip (Lord Kennedy of Southwark) (Lab)"
    assert A.printed_name(printed, {"Lord Kennedy of Southwark"}) == "Lord Kennedy of Southwark"
    first = [{"mnis_id": 5, "role": "office_holder", "boilerplate": False, "speaker_name": printed}]
    item = gold("db0233", category="who_spoke", facts={"debate_id": "D1", "opened_by": "Lord Kennedy of Southwark",
                                                       "mnis_id": 5})
    src = FakeSources(debate=DEBATE, first_speeches=first, member_names=lambda mnis: {"Lord Kennedy of Southwark"})
    assert A.audit_item(item, src) == []


def test_a_different_member_with_the_same_surname_is_flagged():
    src = FakeSources(speech=_speech("Angela Rayner"), member_names=lambda mnis: {"Angela Rayner"})
    item = gold("hum0102", category="paraphrase", question="What was said?",
                nuggets=[nugget("Ruth Rayner argued for the change.")], evidence=[evidence()])
    assert [f.check for f in A.audit_item(item, src)] == ["attribution.speaker"]


def test_a_speech_inside_the_sealed_window_stops_the_item():
    src = FakeSources(speech={"date": date(2026, 6, 3)})
    item = gold("hum0103", category="paraphrase", question="What was said?",
                nuggets=[nugget("Ann Member said it.")], evidence=[evidence()])
    assert [f.check for f in A.audit_item(item, src)] == ["sealed_window"]


# --- bill types ----------------------------------------------------------------------------------------------------


def test_a_bill_type_the_bills_api_contradicts_is_flagged():
    """Hum0083 calls a Government Bill (type 1) a private Member's Bill."""
    src = FakeSources(bills_of_speech=lambda speech_id: [{"id": 4059, "type": 1}])
    item = gold("hum0083", category="colloquial", question="is the goverment scrapping the cap",
                nuggets=[nugget("A private Member's Bill to remove the limit was debated.")],
                evidence=[evidence(quote="This Bill removes the two-child limit entirely.")])
    findings = A.audit_item(item, src)
    assert [(f.check, f.source) for f in findings] == [("bill.type", "bill.id=4059")]
    assert findings[0].found == "Government Bill"


def test_a_forbidden_bill_type_the_bills_api_confirms_is_flagged():
    """The gold forbids saying something the Bills API says is true: the forbidden claim is the error."""
    src = FakeSources(bills_of_speech=lambda speech_id: [{"id": 4056, "type": 8}])
    item = gold("hum0104", category="colloquial", question="whose bill was it",
                nuggets=[nugget("It was debated on 3 February 2026.")], evidence=[evidence()],
                forbidden=["Calling it a private Member's Bill."])
    assert [f.check for f in A.audit_item(item, src)] == ["bill.type.forbidden"]


def test_a_negated_bill_type_is_read_as_a_denial():
    src = FakeSources(bills_of_speech=lambda speech_id: [{"id": 4059, "type": 1}])
    item = gold("hum0105", category="colloquial", question="whose bill was it",
                nuggets=[nugget("It was not a private Member's Bill: the Government brought it.")], evidence=[evidence()])
    assert A.audit_item(item, src) == []


def test_bills_of_both_kinds_on_one_debate_are_left_alone():
    src = FakeSources(bills_of_speech=lambda speech_id: [{"id": 4059, "type": 1}, {"id": 4056, "type": 8}])
    item = gold("hum0106", category="colloquial", question="whose bill was it",
                nuggets=[nugget("A private Member's Bill was debated.")], evidence=[evidence()])
    assert A.audit_item(item, src) == []  # ambiguous is not objective


# --- output and the corrections ledger -----------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["test", "heldout"])
def test_confirmation_split_output_holds_ids_checks_and_sources_never_text_or_values(split):
    item = gold(split=split, facts=DIVISION)
    text = A.render(A.audit_item(item, FakeSources(division=RECORD)), {item.id: item})
    assert "db0006" in text and "division.result" in text and "division.id=421" in text
    for leak in ("agreed", "negatived", "regulations", "28 January", "Commons"):
        assert leak not in text, leak


def test_the_dev_output_shows_what_the_gold_says_and_what_the_record_says():
    item = gold(facts=DIVISION)
    text = A.render(A.audit_item(item, FakeSources(division=RECORD)), {item.id: item})
    assert "gold 'agreed'" in text and "record 'negatived'" in text


def test_the_summary_counts_sealed_items_without_naming_them():
    item = gold(facts=DIVISION, day="2026-06-01")
    text = A.render(A.audit_item(item, FakeSources()), {item.id: item})
    assert "db0006" not in text and "1 item(s) inside the sealed window" in text


@pytest.mark.parametrize("record,ok", [
    ({"id": "hum0083", "field": "nuggets[0].text", "new": "x", "reason": "the Bills API says type 1",
      "source": {"kind": "api", "ref": "bills-api /Bills/4059 billTypeId=1"}, "approved_by": "owner",
      "date": "2026-10-10"}, True),
    ({"id": "hum0083", "field": "nuggets[0].text", "new": "x", "reason": "it reads better", "approved_by": "owner",
      "date": "2026-10-10"}, False),  # no source
    ({"id": "hum0083", "field": "nuggets[0].text", "new": "x", "reason": "r", "source": {"kind": "opinion", "ref": "me"},
      "approved_by": "owner", "date": "2026-10-10"}, False),  # not a primary source
    ({"id": "hum0083", "field": "nuggets[0].text", "new": "x", "reason": "r",
      "source": {"kind": "sql", "ref": "bill.id=4059"}, "date": "2026-10-10"}, False),  # nobody approved it
    ({"id": "hum0083", "field": "nuggets[0].text", "new": "x", "reason": "r",
      "source": {"kind": "sql", "ref": "bill.id=4059"}, "approved_by": "claude", "date": "2026-10-10"}, False),
])
def test_a_correction_needs_a_primary_source_and_the_owners_approval(record, ok):
    assert (A.correction_problems(record) == []) is ok


def test_an_unloaded_corpus_is_refused_by_name_not_audited(scratch_dsn, monkeypatch):
    """CI's database is migrated and empty. Against it the audit reported 140 dev findings ("division.missing" for
    every division item), a summary that cannot tell "nothing is loaded" from "the gold is wrong", and the live
    tests that skip only when the database is unreachable failed.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "pg_dsn", scratch_dsn)
    with pytest.raises(A.CorpusNotLoaded) as refused:
        A.SqlSources.connect()
    for table in ("division", "bill", "debate", "speech", "member_id_map"):
        assert table in str(refused.value)


def test_the_live_audit_finds_d028_and_passes_its_correction():
    from dataclasses import replace

    from goldset.schema import Nugget, load

    try:
        sources = A.SqlSources.connect()
    except Exception as exc:  # noqa: BLE001 - no database is a skip, not a failure
        pytest.skip(f"hansard_v2 is not reachable ({type(exc).__name__})")
    corrected = {i.id: i for i in load()}["hum0083"]
    before = replace(corrected, nuggets=(Nugget("A private Member's Bill to remove the limit was debated on 3 February "
                                                "2026.", True),))
    try:
        old, new = A.audit_item(before, sources), A.audit_item(corrected, sources)
    finally:
        sources.close()
    assert any(f.check == "bill.type" and f.found == "Government Bill" for f in old), old
    assert new == []
