"""Commons + Lords divisions."""

import asyncio
import copy
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import psycopg
import pytest

from ingest.divisions import (
    DIVISION_COLUMNS,
    LORDS_ID_OFFSET,
    VOTE_COLUMNS,
    DebateInfo,
    TallyEvent,
    fetch_commons,
    fetch_lords,
    link_debate,
    normalise_title,
    parse_commons,
    parse_lords,
    parse_tally,
    store,
)
from ingest.http import PoliteClient
from ingest.runtime import run

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "divisions.json").read_text(encoding="utf-8"))


def real_commons(division_id: str) -> dict:
    return copy.deepcopy(FIXTURES["commons"][division_id])


def real_lords(division_id: str) -> dict:
    return copy.deepcopy(FIXTURES["lords"][division_id])


def member(mid, name="Member", party="Labour", abbr="Lab", sub=None, frm="Some Seat", proxy=None):
    return {
        "MemberId": mid, "Name": name, "Party": party, "SubParty": sub, "PartyColour": "d50000",
        "PartyAbbreviation": abbr, "MemberFrom": frm, "ListAs": name, "ProxyName": proxy,
    }  # fmt: skip


def commons(ayes=(), noes=(), nvr=(), aye_tellers=(), no_tellers=(), aye_count=None, no_count=None, **extra):
    d = {
        "DivisionId": 100, "Date": "2026-09-01T15:00:00", "PublicationUpdated": "2026-09-02T10:00:00",
        "Number": 7, "IsDeferred": False, "EVELType": "", "EVELCountry": "", "Title": "Some Bill: Second Reading",
        "AyeCount": len(ayes) if aye_count is None else aye_count,  # the tellers do not count themselves
        "NoCount": len(noes) if no_count is None else no_count,
        "DoubleMajorityAyeCount": None, "DoubleMajorityNoCount": None,
        "AyeTellers": list(aye_tellers), "NoTellers": list(no_tellers),
        "Ayes": list(ayes), "Noes": list(noes), "NoVoteRecorded": list(nvr),
        "FriendlyDescription": None, "FriendlyTitle": None, "RemoteVotingStart": None, "RemoteVotingEnd": None,
    }  # fmt: skip
    d.update(extra)
    return d


def lords_member(mid, name="Lord X", party="Labour", frm="Life peer"):
    return {
        "memberId": mid, "name": name, "listAs": name, "memberFrom": frm, "party": party,
        "partyColour": "d50000", "partyAbbreviation": "Lab", "partyIsMainParty": True,
    }  # fmt: skip


def lords(contents=(), not_contents=(), content_tellers=(), not_content_tellers=(), content=None, not_content=None, **extra):
    d = {
        "divisionId": 500, "date": "2026-09-07T00:00:00", "number": 3, "notes": None, "title": "Some Bill [HL]",
        "isWhipped": True, "isGovernmentContent": True,
        "authoritativeContentCount": len(contents) if content is None else content,
        "authoritativeNotContentCount": len(not_contents) if not_content is None else not_content,
        "divisionHadTellers": True, "tellerContentCount": len(content_tellers), "tellerNotContentCount": len(not_content_tellers),
        "memberContentCount": len(contents) - len(content_tellers), "memberNotContentCount": len(not_contents) - len(not_content_tellers),
        "sponsoringMemberId": 1, "isHouse": True, "isInquorate": False,
        "amendmentMotionNotes": "<p>Lord A moved amendment 1. The House divided:</p>", "isGovernmentWin": True,
        "remoteVotingStart": "2026-09-07T17:11:53", "remoteVotingEnd": "2026-09-07T17:19:53",
        "divisionWasExclusivelyRemote": False,
        "contents": list(contents), "notContents": list(not_contents),
        "contentTellers": list(content_tellers), "notContentTellers": list(not_content_tellers),
    }  # fmt: skip
    d.update(extra)
    return d


def votes_by_member(parsed):
    return {v["mnis_id"]: v for v in parsed.votes}


# =========================================================================== Commons: real payloads
# ===========================================================================


def test_commons_real_division_fields():
    p = parse_commons(real_commons("2428"))
    d = p.division
    assert (d["id"], d["source_id"], d["house"], d["number"]) == (2428, 2428, "Commons", 75)
    assert d["division_date"] == date(2026, 9, 11) and d["held_at"] == datetime(2026, 9, 11, 14, 30)
    assert d["title"] == "Terminally Ill Adults (End of Life) Bill: Second Reading"
    assert (d["ayes"], d["noes"]) == (270, 286)  # the OFFICIAL counts
    assert d["result"] == "negatived" and d["motion_type"] == "division" and d["is_deferred"] is False
    assert d["government_won"] is None and d["debate_id"] is None


def test_commons_real_vote_fields_are_as_at_the_vote():
    v = votes_by_member(parse_commons(real_commons("2428")))[177]
    assert (v["member_name"], v["party"], v["party_abbreviation"], v["sub_party"], v["member_from"]) == (
        "Gareth Thomas", "Labour", "Lab", "Co-op", "Harrow West",
    )
    assert v["lobby"] == "aye" and v["is_teller"] is False and v["division_id"] == 2428


def test_commons_real_no_vote_recorded_and_tellers():
    votes = votes_by_member(parse_commons(real_commons("2428")))
    assert votes[5131]["lobby"] == "no_vote"  # Jack Abbott, first NoVoteRecorded
    assert votes[4456]["is_teller"] and votes[4456]["lobby"] == "aye"  # Jeff Smith, aye teller
    assert votes[5309]["is_teller"] and votes[5309]["lobby"] == "no"  # Katrina Murray, no teller


@pytest.mark.parametrize("division_id,member_id", [("2428", 4595), ("2414", 5231)])
def test_commons_real_member_in_both_lobbies_becomes_a_conflict_not_a_coin_flip(division_id, member_id):
    p = parse_commons(real_commons(division_id))
    v = votes_by_member(p)[member_id]
    assert v["lobby"] == "conflict"
    assert {"listed_aye", "listed_no"} <= set(v["flags"])
    assert "conflicting_votes:1" in p.division["anomalies"]
    assert sum(1 for x in p.votes if x["mnis_id"] == member_id) == 1  # one row: the PK holds


def test_commons_real_deferred_division_with_null_teller_lists():
    p = parse_commons(real_commons("2424"))
    assert p.division["is_deferred"] is True and p.division["number"] == 72
    assert not any(v["is_teller"] for v in p.votes)
    assert p.division["held_at"] == datetime(2026, 9, 9, 0, 0)  # midnight placeholder


def test_commons_primary_key_is_unique_across_every_real_fixture():
    for division_id in FIXTURES["commons"]:
        p = parse_commons(real_commons(division_id))
        keys = [(v["division_id"], v["mnis_id"]) for v in p.votes]
        assert len(keys) == len(set(keys)), division_id


def test_commons_raw_keeps_metadata_but_not_the_member_lists():
    raw = parse_commons(real_commons("2428")).division["raw"]
    assert "Title" in raw and "EVELType" in raw and "PublicationUpdated" in raw
    for key in ("Ayes", "Noes", "NoVoteRecorded", "AyeTellers", "NoTellers"):
        assert key not in raw
    json.dumps(raw)


def test_every_row_has_exactly_the_declared_columns():
    p = parse_commons(real_commons("2428"))
    assert set(p.division) == set(DIVISION_COLUMNS)
    assert all(set(v) == set(VOTE_COLUMNS) for v in p.votes)


# =========================================================================== Commons: synthetic edge cases
# ===========================================================================


@pytest.mark.parametrize("aye,no,result", [(10, 5, "agreed"), (5, 10, "negatived"), (7, 7, "tied")])
def test_result_from_official_counts(aye, no, result):
    assert parse_commons(commons(aye_count=aye, no_count=no)).division["result"] == result


def test_result_is_none_when_counts_are_missing():
    assert parse_commons(commons(AyeCount=None)).division["result"] is None


def test_conflicting_member_is_excluded_from_both_listed_tallies():
    p = parse_commons(commons(ayes=[member(1), member(2)], noes=[member(1), member(3)]))
    assert votes_by_member(p)[1]["lobby"] == "conflict"
    assert (p.division["ayes_listed"], p.division["noes_listed"]) == (1, 1)


def test_voter_also_listed_as_no_vote_keeps_the_vote_and_is_flagged():
    p = parse_commons(commons(ayes=[member(1)], nvr=[member(1)]))
    v = votes_by_member(p)[1]
    assert v["lobby"] == "aye" and "also_no_vote" in v["flags"]
    assert "also_no_vote:1" in p.division["anomalies"]
    assert len(p.votes) == 1


def test_same_member_twice_in_one_list_is_deduplicated_and_flagged():
    p = parse_commons(commons(ayes=[member(1), member(1)]))
    assert len(p.votes) == 1 and "duplicate_listing" in p.votes[0]["flags"]
    assert "duplicate_listing:1" in p.division["anomalies"]
    assert p.votes[0]["lobby"] == "aye"


def test_teller_also_listed_in_the_same_lobby_is_not_an_anomaly():
    # A teller does not count themself, so the figure is 0.
    p = parse_commons(commons(ayes=[member(1)], aye_tellers=[member(1)], aye_count=0, no_count=0))
    v = votes_by_member(p)[1]
    assert v["is_teller"] is True and v["lobby"] == "aye"
    assert p.division["anomalies"] == []
    assert len(p.votes) == 1


def test_teller_listed_in_the_opposite_lobby_is_a_conflict():
    p = parse_commons(commons(noes=[member(1)], aye_tellers=[member(1)]))
    v = votes_by_member(p)[1]
    assert v["lobby"] == "conflict" and v["is_teller"] is True


def test_separate_teller_rows():
    p = parse_commons(commons(ayes=[member(1)], aye_tellers=[member(2), member(3)], no_tellers=[member(4)]))
    votes = votes_by_member(p)
    assert [votes[i]["is_teller"] for i in (1, 2, 3, 4)] == [False, True, True, True]
    assert [votes[i]["lobby"] for i in (1, 2, 3, 4)] == ["aye", "aye", "aye", "no"]


def test_proxy_vote_is_recorded():
    v = votes_by_member(parse_commons(commons(ayes=[member(1, proxy="Some Whip")])))[1]
    assert v["proxy_name"] == "Some Whip"


def test_missing_lists_do_not_crash():
    d = commons()
    for key in ("Ayes", "Noes", "NoVoteRecorded", "AyeTellers", "NoTellers"):
        d[key] = None
    p = parse_commons(d)
    assert p.votes == [] and (p.division["ayes_listed"], p.division["noes_listed"], p.division["no_vote_count"]) == (0, 0, 0)


def test_missing_optional_member_fields():
    v = votes_by_member(parse_commons(commons(ayes=[{"MemberId": 9, "Name": "Solo"}])))[9]
    assert v["party"] is None and v["sub_party"] is None and v["member_from"] is None and v["member_name"] == "Solo"


def test_entry_without_member_id_is_skipped_and_reported():
    p = parse_commons(commons(ayes=[member(1), {"Name": "Ghost"}]))
    assert [v["mnis_id"] for v in p.votes] == [1]
    assert "vote_without_member_id:1" in p.division["anomalies"]


def test_no_vote_count_and_empty_division():
    p = parse_commons(commons(nvr=[member(1), member(2)]))
    assert p.division["no_vote_count"] == 2
    empty = parse_commons(commons(aye_count=0, no_count=0)).division
    assert empty["tally_reconciled"] is True and empty["anomalies"] == []


# ---- reconciliation: official counts vs stored voters -------------------------------------


def test_reconciled_when_official_equals_listed_excluding_tellers():
    p = parse_commons(commons(ayes=[member(1), member(2)], aye_tellers=[member(3)], aye_count=2, no_count=0))
    assert p.division["tally_reconciled"] is True and p.division["anomalies"] == []


def test_a_list_short_by_exactly_its_tellers_does_not_reconcile():
    """This loader used to accept a figure equal to the names *with* the tellers, on the belief that the Votes API
    sometimes counts them. It does not: Hansard printed the same figures as the API in all 26 divisions whose names
    fall exactly two short on one side (`hansard_divisions`), and the Deputy Speaker warned before one of them that
    pass readers in both lobbies were down.
    """
    p = parse_commons(commons(ayes=[member(1), member(2)], aye_tellers=[member(3), member(4)], aye_count=4, no_count=0))
    assert p.division["tally_reconciled"] is False
    assert "aye_count_mismatch:official=4,named=2" in p.division["anomalies"]


def test_an_unreconciled_side_records_its_named_members_and_its_figure():
    p = parse_commons(commons(ayes=[member(1), member(2)], aye_tellers=[member(3)], aye_count=9, no_count=0))
    assert p.division["tally_reconciled"] is False
    assert "aye_count_mismatch:official=9,named=2" in p.division["anomalies"]


def test_each_side_is_reconciled_independently():
    p = parse_commons(commons(ayes=[member(1)], noes=[member(2)], aye_count=1, no_count=5))
    assert p.division["tally_reconciled"] is False
    assert not any(a.startswith("aye_count_mismatch") for a in p.division["anomalies"])
    assert "no_count_mismatch:official=5,named=1" in p.division["anomalies"]


def test_the_names_each_lobby_records_are_counted_as_its_figure_counts_them():
    """`ayes_named` / `noes_named`: Commons tellers apart; a member named in both lobbies counts in each, as the
    tellers counted them; a member named twice in one list is one member.
    """
    p = parse_commons(commons(ayes=[member(1), member(2), member(2)], noes=[member(1), member(3)],
                              aye_tellers=[member(4)], no_tellers=[member(5)], aye_count=2, no_count=2))
    assert (p.division["ayes_named"], p.division["noes_named"]) == (2, 2)
    assert p.division["tally_reconciled"] is True  # the two-lobby member is in both figures


def test_a_commons_teller_named_in_the_other_lobby_counts_there_only():
    p = parse_commons(commons(ayes=[member(1)], noes=[member(2), member(3)], aye_tellers=[member(3)],
                              aye_count=1, no_count=2))
    assert (p.division["ayes_named"], p.division["noes_named"]) == (1, 2)
    assert p.division["tally_reconciled"] is True


def test_an_entry_without_a_member_id_is_not_a_named_member():
    p = parse_commons(commons(ayes=[member(1), {"Name": "Ghost"}], aye_count=2, no_count=0))
    assert p.division["ayes_named"] == 1 and p.division["tally_reconciled"] is False


def test_unrecorded_figures_leave_the_reconciliation_unknown():
    p = parse_commons(commons(ayes=[member(1)], AyeCount=None))
    assert p.division["tally_reconciled"] is None and p.division["ayes_named"] == 1


def test_real_unreconciled_division_is_flagged():
    # 2413 official Aye 166 vs 3 trimmed voters: whatever the trimming, it must not claim reconciliation.
    assert parse_commons(real_commons("2413")).division["tally_reconciled"] is False


# ---- identity / dates -------------------------------------------------------------------------


def test_friendly_description_is_the_motion_text_when_present():
    assert parse_commons(commons(FriendlyDescription="Reasoned amendment")).division["motion_text"] == "Reasoned amendment"
    assert parse_commons(commons()).division["motion_text"] == "Some Bill: Second Reading"


def test_date_only_time_and_deferred_flag():
    d = parse_commons(commons(Date="2026-09-09T00:00:00", IsDeferred=True)).division
    assert d["is_deferred"] and d["held_at"] == datetime(2026, 9, 9)


# =========================================================================== Lords
# ===========================================================================


def test_lords_real_division_fields():
    p = parse_lords(real_lords("3707"))
    d = p.division
    assert (d["id"], d["source_id"], d["house"], d["number"]) == (LORDS_ID_OFFSET + 3707, 3707, "Lords", 2)
    assert d["division_date"] == date(2026, 9, 7) and d["title"] == "Financial Services and Markets Bill [HL]"
    assert (d["ayes"], d["noes"]) == (246, 165)  # authoritative Content / Not-Content
    assert d["result"] == "agreed" and d["government_won"] is False and d["motion_type"] == "division"
    assert d["motion_text"].startswith("Baroness Neville Rolfe moved amendment 10") and "<" not in d["motion_text"]


def test_lords_vote_mapping_and_tellers_inside_the_lists():
    p = parse_lords(real_lords("3707"))
    raw = real_lords("3707")
    teller_ids = {m["memberId"] for m in raw["contentTellers"] + raw["notContentTellers"]}
    votes = votes_by_member(p)
    listed = {m["memberId"] for m in raw["contents"] + raw["notContents"]}
    for mid in teller_ids & listed:
        assert votes[mid]["is_teller"] is True
    assert {votes[m["memberId"]]["lobby"] for m in raw["contents"]} == {"aye"}
    assert {votes[m["memberId"]]["lobby"] for m in raw["notContents"]} == {"no"}
    assert "conflicting_votes" not in " ".join(p.division["anomalies"])
    assert len(p.votes) == len({v["mnis_id"] for v in p.votes})


def test_lords_vote_fields():
    v = votes_by_member(parse_lords(lords(contents=[lords_member(9, "Lord Nine", "Crossbench", "Hereditary")])))[9]
    assert (v["member_name"], v["party"], v["party_abbreviation"], v["member_from"]) == ("Lord Nine", "Crossbench", "Lab", "Hereditary")
    assert v["sub_party"] is None and v["proxy_name"] is None


def test_lords_reconciles_when_tellers_are_inside_the_lists():
    d = lords(contents=[lords_member(1), lords_member(2)], content_tellers=[lords_member(2)], not_contents=[lords_member(3)])
    assert parse_lords(d).division["tally_reconciled"] is True


def test_lords_tellers_count_in_their_lobby_wherever_they_are_listed():
    """The Lords' authoritative counts include the tellers, so a teller named only in the tellers' list still counts.
    """
    p = parse_lords(lords(contents=[lords_member(1)], content_tellers=[lords_member(2)],
                          not_contents=[lords_member(3), lords_member(1)], content=2, not_content=2))
    assert (p.division["ayes_named"], p.division["noes_named"]) == (2, 2)
    assert p.division["tally_reconciled"] is True  # member 1, in both lobbies, counts in each


def test_lords_unreconciled_is_reported():
    p = parse_lords(lords(contents=[lords_member(1)], content=5))
    assert p.division["tally_reconciled"] is False
    assert any(a.startswith("aye_count_mismatch") for a in p.division["anomalies"])


def test_lords_member_in_both_lobbies_is_a_conflict():
    p = parse_lords(lords(contents=[lords_member(1)], not_contents=[lords_member(1)]))
    assert votes_by_member(p)[1]["lobby"] == "conflict"


def test_lords_motion_text_falls_back_to_notes_then_title():
    assert parse_lords(lords(amendmentMotionNotes=None, notes="Some note")).division["motion_text"] == "Some note"
    assert parse_lords(lords(amendmentMotionNotes=None, notes=None)).division["motion_text"] == "Some Bill [HL]"


def test_lords_motion_text_unescapes_html():
    d = lords(amendmentMotionNotes="<p>moved &ldquo;principles&rdquo; &amp; more. The House divided:</p>")
    assert parse_lords(d).division["motion_text"] == "moved “principles” & more. The House divided:"


def test_lords_ties_and_missing_government_flag():
    assert parse_lords(lords(content=5, not_content=5)).division["result"] == "tied"
    assert parse_lords(lords(isGovernmentWin=None)).division["government_won"] is None


def test_lords_held_at_uses_remote_voting_start_when_present_else_midnight():
    assert parse_lords(lords()).division["held_at"] == datetime(2026, 9, 7, 17, 11, 53)
    assert parse_lords(lords(remoteVotingStart=None)).division["held_at"] == datetime(2026, 9, 7, 0, 0)


def test_lords_raw_drops_member_lists():
    raw = parse_lords(lords(contents=[lords_member(1)])).division["raw"]
    assert "isGovernmentWin" in raw and "contents" not in raw and "notContents" not in raw and "contentTellers" not in raw
    json.dumps(raw)


def test_lords_and_commons_ids_never_collide_for_the_same_source_id():
    c = parse_commons(commons(DivisionId=500)).division
    l = parse_lords(lords(divisionId=500)).division
    assert c["source_id"] == l["source_id"] == 500
    assert c["id"] != l["id"] and l["id"] == 500 + LORDS_ID_OFFSET


def test_vote_rows_carry_the_namespaced_division_id():
    p = parse_lords(lords(divisionId=500, contents=[lords_member(1)]))
    assert p.votes[0]["division_id"] == LORDS_ID_OFFSET + 500


# =========================================================================== Linking a division to its Hansard debate
# ===========================================================================


def dt(day, h=0, m=0):
    return datetime(2026, 9, day, h, m)


def debate(did, day, heading, house="Commons", times=(), section=None):
    """`times` are the timecodes of the debate's speeches."""
    return DebateInfo(did, date(2026, 9, day), house, heading, section or f"{house} Chamber", tuple(times))


def div(house="Commons", day=11, h=14, m=30, title="Some Bill: Second Reading", ayes=270, noes=286, deferred=False):
    return {
        "house": house, "division_date": date(2026, 9, day), "held_at": dt(day, h, m),
        "title": title, "ayes": ayes, "noes": noes, "is_deferred": deferred,
    }  # fmt: skip


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ayes 262, Noes 159", (262, 159)),
        ("Ayes 262 Noes 159", (262, 159)),
        ("Contents 246, Not-Contents 165", (246, 165)),
        ("Content 3, Not Content 4", (3, 4)),
        ("Ayes 5", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_tally(text, expected):
    assert parse_tally(text) == expected


@pytest.mark.parametrize(
    "a,b",
    [
        ("Health Bill", "health bill"),
        ("Draft Plant Health Regulations 2026", "Plant Health Regulations 2026"),
        ("Social Housing Bill [Lords]", "Social Housing Bill"),
        ("Financial Services and Markets Bill [HL]", "Financial Services and Markets Bill"),
        ("Terminally Ill Adults (End of Life) Bill", "Terminally Ill Adults  (End of Life)   Bill"),
        ("The Health Bill", "Health Bill"),
    ],
)
def test_normalise_title_equates_cosmetic_differences(a, b):
    assert normalise_title(a) == normalise_title(b)


def test_normalise_title_keeps_real_differences_apart():
    assert normalise_title("Health Bill") != normalise_title("Health and Care Bill")


def test_link_by_tally_event_is_strongest():
    debates = [debate("A", 11, "Unrelated heading"), debate("B", 11, "Other heading")]
    tallies = [TallyEvent("B", date(2026, 9, 11), "Commons", 270, 286)]
    assert link_debate(div(), debates, tallies) == ("B", "tally_event")


def test_tally_must_match_house_date_and_both_counts():
    debates = [debate("A", 11, "Some Bill")]
    for t in (
        TallyEvent("A", date(2026, 9, 11), "Lords", 270, 286),
        TallyEvent("A", date(2026, 9, 10), "Commons", 270, 286),
        TallyEvent("A", date(2026, 9, 11), "Commons", 270, 285),
    ):
        assert link_debate(div(), debates, [t])[1] != "tally_event"


def test_ambiguous_tally_falls_through_to_title():
    debates = [debate("A", 11, "Some Bill"), debate("B", 11, "Another Bill")]
    tallies = [TallyEvent("A", date(2026, 9, 11), "Commons", 270, 286), TallyEvent("B", date(2026, 9, 11), "Commons", 270, 286)]
    assert link_debate(div(), debates, tallies) == ("A", "title_exact")


def test_link_by_title_base_before_the_stage_suffix():
    debates = [debate("T", 11, "Terminally Ill Adults (End of Life) Bill"), debate("X", 11, "25th Anniversary of 9/11")]
    d = div(title="Terminally Ill Adults (End of Life) Bill: Second Reading")
    assert link_debate(d, debates, []) == ("T", "title_exact")


def test_link_ignores_lords_bracket_marker_and_draft():
    assert link_debate(div(title="Social Housing Bill [Lords]: Reasoned Amendment to Second Reading"),
                       [debate("S", 11, "Social Housing Bill [Lords]")], []) == ("S", "title_exact")
    assert link_debate(div(house="Lords", title="Financial Services and Markets Bill [HL]"),
                       [debate("F", 11, "Financial Services and Markets Bill [HL]", "Lords")], []) == ("F", "title_exact")
    assert link_debate(div(title="Plant Health Regulations 2026"),
                       [debate("P", 11, "Draft Plant Health Regulations 2026")], []) == ("P", "title_exact")


def test_house_must_match():
    assert link_debate(div(house="Lords", title="Health Bill"), [debate("H", 11, "Health Bill", "Commons")], [])[0] is None


def test_title_prefix_match_needs_a_reasonably_long_title():
    assert link_debate(div(title="Health Bill: Report Stage: New Clause 6"), [debate("H", 11, "Health Bill: Report Stage")], [])[1] == "title_prefix"
    # "Closure motion" must not attach to an unrelated debate that merely starts the same way.
    assert link_debate(div(title="Closure motion", h=9), [debate("C", 11, "Closure of Post Offices")], [])[0] is None


def test_a_short_title_is_not_matched_as_a_prefix_of_a_longer_heading():
    # "Health Bill" (11 chars) is a prefix of "Health Bill Committee" but that is a different debate.
    assert link_debate(div(title="Health Bill"), [debate("H", 11, "Health Bill Committee")], [])[0] is None
    assert link_debate(div(title="Health Bill: Amendment 1"), [debate("H", 11, "Health Bill Committee")], [])[0] is None


def test_same_date_beats_other_dates_for_a_normal_division():
    debates = [debate("old", 8, "Health Bill"), debate("today", 11, "Health Bill")]
    assert link_debate(div(title="Health Bill: Report Stage"), debates, []) == ("today", "title_exact")
    assert link_debate(div(title="Health Bill: Report Stage"), [debate("old", 8, "Health Bill")], [])[0] is None


def test_deferred_division_links_to_an_earlier_debate_by_title():
    d = div(day=9, h=0, m=0, title="Draft Plant Health Regulations 2026", deferred=True)
    debates = [debate("committee", 2, "Draft Plant Health Regulations 2026", section="General Committees"),
               debate("later", 12, "Draft Plant Health Regulations 2026")]
    assert link_debate(d, debates, []) == ("committee", "title_deferred")


def test_deferred_division_does_not_link_beyond_60_days():
    d = div(day=9, title="Draft Plant Health Regulations 2026", deferred=True)
    far = DebateInfo("far", date(2026, 6, 1), "Commons", "Draft Plant Health Regulations 2026", "General Committees", ())
    assert link_debate(d, [far], [])[0] is None


def test_several_same_day_candidates_are_split_by_time():
    debates = [
        debate("morning", 8, "Health Bill", times=[dt(8, 9), dt(8, 12)]),
        debate("evening", 8, "Health Bill", times=[dt(8, 17), dt(8, 19)]),
    ]
    assert link_debate(div(day=8, h=18, m=1, title="Health Bill: Amendment 1"), debates, []) == ("evening", "title_exact+time")
    assert link_debate(div(day=8, h=9, m=30, title="Health Bill: Amendment 1"), debates, []) == ("morning", "title_exact+time")


def test_time_window_fallback_when_no_title_matches():
    debates = [debate("T", 11, "Terminally Ill Adults (End of Life) Bill", times=[dt(11, 9, 30), dt(11, 14, 11)]),
               debate("N", 11, "25th Anniversary of 9/11", times=[dt(11, 9, 30), dt(11, 9, 40)])]
    assert link_debate(div(h=14, m=15, title="Closure motion"), debates, []) == ("T", "time_window")


def test_a_stray_late_timecode_does_not_make_a_debate_win_by_span():
    # Real bug: "25th Anniversary of 9/11" has speeches at 09:30... and one at 15:03, so its SPAN contains 14:15.
    debates = [debate("anniversary", 11, "25th Anniversary of 9/11", times=[dt(11, 9, 30), dt(11, 10, 0), dt(11, 15, 3)]),
               debate("T", 11, "Terminally Ill Adults (End of Life) Bill", times=[dt(11, 9, 35), dt(11, 14, 11)])]
    assert link_debate(div(h=14, m=15, title="Closure motion"), debates, []) == ("T", "time_window")


def test_only_speeches_before_the_division_count():
    debates = [debate("later", 11, "Some Debate", times=[dt(11, 16, 0), dt(11, 17, 0)])]
    assert link_debate(div(h=14, m=15, title="Closure motion"), debates, []) == (None, "none")


def test_time_window_needs_the_division_to_be_close():
    debates = [debate("T", 11, "Some Debate", times=[dt(11, 9), dt(11, 10)])]
    assert link_debate(div(h=15, title="Closure motion"), debates, []) == (None, "none")


def test_time_window_is_skipped_for_deferred_divisions():
    debates = [debate("T", 9, "Some Debate", times=[dt(9, 0), dt(9, 1)])]
    assert link_debate(div(day=9, h=0, title="Unknown motion", deferred=True), debates, []) == (None, "none")


def test_no_candidates_means_no_link():
    assert link_debate(div(), [], []) == (None, "none")


def test_link_is_deterministic_when_candidates_tie():
    debates = [debate("b", 11, "Some Bill"), debate("a", 11, "Some Bill")]
    assert link_debate(div(), debates, [])[0] == "a"


# =========================================================================== Fetching
# ===========================================================================


def run_with(handler, body, **kwargs):
    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = PoliteClient(http, **{"min_interval": 0, "backoff_base": 0, **kwargs})
            return await body(client)

    return asyncio.run(main())


def test_fetch_commons_pages_through_the_search_then_fetches_each_detail():
    seen = []
    summaries = [{"DivisionId": i} for i in (11, 12, 13)]

    def handler(request):
        seen.append((request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/divisions.json/search"):
            skip = int(request.url.params["queryParameters.skip"])
            take = int(request.url.params["queryParameters.take"])
            return httpx.Response(200, json=summaries[skip : skip + take])
        div_id = int(request.url.path.rsplit("/", 1)[1].split(".")[0])
        return httpx.Response(200, json=commons(DivisionId=div_id))

    details, failed = run_with(handler, lambda c: fetch_commons(c, date(2026, 8, 10), date(2026, 9, 19), page_size=2))
    assert [d["DivisionId"] for d in details] == [11, 12, 13] and failed == []
    searches = [p for path, p in seen if path.endswith("/search")]
    assert [p["queryParameters.skip"] for p in searches] == ["0", "2"]
    assert searches[0]["queryParameters.startDate"] == "2026-08-10" and searches[0]["queryParameters.endDate"] == "2026-09-19"
    assert sorted(path for path, _ in seen if not path.endswith("/search")) == [
        "/data/division/11.json", "/data/division/12.json", "/data/division/13.json",
    ]


def test_fetch_commons_reports_details_that_404_or_fail():
    def handler(request):
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json=[{"DivisionId": 1}, {"DivisionId": 2}, {"DivisionId": 3}])
        if request.url.path.endswith("/2.json"):
            return httpx.Response(404)
        if request.url.path.endswith("/3.json"):
            return httpx.Response(500)
        return httpx.Response(200, json=commons(DivisionId=1))

    details, failed = run_with(handler, lambda c: fetch_commons(c, date(2026, 9, 1), date(2026, 9, 2)), max_attempts=2)
    assert [d["DivisionId"] for d in details] == [1]
    assert sorted(failed) == [2, 3]


def test_fetch_commons_empty_window():
    handler = lambda request: httpx.Response(200, json=[])  # noqa: E731
    assert run_with(handler, lambda c: fetch_commons(c, date(2026, 9, 1), date(2026, 9, 2))) == ([], [])


def test_fetch_lords_pages_and_returns_full_details():
    seen = []
    items = [lords(divisionId=i) for i in (21, 22, 23)]

    def handler(request):
        seen.append(dict(request.url.params))
        skip, take = int(request.url.params["skip"]), int(request.url.params["take"])
        return httpx.Response(200, json=items[skip : skip + take])

    details = run_with(handler, lambda c: fetch_lords(c, date(2026, 8, 10), date(2026, 9, 19), page_size=2))
    assert [d["divisionId"] for d in details] == [21, 22, 23]
    assert [p["skip"] for p in seen] == ["0", "2"]
    assert seen[0]["startDate"] == "2026-08-10" and seen[0]["endDate"] == "2026-09-19"


def test_fetch_search_failure_propagates():
    from ingest.http import FetchError

    with pytest.raises(FetchError):
        run_with(lambda request: httpx.Response(500), lambda c: fetch_commons(c, date(2026, 9, 1), date(2026, 9, 2)), max_attempts=2)


# --------------------------------------------------------------------------- store: the write layer, against the THROW-
# AWAY scratch database, built from db/schema.sql plus every migration.

DIV_A, DIV_B = 999_101, 999_102


def voters(division_id, ayes=(), noes=()):
    return parse_commons(commons(DivisionId=division_id, ayes=[member(m) for m in ayes], noes=[member(m) for m in noes]))


@pytest.fixture
def stored(scratch_dsn):
    """stored(step, step,...) stores each parsed division in order; returns the votes present after each step."""

    def steps(*parsed_divisions):
        async def go():
            async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
                async with conn.transaction(force_rollback=True):
                    snapshots = []
                    for parsed in parsed_divisions:
                        await store(conn, parsed)
                        rows = await (await conn.execute("SELECT division_id, mnis_id, lobby FROM division_vote WHERE division_id BETWEEN 999000 AND 999999")).fetchall()
                        snapshot: dict[int, dict[int, str]] = {}
                        for division_id, mnis_id, lobby in rows:
                            snapshot.setdefault(division_id, {})[mnis_id] = lobby
                        snapshots.append(snapshot)
                    return snapshots

        return run(go())

    return steps


def test_store_writes_every_vote(stored):
    (after,) = stored(voters(DIV_A, ayes=[1, 2, 3], noes=[4, 5]))
    assert after == {DIV_A: {1: "aye", 2: "aye", 3: "aye", 4: "no", 5: "no"}}


def test_store_removes_votes_the_api_no_longer_lists(stored):
    _, after = stored(voters(DIV_A, ayes=[1, 2, 3], noes=[4, 5]), voters(DIV_A, ayes=[1, 3], noes=[5]))
    assert after == {DIV_A: {1: "aye", 3: "aye", 5: "no"}}


def test_store_applies_a_corrected_vote(stored):
    _, after = stored(voters(DIV_A, ayes=[1, 2], noes=[3]), voters(DIV_A, ayes=[1], noes=[2, 3]))
    assert after[DIV_A] == {1: "aye", 2: "no", 3: "no"}


def test_store_is_idempotent(stored):
    first, second = stored(voters(DIV_A, ayes=[1, 2], noes=[3]), voters(DIV_A, ayes=[1, 2], noes=[3]))
    assert first == second


def test_store_pruning_one_division_leaves_another_alone(stored):
    *_, after = stored(voters(DIV_A, ayes=[1, 2]), voters(DIV_B, ayes=[1, 7, 8]), voters(DIV_A, ayes=[1]))
    assert after == {DIV_A: {1: "aye"}, DIV_B: {1: "aye", 7: "aye", 8: "aye"}}


def test_store_never_wipes_votes_when_the_api_returns_no_voters(stored):
    """No member lists at all (a partial response) must not be read as 'every vote was withdrawn'."""
    before, after = stored(voters(DIV_A, ayes=[1, 2], noes=[3]), voters(DIV_A))
    assert after == before == {DIV_A: {1: "aye", 2: "aye", 3: "no"}}


# --------------------------------------------------------------------------- Remote divisions
# ---------------------------------------------------------------------------


def test_a_commons_remote_division_is_marked_remote_and_keeps_its_window():
    """W3 holds 14 Commons divisions: 4 physical in March and 10 remote from 12 May. They are counted by different
    mechanisms, which is why they are the natural test of whether the tellers' count and the listed voters
    reconcile.
    """
    parsed = parse_commons(commons(ayes=[member(1)], noes=[member(2)],
                                   RemoteVotingStart="2020-05-20T18:28:00",
                                   RemoteVotingEnd="2020-05-20T18:43:00"))
    assert parsed.division["is_remote"] is True
    assert parsed.division["remote_start"].isoformat() == "2020-05-20T18:28:00"
    assert parsed.division["remote_end"].isoformat() == "2020-05-20T18:43:00"


def test_a_physical_commons_division_is_not_marked_remote():
    parsed = parse_commons(commons(ayes=[member(1)], noes=[member(2)]))
    assert parsed.division["is_remote"] is False
    assert parsed.division["remote_start"] is None


def test_a_lords_division_is_remote_only_when_the_lords_api_says_exclusively_remote():
    """The trap. `remoteVotingStart` is set on EVERY Lords division -- all 117 in W4 -- so reading it the Commons way
    would mark the whole House remote.
    """
    ordinary = parse_lords(lords(contents=[lords_member(1)], not_contents=[lords_member(2)]))
    assert ordinary.division["remote_start"] is not None, "the timestamp is still stored"
    assert ordinary.division["is_remote"] is False, "but it does not make the division remote"

    remote = parse_lords(lords(contents=[lords_member(1)], not_contents=[lords_member(2)],
                               divisionWasExclusivelyRemote=True))
    assert remote.division["is_remote"] is True


def test_an_unparseable_remote_timestamp_is_absence_not_a_crash():
    parsed = parse_commons(commons(ayes=[member(1)], RemoteVotingStart="not a date"))
    assert parsed.division["remote_start"] is None and parsed.division["is_remote"] is False
