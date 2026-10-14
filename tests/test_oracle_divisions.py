"""The division oracle re-derives every result, tally and member vote from the raw Votes API payloads, with a parser
that shares no code with `src/ingest/`.
"""

from __future__ import annotations

import ast
import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from eval.oracles import divisions as O

ROOT = Path(__file__).resolve().parents[1]


def cm(mid, party="Labour", proxy=None):
    return {"MemberId": mid, "Name": f"Member {mid}", "Party": party, "ProxyName": proxy}


def lm(mid, party="Crossbench"):
    return {"memberId": mid, "name": f"Lord {mid}", "party": party}


COMMONS = {"DivisionId": 1810, "Date": "2024-05-08T15:26:00", "Number": 146, "Title": "Opposition Day: Housing",
           "AyeCount": 3, "NoCount": 2, "Ayes": [cm(1), cm(2, proxy="Proxy Person")], "Noes": [cm(3, "Conservative")],
           "AyeTellers": [cm(4)], "NoTellers": [cm(5, "Conservative")], "NoVoteRecorded": [cm(6)]}
LORDS = {"divisionId": 3142, "date": "2024-05-14T17:00:00", "number": 2, "title": "Health Bill: amendment 1",
         "authoritativeContentCount": 2, "authoritativeNotContentCount": 3, "contents": [lm(11)],
         "notContents": [lm(12), lm(13)], "contentTellers": [lm(14)], "notContentTellers": [lm(15)]}


def test_a_commons_division_is_read_from_its_own_payload():
    t = O.parse_commons(COMMONS)
    assert (t.division_id, t.house, t.date, t.number, t.ayes, t.noes) == (1810, "Commons", date(2024, 5, 8), 146, 3, 2)
    assert (t.ayes_listed, t.noes_listed, t.result) == (3, 2, "agreed")  # tellers are named, so they are listed
    votes = {v.mnis_id: (v.lobby, v.is_teller, v.proxy_name) for v in t.votes}
    assert votes == {1: ("aye", False, None), 2: ("aye", False, "Proxy Person"), 3: ("no", False, None),
                     4: ("aye", True, None), 5: ("no", True, None), 6: ("no_vote", False, None)}


def test_a_lords_division_is_read_from_its_own_payload_with_the_corpus_id():
    t = O.parse_lords(LORDS)
    assert (t.division_id, t.house, t.date, t.ayes, t.noes, t.result) == (1003142, "Lords", date(2024, 5, 14), 2, 3,
                                                                          "negatived")
    assert (t.ayes_listed, t.noes_listed) == (2, 3)
    assert {v.mnis_id: (v.lobby, v.is_teller) for v in t.votes} == {11: ("aye", False), 12: ("no", False),
                                                                    13: ("no", False), 14: ("aye", True),
                                                                    15: ("no", True)}


def test_a_member_in_both_lobbies_is_a_conflict_counted_in_neither_and_a_tie_has_no_result():
    t = O.parse_commons({**COMMONS, "Noes": [cm(1)], "AyeCount": 2, "NoCount": 2})
    assert {v.mnis_id: v.lobby for v in t.votes}[1] == "conflict"
    assert (t.ayes_listed, t.noes_listed) == (2, 1)  # member 1 is named in both lists, so is listed in neither
    assert t.result == "tie"  # the Chair's casting vote decides; the counts alone do not


def test_a_lords_teller_named_in_the_lobby_list_too_is_counted_once():
    """The Lords API names each teller in `contents` as well as in `contentTellers`, so counting list lengths put every
    Lords division 2 over on each side. Members are counted, not list entries.
    """
    t = O.parse_lords({**LORDS, "contents": [lm(11), lm(14)], "notContents": [lm(12), lm(13), lm(15)]})
    assert (t.ayes_listed, t.noes_listed) == (2, 3)
    assert {v.mnis_id: (v.lobby, v.is_teller) for v in t.votes}[14] == ("aye", True)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)


@pytest.fixture
def raw(tmp_path):
    base = tmp_path / "raw"
    _write(base / "commons_divisions" / "W0" / "index_000000.json.gz", [{"DivisionId": 1, "Date": "2026-09-08T00:00:00"}])
    _write(base / "commons_divisions" / "W5" / "index_000000.json.gz", [{"DivisionId": 2, "Date": "2026-05-20T00:00:00"}])
    _write(base / "commons_divisions" / "detail" / "1.json.gz", {**COMMONS, "DivisionId": 1, "Date": "2026-09-08T00:00:00"})
    _write(base / "commons_divisions" / "detail" / "2.json.gz", {**COMMONS, "DivisionId": 2, "Date": "2026-05-20T00:00:00"})
    _write(base / "lords_divisions" / "W0" / "index_000000.json.gz", [{**LORDS, "divisionId": 7, "date": "2026-09-09T00:00:00"}])
    _write(base / "lords_divisions" / "W5" / "index_000000.json.gz", [{**LORDS, "divisionId": 8, "date": "2026-06-01T00:00:00"}])
    return base


def test_the_oracle_never_opens_a_payload_from_the_sealed_window(raw, monkeypatch):
    opened = []
    real = gzip.open

    def spy(path, *a, **k):
        opened.append(Path(path))
        return real(path, *a, **k)

    monkeypatch.setattr(O.gzip, "open", spy)
    found = {t.division_id for t in O.truths(raw)}
    assert found == {1, 1000007}
    assert not any("W5" in p.parts or p.name == "2.json.gz" for p in opened), opened


def test_a_payload_dated_inside_the_window_is_skipped_wherever_it_is_filed(raw):
    _write(raw / "commons_divisions" / "W0" / "index_000001.json.gz", [{"DivisionId": 3, "Date": "2026-06-10T00:00:00"}])
    _write(raw / "commons_divisions" / "detail" / "3.json.gz", {**COMMONS, "DivisionId": 3, "Date": "2026-06-10T00:00:00"})
    assert 3 not in {t.division_id for t in O.truths(raw)}


def test_a_disagreement_with_the_database_is_named_field_by_field():
    t = O.parse_commons(COMMONS)
    record = {"id": 1810, "house": "Commons", "division_date": date(2024, 5, 8), "number": 146, "result": "negatived",
              "ayes": 3, "noes": 9, "ayes_listed": 3, "noes_listed": 2, "ayes_named": 2, "noes_named": 1,
              "tally_reconciled": False}
    votes = {1: {"lobby": "aye", "is_teller": False, "proxy_name": None},
             2: {"lobby": "no", "is_teller": False, "proxy_name": "Proxy Person"},
             4: {"lobby": "aye", "is_teller": False, "proxy_name": None},
             5: {"lobby": "no", "is_teller": True, "proxy_name": None},
             6: {"lobby": "no_vote", "is_teller": False, "proxy_name": None},
             99: {"lobby": "aye", "is_teller": False, "proxy_name": None}}
    found = {(d.check, d.mnis_id) for d in O.compare(t, record, votes)}
    assert found == {("division.result", None), ("division.noes", None), ("vote.lobby", 2), ("vote.missing", 3),
                     ("vote.is_teller", 4), ("vote.extra", 99)}


def test_the_database_agreeing_is_no_disagreement():
    t = O.parse_commons(COMMONS)
    record = {"id": 1810, "house": "Commons", "division_date": date(2024, 5, 8), "number": 146, "result": "agreed",
              "ayes": 3, "noes": 2, "ayes_listed": 3, "noes_listed": 2, "ayes_named": 2, "noes_named": 1,
              "tally_reconciled": False}
    votes = {v.mnis_id: {"lobby": v.lobby, "is_teller": v.is_teller, "proxy_name": v.proxy_name} for v in t.votes}
    assert O.compare(t, record, votes) == []


@pytest.mark.parametrize("change,reconciled", [
    ({}, False),  # 2 Ayes named against 3 counted
    ({"AyeCount": 2, "NoCount": 1}, True),  # names and figures agree, tellers apart
    ({"AyeCount": 2, "NoCount": 2, "Noes": [cm(3), cm(1)]}, True),  # a two-lobby member: counted in both lobbies
    ({"AyeCount": 2, "NoCount": 3, "Noes": [cm(3), cm(1)]}, False),  # ...so a list still short of its figure fails
    ({"AyeCount": 4, "NoCount": 3}, False),  # short by exactly the tellers: still short
])
def test_a_commons_division_reconciles_when_the_printed_names_match_the_tellers_figures(change, reconciled):
    """The record's own definition. The names printed for each lobby, tellers excluded (they are listed apart and do
    not count themselves), against the tellers' figures.
    """
    assert O.parse_commons({**COMMONS, **change}).tally_reconciled is reconciled


def test_a_lords_division_reconciles_when_its_members_match_the_authoritative_counts():
    """The Lords' authoritative counts include the tellers, so each member named for a lobby, tellers in, counted once.
    """
    assert O.parse_lords(LORDS).tally_reconciled is True
    assert O.parse_lords({**LORDS, "authoritativeContentCount": 5}).tally_reconciled is False


def test_the_names_each_lobby_records_are_counted_by_the_record_s_own_definition():
    """`ayes_named` and `noes_named`: the members each lobby's list names, each once. Commons tellers apart, Lords
    tellers in (as each House's figures count them); a member named in both lobbies counts in each, as the tellers
    counted them; a member named twice in one list is one member.
    """
    t = O.parse_commons({**COMMONS, "Ayes": [cm(1), cm(2), cm(2)], "Noes": [cm(3), cm(1)]})
    assert (t.ayes_named, t.noes_named) == (2, 2)
    lords = O.parse_lords({**LORDS, "contents": [lm(11), lm(14), lm(12)], "notContents": [lm(12), lm(13), lm(15)]})
    assert (lords.ayes_named, lords.noes_named) == (3, 3)  # 11, 12, 14 | 12, 13, 15: each teller once


def test_a_commons_teller_named_in_the_other_lobby_counts_there_and_not_for_the_side_they_told():
    t = O.parse_commons({**COMMONS, "Noes": [cm(3), cm(4)]})  # member 4 tells for the Ayes and is named a No
    assert (t.ayes_named, t.noes_named) == (2, 2)


def _hansard(path: Path, results: list[dict]) -> None:
    _write(path, {"Results": results, "TotalResultCount": len(results)})


def test_hansard_s_printed_figures_are_a_second_source_and_a_conflict_is_reported(raw):
    """The Votes API and Hansard's own division records disagree on 4 Commons divisions outside the window. In division
    2264 the API counts 256 Noes, Hansard prints 286, and the list names 288.
    """
    _hansard(raw / "hansard_divisions" / "W0" / "Commons" / "index_000000.json.gz",
             [{"Date": "2026-09-08T15:00:00", "Number": "146", "AyesCount": 3, "NoesCount": 9, "House": "Commons"}])
    _hansard(raw / "hansard_divisions" / "W0" / "Lords" / "index_000000.json.gz",
             [{"Date": "2026-09-09T17:00:00", "Number": "2", "AyesCount": 2, "NoesCount": 3, "House": "Lords"}])
    found = {t.division_id: t for t in O.truths(raw)}
    assert found[1].printed == (3, 9) and found[1000007].printed == (2, 3)
    assert [(c.division_id, c.votes_api, c.hansard) for c in O.source_conflicts(found[1])] == [(1, (3, 2), (3, 9))]
    assert O.source_conflicts(found[1000007]) == []


def test_hansard_s_records_are_never_read_from_the_sealed_window(raw, monkeypatch):
    _hansard(raw / "hansard_divisions" / "W5" / "Commons" / "index_000000.json.gz",
             [{"Date": "2026-05-20T15:00:00", "Number": "146", "AyesCount": 1, "NoesCount": 1, "House": "Commons"}])
    _hansard(raw / "hansard_divisions" / "W0" / "Commons" / "index_000000.json.gz",
             [{"Date": "2026-06-10T15:00:00", "Number": "9", "AyesCount": 1, "NoesCount": 1, "House": "Commons"}])
    opened = []
    real = gzip.open

    def spy(path, *a, **k):
        opened.append(Path(path))
        return real(path, *a, **k)

    monkeypatch.setattr(O.gzip, "open", spy)
    assert O.printed_figures(raw) == {}  # the W5 folder unopened; the misfiled sealed date skipped
    assert not any("W5" in p.parts for p in opened), opened


def test_a_division_hansard_does_not_record_has_no_printed_figures_and_no_conflict(raw):
    t = {t.division_id: t for t in O.truths(raw)}[1]
    assert t.printed is None and O.source_conflicts(t) == []


def test_a_stored_reconciliation_the_payload_contradicts_is_a_disagreement():
    t = O.parse_commons({**COMMONS, "AyeCount": 2, "NoCount": 1})
    record = {"id": 1810, "house": "Commons", "division_date": date(2024, 5, 8), "number": 146, "result": "agreed",
              "ayes": 2, "noes": 1, "ayes_listed": 3, "noes_listed": 2, "ayes_named": 2, "noes_named": 0,
              "tally_reconciled": False}
    votes = {v.mnis_id: {"lobby": v.lobby, "is_teller": v.is_teller, "proxy_name": v.proxy_name} for v in t.votes}
    assert [(d.check, d.oracle, d.database) for d in O.compare(t, record, votes)] == [
        ("division.noes_named", 1, 0), ("division.tally_reconciled", True, False)]


def test_the_oracle_shares_no_code_with_ingest():
    """An oracle built on ingest's parsers would agree with ingest's bugs."""
    tree = ast.parse((ROOT / "src" / "eval" / "oracles" / "divisions.py").read_text(encoding="utf-8"))
    modules = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(m == "ingest" or m.startswith(("ingest.", "answer", "retrieval")) for m in modules), modules


def test_the_report_lists_every_source_conflict_apart_from_the_database_s_disagreements():
    """A conflict between the Votes API and Hansard is not the database's fault, and must never be hidden in a count of
    the database's faults either: it has its own list, in full.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("oracle_divisions_script", ROOT / "scripts" / "oracle_divisions.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    conflict = O.SourceConflict(2264, (73, 256), (73, 286))
    report = script.summarise([], divisions=1, votes=2, conflicts=[conflict], printed=1)
    assert report["disagreements"] == 0 and report["divisions_with_hansard_figures"] == 1
    assert report["source_conflicts"] == [{"division_id": 2264, "votes_api": [73, 256], "hansard": [73, 286]}]
    assert {"ayes_named", "noes_named", "tally_reconciled"} <= set(script.COLUMNS)


def test_hansard_records_that_disagree_under_one_key_are_left_out_never_picked(raw):
    """Review pass 1: the printed figures were keyed by House, date and number, so a second record under a key would
    have overwritten the first silently. None exists today (417 keys); at ten years one may.
    """
    _hansard(raw / "hansard_divisions" / "W0" / "Commons" / "index_000000.json.gz",
             [{"Date": "2026-09-08T15:00:00", "Number": "146", "AyesCount": 3, "NoesCount": 9, "House": "Commons"},
              {"Date": "2026-09-08T16:00:00", "Number": "146", "AyesCount": 3, "NoesCount": 8, "House": "Commons"},
              {"Date": "2026-09-09T16:00:00", "Number": "7", "AyesCount": 1, "NoesCount": 2, "House": "Commons"},
              {"Date": "2026-09-09T16:00:00", "Number": "7", "AyesCount": 1, "NoesCount": 2, "House": "Commons"}])
    printed = O.printed_figures(raw)
    assert ("Commons", date(2026, 9, 8), "146") not in printed
    assert printed[("Commons", date(2026, 9, 9), "7")] == (1, 2)  # an exact repeat is one record


def test_a_list_entry_without_a_member_id_is_skipped_as_the_loader_skips_it():
    """The loader reports such an entry and counts no one for it; the oracle must agree, not crash."""
    t = O.parse_commons({**COMMONS, "Ayes": [cm(1), cm(2), {"Name": "Ghost"}]})
    assert t.ayes_named == 2 and {v.mnis_id for v in t.votes} == {1, 2, 3, 4, 5, 6}


# --- review pass 2: the record's semantics, independent of list order ---------------------------------


def test_a_vote_and_a_no_vote_line_for_one_member_is_the_vote():
    """Review pass 2: a later NoVoteRecorded line overwrote the vote. A recorded vote is the record."""
    t = O.parse_commons({**COMMONS, "NoVoteRecorded": [cm(1)]})
    assert {v.mnis_id: v.lobby for v in t.votes}[1] == "aye"


def test_a_member_in_both_lobbies_stays_a_conflict_whatever_comes_after():
    """Ayes, then Noes, then the Noes' tellers: the conflict must survive the teller line, which used to overwrite it.
    """
    t = O.parse_commons({**COMMONS, "Noes": [cm(3), cm(1)], "NoTellers": [cm(5), cm(1)]})
    one = {v.mnis_id: v for v in t.votes}[1]
    assert (one.lobby, one.is_teller) == ("conflict", True)


def test_unrecorded_figures_give_no_result_and_no_reconciliation():
    t = O.parse_commons({**COMMONS, "AyeCount": None})
    assert (t.ayes, t.result, t.tally_reconciled) == (None, None, None)


@pytest.mark.parametrize("change,result", [
    ({"DoubleMajorityAyeCount": 2, "DoubleMajorityNoCount": 1}, "agreed"),  # both majorities carried it
    ({"DoubleMajorityAyeCount": 1, "DoubleMajorityNoCount": 2}, None),  # the two disagree: withheld
    ({"DoubleMajorityAyeCount": 2, "DoubleMajorityNoCount": None}, None),  # one of two counts: withheld
])
def test_an_english_votes_division_needs_both_majorities(change, result):
    """EVEL (2015-21): a double-majority division passed only with a majority of the House AND of the MPs concerned. At
    ten years these divisions are in the record; without this they would score as false harms.
    """
    assert O.parse_commons({**COMMONS, **change}).result == result


def test_the_oracle_checks_the_title_the_deferral_and_each_vote_s_party():
    t = O.parse_commons({**COMMONS, "IsDeferred": True})
    record = {"id": 1810, "house": "Commons", "division_date": date(2024, 5, 8), "number": 146, "result": "agreed",
              "ayes": 3, "noes": 2, "ayes_listed": 3, "noes_listed": 2, "ayes_named": 2, "noes_named": 1,
              "tally_reconciled": False, "title": "Opposition Day: Housing (amended)", "is_deferred": False}
    votes = {v.mnis_id: {"lobby": v.lobby, "is_teller": v.is_teller, "proxy_name": v.proxy_name, "party": v.party}
             for v in t.votes}
    votes[3] = {**votes[3], "party": "Labour"}
    found = {(d.check, d.mnis_id) for d in O.compare(t, record, votes)}
    assert found == {("division.title", None), ("division.is_deferred", None), ("vote.party", 3)}
