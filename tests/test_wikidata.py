"""Wikidata birth dates. Correctness traps, all seen in the real data: * year-only births are stored as 1 January (149
members) and must not become a fake birthday; * 227 members have several DOB statements (a day-precision preferred
value next to a month-precision one); * 110 members have none. The property ids: P10428 is the MNIS id (P6213 is the
Parliament thesaurus id, NOT the MNIS id).
"""

from datetime import date

import pytest

from ingest.wikidata import (
    CORE_COLUMNS,
    EXTRA_COLUMNS,
    Dob,
    DobCandidate,
    build_update,
    choose_dob,
    core_query,
    dob_bounds,
    extras_query,
    parse_core,
    parse_dob_statements,
    parse_extras,
    parse_time,
)

TODAY = date(2026, 9, 20)


def cand(value, precision, rank="NormalRank"):
    return DobCandidate(value, precision, rank)


# --------------------------------------------------------------------------- parse_time / bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1953-09-27T00:00:00Z", date(1953, 9, 27)),
        ("+1953-09-27T00:00:00Z", date(1953, 9, 27)),
        ("1953-00-00T00:00:00Z", date(1953, 1, 1)),
        ("1953-09-00T00:00:00Z", date(1953, 9, 1)),
        ("-0100-01-01T00:00:00Z", None),
        ("0000-01-01T00:00:00Z", None),
        ("garbage", None),
        ("", None),
        (None, None),
        ("1953-02-30T00:00:00Z", None),
    ],
)
def test_parse_time(value, expected):
    assert parse_time(value) == expected


@pytest.mark.parametrize(
    "value,precision,bounds",
    [
        (date(1953, 1, 1), "year", (date(1953, 1, 1), date(1953, 12, 31))),
        (date(2024, 2, 1), "month", (date(2024, 2, 1), date(2024, 2, 29))),
        (date(2023, 2, 1), "month", (date(2023, 2, 1), date(2023, 2, 28))),
        (date(1953, 9, 27), "day", (date(1953, 9, 27), date(1953, 9, 27))),
    ],
)
def test_dob_bounds(value, precision, bounds):
    assert dob_bounds(value, precision) == bounds


# --------------------------------------------------------------------------- parse_dob_statements
# ---------------------------------------------------------------------------


def test_parse_statements_maps_wikidata_precision_codes():
    got = parse_dob_statements("1953-09-27T00:00:00Z~11~PreferredRank|1953-09-01T00:00:00Z~10~NormalRank|1953-01-01T00:00:00Z~9~NormalRank")
    assert got == [
        cand(date(1953, 9, 27), "day", "PreferredRank"),
        cand(date(1953, 9, 1), "month"),
        cand(date(1953, 1, 1), "year"),
    ]


def test_parse_statements_treats_time_of_day_precision_as_day_and_marks_coarse():
    got = parse_dob_statements("1953-09-27T00:00:00Z~13~NormalRank|1950-01-01T00:00:00Z~8~NormalRank")
    assert got[0].precision == "day" and got[1].precision == "coarse"


@pytest.mark.parametrize("field", [None, "", "|", "~~", "notadate~11~NormalRank", "1953-09-27T00:00:00Z~x~NormalRank", "1953-09-27T00:00:00Z~11"])
def test_parse_statements_skips_empty_or_malformed_pieces(field):
    assert parse_dob_statements(field) == []


def test_parse_statements_tolerates_a_trailing_empty_piece():
    assert len(parse_dob_statements("1953-01-01T00:00:00Z~9~NormalRank|")) == 1


# --------------------------------------------------------------------------- choose_dob
# ---------------------------------------------------------------------------


def test_no_statements():
    assert choose_dob([], TODAY) == Dob(None, None, None, None, None, None)


def test_a_full_date_becomes_date_of_birth():
    d = choose_dob([cand(date(1953, 9, 27), "day")], TODAY)
    assert (d.date_of_birth, d.precision, d.earliest, d.latest, d.year) == (date(1953, 9, 27), "day", date(1953, 9, 27), date(1953, 9, 27), 1953)
    assert d.note is None


def test_a_year_only_birth_is_never_turned_into_a_birthday():
    d = choose_dob([cand(date(1953, 1, 1), "year")], TODAY)
    assert d.date_of_birth is None
    assert (d.precision, d.earliest, d.latest, d.year) == ("year", date(1953, 1, 1), date(1953, 12, 31), 1953)


def test_a_month_only_birth_keeps_its_bounds():
    d = choose_dob([cand(date(1959, 6, 1), "month")], TODAY)
    assert d.date_of_birth is None and (d.precision, d.earliest, d.latest) == ("month", date(1959, 6, 1), date(1959, 6, 30))


def test_preferred_day_beats_normal_month_when_consistent():
    d = choose_dob([cand(date(1952, 9, 21), "day", "PreferredRank"), cand(date(1952, 9, 1), "month")], TODAY)
    assert d.date_of_birth == date(1952, 9, 21) and d.note is None


def test_normal_day_and_month_in_agreement_keeps_the_day():
    d = choose_dob([cand(date(1959, 6, 28), "day"), cand(date(1959, 6, 1), "month")], TODAY)
    assert d.date_of_birth == date(1959, 6, 28) and d.note is None


def test_two_different_full_dates_are_a_conflict_and_nothing_is_guessed():
    d = choose_dob([cand(date(1959, 6, 28), "day"), cand(date(1959, 7, 3), "day")], TODAY)
    assert d.date_of_birth is None and d.precision is None and d.note == "conflicting_dob"


def test_two_preferred_statements_that_disagree_are_a_conflict():
    d = choose_dob([cand(date(1959, 6, 28), "day", "PreferredRank"), cand(date(1959, 7, 3), "day", "PreferredRank")], TODAY)
    assert d.note == "conflicting_dob" and d.date_of_birth is None


def test_preferred_rank_wins_even_over_a_more_precise_normal_statement():
    d = choose_dob([cand(date(1950, 1, 1), "year", "PreferredRank"), cand(date(1950, 3, 2), "day")], TODAY)
    assert d.precision == "year" and d.date_of_birth is None


def test_deprecated_statements_are_ignored():
    d = choose_dob([cand(date(1950, 3, 2), "day", "DeprecatedRank"), cand(date(1951, 1, 1), "year")], TODAY)
    assert d.precision == "year" and d.year == 1951


def test_only_deprecated_statements_means_no_dob():
    assert choose_dob([cand(date(1950, 3, 2), "day", "DeprecatedRank")], TODAY).date_of_birth is None


def test_coarse_precision_is_ignored_with_a_note():
    d = choose_dob([cand(date(1950, 1, 1), "coarse")], TODAY)
    assert d.year is None and d.note == "coarse_dob_ignored"


def test_duplicate_statements_collapse():
    d = choose_dob([cand(date(1953, 9, 27), "day"), cand(date(1953, 9, 27), "day", "PreferredRank")], TODAY)
    assert d.date_of_birth == date(1953, 9, 27) and d.note is None


def test_an_inconsistent_less_precise_statement_is_noted_but_does_not_override():
    d = choose_dob([cand(date(1950, 1, 2), "day", "PreferredRank"), cand(date(1949, 1, 1), "year")], TODAY)
    assert d.date_of_birth == date(1950, 1, 2) and d.note == "inconsistent_candidates"


@pytest.mark.parametrize("value", [date(1750, 1, 1), date(1879, 5, 5), date(2015, 1, 1), date(2026, 9, 1)])
def test_implausible_births_are_rejected(value):
    d = choose_dob([cand(value, "day")], TODAY)
    assert d.date_of_birth is None and d.note == "implausible_dob"


def test_a_birth_that_would_make_a_serving_member_over_105_is_rejected():
    # Real: a current Lord Bishop matched a year-only 1914 (age 111): a wrong or mismatched Wikidata entry.
    d = choose_dob([cand(date(1914, 1, 1), "year")], TODAY)
    assert d.year is None and d.note == "implausible_dob"


@pytest.mark.parametrize("value,ok", [(date(1921, 9, 20), True), (date(1921, 9, 19), False), (date(1925, 6, 1), True)])
def test_the_upper_age_boundary(value, ok):
    assert (choose_dob([cand(value, "day")], TODAY).date_of_birth == value) is ok


def test_a_year_that_straddles_the_upper_age_boundary_is_kept():
    # Born in 1921: they may be 104 or 105 today, so it cannot be ruled out.
    assert choose_dob([cand(date(1921, 1, 1), "year")], TODAY).year == 1921


def test_someone_who_turns_18_today_is_plausible():
    assert choose_dob([cand(date(2008, 9, 20), "day")], TODAY).date_of_birth == date(2008, 9, 20)
    assert choose_dob([cand(date(2008, 9, 21), "day")], TODAY).note == "implausible_dob"


# --------------------------------------------------------------------------- parse_core / parse_extras
# ---------------------------------------------------------------------------


def row(mnis="172", item="Q153454", dobs="1953-09-27T00:00:00Z~11~NormalRank", dods="", twfy="10001", dc="2567"):
    b = {"mnis": {"value": mnis}, "item": {"value": f"http://www.wikidata.org/entity/{item}"}, "dobs": {"value": dobs}, "dods": {"value": dods}}
    if twfy is not None:
        b["twfy_id"] = {"value": twfy}
    if dc is not None:
        b["dc_id"] = {"value": dc}
    return b


def test_parse_core_full_row():
    result = parse_core([row()], TODAY)
    r = result.rows[172]
    assert r["wikidata_id"] == "Q153454"
    assert (r["date_of_birth"], r["dob_precision"], r["birth_year"]) == (date(1953, 9, 27), "day", 1953)
    assert (r["dob_earliest"], r["dob_latest"]) == (date(1953, 9, 27), date(1953, 9, 27))
    assert (r["twfy_id"], r["person_id"], r["democracy_club_id"]) == ("10001", "uk.org.publicwhip/person/10001", "2567")
    assert r["date_of_death"] is None
    assert r["wikidata"]["dob"]["candidates"] == [{"value": "1953-09-27", "precision": "day", "rank": "NormalRank"}]
    assert result.problems == []


def test_parse_core_year_only_row():
    r = parse_core([row(mnis="4972", dobs="1953-01-01T00:00:00Z~9~NormalRank|")], TODAY).rows[4972]
    assert r["date_of_birth"] is None and (r["dob_precision"], r["birth_year"]) == ("year", 1953)
    assert (r["dob_earliest"], r["dob_latest"]) == (date(1953, 1, 1), date(1953, 12, 31))


def test_parse_core_member_with_no_dob_is_still_returned_with_its_ids():
    r = parse_core([row(dobs="", twfy="26114", dc=None)], TODAY).rows[172]
    assert r["date_of_birth"] is None and r["dob_precision"] is None and r["birth_year"] is None
    assert r["twfy_id"] == "26114" and r["democracy_club_id"] is None


def test_parse_core_records_the_reason_when_a_dob_is_rejected():
    r = parse_core([row(dobs="1759-01-01T00:00:00Z~11~NormalRank")], TODAY).rows[172]
    assert r["date_of_birth"] is None and r["wikidata"]["dob"]["note"] == "implausible_dob"


def test_parse_core_date_of_death_only_when_a_full_day():
    assert parse_core([row(dods="1998-05-01T00:00:00Z~11")], TODAY).rows[172]["date_of_death"] == date(1998, 5, 1)
    r = parse_core([row(dods="1998-01-01T00:00:00Z~9")], TODAY).rows[172]
    assert r["date_of_death"] is None and r["wikidata"]["death_year"] == 1998


def test_parse_core_non_numeric_twfy_is_ignored_not_turned_into_a_person_id():
    r = parse_core([row(twfy="abc")], TODAY).rows[172]
    assert r["twfy_id"] is None and r["person_id"] is None


def test_parse_core_skips_non_numeric_mnis_and_reports_it():
    result = parse_core([row(mnis="LEYIBvV9")], TODAY)
    assert result.rows == {} and result.problems == ["non_numeric_mnis:LEYIBvV9"]


def test_parse_core_drops_a_mnis_id_that_appears_on_two_items():
    result = parse_core([row(item="Q1"), row(item="Q2")], TODAY)
    assert 172 not in result.rows and "duplicate_mnis:172" in result.problems


def test_parse_core_drops_a_twfy_id_shared_by_two_members():
    result = parse_core([row(mnis="1", twfy="500"), row(mnis="2", item="Q2", twfy="500")], TODAY)
    assert result.rows[1]["person_id"] is None and result.rows[2]["person_id"] is None
    assert "duplicate_twfy:500" in result.problems
    assert result.rows[1]["twfy_id"] == "500"  # the raw fact is kept; only the unique key is withheld


def test_parse_extras():
    rows = [{"mnis": {"value": "172"}, "pob": {"value": "Paddington"}, "wikipedia": {"value": "https://en.wikipedia.org/wiki/Diane_Abbott"}},
            {"mnis": {"value": "39"}}]
    got = parse_extras(rows)
    assert got[172] == {"place_of_birth": "Paddington", "wikipedia_url": "https://en.wikipedia.org/wiki/Diane_Abbott"}
    assert got[39] == {"place_of_birth": None, "wikipedia_url": None}


# --------------------------------------------------------------------------- the UPDATE statement
# ---------------------------------------------------------------------------


def test_update_refreshes_person_id_from_the_source_instead_of_keeping_a_stale_one():
    sql = build_update(CORE_COLUMNS)
    assert "person_id = u.person_id" in sql and "coalesce" not in sql.lower()


def test_update_has_one_array_per_column_plus_mnis_id_and_person_id():
    for columns in (CORE_COLUMNS, CORE_COLUMNS + EXTRA_COLUMNS):
        assert build_update(columns).count("%s") == len(columns) + 2


def test_core_only_update_never_touches_the_extras_columns():
    core, full = build_update(CORE_COLUMNS), build_update(CORE_COLUMNS + EXTRA_COLUMNS)
    assert "place_of_birth" not in core and "wikipedia_url" not in core
    assert "place_of_birth = u.place_of_birth" in full and "wikipedia_url = u.wikipedia_url" in full


def test_update_casts_the_json_column_and_stamps_the_fetch_time():
    sql = build_update(CORE_COLUMNS)
    assert "wikidata = u.wikidata::jsonb" in sql and "wikidata_fetched_at = now()" in sql


# --------------------------------------------------------------------------- queries
# ---------------------------------------------------------------------------


def test_core_query_uses_the_mnis_property_and_only_our_ids():
    q = core_query([172, 4972, 172])
    assert "wdt:P10428" in q and "P6213" not in q
    assert '"172"' in q and '"4972"' in q and q.count('"172"') == 1
    assert "P569" in q and "P570" in q and "P2171" in q and "P6465" in q


def test_queries_reject_anything_that_is_not_an_integer():
    with pytest.raises(ValueError):
        core_query(["172", '1" } DROP'])
    with pytest.raises(ValueError):
        extras_query(["x"])


def test_extras_query_covers_place_of_birth_and_wikipedia():
    q = extras_query([172])
    assert "P19" in q and "wikipedia.org" in q and '"172"' in q
