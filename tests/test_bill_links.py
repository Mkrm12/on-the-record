"""The bill links: a bill sitting to its Hansard debate, and a division to the bill stage it was taken on."""

from __future__ import annotations

from datetime import date

import pytest

from ingest import bills as B
from ingest.divisions import DebateInfo


# --- the names a bill has been known by --------------------------------------------------------------------


@pytest.mark.parametrize(("title", "expected"), [
    ("Pension Schemes Act 2026", {"Pension Schemes Act 2026", "Pension Schemes Bill"}),
    ("Employment Rights Act 2025", {"Employment Rights Act 2025", "Employment Rights Bill"}),
    ("Biodiversity Beyond National Jurisdiction Act 2026",
     {"Biodiversity Beyond National Jurisdiction Act 2026", "Biodiversity Beyond National Jurisdiction Bill"}),
    ("Finance Act 2020 ", {"Finance Act 2020", "Finance Bill"}),
    ("Crime and Policing Bill", {"Crime and Policing Bill"}),
    ("Financial Services and Markets Bill [HL]", {"Financial Services and Markets Bill [HL]"}),
])
def test_a_bill_is_known_by_its_act_name_and_its_bill_name(title, expected):
    assert set(B.bill_name_forms({"title": title})) == expected


def test_the_former_short_title_is_one_of_the_names():
    forms = B.bill_name_forms({"title": "Victims and Courts Act 2026", "former_short_title": "Victims Bill"})
    assert "Victims Bill" in forms and "Victims and Courts Bill" in forms


def test_a_numbered_finance_act_keeps_its_number_as_a_bill():
    """"Finance (No. 2) Act 2024" was "Finance (No. 2) Bill" — the number is part of the name, not decoration."""
    assert "Finance (No. 2) Bill" in B.bill_name_forms({"title": "Finance (No. 2) Act 2024"})


def test_an_act_word_inside_a_title_is_not_rewritten():
    """Only a trailing "Act YYYY" is the Royal Assent rename. "Representation of the People Act 1983 (Amendment) Bill"
    names an earlier Act and must not become anything else.
    """
    forms = B.bill_name_forms({"title": "Representation of the People Act 1983 (Amendment) Bill"})
    assert forms == ("Representation of the People Act 1983 (Amendment) Bill",)


def test_names_are_stripped_and_deduplicated():
    assert B.bill_name_forms({"title": "  Health Bill  ", "former_short_title": "Health Bill"}) == ("Health Bill",)


def test_a_bill_with_no_title_has_no_names():
    assert B.bill_name_forms({"title": None}) == ()
    assert B.bill_name_forms({}) == ()


def test_parse_bill_strips_the_stored_title():
    """"Finance Act 2020 " was stored with its trailing space, from the source."""
    parsed = B.parse_bill({"billId": 1, "shortTitle": "Finance Act 2020 "}, {})
    assert parsed["title"] == "Finance Act 2020"


# --- sitting -> debate ----------------------------------------------------------------------------------


def debate(i, heading, *, day=date(2026, 1, 22), house="Lords"):
    return DebateInfo(i, day, house, heading, "Lords Chamber")


def test_an_act_title_now_finds_the_bill_debate():
    debates = [debate("d1", "Arrangement of Business"), debate("d2", "Pension Schemes Bill")]
    found, basis = B.link_sitting(B.bill_name_forms({"title": "Pension Schemes Act 2026"}), "Lords",
                                  date(2026, 1, 22), debates)  # fmt: skip
    assert found == "d2" and basis.startswith("title_exact")


def test_a_single_title_string_is_still_accepted():
    """Existing callers pass one title; that must keep working, not silently match nothing."""
    found, _ = B.link_sitting("Pension Schemes Bill", "Lords", date(2026, 1, 22), [debate("d2", "Pension Schemes Bill")])
    assert found == "d2"


def test_a_different_day_is_not_a_match():
    debates = [debate("d2", "Pension Schemes Bill", day=date(2026, 1, 23))]
    assert B.link_sitting(("Pension Schemes Bill",), "Lords", date(2026, 1, 22), debates) == (None, "none")


def test_a_different_house_is_not_a_match():
    debates = [debate("d2", "Pension Schemes Bill", house="Commons")]
    assert B.link_sitting(("Pension Schemes Bill",), "Lords", date(2026, 1, 22), debates) == (None, "none")


def test_an_unrelated_heading_that_shares_a_word_is_not_a_match():
    debates = [debate("d2", "Pension Credit")]
    assert B.link_sitting(("Pension Schemes Bill",), "Lords", date(2026, 1, 22), debates) == (None, "none")


def test_two_debates_on_one_bill_one_day_are_reported_as_multi():
    """A Committee of the whole House split across two headings is two debates; the link says so rather than pretending
    there is one.
    """
    debates = [debate("a", "Pension Schemes Bill"), debate("b", "Pension Schemes Bill")]
    found, basis = B.link_sitting(("Pension Schemes Bill",), "Lords", date(2026, 1, 22), debates)
    assert found == "a" and basis.endswith("+multi")


# --- division -> bill stage -----------------------------------------------------------------------------


def stage(stage_id, title, abbreviation, *, house="Commons", day=date(2025, 11, 19), debates=()):
    return B.StageRef(stage_id, 100 + stage_id, title, abbreviation, house, frozenset({day}), frozenset(debates),
                      bill_titles=frozenset(B.normalise_title(t) for t in B.bill_name_forms({"title": title})))


def division(title, *, house="Commons", day=date(2025, 11, 19), deferred=False, debate_id=None):
    return {"house": house, "division_date": day, "title": title, "is_deferred": deferred,
            "debate_id": debate_id, "debate_link": "hansard_division_list"}


def test_a_division_on_a_renamed_bill_finds_its_stage():
    """"Border Security, Asylum and Immigration Bill: motion to disagree with Lords Amendment 7" against a bill stored
    as "... Act 2025" — the case that left 0 of 586 linked.
    """
    stages = [stage(1, "Border Security, Asylum and Immigration Act 2025", "COLA")]
    found, method = B.link_division_stage(
        division("Border Security, Asylum and Immigration Bill: motion to disagree with Lords Amendment 7"), stages)
    assert found == 1 and method == "title_stage"


def test_the_stage_keyword_chooses_between_two_stages_the_same_day():
    """Programme motion and Report stage often sit together; the title says which."""
    stages = [stage(1, "Health Act 2026", "PM"), stage(2, "Health Act 2026", "RS")]
    assert B.link_division_stage(division("Health Bill: Report Stage: New Clause 6"), stages)[0] == 2


def test_two_equally_good_stages_are_refused_not_guessed():
    stages = [stage(1, "Health Act 2026", "RS"), stage(2, "Health Act 2026", "RS")]
    assert B.link_division_stage(division("Health Bill: Report Stage"), stages) == (None, "ambiguous")


def test_a_deferred_division_is_never_a_bill_stage():
    """Deferred divisions are statutory instruments."""
    stages = [stage(1, "Health Act 2026", "RS")]
    assert B.link_division_stage(division("Health Bill", deferred=True), stages) == (None, "not_a_bill")


def test_a_division_on_another_bill_is_not_linked():
    stages = [stage(1, "Health Act 2026", "RS")]
    assert B.link_division_stage(division("Wealth Bill: Report Stage"), stages) == (None, "none")


def test_a_lords_title_with_no_stage_links_by_title_and_date():
    """Lords division titles are usually just the bill; the date and House decide."""
    stages = [stage(1, "Pension Schemes Act 2026", "CS", house="Lords", day=date(2026, 1, 22))]
    found, method = B.link_division_stage(division("Pension Schemes Bill", house="Lords", day=date(2026, 1, 22)),
                                          stages)  # fmt: skip
    assert found == 1 and method == "title_date"


def test_a_stageref_without_titles_falls_back_to_its_bill_title():
    """StageRef is constructed positionally in older code; the name forms default from `bill_title`."""
    ref = B.StageRef(1, 101, "Health Act 2026", "RS", "Commons", frozenset({date(2025, 11, 19)}), frozenset())
    assert B.normalise_title("Health Bill") in ref.names()
