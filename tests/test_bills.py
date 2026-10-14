"""Bills API: parsing, per-date stage sittings, linking sittings to debates, and linking a division to its bill stage
(the two-step path a division takes: bill title + stage keyword + date, else its debate).
"""

import asyncio
import copy
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from ingest.bills import (
    StageRef,
    bill_relevant,
    fetch_bill,
    fetch_recent_bills,
    link_division_stage,
    link_sitting,
    parse_bill,
    parse_stages,
    stage_keywords,
)
from ingest.divisions import DebateInfo
from ingest.http import PoliteClient

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "bills.json").read_text(encoding="utf-8"))


def real(bill_id):
    return copy.deepcopy(FIXTURES[str(bill_id)])


def d(day, month=9):
    return date(2026, month, day)


# --------------------------------------------------------------------------- parse_bill
# ---------------------------------------------------------------------------


def test_parse_bill_real_fields():
    f = real(4124)
    b = parse_bill(f["list"], f["detail"])
    assert (b["id"], b["title"]) == (4124, "Health Bill")
    assert b["long_title"] and b["current_house"] in ("Commons", "Lords") and b["originating_house"] in ("Commons", "Lords")
    assert (b["is_act"], b["is_defeated"], b["is_withdrawn"]) == (False, False, False)
    assert isinstance(b["last_update"], datetime)
    assert set(b["raw"]) == {"list", "detail"}
    json.dumps(b["raw"], default=str)


def test_parse_bill_withdrawn_and_missing_detail():
    f = real(4124)
    f["list"]["billWithdrawn"] = "2026-09-01T00:00:00"
    b = parse_bill(f["list"], None)
    assert b["is_withdrawn"] is True and b["long_title"] is None and b["raw"]["detail"] is None


def test_parse_bill_keeps_the_former_title():
    f = real(4124)
    f["list"]["formerShortTitle"] = "Old Name Bill"
    assert parse_bill(f["list"], f["detail"])["former_short_title"] == "Old Name Bill"


# --------------------------------------------------------------------------- parse_stages
# ---------------------------------------------------------------------------


def test_parse_stages_real_multi_sitting_stage():
    f = real(4124)
    stages, sittings = parse_stages(4124, f["stages"])
    rs = next(s for s in stages if s["abbreviation"] == "RS")
    assert rs["stage_name"] == "Report stage" and rs["house"] == "Commons" and rs["bill_id"] == 4124
    assert (rs["first_sitting"], rs["last_sitting"], rs["date"]) == (d(7), d(8), d(7))
    rs_sittings = [s for s in sittings if s["bill_stage_id"] == rs["id"]]
    assert sorted(s["date"] for s in rs_sittings) == [d(7), d(8)]
    assert all(s["bill_id"] == 4124 and s["house"] == "Commons" and s["debate_id"] is None for s in rs_sittings)


def test_parse_stages_counts_every_sitting():
    f = real(4124)
    stages, sittings = parse_stages(4124, f["stages"])
    assert len(stages) == len(f["stages"]["items"])
    assert len(sittings) == sum(len(s["stageSittings"]) for s in f["stages"]["items"])
    cs = next(s for s in stages if s["abbreviation"] == "CS")
    assert len([x for x in sittings if x["bill_stage_id"] == cs["id"]]) == 9


def test_parse_stages_a_stage_with_no_sittings_has_no_dates():
    stages, sittings = parse_stages(4129, real(4129)["stages"])
    empty = [s for s in stages if s["date"] is None]
    assert empty and all(s["first_sitting"] is None and s["last_sitting"] is None for s in empty)
    assert all(x["bill_stage_id"] not in {s["id"] for s in empty} for x in sittings)


def synthetic_stages(**overrides):
    stage = {"id": 900, "stageId": 11, "sessionId": 40, "description": "Royal Assent", "abbreviation": "RA",
             "house": "Unassigned", "sortOrder": 9, "stageSittings": [{"id": 5, "stageId": 11, "billStageId": 900, "billId": 1, "date": "2026-09-10T00:00:00"}]}
    stage.update(overrides)
    return {"items": [stage], "totalResults": 1, "itemsPerPage": 20}


def test_parse_stages_unassigned_house_becomes_null():
    stages, sittings = parse_stages(1, synthetic_stages())
    assert stages[0]["house"] is None and sittings[0]["house"] is None


def test_parse_stages_handles_missing_items_and_null_sittings():
    assert parse_stages(1, {"items": []}) == ([], [])
    assert parse_stages(1, None) == ([], [])
    stages, sittings = parse_stages(1, synthetic_stages(stageSittings=None))
    assert stages[0]["date"] is None and sittings == []


def test_parse_stages_sitting_date_with_time_is_truncated_to_the_day():
    _, sittings = parse_stages(1, synthetic_stages())
    assert sittings[0]["date"] == date(2026, 9, 10)


# --------------------------------------------------------------------------- bill_relevant
# ---------------------------------------------------------------------------


def sit(day, month=9):
    return {"date": date(2026, month, day)}


def test_a_bill_is_relevant_when_any_sitting_falls_in_the_window_inclusive():
    start, end = date(2026, 8, 10), date(2026, 9, 19)
    assert bill_relevant([sit(10, 8)], start, end) and bill_relevant([sit(19)], start, end)
    assert not bill_relevant([sit(9, 8)], start, end) and not bill_relevant([sit(20)], start, end)
    assert bill_relevant([sit(1, 3), sit(5)], start, end)
    assert not bill_relevant([], start, end)


def test_real_health_bill_is_relevant_and_a_bill_with_only_future_sittings_is_not():
    start, end = date(2026, 8, 10), date(2026, 9, 19)
    _, sittings = parse_stages(4124, real(4124)["stages"])
    assert bill_relevant(sittings, start, end)
    assert not bill_relevant([{"date": date(2026, 11, 27)}], start, end)


# --------------------------------------------------------------------------- link_sitting
# ---------------------------------------------------------------------------


def deb(did, day, heading, house="Commons", section=None):
    return DebateInfo(did, d(day), house, heading, section or f"{house} Chamber", ())


def test_sitting_links_to_the_debate_with_the_bill_title_on_that_day_and_house():
    debates = [deb("A", 7, "Health Bill"), deb("B", 7, "Other Bill")]
    assert link_sitting("Health Bill", "Commons", d(7), debates) == ("A", "title_exact")


def test_sitting_link_ignores_hl_and_lords_markers():
    assert link_sitting("Social Housing Bill [HL]", "Commons", d(10), [deb("S", 10, "Social Housing Bill [Lords]")]) == ("S", "title_exact")


@pytest.mark.parametrize(
    "debates",
    [
        [deb("A", 8, "Health Bill")],  # wrong day
        [deb("A", 7, "Health Bill", "Lords")],  # wrong house
        [deb("A", 7, "Health and Care Bill")],  # different bill
        [],
    ],
)
def test_sitting_does_not_link_across_days_houses_or_bills(debates):
    assert link_sitting("Health Bill", "Commons", d(7), debates) == (None, "none")


def test_several_debates_for_one_bill_on_one_day_pick_the_lowest_id_and_say_so():
    debates = [deb("b", 7, "Health Bill"), deb("a", 7, "Health Bill", section="Grand Committee")]
    assert link_sitting("Health Bill", "Commons", d(7), debates) == ("a", "title_exact+multi")


def test_sitting_with_no_house_links_by_title_and_date_alone():
    assert link_sitting("Royal Assent Bill", None, d(7), [deb("A", 7, "Royal Assent Bill")]) == ("A", "title_exact")


# --------------------------------------------------------------------------- stage_keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Health Bill: Report Stage: New Clause 6", {"RS"}),
        ("Health Bill: Report Stage: Amendment 1", {"RS"}),
        ("Terminally Ill Adults (End of Life) Bill: Second Reading", {"2R"}),
        ("Social Housing Bill [Lords]: Reasoned Amendment to Second Reading", {"2R"}),
        ("Representation of the People Bill: Third Reading", {"3R"}),
        ("X Bill: Committee Stage", {"CS", "COTWH"}),
        ("X Bill: Committee of the whole House", {"CS", "COTWH"}),
        ("X Bill: Consideration of Lords Amendments", {"COLA"}),
        ("X Bill: Lords amendments", {"COLA"}),
        ("X Bill: Programme motion", {"PM"}),
        ("X Bill: Money Resolution", {"MR"}),
        ("X Bill: Ways and Means Resolution", {"WAMR"}),
        ("Closure motion", set()),
        ("Financial Services and Markets Bill [HL]", set()),
        ("Motion to sit in private", set()),
        (None, set()),
    ],
)
def test_stage_keywords(title, expected):
    assert stage_keywords(title) == expected


def test_a_bill_whose_title_contains_a_stage_word_only_counts_after_the_colon():
    assert stage_keywords("Second Reading Reform Bill: Report Stage") == {"RS"}
    assert stage_keywords("Second Reading Reform Bill") == set()


# --------------------------------------------------------------------------- link_division_stage
# ---------------------------------------------------------------------------

HEALTH, FSM = "Health Bill", "Financial Services and Markets Bill [HL]"
STAGES = [
    StageRef(1, 10, HEALTH, "PM", "Commons", frozenset({d(7)}), frozenset({"d1"})),
    StageRef(2, 10, HEALTH, "RS", "Commons", frozenset({d(7), d(8)}), frozenset({"d1"})),
    StageRef(3, 10, HEALTH, "3R", "Commons", frozenset({d(8)}), frozenset({"d2"})),
    StageRef(4, 20, FSM, "RS", "Lords", frozenset({d(7), d(9)}), frozenset({"d3"})),
    StageRef(5, 20, FSM, "CS", "Lords", frozenset({d(1)}), frozenset()),
    StageRef(6, 30, "Social Housing Bill [HL]", "2R", "Commons", frozenset({d(10)}), frozenset({"d4"})),
]


def division(title, day, house="Commons", debate_id=None, deferred=False):
    return {"title": title, "division_date": d(day), "house": house, "debate_id": debate_id, "is_deferred": deferred}


@pytest.mark.parametrize(
    "title,day,house,expected",
    [
        ("Health Bill: Report Stage: New Clause 6", 7, "Commons", (2, "title_stage")),
        ("Health Bill: Report Stage: Amendment 1", 8, "Commons", (2, "title_stage")),
        ("Health Bill: Third Reading", 8, "Commons", (3, "title_stage")),
        ("Health Bill: Programme motion", 7, "Commons", (1, "title_stage")),
        (FSM, 7, "Lords", (4, "title_date")),
        (FSM, 9, "Lords", (4, "title_date")),
        ("Social Housing Bill [Lords]: Reasoned Amendment to Second Reading", 10, "Commons", (6, "title_stage")),
    ],
)
def test_division_links_by_bill_title_stage_keyword_and_date(title, day, house, expected):
    assert link_division_stage(division(title, day, house), STAGES) == expected


def test_no_keyword_and_two_stages_that_day_is_ambiguous_not_a_guess():
    assert link_division_stage(division("Health Bill", 7), STAGES) == (None, "ambiguous")


def test_keyword_for_a_stage_that_is_not_sitting_that_day_does_not_link():
    assert link_division_stage(division("Health Bill: Second Reading", 7), STAGES) == (None, "none")


def test_house_and_date_must_match():
    assert link_division_stage(division(FSM, 7, "Commons"), STAGES) == (None, "none")
    assert link_division_stage(division("Health Bill: Report Stage", 12), STAGES) == (None, "none")


def test_closure_motion_links_through_its_debate_when_only_one_stage_uses_it():
    assert link_division_stage(division("Closure motion", 8, debate_id="d2"), STAGES) == (3, "debate")


def test_a_stage_reached_through_a_time_guessed_debate_is_labelled_weak():
    # Two inferences deep (time -> debate -> stage): keep the doubt visible instead of looking as sure as a title match.
    weak = {**division("Closure motion", 8, debate_id="d2"), "debate_link": "time_window"}
    assert link_division_stage(weak, STAGES) == (3, "debate_weak")
    for strong in ("title_exact", "tally_event", "title_deferred"):
        assert link_division_stage({**weak, "debate_link": strong}, STAGES) == (3, "debate")


def test_closure_motion_in_a_debate_shared_by_two_stages_is_ambiguous():
    assert link_division_stage(division("Closure motion", 7, debate_id="d1"), STAGES) == (None, "ambiguous")


def test_division_with_no_bill_and_no_debate_does_not_link():
    # `not_a_bill`, not `none`: `none` now means "names a bill and did not link", so the miss rate counts real misses
    # rather than every Opposition Day motion and Budget resolution.
    assert link_division_stage(division("Motion to sit in private", 4), STAGES) == (None, "not_a_bill")
    assert link_division_stage(division("Closure motion", 8, debate_id="unknown"), STAGES) == (None, "not_a_bill")


def test_deferred_divisions_are_statutory_instruments_not_bill_stages():
    assert link_division_stage(division("Health Bill: Report Stage", 7, deferred=True), STAGES) == (None, "not_a_bill")


def test_bill_title_match_is_not_a_substring_match():
    assert link_division_stage(division("Health Bill Reform: Report Stage", 7), STAGES) == (None, "none")


# --------------------------------------------------------------------------- fetching
# ---------------------------------------------------------------------------


def run_with(handler, body, **kwargs):
    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await body(PoliteClient(http, **{"min_interval": 0, "backoff_base": 0, **kwargs}))

    return asyncio.run(main())


def test_fetch_recent_bills_pages_until_it_passes_the_cutoff():
    pages = {
        0: [{"billId": 1, "lastUpdate": "2026-09-18T10:00:00"}, {"billId": 2, "lastUpdate": "2026-09-10T10:00:00"}],
        2: [{"billId": 3, "lastUpdate": "2026-08-20T10:00:00"}, {"billId": 4, "lastUpdate": "2026-07-01T10:00:00"}],
        4: [{"billId": 5, "lastUpdate": "2026-06-01T10:00:00"}],
    }
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"items": pages[int(request.url.params["Skip"])], "totalResults": 5})

    items = run_with(handler, lambda c: fetch_recent_bills(c, date(2026, 8, 10), page_size=2))
    assert [b["billId"] for b in items] == [1, 2, 3]
    assert [p["Skip"] for p in seen] == ["0", "2"]  # page 3 never requested: page 2 already crossed the cutoff
    assert seen[0]["SortOrder"] == "DateUpdatedDescending" and seen[0]["Take"] == "2"


def test_fetch_recent_bills_stops_on_an_empty_page():
    handler = lambda request: httpx.Response(200, json={"items": [], "totalResults": 0})  # noqa: E731
    assert run_with(handler, lambda c: fetch_recent_bills(c, date(2026, 8, 10))) == []


def test_fetch_bill_returns_detail_and_stages():
    def handler(request):
        if request.url.path.endswith("/Stages"):
            return httpx.Response(200, json={"items": [{"id": 1}]})
        return httpx.Response(200, json={"billId": 7})

    detail, stages = run_with(handler, lambda c: fetch_bill(c, 7))
    assert detail == {"billId": 7} and stages == {"items": [{"id": 1}]}


def test_fetch_bill_that_404s_is_reported_as_missing():
    handler = lambda request: httpx.Response(404)  # noqa: E731
    assert run_with(handler, lambda c: fetch_bill(c, 7)) == (None, None)


# A stage's House is Commons, Lords, or unknown ---------------------------------------------------------


@pytest.mark.parametrize("house, want", [("Commons", "Commons"), ("Lords", "Lords"), ("Unassigned", None),
                                         (None, None), ("Both", None), ("", None), ("Joint", None)])
def test_a_stage_house_the_schema_does_not_know_is_stored_as_unknown(house, want):
    from ingest.bills import parse_stages

    stages, _ = parse_stages(1, {"items": [{"id": 5, "house": house, "description": "2nd reading", "sittings": []}]})
    assert stages[0]["house"] == want
