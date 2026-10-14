"""Hansard contribution metadata the original ingest dropped: HRSTag, HansardSection, ItemId, Rank, the API's
MemberName, and (for the debate) DebateSectionId and DebateSectionOrder, the running order of debates within a
sitting that the Timeline block depends on.
"""

import asyncio
import copy
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from ingest.hansard_meta import fetch_day, merge_items, parse_item, text_differs
from ingest.http import PoliteClient

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "hansard_items.json").read_text(encoding="utf-8"))


def item(i=0):
    return copy.deepcopy(FIXTURE["items"][i])


def test_parse_item_extracts_the_dropped_speech_fields():
    it = item()
    speech, debate = parse_item(it)
    assert speech["id"] == it["ContributionExtId"]
    assert speech["item_id"] == it["ItemId"] and isinstance(speech["item_id"], int)
    # Canonicalised: the API writes "hs_para" and "hs_Para" for one tag (43,176 and 103,363 rows in this corpus), so the
    # stored form is the canonical one and the raw spelling survives in `speech["meta"]`.
    assert speech["hrs_tag"] == "hs_Para" and speech["hansard_section"] == it["HansardSection"]
    assert speech["meta"]["HRSTag"] == it["HRSTag"], "the source's own spelling is kept verbatim in meta"
    assert speech["api_rank"] == it["Rank"] and speech["api_member_name"] == it["MemberName"]


def test_parse_item_extracts_the_debate_ordering():
    it = item()
    _, debate = parse_item(it)
    assert debate == {"id": it["DebateSectionExtId"], "section_id": it["DebateSectionId"], "section_order": it["DebateSectionOrder"]}


def test_meta_keeps_everything_except_the_two_long_texts():
    speech, _ = parse_item(item())
    assert "ContributionText" not in speech["meta"] and "ContributionTextFull" not in speech["meta"]
    assert {"HRSTag", "HansardSection", "Timecode", "Section", "House", "SittingDate"} <= set(speech["meta"])
    json.dumps(speech["meta"])


def test_parse_item_tolerates_missing_and_null_fields():
    speech, debate = parse_item({"ContributionExtId": "X", "DebateSectionExtId": "D"})
    assert speech["id"] == "X" and speech["item_id"] is None and speech["hrs_tag"] is None and speech["api_rank"] is None
    assert debate == {"id": "D", "section_id": None, "section_order": None}


def test_parse_item_without_an_id_returns_none():
    assert parse_item({"ItemId": 1}) is None
    assert parse_item({"ContributionExtId": "X"}) is None  # no debate: cannot place it


def test_merge_items_dedupes_by_contribution_id():
    a, b = item(0), item(1)
    merged = merge_items([a, b, copy.deepcopy(a)])
    assert set(merged) == {a["ContributionExtId"], b["ContributionExtId"]}


def test_merge_items_skips_unusable_items():
    assert merge_items([{"ItemId": 1}, {}]) == {}


@pytest.mark.parametrize(
    "api_full,api_short,stored,expected",
    [
        ("Hello world.", "Hello", "Hello world.", False),
        ("Hello\x00 world.", None, "Hello world.", False),  # the ingest strips NULs
        (None, "Short text", "Short text", False),
        ("Hello world.", None, "Hello there.", True),
        ("Hello world.", None, None, True),
        (None, None, "", False),
        (None, None, None, False),
    ],
)
def test_text_differs(api_full, api_short, stored, expected):
    assert text_differs({"ContributionTextFull": api_full, "ContributionText": api_short}, stored) is expected


def run_with(handler, body, **kwargs):
    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await body(PoliteClient(http, **{"min_interval": 0, "backoff_base": 0, **kwargs}))

    return asyncio.run(main())


def test_fetch_day_pages_one_sitting_day_in_the_requested_order():
    seen, results = [], [{"ContributionExtId": str(i)} for i in range(5)]

    def handler(request):
        p = request.url.params
        seen.append(dict(p))
        skip, take = int(p["queryParameters.skip"]), int(p["queryParameters.take"])
        return httpx.Response(200, json={"Results": results[skip : skip + take], "TotalResultCount": 5})

    got = run_with(handler, lambda c: fetch_day(c, date(2026, 9, 11), "SittingDateDesc", page_size=2))
    assert [r["ContributionExtId"] for r in got] == ["0", "1", "2", "3", "4"]
    assert [p["queryParameters.skip"] for p in seen] == ["0", "2", "4"]
    assert seen[0]["queryParameters.startDate"] == seen[0]["queryParameters.endDate"] == "2026-09-11"
    assert seen[0]["queryParameters.orderBy"] == "SittingDateDesc"


def test_fetch_day_with_no_contributions():
    handler = lambda request: httpx.Response(200, json={"Results": [], "TotalResultCount": 0})  # noqa: E731
    assert run_with(handler, lambda c: fetch_day(c, date(2026, 9, 12), "SittingDateAsc")) == []


def test_fetch_day_stops_at_the_reported_total_even_if_pages_keep_coming():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"Results": [{"ContributionExtId": "x"}, {"ContributionExtId": "y"}], "TotalResultCount": 4})

    got = run_with(handler, lambda c: fetch_day(c, date(2026, 9, 11), "SittingDateAsc", page_size=2))
    assert len(got) == 4 and len(calls) == 2


@pytest.mark.parametrize(("raw", "canonical"), [
    ("hs_para", "hs_Para"), ("hs_Para", "hs_Para"), ("HS_PARA", "hs_Para"),
    ("hs_tabledby", "hs_TabledBy"), ("question", "Question"), ("newdebate", "NewDebate"),
    ("hs_SomethingNew", "hs_SomethingNew"), (None, None), ("", None),
])
def test_the_tag_is_canonicalised_and_an_unknown_one_is_kept_as_given(raw, canonical):
    from ingest.hansard_meta import canonical_hrs_tag

    assert canonical_hrs_tag(raw) == canonical
