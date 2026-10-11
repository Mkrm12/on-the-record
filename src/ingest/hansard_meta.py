"""Hansard contribution metadata the original ingest dropped."""

from datetime import date
from typing import Any

from ingest.http import PoliteClient

SPOKEN_URL = "https://hansard-api.parliament.uk/search/contributions/Spoken.json"
_TEXT_KEYS = ("ContributionText", "ContributionTextFull")


# The API writes one tag in two spellings — `hs_Para` 103,363 times and `hs_para` 43,176 times in this corpus — so any
# filter on the tag would silently skip a third of the rows it meant.
CANONICAL_HRS_TAGS = {t.casefold(): t for t in ("hs_Para", "hs_TabledBy", "Question", "NewDebate")}


def canonical_hrs_tag(value: str | None) -> str | None:
    if not value:
        return None
    return CANONICAL_HRS_TAGS.get(value.casefold(), value)


def parse_item(item: dict) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """(speech fields, debate fields), or None when the item cannot be placed (no contribution or debate id)."""
    speech_id, debate_id = item.get("ContributionExtId"), item.get("DebateSectionExtId")
    if not speech_id or not debate_id:
        return None
    speech = {
        "id": speech_id, "item_id": item.get("ItemId"), "hrs_tag": canonical_hrs_tag(item.get("HRSTag")),
        "hansard_section": item.get("HansardSection"), "api_rank": item.get("Rank"),
        "api_member_name": item.get("MemberName"),
        "meta": {k: v for k, v in item.items() if k not in _TEXT_KEYS},
    }  # fmt: skip
    debate = {"id": debate_id, "section_id": item.get("DebateSectionId"), "section_order": item.get("DebateSectionOrder")}
    return speech, debate


def merge_items(items: list[dict]) -> dict[str, dict]:
    """Deduplicate by ContributionExtId (first wins), dropping items that cannot be placed."""
    merged: dict[str, dict] = {}
    for item in items:
        if parse_item(item) is not None:
            merged.setdefault(item["ContributionExtId"], item)
    return merged


def text_differs(item: dict, stored: str | None) -> bool:
    """Has Hansard changed the text since we stored it? (The ingest strips NUL bytes and nothing else.)"""
    api = (item.get("ContributionTextFull") or item.get("ContributionText") or "").replace("\x00", "")
    return api != (stored or "")


async def fetch_day(client: PoliteClient, day: date, order: str, page_size: int = 50) -> list[dict]:
    """Every contribution for one sitting day in one sort order. A day is at most a few hundred rows, so the deep-
    offset 500 errors that hit month-long queries never arise.
    """
    items: list[dict] = []
    skip = 0
    while True:
        payload = await client.get_json(SPOKEN_URL, {
            "queryParameters.startDate": day.isoformat(), "queryParameters.endDate": day.isoformat(),
            "queryParameters.orderBy": order, "queryParameters.take": page_size, "queryParameters.skip": skip,
        })  # fmt: skip
        batch = (payload or {}).get("Results") or []
        items.extend(batch)
        skip += page_size
        if not batch or skip >= ((payload or {}).get("TotalResultCount") or 0):
            break
    return items
