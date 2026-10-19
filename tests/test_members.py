"""Members API parsing and the polite async client."""

import asyncio
import copy
import json
import time
from datetime import date
from pathlib import Path

import httpx
import pytest

from ingest.members import (
    COLUMNS,
    OPTIONAL_ENDPOINTS,
    FetchError,
    MembersClient,
    parse_member,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "members.json").read_text(encoding="utf-8"))


def payloads(mnis_id: str) -> dict:
    return copy.deepcopy(FIXTURES[mnis_id])


# --------------------------------------------------------------------------- parse_member
# ---------------------------------------------------------------------------


def test_row_has_exactly_the_declared_columns():
    row = parse_member(172, payloads("172"), [])
    assert set(row) == {"mnis_id", *COLUMNS}


def test_commons_member_typed_fields():
    row = parse_member(172, payloads("172"), [])
    assert row["mnis_id"] == 172
    assert row["name"] == "Ms Diane Abbott"
    assert row["name_list_as"] == "Abbott, Ms Diane"
    assert row["name_full_title"] == "Rt Hon Diane Abbott MP"
    assert row["name_address_as"] == "Ms Abbott"
    assert row["gender"] == "F"
    assert (row["party"], row["party_id"], row["party_abbreviation"]) == ("Labour", 15, "Lab")
    assert row["party_colour"] == "d50000"
    assert row["house"] == "Commons"
    assert row["constituency"] == row["membership_from"] == "Hackney North and Stoke Newington"
    assert row["membership_from_id"] == 4074
    assert row["is_active"] is True
    assert row["status"] == "Current Member"
    assert row["membership_start_date"] == date(1987, 6, 11)
    assert row["membership_end_date"] is None
    assert row["membership_end_reason"] is None
    assert row["thumbnail_url"].startswith("https://members-api.parliament.uk/")


def test_commons_member_election_result():
    row = parse_member(172, payloads("172"), [])
    assert row["last_election_result"] == "Lab Hold"
    assert row["last_election_date"] == date(2024, 7, 4)
    assert row["last_election_majority"] == 15080
    assert row["last_election_turnout"] == 40959
    assert row["last_election_electorate"] == 77812


def test_lord_has_no_constituency_and_no_election():
    row = parse_member(4972, payloads("4972"), [])
    assert row["house"] == "Lords"
    # "Life peer" is a membership type, not a constituency; don't store it as one.
    assert row["constituency"] is None
    assert row["membership_from"] == "Life peer"
    assert row["last_election_result"] is None
    assert row["last_election_majority"] is None
    assert row["missing_endpoints"] == []  # a 404 election is absence, not failure


def test_synopsis_is_plain_text():
    row = parse_member(172, payloads("172"), [])
    assert "<" not in row["synopsis"]
    assert row["synopsis"].startswith("The Rt Hon Ms Diane Abbott is the Labour MP for Hackney North")


def test_raw_keeps_every_endpoint_unwrapped_and_json_serialisable():
    row = parse_member(172, payloads("172"), [])
    assert set(row["raw"]) == {"profile", *OPTIONAL_ENDPOINTS}
    assert row["raw"]["profile"]["nameDisplayAs"] == "Ms Diane Abbott"  # "value" wrapper removed
    assert isinstance(row["raw"]["contact"], list)
    assert row["raw"]["biography"]["representations"][0]["name"] == "Hackney North and Stoke Newington"
    assert row["raw"]["focus"][0]["category"]
    json.dumps(row["raw"])


def test_absent_optional_endpoint_is_null_in_raw():
    row = parse_member(4972, payloads("4972"), [])
    assert "election" in row["raw"] and row["raw"]["election"] is None


def test_member_since_is_earliest_house_membership():
    p = payloads("172")
    p["biography"]["value"]["houseMemberships"] = [
        {"house": 1, "startDate": "2010-05-06T00:00:00"},
        {"house": 1, "startDate": "1987-06-11T00:00:00"},
        {"house": 2, "startDate": "2020-01-01T00:00:00"},
    ]
    assert parse_member(172, p, [])["member_since"] == date(1987, 6, 11)


def test_member_since_falls_back_to_current_membership_start():
    p = payloads("172")
    del p["biography"]
    assert parse_member(172, p, ["biography"])["member_since"] == date(1987, 6, 11)


def test_failed_endpoints_are_recorded_and_left_out_of_raw():
    p = payloads("172")
    del p["contact"]
    row = parse_member(172, p, ["contact"])
    assert row["missing_endpoints"] == ["contact"]
    assert "contact" not in row["raw"]
    assert row["name"] == "Ms Diane Abbott"  # everything else still parsed


def test_unknown_member_becomes_a_stub_named_from_hansard():
    row = parse_member(99999999, {"profile": None}, ["profile"], fallback_name="Mr Speaker")
    assert row["mnis_id"] == 99999999
    assert row["name"] == "Mr Speaker"
    assert row["missing_endpoints"] == ["profile"]
    assert row["raw"] == {}
    assert row["party"] is None and row["house"] is None and row["is_active"] is None


# --------------------------------------------------------------------------- MembersClient: retries, 404 semantics,
# rate limiting ---------------------------------------------------------------------------


def with_client(handler, body, **kwargs):
    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = MembersClient(http, **{"min_interval": 0, "backoff_base": 0, **kwargs})
            return await body(client)

    return asyncio.run(main())


def test_get_returns_json():
    handler = lambda request: httpx.Response(200, json={"value": 1})  # noqa: E731
    assert with_client(handler, lambda c: c.get("172")) == {"value": 1}


def test_404_returns_none_without_retrying():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(404, text="not found")

    assert with_client(handler, lambda c: c.get("999")) is None
    assert calls == ["/api/Members/999"]


def test_retries_429_honouring_retry_after_then_succeeds():
    responses = iter([httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json={"ok": 1})])
    calls = []

    def handler(request):
        calls.append(1)
        return next(responses)

    assert with_client(handler, lambda c: c.get("1")) == {"ok": 1}
    assert len(calls) == 2


def test_retries_5xx_and_transport_errors():
    seq = iter(["boom", 503, 500, 200])
    calls = []

    def handler(request):
        calls.append(1)
        item = next(seq)
        if item == "boom":
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(item, json={"ok": True})

    assert with_client(handler, lambda c: c.get("1"), max_attempts=5) == {"ok": True}
    assert len(calls) == 4


def test_gives_up_after_max_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500)

    with pytest.raises(FetchError):
        with_client(handler, lambda c: c.get("1"), max_attempts=3)
    assert len(calls) == 3


def test_non_retryable_4xx_fails_immediately():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403)

    with pytest.raises(FetchError):
        with_client(handler, lambda c: c.get("1"), max_attempts=5)
    assert len(calls) == 1


def test_min_interval_spaces_out_request_starts():
    handler = lambda request: httpx.Response(200, json={})  # noqa: E731

    async def body(client):
        start = time.monotonic()
        await asyncio.gather(*(client.get(str(i)) for i in range(5)))
        return time.monotonic() - start

    elapsed = with_client(handler, body, min_interval=0.05, concurrency=5)
    assert elapsed >= 0.15  # 4 gaps of 50ms; loose lower bound to tolerate timer granularity


def test_concurrency_is_capped():
    in_flight = 0
    peak = 0

    async def handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json={})

    async def body(client):
        await asyncio.gather(*(client.get(str(i)) for i in range(20)))

    with_client(handler, body, concurrency=3)
    assert peak <= 3


# --------------------------------------------------------------------------- fetch_member
# ---------------------------------------------------------------------------


def test_fetch_member_collects_every_endpoint():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"value": {"id": 172}})

    fetched = with_client(handler, lambda c: c.fetch_member(172))
    assert fetched.mnis_id == 172
    assert set(fetched.payloads) == {"profile", *OPTIONAL_ENDPOINTS}
    assert fetched.missing == []
    assert len(seen) == 1 + len(OPTIONAL_ENDPOINTS)
    assert "/api/Members/172" in seen and "/api/Members/172/RegisteredInterests" in seen


def test_fetch_member_unknown_profile_stops_early():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(404)

    fetched = with_client(handler, lambda c: c.fetch_member(999))
    assert fetched.payloads == {"profile": None}
    assert fetched.missing == ["profile"]
    assert seen == ["/api/Members/999"]  # no wasted calls for a member that does not exist


def test_fetch_member_optional_404_is_absence_but_persistent_failure_is_missing():
    def handler(request):
        path = request.url.path
        if path.endswith("/LatestElectionResult"):
            return httpx.Response(404)
        if path.endswith("/Contact"):
            return httpx.Response(500)
        return httpx.Response(200, json={"value": {}})

    fetched = with_client(handler, lambda c: c.fetch_member(4972), max_attempts=2)
    assert fetched.payloads["election"] is None
    assert "contact" not in fetched.payloads
    assert fetched.missing == ["contact"]


def test_fetch_member_profile_failure_propagates():
    handler = lambda request: httpx.Response(500)  # noqa: E731
    with pytest.raises(FetchError):
        with_client(handler, lambda c: c.fetch_member(1), max_attempts=2)
