"""A polite Members API client and the row parser."""

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import Any

from ingest.text import clean_text
from ingest.http import FetchError, PoliteClient

BASE_URL = "https://members-api.parliament.uk/api/Members"
HOUSES = {1: "Commons", 2: "Lords"}

# raw key -> URL suffix.
OPTIONAL_ENDPOINTS = {
    "biography": "Biography",  # posts, committees, party affiliations, past seats
    "contact": "Contact",
    "synopsis": "Synopsis",
    "experience": "Experience",
    "focus": "Focus",  # declared areas of interest
    "staff": "Staff",
    "interests": "RegisteredInterests",
    "election": "LatestElectionResult",  # majority, turnout, candidates (Commons only)
}

# Typed columns written by the upsert, in table order.
COLUMNS = (
    "name", "name_list_as", "name_full_title", "name_address_as", "gender",
    "party", "party_id", "party_abbreviation", "party_colour",
    "house", "membership_from", "membership_from_id", "constituency",
    "member_since", "membership_start_date", "membership_end_date", "membership_end_reason",
    "is_active", "status", "thumbnail_url", "synopsis",
    "last_election_result", "last_election_date", "last_election_majority",
    "last_election_turnout", "last_election_electorate",
    "raw", "missing_endpoints",
)  # fmt: skip

@dataclass
class FetchedMember:
    mnis_id: int
    payloads: dict[str, Any]  # endpoint key -> response; None where the API said 404
    missing: list[str]  # endpoints that failed after retries (a 404 is absence, not failure)


# --------------------------------------------------------------------------- Client
# ---------------------------------------------------------------------------


class MembersClient(PoliteClient):
    """Members API client: PoliteClient plus the nine-endpoint member fetch."""

    async def get(self, path: str) -> Any | None:
        """GET {BASE_URL}/{path} as JSON. Returns None on 404; raises FetchError otherwise on failure."""
        return await self.get_json(f"{BASE_URL}/{path}")

    async def fetch_member(self, mnis_id: int) -> FetchedMember:
        """Fetch the profile, then every optional endpoint concurrently."""
        profile = await self.get(str(mnis_id))
        if profile is None:
            return FetchedMember(mnis_id, {"profile": None}, ["profile"])

        async def one(key: str, suffix: str) -> tuple[str, Any, bool]:
            try:
                return key, await self.get(f"{mnis_id}/{suffix}"), False
            except FetchError:
                return key, None, True

        results = await asyncio.gather(*(one(k, s) for k, s in OPTIONAL_ENDPOINTS.items()))
        payloads: dict[str, Any] = {"profile": profile}
        missing: list[str] = []
        for key, data, failed in results:
            if failed:
                missing.append(key)
            else:
                payloads[key] = data
        return FetchedMember(mnis_id, payloads, sorted(missing))


# --------------------------------------------------------------------------- Parsing
# ---------------------------------------------------------------------------


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value[:10]) if value else None


def parse_member(
    mnis_id: int,
    payloads: dict[str, Any],
    missing: list[str],
    fallback_name: str | None = None,
) -> dict[str, Any]:
    """Turn fetched payloads into a member_id_map row (keys: mnis_id plus COLUMNS)."""
    raw = {key: _unwrap(data) for key, data in payloads.items() if key in OPTIONAL_ENDPOINTS or key == "profile"}
    profile = raw.pop("profile", None)
    row: dict[str, Any] = {"mnis_id": mnis_id, **dict.fromkeys(COLUMNS)}
    row["missing_endpoints"] = list(missing)

    if profile is None:
        row["name"] = fallback_name
        row["raw"] = {}
        return row

    party = profile.get("latestParty") or {}
    membership = profile.get("latestHouseMembership") or {}
    status = membership.get("membershipStatus") or {}
    # The API leaves an MP who went at a dissolution as "Current Member" (statusIsActive true, dated from the election
    # before) while giving the dissolution as the membership's end: 222 former MPs in this corpus were stored as
    # sitting.
    end = _date(membership.get("membershipEndDate"))
    start = _date(membership.get("membershipStartDate"))
    if start is not None and end is not None and end < start:
        # The source contradicts itself.
        start = None
    ended = end is not None and end <= date.today()
    house = HOUSES.get(membership.get("house"))
    election = raw.get("election") or {}
    biography = raw.get("biography") or {}
    starts = [_date(m.get("startDate")) for m in biography.get("houseMemberships") or []]
    starts = [s for s in starts if s]
    synopsis = raw.get("synopsis")  # HTML with a link to the constituency page

    row.update(
        name=profile.get("nameDisplayAs"),
        name_list_as=profile.get("nameListAs"),
        name_full_title=profile.get("nameFullTitle"),
        name_address_as=profile.get("nameAddressAs"),
        gender=profile.get("gender"),
        party=party.get("name"),
        party_id=party.get("id"),
        party_abbreviation=party.get("abbreviation"),
        party_colour=party.get("backgroundColour"),
        house=house,
        membership_from=membership.get("membershipFrom"),
        membership_from_id=membership.get("membershipFromId"),
        # For Lords "membershipFrom" is a type ("Life peer"), not a constituency.
        constituency=membership.get("membershipFrom") if house == "Commons" else None,
        membership_start_date=start,
        membership_end_date=end,
        membership_end_reason=membership.get("membershipEndReason"),
        is_active=False if ended else status.get("statusIsActive"),
        status=None if ended else status.get("statusDescription"),
        thumbnail_url=profile.get("thumbnailUrl"),
        synopsis=(clean_text(synopsis) or None) if isinstance(synopsis, str) else None,
        last_election_result=election.get("result"),
        last_election_date=_date(election.get("electionDate")),
        last_election_majority=election.get("majority"),
        last_election_turnout=election.get("turnout"),
        last_election_electorate=election.get("electorate"),
        raw={"profile": profile, **raw},
    )
    row["member_since"] = min(starts) if starts else row["membership_start_date"]
    return row
