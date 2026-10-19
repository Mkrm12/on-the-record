"""A member's record says what the API said — absence is not failure, and an ended membership has ended."""

from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import date, timedelta

import pytest

from ingest import corpus_cache as CC
from ingest.corpus_fetch import SCHEMA, member_path
from ingest.members import parse_member

SUFFIXES = list(CC.MEMBER_ENDPOINT_KEYS)  # "" is the profile


def profile(end=None, active=True, description="Current Member", house=1, reason=None):
    status = None if active is None else {"statusIsActive": active, "statusDescription": description,
                                          "statusStartDate": "2019-12-12T00:00:00"}
    return {"nameDisplayAs": "A Member", "latestParty": {"name": "Party", "id": 1},
            "latestHouseMembership": {"house": house, "membershipFrom": "Somewhere", "membershipStatus": status,
                                      "membershipStartDate": "2010-05-06T00:00:00",
                                      "membershipEndDate": end and f"{end}T00:00:00", "membershipEndReason": reason}}


# Is_active and status ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("end, reason", [("2024-05-30", "Dissolution"), ("2019-11-06", "Dissolution"),
                                         ("2017-05-03", "Dissolution"), ("2021-03-16", "Death"),
                                         ("2023-06-12", "Resigned")])
def test_an_ended_membership_is_not_active_whatever_the_stale_status_says(end, reason):
    row = parse_member(69, {"profile": profile(end=end, reason=reason)}, [])
    assert row["is_active"] is False and row["status"] is None
    assert row["membership_end_date"] == date.fromisoformat(end) and row["membership_end_reason"] == reason


@pytest.mark.parametrize("house", [1, 2])
def test_a_sitting_member_is_active_and_keeps_the_apis_status(house):
    row = parse_member(1, {"profile": profile(house=house)}, [])
    assert row["is_active"] is True and row["status"] == "Current Member"


def test_leave_of_absence_is_a_member_who_is_not_active():
    row = parse_member(2, {"profile": profile(active=False, description="Leave of Absence", house=2)}, [])
    assert row["is_active"] is False and row["status"] == "Leave of Absence"


def test_an_ended_membership_with_no_status_is_not_active_rather_than_unknown():
    row = parse_member(3, {"profile": profile(end="2022-01-01", active=None)}, [])
    assert row["is_active"] is False and row["status"] is None


def test_an_end_date_still_in_the_future_has_not_ended_yet():
    later = (date.today() + timedelta(days=30)).isoformat()
    row = parse_member(4, {"profile": profile(end=later)}, [])
    assert row["is_active"] is True


def test_no_status_and_no_end_is_unknown_not_invented():
    row = parse_member(5, {"profile": profile(active=None)}, [])
    assert row["is_active"] is None and row["status"] is None


# The cache's reading of absence ------------------------------------------------------------------------------


def _cache(tmp_path, mnis_id: int, present: set[str], recorded: str | None, manifest: bool = True):
    root = tmp_path / "cache with spaces"  # the repo lives under "On the record": the manifest URI must survive it
    for suffix in present:
        path = root / member_path(mnis_id, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = profile() if suffix == "" else {"value": {"endpoint": suffix}}
        with gzip.open(path, "wb") as fh:
            fh.write(json.dumps(payload).encode())
    root.mkdir(parents=True, exist_ok=True)
    if manifest:
        con = sqlite3.connect(root / "manifest.sqlite")
        con.executescript(SCHEMA)
        con.execute("INSERT INTO unit (source, window_id, key, status, last_error) VALUES ('members', '_shared', ?, "
                    "'done', ?)", (str(mnis_id), recorded))
        con.commit()
        con.close()
    return CC.CorpusCache(root)


def test_a_404_is_neither_a_payload_nor_missing(tmp_path):
    cache = _cache(tmp_path, 8, set(SUFFIXES) - {"LatestElectionResult"}, "LatestElectionResult:404")
    payloads, missing = cache.member(8)
    assert missing == [] and "election" not in payloads and "profile" in payloads


def test_a_failed_endpoint_is_missing(tmp_path):
    cache = _cache(tmp_path, 4021, set(SUFFIXES) - {"LatestElectionResult", "RegisteredInterests"},
                   "RegisteredInterests:failed;LatestElectionResult:404")
    assert cache.member(4021)[1] == ["interests"]


def test_without_a_manifest_every_file_less_endpoint_is_missing(tmp_path):
    """Nothing known to be absent: the conservative reading is a re-fetch, never a silently incomplete member."""
    cache = _cache(tmp_path, 9, set(SUFFIXES) - {"LatestElectionResult"}, None, manifest=False)
    assert cache.member(9)[1] == ["election"]


def test_an_endpoint_never_recorded_is_missing(tmp_path):
    cache = _cache(tmp_path, 10, set(SUFFIXES) - {"Staff"}, "LatestElectionResult:404")
    assert cache.member(10)[1] == ["staff"]


def test_a_profile_404_is_absence_and_parses_to_a_stub(tmp_path):
    cache = _cache(tmp_path, 11, set(), "profile:404")
    payloads, missing = cache.member(11)
    assert "profile" not in payloads and "profile" not in missing
    row = parse_member(11, payloads, missing, fallback_name="Hansard's Name")
    assert row["name"] == "Hansard's Name"


def test_another_members_404_does_not_leak(tmp_path):
    cache = _cache(tmp_path, 12, set(SUFFIXES) - {"LatestElectionResult"}, "LatestElectionResult:404")
    assert cache.member_absent(13) == frozenset() and cache.member_absent(12) == frozenset({"LatestElectionResult"})


@pytest.mark.parametrize("recorded, want", [
    ("LatestElectionResult:404", {"LatestElectionResult"}),
    ("RegisteredInterests:failed;LatestElectionResult:404", {"LatestElectionResult"}),
    ("profile:404", {""}),
    ("Staff:404;Focus:404;Contact:failed", {"Staff", "Focus"}),
    ("", set()),
    ("LatestElectionResult:4040", set()),
])  # fmt: skip
def test_the_manifest_tokens_are_read_exactly(tmp_path, recorded, want):
    cache = _cache(tmp_path, 14, set(), recorded or None)
    assert cache.member_absent(14) == frozenset(want)


# An unreadable file is damage, not absence ------------------------------------------------------------------


def test_a_file_not_in_the_cache_is_none(tmp_path):
    assert CC.CorpusCache(tmp_path)._maybe("members/1/profile.json.gz") is None


@pytest.mark.parametrize("content", [b"not gzip at all", gzip.compress(b"{not json"), gzip.compress(b"\xff\xfe")[:10]])
def test_an_unreadable_file_raises_instead_of_reading_as_none(tmp_path, content):
    path = tmp_path / "members" / "1" / "profile.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    with pytest.raises(CC.CacheError) as err:
        CC.CorpusCache(tmp_path)._maybe("members/1/profile.json.gz")
    assert not isinstance(err.value, CC.CacheMiss)


def test_a_cache_miss_is_still_a_cache_error_for_callers_that_catch_the_family(tmp_path):
    with pytest.raises(CC.CacheError):
        CC.CorpusCache(tmp_path)._read("members/1/profile.json.gz")


# A membership that ends before it begins is the source contradicting itself ---------------------------


def test_a_reversed_membership_keeps_its_end_and_drops_the_impossible_start():
    """The CHECK refuses end < start, and one such member aborted the whole members load. The END says the membership
    is over, so it is kept; "ended, start unknown" is less wrong than a former member shown as sitting.
    """
    p = profile(end="2019-11-06", reason="Dissolution")
    p["latestHouseMembership"]["membershipStartDate"] = "2021-05-06T00:00:00"
    row = parse_member(70, {"profile": p}, [])
    assert row["membership_start_date"] is None
    assert row["membership_end_date"] == date(2019, 11, 6)
    assert row["is_active"] is False
