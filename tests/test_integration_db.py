"""Invariants of the LIVE database after the Phase 1 pipeline has run."""

import psycopg
import pytest
from dbhelpers import require, scalar

from app.migrate import discover


# --- migrations --------------------------------------------------------------------------------


def test_the_applied_migrations_are_a_gapless_prefix_of_the_ones_on_disk(db):
    """The live database must never hold a migration this checkout does not have, and must never have skipped one in
    the middle.
    """
    files = discover()
    applied = {r[0] for r in db.execute("SELECT version FROM schema_migrations")}
    unknown = applied - set(files)
    assert not unknown, f"the database records migrations this checkout does not have: {sorted(unknown)}"
    assert [f for f in files if f in applied] == files[:len(applied)], \
        f"a migration in the middle was skipped: applied {sorted(applied)}, on disk {files}"


# Members ----------------------------------------------------------------------------


def test_every_speaker_and_voter_has_a_member_row(db):
    require(db, "SELECT count(*) FROM member_id_map", "members not ingested")
    missing_speakers = scalar(db, "SELECT count(DISTINCT s.mnis_id) FROM speech s LEFT JOIN member_id_map m USING (mnis_id) WHERE s.mnis_id IS NOT NULL AND m.mnis_id IS NULL")
    assert missing_speakers == 0
    missing_voters = scalar(db, "SELECT count(DISTINCT v.mnis_id) FROM division_vote v LEFT JOIN member_id_map m USING (mnis_id) WHERE m.mnis_id IS NULL")
    assert missing_voters == 0, "run scripts/ingest_members.py to fetch the new voters"


# The member endpoints that may be absent or fail without making the member unknown.
OPTIONAL_MEMBER_ENDPOINTS = ["biography", "contact", "election", "experience", "focus", "interests", "staff", "synopsis"]


def test_no_member_lacks_a_profile_and_failed_fetches_are_rare(db):
    """`missing_endpoints` lists fetches that FAILED. A 404 is absence, not failure: reading every file-less endpoint
    as missing once recorded 1,284 of 1,933 members as incomplete -- every peer and former MP, for whom the API has
    no LatestElectionResult.
    """
    require(db, "SELECT count(*) FROM member_id_map", "members not ingested")
    total = scalar(db, "SELECT count(*) FROM member_id_map")
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE 'profile' = ANY(missing_endpoints)") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map, unnest(missing_endpoints) e WHERE e <> ALL(%s)",
                  OPTIONAL_MEMBER_ENDPOINTS) == 0
    incomplete = scalar(db, "SELECT count(*) FROM member_id_map WHERE cardinality(missing_endpoints) > 0")
    assert incomplete <= max(5, total // 200), f"{incomplete} of {total} members have failed fetches"


def test_no_ended_membership_is_marked_active(db):
    """The API keeps "Current Member" on MPs who left at a dissolution; 222 were stored as sitting."""
    require(db, "SELECT count(*) FROM member_id_map", "members not ingested")
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE is_active AND membership_end_date <= current_date") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE status IS NOT NULL "
                      "AND membership_end_date <= current_date") == 0


def test_temporal_views_reflect_party_changes(db):
    if not scalar(db, "SELECT count(*) FROM member_id_map WHERE mnis_id = 4972"):
        pytest.skip("member 4972 not ingested")
    assert scalar(db, "SELECT member_party_on(4972, DATE '2023-01-01')") == "Crossbench"
    assert scalar(db, "SELECT member_party_on(4972, DATE '2025-01-01')") == "Labour"


def test_party_history_covers_every_member(db):
    require(db, "SELECT count(*) FROM member_id_map", "members not ingested")
    uncovered = scalar(db, "SELECT count(*) FROM member_id_map m WHERE NOT EXISTS (SELECT 1 FROM member_party_history h WHERE h.mnis_id = m.mnis_id)")
    assert uncovered <= 5, f"{uncovered} members have no party history"


def test_speech_context_is_one_row_per_speech(db):
    assert scalar(db, "SELECT count(*) FROM speech_context") == scalar(db, "SELECT count(*) FROM speech")


# Rules ---------------------------------------------------------------------------------


def test_boilerplate_flag_and_reason_agree(db):
    require(db, "SELECT count(*) FROM speech WHERE speaker_role IS NOT NULL", "rules not run")
    assert scalar(db, "SELECT count(*) FROM speech WHERE is_boilerplate <> (boilerplate_reason IS NOT NULL)") == 0


def test_every_speech_has_a_speaker_role(db):
    require(db, "SELECT count(*) FROM speech WHERE speaker_role IS NOT NULL", "rules not run")
    assert scalar(db, "SELECT count(*) FROM speech WHERE speaker_role IS NULL") == 0


def test_chunks_of_boilerplate_speeches_are_flagged(db):
    require(db, "SELECT count(*) FROM speech WHERE is_boilerplate", "rules not run")
    unflagged = scalar(db, """SELECT count(*) FROM chunk c JOIN speech s ON s.id = c.speech_id
                              WHERE s.is_boilerplate AND s.boilerplate_reason <> 'empty' AND NOT c.is_boilerplate""")
    assert unflagged == 0


def test_procedure_events_point_at_their_own_debate_and_have_a_rule(db):
    require(db, "SELECT count(*) FROM procedure_event", "no events")
    assert scalar(db, "SELECT count(*) FROM procedure_event e JOIN speech s ON s.id = e.speech_id WHERE s.debate_id <> e.debate_id") == 0
    assert scalar(db, "SELECT count(*) FROM procedure_event WHERE rule IS NULL OR speech_id IS NULL") == 0


def test_motion_events_carry_a_classified_motion_type_when_they_have_text(db):
    require(db, "SELECT count(*) FROM procedure_event", "no events")
    assert scalar(db, "SELECT count(*) FROM procedure_event WHERE event_type = 'motion_moved' AND description IS NOT NULL AND motion_type IS NULL") == 0


def test_every_debate_has_a_type(db):
    require(db, "SELECT count(*) FROM debate WHERE debate_type IS NOT NULL", "rules not run")
    assert scalar(db, "SELECT count(*) FROM debate WHERE debate_type IS NULL") == 0


# Divisions -------------------------------------------------------------------------------


def test_lords_and_commons_division_ids_are_namespaced(db):
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division WHERE house = 'Lords' AND id < 1000000") == 0
    assert scalar(db, "SELECT count(*) FROM division WHERE house = 'Commons' AND id >= 1000000") == 0


def test_votes_belong_to_divisions_and_have_a_known_lobby(db):
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division_vote v LEFT JOIN division d ON d.id = v.division_id WHERE d.id IS NULL") == 0
    assert scalar(db, "SELECT count(*) FROM division_vote WHERE lobby NOT IN ('aye', 'no', 'no_vote', 'conflict')") == 0


def test_conflicting_votes_record_both_listings(db):
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division_vote WHERE lobby = 'conflict' AND NOT (flags @> ARRAY['listed_aye', 'listed_no'])") == 0


def test_listed_counts_match_the_stored_votes(db):
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    wrong = scalar(db, """
        SELECT count(*) FROM division d WHERE
            d.ayes_listed <> (SELECT count(*) FROM division_vote v WHERE v.division_id = d.id AND v.lobby = 'aye') OR
            d.noes_listed <> (SELECT count(*) FROM division_vote v WHERE v.division_id = d.id AND v.lobby = 'no') OR
            d.no_vote_count <> (SELECT count(*) FROM division_vote v WHERE v.division_id = d.id AND v.lobby = 'no_vote')""")
    assert wrong == 0


def test_a_reconciled_division_really_matches_its_official_tally(db):
    """Reconciled means the members the lists name, as each House's figures count them, equal the figures. The old form
    of this test accepted a figure equal to the listed voters with or without the tellers, which is the rule the
    payloads and Hansard's printed figures disproved.
    """
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    unnamed = scalar(db, "SELECT count(*) FROM division WHERE ayes IS NOT NULL AND (ayes_named IS NULL OR noes_named IS NULL)")
    assert unnamed == 0, "every division with figures carries its named counts (migration 024)"
    liars = scalar(db, """
        SELECT count(*) FROM division d
         WHERE d.tally_reconciled IS DISTINCT FROM (d.ayes_named = d.ayes AND d.noes_named = d.noes)
           AND d.ayes IS NOT NULL AND d.noes IS NOT NULL""")
    assert liars == 0


def test_a_division_links_only_to_a_debate_of_its_own_house(db):
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division d JOIN debate b ON b.id = d.debate_id WHERE b.house <> d.house") == 0


def test_a_non_deferred_division_links_only_to_a_debate_on_the_same_day(db):
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division d JOIN debate b ON b.id = d.debate_id WHERE d.debate_link NOT LIKE 'title_deferred%' AND b.date <> d.division_date") == 0


# How a division reached its debate (ingest.load, divisions.link_debate, divisions.link_deferred)...
DIVISION_LINKED = ["hansard_division_list", "tally_event", "time_window",
                   "title_exact", "title_prefix", "title_deferred",
                   "title_exact+time", "title_prefix+time", "title_deferred+time"]
# ... and why it did not: no evidence, or Hansard names a section that never becomes a debate row (a deferred division's
# empty "Deferred Division" list, or a section outside the loaded windows).
DIVISION_UNLINKED = ["none", "hansard_link_outside_window"]


def test_every_division_says_how_it_was_linked(db):
    """A closed vocabulary, both ways: a linked label always has a debate and an unlinked one never does."""
    require(db, "SELECT count(*) FROM division", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division WHERE debate_link IS NULL") == 0
    assert scalar(db, "SELECT count(*) FROM division WHERE debate_link <> ALL(%s)", DIVISION_LINKED + DIVISION_UNLINKED) == 0
    assert scalar(db, "SELECT count(*) FROM division WHERE debate_link = ANY(%s) AND debate_id IS NULL", DIVISION_LINKED) == 0
    assert scalar(db, "SELECT count(*) FROM division WHERE debate_link = ANY(%s) AND debate_id IS NOT NULL",
                  DIVISION_UNLINKED) == 0


def test_store_updates_changed_votes_and_removes_stale_ones_then_rolls_back(scratch_dsn):
    """Exercise ingest.divisions.store against the real schema, always rolled back."""

    from ingest.divisions import parse_commons, store
    from ingest.runtime import run

    def payload(ayes, noes=()):
        def m(mid, party):
            return {"MemberId": mid, "Name": f"Test {mid}", "Party": party, "PartyAbbreviation": "T", "MemberFrom": "Seat"}
        return {
            "DivisionId": 99999991, "Date": "2026-09-01T12:00:00", "Number": 1, "IsDeferred": False, "Title": "Test division",
            "AyeCount": len(ayes), "NoCount": len(noes), "AyeTellers": None, "NoTellers": None,
            "Ayes": [m(i, p) for i, p in ayes], "Noes": [m(i, p) for i, p in noes], "NoVoteRecorded": [],
        }

    async def scenario():
        async with await psycopg.AsyncConnection.connect(scratch_dsn) as conn:
            async with conn.transaction():
                await store(conn, parse_commons(payload(ayes=[(9001, "Old"), (9002, "Old"), (9003, "Old")])))
                first = await (await conn.execute("SELECT count(*) FROM division_vote WHERE division_id = 99999991")).fetchone()
                # Re-fetch: 9003 is gone, 9002 switched lobby and party, 9004 is new.
                await store(conn, parse_commons(payload(ayes=[(9001, "New")], noes=[(9002, "Other"), (9004, "Other")])))
                rows = await (await conn.execute(
                    "SELECT mnis_id, lobby, party FROM division_vote WHERE division_id = 99999991 ORDER BY mnis_id")).fetchall()
                divisions = await (await conn.execute("SELECT count(*) FROM division WHERE id = 99999991")).fetchone()
                raise psycopg.Rollback()
        return first[0], rows, divisions[0]

    first, rows, divisions = run(scenario())
    assert first == 3
    assert rows == [(9001, "aye", "New"), (9002, "no", "Other"), (9004, "no", "Other")]  # updated, switched, stale 9003 removed
    assert divisions == 1
    with psycopg.connect(scratch_dsn) as check:
        left_behind = check.execute("SELECT count(*) FROM division WHERE id = 99999991").fetchone()[0]
    assert left_behind == 0, "the test must leave nothing behind"


def test_vote_party_is_as_at_the_vote_not_overwritten_by_the_current_party(db):
    # Votes store the party printed in the division record; members store today's.
    require(db, "SELECT count(*) FROM division_vote WHERE party IS NOT NULL", "divisions not ingested")
    assert scalar(db, "SELECT count(*) FROM division_vote WHERE party IS NULL AND lobby IN ('aye', 'no')") == 0
