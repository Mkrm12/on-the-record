"""Retrieval finds a member by the name they spoke under, not only by the name they hold now."""

from __future__ import annotations

from datetime import date

import pytest

from app import safety
from retrieval.hybrid import _SPEAKERS_DATED_SQL, _SPEAKERS_SQL
from retrieval.speakers import SpeakerIndex

SERVED = (date(2010, 5, 6), None)


def index(*aliases):
    rows = [(1438, "Lord Beamish", "Beamish, L.", "The Lord Beamish", *SERVED)]
    rows += [(1438, alias, None, None, *SERVED) for alias in aliases]
    return SpeakerIndex.from_rows(rows)


def test_the_name_the_record_uses_finds_the_member():
    got = index("Mr Kevan Jones").resolve("What did Kevan Jones say about the Post Office Horizon scandal?")
    assert got.member_ids == (1438,)


def test_the_present_name_still_finds_them():
    assert index("Mr Kevan Jones").resolve("What did Lord Beamish say?").member_ids == (1438,)


def test_without_the_alias_the_defect_is_there():
    assert index().resolve("What did Kevan Jones say about the Post Office?").member_ids == ()


def test_an_alias_never_makes_a_bare_surname_resolve():
    """"Jones" is a hundred members; an alias adds full-name keys only."""
    assert index("Mr Kevan Jones").resolve("What did Mr Jones say?").member_ids == ()


def test_an_alias_keeps_the_members_service_dates():
    idx = index("Mr Kevan Jones")
    assert idx.serving(1438, date(2020, 1, 1)) and not idx.serving(1438, date(2001, 1, 1))


# --- the SQL: which printed names become aliases --------------------------------------------------------------------


@pytest.fixture
def record(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WA', 'test', '2021-01-01', '2024-12-31', 'dev_test')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                    "VALUES ('dA', '2022-03-01', 'Commons', 'A Debate', 'Chamber', 'WA', 'test')")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, name_list_as, name_full_title, house, "
                    "membership_start_date) VALUES (1438, 'Lord Beamish', 'Beamish, L.', 'The Lord Beamish', 'Lords', "
                    "'2025-01-01'), (5, 'Ann Lee', 'Lee, Ann', NULL, 'Commons', '2019-12-12')")
    speeches = [(1438, "Mr Kevan Jones")] * 2 + [(1438, "Mr Jones")] * 3 + [(1438, "Kevan Jones")] + \
               [(5, "Ann Lee")] * 2 + [(None, "Some Person")] * 2
    for seq, (mnis_id, person) in enumerate(speeches):
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_person, text, window_id, "
                        "mnis_id) VALUES (%s, 'dA', %s, %s, %s, 'Words.', 'WA', %s)",
                        (f"s{seq}", seq, person, person, mnis_id))
    return scratch


def test_a_full_name_printed_twice_is_an_alias_and_nothing_else_is(record):
    rows = record.execute(_SPEAKERS_SQL).fetchall()
    assert sorted(r[:2] for r in rows if r[0] == 1438) == [(1438, "Lord Beamish"), (1438, "Mr Kevan Jones")], \
        "not the terse 'Mr Jones', not 'Kevan Jones' printed once"
    assert all(r[0] is not None for r in rows), "an unattributed label is nobody's alias"


def test_the_dated_rows_give_an_alias_the_members_own_dates(record):
    rows = {r[1]: r for r in record.execute(_SPEAKERS_DATED_SQL).fetchall()}
    assert rows["Mr Kevan Jones"][0] == 1438 and rows["Mr Kevan Jones"][4:] == rows["Lord Beamish"][4:]
    assert rows["Mr Kevan Jones"][2:4] == (None, None), "no list_as: an alias makes no surname key"


def add_speeches(conn, start, pairs):
    for seq, (mnis_id, person) in enumerate(pairs, start=start):
        conn.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_person, text, window_id, mnis_id) "
                     "VALUES (%s, 'dA', %s, %s, %s, 'Words.', 'WA', %s)", (f"x{seq}", seq, person, person, mnis_id))


def test_a_label_the_source_attached_to_the_wrong_member_is_not_their_alias(record):
    """Measured on hansard_v2: speeches labelled "Baroness Blake of Leeds" carrying Baroness Smith of Malvern's id. The
    label names ANOTHER member; as an alias it would pull the wrong member's speeches into every search for her.
    """
    record.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (269, 'Baroness Smith of Malvern', "
                   "'Lords'), (4912, 'Baroness Blake of Leeds', 'Lords')")
    add_speeches(record, 100, [(269, "Baroness Blake of Leeds")] * 2)
    aliases = [r for r in record.execute(_SPEAKERS_SQL).fetchall() if r[1] == "Baroness Blake of Leeds"]
    assert aliases == [], "not an alias of 269, and 4912 needs none: it is her own name"


def test_a_name_printed_for_two_members_belongs_to_the_one_it_was_printed_for_most(record):
    record.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (71, 'Lord A', 'Lords'), "
                   "(72, 'Lord B', 'Lords')")
    add_speeches(record, 200, [(71, "Chris Sample")] * 2 + [(72, "Chris Sample")] * 5)
    owners = {r[0] for r in record.execute(_SPEAKERS_SQL).fetchall() if r[1] == "Chris Sample"}
    assert owners == {72}


def test_the_index_built_from_the_database_resolves_the_printed_name(record):
    idx = SpeakerIndex.from_rows(record.execute(_SPEAKERS_DATED_SQL).fetchall())
    assert idx.resolve("What did Kevan Jones say?").member_ids == (1438,)
    assert idx.resolve("What did Ann Lee say?").member_ids == (5,)
