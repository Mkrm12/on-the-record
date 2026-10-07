"""The integrity gate is itself tested: every check runs, finds nothing where there is nothing, and FIRES on a planted
violation.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import psycopg
import pytest

from app import safety

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("audit_invariants", REPO / "scripts" / "audit_invariants.py")
AUDIT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AUDIT)
CHECKS = {code: (what, sql) for code, what, sql in AUDIT.CHECKS}
DAY = date(2025, 11, 19)


def violations(conn, code: str) -> int:
    return conn.execute(f"SELECT count(*) FROM ({CHECKS[code][1]}) q").fetchone()[0]


@pytest.fixture
def db(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    # Everything the checks read, emptied: the fixture's list plus what a check reads that it does not cascade to.
    scratch.execute("TRUNCATE member_office, procedure_event, division_vote, bill_stage_sitting, bill_stage CASCADE")
    return scratch


def window(conn, wid="W1", start="2025-11-01", end="2025-11-30"):
    conn.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                 "VALUES (%s, 'test', %s, %s, 'dev_test') ON CONFLICT DO NOTHING", (wid, start, end))


def debate(conn, did="d1", *, day=DAY, house="Commons", heading="A debate", wid="W1"):
    window(conn, wid)
    conn.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                 "VALUES (%s, %s, %s, %s, %s, %s, 'test')",
                 (did, day, house, heading, f"{house} Chamber", wid))


def speech(conn, sid="s1", did="d1", *, seq=0, text="Words.", wid="W1", mnis_id=None):
    conn.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id, mnis_id) "
                 "VALUES (%s, %s, %s, 'Ms Test', %s, %s, %s)", (sid, did, seq, text, wid, mnis_id))


def member(conn, mnis_id=1, **cols):
    base = {"mnis_id": mnis_id, "name": "A Member", "house": "Commons", "party": "Party"} | cols
    names = ", ".join(base)
    conn.execute(f"INSERT INTO member_id_map ({names}) VALUES ({', '.join(['%s'] * len(base))})", tuple(base.values()))


def division(conn, vid=1, *, title="Health Bill: Report Stage", bill_link=None, bill_stage_id=None, **cols):
    base = {"id": vid, "source_id": vid, "house": "Commons", "division_date": DAY, "title": title,
            "is_deferred": False, "debate_link": "none", "bill_link": bill_link, "bill_stage_id": bill_stage_id} | cols
    names = ", ".join(base)
    conn.execute(f"INSERT INTO division ({names}) VALUES ({', '.join(['%s'] * len(base))})", tuple(base.values()))


def bill_stage(conn, stage_id=501, bill_id=4001):
    conn.execute("INSERT INTO bill (id, title) VALUES (%s, 'Health Act 2026') ON CONFLICT DO NOTHING", (bill_id,))
    conn.execute("INSERT INTO bill_stage (id, bill_id, stage_name, house, abbreviation) "
                 "VALUES (%s, %s, 'Report stage', 'Commons', 'RS')", (stage_id, bill_id))


# --- the catalogue itself -------------------------------------------------------------------------------------------


def test_every_check_code_is_unique():
    codes = [code for code, _, _ in AUDIT.CHECKS]
    assert len(codes) == len(set(codes))


def test_info_and_expected_name_real_checks():
    assert set(AUDIT.INFO) <= set(CHECKS) and set(AUDIT.EXPECTED) <= set(CHECKS)
    assert not set(AUDIT.INFO) & set(AUDIT.EXPECTED)


def test_most_checks_are_hard():
    """Softening is the easy way to a green gate; it is allowed only with a stated reason (the INFO dict's values).
    """
    assert all(reason.strip() for reason in AUDIT.INFO.values())
    assert len(CHECKS) - len(AUDIT.INFO) - len(AUDIT.EXPECTED) >= 0.8 * len(CHECKS)


@pytest.mark.parametrize("code", sorted(CHECKS))
def test_the_check_is_a_read_only_select(code):
    sql = CHECKS[code][1].lstrip().upper()
    assert sql.startswith(("SELECT", "WITH")), code
    for verb in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "TRUNCATE "):
        assert verb not in sql, (code, verb)


@pytest.mark.parametrize("code", sorted(CHECKS))
def test_the_check_runs_on_the_migrated_schema_and_finds_nothing_in_an_empty_database(db, code):
    """A typo'd column errors here, not in the middle of a ten-year load's gate."""
    assert violations(db, code) == 0


# --- planted violations: each check fires ------------------------------------------------------------------------


@pytest.mark.parametrize("anomalies, fires", [
    ([], 1),  # withheld with no reason: the gate objects
    (["aye_count_mismatch:official=300,listed=0,with_tellers=0"], 1),  # a reason, but not one that withholds a result
    (["double_majority_disagrees:whole=agreed,subset=negatived"], 0),
    (["double_majority_incomplete"], 0),
])  # fmt: skip
def test_D20_a_division_with_both_counts_and_no_result_must_say_why(db, anomalies, fires):
    """The loader withholds a result only for an EVEL double majority that disagrees or is incomplete."""
    division(db, ayes=300, noes=200, result=None, anomalies=anomalies)
    assert violations(db, "D20") == fires


def test_D20_leaves_a_division_without_counts_to_the_other_checks(db):
    division(db, ayes=None, noes=None, result=None, anomalies=[])
    assert violations(db, "D20") == 0


def test_D19_fires_on_a_stage_debate_no_sitting_supports(db):
    """A stage keeps a debate after its sittings were unlinked."""
    debate(db)
    bill_stage(db)
    db.execute("UPDATE bill_stage SET debate_id = 'd1' WHERE id = 501")
    assert violations(db, "D19") == 1
    db.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, house, debate_id, debate_link) "
               "VALUES (901, 501, 4001, %s, 'Commons', 'd1', 'title_exact')", (DAY,))
    assert violations(db, "D19") == 0


def test_D19_fires_on_a_linked_sitting_whose_stage_has_no_debate(db):
    debate(db)
    bill_stage(db)
    db.execute("INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, house, debate_id, debate_link) "
               "VALUES (901, 501, 4001, %s, 'Commons', 'd1', 'title_exact')", (DAY,))
    assert violations(db, "D19") == 1


def test_X23_fires_on_a_speech_with_no_order(db):
    debate(db)
    db.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id) "
               "VALUES ('s9', 'd1', NULL, 'Ms Test', 'Words.', 'W1')")
    assert violations(db, "X23") == 1


def test_link_all_clears_a_stage_debate_that_no_sitting_supports(db, scratch_dsn):
    """The fix behind D19, end to end: bills.link_all derives stage debates both ways."""
    from ingest import bills as B
    from ingest.runtime import run

    debate(db)
    bill_stage(db)
    db.execute("UPDATE bill_stage SET debate_id = 'd1' WHERE id = 501")

    async def go():
        conn = await psycopg.AsyncConnection.connect(scratch_dsn, autocommit=True)
        try:
            await B.link_all(conn)
        finally:
            await conn.close()

    run(go())
    assert db.execute("SELECT debate_id FROM bill_stage WHERE id = 501").fetchone()[0] is None
    assert violations(db, "D19") == 0


def test_R15_fires_on_a_linked_label_without_a_stage(db):
    division(db, bill_link="title_stage")
    assert violations(db, "R15") == 1


def test_R15_fires_on_an_unlinked_label_with_a_stage(db):
    bill_stage(db)
    division(db, bill_link="none", bill_stage_id=501)
    assert violations(db, "R15") == 1


def test_R15_fires_on_a_label_outside_the_vocabulary(db):
    bill_stage(db)
    division(db, bill_link="title_guess", bill_stage_id=501)
    assert violations(db, "R15") == 1


@pytest.mark.parametrize("label", ["title_stage", "title_date+numbered", "title_amendment+fuzzy", "debate", "debate_weak"])
def test_R15_accepts_every_linked_label_with_a_stage(db, label):
    bill_stage(db)
    division(db, bill_link=label, bill_stage_id=501)
    assert violations(db, "R15") == 0


@pytest.mark.parametrize("label", ["none", "ambiguous", "not_a_bill"])
def test_R15_accepts_every_unlinked_label_without_one(db, label):
    division(db, bill_link=label)
    assert violations(db, "R15") == 0


def test_X21_fires_on_an_ended_membership_marked_active(db):
    member(db, is_active=True, status="Current Member", membership_end_date=date(2024, 5, 30))
    assert violations(db, "X21") == 1


def test_X21_is_quiet_for_an_ended_membership_marked_ended(db):
    member(db, is_active=False, status=None, membership_end_date=date(2024, 5, 30))
    assert violations(db, "X21") == 0


def test_X22_fires_on_a_failed_profile_and_not_on_a_failed_optional_endpoint(db):
    member(db, 1, missing_endpoints=["profile"])
    member(db, 2, missing_endpoints=["interests"])
    assert violations(db, "X22") == 1


def test_C01_fires_on_a_debate_with_no_speech(db):
    debate(db)
    assert violations(db, "C01") == 1
    speech(db)
    assert violations(db, "C01") == 0


def test_C03_fires_on_a_division_with_no_votes(db):
    division(db, bill_link="not_a_bill")
    assert violations(db, "C03") == 1


def test_C04_fires_on_a_bill_with_no_stage(db):
    db.execute("INSERT INTO bill (id, title) VALUES (1, 'Lonely Bill')")
    assert violations(db, "C04") == 1


def test_C06_fires_on_a_speaker_with_no_party(db):
    member(db, 7, party=None)
    debate(db)
    speech(db, mnis_id=7)
    assert violations(db, "C06") == 1


def test_X17_fires_on_an_empty_division_title(db):
    division(db, title="   ", bill_link="not_a_bill")
    assert violations(db, "X17") == 1


def test_X18_is_now_the_schemas_job_an_empty_or_padded_bill_title_is_refused(db):
    """Migration 017: bill.title NOT NULL and trimmed. X18 stays as the second line."""
    for title in (" ", "", "Finance Act 2020 ", " Finance Act 2020"):
        with pytest.raises(psycopg.errors.CheckViolation):
            db.execute("INSERT INTO bill (id, title) VALUES (1, %s)", (title,))
    with pytest.raises(psycopg.errors.NotNullViolation):
        db.execute("INSERT INTO bill (id, title) VALUES (1, NULL)")
    assert violations(db, "X18") == 0


def test_the_schema_refuses_a_rule_made_event_without_its_key(db):
    """Migration 017: ON CONFLICT (speech_id, event_type, char_start) is only a key if none of them can be NULL — NULLs
    never collide, so a keyless rule event would be inserted again on every run.
    """
    debate(db)
    speech(db)
    for speech_id, char_start in (("s1", None), (None, 5)):
        with pytest.raises(psycopg.errors.CheckViolation):
            db.execute("INSERT INTO procedure_event (debate_id, speech_id, seq, event_type, raw_phrase, rule, char_start) "
                       "VALUES ('d1', %s, 0, 'petition_presented', 'x', 'a_rule', %s)", (speech_id, char_start))
    # a hand-made, debate-level event (rule IS NULL) may have neither
    db.execute("INSERT INTO procedure_event (debate_id, event_type, raw_phrase) VALUES ('d1', 'manual_note', 'x')")
    with pytest.raises(psycopg.errors.NotNullViolation):
        db.execute("INSERT INTO procedure_event (debate_id, raw_phrase) VALUES ('d1', 'no type')")


def test_the_schema_refuses_a_debate_with_no_corpus_version(db):
    window(db)
    with pytest.raises(psycopg.errors.NotNullViolation):
        db.execute("INSERT INTO debate (id, date, house, heading, section, window_id) "
                   "VALUES ('d8', '2025-11-19', 'Commons', 'x', 'x', 'W1')")


def test_X07_fires_on_a_tally_no_house_could_cast(db):
    division(db, bill_link="not_a_bill", ayes=950, noes=10, result="agreed")
    assert violations(db, "X07") == 1


def test_X01_fires_on_a_debate_outside_its_windows_dates(db):
    debate(db, day=date(2025, 12, 25))  # W1 is November
    assert violations(db, "X01") == 1


def test_D07_fires_on_a_speech_filed_under_another_window(db):
    debate(db)
    window(db, "W2", "2025-12-01", "2025-12-31")
    speech(db, wid="W2")
    assert violations(db, "D07") == 1


def test_D16_and_D17_fire_on_stale_window_counts(db):
    window(db)
    db.execute("UPDATE corpus_window SET speeches = 5, debates = 3 WHERE window_id = 'W1'")
    assert violations(db, "D16") == 1 and violations(db, "D17") == 1


def test_U01_fires_on_a_repeated_seq(db):
    debate(db)
    speech(db, "s1", seq=0)
    try:
        speech(db, "s2", seq=0)
    except psycopg.errors.UniqueViolation:
        pytest.skip("the schema already refuses a repeated seq; U01 is a second line")
    assert violations(db, "U01") == 1


def test_U07_fires_on_a_contribution_loaded_twice(db):
    debate(db)
    text = "The same long contribution, loaded twice by a paging bug. " * 5
    speech(db, "s1", seq=0, text=text)
    speech(db, "s2", seq=1, text=text)
    assert violations(db, "U07") == 1


@pytest.mark.parametrize("table, column, insert", [
    ("debate", "heading", lambda c: debate(c, heading="Bad � heading")),
    ("division", "title", lambda c: division(c, title="Children�s Bill", bill_link="not_a_bill")),
    ("bill", "title", lambda c: c.execute("INSERT INTO bill (id, title) VALUES (9, 'Mojibake � Act 2026')")),
])  # fmt: skip
def test_X20_fires_on_a_replacement_character_in_stored_text(db, table, column, insert):
    """U+FFFD is what a decoding failure upstream leaves behind: text that was mangled before it reached us."""
    insert(db)
    assert violations(db, "X20") == 1


# --- where the schema now forbids it --------------------------------------------------------------------------------


@pytest.mark.parametrize("cols", [
    {"result": "agreed", "ayes": 10, "noes": 300},
    {"result": "negatived", "ayes": 300, "noes": 10},
    {"ayes": -1, "noes": 10, "result": "negatived"},
])  # fmt: skip
def test_the_schema_refuses_a_result_its_tally_contradicts(db, cols):
    with pytest.raises(psycopg.errors.CheckViolation):
        division(db, bill_link="not_a_bill", **cols)


def test_the_schema_refuses_an_office_that_ends_before_it_starts(db):
    member(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("INSERT INTO member_office (mnis_id, office_key, office, start_date, end_date, source) "
                   "VALUES (1, 'k', 'Minister', '2025-01-02', '2025-01-01', 'mnis')")


def test_the_schema_refuses_a_speech_in_no_window(db):
    debate(db)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        speech(db, wid="NOPE")


def test_the_schema_refuses_an_unknown_house(db):
    window(db)
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                   "VALUES ('d9', '2025-11-19', 'Senate', 'x', 'x', 'W1', 'test')")


# --- I01: the vector index is the form frozen retrieval queries ------------------------------------------------


@pytest.mark.parametrize(("state", "fires"), [("frozen", 0), ("other", 1), ("missing", 1), ("invalid", 1)])
def test_I01_once_there_are_vectors_the_index_must_be_the_frozen_form_and_valid(db, state, fires):
    """The wrong form is not an error anywhere else: every search just becomes a sequential scan."""
    from retrieval import vector_index as V

    frozen = V.frozen_form()
    other = next(f for f in V.FORMS if f != frozen)
    debate(db)
    speech(db)
    db.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body, is_boilerplate, embedding) "
               "VALUES ('s1', 'd1', 0, 0, 'Words.', 'Words.', false, %s::vector)", (str([1.0] + [0.0] * 383),))
    db.execute(f"DROP INDEX IF EXISTS {V.INDEX_NAME}")
    try:
        if state in ("frozen", "invalid"):
            db.execute(V.index_sql(frozen))
        elif state == "other":
            db.execute(V.index_sql(other))
        if state == "invalid":
            try:
                db.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass", (V.INDEX_NAME,))
            except psycopg.errors.InsufficientPrivilege:
                pytest.skip("cannot mark an index invalid without superuser")
        assert violations(db, "I01") == fires
    finally:
        db.execute(f"DROP INDEX IF EXISTS {V.INDEX_NAME}")
        db.execute(V.index_sql("halfvec"))  # as migration 023 leaves the scratch database
