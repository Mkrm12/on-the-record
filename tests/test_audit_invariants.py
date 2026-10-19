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


def _votes(conn, rows, did=1):
    for mnis_id, lobby, teller, flags in rows:
        conn.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, is_teller, flags) VALUES (%s, %s, %s, %s, %s)",
                     (did, mnis_id, lobby, teller, list(flags)))  # fmt: skip


def test_D21_is_quiet_when_the_named_counts_are_the_vote_rows_own(db):
    """Commons tellers apart, and a member named in both lobbies counted in each."""
    division(db, ayes=2, noes=2, ayes_named=2, noes_named=2, tally_reconciled=True)
    _votes(db, [(1, "aye", False, ()), (2, "no", False, ()), (3, "conflict", False, ("listed_aye", "listed_no")),
                (4, "aye", True, ()), (5, "no", True, ())])
    assert violations(db, "D21") == 0


def test_D21_fires_when_a_reload_changed_the_votes_and_not_the_counts(db):
    division(db, ayes=2, noes=1, ayes_named=2, noes_named=1, tally_reconciled=True)
    _votes(db, [(1, "aye", False, ()), (2, "aye", False, ()), (3, "no", False, ()), (6, "no", False, ())])
    assert violations(db, "D21") == 1


def test_D21_lists_a_division_whose_two_lobby_teller_cannot_be_placed(db):
    division(db, ayes=1, noes=1, ayes_named=1, noes_named=1, tally_reconciled=True)
    _votes(db, [(1, "conflict", True, ("listed_aye", "listed_no"))])
    assert violations(db, "D21") == 1


def test_D21_counts_the_lords_tellers_in(db):
    division(db, 1_000_001, source_id=1, house="Lords", ayes=2, noes=1, ayes_named=2, noes_named=1,
             tally_reconciled=True)
    _votes(db, [(11, "aye", False, ()), (12, "aye", True, ()), (13, "no", False, ())], did=1_000_001)
    assert violations(db, "D21") == 0


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


def test_a_reports_examples_never_come_from_the_sealed_window(db):
    """Reports/audit_invariants.json, which the mirror publishes, listed two W5 divisions (id, House, date and title)
    as R14's examples. The sealed window appears in a report as counts only.
    """
    division(db, 1, title="An open-window division", division_date=date(2026, 9, 9), debate_id=None)
    division(db, 2, title="A sealed-window division", division_date=date(2026, 5, 20), debate_id=None)
    debate(db, "d5", day=date(2026, 6, 2), wid="W1")
    speech(db, "s5", "d5")
    sql = CHECKS["R14"][1]
    examples, withheld = AUDIT.examples_outside_window(db, sql)
    assert [e[0] for e in examples] == ["1"] and withheld == 1
    assert violations(db, "R14") == 2  # the count still includes the window
    # a row that names a sealed debate or speech by id, with no date of its own, is withheld too
    rows, held = AUDIT.examples_outside_window(db, "SELECT 'd5' AS debate_id UNION ALL SELECT 's5' UNION ALL SELECT 'x'")
    assert rows == [["x"]] and held == 2
    # review pass 2: a row that names a sealed DIVISION by id alone (D21, the X checks) is withheld too
    rows, held = AUDIT.examples_outside_window(db, "SELECT 2 AS division_id UNION ALL SELECT 1")
    assert rows == [["1"]] and held == 1


# --- S05: the procedure layer's population invariants (P01-P04) -------------------------------------------------------
# Computed checks (`AUDIT.COMPUTED`): their rule lives in Python (`procedure.opener`, `procedure.threads`), and each is
# judged against an independent definition where one exists (the MR-08 oracle's Chair).

COMPUTED = {code: (what, fn) for code, what, fn in AUDIT.COMPUTED}


def computed(conn, code: str) -> list:
    return COMPUTED[code][1](conn)


def ordered_debate(conn, did, order, heading, *, day=DAY, section="Commons Chamber"):
    window(conn)
    conn.execute("INSERT INTO debate (id, date, house, heading, section, section_order, window_id, corpus_version) "
                 "VALUES (%s, %s, 'Commons', %s, %s, %s, 'W1', 'test')", (did, day, heading, section, order))


def said(conn, sid, did, seq, printed, role, mnis=None, text="Words."):
    conn.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, speaker_role, mnis_id, text, window_id) "
                 "VALUES (%s, %s, %s, %s, %s, %s, %s, 'W1')", (sid, did, seq, printed, role, mnis, text))


def test_the_computed_checks_have_codes_of_their_own_and_are_hard():
    assert not set(COMPUTED) & set(CHECKS) and {"P01", "P02", "P04", "P05", "P06"} <= set(COMPUTED) and "P03" in CHECKS
    assert not ({"P01", "P02", "P03", "P04", "P05", "P06"} & (set(AUDIT.INFO) | set(AUDIT.EXPECTED)))


@pytest.mark.parametrize("code", sorted(COMPUTED))
def test_every_computed_check_finds_nothing_in_an_empty_database(db, code):
    assert computed(db, code) == []


def test_P01_fires_when_the_opener_chosen_is_the_chair_by_the_oracles_own_definition(db, monkeypatch):
    """The opener rule and the oracle's Chair are separate code: if the rule ever let a Chair form through, the gate
    sees it by the oracle's definition.
    """
    ordered_debate(db, "d1", 1, "Climate Change")
    said(db, "s1", "d1", 0, "Madam Deput y Speaker (Caroline Nokes)", "member", 4048, "I call Luke Murphy.")
    said(db, "s2", "d1", 1, "Luke Murphy (Basingstoke) (Lab)", "member", 8, "I beg to move.")
    assert computed(db, "P01") == []
    monkeypatch.setattr(AUDIT.OP, "presiding", lambda printed, role: role in AUDIT.OP.PRESIDING_ROLES)  # a regressed rule
    assert [row[0] for row in computed(db, "P01")] == ["d1"]


def test_P02_fires_on_a_printing_of_the_chair_stored_without_a_presiding_role(db):
    ordered_debate(db, "d1", 1, "Coronavirus Bill")
    said(db, "s1", "d1", 0, "The Chairman of Ways and Means (Dame Eleanor Laing)", "office_holder", 1)
    said(db, "s2", "d1", 1, "The Chair of the Treasury Committee (Dame Meg Hillier)", "office_holder", 2)
    rows = computed(db, "P02")
    assert [r[0] for r in rows] == ["The Chairman of Ways and Means (Dame Eleanor Laing)"]


def test_P03_fires_on_a_record_with_no_place_in_the_day_or_a_place_shared(db):
    ordered_debate(db, "d1", 1, "A")
    ordered_debate(db, "d2", 1, "B")
    ordered_debate(db, "d3", None, "C")
    assert violations(db, "P03") == 3


def test_P04_is_quiet_on_what_threads_build_and_fires_on_two_threads_nothing_separates(db, monkeypatch):
    ordered_debate(db, "d1", 1, "Student Loans")
    ordered_debate(db, "d2", 2, "Student Loans")
    assert computed(db, "P04") == []
    monkeypatch.setattr(AUDIT.T, "build", lambda metas: [AUDIT.T.Thread((m.debate_id,), m.house, m.date, m.section,
                                                                        m.heading) for m in metas])  # a regressed build
    assert computed(db, "P04") == [["d1", "d2"]]


def test_D22_counts_a_division_whose_two_sources_part_and_only_then(db):
    division(db, 1, ayes=73, noes=256)
    division(db, 2, ayes=300, noes=200)
    db.execute("UPDATE division SET printed_ayes = 73, printed_noes = 286 WHERE id = 1")
    db.execute("UPDATE division SET printed_ayes = 300, printed_noes = 200 WHERE id = 2")
    assert violations(db, "D22") == 1 and "D22" in AUDIT.INFO


def test_P05_fires_when_the_opener_chosen_says_nothing_by_the_oracles_own_reading(db, monkeypatch):
    """QA pass 6: a member who "rose—" and was not called is not the opener. The rule (`procedure.opener.non_speech`)
    and the oracle's reading (`eval.oracles.openers.says_nothing`, an HTML parser) are separate code.
    """
    ordered_debate(db, "d1", 1, "Rail Fares")
    said(db, "s1", "d1", 0, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", 4048, "Order.")
    said(db, "s2", "d1", 1, "Sir Desmond Swayne (New Forest West) (Con)", "member", 7, "<em>rose\u2014</em>")
    said(db, "s3", "d1", 2, "The Minister (Ann Minister)", "minister", 8, "I beg to move.")
    assert computed(db, "P05") == []
    monkeypatch.setattr(AUDIT.OP, "non_speech", lambda text: False)  # a regressed rule
    assert [row[0] for row in computed(db, "P05")] == ["d1"]


def test_P06_fires_when_the_chairs_own_statement_is_given_a_member_as_opener(db, monkeypatch):
    ordered_debate(db, "d1", 1, "Speaker\u2019s Statement")
    said(db, "s1", "d1", 0, "Mr Speaker", "speaker", 467, "I wish to make a short statement.")
    said(db, "s2", "d1", 1, "Sir Desmond Swayne (New Forest West) (Con)", "member", 7, "On a point of order, Mr Speaker.")
    assert computed(db, "P06") == []
    monkeypatch.setattr(AUDIT.OP, "business_of_the_chair", lambda heading, kind: False)  # a regressed rule
    assert [row[0] for row in computed(db, "P06")] == ["d1"]


def test_the_computed_procedure_checks_read_the_record_once_per_gate_run(db, monkeypatch):
    """P01, P05, P06 and P07 judge the same record; the gate reads every debate's speeches once for all four."""
    ordered_debate(db, "d1", 1, "Rail Fares")
    said(db, "s1", "d1", 0, "The Minister (Ann Minister)", "minister", 8, "I beg to move.")
    reads = []
    real = AUDIT.OP.records_sync
    monkeypatch.setattr(AUDIT.OP, "records_sync", lambda conn: reads.append(1) or real(conn))
    with AUDIT.shared_records(db):
        for code in ("P01", "P05", "P06", "P07"):
            assert computed(db, code) == []
    assert len(reads) == 1
    assert computed(db, "P01") == [] and len(reads) == 2, "outside a gate run, each check reads the record afresh"


def test_P05_fires_too_where_the_rule_finds_no_opener_and_the_oracle_reads_a_speech(db, monkeypatch):
    """QA pass 7 (Med-2): P05 judged only an opener the rule named, so a rule calling a real speech "nothing" was
    invisible to the gate.
    """
    ordered_debate(db, "d1", 1, "Rail Fares")
    said(db, "s1", "d1", 0, "Madam Deputy Speaker (Caroline Nokes)", "deputy_speaker", 4048, "Order.")
    said(db, "s2", "d1", 1, "The Minister (Ann Minister)", "minister", 8, "I beg to move.")
    assert computed(db, "P05") == []
    monkeypatch.setattr(AUDIT.OP, "non_speech", lambda text: True)  # a regressed rule: everything says nothing
    assert [row[0] for row in computed(db, "P05")] == ["d1"]


def test_P07_fires_where_the_rule_and_the_oracle_read_any_contribution_differently(db, monkeypatch):
    """S05's real-data check: P05 judges openers only, so a contribution later in a record that the two readers part on
    was invisible to the gate. The readers part today on malformed markup the record does not hold ("</ em>", a tag
    never ended); a load that brings it fails here.
    """
    ordered_debate(db, "d1", 1, "Rail Fares")
    said(db, "s1", "d1", 0, "The Minister (Ann Minister)", "minister", 8, "I beg to move.")
    said(db, "s2", "d1", 1, "Richard Drax (South Dorset) (Con) [V]", "member", 7, "<em>[Inaudible.]</em>")
    said(db, "s3", "d1", 2, "Sir Desmond Swayne (New Forest West) (Con)", "member", 9, "<em>rose—</em>")
    assert computed(db, "P07") == []
    regressed = AUDIT.OP.non_speech
    monkeypatch.setattr(AUDIT.OP, "non_speech", lambda text: "Inaudible" in (text or "") or regressed(text))
    assert [row[2] for row in computed(db, "P07")] == ["s2"]


# --- E01 to E04: the procedural eras ----------------------------------------------------------------------


def test_the_era_checks_are_in_the_gate_and_hard():
    from procedure import eras as ERAS

    computed = {check for check, _, _ in AUDIT.COMPUTED}
    assert set(ERAS.CHECKS) <= computed
    assert not set(ERAS.CHECKS) & set(AUDIT.INFO), "a flag outside its era is a payload read wrongly: it fails the gate"


def test_the_era_checks_read_the_record_and_report_by_division(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (7001, 'A Member', 'Commons')")
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                    "ayes_listed, noes_listed, is_remote) VALUES (41, 41, 'Commons', 1, '2024-05-13', 'Late Remote', 1, "
                    "0, 'agreed', 1, 0, true), (42, 42, 'Commons', 2, '2020-05-13', 'Remote', 1, 0, 'agreed', 1, 0, true)")
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, proxy_name) "
                    "VALUES (42, 7001, 'aye', 'A Member', 'A Whip')")
    with psycopg.connect(scratch_dsn) as conn:
        checks = {check: fn for check, _, fn in AUDIT.COMPUTED}
        assert checks["E01"](conn) == [[41, "Commons", "2024-05-13"]]
        assert checks["E03"](conn) == [], "a proxy vote in May 2020 is inside the Commons' proxy era"
