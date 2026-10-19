"""The procedural eras across ten years, as a versioned table (`config/eras.toml`) the routes and the gate read."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from procedure import eras as E


def test_the_table_loads_and_every_era_is_well_formed():
    table = E.load()
    ids = [e.id for e in table]
    assert len(ids) == len(set(ids)), "era ids are unique"
    for e in table:
        assert e.house in E.HOUSES and e.status in E.STATUSES, e
        assert e.end is None or e.start <= e.end, e
        assert e.source and e.name and e.payload, f"{e.id}: every era says where it comes from and how it shows"
        if e.status == "attested":
            assert e.seen_first and e.seen_last and e.covers(e.seen_first) and e.covers(e.seen_last), \
                f"{e.id}: what the record shows lies inside the era it attests"


@pytest.mark.parametrize("era, day, house, inside", [
    ("evel", date(2020, 3, 4), "Commons", True),
    ("evel", date(2020, 3, 4), "Lords", False),  # a Commons procedure
    ("evel", date(2021, 7, 14), "Commons", False),  # repealed 13 July 2021
    ("commons_remote_divisions", date(2020, 5, 12), "Commons", True),
    ("commons_remote_divisions", date(2020, 6, 2), "Commons", False),  # physical divisions again from 2 June 2020
    ("commons_proxy_voting", date(2019, 1, 27), "Commons", False),
    ("commons_proxy_voting", date(2026, 9, 11), "Commons", True),  # open-ended
    ("lords_remote_divisions", date(2020, 5, 20), "Lords", False),
])
def test_an_era_covers_its_house_and_its_dates_only(era, day, house, inside):
    assert E.by_id(era).covers(day, house) is inside


def test_the_eras_on_a_day_are_listed_for_its_house():
    on = {e.id for e in E.on(date(2020, 5, 13), "Commons")}
    assert {"evel", "commons_remote_divisions", "commons_proxy_voting"} <= on
    assert "commons_remote_divisions" not in {e.id for e in E.on(date(2020, 5, 13), "Lords")}


# --- the property checks the gate runs (E01 to E04), on synthetic rows
# ---------------------------------------------------


def _row(**kw):
    base = {"id": 1, "house": "Commons", "division_date": date(2025, 11, 19), "is_remote": False, "is_deferred": False,
            "double_majority_ayes": None, "double_majority_noes": None, "proxy_votes": 0}
    return {**base, **kw}


@pytest.mark.parametrize("row, check", [
    (_row(is_remote=True, division_date=date(2024, 5, 13)), "E01"),  # remote outside the remote era
    (_row(is_remote=True, house="Lords", division_date=date(2020, 5, 20)), "E01"),
    (_row(double_majority_ayes=200, division_date=date(2024, 5, 13)), "E02"),  # a double count after EVEL
    (_row(double_majority_ayes=200, house="Lords", division_date=date(2020, 3, 4)), "E02"),
    (_row(proxy_votes=3, division_date=date(2018, 6, 1)), "E03"),  # proxies before the pilot
    (_row(proxy_votes=3, house="Lords"), "E03"),  # the Lords has no proxy voting
    (_row(is_deferred=True, house="Lords"), "E04"),  # a Commons procedure only
])
def test_a_flag_outside_every_era_that_allows_it_is_a_violation(row, check):
    assert E.violations([row])[check] == [[row["id"], row["house"], row["division_date"].isoformat()]]


def test_flags_inside_their_eras_are_not():
    rows = [_row(is_remote=True, division_date=date(2020, 5, 13)),
            _row(double_majority_ayes=200, double_majority_noes=100, division_date=date(2020, 3, 4)),
            _row(proxy_votes=12, division_date=date(2026, 9, 11)),
            _row(is_deferred=True)]
    assert all(not v for v in E.violations(rows).values())


# --- what the reader is told
# ------------------------------------------------------------------------------------------------


def test_a_remote_division_says_how_members_voted():
    notes = E.division_notes(_row(is_remote=True, division_date=date(2020, 5, 13)))
    assert any("remote" in n.lower() for n in notes)


def test_a_deferred_division_says_when_the_question_was_put_and_when_the_ballot_was():
    notes = E.division_notes(_row(is_deferred=True, division_date=date(2025, 11, 19)), put_on=date(2025, 11, 11))
    assert any("deferred" in n.lower() and "11 November 2025" in n and "19 November 2025" in n for n in notes)


def test_an_ordinary_division_carries_no_era_note():
    assert E.division_notes(_row()) == []


def test_a_double_majority_division_states_both_counts():
    notes = E.division_notes(_row(double_majority_ayes=250, double_majority_noes=180, evel_country="England",
                                  division_date=date(2020, 3, 4)))
    assert any("double-majority" in n and "250 to 180" in n for n in notes)


# --- sittings past midnight
# -------------------------------------------------------------------------------------------------


def test_a_division_held_after_midnight_belongs_to_the_sitting_it_was_part_of():
    """The Lords divided at 00:40 on 19 March 2026 in the sitting of 18 March: both days name it."""
    held = datetime(2026, 3, 19, 0, 40, 55)
    assert E.calendar_days(date(2026, 3, 18), held) == {date(2026, 3, 18), date(2026, 3, 19)}
    assert E.calendar_days(date(2026, 3, 18), datetime(2026, 3, 18, 22, 0)) == {date(2026, 3, 18)}
    assert E.calendar_days(date(2026, 3, 18), None) == {date(2026, 3, 18)}
    # a deferred ballot's midnight timestamp is a date, not a time past midnight
    assert E.calendar_days(date(2025, 11, 19), datetime(2025, 11, 19, 0, 0)) == {date(2025, 11, 19)}


def test_a_division_held_after_midnight_says_which_sitting_hansard_dates_it_to():
    """Listed under the 19th, the 00:40 division read as one of that day's sitting; the fact itself now says where it
    belongs, so every answer that shows it carries the note.
    """
    late = _row(house="Lords", division_date=date(2026, 3, 18), held_at=datetime(2026, 3, 19, 0, 40, 55))
    notes = E.division_notes(late)
    assert any("after midnight" in n and "00:40 on 19 March 2026" in n and "sitting of 18 March 2026" in n
               for n in notes), notes
    assert E.division_notes(_row(division_date=date(2026, 3, 18), held_at=datetime(2026, 3, 18, 22, 0))) == []
    deferred = _row(division_date=date(2025, 11, 19), held_at=datetime(2025, 11, 19, 0, 0))
    assert E.division_notes(deferred) == [], "a deferred ballot's midnight stamp is a date, not a time past midnight"
    later = _row(division_date=date(2026, 3, 18), held_at=datetime(2026, 3, 20, 0, 40))
    assert E.division_notes(later) == [], "only the next day is the same sitting's (calendar_days); later is not"


def test_a_division_is_placed_against_the_day_asked_only_when_its_sitting_differs():
    assert E.other_sitting(date(2026, 3, 19), date(2026, 3, 19)) == ""
    assert E.other_sitting("2026-03-18", date(2026, 3, 19)) == "in the sitting of 18 March 2026"
    assert E.other_sitting(date(2026, 3, 18), "2026-03-19") == "in the sitting of 18 March 2026"
    assert E.other_sitting(date(2026, 3, 18), None) == ""
    assert E.other_sitting(None, date(2026, 3, 19)) == ""


# --- the division facts carry the notes (result AND tally)
# ---------------------------------------------------------------


@pytest.fixture
def divisions(scratch, scratch_dsn):
    from app import safety

    safety.require_test(scratch_dsn)
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WX', 'v2', '2020-03-01', '2026-12-31', 'dev')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                    "VALUES ('put1', '2025-11-11', 'Commons', 'Draft Order', 'Commons Chamber', 'WX', 'v2')")
    rows = ((1, "2020-05-13", "Remote One", True, False, None, None, None),
            (2, "2025-11-19", "Deferred One", False, True, "put1", None, None),
            (3, "2020-03-04", "English One", False, False, None, 250, 180))
    for i, day, title, remote, deferred, debate, dm_ayes, dm_noes in rows:
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, title, ayes, noes, result, "
                        "ayes_listed, noes_listed, is_remote, is_deferred, debate_id, double_majority_ayes, "
                        "double_majority_noes, evel_country) VALUES (%s, %s, 'Commons', 1, %s, %s, 300, 200, 'agreed', "
                        "300, 200, %s, %s, %s, %s, %s, CASE WHEN %s::int IS NULL THEN NULL ELSE 'England' END)",
                        (i, i, day, title, remote, deferred, debate, dm_ayes, dm_noes, dm_ayes))
    return scratch_dsn


def _fact(dsn, category, division_id):
    import psycopg

    from answer import facts as F
    from ingest.runtime import run

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await F.answer(conn, category, {"division_id": division_id})
        finally:
            await conn.close()
    return run(go())


@pytest.mark.parametrize("category", ["division_result", "division_tally"])
def test_both_division_facts_state_what_the_era_changed(divisions, category):
    assert "remote division" in _fact(divisions, category, 1).caveat
    deferred = _fact(divisions, category, 2).caveat
    assert "deferred division" in deferred and "11 November 2025" in deferred and "19 November 2025" in deferred
    assert "250 to 180" in _fact(divisions, category, 3).caveat, "V-02: a tally under EVEL states both counts too"


# --- the calendar in the resolver
# -----------------------------------------------------------------------------------------


@pytest.fixture
def late(scratch, scratch_dsn):
    from app import safety

    safety.require_test(scratch_dsn)
    rows = ((1_003_585, "2026-03-18", "2026-03-19 00:40:55", "Late Bill"),  # held after midnight
            (1_003_590, "2026-03-18", "2026-03-18 22:00:00", "Evening Bill"))  # held before it
    for i, day, held, title in rows:
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, held_at, title, ayes, noes, "
                        "result, ayes_listed, noes_listed) VALUES (%s, %s, 'Lords', 1, %s, %s, %s, 100, 50, 'agreed', "
                        "100, 50)", (i, i - 1_000_000, day, held, title))
    scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, held_at, title, ayes, noes, "
                    "result, ayes_listed, noes_listed, is_deferred) VALUES (2189, 2189, 'Commons', 1, '2025-11-19', "
                    "'2025-11-19 00:00:00', 'Deferred Order', 300, 200, 'agreed', 300, 200, true)")
    return scratch_dsn


def _ask(dsn, question):
    import psycopg

    from answer import contract as K
    from answer import pipeline as PL
    from answer import router as R
    from ingest.runtime import run

    prov = K.Provenance(route="structured", corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76", recipe="r",
                        prompt_version="p", model_id="m")  # fmt: skip

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await PL.structured_answer(conn, question, R.route(question), prov)
        finally:
            await conn.close()
    return run(go())


@pytest.mark.parametrize("day", ["18 March 2026", "19 March 2026"])
def test_a_division_after_midnight_is_found_by_its_sitting_and_its_calendar_day(late, day):
    answer, notes = _ask(late, f"What was the result of the Lords division on Late Bill on {day}?")
    assert answer is not None and answer.structured_facts, notes
    assert answer.structured_facts[0].value["division_id"] == 1_003_585


def test_a_division_before_midnight_is_not_found_by_the_next_day(late):
    answer, notes = _ask(late, "What was the result of the Lords division on Evening Bill on 19 March 2026?")
    assert answer is None or not answer.structured_facts, notes


def test_a_deferred_ballots_midnight_stamp_does_not_move_it(late):
    answer, notes = _ask(late, "What was the result of the Commons division on Deferred Order on 20 November 2025?")
    assert answer is None or not answer.structured_facts, notes


# --- a calendar day that holds two sittings ---------------------------------------------------------------------


@pytest.fixture
def mixed(scratch, scratch_dsn):
    """The Lords on 19 March 2026: two divisions after midnight in the sitting of the 18th, then the 19th's own two.
    One late division shares its number with one of the 19th's (numbers restart each sitting).
    """
    from app import safety

    safety.require_test(scratch_dsn)
    rows = ((1_003_585, 10, "2026-03-18", "2026-03-19 00:40:55", "Crime Bill", "Amendment 429ZA", "negatived", 100, 150),
            (1_003_586, 1, "2026-03-18", "2026-03-19 00:51:50", "Crime Bill", "Amendment 430", "agreed", 120, 90),
            (1_003_588, 1, "2026-03-19", "2026-03-19 12:27:24", "Pension Bill", "Amendment 49", "agreed", 217, 107),
            (1_003_589, 2, "2026-03-19", "2026-03-19 13:17:02", "Pension Bill", "Amendment 52", "agreed", 217, 113))
    for i, number, day, held, title, motion, result, ayes, noes in rows:
        scratch.execute("INSERT INTO division (id, source_id, house, number, division_date, held_at, title, motion_text, "
                        "ayes, noes, result, ayes_listed, noes_listed) VALUES (%s, %s, 'Lords', %s, %s, %s, %s, %s, %s, "
                        "%s, %s, %s, %s)", (i, i - 1_000_000, number, day, held, title, motion, ayes, noes, result, ayes,
                                            noes))  # fmt: skip
    return scratch_dsn


def test_a_day_s_table_is_headed_by_the_day_asked_and_places_the_other_sitting_s_divisions(mixed):
    answer, notes = _ask(mixed, "What were the results of the Lords divisions on 19 March 2026?")
    assert answer is not None and len(answer.structured_facts) == 4, notes
    lead = answer.rendered.splitlines()[0]
    assert "19 March 2026" in lead and "2 of them in the sitting of 18 March 2026" in lead, lead
    rows = [line for line in answer.rendered.splitlines() if line.startswith("| ") and "Bill" in line]
    assert all("sitting of 18 March 2026" in r for r in rows if "Crime Bill" in r), rows
    assert not any("sitting of" in r for r in rows if "Pension Bill" in r), rows
    late = [f for f in answer.structured_facts if f.value["date"] == "2026-03-18"]
    assert len(late) == 2 and all("sitting of 18 March 2026" in f.caveat for f in late)


def test_the_other_sitting_s_own_day_lists_them_as_its_own(mixed):
    answer, notes = _ask(mixed, "What were the results of the Lords divisions on 18 March 2026?")
    assert answer is not None and len(answer.structured_facts) == 2, notes
    lines = answer.rendered.splitlines()
    assert "18 March 2026" in lines[0] and "sitting of" not in lines[0]
    assert not any("(sitting of" in line for line in lines if line.startswith("| ")), "the sitting's own, unlabelled"


def test_a_division_number_on_a_day_means_that_day_s_sitting_first(mixed):
    """Division numbers restart each sitting: "division 1 on 19 March" is the 19th's division 1, not the 18th's late
    one.
    """
    answer, notes = _ask(mixed, "What was the result of Lords division 1 on 19 March 2026?")
    assert answer is not None and [f.value["division_id"] for f in answer.structured_facts] == [1_003_588], notes


def test_a_late_division_asked_by_its_sitting_s_number_is_still_found(mixed):
    answer, notes = _ask(mixed, "What was the result of Lords division 10 on 19 March 2026?")
    assert answer is not None and [f.value["division_id"] for f in answer.structured_facts] == [1_003_585], notes
    assert "sitting of 18 March 2026" in answer.rendered


def test_a_passage_question_places_the_sitting_before_s_amendment_divisions(mixed, scratch):
    """The passage path lists the day's divisions on the bill; the 18th's two after midnight are placed, as in "Did the
    Lords pass the Crime and Policing Bill on 19 March 2026?" on the real record.
    """
    scratch.execute("INSERT INTO bill (id, title) VALUES (77, 'Crime Bill')")
    answer, notes = _ask(mixed, "Did the Lords pass the Crime Bill on 19 March 2026?")
    assert answer is not None and answer.facets.get("clarify"), notes
    assert answer.rendered.count("in the sitting of 18 March 2026") == 2, answer.rendered


# --- a count of a day's divisions counts divisions (QA pass 9, High 3)
# -------------------------------------------------


def test_how_many_times_a_house_divided_on_a_day_counts_its_divisions_and_places_the_late_ones(mixed):
    """"How many times did the Lords divide on 19 March 2026?" was answered "The Lords held 11 separate debates": the
    count chose its template by the shape of its parameters (a day meant debates), not by what the question counts.
    """
    answer, notes = _ask(mixed, "How many times did the Lords divide on 19 March 2026?")
    assert answer is not None and answer.structured_facts, notes
    fact = answer.structured_facts[0].value
    assert fact["divisions"] == 4 and "debates" not in fact
    assert "divided 4 times on 19 March 2026, 2 of them after midnight in the sitting of 18 March 2026" in answer.rendered


def test_a_count_of_divisions_on_the_late_sitting_s_own_day_is_its_own(mixed):
    answer, notes = _ask(mixed, "How many divisions were there in the Lords on 18 March 2026?")
    assert answer is not None and answer.structured_facts[0].value["divisions"] == 2, notes
    assert "sitting of" not in answer.rendered


def test_a_count_of_debates_in_a_month_counts_debates(mixed, scratch):
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WM', 'v2', '2026-03-01', '2026-03-31', 'dev')")
    for i, day in enumerate(("2026-03-18", "2026-03-19", "2026-03-19")):
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                        "VALUES (%s, %s, 'Lords', 'Business', 'Lords Chamber', 'WM', 'v2')", (f"dm{i}", day))
    answer, notes = _ask(mixed, "How many debates were there in the Lords in March 2026?")
    assert answer is not None and answer.structured_facts, notes
    fact = answer.structured_facts[0].value
    assert fact.get("debates") == 3 and "divisions" not in fact, fact


def test_a_day_the_record_holds_no_division_on_says_so_as_the_record_s(mixed):
    answer, notes = _ask(mixed, "How many divisions were there in the Lords on 20 March 2026?")
    assert answer is not None and answer.structured_facts[0].value["divisions"] == 0, notes
    assert answer.rendered == "The record holds no division in the Lords on 20 March 2026."


def test_the_debate_a_late_division_was_part_of_says_where_hansard_dates_it(mixed, scratch):
    """QA pass 9, Low 8: "Which debate was division 11 in the Lords on 19 March 2026 part of?" named the 18th's debate
    with no word of the division being held at 00:51 on the 19th.
    """
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('WL', 'v2', '2026-03-01', '2026-03-31', 'dev')")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                    "VALUES ('late1', '2026-03-18', 'Lords', 'Crime Bill', 'Lords Chamber', 'WL', 'v2')")
    scratch.execute("UPDATE division SET debate_id = 'late1' WHERE id = 1003585")
    from answer import facts as F
    from ingest.runtime import run
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(mixed)
        try:
            return await F.division_debate(conn, 1_003_585)
        finally:
            await conn.close()
    fact = run(go())
    assert fact.ok and "after midnight, at 00:40 on 19 March 2026, in the sitting of 18 March 2026" in fact.caveat


@pytest.mark.parametrize("question, subject", [
    ("How many times did the Lords divide on 19 March 2026?", "divisions"),
    ("How many divisions were there in the Commons on 3 March 2026?", "divisions"),
    ("How many times did the House vote on 3 March 2026?", "divisions"),
    ("How many separate debates did the Lords hold on 19 March 2026?", "debates"),
    # "votes" is the members' votes, a tally, not a count of divisions
    ("How many votes were cast in the Lords division on 19 March 2026?", None),
    ("How many speeches were there on 19 March 2026?", None),
])
def test_what_a_count_counts_is_read_from_the_question(question, subject):
    from answer import resolve as RS

    assert RS.count_subject(question) == subject
