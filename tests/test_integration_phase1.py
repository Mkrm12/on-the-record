"""Live-database invariants for the Phase 1 completion work: parent windows, column citations, dates of birth, Bills,
Hansard API metadata, and the views Phase 2 retrieval builds on. Skipped when the database or the relevant pipeline
stage is absent.
"""

import pytest
from dbhelpers import require, scalar

# --- parent windows / boilerplate -----------------------------------------------------------------------


def test_no_parent_window_contains_a_boilerplate_sibling(db):
    require(db, "SELECT count(*) FROM chunk WHERE is_boilerplate", "chunks not built")
    polluted = scalar(db, """
        SELECT count(*) FROM chunk c JOIN chunk b ON b.speech_id = c.speech_id AND b.is_boilerplate AND NOT c.is_boilerplate
        WHERE length(b.body) > 6 AND position(' ' || b.body || ' ' in ' ' || c.parent_body || ' ') > 0""")
    assert polluted == 0


def test_a_boilerplate_chunks_parent_is_itself(db):
    require(db, "SELECT count(*) FROM chunk WHERE is_boilerplate", "chunks not built")
    assert scalar(db, "SELECT count(*) FROM chunk WHERE is_boilerplate AND parent_body <> body") == 0


def test_chunk_time_flags_agree_with_the_rules_engine(db):
    """Both flag with the same function; if they disagree, the rules changed and chunks need --rebuild."""
    from ingest.rules import sentence_boilerplate_reason

    require(db, "SELECT count(*) FROM chunk", "chunks not built")
    rows = db.execute("SELECT c.is_boilerplate, c.body, s.speaker_name FROM chunk c JOIN speech s ON s.id = c.speech_id ORDER BY md5(c.id::text) LIMIT 4000").fetchall()
    wrong = [b for flag, b, name in rows if flag != (sentence_boilerplate_reason(b, name) is not None)]
    assert wrong == [], f"chunks are stale relative to the rules: run `python -m ingest.chunk --rebuild`  e.g. {wrong[:2]}"


def test_embedding_provenance_never_outlives_the_embedding(db):
    require(db, "SELECT count(*) FROM chunk", "chunks not built")
    assert scalar(db, "SELECT count(*) FROM chunk WHERE (embedding_model IS NOT NULL OR embedded_at IS NOT NULL) AND embedding IS NULL") == 0


# --- column citations --------------------------------------------------------------------------------------


def test_column_fields_are_consistent(db):
    require(db, "SELECT count(*) FROM chunk WHERE column_ref IS NOT NULL", "columns not built")
    assert scalar(db, "SELECT count(*) FROM chunk WHERE column_number IS DISTINCT FROM nullif(substring(column_ref from '^[0-9]+'), '')::int") == 0
    assert scalar(db, "SELECT count(*) FROM chunk WHERE (column_ref IS NULL) <> (column_basis IS NULL)") == 0
    assert scalar(db, "SELECT count(*) FROM chunk WHERE column_end_ref IS NOT NULL AND column_end_ref = column_ref") == 0
    assert scalar(db, "SELECT count(*) FROM chunk WHERE column_ref IS NOT NULL AND column_ref !~ '^[0-9]+(GC|WH)?$'") == 0


def test_grand_committee_and_westminster_hall_refs_keep_their_suffix(db):
    require(db, "SELECT count(*) FROM chunk WHERE column_ref IS NOT NULL", "columns not built")
    for section in ("Grand Committee", "Westminster Hall"):
        assert scalar(db, "SELECT count(*) FROM chunk c JOIN debate d ON d.id = c.debate_id WHERE d.section = %s AND c.column_ref ~ '^[0-9]+$'", section) == 0


def test_columns_do_not_run_backwards_within_a_debate(db):
    require(db, "SELECT count(*) FROM chunk WHERE column_ref IS NOT NULL", "columns not built")
    backwards = scalar(db, """
        SELECT count(*) FROM (
            SELECT column_number, lag(column_number) OVER (PARTITION BY debate_id ORDER BY seq) prev
            FROM chunk WHERE column_number IS NOT NULL) x WHERE column_number < prev""")
    # The API omits a few contributions so a run can jump forward; going back by more than a stray marker is a bug.
    assert backwards <= 5, f"{backwards} chunks step back a column"


def test_inferred_columns_only_precede_the_first_marker_of_their_debate(db):
    require(db, "SELECT count(*) FROM chunk WHERE column_basis = 'inferred'", "no inferred columns")
    bad = scalar(db, """
        SELECT count(*) FROM chunk i WHERE i.column_basis = 'inferred' AND EXISTS (
            SELECT 1 FROM chunk m WHERE m.debate_id = i.debate_id AND m.column_basis = 'marker' AND m.seq < i.seq)""")
    assert bad == 0


def test_marker_aware_text_matches_clean_text_on_real_speeches(db):
    """Offsets are only valid if the text that carries them is the text everything else uses."""
    from ingest.text import clean_text, clean_text_with_markers

    require(db, "SELECT count(*) FROM speech WHERE text LIKE '%column-number%'", "no marked speeches")
    rows = db.execute("SELECT text FROM speech WHERE text LIKE '%column-number%' ORDER BY md5(id) LIMIT 400").fetchall()
    assert [i for i, (t,) in enumerate(rows) if clean_text_with_markers(t)[0] != clean_text(t)] == []


# --- dates of birth ----------------------------------------------------------------------------------------


def test_full_dates_are_only_ever_day_precision(db):
    require(db, "SELECT count(*) FROM member_id_map WHERE dob_precision IS NOT NULL", "dob not backfilled")
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE date_of_birth IS NOT NULL AND dob_precision <> 'day'") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE dob_precision IN ('month', 'year') AND date_of_birth IS NOT NULL") == 0


def test_dob_bounds_are_consistent(db):
    require(db, "SELECT count(*) FROM member_id_map WHERE dob_precision IS NOT NULL", "dob not backfilled")
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE dob_precision IS NOT NULL AND (dob_earliest IS NULL OR dob_latest IS NULL OR birth_year IS NULL OR dob_earliest > dob_latest)") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE dob_precision = 'day' AND (date_of_birth IS DISTINCT FROM dob_earliest OR dob_earliest <> dob_latest)") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE dob_precision = 'year' AND (extract(month FROM dob_earliest) <> 1 OR extract(month FROM dob_latest) <> 12)") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE birth_year <> extract(year FROM dob_earliest)") == 0


def test_no_member_is_implausibly_young_or_old(db):
    require(db, "SELECT count(*) FROM member_id_map WHERE dob_precision IS NOT NULL", "dob not backfilled")
    assert scalar(db, "SELECT count(*) FROM member_age WHERE age_max < 18 OR age_min > 105") == 0


def test_age_function_respects_the_birthday(db):
    if scalar(db, "SELECT count(*) FROM member_id_map WHERE dob_precision = 'day'") == 0:
        pytest.skip("no full birth dates backfilled")
    mnis, dob = db.execute("SELECT mnis_id, date_of_birth FROM member_id_map WHERE dob_precision = 'day' AND date_of_birth < DATE '2000-01-01' AND extract(month FROM date_of_birth) <> 2 LIMIT 1").fetchone()
    sixtieth = dob.replace(year=dob.year + 60)
    assert (scalar(db, "SELECT member_age_on(%s, %s::date - 1)", mnis, sixtieth), scalar(db, "SELECT member_age_on(%s, %s::date)", mnis, sixtieth)) == (59, 60)


def test_a_member_without_a_full_date_has_no_exact_age_but_has_bounds(db):
    row = db.execute("SELECT mnis_id FROM member_id_map WHERE dob_precision IN ('month', 'year') LIMIT 1").fetchone()
    if row is None:
        pytest.skip("no partial-precision members")
    assert scalar(db, "SELECT member_age_on(%s, DATE '2026-01-01')", row[0]) is None
    lo, hi = db.execute("SELECT age_min, age_max FROM member_age WHERE mnis_id = %s", (row[0],)).fetchone()
    assert 0 <= hi - lo <= 1


def test_person_id_is_the_theyworkforyou_form(db):
    require(db, "SELECT count(*) FROM member_id_map WHERE person_id IS NOT NULL", "person ids not backfilled")
    assert scalar(db, r"SELECT count(*) FROM member_id_map WHERE person_id IS NOT NULL AND person_id !~ '^uk\.org\.publicwhip/person/[0-9]+$'") == 0
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE person_id IS NOT NULL AND person_id <> 'uk.org.publicwhip/person/' || twfy_id") == 0


def test_wikidata_provenance_is_recorded(db):
    require(db, "SELECT count(*) FROM member_id_map WHERE wikidata_id IS NOT NULL", "wikidata not backfilled")
    assert scalar(db, "SELECT count(*) FROM member_id_map WHERE wikidata_id IS NOT NULL AND (wikidata->>'source' IS NULL OR wikidata_fetched_at IS NULL)") == 0


def test_party_base_folds_co_op_into_labour(db):
    assert scalar(db, "SELECT party_base('Labour (Co-op)')") == "Labour"
    assert scalar(db, "SELECT party_base('Labour')") == "Labour"
    assert scalar(db, "SELECT party_base('Liberal Democrat')") == "Liberal Democrat"
    assert scalar(db, "SELECT party_base(NULL)") is None


# --- Hansard API metadata -----------------------------------------------------------------------------------


def test_almost_every_speech_has_its_api_metadata(db):
    require(db, "SELECT count(*) FROM speech WHERE item_id IS NOT NULL", "speech metadata not backfilled")
    assert scalar(db, "SELECT count(*) FROM speech WHERE item_id IS NULL") <= 10  # 2 were unserved by the API at first backfill
    assert scalar(db, "SELECT count(*) FROM speech WHERE item_id IS NOT NULL AND (meta ? 'HRSTag') IS NOT TRUE") == 0
    assert scalar(db, "SELECT count(*) FROM speech WHERE meta ? 'ContributionText' OR meta ? 'ContributionTextFull'") == 0


def test_debate_running_order_is_unique_within_a_section_of_a_sitting(db):
    require(db, "SELECT count(*) FROM debate WHERE section_order IS NOT NULL", "debate order not backfilled")
    assert scalar(db, "SELECT count(*) FROM (SELECT 1 FROM debate WHERE section_order IS NOT NULL GROUP BY date, section, section_order HAVING count(*) > 1) x") == 0


# --- bills ---------------------------------------------------------------------------------------------------


def test_bill_stage_dates_come_from_their_sittings(db):
    require(db, "SELECT count(*) FROM bill", "bills not ingested")
    assert scalar(db, "SELECT count(*) FROM bill_stage st WHERE st.first_sitting IS DISTINCT FROM st.date") == 0
    assert scalar(db, "SELECT count(*) FROM bill_stage st WHERE st.first_sitting IS DISTINCT FROM (SELECT min(t.date) FROM bill_stage_sitting t WHERE t.bill_stage_id = st.id)") == 0
    assert scalar(db, "SELECT count(*) FROM bill_stage st WHERE st.last_sitting IS DISTINCT FROM (SELECT max(t.date) FROM bill_stage_sitting t WHERE t.bill_stage_id = st.id)") == 0


def test_a_sitting_links_only_to_a_debate_of_that_day_and_house(db):
    require(db, "SELECT count(*) FROM bill_stage_sitting WHERE debate_id IS NOT NULL", "no linked sittings")
    assert scalar(db, "SELECT count(*) FROM bill_stage_sitting t JOIN debate d ON d.id = t.debate_id WHERE d.date <> t.date OR (t.house IS NOT NULL AND d.house <> t.house)") == 0


def test_a_division_links_to_a_stage_of_its_own_house_on_its_own_day(db):
    require(db, "SELECT count(*) FROM division WHERE bill_stage_id IS NOT NULL", "divisions not linked to bills")
    assert scalar(db, "SELECT count(*) FROM division d JOIN bill_stage st ON st.id = d.bill_stage_id WHERE st.house IS NOT NULL AND st.house <> d.house") == 0
    assert scalar(db, """SELECT count(*) FROM division d WHERE d.bill_stage_id IS NOT NULL AND d.bill_link NOT LIKE 'debate%' AND NOT EXISTS
                         (SELECT 1 FROM bill_stage_sitting t WHERE t.bill_stage_id = d.bill_stage_id AND t.date = d.division_date)""") == 0


def test_every_division_says_how_its_bill_was_found(db):
    require(db, "SELECT count(*) FROM division WHERE bill_link IS NOT NULL", "divisions not linked to bills")
    assert scalar(db, "SELECT count(*) FROM division WHERE bill_link IS NULL") == 0
    assert scalar(db, "SELECT count(*) FROM division WHERE bill_stage_id IS NULL AND bill_link IN ('title_stage', 'title_date', 'debate', 'debate_weak')") == 0
    assert scalar(db, "SELECT count(*) FROM division WHERE bill_stage_id IS NOT NULL AND bill_link IN ('none', 'ambiguous')") == 0


def test_weak_links_are_never_reported_as_strong(db):
    require(db, "SELECT count(*) FROM division WHERE bill_link IS NOT NULL", "divisions not linked to bills")
    assert scalar(db, "SELECT count(*) FROM division WHERE bill_link = 'debate' AND debate_link = 'time_window'") == 0


def test_bill_views_resolve(db):
    require(db, "SELECT count(*) FROM bill", "bills not ingested")
    assert scalar(db, "SELECT count(*) FROM bill_sponsor") > 0
    assert scalar(db, "SELECT count(*) FROM debate_bill WHERE debate_id NOT IN (SELECT id FROM debate)") == 0


# --- Phase 2 foundation ---------------------------------------------------------------------------------------


def test_chunk_context_is_one_row_per_chunk(db):
    require(db, "SELECT count(*) FROM chunk", "chunks not built")
    assert scalar(db, "SELECT count(*) FROM chunk_context") == scalar(db, "SELECT count(*) FROM chunk")


def test_chunk_context_carries_everything_retrieval_and_citation_need(db):
    require(db, "SELECT count(*) FROM chunk_context WHERE column_basis = 'marker' AND party_on_date IS NOT NULL", "no fully-populated chunk yet")
    url, party, day, house, heading, debate_type, column_ref = db.execute("""
        SELECT hansard_url, party_on_date, date, house, heading, debate_type, column_ref FROM chunk_context
        WHERE column_basis = 'marker' AND party_on_date IS NOT NULL LIMIT 1""").fetchone()
    assert url.startswith(f"https://hansard.parliament.uk/{house}/{day}/debates/") and "#contribution-" in url
    assert party and heading and debate_type and column_ref


def test_hansard_url_function_shapes_the_slug_and_anchor(db):
    url = scalar(db, "SELECT hansard_url('Commons', DATE '2026-09-11', 'ABC-123', 'Terminally Ill Adults (End of Life) Bill', 'SPEECH-1')")
    assert url == "https://hansard.parliament.uk/Commons/2026-09-11/debates/ABC-123/Terminally-Ill-Adults-End-of-Life-Bill#contribution-SPEECH-1"
    assert scalar(db, "SELECT hansard_url('Lords', DATE '2026-09-11', 'X', NULL)") == "https://hansard.parliament.uk/Lords/2026-09-11/debates/X/debate"
    assert scalar(db, "SELECT hansard_url('Lords', DATE '2026-09-11', 'X', '!!!')") == "https://hansard.parliament.uk/Lords/2026-09-11/debates/X/debate"
