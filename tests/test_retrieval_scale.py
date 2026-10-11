"""The scale fixes, and the one property they all have to share."""

from datetime import date

import pytest
from corpus import Corpus, vec

from retrieval.hybrid import SearchConfig, SearchFilters, _DENSE_HNSW, _FUSE_SQL, fuse_sql
from retrieval.offices import OfficeHolder, OfficeIndex, office_key
from retrieval.speakers import SpeakerIndex, common_df_threshold

W0_DOCS = 90_040  # the corpus every Phase 2 number was measured on
V2_DOCS = 1_190_000  # roughly what four more windows make it


def test_the_absolute_threshold_is_untouched_when_no_ratio_is_given():
    assert common_df_threshold(W0_DOCS, absolute=50, ratio=None) == 50
    assert common_df_threshold(V2_DOCS, absolute=50, ratio=None) == 50


def test_the_relative_threshold_reproduces_the_old_one_exactly_on_w0():
    """Behaviour-preserving has to mean exactly, not approximately: a threshold of 49 or 51 would change which surnames
    resolve, and the scale-tax number would then be measuring a code change as well as a corpus change.
    """
    ratio = 50 / W0_DOCS
    assert common_df_threshold(W0_DOCS, absolute=50, ratio=ratio) == 50


def test_the_relative_threshold_grows_with_the_corpus_and_the_absolute_one_does_not():
    """At 1.19M chunks almost every surname appears in 50 of them, so 'a word that is common in the corpus needs a
    title' stops guarding anything.
    """
    ratio = 50 / W0_DOCS
    assert common_df_threshold(V2_DOCS, absolute=50, ratio=ratio) == 661
    assert common_df_threshold(V2_DOCS, absolute=50, ratio=None) == 50


def test_the_threshold_never_falls_below_one():
    """A tiny corpus must not make every word common, which df >= 0 would."""
    assert common_df_threshold(10, absolute=50, ratio=1e-9) == 1


def _index_with_two_smiths():
    rows = [
        (1, "Alice Smith", "Smith, Alice", None, date(2015, 5, 8), date(2019, 11, 6)),
        (2, "Bob Smith", "Smith, Bob", None, date(2019, 12, 13), None),
    ]
    return SpeakerIndex.from_rows(rows)


def test_without_a_date_a_shared_surname_stays_ambiguous():
    resolution = _index_with_two_smiths().resolve("What did Mr Smith say?")
    assert set(resolution.member_ids) == {1, 2} and resolution.ambiguous


def test_a_date_narrows_a_shared_surname_to_whoever_was_sitting():
    """'The Prime Minister' is four people across this corpus and Smith is two; a question filtered to a window has a
    date, and the date is the answer.
    """
    index = _index_with_two_smiths()
    early = index.resolve("What did Mr Smith say?", date(2017, 6, 1))
    late = index.resolve("What did Mr Smith say?", date(2024, 6, 1))
    assert early.member_ids == (1,) and not early.ambiguous
    assert late.member_ids == (2,) and not late.ambiguous


def test_a_date_nobody_was_sitting_on_keeps_every_candidate_rather_than_none():
    """Narrowing to nobody would silently turn a named-speaker question into an unnamed one. A date may remove
    candidates; it may not remove the last of them.
    """
    resolution = _index_with_two_smiths().resolve("What did Mr Smith say?", date(2019, 12, 1))
    assert set(resolution.member_ids) == {1, 2}


def test_a_member_with_no_recorded_dates_is_never_filtered_out():
    index = SpeakerIndex.from_rows([(3, "Carol Jones", "Jones, Carol", None, None, None)])
    assert index.resolve("Ms Jones asked", date(1990, 1, 1)).member_ids == (3,)


def test_an_index_built_without_date_columns_still_works():
    """The old four-column query must keep working: it is what the flag-off path runs."""
    index = SpeakerIndex.from_rows([(1, "Alice Smith", "Smith, Alice", None)])
    assert index.resolve("Alice Smith", date(2020, 1, 1)).member_ids == (1,)


def _pm_index():
    return OfficeIndex([
        OfficeHolder(1423, "The Prime Minister", date(2019, 7, 24), date(2022, 9, 6), "hansard_attribution", 500),
        OfficeHolder(4483, "The Prime Minister", date(2022, 10, 25), date(2024, 7, 5), "hansard_attribution", 400),
        OfficeHolder(4514, "The Prime Minister", date(2024, 7, 5), date(2026, 8, 31), "hansard_attribution", 900),
        OfficeHolder(1427, "Prime Minister", date(2026, 9, 1), None, "hansard_attribution", 120),
    ])


@pytest.mark.parametrize("raw", ["The Prime Minister", "the prime minister", " Prime Minister.", "PRIME MINISTER"])
def test_one_office_however_hansard_spells_it(raw):
    assert office_key(raw) == "prime minister"


def test_an_office_with_a_date_resolves_to_one_person():
    for day, expected in ((date(2020, 4, 1), 1423), (date(2023, 1, 1), 4483),
                          (date(2025, 11, 1), 4514), (date(2026, 9, 10), 1427)):
        resolution = _pm_index().resolve("the Prime Minister", day)
        assert resolution.member_ids == (expected,), day
        assert not resolution.ambiguous


def test_an_office_with_no_date_returns_every_holder_and_says_it_is_ambiguous():
    """The honest answer to 'what did the Prime Minister say about X' over a corpus holding four of them. Guessing the
    most recent would be wrong in three windows out of four.
    """
    resolution = _pm_index().resolve("the Prime Minister")
    assert resolution.ambiguous
    assert set(resolution.member_ids) == {1423, 4483, 4514, 1427}
    assert "no date was given" in resolution.reason


def test_an_office_nobody_held_on_that_date_resolves_to_nobody_not_to_the_nearest():
    resolution = _pm_index().resolve("the Prime Minister", date(2022, 9, 20))
    assert not resolution and resolution.reason == "office not held on that date"


def test_an_office_several_people_hold_at_once_is_ambiguous_even_with_a_date():
    index = OfficeIndex([OfficeHolder(1, "Minister of State", None, None, "mnis_biography", 0)])
    resolution = index.resolve("Minister of State", date(2025, 1, 1))
    assert resolution.ambiguous and "at once" in resolution.reason


def test_an_office_that_is_not_in_the_corpus_says_so():
    assert _pm_index().resolve("Lord High Executioner", date(2025, 1, 1)).reason == "office not in the corpus"


# --------------------------------------------------------------------------- First-class filters
# ---------------------------------------------------------------------------


def test_no_filter_produces_exactly_the_block_11_query():
    """The scale tax is the cost of a bigger haystack, and it is only that if the query is the same query. A stray
    space would not change the plan, but it would mean nobody could check this claim by eye again.
    """
    from app import sealed

    def block_11(mask: str) -> str:
        return _FUSE_SQL.format(dense_leg=_DENSE_HNSW.format(filter=mask, order="c.embedding <=> %(qvec)s::vector"),
                                filter=mask)

    with sealed.lifted_for_exam("the W5 exam: the unmasked query (test)"):
        assert fuse_sql(SearchFilters(), exact=False, dense_index="vector") == block_11("")
    mask = sealed.sql_exclusion("c.debate_date")[0]
    assert mask and fuse_sql(SearchFilters(), exact=False, dense_index="vector") == block_11(mask)
    halfvec = fuse_sql(SearchFilters(), exact=False, dense_index="halfvec")
    block_11 = block_11(mask)
    differing = [(a, b) for a, b in zip(block_11.splitlines(), halfvec.splitlines()) if a != b]
    assert differing == [("        ORDER BY c.embedding <=> %(qvec)s::vector",
                          "        ORDER BY c.embedding::halfvec(384) <=> %(qvec)s::halfvec(384)")]
    assert "window_id" not in fuse_sql(SearchFilters(), exact=False)


def test_each_filter_adds_its_own_predicate_and_nothing_else():
    """...and the sealed-window mask is always last, gone only in the W5 exam's own run."""
    from app import sealed

    mask = sealed.sql_exclusion("c.debate_date")[0]
    assert SearchFilters().predicate() == mask
    assert SearchFilters(windows=("W1",)).predicate() == " AND c.window_id = ANY(%(f_windows)s::text[])" + mask
    with sealed.lifted_for_exam("the W5 exam: the unmasked predicate (test)"):
        assert SearchFilters().predicate() == ""
    both = SearchFilters(houses=("Lords",), sections=("Grand Committee",)).predicate()
    assert both.count(" AND ") == 3 and "house" in both and "section" in both  # two filters, then the mask


def test_a_filter_reaches_every_leg_including_the_speaker_one():
    """A window filter that the dense leg honours and the keyword leg does not would return results from outside the
    window and look like a retrieval bug for weeks.
    """
    sql = fuse_sql(SearchFilters(windows=("W1",)), exact=False)
    assert sql.count("c.window_id = ANY(%(f_windows)s::text[])") == 3


def test_the_exact_scan_variant_materialises_the_filtered_set_first():
    """An HNSW scan under a selective filter gives up after 20,000 tuples and can miss the one sentence that matters
    (gotcha 39). MATERIALIZED is what stops the planner using the vector index for the ordering.
    """
    sql = fuse_sql(SearchFilters(windows=("W1",)), exact=True)
    assert "scoped AS MATERIALIZED" in sql
    assert sql.index("scoped AS MATERIALIZED") < sql.index("dense AS")


def test_filters_must_be_a_filters_object_not_a_dict():
    with pytest.raises(TypeError):
        SearchConfig(filters={"windows": ["W1"]})


def test_the_scale_fix_flags_all_default_to_the_old_behaviour():
    cfg = SearchConfig()
    assert cfg.common_df_ratio is None
    assert cfg.speaker_date_aware is False
    assert cfg.stored_lexicon is False
    assert cfg.exact_scan_max_rows == 0
    assert not cfg.filters


# --------------------------------------------------------------------------- Against the scratch database
# ---------------------------------------------------------------------------


def _tagged_corpus(scratch):
    """Two windows, two houses, so a filter has something to exclude."""
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role)"
        " VALUES ('W1','v2','2024-05-01','2024-07-31','dev_test'),"
        "        ('W2','v2','2025-10-01','2025-12-31','dev_test') ON CONFLICT DO NOTHING")
    corpus = Corpus(scratch)
    for window, debate_id, house, day in (("W1", "d1", "Commons", "2024-05-08"),
                                          ("W2", "d2", "Lords", "2025-10-14")):
        corpus.debate(debate_id, heading=f"Debate {debate_id}", date=day, house=house)
        corpus.speech(debate_id, f"s-{debate_id}", ["The first sentence about fishing quotas.",
                                                    "A second sentence about fishing quotas."],
                      mnis_id=4514, speaker_name="Ms Test", vectors=[vec((0, 1.0)), vec((0, 1.0))])
        scratch.execute("UPDATE debate SET window_id = %s WHERE id = %s", (window, debate_id))
        scratch.execute("UPDATE speech SET window_id = %s WHERE debate_id = %s", (window, debate_id))
    return corpus


def test_the_stored_lexicon_matches_the_ts_stat_one(scratch, scratch_dsn):
    """`stored=True` may make a refresh faster. It may not make it different."""
    from app.db import make_pool
    from ingest.runtime import run
    from retrieval.lexicon import build_lexicon, load_lexicon, store_lexicon

    _tagged_corpus(scratch)

    async def go():
        pool = make_pool(scratch_dsn, min_size=1, max_size=2)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                computed = await build_lexicon(conn)
                # Before it is built, the flag falls back rather than returning nothing.
                fallback = await load_lexicon(conn, stored=True)
                await store_lexicon(conn)
                stored = await load_lexicon(conn, stored=True)
                return computed, fallback, stored
        finally:
            await pool.close()

    computed, fallback, stored = run(go())
    assert computed.df and stored.df == computed.df
    assert (stored.n_docs, round(stored.avg_len, 6)) == (computed.n_docs, round(computed.avg_len, 6))
    assert fallback.df == computed.df, "an unbuilt table must fall back, never return an empty corpus"


def test_the_chunk_filter_columns_actually_filter(scratch):
    """The predicate is only worth anything if the denormalised columns are populated: a filter over columns nobody
    filled returns nothing, which looks exactly like a corpus with no matching debates.
    """
    _tagged_corpus(scratch)
    scratch.execute("UPDATE chunk c SET window_id = d.window_id, house = d.house, debate_date = d.date,"
                    " section = d.section, mnis_id = s.mnis_id"
                    " FROM debate d, speech s WHERE d.id = c.debate_id AND s.id = c.speech_id")
    rows = scratch.execute(
        "SELECT count(*) FROM chunk c WHERE NOT c.is_boilerplate"
        + SearchFilters(windows=("W1",)).predicate(),
        SearchFilters(windows=("W1",)).params()).fetchone()[0]
    total = scratch.execute("SELECT count(*) FROM chunk WHERE NOT is_boilerplate").fetchone()[0]
    assert 0 < rows < total
    lords = scratch.execute(
        "SELECT count(*) FROM chunk c WHERE NOT c.is_boilerplate"
        + SearchFilters(houses=("Lords",)).predicate(),
        SearchFilters(houses=("Lords",)).params()).fetchone()[0]
    assert lords == total - rows


def test_the_exact_scan_plan_is_never_used_without_a_filter(scratch, scratch_dsn):
    """`exact_scan_max_rows` is a filter policy. Applied to an unfiltered search it would turn every query into a scan
    of 1.2M vectors -- and it would do it silently, because the results would still be right.
    """
    from app.db import make_pool
    from ingest.runtime import run
    from retrieval.hybrid import HybridSearcher

    _tagged_corpus(scratch)
    scratch.execute("UPDATE chunk c SET window_id = d.window_id, house = d.house, debate_date = d.date,"
                    " section = d.section, mnis_id = s.mnis_id"
                    " FROM debate d, speech s WHERE d.id = c.debate_id AND s.id = c.speech_id")

    async def decide(filters):
        pool = make_pool(scratch_dsn, min_size=1, max_size=2)
        await pool.open(wait=True)
        try:
            searcher = HybridSearcher(pool, gpu=None)
            cfg = SearchConfig(filters=filters, exact_scan_max_rows=1_000_000)
            diag = {"queries": 0}
            async with pool.connection() as conn:
                return await searcher._scope_is_small(conn, cfg, {**filters.params()}, diag), diag
        finally:
            await pool.close()

    unfiltered, diag = run(decide(SearchFilters()))
    assert unfiltered is False and "scope_rows" not in diag, "no filter means no probe and no exact scan"
    filtered, diag = run(decide(SearchFilters(windows=("W1",))))
    assert filtered is True and diag["scope_rows"]
