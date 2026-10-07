"""The offline loader: what it must never do, and what it must do the same way twice."""

import gzip
import json
import socket
from datetime import date
from pathlib import Path

import pytest

from app import safety
from ingest.corpus_cache import CorpusCache
from ingest.corpus_fetch import (
    WindowSpec,
    bill_path,
    commons_detail_path,
    commons_index_path,
    contributions_path,
    corrections_path,
    hansard_divisions_path,
    member_path,
)
from ingest import load as L
from ingest.load import clean, shape_correction, shape_day

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "corpus_fetch.json").read_text(encoding="utf-8"))
WINDOW = WindowSpec(id="T1", start=date(2026, 9, 1), end=date(2026, 9, 2), role="dev_test", active=True)


# --------------------------------------------------------------------------- Building a cache the way the fetch tool
# writes one ---------------------------------------------------------------------------


def write(root: Path, rel: str, payload) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(json.dumps(payload).encode("utf-8"))


def contribution(cid: str, *, debate: str = "D1", day: str = "2026-09-01", order: int = 1,
                 member: int | None = 4514, **over) -> dict:
    template = FIXTURES["contributions_page"]["Results"][0]
    return {**template, "ContributionExtId": cid, "DebateSectionExtId": debate,
            "SittingDate": f"{day}T00:00:00", "OrderInDebateSection": order, "MemberId": member,
            "House": "Commons", "Section": "Commons Chamber", "DebateSection": "A Debate",
            "AttributedTo": "Ms Test Member (Testville) (Lab)",
            "ContributionTextFull": f"<p>Body of {cid}.</p>", "ItemId": 1000 + order,
            **over}


def day_page(root: Path, day: str, items: list[dict], *, listed: int | None = None,
             orders=("SittingDateAsc", "SittingDateDesc"), window: str = "T1") -> None:
    for order in orders:
        write(root, contributions_path(window, date.fromisoformat(day), order, 0),
              {**FIXTURES["contributions_page"], "Results": items,
               "TotalResultCount": len(items) if listed is None else listed})


def build_cache(tmp_path: Path, days: dict[str, list[dict]] | None = None, **extra) -> Path:
    root = tmp_path / "cache"
    for day, items in (days or {"2026-09-01": [contribution("C1"), contribution("C2", order=2)]}).items():
        day_page(root, day, items, listed=extra.get("listed"))
    return root


# --------------------------------------------------------------------------- Never the network
# ---------------------------------------------------------------------------


def test_the_loader_reads_a_cache_without_opening_a_socket(tmp_path, monkeypatch):
    """`hansard_v2` is reproducible only because its input is files on this disk. A load that reaches the API for one
    missing field would make the corpus depend on Parliament's uptime and on what the API happens to serve that day
    -- which, measured, is not what it served last week.
    """
    def forbidden(*a, **k):
        raise AssertionError("the loader made a network call")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    cache = CorpusCache(build_cache(tmp_path))
    items = cache.day_items(WINDOW, date(2026, 9, 1))
    debates, speeches, unplaceable = shape_day(items.items.values(), WINDOW, "v2")
    assert len(debates) == 1 and len(speeches) == 2 and not unplaceable


def test_a_missing_cache_file_is_an_absence_not_a_crash(tmp_path):
    cache = CorpusCache(tmp_path / "nothing-here")
    assert cache.days(WINDOW) == []
    assert cache.commons_divisions(WINDOW) == []
    assert cache.member(4514) == ({}, ["profile", "biography", "contact", "synopsis", "experience",
                                       "focus", "staff", "interests", "election"])


def test_a_cache_path_that_escapes_the_root_is_refused(tmp_path):
    from ingest.corpus_cache import CacheError
    cache = CorpusCache(tmp_path)
    with pytest.raises(CacheError):
        cache._read("../../etc/passwd")


# --------------------------------------------------------------------------- What the source disagrees with itself
# about ---------------------------------------------------------------------------


def test_a_contribution_served_under_two_debate_ids_keeps_the_first_and_records_the_clash(tmp_path):
    """The API served 26 General Committee contributions under both `4c458239-...` and `afccb16f-...`, two records of
    one sitting. The old ingest took its debate rows from the un-deduplicated pages and so created a debate with no
    speeches at all.
    """
    root = tmp_path / "cache"
    day_page(root, "2026-09-01", [contribution("C1", debate="FIRST")], orders=("SittingDateAsc",))
    write(root, contributions_path("T1", date(2026, 9, 1), "SittingDateDesc", 0),
          {**FIXTURES["contributions_page"], "TotalResultCount": 1,
           "Results": [contribution("C1", debate="SECOND")]})
    items = CorpusCache(root).day_items(WINDOW, date(2026, 9, 1))
    assert items.debate_conflicts == {"C1": ("FIRST", "SECOND")}
    debates, speeches, _ = shape_day(items.items.values(), WINDOW, "v2")
    assert [d.id for d in debates] == ["FIRST"], "the first sort order wins, as it did in the old ingest"
    assert [s.debate_id for s in speeches] == ["FIRST"]


def test_a_contribution_the_api_lists_but_will_not_serve_is_counted_not_fatal(tmp_path):
    """The API listed 9,021 contributions for W0 and served 8,985. That shortfall is the source's, and a loader that
    treated it as an error would never finish a window.
    """
    root = tmp_path / "cache"
    day_page(root, "2026-09-01", [contribution("C1")], listed=5)
    items = CorpusCache(root).day_items(WINDOW, date(2026, 9, 1))
    assert items.listed == 5 and len(items.items) == 1 and items.unserved == 4


def test_an_item_that_cannot_be_placed_is_returned_rather_than_dropped(tmp_path):
    """No contribution id, or no debate id, means the row cannot be located in the record. Dropping it silently is how
    a gap becomes invisible.
    """
    good = contribution("C1")
    no_debate = {**contribution("C2", order=2), "DebateSectionExtId": None}
    debates, speeches, unplaceable = shape_day([good, no_debate], WINDOW, "v2")
    assert [s.id for s in speeches] == ["C1"]
    assert unplaceable == ["C2"]
    assert len(debates) == 1


# --------------------------------------------------------------------------- Shaping
# ---------------------------------------------------------------------------


def test_the_stored_text_is_the_api_text_with_only_nul_removed():
    """Reproducing `hansard` byte for byte depends on this: the original ingest stripped NUL and nothing else, and HTML
    cleaning happens later, in the chunker. Anything cleverer here changes every text hash in the corpus.
    """
    raw = "<p>Hello\x00 <i>there</i></p>"
    assert clean(raw) == "<p>Hello <i>there</i></p>"
    _, speeches, _ = shape_day([contribution("C1", ContributionTextFull=raw)], WINDOW, "v2")
    assert speeches[0].text == "<p>Hello <i>there</i></p>"


def test_the_full_text_is_preferred_over_the_short_one():
    item = contribution("C1", ContributionTextFull="the full text", ContributionText="the short one")
    _, speeches, _ = shape_day([item], WINDOW, "v2")
    assert speeches[0].text == "the full text"


def test_the_window_tag_is_written_on_every_row():
    debates, speeches, _ = shape_day([contribution("C1")], WINDOW, "v2")
    assert debates[0].window_id == "T1" and debates[0].corpus_version == "v2"
    assert speeches[0].window_id == "T1"


def test_meta_keeps_every_api_field_except_the_two_long_texts():
    _, speeches, _ = shape_day([contribution("C1")], WINDOW, "v2")
    meta = json.loads(speeches[0].meta)
    assert "ContributionText" not in meta and "ContributionTextFull" not in meta
    assert meta["HansardSection"] and meta["ContributionExtId"] == "C1"


def test_a_correction_keeps_its_own_date_not_the_day_it_was_filed_under():
    item = contribution("X1", day="2026-09-02")
    row = shape_correction(item, date(2026, 9, 1), WINDOW)
    assert row["date"] == date(2026, 9, 2) and row["id"] == "X1" and row["window_id"] == "T1"
    assert "ContributionTextFull" not in row["raw"]


# --------------------------------------------------------------------------- Reading the other sources
# ---------------------------------------------------------------------------


def test_divisions_bills_and_members_are_read_from_the_paths_the_fetch_wrote(tmp_path):
    """The one place the reader and the fetcher can disagree is the filenames. They share the path helpers precisely so
    a rename cannot silently make a source look absent.
    """
    root = tmp_path / "cache"
    write(root, commons_index_path("T1", 0), FIXTURES["commons_division_search"])
    division_id = FIXTURES["commons_division_search"][0]["DivisionId"]
    write(root, commons_detail_path(division_id), FIXTURES["commons_division_detail"])
    write(root, hansard_divisions_path("T1", "Commons", 0), FIXTURES["hansard_divisions"])
    write(root, bill_path(4321, "detail"), FIXTURES["bill_detail"])
    write(root, bill_path(4321, "stages"), FIXTURES["bill_stages"])
    write(root, member_path(4514, ""), FIXTURES["member_profile"])
    write(root, member_path(4514, "Biography"), FIXTURES["member_biography"])
    write(root, corrections_path("T1", date(2026, 9, 1)),
          {**FIXTURES["contributions_page"], "Results": [contribution("X1")]})

    cache = CorpusCache(root)
    assert len(cache.commons_divisions(WINDOW)) == len(FIXTURES["commons_division_search"])
    assert cache.commons_detail(division_id)["DivisionId"] == division_id
    assert cache.hansard_divisions(WINDOW, "Commons")
    detail, stages = cache.bill(4321)
    assert detail and stages
    assert cache.member_ids() == [4514]
    payloads, missing = cache.member(4514)
    assert set(payloads) == {"profile", "biography"} and "election" in missing
    assert [cid for _, item in cache.corrections(WINDOW) for cid in [item["ContributionExtId"]]] == ["X1"]


def test_a_day_is_merged_across_both_sort_orders(tmp_path):
    """One order is not the record; the union is (gotcha 10)."""
    root = tmp_path / "cache"
    write(root, contributions_path("T1", date(2026, 9, 1), "SittingDateAsc", 0),
          {**FIXTURES["contributions_page"], "TotalResultCount": 3,
           "Results": [contribution("A"), contribution("B", order=2)]})
    write(root, contributions_path("T1", date(2026, 9, 1), "SittingDateDesc", 0),
          {**FIXTURES["contributions_page"], "TotalResultCount": 3,
           "Results": [contribution("B", order=2), contribution("C", order=3)]})
    items = CorpusCache(root).day_items(WINDOW, date(2026, 9, 1))
    assert sorted(items.items) == ["A", "B", "C"]


# --------------------------------------------------------------------------- Which database may be written to
# ---------------------------------------------------------------------------


def test_the_rollback_database_cannot_be_written_to_by_accident():
    """`hansard` holds the only corpus anything has ever been measured on, and unlike `hansard_v2` it cannot be rebuilt
    from the cache: the path that built it no longer exists. A stale PG_DSN from an earlier shell must not be able
    to overwrite it.
    """
    with pytest.raises(safety.WrongDatabase, match="rollback"):
        safety.require("postgresql://u:p@127.0.0.1:5432/hansard")
    assert safety.require("postgresql://u:p@127.0.0.1:5432/hansard_v2").endswith("hansard_v2")


def test_a_dsn_with_no_database_is_refused_rather_than_guessed():
    """libpq would fall back to the OS user's name. Nobody has ever wanted that."""
    with pytest.raises(safety.WrongDatabase, match="names no database"):
        safety.require("postgresql://u:p@127.0.0.1:5432/")


def test_a_test_may_only_use_a_throw_away_database():
    with pytest.raises(safety.WrongDatabase):
        safety.require_test("postgresql://u:p@127.0.0.1:5432/hansard_v2")
    assert safety.require_test("postgresql://u:p@127.0.0.1:5432/hansard_rt_test_9")


# --------------------------------------------------------------------------- Against a real (throw-away) database
# ---------------------------------------------------------------------------


def _load(dsn, root: Path, window: WindowSpec = WINDOW, **kw):
    """Run the real async loader against the scratch database and return the WindowLoad."""
    from app.db import make_pool
    from ingest.load import load_window
    from ingest.runtime import run

    async def go():
        pool = make_pool(dsn, min_size=1, max_size=2)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                return await load_window(conn, CorpusCache(root), window, "v2", **kw)
        finally:
            await pool.close()

    return run(go())


def _division_cache(tmp_path: Path) -> Path:
    """A cache holding one day, one Commons division and Hansard's own list of it."""
    root = tmp_path / "cache"
    day_page(root, "2026-09-01", [contribution("C1", debate="D1")])
    detail = {**FIXTURES["commons_division_detail"], "DivisionId": 99, "Number": 7,
              "Date": "2026-09-01T16:00:00"}
    write(root, commons_index_path("T1", 0), [{**FIXTURES["commons_division_search"][0],
                                               "DivisionId": 99, "Number": 7,
                                               "Date": "2026-09-01T16:00:00"}])
    write(root, commons_detail_path(99), detail)
    write(root, hansard_divisions_path("T1", "Commons", 0),
          {"TotalResultCount": 1,
           "Results": [{**FIXTURES["hansard_divisions"]["Results"][0], "Id": 1, "Number": "7",
                        "Date": "2026-09-01T16:00:00", "House": "Commons",
                        "DebateSectionExtId": "D1", "IsCommitteeDivision": False}]})
    return root


def test_loading_the_same_window_twice_changes_nothing(scratch, scratch_dsn, tmp_path):
    """The orchestrator is resumable, which means it re-runs steps that already finished. An upsert that rewrites every
    row would turn each re-run into a table rewrite of 1.2M rows and would churn the index for nothing (gotcha 5).
    """
    root = build_cache(tmp_path)
    first = _load(scratch_dsn, root, with_divisions=False, with_bills=False)
    assert first.speeches == 2 and first.rows_changed["speech"] == 2
    second = _load(scratch_dsn, root, with_divisions=False, with_bills=False)
    assert second.speeches == 2, "it still reads every row"
    assert second.rows_changed == {"debate": 0, "speech": 0}, "but it writes none of them"


def test_the_window_tag_reaches_the_database(scratch, scratch_dsn, tmp_path):
    _load(scratch_dsn, build_cache(tmp_path), with_divisions=False, with_bills=False)
    assert scratch.execute("SELECT window_id, corpus_version FROM debate").fetchall() == [("T1", "v2")]
    assert {r[0] for r in scratch.execute("SELECT window_id FROM speech").fetchall()} == {"T1"}
    row = scratch.execute("SELECT role, start_date, end_date FROM corpus_window").fetchone()
    assert row == ("dev_test", date(2026, 9, 1), date(2026, 9, 2))


def test_a_division_is_linked_by_hansards_own_statement_of_the_link(scratch, scratch_dsn, tmp_path):
    """`link_debate` infers the division-to-debate link from tallies, titles and times because the old ingest had
    nothing better. Hansard's division list states it outright (gotcha 52).
    """
    load = _load(scratch_dsn, _division_cache(tmp_path), with_bills=False)
    assert load.divisions == 1
    assert load.division_links == {"hansard_division_list": 1}
    assert scratch.execute("SELECT debate_id, debate_link FROM division").fetchone() == ("D1", "hansard_division_list")


def test_a_hansard_link_to_a_debate_outside_the_window_is_recorded_not_invented(scratch, scratch_dsn, tmp_path):
    """Hansard can name a debate the window does not contain -- a division taken on a day outside it. Storing that id
    would break the foreign key; guessing another debate would be worse.
    """
    root = _division_cache(tmp_path)
    write(root, hansard_divisions_path("T1", "Commons", 0),
          {"TotalResultCount": 1,
           "Results": [{**FIXTURES["hansard_divisions"]["Results"][0], "Id": 1, "Number": "7",
                        "Date": "2026-09-01T16:00:00", "House": "Commons",
                        "DebateSectionExtId": "SOMEWHERE-ELSE", "IsCommitteeDivision": False}]})
    load = _load(scratch_dsn, root, with_bills=False)
    assert load.division_links == {"hansard_link_outside_window": 1}
    assert scratch.execute("SELECT debate_id, debate_link FROM division").fetchone() == (None, "hansard_link_outside_window")


def test_the_unserved_count_survives_into_the_load_report(scratch, scratch_dsn, tmp_path):
    root = tmp_path / "cache"
    day_page(root, "2026-09-01", [contribution("C1")], listed=4)
    load = _load(scratch_dsn, root, with_divisions=False, with_bills=False)
    assert load.listed == 4 and load.served == 1
    assert load.unserved_by_day == {"2026-09-01": 3}
    assert load.as_dict()["unserved"] == 3


def test_corrections_are_stored_and_deduplicated(scratch, scratch_dsn, tmp_path):
    root = build_cache(tmp_path)
    write(root, corrections_path("T1", date(2026, 9, 1)),
          {**FIXTURES["contributions_page"], "Results": [contribution("X1"), contribution("X1")]})
    load = _load(scratch_dsn, root, with_divisions=False, with_bills=False)
    assert load.corrections == 1
    row = scratch.execute("SELECT id, window_id, corrected_speech_id FROM hansard_correction").fetchone()
    assert row == ("X1", "T1", None), "the link to the statement it corrects is Phase 3 work, left NULL"


def test_chunk_filter_columns_are_copied_down_from_debate_and_speech(scratch, scratch_dsn, tmp_path):
    """A filtered vector search needs its predicate on the indexed table: pgvector cannot push a join down into the
    HNSW scan, so a window or house filter that lives only on `debate` degenerates into a scan of the whole table at
    1.2M rows.
    """
    from app.db import make_pool
    from ingest.load import sync_chunk_filters
    from ingest.runtime import run

    _load(scratch_dsn, build_cache(tmp_path), with_divisions=False, with_bills=False)
    scratch.execute(
        "INSERT INTO chunk (speech_id, debate_id, seq, body, sent_idx, parent_body)"
        " SELECT id, debate_id, 1, 'a sentence', 0, 'a sentence' FROM speech")

    async def go():
        pool = make_pool(scratch_dsn, min_size=1, max_size=2)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                return await sync_chunk_filters(conn)
        finally:
            await pool.close()

    assert run(go()) == 2
    rows = scratch.execute("SELECT DISTINCT window_id, debate_date, house, section, mnis_id FROM chunk").fetchall()
    assert rows == [("T1", date(2026, 9, 1), "Commons", "Commons Chamber", 4514)]
    assert run(go()) == 0, "a second sync must not rewrite rows that already agree"


# --------------------------------------------------------------------------- Gaps the mutation suite found: each of
# these is a test that was missing ---------------------------------------------------------------------------


def test_the_listed_total_comes_from_the_first_page_not_the_last(tmp_path):
    """Page 0 is the one always re-read at fetch time, so its TotalResultCount is the day's current size. A later page
    can carry a stale total from before the day grew, and taking that one would quietly understate what the source
    claims to hold.
    """
    root = tmp_path / "cache"
    items = [contribution(f"C{i}", order=i) for i in range(3)]
    for order in ("SittingDateAsc", "SittingDateDesc"):
        write(root, contributions_path("T1", date(2026, 9, 1), order, 0),
              {**FIXTURES["contributions_page"], "Results": items[:2], "TotalResultCount": 9})
        write(root, contributions_path("T1", date(2026, 9, 1), order, 100),
              {**FIXTURES["contributions_page"], "Results": items[2:], "TotalResultCount": 3})
    items_read = CorpusCache(root).day_items(WINDOW, date(2026, 9, 1))
    assert items_read.listed == 9 and len(items_read.items) == 3
    assert items_read.unserved == 6


def test_serving_more_than_was_listed_is_zero_unserved_not_a_negative_number():
    """A negative shortfall would be added into a corpus-wide total and quietly cancel a real one somewhere else."""
    from ingest.corpus_cache import DayItems
    more = DayItems(day=date(2026, 9, 1), items={"a": {}, "b": {}}, listed=1)
    assert more.unserved == 0


@pytest.mark.parametrize("rel", ["../outside.json.gz", "a/../../outside.json.gz",
                                 "/etc/passwd", "a//../b"])
def test_a_path_that_climbs_out_of_the_cache_is_refused(tmp_path, rel):
    """The character class alone is not a guard: every file here ends `.json.gz`, so dots have to be allowed, and
    `../../secrets` matches it perfectly well.
    """
    from ingest.corpus_cache import CacheError
    (tmp_path / "root").mkdir()
    (tmp_path / "outside.json.gz").write_bytes(b"not yours")
    cache = CorpusCache(tmp_path / "root")
    with pytest.raises(CacheError, match="plain relative name"):
        cache._read(rel)


def test_a_cached_day_outside_the_window_is_not_loaded(tmp_path):
    """Windows must not overlap, and a day that leaked into the wrong window's directory would be counted twice across
    the corpus and confound every per-window comparison.
    """
    root = tmp_path / "cache"
    day_page(root, "2026-09-01", [contribution("C1")])
    day_page(root, "2026-09-30", [contribution("C2", day="2026-09-30")])
    assert CorpusCache(root).days(WINDOW) == [date(2026, 9, 1)]


def test_one_debate_is_written_once_however_many_days_are_flushed_together(scratch, scratch_dsn, tmp_path):
    """`ON CONFLICT DO UPDATE` refuses to touch the same row twice in one statement, so a batch holding two copies of a
    debate fails outright. It only takes one sitting spanning a batch boundary, and the run is hours in by then.
    """
    from app.db import make_pool
    from ingest.load import WindowLoad, _flush_text, shape_day
    from ingest.runtime import run

    debates, speeches, _ = shape_day([contribution("C1"), contribution("C2", order=2)], WINDOW, "v2")
    load = WindowLoad(window_id="T1")
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role)"
        " VALUES ('T1','v2','2026-09-01','2026-09-02','dev_test') ON CONFLICT DO NOTHING")

    async def go():
        pool = make_pool(scratch_dsn, min_size=1, max_size=2)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                await _flush_text(conn, debates + debates, speeches + speeches, load)
        finally:
            await pool.close()

    run(go())
    assert scratch.execute("SELECT count(*) FROM debate").fetchone()[0] == 1
    assert scratch.execute("SELECT count(*) FROM speech").fetchone()[0] == 2


def test_a_database_that_is_neither_the_target_nor_the_rollback_is_still_refused():
    """The protected-name branch catches `hansard`. A typo like `hansard_v3` reaches the second check, and without it a
    whole run would build the wrong database in silence.
    """
    with pytest.raises(safety.WrongDatabase, match="hansard_v3"):
        safety.require("postgresql://u:p@127.0.0.1:5432/hansard_v3", "hansard_v2")


@pytest.mark.parametrize("expected", ["hansard", "HANSARD"])
def test_a_protected_database_is_refused_even_when_it_is_the_expected_one(expected):
    """The loophole the audit found: `require(dsn, expected="hansard")` used to PASS, because the protected check only
    fired when `expected` was not itself protected. Four write paths forward a user-supplied `--expect` or
    `--database` as `expected` — including `load_corpus_v2.py`, the one that writes the most — so `--expect hansard`
    with PG_DSN pointing at the rollback would have loaded into it.
    """
    with pytest.raises(safety.WrongDatabase, match="rollback"):
        safety.require("postgresql://u:p@127.0.0.1:5432/hansard", expected.lower())


def test_the_protected_refusal_never_leaks_the_password():
    with pytest.raises(safety.WrongDatabase) as caught:
        safety.require("postgresql://u:s3cret@127.0.0.1:5432/hansard", "hansard")
    assert "s3cret" not in str(caught.value)


# --------------------------------------------------------------------------- Items the loader cannot place are
# quarantined, never crashed on or stored half-formed.


def test_an_item_with_no_sitting_date_is_quarantined_not_crashed_on():
    """The latent crash: with no SittingDate the debate row was never created but the speech was kept, so the insert
    violated speech -> debate and aborted the whole window. At ten years of items, "never happens" is a date.
    """
    good = contribution("C1")
    undated = contribution("C2", order=2, debate="D2", SittingDate=None)
    debates, speeches, unplaceable = shape_day([good, undated], WINDOW, "v2")
    assert {s.id for s in speeches} == {"C1"}
    assert "C2" in unplaceable
    assert all(s.debate_id in {d.id for d in debates} for s in speeches), "every kept speech has its debate"


def test_an_item_with_no_house_is_quarantined():
    """A debate with no House cannot be filtered, attributed or reconciled; storing one is a silent NULL."""
    debates, speeches, unplaceable = shape_day([contribution("C1", House=None)], WINDOW, "v2")
    assert speeches == [] and debates == [] and unplaceable == ["C1"]


def test_a_malformed_sitting_date_is_quarantined():
    _, speeches, unplaceable = shape_day([contribution("C1", SittingDate="not a date")], WINDOW, "v2")
    assert speeches == [] and unplaceable == ["C1"]


def test_a_contribution_with_no_text_is_stored_as_empty_not_null():
    """Hansard's "tabled by" markers carry no text; stored as '' they are an empty contribution, which is what they
    are. NULL would be a missing value, which they are not.
    """
    _, speeches, _ = shape_day([contribution("C1", ContributionTextFull=None, ContributionText=None)], WINDOW, "v2")
    assert speeches[0].text == ""


def test_a_later_item_of_an_already_placed_debate_needs_no_date_of_its_own():
    """Quarantine is per item: once a debate is placed, a later item in it is kept even if its own date is missing,
    because the debate — not the item — carries the date.
    """
    first = contribution("C1")
    second = contribution("C2", order=2, SittingDate=None)
    _, speeches, unplaceable = shape_day([first, second], WINDOW, "v2")
    assert {s.id for s in speeches} == {"C1", "C2"} and unplaceable == []


# A record that breaks a schema rule is quarantined and counted, never allowed to abort a window --


def test_an_undated_division_is_quarantined_and_the_window_still_loads(scratch, scratch_dsn, tmp_path):
    """division_date is NOT NULL, and the loader read `.isoformat` of it before the database could object: one undated
    record crashed the whole window. A ten-year load will meet records the six windows never showed.
    """
    root = _division_cache(tmp_path)
    detail = {**FIXTURES["commons_division_detail"], "DivisionId": 99, "Number": 7, "Date": None}
    write(root, commons_detail_path(99), detail)
    load = _load(scratch_dsn, root, with_bills=False)
    assert load.speeches == 1, "the window's text still loaded"
    assert load.divisions == 0 and load.quarantined == {"division_without_date": 1}
    assert load.as_dict()["quarantined"] == {"division_without_date": 1}, "and the report says so"
    assert scratch.execute("SELECT count(*) FROM division").fetchone()[0] == 0


# Quarantine keeps one bad record from aborting a window; a CEILING keeps a systematic loss from -- --- loading as a
# success.


def window_load(served=1000, unplaceable=0, divisions=100, undated=0, bills=50, untitled=0):
    load = L.WindowLoad(window_id="W")
    load.served, load.unplaceable, load.divisions, load.bills = served, [f"u{i}" for i in range(unplaceable)], \
        divisions, bills
    load.quarantined.update({k: v for k, v in (("division_without_date", undated), ("bill_without_title", untitled))
                             if v})
    return load


@pytest.mark.parametrize("kw, flagged", [
    ({}, []),
    ({"unplaceable": 25}, []),  # the absolute floor: a handful is not systematic
    ({"unplaceable": 26}, ["unplaceable"]),
    ({"served": 100_000, "unplaceable": 500}, []),  # 0.5% of a big window
    ({"served": 100_000, "unplaceable": 501}, ["unplaceable"]),
    ({"undated": 3}, []),
    ({"undated": 4}, ["division_without_date"]),
    ({"divisions": 1000, "undated": 21}, ["division_without_date"]),  # 21 of 1,021 is over 2%
    ({"untitled": 4}, ["bill_without_title"]),
    ({"divisions": 0, "undated": 50}, ["division_without_date"]),  # a window whose EVERY division was undated
])  # fmt: skip
def test_a_systematic_loss_is_a_problem_and_a_stray_record_is_not(kw, flagged):
    problems = window_load(**kw).problems()
    assert [p.split(":")[0] for p in problems] == flagged


def test_the_report_carries_the_problems():
    assert window_load(unplaceable=26).as_dict()["problems"] == ["unplaceable: 26 of 1000 (ceiling 25)"]


def test_the_loader_fails_on_a_problem_unless_the_owner_accepts_it():
    import importlib.util

    spec = importlib.util.spec_from_file_location("load_corpus_v2_q02", ROOT / "scripts" / "load_corpus_v2.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    clean, bad = window_load(), window_load(unplaceable=26)
    assert script.verdict([clean]) == 0
    assert script.verdict([clean, bad]) == 1, "one bad window fails the run: the decade pipeline stops there"
    assert script.verdict([bad], accept_anomalies=True) == 0


def test_a_pipeline_step_is_told_which_windows_were_accepted(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("load_corpus_v2_q02b", ROOT / "scripts" / "load_corpus_v2.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    bad = window_load(unplaceable=26)  # window "W"
    monkeypatch.setenv(script.ACCEPT_ENV, "C2016Q1, W")
    assert script.verdict([bad]) == 0
    monkeypatch.setenv(script.ACCEPT_ENV, "C2016Q1")
    assert script.verdict([bad]) == 1, "accepting one window never accepts another"
