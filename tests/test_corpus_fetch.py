"""Corpus v2 fetch tool: the failure modes, not the happy path."""

import asyncio
import gzip
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from ingest.corpus_fetch import (
    DONE,
    FAILED,
    KNOWN_GAP,
    NOT_SITTING,
    SHARED,
    SOURCE_ORDER,
    Breaker,
    BreakerPolicy,
    Cache,
    ConfigError,
    Context,
    DiskGuard,
    DiskPolicy,
    Interrupted,
    Manifest,
    ManifestWriter,
    StopFlag,
    Unit,
    UnitResult,
    WindowSpec,
    already_complete,
    build_client,
    build_report,
    clear_failed,
    contributions_path,
    fetch_window,
    fingerprint,
    load_config,
    load_known_member_ids,
    parse_and_check,
    plan_units,
    probe_path,
    reconcile_divisions,
    safe_params,
    scan_window,
    verify_window,
)
from ingest.runtime import run

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "corpus_fetch.json").read_text(encoding="utf-8"))
REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "corpus_v2.toml"


# --------------------------------------------------------------------------- Config helpers
# ---------------------------------------------------------------------------


def write_config(tmp_path: Path, windows: str, **overrides) -> Path:
    body = [
        "[corpus]", 'version = "v2"', 'cache_root = "cache"', 'log_dir = "logs"', "",
        "[http]", f'user_agent = "{overrides.get("user_agent", "on-the-record/test")}"',
        f'concurrency = {overrides.get("concurrency", 2)}',
        f'max_concurrency = {overrides.get("max_concurrency", 4)}',
        "min_interval = 0.0", "connect_timeout = 1.0", "read_timeout = 2.0",
        f'max_attempts = {overrides.get("max_attempts", 3)}',
        "backoff_base = 0.001", "backoff_cap = 0.01",
        f'page_size = {overrides.get("page_size", 100)}', "",
        "[breaker]", f'consecutive_failures = {overrides.get("consecutive_failures", 10)}',
        "pause_seconds = 0.001", "pause_cap = 0.002",
        f'max_pauses = {overrides.get("max_pauses", 2)}', "",
        "[disk]", f'min_free_gb = {overrides.get("min_free_gb", 0.0)}', "check_every_units = 1", "",
        "[sources]", f'enabled = {overrides.get("sources", json.dumps(["probe", "contributions"]))}', "",
        "[members]", 'endpoints = ["", "Biography"]',
        'known_ids_file = "config/known_mnis_ids.txt"',
        f'fetch_known = {str(overrides.get("fetch_known", True)).lower()}', "",
        "[estimate]", "chunks_per_contribution = 10.07", "served_fraction = 0.9961",
        "bytes_per_contribution = 2600", "",
        windows,
    ]
    path = tmp_path / "corpus_v2.toml"
    path.write_text("\n".join(body), encoding="utf-8")
    return path


ONE_WINDOW = """
[[windows]]
id = "T1"
start = 2026-09-01
end = 2026-09-02
role = "dev_test"
active = true
"""


def cfg_for(tmp_path: Path, windows: str = ONE_WINDOW, **overrides):
    return load_config(write_config(tmp_path, windows, **overrides), repo_root=tmp_path,
                       today=date(2026, 9, 23))


# --------------------------------------------------------------------------- A fake Hansard, with a dial for every way
# it can misbehave ---------------------------------------------------------------------------


class FakeAPI:
    """Serves the recorded payloads, and can be told to fail in specific, realistic ways."""

    def __init__(self, *, day_totals: dict[str, int] | None = None, tag: str = "") -> None:
        self.day_totals = day_totals or {}
        self.tag = tag  # marks the ids this instance serves, so a reused stale page shows up
        self.calls: list[tuple[str, dict]] = []
        self.fail_times: dict[str, int] = {}  # url fragment -> how many times to fail first
        self.fail_status: dict[str, int] = {}  # url fragment -> the status to fail with
        self.raise_exc: dict[str, int] = {}  # url fragment -> how many times to raise
        self.exc_factory = lambda: httpx.ConnectError("connection reset by peer")
        self.retry_after: str | None = None
        self.hard_fail: set[str] = set()  # url fragment -> always this status
        self.hard_status = 403
        self.drop_ids: dict[str, set[str]] = {}  # order -> ids that order will not return
        # Divisions.
        self.divisions: list[tuple[str, int]] | None = None
        self.hansard_divisions: list[tuple[str, int]] | None = None
        self.page_cap: int | None = None
        self.ignore_skip = False
        self.deferred: set[tuple[str, int]] = set()
        self.committee: set[tuple[str, int]] = set()

    def _commons_search(self, params: dict) -> httpx.Response:
        if self.divisions is None:
            return httpx.Response(200, json=FIXTURES["commons_division_search"]
                                  if not params.get("queryParameters.skip") else [])
        skip = 0 if self.ignore_skip else int(params.get("queryParameters.skip", 0))
        take = int(params.get("queryParameters.take", 100))
        template = FIXTURES["commons_division_search"][0]
        window = self.divisions[skip:skip + min(take, self.page_cap or take)]
        return httpx.Response(200, json=[
            {**template, "DivisionId": 9000 + skip + i, "Date": f"{day}T16:00:00", "Number": number,
             "IsDeferred": (day, number) in self.deferred}
            for i, (day, number) in enumerate(window)])

    def _hansard_divisions(self, params: dict) -> httpx.Response:
        if self.hansard_divisions is None:
            return httpx.Response(200, json=FIXTURES["hansard_divisions"])
        house = params.get("queryParameters.house")
        rows = self.hansard_divisions if house == "Commons" else []
        skip = int(params.get("queryParameters.skip", 0))
        take = int(params.get("queryParameters.take", 100))
        template = FIXTURES["hansard_divisions"]["Results"][0]
        return httpx.Response(200, json={
            "TotalResultCount": len(rows),
            "Results": [{**template, "Id": 70000 + skip + i, "Date": f"{day}T16:00:00",
                         "Number": str(number), "House": house,
                         "IsCommitteeDivision": (day, number) in self.committee}
                        for i, (day, number) in enumerate(rows[skip:skip + take])]})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _match(self, table: dict, url: str) -> str | None:
        return next((k for k in table if k in url), None)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        params = dict(request.url.params)
        self.calls.append((url, params))

        if key := self._match({k: 1 for k in self.hard_fail}, url):
            return httpx.Response(self.hard_status, text=f"forbidden: {key}")
        if key := self._match(self.raise_exc, url):
            if self.raise_exc[key] > 0:
                self.raise_exc[key] -= 1
                raise self.exc_factory()
        if key := self._match(self.fail_times, url):
            if self.fail_times[key] > 0:
                self.fail_times[key] -= 1
                headers = {"Retry-After": self.retry_after} if self.retry_after else {}
                return httpx.Response(self.fail_status.get(key, 503), text="upstream sulking",
                                      headers=headers)

        if "overview/calendar" in url:
            return httpx.Response(200, json=FIXTURES["calendar_month"])
        if "overview/sectionsforday" in url:
            return httpx.Response(200, json=FIXTURES["sections_for_day"])
        if "contributions/Corrections" in url:
            return httpx.Response(200, json=FIXTURES["corrections_empty"])
        if "contributions/Spoken" in url:
            return self._spoken(params)
        if "search/divisions" in url:
            return self._hansard_divisions(params)
        if "commonsvotes-api" in url and "search" in url:
            return self._commons_search(params)
        if "commonsvotes-api" in url:
            return httpx.Response(200, json=FIXTURES["commons_division_detail"])
        if "lordsvotes-api" in url:
            return httpx.Response(200, json=FIXTURES["lords_division_search"]
                                  if not params.get("skip") else [])
        if "/Sittings" in url:
            return httpx.Response(200, json=FIXTURES["bills_sittings"] if not params.get("Skip")
                                  else {"items": [], "totalResults": 0, "itemsPerPage": 100})
        if "/Stages" in url:
            return httpx.Response(200, json=FIXTURES["bill_stages"])
        if "bills-api" in url:
            return httpx.Response(200, json=FIXTURES["bill_detail"])
        if "Biography" in url:
            return httpx.Response(200, json=FIXTURES["member_biography"])
        if "members-api" in url:
            return httpx.Response(200, json=FIXTURES["member_profile"])
        return httpx.Response(404, text="no such thing")

    def _spoken(self, params: dict) -> httpx.Response:
        day = params.get("queryParameters.startDate", "")
        order = params.get("queryParameters.orderBy", "")
        take = int(params.get("queryParameters.take", 100))
        skip = int(params.get("queryParameters.skip", 0))
        total = self.day_totals.get(day, 0)
        template = FIXTURES["contributions_page"]["Results"][0]
        drop = self.drop_ids.get(order, set())
        items = []
        for i in range(skip, min(skip + take, total)):
            cid = f"{day}{self.tag}-{i:04d}"
            if cid in drop:
                continue
            items.append({**template, "ContributionExtId": cid,
                          "SittingDate": f"{day}T00:00:00",
                          "House": "Commons" if i % 3 else "Lords",
                          "Section": "Commons Chamber" if i % 3 else "Lords Chamber",
                          "DebateSectionExtId": f"{day}-debate-{i // 10}",
                          "MemberId": 4000 + (i % 5)})
        return httpx.Response(200, json={**FIXTURES["contributions_page"],
                                         "Results": items, "TotalResultCount": total})


def go(cfg, window_id="T1", api: FakeAPI | None = None, **kwargs):
    """Run a fetch against the fake API on a Windows-safe event loop."""
    api = api or FakeAPI()
    return run(fetch_window(cfg, cfg.window(window_id), transport=api.transport(), **kwargs))


# --------------------------------------------------------------------------- Config validation
# ---------------------------------------------------------------------------


def test_real_config_loads_and_lists_the_windows_in_fetch_order():
    cfg = load_config(CONFIG, today=date(2026, 9, 23))
    assert [w.id for w in cfg.active_windows] == ["W0", "W4", "W2", "W1", "W3", "W5"]

    # W5 is the exam.
    w5 = next(w for w in cfg.active_windows if w.id == "W5")
    assert w5.role == "heldout", "W5 is the exam window and must never be dev or test"
    assert (w5.start, w5.end) == (date(2026, 4, 1), date(2026, 7, 31))
    assert {w.id for w in cfg.active_windows if w.role == "heldout"} == {"W3", "W5"}
    assert cfg.window("W3").role == "heldout"
    # Hansard_v2 is built from the cache ALONE, so the one window used as the rebuild oracle cannot be the one window
    # missing from the cache.
    assert cfg.window("W0").active is True
    assert cfg.fetch_known_members is True  # ditto: a member held only in `hansard` is not held
    assert cfg.http.page_size == 100  # take=200 returns HTTP 500; never raise this blind


def test_user_agent_may_not_contain_an_email_address(tmp_path):
    with pytest.raises(ConfigError, match="must not contain"):
        cfg_for(tmp_path, user_agent="on-the-record (owner@example.invalid)")


def test_reversed_window_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="after end"):
        cfg_for(tmp_path, '[[windows]]\nid="B"\nstart=2026-09-10\nend=2026-09-01\n')


def test_future_window_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="in the future"):
        cfg_for(tmp_path, '[[windows]]\nid="B"\nstart=2026-09-10\nend=2027-01-01\n')


def test_overlapping_active_windows_are_refused(tmp_path):
    both = ('[[windows]]\nid="A"\nstart=2026-09-01\nend=2026-09-10\n\n'
            '[[windows]]\nid="B"\nstart=2026-09-05\nend=2026-09-12\n')
    with pytest.raises(ConfigError, match="overlap"):
        cfg_for(tmp_path, both)


def test_inactive_windows_may_overlap_because_candidates_must(tmp_path):
    both = ('[[windows]]\nid="A"\nstart=2026-09-01\nend=2026-09-10\n\n'
            '[[windows]]\nid="B"\nstart=2026-09-05\nend=2026-09-12\nactive=false\n')
    assert [w.id for w in cfg_for(tmp_path, both).windows] == ["A", "B"]


def test_unknown_source_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="unknown source"):
        cfg_for(tmp_path, ONE_WINDOW, sources=json.dumps(["probe", "telepathy"]))


def test_duplicate_window_id_is_refused(tmp_path):
    both = ('[[windows]]\nid="A"\nstart=2026-09-01\nend=2026-09-02\n\n'
            '[[windows]]\nid="A"\nstart=2026-09-05\nend=2026-09-06\nactive=false\n')
    with pytest.raises(ConfigError, match="duplicate"):
        cfg_for(tmp_path, both)


def test_concurrency_is_capped_at_four_because_this_is_someone_elses_api(tmp_path):
    with pytest.raises(ConfigError, match="capped at 4"):
        cfg_for(tmp_path, ONE_WINDOW, max_concurrency=8, concurrency=8)
    with pytest.raises(ConfigError, match="exceeds max_concurrency"):
        cfg_for(tmp_path, ONE_WINDOW, concurrency=4, max_concurrency=2)


def test_page_size_above_one_hundred_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="outside 1..100"):
        cfg_for(tmp_path, ONE_WINDOW, page_size=200)


def test_unknown_window_names_the_ones_that_exist(tmp_path):
    cfg = cfg_for(tmp_path)
    with pytest.raises(ConfigError, match="config has T1"):
        cfg.window("nope")


def test_known_member_ids_file_tolerates_comments_and_junk(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# mnis ids\n4514\n\n1427\nnot-a-number\n", encoding="utf-8")
    assert load_known_member_ids(path) == {4514, 1427}
    assert load_known_member_ids(tmp_path / "absent.txt") == set()


def test_a_byte_order_mark_does_not_swallow_the_first_member(tmp_path):
    """Windows PowerShell 5.1's `Out-File -Encoding utf8` writes a BOM. Reading as plain utf-8 turns the first line
    into "﻿8" and loses that member without a word -- which is how this was found, exporting 1,268 ids and getting
    1,267 back.
    """
    path = tmp_path / "ids.txt"
    path.write_text("8\n14\n36\n", encoding="utf-8-sig")
    assert load_known_member_ids(path) == {8, 14, 36}


# --------------------------------------------------------------------------- Cache: atomic writes
# ---------------------------------------------------------------------------


def test_write_is_atomic_and_hashes_the_raw_bytes(tmp_path):
    cache = Cache(tmp_path)
    written = cache.write("a/b.json.gz", b'{"x":1}')
    assert cache.read("a/b.json.gz") == b'{"x":1}'
    assert cache.sha256("a/b.json.gz") == written.sha256
    assert written.raw_bytes == 7


def test_gzip_is_deterministic_so_a_rewrite_is_byte_identical(tmp_path):
    cache = Cache(tmp_path)
    cache.write("a.json.gz", b"hello")
    first = (tmp_path / "a.json.gz").read_bytes()
    cache.write("a.json.gz", b"hello")
    assert (tmp_path / "a.json.gz").read_bytes() == first  # mtime=0, so no spurious diffs


def test_a_kill_mid_write_leaves_no_file_that_looks_complete(tmp_path, monkeypatch):
    cache = Cache(tmp_path)
    import os as real_os

    def die(*a, **k):
        raise KeyboardInterrupt("killed between the write and the rename")

    monkeypatch.setattr(real_os, "replace", die)
    with pytest.raises(KeyboardInterrupt):
        cache.write("day/page.json.gz", b'{"Results":[]}')
    assert not (tmp_path / "day" / "page.json.gz").exists()
    assert list((tmp_path / "day").glob("*.part")) == []  # and no litter either


def test_sweep_partials_removes_leftovers_but_never_a_real_file(tmp_path):
    cache = Cache(tmp_path)
    cache.write("keep.json.gz", b"x")
    (tmp_path / ".keep.json.gz.999.deadbeef.part").write_bytes(b"junk")
    assert cache.sweep_partials() == 1
    assert cache.exists("keep.json.gz")


def test_sha256_of_a_truncated_file_is_none_not_a_crash(tmp_path):
    cache = Cache(tmp_path)
    cache.write("x.json.gz", b"hello")
    (tmp_path / "x.json.gz").write_bytes(b"not gzip at all")
    assert cache.sha256("x.json.gz") is None


# --------------------------------------------------------------------------- HTTP resilience
# ---------------------------------------------------------------------------


def _client(api: FakeAPI, **over):
    from ingest.corpus_fetch import HttpPolicy

    policy = HttpPolicy(user_agent="test", concurrency=1, max_concurrency=4, min_interval=0.0,
                        connect_timeout=1.0, read_timeout=1.0, write_timeout=1.0, pool_timeout=1.0,
                        max_attempts=over.get("max_attempts", 4), backoff_base=0.001,
                        backoff_cap=0.002, retry_after_cap=0.01, page_size=100,
                        slowdown_factor=2.0, slowdown_decay=over.get("decay", 60.0))
    stop = over.get("stop") or StopFlag()
    return build_client(policy, stop, transport=api.transport())


@pytest.mark.parametrize("status", [500, 502, 503, 504, 408, 425, 429])
def test_retryable_statuses_are_retried_and_then_succeed(status):
    api = FakeAPI()
    api.fail_times["calendar"] = 2
    api.fail_status["calendar"] = status
    http, client = _client(api)

    async def main():
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()
        return result

    result = run(main())
    assert result.ok and result.attempts == 3 and result.retries == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_other_4xx_is_not_retried_and_keeps_a_body_snippet(status):
    api = FakeAPI()
    api.hard_fail = {"calendar"}
    api.hard_status = status
    http, client = _client(api)

    async def main():
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()
        return result

    result = run(main())
    assert not result.ok and result.attempts == 1 and result.status == status
    assert "forbidden" in (result.error or "")


def test_allow_404_turns_a_missing_optional_endpoint_into_a_normal_answer():
    api = FakeAPI()
    api.hard_fail = {"calendar"}
    api.hard_status = 404
    http, client = _client(api)

    async def main():
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json",
                                      allow_404=True)
        await http.aclose()
        return result

    assert run(main()).is_404


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("reset"), httpx.ReadError("reset"), httpx.RemoteProtocolError("bad"),
    httpx.ConnectTimeout("slow"), httpx.ReadTimeout("slow"), httpx.PoolTimeout("busy"),
])
def test_connection_resets_and_timeouts_are_retried(exc):
    """Gotcha 21: connections from this machine are intermittently reset. Four hours of fetching will hit every one of
    these.
    """
    api = FakeAPI()
    api.raise_exc["calendar"] = 2
    api.exc_factory = lambda: exc
    http, client = _client(api)

    async def main():
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()
        return result

    assert run(main()).ok


def test_retries_are_bounded_and_the_error_says_how_many():
    api = FakeAPI()
    api.raise_exc["calendar"] = 99
    http, client = _client(api, max_attempts=3)

    async def main():
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()
        return result

    result = run(main())
    assert not result.ok and result.attempts == 3 and "after 3 attempts" in (result.error or "")


def test_retry_after_is_honoured_and_capped():
    api = FakeAPI()
    api.fail_times["calendar"] = 1
    api.fail_status["calendar"] = 429
    api.retry_after = "9999"  # a hostile value; the cap must apply
    http, client = _client(api)

    async def main():
        import time
        t0 = time.monotonic()
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        elapsed = time.monotonic() - t0
        await http.aclose()
        return result, elapsed

    result, elapsed = run(main())
    assert result.ok and elapsed < 5  # capped at retry_after_cap (0.01s here), not 9,999s


def test_a_429_slows_every_later_request_down():
    api = FakeAPI()
    api.fail_times["calendar"] = 1
    api.fail_status["calendar"] = 429
    http, client = _client(api)

    async def main():
        await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()

    run(main())
    assert client.rate_limited == 1
    assert client._extra_interval >= 0.0 and client._slow_until > 0  # the slow-down is armed


def test_an_http_date_retry_after_falls_back_to_backoff_rather_than_crashing():
    api = FakeAPI()
    api.fail_times["calendar"] = 1
    api.fail_status["calendar"] = 429
    api.retry_after = "Wed, 21 Oct 2026 07:28:00 GMT"
    http, client = _client(api)

    async def main():
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()
        return result

    assert run(main()).ok


def test_backoff_uses_full_jitter_so_retries_do_not_synchronise():
    api = FakeAPI()
    http, client = _client(api)
    samples = [client._jitter(4) for _ in range(400)]
    run(http.aclose())
    ceiling = min(client.policy.backoff_cap, client.policy.backoff_base * 8)
    assert all(0.0 <= s <= ceiling for s in samples)
    assert min(samples) < ceiling * 0.25  # full jitter reaches near zero; equal jitter would not


def test_the_real_user_agent_is_sent_and_has_no_at_sign():
    cfg = load_config(CONFIG, today=date(2026, 9, 23))
    api = FakeAPI()
    http, client = build_client(cfg.http, StopFlag(), transport=api.transport())

    async def main():
        await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()

    run(main())
    assert "@" not in cfg.http.user_agent
    assert "on-the-record" in cfg.http.user_agent


# --------------------------------------------------------------------------- Ctrl+C
# ---------------------------------------------------------------------------


def test_ctrl_c_during_a_backoff_sleep_stops_within_a_beat():
    stop = StopFlag()
    api = FakeAPI()
    http, client = _client(api, stop=stop)

    async def main():
        stop.request()
        with pytest.raises(Interrupted):
            await client.sleep(3600)  # a long pause must not hold Ctrl+C for an hour
        await http.aclose()

    run(main())


def test_ctrl_c_mid_run_saves_state_and_exits_130(tmp_path):
    cfg = cfg_for(tmp_path)
    stop = StopFlag()
    api = FakeAPI(day_totals={"2026-09-01": 250, "2026-09-02": 250})
    calls = {"n": 0}
    original = api._handle

    def trip(request):
        calls["n"] += 1
        if calls["n"] == 4:
            stop.request()
        return original(request)

    outcome = run(fetch_window(cfg, cfg.window("T1"), transport=httpx.MockTransport(trip), stop=stop))
    assert outcome.interrupted and outcome.exit_code == 130
    assert "state saved" in outcome.message
    # Whatever was finished before the interrupt is in the manifest, committed.
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        assert manifest.conn.execute("SELECT count(*) FROM unit").fetchone()[0] >= 1
        assert manifest.conn.execute(
            "SELECT outcome FROM run ORDER BY id DESC LIMIT 1").fetchone()[0] == "interrupted"
    finally:
        manifest.close()


def test_a_second_run_after_an_interrupt_finishes_the_job(tmp_path):
    cfg = cfg_for(tmp_path)
    stop = StopFlag()
    api = FakeAPI(day_totals={"2026-09-01": 250, "2026-09-02": 250})
    calls = {"n": 0}
    original = api._handle

    def trip(request):
        calls["n"] += 1
        if calls["n"] == 3:
            stop.request()
        return original(request)

    run(fetch_window(cfg, cfg.window("T1"), transport=httpx.MockTransport(trip), stop=stop))
    second = go(cfg, api=FakeAPI(day_totals={"2026-09-01": 250, "2026-09-02": 250}))
    assert second.exit_code == 0 and not second.interrupted
    assert verify_window(cfg, cfg.window("T1")).clean


# --------------------------------------------------------------------------- Resume and idempotence
# ---------------------------------------------------------------------------


def test_a_second_run_of_a_finished_window_fetches_nothing(tmp_path):
    cfg = cfg_for(tmp_path)
    totals = {"2026-09-01": 150, "2026-09-02": 0}
    first = go(cfg, api=FakeAPI(day_totals=totals))
    assert first.exit_code == 0 and first.units_done > 0

    api2 = FakeAPI(day_totals=totals)
    second = run(fetch_window(cfg, cfg.window("T1"), transport=api2.transport()))
    assert api2.calls == [], "a finished window must make zero requests"
    assert second.units_done == 0 and second.units_skipped == first.units_done


def test_a_deleted_cache_file_makes_the_unit_work_again(tmp_path):
    cfg = cfg_for(tmp_path)
    totals = {"2026-09-01": 50, "2026-09-02": 0}
    go(cfg, api=FakeAPI(day_totals=totals))
    (cfg.cache_root / probe_path("T1", date(2026, 9, 1))).unlink()
    api2 = FakeAPI(day_totals=totals)
    run(fetch_window(cfg, cfg.window("T1"), transport=api2.transport()))
    assert any("Spoken" in url for url, _ in api2.calls)


def test_a_corrupted_cache_file_makes_the_unit_work_again(tmp_path):
    cfg = cfg_for(tmp_path)
    totals = {"2026-09-01": 50, "2026-09-02": 0}
    go(cfg, api=FakeAPI(day_totals=totals))
    target = cfg.cache_root / probe_path("T1", date(2026, 9, 1))
    target.write_bytes(gzip.compress(b'{"TotalResultCount": 999999}'))  # valid gzip, wrong hash
    api2 = FakeAPI(day_totals=totals)
    run(fetch_window(cfg, cfg.window("T1"), transport=api2.transport()))
    assert any("Spoken" in url for url, _ in api2.calls)


def test_force_refetches_even_a_complete_window(tmp_path):
    cfg = cfg_for(tmp_path)
    totals = {"2026-09-01": 50, "2026-09-02": 0}
    go(cfg, api=FakeAPI(day_totals=totals))
    api2 = FakeAPI(day_totals=totals)
    run(fetch_window(cfg, cfg.window("T1"), transport=api2.transport(), force=True))
    assert api2.calls


def test_a_failed_unit_is_retried_on_the_next_run(tmp_path):
    cfg = cfg_for(tmp_path)
    api = FakeAPI(day_totals={"2026-09-01": 50, "2026-09-02": 50})
    api.hard_fail = {"startDate=2026-09-02"}
    outcome = go(cfg, api=api)
    assert outcome.units_failed >= 1 and outcome.exit_code == 3

    api2 = FakeAPI(day_totals={"2026-09-01": 50, "2026-09-02": 50})
    second = run(fetch_window(cfg, cfg.window("T1"), transport=api2.transport()))
    assert second.units_failed == 0
    assert any("2026-09-02" in str(params) for _, params in api2.calls)


def test_one_bad_day_never_ends_the_run(tmp_path):
    cfg = cfg_for(tmp_path, sources=json.dumps(["probe"]))
    api = FakeAPI(day_totals={"2026-09-01": 10, "2026-09-02": 10})
    api.hard_fail = {"startDate=2026-09-01"}
    outcome = go(cfg, api=api)
    assert outcome.units_failed == 1 and outcome.units_done == 1  # the other day still landed


# --------------------------------------------------------------------------- Both sort orders, merging and known gaps
# ---------------------------------------------------------------------------


def test_both_sort_orders_are_read_and_merged_by_id(tmp_path):
    cfg = cfg_for(tmp_path)
    api = FakeAPI(day_totals={"2026-09-01": 120, "2026-09-02": 0})
    go(cfg, api=api)
    orders = {p.get("queryParameters.orderBy") for url, p in api.calls if "Spoken" in url}
    assert orders == {"SittingDateAsc", "SittingDateDesc"}
    scan = scan_window(Cache(cfg.cache_root), cfg.window("T1"))
    assert scan.contributions == 120  # merged, not 240


def test_a_sort_order_disagreement_is_recorded_not_hidden(tmp_path):
    cfg = cfg_for(tmp_path)
    api = FakeAPI(day_totals={"2026-09-01": 100, "2026-09-02": 0})
    api.drop_ids["SittingDateAsc"] = {"2026-09-01-0007"}
    go(cfg, api=api)
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        row = manifest.unit("contributions", "T1", "2026-09-01")
        assert "sort_order_disagreement" in (row["last_error"] or "")
        assert "desc_only=1" in row["last_error"]
        assert row["items"] == 100  # the union is the record, so nothing was lost
    finally:
        manifest.close()


def test_a_contribution_the_api_will_not_serve_is_a_known_gap_not_a_failure(tmp_path):
    """That is the source's gap, not ours."""
    cfg = cfg_for(tmp_path)
    api = FakeAPI(day_totals={"2026-09-01": 100, "2026-09-02": 0})
    api.drop_ids["SittingDateAsc"] = {"2026-09-01-0005"}
    api.drop_ids["SittingDateDesc"] = {"2026-09-01-0005"}
    go(cfg, api=api)
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        row = manifest.unit("contributions", "T1", "2026-09-01")
        assert row["status"] == KNOWN_GAP
        assert "unserved:1" in row["last_error"]
    finally:
        manifest.close()
    assert verify_window(cfg, cfg.window("T1")).clean  # a known gap is not a problem


def test_a_non_sitting_day_is_recorded_as_such_not_left_blank(tmp_path):
    cfg = cfg_for(tmp_path)
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0}))
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        rows = manifest.units(window_id="T1", source="probe")
        assert {r["status"] for r in rows} == {NOT_SITTING}
    finally:
        manifest.close()


def test_a_day_row_is_written_for_both_houses_even_when_one_did_not_sit(tmp_path):
    cfg = cfg_for(tmp_path)
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 30, "2026-09-02": 0}))
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        rows = {(r["date"], r["house"]): r for r in manifest.days("T1")}
        assert ("2026-09-01", "Commons") in rows and ("2026-09-01", "Lords") in rows
        assert rows[("2026-09-01", "Commons")]["merged"] > 0
    finally:
        manifest.close()


def test_a_page_the_api_refuses_marks_the_day_failed_and_says_which_page(tmp_path):
    cfg = cfg_for(tmp_path, page_size=100)
    api = FakeAPI(day_totals={"2026-09-01": 300, "2026-09-02": 0})
    api.hard_fail = {"skip=200"}
    go(cfg, api=api)
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        row = manifest.unit("contributions", "T1", "2026-09-01")
        assert row["status"] == FAILED and "page_failed" in row["last_error"]
    finally:
        manifest.close()


# --------------------------------------------------------------------------- Manifest
# ---------------------------------------------------------------------------


def test_the_manifest_records_a_schema_fingerprint_so_api_drift_is_visible(tmp_path):
    cfg = cfg_for(tmp_path)
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 10, "2026-09-02": 0}))
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        row = manifest.unit("probe", "T1", "2026-09-01")
        assert row["fingerprint"].startswith("obj{") and "TotalResultCount" in row["fingerprint"]
    finally:
        manifest.close()


def test_fingerprint_distinguishes_shapes():
    assert fingerprint({"b": 1, "a": 2}) == "obj{a,b}"
    assert fingerprint([{"x": 1}]) == "list[obj{x}]"
    assert fingerprint([]) == "list[]"
    assert fingerprint("2026-09-15") == "str"
    assert fingerprint({"a": 1}) != fingerprint({"a": 1, "b": 2})  # a new key must show


def test_manifest_writer_is_the_only_writer_and_flushes_on_close(tmp_path):
    manifest = Manifest(tmp_path / "m.sqlite")

    async def main():
        writer = ManifestWriter(manifest, batch=1000, interval=99.0)  # never auto-commits
        writer.start()
        for i in range(10):
            writer.record_unit("probe", "T1", str(i), UnitResult(status=DONE, path=f"p{i}"))
        await writer.close()

    run(main())
    reader = Manifest(tmp_path / "m.sqlite", read_only=True)
    try:
        assert reader.conn.execute("SELECT count(*) FROM unit").fetchone()[0] == 10
    finally:
        reader.close()
        manifest.close()


def test_manifest_writer_flush_waits_for_the_queue_to_drain(tmp_path):
    manifest = Manifest(tmp_path / "m.sqlite")

    async def main():
        writer = ManifestWriter(manifest, batch=1000, interval=99.0)
        writer.start()
        writer.record_unit("probe", "T1", "x", UnitResult(status=DONE))
        await writer.flush()
        n = manifest.conn.execute("SELECT count(*) FROM unit").fetchone()[0]
        await writer.close()
        return n

    assert run(main()) == 1


def test_attempts_accumulate_across_runs_rather_than_resetting(tmp_path):
    manifest = Manifest(tmp_path / "m.sqlite")
    try:
        manifest.apply([(__import__("ingest.corpus_fetch", fromlist=["UPSERT_UNIT"]).UPSERT_UNIT,
                         ("probe", "T1", "k", None, None, None, FAILED, 503, 3, None, 0, None,
                          None, "t", "boom"))] * 2)
        manifest.commit()
        assert manifest.unit("probe", "T1", "k")["attempts"] == 6
    finally:
        manifest.close()


def test_the_manifest_is_in_wal_mode_so_a_reader_never_blocks_the_writer(tmp_path):
    manifest = Manifest(tmp_path / "m.sqlite")
    try:
        assert manifest.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        manifest.close()


def test_params_never_carry_anything_that_looks_like_a_credential():
    assert json.loads(safe_params({"date": "2026-09-01", "api_key": "hunter2"})) == {"date": "2026-09-01"}
    assert safe_params(None) is None
    for name in ("token", "secret", "password", "AUTH"):
        assert name.casefold() not in (safe_params({name: "x", "ok": 1}) or "").casefold()


# --------------------------------------------------------------------------- verify
# ---------------------------------------------------------------------------


def _fetched(tmp_path, totals=None, **over):
    cfg = cfg_for(tmp_path, **over)
    go(cfg, api=FakeAPI(day_totals=totals or {"2026-09-01": 60, "2026-09-02": 40}))
    return cfg


def test_verify_is_clean_after_a_complete_fetch(tmp_path):
    cfg = _fetched(tmp_path)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean, [str(p) for p in report.problems]
    assert report.checked_units > 0


def test_verify_catches_a_deleted_file(tmp_path):
    cfg = _fetched(tmp_path)
    (cfg.cache_root / probe_path("T1", date(2026, 9, 1))).unlink()
    kinds = {p.kind for p in verify_window(cfg, cfg.window("T1")).problems}
    assert "missing_file" in kinds


def test_verify_catches_a_hash_mismatch(tmp_path):
    cfg = _fetched(tmp_path)
    target = cfg.cache_root / probe_path("T1", date(2026, 9, 2))
    target.write_bytes(gzip.compress(b'{"TotalResultCount": 1}'))
    kinds = {p.kind for p in verify_window(cfg, cfg.window("T1")).problems}
    assert "hash_mismatch" in kinds


def test_verify_catches_a_sitting_day_with_no_debates(tmp_path):
    cfg = _fetched(tmp_path)
    manifest = Manifest(cfg.manifest_path)
    try:
        manifest.conn.execute("UPDATE day SET debates = 0 WHERE date = ?", ("2026-09-01",))
        manifest.commit()
    finally:
        manifest.close()
    kinds = {p.kind for p in verify_window(cfg, cfg.window("T1")).problems}
    assert "sitting_day_without_debates" in kinds


def test_verify_catches_an_unreadable_page(tmp_path):
    cfg = _fetched(tmp_path)
    page = cfg.cache_root / contributions_path("T1", date(2026, 9, 1), "SittingDateAsc", 0)
    page.write_bytes(b"this is not gzip")
    kinds = {p.kind for p in verify_window(cfg, cfg.window("T1")).problems}
    assert "unreadable_page" in kinds or "hash_mismatch" in kinds


def test_verify_catches_a_missing_follow_on_member(tmp_path):
    cfg = cfg_for(tmp_path, sources=json.dumps(["probe", "contributions", "members"]))
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 30, "2026-09-02": 0}))
    assert verify_window(cfg, cfg.window("T1")).clean
    manifest = Manifest(cfg.manifest_path)
    try:
        manifest.conn.execute("DELETE FROM unit WHERE source='members'")
        manifest.commit()
    finally:
        manifest.close()
    kinds = {p.kind for p in verify_window(cfg, cfg.window("T1")).problems}
    assert "member_missing" in kinds


def test_verify_reports_a_pending_unit(tmp_path):
    cfg = _fetched(tmp_path)
    manifest = Manifest(cfg.manifest_path)
    try:
        manifest.conn.execute("UPDATE unit SET status='pending' WHERE source='probe'")
        manifest.commit()
    finally:
        manifest.close()
    assert "pending_unit" in {p.kind for p in verify_window(cfg, cfg.window("T1")).problems}


def test_clear_failed_leaves_the_cache_alone(tmp_path):
    cfg = cfg_for(tmp_path)
    api = FakeAPI(day_totals={"2026-09-01": 40, "2026-09-02": 40})
    api.hard_fail = {"startDate=2026-09-02"}
    go(cfg, api=api)
    before = len(list(cfg.cache_root.rglob("*.json.gz")))
    assert clear_failed(cfg, "T1") >= 1
    assert len(list(cfg.cache_root.rglob("*.json.gz"))) == before


# --------------------------------------------------------------------------- Circuit breaker and disk guard
# ---------------------------------------------------------------------------


TEN_DAYS = """
[[windows]]
id = "T1"
start = 2026-09-01
end = 2026-09-10
role = "dev_test"
active = true
"""


def test_the_breaker_trips_pauses_and_then_gives_up_cleanly(tmp_path):
    """A dead network must not burn four hours retrying. It pauses, then stops."""
    cfg = cfg_for(tmp_path, TEN_DAYS, sources=json.dumps(["probe"]), concurrency=1,
                  consecutive_failures=2, max_pauses=1)
    api = FakeAPI(day_totals={f"2026-09-{d:02d}": 10 for d in range(1, 11)})
    api.hard_fail = {"Spoken"}
    outcome = go(cfg, api=api)
    assert outcome.breaker_stopped and outcome.exit_code == 4
    assert "circuit-breaker pause" in outcome.message
    assert len(api.calls) < 10, "it should give up well before trying every day"


def test_the_breaker_resets_after_a_success():
    breaker = Breaker(BreakerPolicy(consecutive_failures=3, pause_seconds=1, pause_cap=8, max_pauses=2))
    breaker.record(False)
    breaker.record(False)
    breaker.record(True)
    assert not breaker.tripped and breaker.consecutive == 0


def test_breaker_pauses_double_up_to_the_cap():
    breaker = Breaker(BreakerPolicy(consecutive_failures=1, pause_seconds=10, pause_cap=25, max_pauses=5))
    assert [breaker.take_pause() for _ in range(4)] == [10, 20, 25, 25]
    assert breaker.consecutive == 0  # taking a pause clears the run of failures


def test_the_disk_guard_refuses_to_start_when_the_disk_is_nearly_full(tmp_path):
    cfg = cfg_for(tmp_path, min_free_gb=10_000_000.0)  # no laptop has this much
    outcome = go(cfg, api=FakeAPI(day_totals={"2026-09-01": 10}))
    assert outcome.disk_stopped and outcome.exit_code == 5
    assert "refusing to start" in outcome.message


def test_the_disk_guard_only_checks_every_n_units(tmp_path):
    guard = DiskGuard(tmp_path, DiskPolicy(min_free_gb=10_000_000.0, check_every_units=3))
    assert [guard.low() is not None for _ in range(6)] == [False, False, True, False, False, True]
    assert guard.low(force=True) is not None


# --------------------------------------------------------------------------- Payload validation
# ---------------------------------------------------------------------------


def test_an_unparseable_payload_is_recorded_not_raised():
    payload, anomalies = parse_and_check("probe", b"<html>502 Bad Gateway</html>")
    assert payload is None and anomalies == ["unparseable:JSONDecodeError"]


def test_a_wrong_container_is_recorded():
    _, anomalies = parse_and_check("calendar", b'{"not": "a list"}')
    assert anomalies == ["shape:expected_list_got_dict"]


def test_missing_required_keys_are_named():
    _, anomalies = parse_and_check("contributions", b'{"Results": []}')
    assert anomalies == ["missing_keys:TotalResultCount"]


def test_the_real_recorded_payloads_all_validate():
    for source, key in (("contributions", "contributions_page"), ("calendar", "calendar_month"),
                        ("sections", "sections_for_day"), ("commons_detail", "commons_division_detail"),
                        ("lords_divisions", "lords_division_search"), ("bills", "bills_sittings"),
                        ("members", "member_profile"), ("corrections", "corrections_empty")):
        raw = json.dumps(FIXTURES[key]).encode()
        payload, anomalies = parse_and_check(source, raw)
        assert payload is not None and anomalies == [], (source, anomalies)


def test_an_odd_payload_does_not_stop_the_run(tmp_path):
    cfg = cfg_for(tmp_path, sources=json.dumps(["probe"]))

    def handler(request):
        return httpx.Response(200, json={"Surprise": True})  # no TotalResultCount at all

    outcome = run(fetch_window(cfg, cfg.window("T1"), transport=httpx.MockTransport(handler)))
    assert outcome.exit_code == 0
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        assert "missing_keys" in (manifest.unit("probe", "T1", "2026-09-01")["last_error"] or "")
    finally:
        manifest.close()


# --------------------------------------------------------------------------- Unit planning, scanning, report
# ---------------------------------------------------------------------------


def _ctx(cfg, window_id="T1"):
    manifest = Manifest(cfg.manifest_path)
    return Context(cfg=cfg, window=cfg.window(window_id), cache=Cache(cfg.cache_root), client=None,
                   writer=None, manifest=manifest, stop=StopFlag(),
                   breaker=Breaker(cfg.breaker), disk=DiskGuard(cfg.cache_root, cfg.disk))


def test_probe_units_cover_every_calendar_day_not_only_the_sitting_ones(tmp_path):
    """A calendar that forgets a sitting is exactly the failure a second signal exists to catch."""
    cfg = cfg_for(tmp_path)
    ctx = _ctx(cfg)
    try:
        assert len(plan_units(ctx, "probe")) == 2  # 1 and 2 September, sitting or not
    finally:
        ctx.manifest.close()


def test_contribution_units_follow_the_probes(tmp_path):
    cfg = cfg_for(tmp_path)
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 20, "2026-09-02": 0}))
    ctx = _ctx(cfg)
    try:
        keys = [u.key for u in plan_units(ctx, "contributions")]
    finally:
        ctx.manifest.close()
    assert "2026-09-01" in keys


def test_shared_units_belong_to_no_window_so_they_are_fetched_once(tmp_path):
    cfg = cfg_for(tmp_path, sources=json.dumps(["probe", "contributions", "members"]))
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 30, "2026-09-02": 0}))
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        rows = manifest.units(source="members")
        assert rows and {r["window_id"] for r in rows} == {SHARED}
    finally:
        manifest.close()


@pytest.mark.parametrize("fetch_known, expected", [(False, 0), (True, 5)])
def test_known_member_ids_are_skipped_only_when_the_config_says_so(tmp_path, fetch_known, expected):
    """Skipping members the `hansard` database already holds was right while the cache supplemented a database, and
    wrong the moment hansard_v2 is built from the cache ALONE: their records would then exist nowhere the offline
    loader can read. The default is now to fetch everyone seen; the old behaviour is one config line away.
    """
    cfg = cfg_for(tmp_path, sources=json.dumps(["probe", "contributions", "members"]),
                  fetch_known=fetch_known)
    (tmp_path / "config").mkdir(exist_ok=True)
    known = tmp_path / "config" / "known_mnis_ids.txt"
    known.write_text("\n".join(str(4000 + i) for i in range(5)), encoding="utf-8")
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 30, "2026-09-02": 0}))
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        assert len(manifest.units(source="members")) == expected
    finally:
        manifest.close()


def test_scan_window_reads_the_files_rather_than_trusting_the_manifest(tmp_path):
    cfg = cfg_for(tmp_path)
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 45, "2026-09-02": 0}))
    scan = scan_window(Cache(cfg.cache_root), cfg.window("T1"))
    assert scan.contributions == 45
    assert scan.houses["Commons"] + scan.houses["Lords"] == 45
    assert scan.member_ids
    page = cfg.cache_root / contributions_path("T1", date(2026, 9, 1), "SittingDateAsc", 0)
    page.unlink()
    assert scan_window(Cache(cfg.cache_root), cfg.window("T1")).contributions <= 45


def test_the_report_carries_the_section_census_and_the_fingerprints(tmp_path):
    cfg = cfg_for(tmp_path, sources=json.dumps(["calendar", "probe", "sections", "contributions"]))
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 60, "2026-09-02": 0}))
    data = build_report(cfg, cfg.window("T1"))
    assert data["contributions"] == 60
    assert any("Chamber" in k for k in data["section_census"])
    assert data["estimated_chunks"] == round(60 * cfg.estimate.chunks_per_contribution)
    assert data["schema_fingerprints"]
    assert data["window"]["id"] == "T1"


def test_already_complete_is_false_for_a_unit_the_manifest_has_never_seen(tmp_path):
    cfg = cfg_for(tmp_path)
    ctx = _ctx(cfg)
    try:
        assert not already_complete(ctx, Unit("probe", "T1", "2026-09-01"))
    finally:
        ctx.manifest.close()


# --------------------------------------------------------------------------- Gaps found by the mutation suite.


def test_a_window_covers_every_month_it_touches_including_the_last():
    """A window that loses its last month loses a third of W2 without a word."""
    w = WindowSpec(id="X", start=date(2025, 10, 1), end=date(2025, 12, 31), role="dev", active=True)
    assert w.months == [(2025, 10), (2025, 11), (2025, 12)]
    assert WindowSpec(id="Y", start=date(2024, 11, 5), end=date(2025, 2, 3), role="dev",
                      active=True).months == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    assert len(w.days) == 92 and w.days[-1] == date(2025, 12, 31)


def test_retry_after_actually_delays_the_retry():
    """Not just 'it did not hang': the header has to be obeyed, or a 429 storm repeats."""
    import time

    api = FakeAPI()
    api.fail_times["calendar"] = 1
    api.fail_status["calendar"] = 429
    api.retry_after = "1"
    http, client = _client(api)
    client.policy = client.policy.__class__(**{**client.policy.__dict__, "retry_after_cap": 5.0})

    async def main():
        t0 = time.monotonic()
        result = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        elapsed = time.monotonic() - t0
        await http.aclose()
        return result, elapsed

    result, elapsed = run(main())
    # backoff_base is 0.001s here, so anything near a second can only be Retry-After.
    assert result.ok and elapsed >= 0.9, f"Retry-After was ignored ({elapsed:.3f}s)"


def test_allow_404_is_a_success_not_merely_a_404():
    """is_404 is true either way; what matters is whether the caller is told it is OK."""
    api = FakeAPI()
    api.hard_fail = {"calendar"}
    api.hard_status = 404
    http, client = _client(api)

    async def main():
        allowed = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json",
                                       allow_404=True)
        refused = await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()
        return allowed, refused

    allowed, refused = run(main())
    assert allowed.ok and allowed.is_404 and allowed.content is not None
    assert not refused.ok and refused.content is None


def test_ctrl_c_is_noticed_before_a_request_is_even_sent():
    """On the first press nothing further should go out to Parliament's servers."""
    stop = StopFlag()
    api = FakeAPI()
    http, client = _client(api, stop=stop)

    async def main():
        stop.request()
        with pytest.raises(Interrupted):
            await client.get_raw("https://hansard-api.parliament.uk/overview/calendar.json")
        await http.aclose()

    run(main())
    assert api.calls == [], "a request was sent after Ctrl+C"


def test_a_stop_arriving_during_a_sleep_is_noticed_within_a_slice():
    """A 60-second backoff must not hold Ctrl+C for 60 seconds."""
    import time

    stop = StopFlag()
    api = FakeAPI()
    http, client = _client(api, stop=stop)

    async def main():
        async def press():
            await asyncio.sleep(0.05)
            stop.request()

        task = asyncio.create_task(press())
        t0 = time.monotonic()
        with pytest.raises(Interrupted):
            await client.sleep(30)
        elapsed = time.monotonic() - t0
        await task
        await http.aclose()
        return elapsed

    elapsed = run(main())
    assert elapsed < 2.0, f"took {elapsed:.2f}s to notice a stop inside a 30s sleep"


def test_a_cached_page_from_a_day_that_has_since_grown_is_refetched(tmp_path):
    """Hansard adds contributions to a day after first publishing it, and a day whose later pages failed gets retried.
    Reusing pages that still carry the old total would silently keep the short version of that day, forever.
    """
    cfg = cfg_for(tmp_path, page_size=100)
    first = FakeAPI(day_totals={"2026-09-01": 300, "2026-09-02": 0})
    first.hard_fail = {"skip=200"}  # the day ends up FAILED, with pages 0 and 100 cached
    go(cfg, api=first)
    page0 = cfg.cache_root / contributions_path("T1", date(2026, 9, 1), "SittingDateAsc", 0)
    assert json.loads(gzip.decompress(page0.read_bytes()))["TotalResultCount"] == 300

    # The day has grown, and Hansard has republished it: the ids carry a new tag.
    second = FakeAPI(day_totals={"2026-09-01": 400, "2026-09-02": 0}, tag="R")
    outcome = run(fetch_window(cfg, cfg.window("T1"), transport=second.transport()))
    assert outcome.units_failed == 0
    assert json.loads(gzip.decompress(page0.read_bytes()))["TotalResultCount"] == 400
    page1 = cfg.cache_root / contributions_path("T1", date(2026, 9, 1), "SittingDateAsc", 100)
    body = json.loads(gzip.decompress(page1.read_bytes()))
    assert body["TotalResultCount"] == 400, "a stale page was reused instead of re-fetched"
    assert body["Results"][0]["ContributionExtId"].startswith("2026-09-01R"), \
        "the reused page still holds the superseded contributions"
    assert scan_window(Cache(cfg.cache_root), cfg.window("T1")).contributions == 400


def test_the_first_sort_order_wins_when_the_two_disagree_about_an_items_content(tmp_path):
    """Merging by id must keep one item per id, deterministically: 'last write wins' makes the corpus depend on the
    order the pages happened to arrive in.
    """
    cfg = cfg_for(tmp_path)
    seen: dict[str, str] = {}

    def handler(request):
        url = str(request.url)
        params = dict(request.url.params)
        if "Spoken" not in url:
            return httpx.Response(404, text="no")
        order = params.get("queryParameters.orderBy", "")
        day = params.get("queryParameters.startDate")
        if day != "2026-09-01":
            return httpx.Response(200, json={"Results": [], "TotalResultCount": 0})
        take = int(params.get("queryParameters.take", 100))
        if take == 1:
            return httpx.Response(200, json={"Results": [], "TotalResultCount": 1})
        first = order.endswith("Asc")
        seen[order] = "FIRST" if first else "SECOND"
        # Same id, different House: whichever listing wins the merge decides the day rows.
        item = {**FIXTURES["contributions_page"]["Results"][0], "ContributionExtId": "same-id",
                "House": "Commons" if first else "Lords",
                "Section": "Commons Chamber" if first else "Lords Chamber",
                "DebateSectionExtId": "d1", "MemberId": 4514}
        return httpx.Response(200, json={"Results": [item], "TotalResultCount": 1})

    cfg_ = cfg
    run(fetch_window(cfg_, cfg_.window("T1"), transport=httpx.MockTransport(handler)))
    assert seen == {"SittingDateAsc": "FIRST", "SittingDateDesc": "SECOND"}
    manifest = Manifest(cfg_.manifest_path, read_only=True)
    try:
        rows = {r["house"]: r["merged"] for r in manifest.days("T1") if r["date"] == "2026-09-01"}
        assert manifest.unit("contributions", "T1", "2026-09-01")["items"] == 1
    finally:
        manifest.close()
    assert rows == {"Commons": 1, "Lords": 0}, \
        "the first sort order must win the merge, not whichever page arrived last"


def test_paging_stops_exactly_on_the_total_without_an_extra_request(tmp_path):
    """A day whose total is an exact multiple of the page size is the case where an off-by-one costs one wasted request
    per order, on every such day, for a year.
    """
    cfg = cfg_for(tmp_path, page_size=100)
    api = FakeAPI(day_totals={"2026-09-01": 200, "2026-09-02": 0})
    go(cfg, api=api)
    pages = [p for url, p in api.calls
             if "Spoken" in url and p.get("queryParameters.startDate") == "2026-09-01"
             and int(p.get("queryParameters.take", 0)) > 1]
    assert len(pages) == 4, [p.get("queryParameters.skip") for p in pages]  # 2 pages x 2 orders
    assert "200" not in {str(p.get("queryParameters.skip")) for p in pages}


def test_a_page_loop_that_never_ends_is_cut_off(tmp_path):
    """A total the API keeps raising, or a page that never shortens, must not spin forever."""
    cfg = cfg_for(tmp_path, page_size=1)
    api = FakeAPI(day_totals={"2026-09-01": 400, "2026-09-02": 0})
    go(cfg, api=api)
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        assert "page_runaway" in (manifest.unit("contributions", "T1", "2026-09-01")["last_error"] or "")
    finally:
        manifest.close()


def test_a_first_pause_longer_than_the_cap_is_still_capped():
    breaker = Breaker(BreakerPolicy(consecutive_failures=1, pause_seconds=100, pause_cap=25,
                                    max_pauses=5))
    assert breaker.take_pause() == 25, "a misconfigured first pause must not stall the run"


def test_every_multi_file_unit_records_a_path_that_really_is_a_directory(tmp_path):
    """A unit that writes several files records the directory holding them, and `verify` checks that path exists. A
    filename PREFIX looks fine until verify reports every such unit as missing -- which is exactly how this was
    found, on a real run.
    """
    cfg = cfg_for(tmp_path, sources=json.dumps(list(SOURCE_ORDER)))
    go(cfg, api=FakeAPI(day_totals={"2026-09-01": 150, "2026-09-02": 0}))
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        rows = manifest.units(window_id="T1")
    finally:
        manifest.close()
    assert rows
    for row in rows:
        if not row["path"] or row["sha256"]:
            continue  # single-file units record the file and are checked by hash instead
        target = cfg.cache_root / row["path"]
        assert target.is_dir(), f"{row['source']}/{row['key']} recorded {row['path']}, not a directory"
        assert any(target.rglob("*.json.gz")), f"{row['path']} is empty"


def test_the_chunk_estimate_uses_the_measured_multiplier(tmp_path):
    """Calibrated on W0: 9,021 listed contributions produced 90,852 chunks."""
    from ingest.corpus_fetch import WindowPlan

    cfg = cfg_for(tmp_path)
    plan = WindowPlan(window=cfg.window("T1"), listed=1000, per_day={"2026-09-01": 1000})
    est = plan.estimate(cfg)
    assert est["chunks"] == pytest.approx(10_070, rel=1e-6)
    assert est["chunks"] > est["served"] * 5  # a chunk count that equals the contributions is wrong


# --------------------------------------------------------------------------- Paging a server that does not honour
# `take`, and the cross-source check that is the only thing that would have caught it
# ---------------------------------------------------------------------------

DIVISION_SOURCES = json.dumps(["probe", "hansard_divisions", "commons_divisions", "lords_divisions"])


def _divisions(n: int, *, start_number: int = 300) -> list[tuple[str, int]]:
    """n divisions spread over the two days of the test window."""
    return [(f"2026-09-0{1 + i % 2}", start_number + i) for i in range(n)]


def _fetch_divisions(tmp_path, api: FakeAPI, **over):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = cfg_for(tmp_path, sources=over.pop("sources", DIVISION_SOURCES), **over)
    go(cfg, api=api)
    return cfg


def _commons_ids(cfg) -> list[int]:
    from ingest.corpus_fetch import _cached_ids
    return _cached_ids(Cache(cfg.cache_root), "commons_divisions/T1", "DivisionId")


def test_a_server_that_caps_its_page_below_take_is_still_paged_to_the_end(tmp_path):
    """The real defect, in miniature. commonsvotes-api returns 25 rows however large a `take` it is sent. Stopping on a
    short page therefore kept the newest 25 divisions of W4 out of 71 and of W2 out of 87 -- while W1 (21) and W3
    (14) were under the cap and complete, which is why it read as a 2024/2020 quirk.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.divisions, api.page_cap = _divisions(71), 25
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    assert len(_commons_ids(cfg)) == 71
    assert len({p.get("queryParameters.skip") for u, p in api.calls if "commonsvotes" in u and "search" in u}) == 4


def test_the_page_cap_is_recorded_rather_than_silently_worked_around(tmp_path):
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.divisions, api.page_cap = _divisions(40), 25
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        assert "server_page_cap:25<100" in (manifest.unit("commons_divisions", "T1", "index")["last_error"] or "")
    finally:
        manifest.close()


@pytest.mark.parametrize("n", [0, 10, 25, 50])
def test_a_list_shorter_than_or_equal_to_one_page_is_read_exactly_once_over(tmp_path, n):
    """Pages of 0, 10 and 25 rows: the cap is only a cap when a short page is FOLLOWED by rows, so a genuinely short
    last page must not be reported as one.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.divisions, api.page_cap = _divisions(n), 25
    cfg = _fetch_divisions(tmp_path, api, page_size=25)
    assert len(_commons_ids(cfg)) == n
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        assert "server_page_cap" not in (manifest.unit("commons_divisions", "T1", "index")["last_error"] or "")
    finally:
        manifest.close()


def test_a_server_that_ignores_skip_stops_the_loop_instead_of_spinning(tmp_path):
    """Paging by what came back rather than by what was asked for turns 'the server ignores skip' from a silent
    infinite loop into one recorded anomaly.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.divisions, api.page_cap, api.ignore_skip = _divisions(80), 25, True
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    assert len(_commons_ids(cfg)) == 25
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        error = manifest.unit("commons_divisions", "T1", "index")["last_error"] or ""
    finally:
        manifest.close()
    assert "page_repeated" in error
    assert sum(1 for u, _ in api.calls if "commonsvotes" in u and "search" in u) == 2


def test_a_stated_total_stops_the_loop_one_request_early_and_is_cross_checked(tmp_path):
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(150)
    _fetch_divisions(tmp_path, api, page_size=100)
    skips = [p.get("queryParameters.skip") for u, p in api.calls
             if "search/divisions" in u and p.get("queryParameters.house") == "Commons"]
    assert skips == ["0", "100"], "TotalResultCount says 150, so a third request is waste"


def test_verify_fails_when_two_official_sources_disagree_about_divisions(tmp_path):
    """The check that was missing. Every internal signal agreed with itself while 46 of W4's Commons divisions were
    absent; only a second source could say so.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(40)
    api.divisions, api.page_cap = _divisions(40), 25
    cfg = _fetch_divisions(tmp_path, api, page_size=25)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean, [str(p) for p in report.problems]

    api2 = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api2.hansard_divisions = _divisions(40)
    api2.divisions = _divisions(25)  # the truncation, as it actually happened
    cfg2 = _fetch_divisions(tmp_path / "second", api2, page_size=25)
    report2 = verify_window(cfg2, cfg2.window("T1"))
    assert not report2.clean
    kinds = {p.kind for p in report2.problems}
    assert "division_missing_from_votes_api" in kinds
    assert sum(p.kind == "division_missing_from_votes_api" for p in report2.problems) == 15


def test_committee_divisions_are_a_known_gap_not_a_failure(tmp_path):
    """A Public Bill Committee division is minuted in Hansard and is not in the Commons Votes API at all. Counting it
    as missing would cry wolf on every window.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(30)
    api.committee = set(_divisions(30)[25:])
    api.divisions = _divisions(30)[:25]
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean, [str(p) for p in report.problems]
    assert any(g.kind == "committee_divisions" for g in report.known_gaps)


def test_one_division_the_two_sources_date_differently_is_paired_not_double_counted(tmp_path):
    """A deferred division is taken on a later sitting day than the debate it belongs to, and a division after midnight
    belongs to the previous sitting day.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = [("2026-09-02", 360)]
    api.divisions = [("2026-09-01", 360)]
    api.deferred = {("2026-09-01", 360)}
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean, [str(p) for p in report.problems]
    assert any(g.kind == "division_dated_differently" for g in report.known_gaps)
    tally = reconcile_divisions(Cache(cfg.cache_root), cfg.window("T1"))["Commons"]
    assert tally.date_shifted == [("2026-09-02", "2026-09-01", 360)]
    assert not tally.only_hansard and not tally.only_votes


def test_a_few_divisions_hansard_omits_are_noise_but_a_systematic_gap_is_not(tmp_path):
    """Forty of them is a paging bug wearing the same clothes."""
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(40)
    api.divisions = _divisions(42)
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean, [str(p) for p in report.problems]
    assert sum(g.kind == "division_not_in_hansard_list" for g in report.known_gaps) == 2

    api2 = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api2.hansard_divisions = _divisions(40)
    api2.divisions = _divisions(80)  # 40 one-sided rows: not noise
    cfg2 = _fetch_divisions(tmp_path / "second", api2, page_size=100)
    kinds = {p.kind for p in verify_window(cfg2, cfg2.window("T1")).problems}
    assert "divisions_missing_from_hansard_list" in kinds


def test_reconciliation_matches_on_day_and_number_not_on_either_apis_own_id(tmp_path):
    """The two sources number their own rows differently; (day, division number) is the only key both of them carry.
    Checked against the real cache: it matches exactly in W1, W3 and both Houses' Lords sets.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(12)
    api.divisions = _divisions(12)
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    tallies = reconcile_divisions(Cache(cfg.cache_root), cfg.window("T1"))
    assert tallies["Commons"].matched and len(tallies["Commons"].hansard) == 12
    assert {d for d, _ in tallies["Commons"].hansard} == {"2026-09-01", "2026-09-02"}


def test_a_contributions_day_is_paged_by_what_came_back_not_by_what_was_asked_for(tmp_path):
    """The Spoken endpoint does honour take=100, but nothing downstream should depend on that: a cap there would skip
    rows between the pages it did fetch.
    """
    cfg = cfg_for(tmp_path, page_size=100)
    served: list[int] = []

    def handler(request):
        params = dict(request.url.params)
        if "Spoken" not in str(request.url):
            return httpx.Response(404, text="no")
        day = params.get("queryParameters.startDate")
        take, skip = int(params["queryParameters.take"]), int(params["queryParameters.skip"])
        total = 120 if day == "2026-09-01" else 0
        if take > 1:
            served.append(skip)
        n = min(take, 40, max(total - skip, 0))  # a 40-row cap on a 100-row request
        template = FIXTURES["contributions_page"]["Results"][0]
        return httpx.Response(200, json={"TotalResultCount": total, "Results": [
            {**template, "ContributionExtId": f"c{i:04d}", "SittingDate": f"{day}T00:00:00",
             "House": "Commons", "Section": "Commons Chamber", "DebateSectionExtId": "d1",
             "MemberId": 4514} for i in range(skip, skip + n)]})

    run(fetch_window(cfg, cfg.window("T1"), transport=httpx.MockTransport(handler)))
    assert sorted(set(served)) == [0, 40, 80], served
    assert scan_window(Cache(cfg.cache_root), cfg.window("T1")).contributions == 120


# --------------------------------------------------------------------------- One opt-in live test
# ---------------------------------------------------------------------------


@pytest.mark.skipif("not config.getoption('--live-api', default=False)",
                    reason="live network test; run with --live-api")
def test_live_one_day_against_the_real_api(tmp_path):
    """Proof against the real thing, off by default so CI stays network-free."""
    cfg = cfg_for(tmp_path, ONE_WINDOW.replace("2026-09-01", "2026-09-02").replace("end = 2026-09-02",
                                                                                   "end = 2026-09-02"))
    outcome = run(fetch_window(cfg, cfg.window("T1"), sources=["probe", "contributions"]))
    assert outcome.exit_code in (0, 3)
    scan = scan_window(Cache(cfg.cache_root), cfg.window("T1"))
    assert scan.contributions > 500


def test_a_window_before_the_votes_apis_records_begin_is_a_known_gap_not_a_failure(tmp_path):
    """The ten-year plan reaches back to 2015. Where Hansard lists divisions and the Votes API's units complete with
    none at all for the window, the voter lists do not exist in that source: a coverage boundary, recorded for the
    loader, and not a per-division failure that would stop the decade run at its first window.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(30)
    api.divisions = []
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean, [str(p) for p in report.problems]
    gaps = [g for g in report.known_gaps if g.kind == "votes_api_no_coverage"]
    assert len(gaps) == 1 and "30 division(s)" in gaps[0].detail


def test_a_votes_api_that_serves_some_but_not_all_is_still_a_failure(tmp_path):
    """The boundary rule must not swallow a paging bug: SOME divisions from the Votes API means it covers the window.
    """
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = _divisions(30)
    api.divisions = _divisions(1)
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    report = verify_window(cfg, cfg.window("T1"))
    assert not report.clean
    assert not any(g.kind == "votes_api_no_coverage" for g in report.known_gaps)


def test_no_divisions_in_either_source_is_neither_a_failure_nor_a_gap(tmp_path):
    """Spring 2020 in the Lords: both sources empty is a fact about the House, not about the sources."""
    api = FakeAPI(day_totals={"2026-09-01": 0, "2026-09-02": 0})
    api.hansard_divisions = []
    api.divisions = []
    cfg = _fetch_divisions(tmp_path, api, page_size=100)
    report = verify_window(cfg, cfg.window("T1"))
    assert report.clean and not any(g.kind == "votes_api_no_coverage" for g in report.known_gaps)
