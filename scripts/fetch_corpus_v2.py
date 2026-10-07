"""Corpus v2 fetch: download a window of Hansard into the local cache. Never touches the database."""

import argparse
import logging
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ingest.corpus_fetch import (
    EXIT_ERROR,
    EXIT_FAILURES,
    EXIT_OK,
    EXIT_USAGE,
    Cache,
    ConfigError,
    Manifest,
    StopFlag,
    WindowSpec,
    clear_failed,
    estimate_runtime,
    failed_units,
    fetch_window,
    load_config,
    plan_windows,
    setup_logging,
    verify_window,
    write_report,
)
from ingest.runtime import run

DEFAULT_CONFIG = "config/corpus_v2.toml"
# A minimum width so the plan table is never silently truncated to ellipses in a narrow window or when the output is
# piped to a file (rich defaults to 80 columns off a tty).
console = Console(width=max(104, Console().width))


# --------------------------------------------------------------------------- Progress
# ---------------------------------------------------------------------------


class Line:
    """One progress line per few seconds: window, pass, x of y, requests, retries, failures, ETA."""

    def __init__(self, every: float = 5.0) -> None:
        self._last = 0.0
        self._every = every

    def __call__(self, progress) -> None:
        import time

        now = time.monotonic()
        finished = progress.done >= progress.total
        if not finished and now - self._last < self._every:
            return
        self._last = now
        eta = progress.eta_seconds()
        console.print(
            f"[{progress.window_id}] {progress.source:<17} {progress.done:>5}/{progress.total:<5} "
            f"skip {progress.skipped:<5} fail {progress.failures:<3} "
            f"req {progress.requests:<6} retry {progress.retries:<4} "
            f"eta {_hms(eta) if eta else '--:--:--'}",
            highlight=False, soft_wrap=True,
        )


def _hms(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


# --------------------------------------------------------------------------- plan
# ---------------------------------------------------------------------------


def cmd_plan(args, cfg) -> int:
    windows = [cfg.window(w) for w in args.window] if args.window else list(cfg.windows)
    if not args.include_inactive and not args.window:
        # Candidates are reported too: choosing W1 is the whole point of this command.
        windows = [w for w in windows if w.active or w.role == "candidate"]
    console.print(f"[bold]Planning {len(windows)} window(s)[/bold]: {', '.join(w.id for w in windows)}")
    console.print("One take=1 probe per calendar day, plus the month calendars. Both are cached, "
                  "so `fetch` reuses every one of them.\n")
    plans, client = run(plan_windows(cfg, windows, on_progress=Line()))
    p50, p95 = client.latency_quantiles()

    table = Table(title="Corpus v2 plan")
    for col, just in (("Window", "left"), ("Dates", "left"), ("Com/Lds", "right"),
                      ("H-days", "right"), ("Sit", "right"), ("Contribs", "right"),
                      ("Requests", "right"), ("Chunks", "right"), ("Cache", "right")):
        table.add_column(col, justify=just, no_wrap=True)
    for plan in plans:
        est = plan.estimate(cfg)
        active = plan.window.active
        w = plan.window
        table.add_row(
            f"{w.id} ({w.role})" if active else f"{w.id} ({w.role}, not fetched)",
            f"{w.start:%Y-%m-%d} to {w.end:%m-%d}",
            f"{plan.commons_days}/{plan.lords_days}", str(plan.house_days),
            str(plan.sitting_days), f"{plan.listed:,}", f"{est['requests']:,.0f}",
            f"{est['chunks']:,.0f}", _gb(est["cache_bytes"]),
            style=None if active else "dim",
        )
    console.print()
    console.print(table)

    console.print()
    console.print(f"Measured request latency this run: p50 {p50:.2f}s, p95 {p95:.2f}s over "
                  f"{len(client.latencies):,} requests at concurrency {cfg.http.concurrency}. "
                  "Probes are small; full pages are slower, so treat the estimate as a floor.")
    active_plans = [p for p in plans if p.window.active]
    if not active_plans:
        console.print("[yellow]No ACTIVE window in this plan[/yellow], so there is nothing to "
                      "fetch from it. Re-run without --window, or set active = true in the config.")
    else:
        low, high = estimate_runtime(active_plans, cfg, p50, p95)
        console.print(f"[bold]Active windows total[/bold]: "
                      f"{sum(p.listed for p in active_plans):,} contributions, "
                      f"{sum(p.estimate(cfg)['requests'] for p in active_plans):,.0f} requests, "
                      f"~{sum(p.estimate(cfg)['chunks'] for p in active_plans):,.0f} chunks, "
                      f"~{_gb(sum(p.estimate(cfg)['cache_bytes'] for p in active_plans))} of cache.")
        console.print(f"[bold]Estimated fetch time[/bold]: {_hms(low)} to {_hms(high)} in total "
                      "(an interval from the measured p50 and p95, not a mean). Per window:")
        for p in active_plans:
            lo, hi = estimate_runtime([p], cfg, p50, p95)
            console.print(f"    {p.window.id:<4} {_hms(lo)} to {_hms(hi)}")

    disagreements = [(p.window.id, d) for p in plans for d in p.disagreements]
    if disagreements:
        console.print()
        console.print(f"[yellow]{len(disagreements)} calendar/contributions disagreement(s)[/yellow] "
                      "(a day one signal calls sitting and the other does not). These are fetched "
                      "anyway and recorded; they are not errors:")
        for wid, detail in disagreements[:25]:
            console.print(f"    {wid}: {detail}")
        if len(disagreements) > 25:
            console.print(f"    ... and {len(disagreements) - 25} more")

    candidates = [p for p in plans if p.window.role == "candidate"]
    if candidates:
        console.print()
        console.print("[bold]W1 alternatives, for your decision[/bold] (none of these are fetched):")
        chosen = next((p for p in plans if p.window.id == "W1"), None)
        for p in candidates:
            console.print(f"    {p.window.id:<12} {p.window.start}..{p.window.end}  "
                          f"{p.sitting_days:>3} sitting days, {p.house_days:>3} house-days, "
                          f"{p.listed:>7,} contributions")
        if chosen:
            console.print(f"    {'W1 (chosen)':<12} {chosen.window.start}..{chosen.window.end}  "
                          f"{chosen.sitting_days:>3} sitting days, {chosen.house_days:>3} house-days, "
                          f"{chosen.listed:>7,} contributions")
        console.print("    To change the choice, edit config/corpus_v2.toml and re-run `plan`.")
    return EXIT_OK


# --------------------------------------------------------------------------- fetch
# ---------------------------------------------------------------------------


def cmd_fetch(args, cfg) -> int:
    window = cfg.window(args.window)
    if args.dry_run:
        return _dry_run(cfg, window, args)
    stop = StopFlag()
    stop.install()
    console.print(f"[bold]Fetching {window.id}[/bold] {window.start} .. {window.end} "
                  f"({window.role}) into {cfg.cache_root.name}/")
    if window.role == "heldout":
        console.print("[yellow]W3 is HELD OUT.[/yellow] Ingest data-quality checks are allowed; "
                      "its retrieval and evaluation results are not to be looked at until the "
                      "final generalisation report.")
    console.print("Ctrl+C stops cleanly and saves state. Re-run the same command to resume.\n")

    try:
        outcome = run(fetch_window(cfg, window, sources=args.source or None, force=args.force,
                                   on_progress=Line(), limit=args.limit_units, stop=stop))
    except KeyboardInterrupt:
        console.print("\n[red]Aborted on the second Ctrl+C.[/red] "
                      "The manifest may be a few units behind the cache; run `verify` before resuming.")
        return 130
    finally:
        stop.restore()

    table = Table(title=f"{window.id}: fetch finished", show_header=False)
    table.add_column("")
    table.add_column("", justify="right")
    table.add_row("Units fetched", f"{outcome.units_done:,}")
    table.add_row("Units skipped (already done)", f"{outcome.units_skipped:,}")
    table.add_row("Units failed", f"[red]{outcome.units_failed:,}[/red]" if outcome.units_failed
                  else f"{outcome.units_failed:,}")
    table.add_row("HTTP requests", f"{outcome.requests:,}")
    table.add_row("Retries", f"{outcome.retries:,}")
    table.add_row("Rate-limited (429)", f"{outcome.rate_limited:,}")
    table.add_row("Bytes written", _gb(outcome.bytes_written))
    table.add_row("Elapsed", _hms(outcome.elapsed))
    console.print()
    console.print(table)
    if outcome.message:
        console.print(f"\n{outcome.message}")
    if outcome.exit_code != EXIT_OK:
        console.print(f"\n[bold]Resume with:[/bold] "
                      f"uv run python scripts/fetch_corpus_v2.py fetch --window {window.id}")
    else:
        console.print(f"\nNow run: uv run python scripts/fetch_corpus_v2.py verify --window {window.id}")
    return outcome.exit_code


def _dry_run(cfg, window: WindowSpec, args) -> int:
    """List the units a fetch would do, without a single request."""
    from ingest.corpus_fetch import Context, DiskGuard, Breaker, already_complete, passes_for, plan_units

    cache = Cache(cfg.cache_root)
    # Read-only, and only when a manifest exists: a dry run must never take a write lock on a manifest another window's
    # fetch is using, and must work before anything is fetched.
    fresh = not cfg.manifest_path.exists()
    manifest = Manifest(cfg.manifest_path, read_only=True) if not fresh else None
    ctx = Context(cfg=cfg, window=window, cache=cache, client=None, writer=None,  # type: ignore[arg-type]
                  manifest=manifest, stop=StopFlag(), breaker=Breaker(cfg.breaker),  # type: ignore[arg-type]
                  disk=DiskGuard(cfg.cache_root, cfg.disk), force=args.force)
    table = Table(title=f"{window.id}: dry run (no requests made)")
    table.add_column("Pass")
    table.add_column("Units", justify="right")
    table.add_column("Already done", justify="right")
    table.add_column("To fetch", justify="right")
    total = 0
    try:
        for source in passes_for(args.source or cfg.sources):
            try:
                units = plan_units(ctx, source)
            except ConfigError as exc:
                table.add_row(source, "?", "?", f"[yellow]{exc}[/yellow]")
                continue
            done = 0 if fresh else sum(1 for u in units if already_complete(ctx, u))
            total += len(units) - done
            table.add_row(source, f"{len(units):,}", f"{done:,}", f"{len(units) - done:,}")
    finally:
        if manifest is not None:
            manifest.close()
    console.print(table)
    console.print(f"\n{total:,} unit(s) would be fetched. Sources that depend on an earlier pass "
                  "(commons_detail, bill, members) show 0 until that pass has run.")
    return EXIT_OK


# --------------------------------------------------------------------------- verify / status / report
# ---------------------------------------------------------------------------


def cmd_verify(args, cfg) -> int:
    windows = [cfg.window(w) for w in args.window] if args.window else list(cfg.active_windows)
    worst = EXIT_OK
    for window in windows:
        report = verify_window(cfg, window)
        console.print(f"\n[bold]{window.id}[/bold]: {report.checked_units:,} units checked")
        if report.scan:
            console.print(f"  cache holds {report.scan.contributions:,} contributions over "
                          f"{len(report.scan.per_day)} days, {len(report.scan.debate_ids):,} debates, "
                          f"{len(report.scan.member_ids):,} distinct members")
        if report.known_gaps:
            kinds = {}
            for p in report.known_gaps:
                kinds[p.kind] = kinds.get(p.kind, 0) + 1
            console.print(f"  [yellow]known gaps[/yellow] (the source's, not ours): "
                          + ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items())))
            for p in report.known_gaps[:10]:
                console.print(f"    {p}")
            if len(report.known_gaps) > 10:
                console.print(f"    ... and {len(report.known_gaps) - 10} more")
        if report.clean:
            console.print("  [green]clean[/green]: every unit accounted for, every hash matches.")
            continue
        worst = EXIT_FAILURES
        kinds = {}
        for p in report.problems:
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        console.print(f"  [red]{len(report.problems)} problem(s)[/red]: "
                      + ", ".join(f"{k} x{n}" for k, n in sorted(kinds.items())))
        for p in report.problems[:30]:
            console.print(f"    {p}")
        if len(report.problems) > 30:
            console.print(f"    ... and {len(report.problems) - 30} more")
        if args.repair:
            n = clear_failed(cfg, window.id)
            console.print(f"  cleared {n} failed/pending manifest row(s); "
                          f"re-run `fetch --window {window.id}` to re-fetch them.")
    if worst != EXIT_OK and not args.repair:
        console.print("\nRe-run `fetch` for the affected window to retry, or add --repair here "
                      "to clear failed rows first.")
    return worst


def cmd_status(args, cfg) -> int:
    if not cfg.manifest_path.exists():
        console.print(f"No manifest yet at {cfg.manifest_path.relative_to(cfg.cache_root.parent.parent)}. "
                      "Nothing has been fetched.")
        return EXIT_OK
    manifest = Manifest(cfg.manifest_path, read_only=True)
    try:
        table = Table(title="Corpus v2 status")
        table.add_column("Window")
        table.add_column("Source")
        for status in ("done", "not_sitting", "known_gap", "failed", "pending"):
            table.add_column(status, justify="right")
        rows = manifest.conn.execute(
            "SELECT window_id, source, status, count(*) n FROM unit GROUP BY window_id, source, status"
        ).fetchall()
        grouped: dict[tuple[str, str], dict[str, int]] = {}
        for r in rows:
            grouped.setdefault((r["window_id"], r["source"]), {})[r["status"]] = r["n"]
        for (wid, source), counts in sorted(grouped.items()):
            table.add_row(wid, source, *[
                (f"[red]{counts.get(s, 0)}[/red]" if s in ("failed", "pending") and counts.get(s)
                 else str(counts.get(s, 0)))
                for s in ("done", "not_sitting", "known_gap", "failed", "pending")])
        console.print(table)

        runs = manifest.conn.execute(
            "SELECT * FROM run ORDER BY id DESC LIMIT 8").fetchall()
        if runs:
            rt = Table(title="Recent runs")
            for c in ("id", "window", "command", "started", "ended", "requests", "retries", "failures", "outcome"):
                rt.add_column(c)
            for r in runs:
                rt.add_row(str(r["id"]), r["window_id"] or "-", r["command"], (r["started_at"] or "")[:19],
                           (r["ended_at"] or "running")[:19], f"{r['requests']:,}", f"{r['retries']:,}",
                           f"{r['failures']:,}", r["outcome"] or "-")
            console.print(rt)
        cache = Cache(cfg.cache_root)
        console.print(f"Cache on disk: {_gb(cache.total_bytes())}")
        stuck = failed_units(cfg)
        if stuck:
            console.print(f"[yellow]{len(stuck)} failed/pending unit(s).[/yellow] "
                          "Re-run `fetch` for the window to retry them.")
    finally:
        manifest.close()
    return EXIT_OK


def cmd_report(args, cfg) -> int:
    windows = [cfg.window(w) for w in args.window] if args.window else list(cfg.active_windows)
    for window in windows:
        path, data = write_report(cfg, window)
        console.print(f"\n[bold]{window.id}[/bold] {window.start} .. {window.end} ({window.role})")
        console.print(f"  sitting days   Commons {data['sitting_days']['Commons']}, "
                      f"Lords {data['sitting_days']['Lords']} "
                      f"(calendar says {data['calendar_days']['Commons']} / {data['calendar_days']['Lords']})")
        console.print(f"  contributions  {data['contributions']:,} over {data['days_with_contributions']} days, "
                      f"{data['debates']:,} debates")
        console.print(f"  divisions      Commons {data['divisions']['commons_listed']} "
                      f"({data['divisions']['commons_detail_cached']} detailed), "
                      f"Lords {data['divisions']['lords_listed']}, "
                      f"Hansard lists {data['divisions']['hansard_listed']}")
        console.print(f"  bills          {data['bills']['distinct_bills']} bills, "
                      f"{data['bills']['sittings']} sittings in window")
        console.print(f"  members        {data['members']['seen']:,} seen, "
                      f"{data['members']['new']:,} new to us")
        console.print(f"  est. chunks    {data['estimated_chunks']:,}")
        console.print(f"  cache          {_gb(data['cache_bytes'])}")
        console.print("  section census:")
        for key, n in list(data["section_census"].items())[:12]:
            console.print(f"      {key:<34} {n:>8,}")
        if data["day_anomalies"]:
            console.print(f"  day anomalies: {data['day_anomalies']}")
        if data["unreadable_pages"]:
            console.print(f"  [red]unreadable pages: {len(data['unreadable_pages'])}[/red]")
        console.print("  schema fingerprints:")
        for source, prints in data["schema_fingerprints"].items():
            flag = "[red] DRIFT[/red]" if len(prints) > 1 else ""
            console.print(f"      {source:<20} {len(prints)} distinct{flag}")
        console.print(f"  written to {path.parent.name}/{path.name}")
    return EXIT_OK


# --------------------------------------------------------------------------- smoke: the one live check with an oracle
# we already trust ---------------------------------------------------------------------------


def cmd_smoke(args, cfg) -> int:
    """Fetch a day we already hold into a THROWAWAY cache and compare ids with the database."""
    day = date.fromisoformat(args.date)
    temp_root = Path(tempfile.mkdtemp(prefix="corpus_v2_smoke_"))
    window = WindowSpec(id="SMOKE", start=day, end=day, role="dev", active=True, note="")
    smoke_cfg = type(cfg)(**{**cfg.__dict__, "cache_root": temp_root})
    console.print(f"[bold]Smoke test[/bold]: fetching {day} into a throwaway cache, then comparing "
                  "its contribution ids with speech.id in the database.")
    try:
        outcome = run(fetch_window(smoke_cfg, window, sources=["probe", "contributions"]))
        fetched = set()
        base = temp_root / "contributions" / "SMOKE" / day.isoformat()
        cache = Cache(temp_root)
        for page in sorted(base.glob("*.json.gz")) if base.is_dir() else []:
            payload = cache.read_json(str(page.relative_to(temp_root)).replace("\\", "/"))
            for item in (payload.get("Results") or []) if isinstance(payload, dict) else []:
                if item.get("ContributionExtId"):
                    fetched.add(item["ContributionExtId"])
        console.print(f"  fetched {len(fetched):,} distinct contributions in "
                      f"{outcome.requests} requests, {_hms(outcome.elapsed)}")

        stored = _stored_speech_ids(day)
        if stored is None:
            console.print("[yellow]  Could not read the database, so the comparison was NOT made.[/yellow] "
                          "Start Postgres and re-run to complete the smoke test.")
            return EXIT_FAILURES
        console.print(f"  database holds {len(stored):,} speeches for {day}")
        missing = stored - fetched
        extra = fetched - stored
        console.print(f"  in the database but not fetched: {len(missing)}")
        console.print(f"  fetched but not in the database: {len(extra)} "
                      "(expected: the API lists a few it would not serve at ingest time)")
        if missing:
            console.print(f"[red]  FAIL[/red]: {len(missing)} stored speech(es) the fetch did not find: "
                          f"{sorted(missing)[:5]}")
            return EXIT_FAILURES
        console.print("[green]  PASS[/green]: every speech the database holds for this day was fetched.")
        return EXIT_OK
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        console.print("  throwaway cache deleted.")


def _stored_speech_ids(day: date) -> set[str] | None:
    try:
        import psycopg

        from app import safety
        from app.config import settings

        # working_dsn: PG_DSN named the rollback, so this compared a v2 fetch with the v1 corpus's speeches.
        with psycopg.connect(safety.working_dsn(settings.pg_dsn), connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT s.id FROM speech s JOIN debate d ON d.id = s.debate_id WHERE d.date = %s",
                        (day.isoformat(),))
            return {r[0] for r in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001 - the database is optional here, by design
        console.print(f"  (database unavailable: {type(exc).__name__})")
        return None


# --------------------------------------------------------------------------- main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="window config (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log INFO to the console too")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="enumerate sitting days and estimate the fetch; cheap")
    p.add_argument("--window", action="append", help="limit to this window (repeatable)")
    p.add_argument("--include-inactive", action="store_true", help="also plan windows marked inactive")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("fetch", help="fetch one window into the cache")
    p.add_argument("--window", required=True)
    p.add_argument("--source", action="append", help="limit to this source (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="list the units, make no requests")
    p.add_argument("--force", action="store_true", help="re-fetch units already marked done")
    p.add_argument("--limit-units", type=int, help="stop each pass after N units (for a trial run)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("verify", help="check the manifest against the cache; makes no requests")
    p.add_argument("--window", action="append")
    p.add_argument("--repair", action="store_true", help="clear failed rows so the next fetch retries them")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("status", help="what has been fetched so far")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("report", help="the per-window report, printed and written to reports/")
    p.add_argument("--window", action="append")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("smoke", help="fetch one day we already hold and check it against the database")
    p.add_argument("--date", default="2026-09-02")
    p.set_defaults(func=cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return EXIT_USAGE
    log_path = setup_logging(cfg, console_level=logging.INFO if args.verbose else logging.WARNING)
    try:
        code = args.func(args, cfg)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130
    except Exception:
        console.print_exception()
        console.print(f"\nFull traceback in {log_path.parent.name}/{log_path.name}")
        return EXIT_ERROR
    return code


if __name__ == "__main__":
    sys.exit(main())
