"""Load windows from the Corpus v2 cache into `hansard_v2`. Offline: no network, ever."""

import argparse
import json
import sys
from datetime import date

from psycopg.conninfo import make_conninfo
from rich.console import Console
from rich.table import Table

from app import safety
from app.config import settings
from app.db import make_pool
from ingest.corpus_cache import CorpusCache
from ingest.corpus_fetch import load_config
from ingest.load import (
    WindowLoad,
    finish_window,
    load_bills,
    load_members,
    load_window,
    sync_chunk_filters,
)
from ingest.runtime import run

console = Console()
DEFAULT_CONFIG = "config/corpus_v2.toml"


async def _run(args) -> int:
    cfg = load_config(args.config, today=date.today())
    windows = ([cfg.window(w) for w in args.window] if args.window
               else list(cfg.active_windows) if args.all else [])
    if args.bills_only and not windows:
        console.print("[red]--bills-only needs --window or --all: bills are found by their sittings' dates[/red]")
        return 2
    if not windows and not (args.members_only or args.sync_only):
        console.print("[red]name at least one --window, or --all[/red]")
        return 2

    dsn = make_conninfo(settings.pg_dsn, dbname=args.database) if args.database else safety.working_dsn(settings.pg_dsn)
    try:
        safety.require(dsn, args.expect)
    except safety.WrongDatabase as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    cache = CorpusCache(cfg.cache_root)
    if not cache.root.is_dir():
        console.print(f"[red]no cache at {cfg.cache_root}; run the fetch first[/red]")
        return 2

    loads: list[WindowLoad] = []
    pool = make_pool(dsn, min_size=1, max_size=2, served=False)  # served=False: ingest writes the whole record
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            if args.bills_only and not args.dry_run:
                # A bill's record changes after we fetch it -- it is renamed at Royal Assent, stages are added -- and a
                # parsing fix (titles stripped) must reach rows already stored.
                from ingest import bills as bills_mod

                for window in windows:
                    load = WindowLoad(window_id=window.id)
                    await load_bills(conn, cache, window, load)
                    console.print(f"[bold]{window.id}[/bold] bills {load.bills:,}, stages {load.bill_stages:,}, "
                                  f"sittings {load.bill_sittings:,}")
                links = await bills_mod.link_all(conn)
                console.print(f"relinked: {json.dumps(links, default=str)}")
                return 0
            for window in (() if args.sync_only or args.members_only else windows):
                console.print(f"[bold]{window.id}[/bold] {window.start} .. {window.end} ({window.role})")
                if args.dry_run:
                    days = cache.days(window)
                    served = sum(len(cache.day_items(window, d).items) for d in days)
                    console.print(f"  dry run: {len(days)} day(s), {served:,} contributions in the cache")
                    continue
                load = await load_window(conn, cache, window, cfg.version,
                                         with_divisions=not args.skip_divisions,
                                         with_bills=not args.skip_bills)
                await finish_window(conn, window)
                loads.append(load)
                _print_window(load)
            if (args.members or args.members_only) and not args.dry_run and not args.sync_only:
                stats = await load_members(conn, cache)
                console.print(f"members: {stats['seen']:,} in the cache, {stats['stubs']:,} stub(s), "
                              f"{stats['changed']:,} row(s) changed")
            if args.sync_filters and not args.dry_run:
                # After chunking, never before it: this copies window, date, house, section and speaker down onto
                # `chunk` so a filtered vector search has its predicate on the indexed table (migration 005).
                changed = await sync_chunk_filters(conn)
                console.print(f"chunk filter columns: {changed:,} row(s) updated")
    finally:
        await pool.close()

    if args.json and loads:
        print(json.dumps([load.as_dict() for load in loads], indent=2, default=str))
    return verdict(loads, accept_anomalies=args.accept_anomalies)


# Windows whose over-ceiling loss the maintainer has inspected and accepted, comma-separated — how a pipeline step
# (which runs this script itself) is told, one window at a time: `$env:LOAD_ACCEPT_ANOMALIES = "C2016Q1"`.
ACCEPT_ENV = "LOAD_ACCEPT_ANOMALIES"


def verdict(loads: list[WindowLoad], *, accept_anomalies: bool = False) -> int:
    """1 when any window lost more than a stray record's worth (`ingest.load.LOAD_CEILINGS`), so the decade pipeline
    stops at that window instead of loading on with its records missing. The load itself is idempotent and already
    committed: after looking, `--accept-anomalies` (this run) or `LOAD_ACCEPT_ANOMALIES` (named windows) records the
    decision and returns 0.
    """
    import os

    accepted = {w.strip() for w in os.environ.get(ACCEPT_ENV, "").split(",") if w.strip()}
    problems = [(load.window_id, p) for load in loads for p in load.problems()]
    for window_id, problem in problems:
        note = " (accepted)" if accept_anomalies or window_id in accepted else ""
        console.print(f"[bold red]{window_id}: {problem}{note}[/bold red]")
    if any(not accept_anomalies and window_id not in accepted for window_id, _ in problems):
        console.print(f"[red]the load lost more than stray records; inspect them (--json lists the unplaceable ids), "
                      f"then re-run with --accept-anomalies, or set {ACCEPT_ENV} to the window ids, if the loss is "
                      f"understood[/red]")
        return 1
    return 0


def _print_window(load: WindowLoad) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("  days", f"{load.days:,}")
    table.add_row("  debates", f"{load.debates:,}")
    table.add_row("  speeches", f"{load.speeches:,} of {load.listed:,} listed")
    if load.unserved:
        table.add_row("  listed but not served", f"[yellow]{load.unserved:,}[/yellow]")
    if load.unplaceable:
        table.add_row("  could not be placed", f"[yellow]{len(load.unplaceable):,}[/yellow]")
    for reason, count in sorted(load.quarantined.items()):
        table.add_row(f"  quarantined: {reason.replace('_', ' ')}", f"[yellow]{count:,}[/yellow]")
    table.add_row("  corrections", f"{load.corrections:,}")
    table.add_row("  divisions / votes", f"{load.divisions:,} / {load.division_votes:,}")
    table.add_row("  bills / stages / sittings",
                  f"{load.bills:,} / {load.bill_stages:,} / {load.bill_sittings:,}")
    table.add_row("  rows changed", ", ".join(f"{k} {v:,}" for k, v in sorted(load.rows_changed.items())) or "0")
    console.print(table)
    if load.division_links:
        console.print("  division links: " +
                      ", ".join(f"{k}={v}" for k, v in load.division_links.most_common()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--window", action="append", help="window id (repeatable)")
    p.add_argument("--all", action="store_true", help="every active window, in config order")
    p.add_argument("--members", action="store_true", default=True,
                   help="also load members (default on; they are shared across windows)")
    p.add_argument("--no-members", dest="members", action="store_false")
    p.add_argument("--members-only", action="store_true")
    p.add_argument("--bills-only", action="store_true",
                   help="re-derive the named windows' bills from the cache, then relink sittings and divisions")
    p.add_argument("--sync-filters", action="store_true", default=True,
                   help="copy window/date/house/section/speaker onto chunk (default on; run after chunking)")
    p.add_argument("--no-sync-filters", dest="sync_filters", action="store_false")
    p.add_argument("--sync-only", action="store_true", help="only sync the chunk filter columns")
    p.add_argument("--skip-divisions", action="store_true")
    p.add_argument("--skip-bills", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--accept-anomalies", action="store_true",
                   help="exit 0 even when a window lost more than the load ceilings allow (after inspecting it)")
    p.add_argument("--json", action="store_true", help="print the per-window load as JSON")
    p.add_argument("--database", help="override the database name in PG_DSN")
    p.add_argument("--expect", default=safety.TARGET,
                   help="the database this run is allowed to write to (default: %(default)s)")
    sys.exit(run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
