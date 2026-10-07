"""Ingest bills, stages and sittings, and link them to debates and divisions."""

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from app.db import pool
from ingest.bills import (
    SQL_DIVISIONS,
    SQL_LINK_DIVISIONS,
    SQL_STAGE_DEBATES,
    StageRef,
    bill_mentioned,
    bill_relevant,
    fetch_bills,
    fetch_recent_bills,
    link_division_stage,
    link_sitting,
    load_stage_refs,
    parse_bill,
    parse_stages,
    store_bill,
)
from ingest.divisions import load_link_context, normalise_title
from ingest.http import PoliteClient
from ingest.runtime import run

DEFAULT_START, DEFAULT_END = date(2026, 8, 10), date(2026, 9, 19)
USER_AGENT = "on-the-record/0.1 (Hansard civic-intelligence research)"
console = Console()


@dataclass
class Report:
    listed: int = 0
    failed: list[int] = field(default_factory=list)
    kept: list[tuple[dict, list[dict], list[dict]]] = field(default_factory=list)
    division_links: dict[str, int] = field(default_factory=dict)
    unmatched_headings: list[str] = field(default_factory=list)
    dry_run: bool = False


def refs_from_parsed(kept) -> list[StageRef]:
    """StageRefs straight from parsed data, for a dry run that must not read what it did not write."""
    refs = []
    for bill, stages, sittings in kept:
        for st in stages:
            mine = [s for s in sittings if s["bill_stage_id"] == st["id"]]
            refs.append(StageRef(st["id"], bill["id"], bill["title"] or "", st["abbreviation"], st["house"],
                                 frozenset(s["date"] for s in mine), frozenset(s["debate_id"] for s in mine if s["debate_id"])))  # fmt: skip
    return refs


async def run_ingest(start: date, end: date, since: date, concurrency: int, min_interval: float, dry_run: bool) -> Report:
    report = Report(dry_run=dry_run)
    await pool.open(wait=True)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers={"User-Agent": USER_AGENT},
                                     limits=httpx.Limits(max_connections=concurrency * 2)) as http:
            client = PoliteClient(http, concurrency=concurrency, min_interval=min_interval)
            with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                          TimeElapsedColumn(), console=console) as progress:  # fmt: skip
                task = progress.add_task("Listing recently updated bills", total=None)
                items = await fetch_recent_bills(client, since)
                report.listed = len(items)
                progress.update(task, description=f"Fetching stages for {len(items)} bills", total=len(items), completed=0)
                fetched, report.failed = await fetch_bills(client, items)
                progress.update(task, completed=len(items))

                async with pool.connection() as conn:
                    debates, _ = await load_link_context(conn)
                    divisions = [
                        dict(zip(("id", "house", "division_date", "title", "debate_id", "is_deferred", "debate_link"), row))
                        for row in await (await conn.execute(SQL_DIVISIONS)).fetchall()
                    ]
                    named = {normalise_title(d.heading) for d in debates} | {normalise_title((d["title"] or "").split(":")[0]) for d in divisions}
                    for item, detail, payload in fetched:
                        bill = parse_bill(item, detail)
                        stages, sittings = parse_stages(bill["id"], payload)
                        if not (bill_relevant(sittings, start, end) or bill_mentioned(bill["title"], named)):
                            continue
                        for s in sittings:
                            s["debate_id"], s["debate_link"] = link_sitting(bill["title"], s["house"], s["date"], debates)
                        report.kept.append((bill, stages, sittings))

                    if not dry_run:
                        storing = progress.add_task("Storing bills, stages and sittings", total=len(report.kept))
                        for bill, stages, sittings in report.kept:
                            await store_bill(conn, bill, stages, sittings)
                            progress.advance(storing)
                        await conn.execute(SQL_STAGE_DEBATES)

                    refs = refs_from_parsed(report.kept) if dry_run else await load_stage_refs(conn)
                    links = [(d["id"], *link_division_stage(d, refs)) for d in divisions]
                    for _, _, method in links:
                        report.division_links[method] = report.division_links.get(method, 0) + 1
                    if not dry_run and links:
                        await conn.execute(SQL_LINK_DIVISIONS, ([i for i, _, _ in links], [s for _, s, _ in links], [m for _, _, m in links]))

                    stored_titles = {normalise_title(b["title"]) for b, _, _ in report.kept}
                    report.unmatched_headings = sorted({
                        d.heading for d in debates
                        if d.heading and d.heading.rstrip().endswith(("Bill", "Bill [HL]", "Bill [Lords]")) and normalise_title(d.heading) not in stored_titles
                    })  # fmt: skip
    finally:
        await pool.close()
    return report


def print_report(r: Report, start: date, end: date) -> None:
    t = Table(title=f"Bills with a sitting in {start} .. {end}" + (" (dry run: nothing written)" if r.dry_run else ""))
    for col in ("Id", "Bill", "House", "Stages", "Sittings in window", "Linked to debate"):
        t.add_column(col, justify="right" if col in ("Id", "Stages", "Sittings in window", "Linked to debate") else "left", no_wrap=True)
    for bill, stages, sittings in sorted(r.kept, key=lambda k: k[0]["title"] or ""):
        inside = [s for s in sittings if start <= s["date"] <= end]
        if inside:
            t.add_row(str(bill["id"]), (bill["title"] or "")[:52], bill["current_house"] or "", str(len(stages)), str(len(inside)),
                      f"{sum(1 for s in inside if s['debate_id'])}/{len(inside)}")  # fmt: skip
    console.print(t)
    s = Table(title="Summary")
    s.add_column("")
    s.add_column("Count", justify="right")
    all_sittings = [x for _, _, ss in r.kept for x in ss]
    in_window = [x for x in all_sittings if start <= x["date"] <= end]
    s.add_row("Recently updated bills listed", f"{r.listed:,}")
    s.add_row("Bills kept (sitting in window, or named in our data)", f"[green]{len(r.kept):,}[/green]")
    s.add_row("Stages / sittings stored", f"{sum(len(st) for _, st, _ in r.kept):,} / {len(all_sittings):,}")
    s.add_row("Window sittings linked to a Hansard debate", f"{sum(1 for x in in_window if x['debate_id'])} of {len(in_window)}")
    for method, n in sorted(r.division_links.items(), key=lambda kv: -kv[1]):
        s.add_row(f"Divisions: {method}", f"{n}")
    if r.failed:
        s.add_row("[red]Bills that failed to fetch[/red]", ", ".join(map(str, r.failed)))
    console.print(s)
    if r.unmatched_headings:
        console.print("[yellow]Debate headings that look like bills but match no stored bill:[/yellow] " + "; ".join(r.unmatched_headings))
        console.print("[dim](draft bills and legislation not yet introduced are not in the Bills API)[/dim]")


def main() -> None:
    # Retired v1 loader: never the rollback, never hansard_v2 (built only from the v2 cache).
    from app import safety
    from app.db import pool as _pool

    safety.refuse_for_v1(_pool.conninfo)
    parser = argparse.ArgumentParser(description="Ingest bills, stages and sittings; link them to debates and divisions.")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--since", type=date.fromisoformat, default=None,
                        help="list bills updated on/after this date (default: 30 days before --start)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--min-interval", type=float, default=0.1)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not be after --end")
    since = args.since or args.start - timedelta(days=30)
    print_report(run(run_ingest(args.start, args.end, since, args.concurrency, args.min_interval, args.dry_run)), args.start, args.end)


if __name__ == "__main__":
    main()
