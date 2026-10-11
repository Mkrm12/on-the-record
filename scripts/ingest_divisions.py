"""Ingest Commons and Lords divisions and every individual vote for a date window."""

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import date

import httpx
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from app.db import pool
from ingest.divisions import (
    Parsed,
    fetch_commons,
    fetch_lords,
    link_debate,
    load_link_context,
    parse_commons,
    parse_lords,
    store,
)
from ingest.http import PoliteClient
from ingest.runtime import run

DEFAULT_START = date(2026, 8, 10)
DEFAULT_END = date(2026, 9, 19)
USER_AGENT = "on-the-record/0.1 (Hansard civic-intelligence research)"

SQL_UNKNOWN_VOTERS = """
    SELECT count(DISTINCT v.mnis_id) FROM division_vote v
    LEFT JOIN member_id_map m ON m.mnis_id = v.mnis_id WHERE m.mnis_id IS NULL
"""

console = Console()


@dataclass
class Report:
    parsed: list[Parsed] = field(default_factory=list)
    failed_commons: list[int] = field(default_factory=list)
    unknown_voters: int | None = None
    dry_run: bool = False


async def run_ingest(start: date, end: date, include_lords: bool, concurrency: int, min_interval: float, dry_run: bool) -> Report:
    report = Report(dry_run=dry_run)
    await pool.open(wait=True)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=concurrency * 2),
        ) as http:
            client = PoliteClient(http, concurrency=concurrency, min_interval=min_interval)
            with Progress(
                SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                TimeElapsedColumn(), console=console,
            ) as progress:  # fmt: skip
                fetching = progress.add_task("Fetching Commons + Lords divisions", total=None)
                commons, lords = await asyncio.gather(
                    fetch_commons(client, start, end),
                    fetch_lords(client, start, end) if include_lords else _none(),
                )
                details, report.failed_commons = commons
                progress.update(fetching, total=1, completed=1)

                parsed = [parse_commons(d) for d in details] + [parse_lords(d) for d in lords]
                parsed.sort(key=lambda p: (p.division["division_date"], p.division["house"], p.division["number"] or 0))

                async with pool.connection() as conn:
                    debates, tallies = await load_link_context(conn)
                    for p in parsed:
                        p.division["debate_id"], p.division["debate_link"] = link_debate(p.division, debates, tallies)

                    if not dry_run:
                        storing = progress.add_task("Storing divisions and votes", total=len(parsed))
                        for p in parsed:
                            await store(conn, p)
                            progress.advance(storing)
                        report.unknown_voters = (await (await conn.execute(SQL_UNKNOWN_VOTERS)).fetchone())[0]
                report.parsed = parsed
    finally:
        await pool.close()
    return report


async def _none() -> list:
    return []


def print_report(report: Report) -> None:
    detail = Table(title="Divisions" + (" (dry run: nothing written)" if report.dry_run else ""))
    for col in ("Date", "House", "No.", "Title", "Official", "Listed", "OK", "Debate link"):
        detail.add_column(col, justify="right" if col in ("No.", "Official", "Listed") else "left", no_wrap=True)
    for p in report.parsed:
        d = p.division
        style = "green" if d["tally_reconciled"] else "yellow"
        detail.add_row(
            str(d["division_date"]), d["house"], str(d["number"]), (d["title"] or "")[:38],
            f"{d['ayes']}/{d['noes']}", f"{d['ayes_listed']}/{d['noes_listed']}",
            f"[{style}]{'yes' if d['tally_reconciled'] else 'NO'}[/{style}]",
            f"{d['debate_link']}" + ("" if d["debate_id"] else " [red](unlinked)[/red]"),
        )  # fmt: skip
    console.print(detail)
    flagged = [p.division for p in report.parsed if p.division["anomalies"]]
    if flagged:
        console.print("[bold]Anomalies[/bold]")
        for d in flagged:
            console.print(f"  {d['division_date']} {d['house']} #{d['number']} ({(d['title'] or '')[:40]}): {'; '.join(d['anomalies'])}", soft_wrap=True)

    divisions = [p.division for p in report.parsed]
    votes = [v for p in report.parsed for v in p.votes]
    summary = Table(title="Summary")
    summary.add_column("")
    summary.add_column("Count", justify="right")
    for house in ("Commons", "Lords"):
        summary.add_row(f"{house} divisions", str(sum(1 for d in divisions if d["house"] == house)))
    summary.add_row("Votes stored (rows)", f"{len(votes):,}")
    summary.add_row("  aye / no / no_vote", " / ".join(f"{sum(1 for v in votes if v['lobby'] == k):,}" for k in ("aye", "no", "no_vote")))
    summary.add_row("  tellers", f"{sum(1 for v in votes if v['is_teller']):,}")
    summary.add_row("  [yellow]conflicting (listed in both lobbies)[/yellow]", str(sum(1 for v in votes if v["lobby"] == "conflict")))
    summary.add_row("Divisions whose official tally != listed voters", f"[yellow]{sum(1 for d in divisions if d['tally_reconciled'] is False)}[/yellow]")
    summary.add_row("Divisions linked to a debate", f"{sum(1 for d in divisions if d['debate_id'])} of {len(divisions)}")
    if report.failed_commons:
        summary.add_row("[red]Commons divisions that failed to fetch[/red]", ", ".join(map(str, report.failed_commons)))
    if report.unknown_voters is not None:
        summary.add_row("Distinct voters with no member_id_map row", str(report.unknown_voters))
    console.print(summary)
    if report.unknown_voters:
        console.print("[yellow]Run `uv run python scripts/ingest_members.py` to fetch metadata for the new voters.[/yellow]")


def main() -> None:
    # Retired v1 loader: never the rollback, never hansard_v2 (built only from the v2 cache).
    from app import safety
    from app.db import pool as _pool

    safety.refuse_for_v1(_pool.conninfo)
    parser = argparse.ArgumentParser(description="Ingest Commons and Lords divisions and votes.")
    parser.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START, help="first date (default: %(default)s)")
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END, help="last date (default: %(default)s)")
    parser.add_argument("--no-lords", action="store_true", help="Commons only")
    parser.add_argument("--dry-run", action="store_true", help="fetch, parse, link and report; write nothing")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--min-interval", type=float, default=0.1, help="min seconds between request starts")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must not be after --end")
    print_report(run(run_ingest(args.start, args.end, not args.no_lords, args.concurrency, args.min_interval, args.dry_run)))


if __name__ == "__main__":
    main()
