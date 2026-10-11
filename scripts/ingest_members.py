"""Map every speaker's MNIS ID to full member metadata."""

import argparse
import asyncio
from dataclasses import dataclass, field

import httpx
from psycopg.types.json import Jsonb
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from app.db import pool
from ingest.members import COLUMNS, FetchError, MembersClient, parse_member
from ingest.runtime import run

USER_AGENT = "on-the-record/0.1 (Hansard civic-intelligence research)"

# Everyone who spoke OR voted: a member who only votes still needs a row for "how did X vote" queries. mode picks the
# name most often used for the ID: the fallback for a stub row.
SQL_TARGETS = """
    WITH ids AS (
        SELECT mnis_id, speaker_name AS name FROM speech WHERE mnis_id IS NOT NULL
        UNION ALL
        SELECT mnis_id, member_name FROM division_vote
    )
    SELECT i.mnis_id, mode() WITHIN GROUP (ORDER BY i.name)
    FROM ids i
    LEFT JOIN member_id_map m ON m.mnis_id = i.mnis_id
    WHERE %(refresh)s OR m.mnis_id IS NULL OR cardinality(m.missing_endpoints) > 0
    GROUP BY i.mnis_id
    ORDER BY i.mnis_id
    LIMIT %(limit)s
"""
SQL_UPSERT = f"""
    INSERT INTO member_id_map (mnis_id, {", ".join(COLUMNS)}, fetched_at)
    VALUES (%(mnis_id)s, {", ".join(f"%({c})s" for c in COLUMNS)}, now())
    ON CONFLICT (mnis_id) DO UPDATE SET
        {", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS)},
        fetched_at = now()
"""

console = Console()


@dataclass
class Stats:
    complete: int = 0
    partial: int = 0  # stored, but some endpoints failed after retries
    stubs: int = 0  # API returned 404 for the member
    failed: list[tuple[int, str]] = field(default_factory=list)  # nothing stored


async def ingest_member(
    client: MembersClient,
    mnis_id: int,
    fallback_name: str | None,
    stats: Stats,
) -> None:
    try:
        fetched = await client.fetch_member(mnis_id)
    except FetchError as exc:
        stats.failed.append((mnis_id, str(exc)))
        return

    try:
        row = parse_member(mnis_id, fetched.payloads, fetched.missing, fallback_name)
        row["raw"] = Jsonb(row["raw"])
        # Acquire a connection only now, so HTTP waits never hold one from the pool.
        async with pool.connection() as conn:
            await conn.execute(SQL_UPSERT, row)
    except Exception as exc:  # one odd member must not abort the batch; it is listed in the summary
        stats.failed.append((mnis_id, f"{type(exc).__name__}: {exc}"))
        return

    if fetched.payloads.get("profile") is None:
        stats.stubs += 1
    elif fetched.missing:
        stats.partial += 1
    else:
        stats.complete += 1


async def run_ingest(concurrency: int, min_interval: float, refresh: bool, limit: int | None) -> Stats:
    stats = Stats()
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            cur = await conn.execute(SQL_TARGETS, {"refresh": refresh, "limit": limit})
            targets = await cur.fetchall()
        if not targets:
            console.print("[green]Nothing to do: every speaker already has a complete member row.[/green]")
            return stats

        # A member is ~9 requests; capping members in flight keeps memory and progress orderly, while the client's own
        # semaphore and gate cap the requests themselves.
        in_flight = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=concurrency * 2),
        ) as http:
            client = MembersClient(http, concurrency=concurrency, min_interval=min_interval)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]Fetching members"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[red]{task.fields[failed]} failed[/red]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("members", total=len(targets), failed=0)

                async def worker(mnis_id: int, fallback_name: str | None) -> None:
                    async with in_flight:
                        await ingest_member(client, mnis_id, fallback_name, stats)
                    progress.update(task, advance=1, failed=len(stats.failed))

                await asyncio.gather(*(worker(m, n) for m, n in targets))
    finally:
        await pool.close()
    return stats


def print_summary(stats: Stats) -> None:
    table = Table(title="Member ingest complete")
    table.add_column("")
    table.add_column("Members", justify="right")
    table.add_row("Complete", f"[green]{stats.complete:,}[/green]")
    table.add_row("Partial (some endpoints failed, will retry)", f"[yellow]{stats.partial:,}[/yellow]")
    table.add_row("Unknown to the API (stub from Hansard name)", f"[yellow]{stats.stubs:,}[/yellow]")
    table.add_row("Failed (nothing stored, will retry)", f"[red]{len(stats.failed):,}[/red]")
    console.print(table)
    for mnis_id, message in stats.failed[:10]:
        console.print(f"  [red]{mnis_id}[/red]: {message}")
    if len(stats.failed) > 10:
        console.print(f"  ... and {len(stats.failed) - 10} more")
    if stats.partial or stats.failed:
        console.print("[yellow]Re-run the same command to retry the incomplete ones.[/yellow]")


def main() -> None:
    # Retired v1 loader: never the rollback, never hansard_v2 (built only from the v2 cache).
    from app import safety
    from app.db import pool as _pool

    safety.refuse_for_v1(_pool.conninfo)
    parser = argparse.ArgumentParser(description="Fetch member metadata from the UK Parliament Members API.")
    parser.add_argument("--refresh", action="store_true", help="re-fetch every member, not just missing ones")
    parser.add_argument("--limit", type=int, default=None, help="only process this many members")
    parser.add_argument("--concurrency", type=int, default=6, help="max simultaneous requests (default: %(default)s)")
    parser.add_argument("--min-interval", type=float, default=0.1,
                        help="min seconds between request starts; 0.1 = at most 10 req/s (default: %(default)s)")
    args = parser.parse_args()

    stats = run(run_ingest(args.concurrency, args.min_interval, args.refresh, args.limit))
    print_summary(stats)


if __name__ == "__main__":
    main()
