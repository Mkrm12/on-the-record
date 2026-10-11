"""Backfill the Hansard API fields the original speech ingest dropped."""

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import httpx
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from app.db import pool
from ingest.hansard_meta import fetch_day, merge_items, parse_item, text_differs
from ingest.http import PoliteClient
from ingest.runtime import run

USER_AGENT = "on-the-record/0.1 (Hansard civic-intelligence research)"
ORDERS = ("SittingDateAsc", "SittingDateDesc")
console = Console()

SQL_UPDATE_SPEECH = """
    UPDATE speech s SET item_id = u.item_id, hrs_tag = u.hrs_tag, hansard_section = u.hansard_section,
                        api_rank = u.api_rank, api_member_name = u.api_member_name, meta = u.meta::jsonb
    FROM unnest(%s::text[], %s::int[], %s::text[], %s::text[], %s::int[], %s::text[], %s::text[])
         AS u(id, item_id, hrs_tag, hansard_section, api_rank, api_member_name, meta)
    WHERE s.id = u.id
      AND (s.item_id, s.hrs_tag, s.hansard_section, s.api_rank, s.api_member_name, s.meta)
          IS DISTINCT FROM (u.item_id, u.hrs_tag, u.hansard_section, u.api_rank, u.api_member_name, u.meta::jsonb)
"""
SQL_UPDATE_DEBATE = """
    UPDATE debate d SET section_id = u.section_id, section_order = u.section_order
    FROM unnest(%s::text[], %s::int[], %s::int[]) AS u(id, section_id, section_order)
    WHERE d.id = u.id AND (d.section_id, d.section_order) IS DISTINCT FROM (u.section_id, u.section_order)
"""


@dataclass
class Report:
    days: int = 0
    api_items: int = 0
    matched: int = 0
    speeches_in_db: int = 0
    missing_from_api: int = 0
    api_only: int = 0
    text_changed: list[str] = field(default_factory=list)
    hrs_tags: Counter = field(default_factory=Counter)
    speech_rows_changed: int = 0
    debates_ordered: int = 0
    debates_total: int = 0
    dry_run: bool = False


async def run_backfill(concurrency: int, min_interval: float, dry_run: bool) -> Report:
    report = Report(dry_run=dry_run)
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            days = [r[0] for r in await (await conn.execute("SELECT DISTINCT date FROM debate ORDER BY date")).fetchall()]
            stored = dict(await (await conn.execute("SELECT id, text FROM speech")).fetchall())
            report.days, report.speeches_in_db = len(days), len(stored)
            report.debates_total = (await (await conn.execute("SELECT count(*) FROM debate")).fetchone())[0]

            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers={"User-Agent": USER_AGENT},
                                         limits=httpx.Limits(max_connections=concurrency * 2)) as http:
                client = PoliteClient(http, concurrency=concurrency, min_interval=min_interval)
                with Progress(SpinnerColumn(), TextColumn("[bold]Reading sitting days"), BarColumn(), MofNCompleteColumn(),
                              TimeElapsedColumn(), console=console) as progress:  # fmt: skip
                    task = progress.add_task("days", total=len(days))

                    async def one_day(day: date) -> list[dict]:
                        both = await asyncio.gather(*(fetch_day(client, day, order) for order in ORDERS))
                        progress.advance(task)
                        return [item for batch in both for item in batch]

                    items = [i for day_items in await asyncio.gather(*(one_day(d) for d in days)) for i in day_items]

            merged = merge_items(items)
            report.api_items = len(merged)
            speech_rows, debate_rows = [], {}
            for speech_id, item in merged.items():
                if speech_id not in stored:
                    report.api_only += 1
                    continue
                speech, debate = parse_item(item)
                report.matched += 1
                report.hrs_tags[speech["hrs_tag"]] += 1
                if text_differs(item, stored[speech_id]):
                    report.text_changed.append(speech_id)
                speech_rows.append(speech)
                debate_rows.setdefault(debate["id"], debate)
            report.missing_from_api = len(stored) - report.matched
            report.debates_ordered = sum(1 for d in debate_rows.values() if d["section_order"] is not None)

            if not dry_run and speech_rows:
                async with conn.transaction():
                    cur = await conn.execute(SQL_UPDATE_SPEECH, (
                        [s["id"] for s in speech_rows], [s["item_id"] for s in speech_rows], [s["hrs_tag"] for s in speech_rows],
                        [s["hansard_section"] for s in speech_rows], [s["api_rank"] for s in speech_rows],
                        [s["api_member_name"] for s in speech_rows], [json.dumps(s["meta"], ensure_ascii=False) for s in speech_rows],
                    ))  # fmt: skip
                    report.speech_rows_changed = cur.rowcount
                    ds = list(debate_rows.values())
                    await conn.execute(SQL_UPDATE_DEBATE, ([d["id"] for d in ds], [d["section_id"] for d in ds], [d["section_order"] for d in ds]))
    finally:
        await pool.close()
    return report


def print_report(r: Report) -> None:
    t = Table(title="Speech metadata backfill" + (" (dry run: nothing written)" if r.dry_run else ""))
    t.add_column("")
    t.add_column("Count", justify="right")
    t.add_row("Sitting days read (both sort orders)", f"{r.days:,}")
    t.add_row("Distinct contributions the API served", f"{r.api_items:,}")
    t.add_row("Matched to a stored speech", f"[green]{r.matched:,}[/green] of {r.speeches_in_db:,}")
    t.add_row("Stored speeches the API did not serve", f"[yellow]{r.missing_from_api:,}[/yellow]")
    t.add_row("API contributions we never stored", f"{r.api_only:,}")
    t.add_row("Stored text differs from the API's text now", f"[yellow]{len(r.text_changed):,}[/yellow]")
    t.add_row("Speech rows changed", f"{r.speech_rows_changed:,}")
    t.add_row("Debates with a running order", f"{r.debates_ordered:,} of {r.debates_total:,}")
    console.print(t)
    console.print("HRSTag values: " + ", ".join(f"{k}={v:,}" for k, v in r.hrs_tags.most_common(8)))
    if r.missing_from_api:
        console.print("[dim]Speeches the API no longer serves are the known unstable-pagination ghosts; their old data is untouched.[/dim]")


def main() -> None:
    # Retired v1 loader: never the rollback, never hansard_v2 (built only from the v2 cache).
    from app import safety
    from app.db import pool as _pool

    safety.refuse_for_v1(_pool.conninfo)
    parser = argparse.ArgumentParser(description="Backfill Hansard API metadata for stored speeches.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--min-interval", type=float, default=0.1)
    args = parser.parse_args()
    print_report(run(run_backfill(args.concurrency, args.min_interval, args.dry_run)))


if __name__ == "__main__":
    main()
