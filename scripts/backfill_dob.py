"""Backfill member dates of birth (and identity facts) from Wikidata, matched on MNIS id."""

import argparse
import json
from datetime import date

import httpx
from rich.console import Console
from rich.table import Table

from app.db import pool
from ingest.http import FetchError, PoliteClient
from ingest.runtime import run
from ingest.wikidata import CORE_COLUMNS, EXTRA_COLUMNS, SPARQL_URL, build_update, core_query, extras_query, parse_core, parse_extras

USER_AGENT = "on-the-record/0.1 (Hansard civic-intelligence research)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}

console = Console()


async def query(client: PoliteClient, sparql: str) -> list[dict]:
    payload = await client.post_json(SPARQL_URL, {"query": sparql, "format": "json"}, headers=HEADERS)
    if payload is None:
        raise FetchError("Wikidata returned 404 for the SPARQL endpoint")
    return payload["results"]["bindings"]


async def run_backfill(core_only: bool, dry_run: bool) -> dict:
    report: dict = {"dry_run": dry_run}
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            ids = [r[0] for r in await (await conn.execute("SELECT mnis_id FROM member_id_map ORDER BY mnis_id")).fetchall()]
            report["members"] = len(ids)
            # One request at a time and at most two attempts: Wikidata's limit is per-query time, and a retry storm
            # after a timeout only makes the next one slower.
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), headers=HEADERS) as http:
                client = PoliteClient(http, concurrency=1, min_interval=1.0, max_attempts=2)
                core = parse_core(await query(client, core_query(ids)), date.today())
                extras = {} if core_only else parse_extras(await query(client, extras_query(ids)))
            report["core"], report["extras"] = core, extras

            columns = CORE_COLUMNS + (() if core_only else EXTRA_COLUMNS)
            matched = sorted(core.rows)
            if not dry_run and matched:
                arrays: list = [matched]
                for name, kind in columns:
                    if name in dict(EXTRA_COLUMNS):
                        arrays.append([extras.get(m, {}).get(name) for m in matched])
                    else:
                        arrays.append([json.dumps(core.rows[m][name], default=str) if kind == "jsonb" else core.rows[m][name] for m in matched])
                arrays.append([core.rows[m]["person_id"] for m in matched])
                async with conn.transaction():
                    # person_id is derived from twfy_id: clear it for the batch so ids can move between members
                    await conn.execute("UPDATE member_id_map SET person_id = NULL WHERE mnis_id = ANY(%s::int[])", (matched,))
                    await conn.execute(build_update(columns), arrays)
            report["ages"] = await (await conn.execute("""
                SELECT count(*) FILTER (WHERE age_exact IS NOT NULL), min(age_exact), max(age_exact),
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY age_exact) FROM member_age""")).fetchone()
    finally:
        await pool.close()
    return report


def print_report(r: dict) -> None:
    core = r["core"]
    rows = core.rows.values()
    precisions = {p: sum(1 for x in rows if x["dob_precision"] == p) for p in ("day", "month", "year")}
    none = sum(1 for x in rows if x["dob_precision"] is None)
    t = Table(title="Wikidata backfill" + (" (dry run: nothing written)" if r["dry_run"] else ""))
    t.add_column("")
    t.add_column("Members", justify="right")
    t.add_row("Members in our table", f"{r['members']:,}")
    t.add_row("Found on Wikidata by MNIS id", f"{len(core.rows):,}")
    t.add_row("  full birth date (date_of_birth set)", f"[green]{precisions['day']:,}[/green]")
    t.add_row("  month only  (age is a 1-year range)", f"{precisions['month']:,}")
    t.add_row("  year only   (age is a 1-2 year range)", f"{precisions['year']:,}")
    t.add_row("  no usable birth date", f"[yellow]{none:,}[/yellow]")
    for note in ("conflicting_dob", "implausible_dob", "inconsistent_candidates", "coarse_dob_ignored"):
        n = sum(1 for x in rows if x["wikidata"]["dob"]["note"] == note)
        if n:
            t.add_row(f"  flagged {note}", f"[yellow]{n}[/yellow]")
    t.add_row("TheyWorkForYou id (person_id)", f"{sum(1 for x in rows if x['person_id']):,}")
    t.add_row("Democracy Club id", f"{sum(1 for x in rows if x['democracy_club_id']):,}")
    t.add_row("Date of death recorded", f"{sum(1 for x in rows if x['date_of_death']):,}")
    if r["extras"]:
        t.add_row("Place of birth / Wikipedia", f"{sum(1 for x in r['extras'].values() if x['place_of_birth']):,} / {sum(1 for x in r['extras'].values() if x['wikipedia_url']):,}")
    console.print(t)
    if core.problems:
        console.print(f"[yellow]Problems ({len(core.problems)}):[/yellow] " + ", ".join(core.problems[:12]))
    exact, lo, hi, median = r["ages"]
    console.print(f"Exact ages in the database now: {exact:,} members (youngest {lo}, oldest {hi}, median {median:.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill dates of birth from Wikidata.")
    parser.add_argument("--core-only", action="store_true", help="skip the slower place-of-birth / Wikipedia query")
    parser.add_argument("--dry-run", action="store_true", help="fetch and report; write nothing")
    args = parser.parse_args()
    print_report(run(run_backfill(args.core_only, args.dry_run)))


if __name__ == "__main__":
    main()
