"""Fill ingestion gaps by querying the Hansard API one day at a time."""

import datetime
import sys
import time

import httpx
import psycopg
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from app import safety
from app.config import settings
from ingest.window import days_newest_first, parse_window

API_URL = "https://hansard-api.parliament.uk/search/contributions/Spoken.json"
DEFAULT_START = datetime.date(2026, 8, 10)
DEFAULT_END = datetime.date(2026, 9, 19)
PAGE_SIZE = 50  # smaller pages = more stable on the API
MAX_ATTEMPTS = 6  # generous retries for per-day queries

INSERT_DEBATE = """
    INSERT INTO debate (id, date, house, heading, section)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""
INSERT_SPEECH = """
    INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, timecode)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
"""

console = Console()


# --------------------------------------------------------------------------- HTTP helpers
# ---------------------------------------------------------------------------

def fetch_page(
    client: httpx.Client,
    date_str: str,
    skip: int,
    take: int,
    order: str,
) -> dict | None:
    """Fetch a single page from the Hansard API for one specific day."""
    params = {
        "queryParameters.startDate": date_str,
        "queryParameters.endDate": date_str,
        "queryParameters.orderBy": order,
        "queryParameters.take": take,
        "queryParameters.skip": skip,
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.get(API_URL, params=params)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                exc.response.status_code == 429 or exc.response.status_code >= 500
            )
            if not retryable or attempt == MAX_ATTEMPTS:
                console.print(
                    f"[bold red]{date_str} skip={skip} permanently failed "
                    f"after {attempt} attempt(s); skipping page[/bold red]"
                )
                return None
            delay = 2 ** attempt
            console.print(
                f"[yellow]{date_str} skip={skip} attempt {attempt} "
                f"failed ({exc!r}); retrying in {delay}s[/yellow]"
            )
            time.sleep(delay)
    return None


def get_api_day_total(client: httpx.Client, date_str: str) -> int | None:
    """Probe the API for a day's total count (fetches 0 results, just the count)."""
    page = fetch_page(client, date_str, skip=0, take=1, order="SittingDateAsc")
    if page is None or "TotalResultCount" not in page:
        return None
    return page["TotalResultCount"]


# --------------------------------------------------------------------------- Row mapping
# ---------------------------------------------------------------------------

def clean(value: str | None) -> str | None:
    return value.replace("\x00", "") if value else value


def to_rows(results: list[dict]) -> tuple[list[tuple], list[tuple], int]:
    """Map API contributions to (debate rows, speech rows, skipped count)."""
    debates: dict[str, tuple] = {}
    speeches: list[tuple] = []
    skipped = 0
    for item in results:
        speech_id = item.get("ContributionExtId")
        debate_id = item.get("DebateSectionExtId")
        if not speech_id or not debate_id:
            skipped += 1
            continue
        debates.setdefault(
            debate_id,
            (
                debate_id,
                item["SittingDate"][:10],
                item.get("House"),
                (item.get("DebateSection") or "").strip() or None,
                item.get("Section"),
            ),
        )
        speeches.append(
            (
                speech_id,
                debate_id,
                item.get("OrderInDebateSection"),
                item.get("AttributedTo"),
                item.get("MemberId"),
                clean(item.get("ContributionTextFull") or item.get("ContributionText")),
                item.get("Timecode"),
            )
        )
    return list(debates.values()), speeches, skipped


# --------------------------------------------------------------------------- Per-day ingestion
# ---------------------------------------------------------------------------

def ingest_day(
    client: httpx.Client,
    conn: psycopg.Connection,
    date_str: str,
    api_total: int,
) -> tuple[int, int, int]:
    """Paginate through a single day with both sort orders."""
    new_debates = new_speeches = skipped = 0

    for order in ("SittingDateAsc", "SittingDateDesc"):
        skip = 0
        while skip < api_total:
            page = fetch_page(client, date_str, skip, PAGE_SIZE, order)
            if page is None or "Results" not in page:
                skip += PAGE_SIZE
                continue

            results = page["Results"]
            if not results:
                break

            debate_rows, speech_rows, page_skipped = to_rows(results)
            with conn.transaction(), conn.cursor() as cur:
                cur.executemany(INSERT_DEBATE, debate_rows)
                new_debates += cur.rowcount
                cur.executemany(INSERT_SPEECH, speech_rows)
                new_speeches += cur.rowcount
            skipped += page_skipped
            skip += PAGE_SIZE

    return new_debates, new_speeches, skipped


# --------------------------------------------------------------------------- Main
# ---------------------------------------------------------------------------

def main() -> None:
    # v1 tooling (superseded by fetch_corpus_v2 + load_corpus_v2).
    dsn = safety.refuse_for_v1(settings.pg_dsn)
    start, end = parse_window(sys.argv[1:], DEFAULT_START, DEFAULT_END, __doc__.splitlines()[0])
    dates = days_newest_first(start, end)  # reverse chronological
    num_days = len(dates)

    total_new_debates = 0
    total_new_speeches = 0
    total_api = 0
    total_stored_before = 0
    deficit_days: list[tuple[str, int, int, int]] = []  # (date, api, stored, found)

    with (
        httpx.Client(timeout=60.0) as client,
        psycopg.connect(dsn, autocommit=True) as conn,
        Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("Scanning days", total=num_days)

        for day in dates:
            date_str = day.isoformat()
            progress.update(task, description=f"Scanning {date_str}")

            # 1.
            api_total = get_api_day_total(client, date_str)
            if api_total is None:
                console.print(
                    f"[bold red]{date_str}: could not reach API; skipping day[/bold red]"
                )
                progress.advance(task)
                continue

            if api_total == 0:
                # Non-sitting day (weekend, recess)
                progress.advance(task)
                continue

            total_api += api_total

            # 2.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM speech s "
                    "JOIN debate d ON d.id = s.debate_id "
                    "WHERE d.date = %s",
                    (date_str,),
                )
                stored = cur.fetchone()[0]
            total_stored_before += stored

            deficit = api_total - stored
            if deficit <= 0:
                # Day is already complete
                progress.advance(task)
                continue

            # 3.
            console.print(
                f"[cyan]{date_str}: API={api_total}, stored={stored}, "
                f"deficit={deficit} -- ingesting[/cyan]"
            )
            day_debates, day_speeches, day_skipped = ingest_day(
                client, conn, date_str, api_total,
            )
            total_new_debates += day_debates
            total_new_speeches += day_speeches
            deficit_days.append((date_str, api_total, stored, day_speeches))
            progress.advance(task)

    # ----------------------------------------------------------------------- Final report
    # -----------------------------------------------------------------------

    # Recount total stored after fill
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT d.id), count(s.id) "
            "FROM debate d "
            "LEFT JOIN speech s ON s.debate_id = d.id "
            "WHERE d.date BETWEEN %s AND %s",
            (start.isoformat(), end.isoformat()),
        )
        final_debates, final_speeches = cur.fetchone()

    console.print()

    if deficit_days:
        detail = Table(title="Days with deficits")
        detail.add_column("Date")
        detail.add_column("API total", justify="right")
        detail.add_column("Stored before", justify="right")
        detail.add_column("Newly found", justify="right")
        for date_str, api_t, stored_b, found in deficit_days:
            found_style = "green" if found > 0 else "yellow"
            detail.add_row(date_str, str(api_t), str(stored_b), f"[{found_style}]{found}[/{found_style}]")
        console.print(detail)

    summary = Table(title=f"Gap fill {start} -> {end}")
    summary.add_column("")
    summary.add_column("Count", justify="right")
    summary.add_row("Days scanned", str(num_days))
    summary.add_row("Days with deficit", str(len(deficit_days)))
    summary.add_row("New debates inserted", f"[green]{total_new_debates:,}[/green]")
    summary.add_row("New speeches inserted", f"[green]{total_new_speeches:,}[/green]")
    summary.add_row("Total debates (final)", f"{final_debates:,}")
    summary.add_row("Total speeches (final)", f"[bold]{final_speeches:,}[/bold]")
    console.print(summary)

    if total_new_speeches == 0:
        console.print(
            "[bold green]No gaps found -- database already has full coverage "
            "of all API-reported speeches.[/bold green]"
        )
    else:
        console.print(
            f"[bold green]Recovered {total_new_speeches:,} previously missing speech(es).[/bold green]"
        )


if __name__ == "__main__":
    main()

