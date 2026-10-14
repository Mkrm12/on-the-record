"""Forensic gap analysis: account for every single API contribution."""

import datetime
import sys
import time

import httpx
import psycopg
from rich.console import Console
from rich.table import Table

from app import safety
from app.config import settings
from ingest.window import days_newest_first, parse_window

API_URL = "https://hansard-api.parliament.uk/search/contributions/Spoken.json"
DEFAULT_START = datetime.date(2026, 8, 10)
DEFAULT_END = datetime.date(2026, 9, 19)

# Tiny pages for maximum stability
PAGE_SIZE = 20
MAX_ATTEMPTS = 8
# Try every sort order the API supports
SORT_ORDERS = (
    "SittingDateAsc",
    "SittingDateDesc",
)

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


def fetch_page(
    client: httpx.Client,
    date_str: str,
    skip: int,
    take: int,
    order: str,
) -> dict | None:
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
                    f"[bold red]{date_str} skip={skip} failed after "
                    f"{attempt} attempts; skipping[/bold red]"
                )
                return None
            delay = 2 ** attempt
            time.sleep(delay)
    return None


def clean(value: str | None) -> str | None:
    return value.replace("\x00", "") if value else value


def forensic_scan_day(
    client: httpx.Client,
    conn: psycopg.Connection,
    date_str: str,
) -> dict:
    """Exhaustively scan a single day and account for every contribution."""
    # Sets to track unique items across all sort orders / pages
    seen_speech_ids: set[str] = set()  # valid ContributionExtId values
    idless_items: list[dict] = []  # items missing required IDs
    idless_fingerprints: set[str] = set()  # dedup idless items

    new_debates = 0
    new_speeches = 0
    api_total = None

    for order in SORT_ORDERS:
        skip = 0
        consecutive_empty = 0

        while True:
            page = fetch_page(client, date_str, skip, PAGE_SIZE, order)
            if page is None or "Results" not in page:
                skip += PAGE_SIZE
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                continue

            if api_total is None and "TotalResultCount" in page:
                api_total = page["TotalResultCount"]

            results = page["Results"]
            if not results:
                break

            consecutive_empty = 0
            debates_batch: dict[str, tuple] = {}
            speeches_batch: list[tuple] = []

            for item in results:
                speech_id = item.get("ContributionExtId")
                debate_id = item.get("DebateSectionExtId")

                if not speech_id or not debate_id:
                    # Build a fingerprint to dedup ID-less items
                    fp = (
                        f"{item.get('AttributedTo', '')}|"
                        f"{item.get('Timecode', '')}|"
                        f"{(item.get('ContributionText') or '')[:80]}"
                    )
                    if fp not in idless_fingerprints:
                        idless_fingerprints.add(fp)
                        idless_items.append({
                            "speaker": item.get("AttributedTo"),
                            "timecode": item.get("Timecode"),
                            "text_preview": (
                                item.get("ContributionText") or ""
                            )[:120],
                            "has_speech_id": bool(speech_id),
                            "has_debate_id": bool(debate_id),
                        })
                    continue

                if speech_id in seen_speech_ids:
                    continue  # already processed in another sort order
                seen_speech_ids.add(speech_id)

                debates_batch.setdefault(
                    debate_id,
                    (
                        debate_id,
                        item["SittingDate"][:10],
                        item.get("House"),
                        (item.get("DebateSection") or "").strip() or None,
                        item.get("Section"),
                    ),
                )
                speeches_batch.append(
                    (
                        speech_id,
                        debate_id,
                        item.get("OrderInDebateSection"),
                        item.get("AttributedTo"),
                        item.get("MemberId"),
                        clean(
                            item.get("ContributionTextFull")
                            or item.get("ContributionText")
                        ),
                        item.get("Timecode"),
                    )
                )

            # Insert any new data
            if debates_batch or speeches_batch:
                with conn.transaction(), conn.cursor() as cur:
                    if debates_batch:
                        cur.executemany(INSERT_DEBATE, list(debates_batch.values()))
                        new_debates += cur.rowcount
                    if speeches_batch:
                        cur.executemany(INSERT_SPEECH, speeches_batch)
                        new_speeches += cur.rowcount

            skip += PAGE_SIZE
            # Safety: don't paginate beyond what the API reports
            if api_total is not None and skip >= api_total:
                break

    # Final DB count for this day
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM speech s "
            "JOIN debate d ON d.id = s.debate_id "
            "WHERE d.date = %s",
            (date_str,),
        )
        final_stored = cur.fetchone()[0]

    return {
        "date": date_str,
        "api_total": api_total or 0,
        "unique_seen": len(seen_speech_ids),
        "idless_count": len(idless_items),
        "idless_items": idless_items,
        "new_debates": new_debates,
        "new_speeches": new_speeches,
        "final_stored": final_stored,
    }


def main() -> None:
    # v1 tooling (superseded by fetch_corpus_v2 + load_corpus_v2).
    dsn = safety.refuse_for_v1(settings.pg_dsn)
    # Target only the deficit days identified by fill_gaps.py But also re-scan ALL days to be absolutely sure
    start, end = parse_window(sys.argv[1:], DEFAULT_START, DEFAULT_END, __doc__.splitlines()[0])
    dates = days_newest_first(start, end)
    num_days = len(dates)

    total_new = 0
    all_results: list[dict] = []

    with (
        httpx.Client(timeout=60.0) as client,
        psycopg.connect(dsn, autocommit=True) as conn,
    ):
        for i, day in enumerate(dates, 1):
            date_str = day.isoformat()

            # Quick probe: does this day have data?
            page = fetch_page(client, date_str, 0, 1, "SittingDateAsc")
            if page is None or page.get("TotalResultCount", 0) == 0:
                console.print(f"[dim]{date_str}: no sitting ({i}/{num_days})[/dim]")
                continue

            api_total = page["TotalResultCount"]

            # Check DB count
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM speech s "
                    "JOIN debate d ON d.id = s.debate_id "
                    "WHERE d.date = %s",
                    (date_str,),
                )
                stored = cur.fetchone()[0]

            if stored >= api_total:
                console.print(
                    f"[green]{date_str}: {stored}/{api_total} "
                    f"-- complete ({i}/{num_days})[/green]"
                )
                continue

            # Deficit day -- full forensic scan
            deficit = api_total - stored
            console.print(
                f"[bold cyan]{date_str}: {stored}/{api_total} "
                f"(deficit={deficit}) -- forensic scan... ({i}/{num_days})[/bold cyan]"
            )
            result = forensic_scan_day(client, conn, date_str)
            all_results.append(result)
            total_new += result["new_speeches"]

            # Print per-item detail for ID-less entries
            if result["idless_items"]:
                console.print(
                    f"  [yellow]{result['idless_count']} ID-less item(s) "
                    f"(count toward API total but cannot be stored):[/yellow]"
                )
                for idx, item in enumerate(result["idless_items"][:5], 1):
                    console.print(
                        f"    {idx}. speaker={item['speaker']!r}, "
                        f"timecode={item['timecode']!r}, "
                        f"has_speech_id={item['has_speech_id']}, "
                        f"has_debate_id={item['has_debate_id']}"
                    )
                    if item["text_preview"]:
                        console.print(
                            f"       text: {item['text_preview']!r}"
                        )
                if result["idless_count"] > 5:
                    console.print(
                        f"    ... and {result['idless_count'] - 5} more"
                    )

    # ------------------------------------------------------------------- Final accounting
    # -------------------------------------------------------------------
    console.print()

    # Grand total recount
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT d.id), count(s.id) "
            "FROM debate d "
            "LEFT JOIN speech s ON s.debate_id = d.id "
            "WHERE d.date BETWEEN %s AND %s",
            (start.isoformat(), end.isoformat()),
        )
        grand_debates, grand_speeches = cur.fetchone()

    if all_results:
        detail = Table(title="Forensic accounting per deficit day")
        detail.add_column("Date")
        detail.add_column("API total", justify="right")
        detail.add_column("Unique IDs seen", justify="right")
        detail.add_column("ID-less items", justify="right")
        detail.add_column("Accounted", justify="right")
        detail.add_column("Ghosts", justify="right")
        detail.add_column("New inserts", justify="right")
        detail.add_column("Final stored", justify="right")

        for r in all_results:
            accounted = r["unique_seen"] + r["idless_count"]
            ghosts = r["api_total"] - accounted
            ghost_style = "green" if ghosts == 0 else "red"
            new_style = "green" if r["new_speeches"] > 0 else "dim"
            detail.add_row(
                r["date"],
                str(r["api_total"]),
                str(r["unique_seen"]),
                str(r["idless_count"]),
                str(accounted),
                f"[{ghost_style}]{ghosts}[/{ghost_style}]",
                f"[{new_style}]{r['new_speeches']}[/{new_style}]",
                str(r["final_stored"]),
            )
        console.print(detail)

    summary = Table(title="Final database state")
    summary.add_column("")
    summary.add_column("Count", justify="right")
    summary.add_row("Total debates", f"{grand_debates:,}")
    summary.add_row("Total speeches", f"[bold]{grand_speeches:,}[/bold]")
    summary.add_row("New speeches this run", f"[green]{total_new:,}[/green]")
    console.print(summary)

    remaining = 0
    for r in all_results:
        remaining += r["api_total"] - r["final_stored"]

    if remaining > 0:
        id_less_total = sum(r["idless_count"] for r in all_results)
        console.print(
            f"\n[bold yellow]Remaining deficit: {remaining} speech(es) across "
            f"{len(all_results)} day(s).[/bold yellow]"
        )
        if id_less_total > 0:
            console.print(
                f"[yellow]Of these, {id_less_total} are ID-less API entries "
                f"that cannot be stored (no ContributionExtId or DebateSectionExtId). "
                f"These are API ghosts, not missing data.[/yellow]"
            )
        ghost_total = sum(
            r["api_total"] - r["unique_seen"] - r["idless_count"]
            for r in all_results
        )
        if ghost_total > 0:
            console.print(
                f"[yellow]{ghost_total} are true ghosts: counted in API total "
                f"but never served in any paginated response.[/yellow]"
            )
    else:
        console.print(
            "\n[bold green]PERFECT: Every API-reported speech is accounted for. "
            "Zero data loss.[/bold green]"
        )


if __name__ == "__main__":
    main()

