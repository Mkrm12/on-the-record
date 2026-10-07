"""Everything computed from the loaded rows, and the per-window load report."""

import argparse
import json
import sys
from typing import Any

from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from rich.console import Console
from rich.table import Table

from app import safety
from app.config import settings
from app.db import make_pool
from ingest.runtime import run
from retrieval.lexicon import store_lexicon
from retrieval.offices import build_offices

console = Console()

# --- the per-window report -------------------------------------------------

REPORT_SQL: dict[str, str] = {
    "debates": "SELECT count(*) FROM debate WHERE window_id = %(w)s",
    "speeches": "SELECT count(*) FROM speech WHERE window_id = %(w)s",
    "chunks": "SELECT count(*) FROM chunk WHERE window_id = %(w)s",
    "chunks_boilerplate": "SELECT count(*) FROM chunk WHERE window_id = %(w)s AND is_boilerplate",
    "chunks_embedded": "SELECT count(*) FROM chunk WHERE window_id = %(w)s AND embedding IS NOT NULL",
    "speeches_boilerplate": "SELECT count(*) FROM speech WHERE window_id = %(w)s AND is_boilerplate",
    # The mapping invariant: every speech that names a member must resolve to a member row.
    "speeches_with_mnis": "SELECT count(*) FROM speech WHERE window_id = %(w)s AND mnis_id IS NOT NULL",
    "orphan_speakers": """
        SELECT count(DISTINCT s.mnis_id) FROM speech s
        LEFT JOIN member_id_map m USING (mnis_id)
        WHERE s.window_id = %(w)s AND s.mnis_id IS NOT NULL AND m.mnis_id IS NULL
    """,
    "orphan_voters": """
        SELECT count(DISTINCT v.mnis_id) FROM division_vote v
        JOIN division x ON x.id = v.division_id
        JOIN corpus_window cw ON cw.window_id = %(w)s
        LEFT JOIN member_id_map m ON m.mnis_id = v.mnis_id
        WHERE x.division_date BETWEEN cw.start_date AND cw.end_date AND m.mnis_id IS NULL
    """,
    "procedure_events": """
        SELECT count(*) FROM procedure_event p JOIN debate d ON d.id = p.debate_id
        WHERE d.window_id = %(w)s
    """,
    "citations_known": """
        SELECT count(*) FROM chunk WHERE window_id = %(w)s AND column_ref IS NOT NULL
    """,
    "corrections": "SELECT count(*) FROM hansard_correction WHERE window_id = %(w)s",
}

GROUPED_SQL: dict[str, str] = {
    "sections": "SELECT section, count(*) FROM debate WHERE window_id = %(w)s GROUP BY 1 ORDER BY 2 DESC",
    "debate_types": "SELECT debate_type, count(*) FROM debate WHERE window_id = %(w)s GROUP BY 1 ORDER BY 2 DESC",
    "event_types": """
        SELECT p.event_type, count(*) FROM procedure_event p JOIN debate d ON d.id = p.debate_id
        WHERE d.window_id = %(w)s GROUP BY 1 ORDER BY 2 DESC
    """,
    "citation_basis": """
        SELECT coalesce(column_basis, 'unknown'), count(*) FROM chunk
        WHERE window_id = %(w)s GROUP BY 1 ORDER BY 2 DESC
    """,
    "speaker_roles": """
        SELECT coalesce(speaker_role, 'unknown'), count(*) FROM speech
        WHERE window_id = %(w)s GROUP BY 1 ORDER BY 2 DESC
    """,
    "division_links": """
        SELECT coalesce(x.debate_link, 'none'), count(*) FROM division x
        JOIN corpus_window cw ON cw.window_id = %(w)s
        WHERE x.division_date BETWEEN cw.start_date AND cw.end_date GROUP BY 1 ORDER BY 2 DESC
    """,
}

DIVISIONS_SQL = """
    SELECT count(*), count(*) FILTER (WHERE tally_reconciled),
           count(*) FILTER (WHERE tally_reconciled IS FALSE),
           count(*) FILTER (WHERE debate_id IS NOT NULL),
           count(*) FILTER (WHERE bill_stage_id IS NOT NULL),
           count(*) FILTER (WHERE is_deferred),
           count(*) FILTER (WHERE house = 'Lords'),
           count(*) FILTER (WHERE is_remote),
           -- I-8's natural experiment: a remote division is counted by a different
           -- mechanism from a physical one, so whether the two reconcile at different
           -- rates is the question W3's 10 remote divisions exist to answer.
           count(*) FILTER (WHERE is_remote AND tally_reconciled),
           count(*) FILTER (WHERE NOT is_remote AND tally_reconciled)
    FROM division x JOIN corpus_window cw ON cw.window_id = %(w)s
    WHERE x.division_date BETWEEN cw.start_date AND cw.end_date
"""

BILLS_SQL = """
    SELECT count(DISTINCT b.id), count(DISTINCT st.id), count(DISTINCT si.id),
           count(DISTINCT si.id) FILTER (WHERE si.debate_id IS NOT NULL)
    FROM bill_stage_sitting si
    JOIN bill b ON b.id = si.bill_id
    JOIN bill_stage st ON st.id = si.bill_stage_id
    JOIN corpus_window cw ON cw.window_id = %(w)s
    WHERE si.date BETWEEN cw.start_date AND cw.end_date
"""

# Offices Hansard attributes inside the window, with how many people held each.
TIME_VARYING_SQL = """
    SELECT s.speaker_office, count(DISTINCT s.mnis_id) AS holders, count(*) AS speeches,
           min(d.date) AS first_day, max(d.date) AS last_day
    FROM speech s JOIN debate d ON d.id = s.debate_id
    WHERE d.window_id = %(w)s AND s.speaker_office IS NOT NULL AND s.mnis_id IS NOT NULL
    GROUP BY 1 HAVING count(DISTINCT s.mnis_id) > 1
    ORDER BY holders DESC, speeches DESC LIMIT 25
"""


async def _scalars(conn, window: str) -> dict[str, int]:
    out = {}
    for name, sql in REPORT_SQL.items():
        row = await (await conn.execute(sql, {"w": window})).fetchone()
        out[name] = int(row[0]) if row and row[0] is not None else 0
    return out


async def _grouped(conn, window: str) -> dict[str, dict[str, int]]:
    out = {}
    for name, sql in GROUPED_SQL.items():
        rows = await (await conn.execute(sql, {"w": window})).fetchall()
        out[name] = {str(key): int(count) for key, count in rows}
    return out


async def window_report(conn, window: str) -> dict[str, Any]:
    counts = await _scalars(conn, window)
    grouped = await _grouped(conn, window)
    (divisions, reconciled, unreconciled, linked, staged, deferred, lords,
     remote, remote_ok, physical_ok) = await (await conn.execute(DIVISIONS_SQL, {"w": window})).fetchone()
    bills, stages, sittings, sittings_linked = \
        await (await conn.execute(BILLS_SQL, {"w": window})).fetchone()
    varying = await (await conn.execute(TIME_VARYING_SQL, {"w": window})).fetchall()
    speeches = counts["speeches"] or 1
    chunks = counts["chunks"] or 1
    return {
        "window_id": window,
        "counts": counts,
        "rates": {
            "mnis_mapping": round(counts["speeches_with_mnis"] / speeches, 4),
            "boilerplate_speeches": round(counts["speeches_boilerplate"] / speeches, 4),
            "boilerplate_chunks": round(counts["chunks_boilerplate"] / chunks, 4),
            "citation_coverage": round(counts["citations_known"] / chunks, 4),
            "embedded": round(counts["chunks_embedded"] / max(chunks - counts["chunks_boilerplate"], 1), 4),
        },
        "divisions": {
            "total": divisions, "lords": lords, "deferred": deferred,
            "tally_reconciled": reconciled, "tally_not_reconciled": unreconciled,
            "linked_to_a_debate": linked, "linked_to_a_bill_stage": staged,
            "link_rate": round(linked / divisions, 4) if divisions else None,
            "remote": remote, "physical": divisions - remote,
            "remote_reconciled": remote_ok, "physical_reconciled": physical_ok,
            "remote_reconcile_rate": round(remote_ok / remote, 4) if remote else None,
            "physical_reconcile_rate": round(physical_ok / (divisions - remote), 4) if divisions - remote else None,
        },
        "bills": {"bills": bills, "stages": stages, "sittings": sittings,
                  "sittings_linked_to_a_debate": sittings_linked,
                  "link_rate": round(sittings_linked / sittings, 4) if sittings else None},
        "time_varying_offices": [
            {"office": office, "holders": holders, "speeches": n,
             "first_day": str(first), "last_day": str(last)}
            for office, holders, n, first, last in varying],
        **grouped,
    }  # fmt: skip


def _problems(report: dict[str, Any]) -> list[str]:
    """The invariants a load has to satisfy. Stated here so the report says whether it passed, rather than leaving
    someone to eyeball a table of numbers.
    """
    out = []
    counts = report["counts"]
    if counts["orphan_speakers"]:
        out.append(f"{counts['orphan_speakers']} speaker MNIS id(s) have no member row "
                   "(member mapping must be 100% with 0 orphans)")
    if counts["orphan_voters"]:
        out.append(f"{counts['orphan_voters']} voter MNIS id(s) have no member row")
    if counts["speeches"] and counts["speeches_with_mnis"] == 0:
        out.append("no speech carries an MNIS id at all")
    if counts["chunks"] == 0 and counts["speeches"]:
        out.append("speeches loaded but nothing was chunked")
    return out


async def _run(args) -> int:
    dsn = make_conninfo(settings.pg_dsn, dbname=args.database) if args.database else settings.pg_dsn
    try:
        safety.require(dsn, args.expect)
    except safety.WrongDatabase as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    built: dict[str, Any] = {}
    reports: list[dict[str, Any]] = []
    pool = make_pool(dsn, min_size=1, max_size=2)
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            if not args.report_only:
                lexicon = await store_lexicon(conn)
                built["lexicon"] = {"lexemes": len(lexicon.df), "n_docs": lexicon.n_docs,
                                    "avg_len": round(lexicon.avg_len, 3)}
                built["offices"] = await build_offices(conn)
                console.print(f"lexicon: {built['lexicon']['lexemes']:,} lexemes over "
                              f"{built['lexicon']['n_docs']:,} chunks")
                console.print(f"offices: {built['offices']}")
            windows = [r[0] for r in await (await conn.execute(
                "SELECT window_id FROM corpus_window ORDER BY start_date")).fetchall()]
            for window in (args.window or windows):
                report = await window_report(conn, window)
                report["problems"] = _problems(report)
                reports.append(report)
                if not args.report_only:
                    async with conn.transaction():
                        await conn.execute(
                            "INSERT INTO load_report (window_id, step, payload) VALUES (%s, 'load', %s)",
                            (window, Jsonb(report)))
    finally:
        await pool.close()

    _print(reports)
    if args.json:
        print(json.dumps({"built": built, "windows": reports}, indent=2, default=str))
    return 1 if any(r["problems"] for r in reports) else 0


def _print(reports: list[dict[str, Any]]) -> None:
    table = Table(title="Per-window load report")
    for column in ("window", "debates", "speeches", "chunks", "MNIS", "boiler", "citations",
                   "divisions", "linked", "bills"):
        table.add_column(column, justify="right" if column != "window" else "left")
    for report in reports:
        counts, rates = report["counts"], report["rates"]
        table.add_row(
            report["window_id"], f"{counts['debates']:,}", f"{counts['speeches']:,}",
            f"{counts['chunks']:,}", f"{rates['mnis_mapping']:.1%}",
            f"{rates['boilerplate_chunks']:.1%}", f"{rates['citation_coverage']:.1%}",
            f"{report['divisions']['total']:,}",
            f"{(report['divisions']['link_rate'] or 0):.0%}",
            f"{report['bills']['bills']:,}")
    console.print(table)
    for report in reports:
        varying = report["time_varying_offices"]
        if varying:
            console.print(f"[dim]{report['window_id']} offices that changed hands inside the window: "
                          + ", ".join(f"{v['office']} ({v['holders']})" for v in varying[:5]) + "[/dim]")
        divisions = report["divisions"]
        if divisions["remote"]:
            console.print(
                f"[dim]{report['window_id']} divisions: {divisions['remote']} remote "
                f"({(divisions['remote_reconcile_rate'] or 0):.0%} reconcile) against "
                f"{divisions['physical']} physical "
                f"({(divisions['physical_reconcile_rate'] or 0):.0%}) -- I-8[/dim]")
        for problem in report["problems"]:
            console.print(f"[bold red]{report['window_id']}: {problem}[/bold red]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--window", action="append", help="only this window (repeatable)")
    p.add_argument("--report-only", action="store_true", help="do not rebuild the lexicon or offices")
    p.add_argument("--json", action="store_true")
    p.add_argument("--database", help="override the database name in PG_DSN")
    p.add_argument("--expect", default=safety.TARGET)
    sys.exit(run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
