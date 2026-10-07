"""Parent/child chunking of speeches."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from psycopg import AsyncConnection
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
from ingest.rules import sentence_boilerplate_reason
from ingest.runtime import run
from ingest.text import (  # noqa: F401  (clean_text and split_sentences are re-exported for callers and tests)
    Marker,
    assign_columns,
    clean_text,
    clean_text_with_markers,
    previous_ref,
    split_sentence_spans,
    split_sentences,
)

DEFAULT_WINDOW = 5
# Reporting only: the embedding model truncates long inputs, so we surface how many children are affected instead of
# silently hiding it.
LONG_SENTENCE_CHARS = 1500

console = Console()


class ChunkDraft(NamedTuple):
    sent_idx: int  # 0-based position within the speech (counts every sentence, boilerplate included)
    body: str  # the child sentence
    parent_body: str  # the window of sentences around it
    is_boilerplate: bool = False
    column_ref: str | None = None  # Hansard column the sentence starts in, as printed
    column_number: int | None = None
    column_end_ref: str | None = None  # only when it runs into another column
    column_basis: str | None = None  # marker | carried | inferred


# --------------------------------------------------------------------------- Parent windows
# ---------------------------------------------------------------------------


def _radius(window: int) -> int:
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number, got {window}")
    return window // 2


def window_bounds(index: int, total: int, window: int) -> tuple[int, int]:
    """Half-open [lo, hi) slice of the sentences around `index`."""
    radius = _radius(window)
    if not 0 <= index < total:
        raise ValueError(f"index {index} out of range for {total} sentences")
    return max(0, index - radius), min(total, index + radius + 1)


def build_chunks(sentences: Sequence[str], window: int = DEFAULT_WINDOW, boilerplate: Sequence[bool] | None = None) -> list[ChunkDraft]:
    """One draft per sentence. Windows are taken over the NON-boilerplate sentences only, so a boilerplate sentence
    never appears in anyone else's parent; a boilerplate child's parent is itself.
    """
    _radius(window)  # reject a bad window even for an empty speech
    total = len(sentences)
    if boilerplate is None:
        boilerplate = [False] * total
    elif len(boilerplate) != total:
        raise ValueError(f"boilerplate mask has {len(boilerplate)} entries for {total} sentences")
    kept = [i for i, flagged in enumerate(boilerplate) if not flagged]
    position = {i: p for p, i in enumerate(kept)}
    chunks = []
    for i, sentence in enumerate(sentences):
        if boilerplate[i]:
            chunks.append(ChunkDraft(i, sentence, sentence, True))
            continue
        lo, hi = window_bounds(position[i], len(kept), window)
        chunks.append(ChunkDraft(i, sentence, " ".join(sentences[k] for k in kept[lo:hi])))
    return chunks


# --------------------------------------------------------------------------- Speech and debate level
# ---------------------------------------------------------------------------

Carry = tuple[str, str] | None  # (column ref, basis) for text before any marker


class Prepared(NamedTuple):
    text: str
    markers: list[Marker]


def prepare(raw_text: str | None) -> Prepared:
    return Prepared(*clean_text_with_markers(raw_text))


def chunk_prepared(prepared: Prepared, window: int, speaker_name: str | None, carry: Carry) -> tuple[list[ChunkDraft], Carry]:
    """Chunk one speech. Returns the drafts and the column carry for the next speech in the debate."""
    spans = split_sentence_spans(prepared.text)
    flags = [sentence_boilerplate_reason(s.text, speaker_name) is not None for s in spans]
    drafts = build_chunks([s.text for s in spans], window, flags)
    columns = assign_columns(spans, prepared.markers, carry)
    drafts = [
        d._replace(column_ref=c.ref, column_number=c.number, column_end_ref=c.end_ref, column_basis=c.basis)
        for d, c in zip(drafts, columns)
    ]
    return drafts, ((prepared.markers[-1].ref, "carried") if prepared.markers else carry)


def chunk_debate_texts(items: Sequence[tuple[str | None, str | None]], window: int = DEFAULT_WINDOW) -> list[list[ChunkDraft]]:
    """Chunk every speech of one debate, in order. `items` are (speaker_name, raw_text)."""
    prepared = [prepare(text) for _, text in items]
    first = next((p.markers[0].ref for p in prepared if p.markers), None)
    inferred = previous_ref(first)
    carry: Carry = (inferred, "inferred") if inferred else None
    out = []
    for (speaker, _), prep in zip(items, prepared):
        drafts, carry = chunk_prepared(prep, window, speaker, carry)
        out.append(drafts)
    return out


def speech_to_chunks(raw_text: str | None, window: int = DEFAULT_WINDOW, speaker_name: str | None = None) -> list[ChunkDraft]:
    """One speech in isolation: no column carry, and chair-only formulas need `speaker_name` to be flagged."""
    return chunk_prepared(prepare(raw_text), window, speaker_name, None)[0]


# --------------------------------------------------------------------------- Database
# ---------------------------------------------------------------------------

SQL_PENDING_DEBATES = """
    SELECT DISTINCT s.debate_id
    FROM speech s
    WHERE btrim(coalesce(s.text, '')) <> ''
      AND (%(rebuild)s OR NOT EXISTS (SELECT 1 FROM chunk c WHERE c.speech_id = s.id))
    ORDER BY s.debate_id
    LIMIT %(limit)s
"""
# With the debate's window, date, House and section and the speech's speaker: the chunk's filter columns are written
# HERE, not only by the later `load_corpus_v2.py --sync-only`.
SQL_DEBATE_SPEECHES = """
    SELECT s.id, s.speaker_name, s.text, d.window_id, d.date, d.house, d.section, s.mnis_id
      FROM speech s JOIN debate d ON d.id = s.debate_id
     WHERE s.debate_id = %s ORDER BY s.seq, s.id
"""
# The embedding (and its provenance) survives only if the sentence is unchanged (CASE has no ELSE, so a changed body
# resets them to NULL for Phase 2 to re-embed).
SQL_UPSERT_CHUNK = """
    INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body,
                       is_boilerplate, column_number, column_ref, column_end_ref, column_basis,
                       window_id, debate_date, house, section, mnis_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (speech_id, sent_idx) DO UPDATE SET
        debate_id       = EXCLUDED.debate_id,
        seq             = EXCLUDED.seq,
        parent_body     = EXCLUDED.parent_body,
        is_boilerplate  = EXCLUDED.is_boilerplate,
        column_number   = EXCLUDED.column_number,
        column_ref      = EXCLUDED.column_ref,
        column_end_ref  = EXCLUDED.column_end_ref,
        column_basis    = EXCLUDED.column_basis,
        window_id       = EXCLUDED.window_id,
        debate_date     = EXCLUDED.debate_date,
        house           = EXCLUDED.house,
        section         = EXCLUDED.section,
        mnis_id         = EXCLUDED.mnis_id,
        embedding       = CASE WHEN chunk.body = EXCLUDED.body THEN chunk.embedding END,
        embedding_model = CASE WHEN chunk.body = EXCLUDED.body THEN chunk.embedding_model END,
        embedded_at     = CASE WHEN chunk.body = EXCLUDED.body THEN chunk.embedded_at END,
        -- the recipe goes with the vector it describes: kept alone it claimed an embedding that no longer existed (D15)
        embedding_recipe = CASE WHEN chunk.body = EXCLUDED.body THEN chunk.embedding_recipe END,
        body            = EXCLUDED.body
    WHERE (chunk.debate_id, chunk.seq, chunk.body, chunk.parent_body, chunk.is_boilerplate,
           chunk.column_number, chunk.column_ref, chunk.column_end_ref, chunk.column_basis,
           chunk.window_id, chunk.debate_date, chunk.house, chunk.section, chunk.mnis_id)
          IS DISTINCT FROM
          (EXCLUDED.debate_id, EXCLUDED.seq, EXCLUDED.body, EXCLUDED.parent_body, EXCLUDED.is_boilerplate,
           EXCLUDED.column_number, EXCLUDED.column_ref, EXCLUDED.column_end_ref, EXCLUDED.column_basis,
           EXCLUDED.window_id, EXCLUDED.debate_date, EXCLUDED.house, EXCLUDED.section, EXCLUDED.mnis_id)
"""
# Drop chunks that no longer exist, e.g. a speech that now has fewer sentences.
SQL_DELETE_STALE = """
    DELETE FROM chunk c
    WHERE c.debate_id = %s
      AND NOT EXISTS (
          SELECT 1 FROM unnest(%s::text[], %s::int[]) AS keep(speech_id, sent_idx)
          WHERE keep.speech_id = c.speech_id AND keep.sent_idx = c.sent_idx
      )
"""


@dataclass
class Stats:
    debates: int = 0
    speeches: int = 0
    empty_speeches: int = 0  # non-blank text that still yielded no sentence
    chunks: int = 0
    boilerplate_chunks: int = 0
    stale_deleted: int = 0
    long_sentences: int = 0
    longest_sentence: int = 0
    column_basis: dict[str | None, int] = None

    def __post_init__(self):
        self.column_basis = {}


async def chunk_debate(conn: AsyncConnection, debate_id: str, window: int, stats: Stats) -> None:
    """Rebuild every chunk of one debate in a single transaction."""
    async with conn.transaction():
        cur = await conn.execute(SQL_DEBATE_SPEECHES, (debate_id,))
        speeches = await cur.fetchall()
        per_speech = chunk_debate_texts([(name, text) for _, name, text, *_ in speeches], window)

        rows: list[tuple] = []
        seq = 0
        for (speech_id, _, _, window_id, day, house, section, mnis_id), drafts in zip(speeches, per_speech):
            stats.speeches += 1
            if not drafts:
                stats.empty_speeches += 1
            for d in drafts:
                rows.append((speech_id, debate_id, seq, d.sent_idx, d.body, d.parent_body,
                             d.is_boilerplate, d.column_number, d.column_ref, d.column_end_ref, d.column_basis,
                             window_id, day, house, section, mnis_id))  # fmt: skip
                seq += 1
                stats.boilerplate_chunks += d.is_boilerplate
                stats.column_basis[d.column_basis] = stats.column_basis.get(d.column_basis, 0) + 1
                stats.longest_sentence = max(stats.longest_sentence, len(d.body))
                stats.long_sentences += len(d.body) > LONG_SENTENCE_CHARS

        async with conn.cursor() as cur:
            await cur.executemany(SQL_UPSERT_CHUNK, rows)
            await cur.execute(
                SQL_DELETE_STALE,
                (debate_id, [r[0] for r in rows], [r[3] for r in rows]),
            )
            stats.stale_deleted += cur.rowcount
        stats.chunks += len(rows)
        stats.debates += 1


async def run_chunking(window: int, rebuild: bool, limit: int | None) -> Stats:
    _radius(window)
    stats = Stats()
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            cur = await conn.execute(SQL_PENDING_DEBATES, {"rebuild": rebuild, "limit": limit})
            debate_ids = [row[0] for row in await cur.fetchall()]
            # That SELECT opened an implicit transaction, and inside an open transaction `conn.transaction` is only a
            # savepoint (gotcha 37).
            await conn.commit()
            await conn.set_autocommit(True)

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]Chunking debates"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.fields[chunks]:,} chunks"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("chunk", total=len(debate_ids), chunks=0)
                for debate_id in debate_ids:
                    await chunk_debate(conn, debate_id, window, stats)
                    progress.update(task, advance=1, chunks=stats.chunks)
            # The window table's chunk counts are only true after chunking; see SQL_REFRESH_WINDOW_COUNTS.
            from ingest.load import refresh_window_counts

            await refresh_window_counts(conn)
    finally:
        await pool.close()
    return stats


def print_summary(stats: Stats, window: int) -> None:
    table = Table(title=f"Chunking complete (window={window})")
    table.add_column("")
    table.add_column("Count", justify="right")
    table.add_row("Debates processed", f"{stats.debates:,}")
    table.add_row("Speeches read", f"{stats.speeches:,}")
    table.add_row("Speeches with no sentences (blank/markup only)", f"{stats.empty_speeches:,}")
    table.add_row("Chunks in sync (unchanged rows are skipped)", f"[bold green]{stats.chunks:,}[/bold green]")
    table.add_row("  flagged boilerplate (left out of parent windows)", f"{stats.boilerplate_chunks:,}")
    for basis in ("marker", "carried", "inferred", None):
        table.add_row(f"  column basis: {basis or 'unknown'}", f"{stats.column_basis.get(basis, 0):,}")
    table.add_row("Stale chunks removed", f"{stats.stale_deleted:,}")
    table.add_row(f"Sentences over {LONG_SENTENCE_CHARS:,} chars", f"{stats.long_sentences:,}")
    table.add_row("Longest sentence (chars)", f"{stats.longest_sentence:,}")
    console.print(table)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk speeches into sentence rows with parent windows.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="sentences per parent window, odd (default: %(default)s)")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-chunk every debate, not just those missing chunks")
    parser.add_argument("--limit", type=int, default=None, help="only process this many debates")
    args = parser.parse_args(argv)
    if args.window < 1 or args.window % 2 == 0:
        parser.error("--window must be a positive odd number")

    stats = run(run_chunking(args.window, args.rebuild, args.limit))
    print_summary(stats, args.window)


if __name__ == "__main__":
    main()
