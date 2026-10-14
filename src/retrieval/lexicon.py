"""Corpus statistics for the keyword leg: how rare each lexeme is, so a match on "Seaton Carew" outweighs one on
"government". Postgres full-text ranking (ts_rank) ignores rarity, so the keyword leg computes BM25-style scores
itself in SQL and takes its IDF weights from here.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from psycopg import AsyncConnection

MAX_TERMS = 24  # a pasted paragraph has a hundred lexemes; the 24 rarest carry the meaning and bound the SQL work

Term = tuple[str, float]  # (lexeme, idf)


@dataclass(frozen=True)
class Lexicon:
    n_docs: int  # non-boilerplate chunks
    avg_len: float  # mean number of distinct lexemes per chunk (what SQL's length(ts) measures)
    df: dict[str, int] = field(default_factory=dict)  # lexeme -> number of chunks containing it

    def __post_init__(self) -> None:
        if not self.avg_len > 0:
            object.__setattr__(self, "avg_len", 1.0)  # the SQL divides by it

    @classmethod
    def empty(cls) -> "Lexicon":
        return cls(0, 1.0, {})

    def idf(self, lexeme: str) -> float:
        """Lucene's BM25 IDF: always positive, and largest for a lexeme that occurs nowhere."""
        df = self.df.get(lexeme, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def select_terms(self, lexemes: Iterable[str], max_terms: int = MAX_TERMS) -> list[Term]:
        """The distinct query lexemes that occur in the corpus, rarest first (ties alphabetical), at most `max_terms`.
        A lexeme that occurs nowhere can match nothing, so it is dropped.
        """
        terms = [(lx, self.idf(lx)) for lx in dict.fromkeys(lexemes) if self.df.get(lx, 0) > 0]
        terms.sort(key=lambda t: (-t[1], t[0]))
        return terms[:max_terms]


def min_mass(terms: Sequence[Term], coverage: float) -> float:
    """The share of the query's total IDF weight a sentence must cover to enter the keyword leg."""
    return coverage * sum(idf for _, idf in terms)


# The `ts_stat` scan, kept as the source of truth and as the thing the stored table is checked against.
SQL_TS_STAT = "SELECT word, ndoc FROM ts_stat('SELECT ts FROM chunk WHERE NOT is_boilerplate')"
SQL_SCALARS = "SELECT count(*), avg(length(ts)) FROM chunk WHERE NOT is_boilerplate"
SQL_STORED = "SELECT lexeme, df FROM lexeme_df"
SQL_STORED_SCALARS = "SELECT key, value FROM corpus_stat WHERE key IN ('n_docs', 'avg_len')"


async def build_lexicon(conn: AsyncConnection) -> Lexicon:
    """Compute the statistics with `ts_stat`. The slow, authoritative path."""
    rows = await (await conn.execute(SQL_TS_STAT)).fetchall()
    n_docs, avg_len = await (await conn.execute(SQL_SCALARS)).fetchone()
    return Lexicon(int(n_docs), float(avg_len or 0.0), {word: int(ndoc) for word, ndoc in rows})


async def store_lexicon(conn: AsyncConnection) -> Lexicon:
    """Recompute the statistics and write them to `lexeme_df` / `corpus_stat`."""
    lexicon = await build_lexicon(conn)
    async with conn.transaction():
        await conn.execute("TRUNCATE lexeme_df")
        if lexicon.df:
            await conn.execute(
                "INSERT INTO lexeme_df (lexeme, df) SELECT * FROM unnest(%s::text[], %s::bigint[])",
                (list(lexicon.df), [lexicon.df[w] for w in lexicon.df]))
        await conn.execute(
            "INSERT INTO corpus_stat (key, value) VALUES ('n_docs', %s), ('avg_len', %s), "
            "('built_at', extract(epoch FROM now())) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = now()",
            (float(lexicon.n_docs), lexicon.avg_len))
    return lexicon


async def read_lexicon(conn: AsyncConnection) -> Lexicon | None:
    """The stored statistics, or None when they have never been built."""
    scalars = dict(await (await conn.execute(SQL_STORED_SCALARS)).fetchall())
    if "n_docs" not in scalars or not scalars["n_docs"]:
        return None
    rows = await (await conn.execute(SQL_STORED)).fetchall()
    if not rows:
        return None
    return Lexicon(int(scalars["n_docs"]), float(scalars.get("avg_len") or 0.0),
                   {word: int(df) for word, df in rows})


async def load_lexicon(conn: AsyncConnection, *, stored: bool = False) -> Lexicon:
    """Statistics over the searchable chunks, from Postgres's own tokenizer so they match the index exactly."""
    if stored:
        lexicon = await read_lexicon(conn)
        if lexicon is not None:
            return lexicon
    return await build_lexicon(conn)
