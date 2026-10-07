"""Hybrid retrieval: dense + keyword + speaker legs fused with Reciprocal Rank Fusion, sibling collapse, a cross-
encoder rerank, and batched expansion of the winners to their surrounding sentences.
"""

import asyncio
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any, NamedTuple, Protocol

import psycopg
from pgvector import Vector
from psycopg import pq
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.db import VectorError, to_vector
from retrieval import arms, vector_index
from retrieval.collapse import OVERLAP_SPAN, Cluster, collapse
from retrieval.gpu_client import GpuError
from retrieval.lexicon import MAX_TERMS, Lexicon, load_lexicon, min_mass
from retrieval.query import MAX_QUERY_CHARS, clean_query
from retrieval.speakers import Resolution, SpeakerIndex, common_df_threshold, strip_mentions

# 100 because the pre-registered primary metric is pooled recall@100 -- the first-stage candidate ceiling, which bounds
# everything the reranker can do.
MAX_TOP_K = 100
RERANK_MODES = ("blend", "replace", "off")


class RetrievalError(Exception):
    """The database part of a search failed (unreachable, timed out, pool exhausted). Never carries the query."""


class Gpu(Protocol):
    async def embed_query(self, text: str) -> Vector: ...
    async def rerank(self, query: str, candidates: Sequence[str]) -> list[float | None]: ...


def _int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")
    return value


@dataclass(frozen=True)
class SearchFilters:
    """First-class metadata filters."""

    windows: tuple[str, ...] = ()
    date_from: Date | None = None
    date_to: Date | None = None
    houses: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    speakers: tuple[int, ...] = ()  # MNIS ids
    debates: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return any((self.windows, self.date_from, self.date_to, self.houses,
                    self.sections, self.speakers, self.debates))

    def predicate(self, alias: str = "c") -> str:
        """SQL to AND into a chunk predicate. Empty string when nothing is filtered."""
        parts = []
        if self.windows:
            parts.append(f"{alias}.window_id = ANY(%(f_windows)s::text[])")
        if self.date_from:
            parts.append(f"{alias}.debate_date >= %(f_date_from)s")
        if self.date_to:
            parts.append(f"{alias}.debate_date <= %(f_date_to)s")
        if self.houses:
            parts.append(f"{alias}.house = ANY(%(f_houses)s::text[])")
        if self.sections:
            parts.append(f"{alias}.section = ANY(%(f_sections)s::text[])")
        if self.speakers:
            parts.append(f"{alias}.mnis_id = ANY(%(f_speakers)s::int[])")
        if self.debates:
            parts.append(f"{alias}.debate_id = ANY(%(f_debates)s::text[])")
        return "".join(f" AND {p}" for p in parts)

    def params(self) -> dict[str, Any]:
        return {"f_windows": list(self.windows), "f_date_from": self.date_from,
                "f_date_to": self.date_to, "f_houses": list(self.houses),
                "f_sections": list(self.sections), "f_speakers": list(self.speakers),
                "f_debates": list(self.debates)}

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in (("windows", list(self.windows)), ("date_from", self.date_from),
                                  ("date_to", self.date_to), ("houses", list(self.houses)),
                                  ("sections", list(self.sections)), ("speakers", list(self.speakers)),
                                  ("debates", list(self.debates))) if v}


@dataclass(frozen=True)
class SearchConfig:
    dense: bool = True
    sparse: bool = True
    speaker: bool = True
    dense_limit: int = 50  # each leg is limited BEFORE the join, or rank inflation multiplies identical rows
    sparse_limit: int = 50
    speaker_limit: int = 50
    fused_limit: int = 100
    rrf_k: int = 60
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # dense, keyword, speaker
    ef_search: int = 100  # 98.3 % recall against an exact scan, against 96.3 % at the default 40
    statement_timeout_ms: int = 5000
    sparse_coverage: float = 0.4  # share of the query's IDF mass a sentence must cover to enter the keyword leg
    max_terms: int = MAX_TERMS
    k1: float = 1.2
    b: float = 0.75
    pool: int = 20  # clusters sent to the reranker
    rerank: str = "blend"
    rerank_k: int = 60
    expand_radius: int = 4  # sentences either side of a hit; 0 switches expansion off
    lexicon_ttl_s: float = 3600.0
    common_df: int = 50  # a surname in at least this many chunks is an ordinary word, not evidence of a name

    # --- Corpus v2 scale fixes.
    common_df_ratio: float | None = None
    # Narrow a shared surname to the members sitting on the filtered date range.
    speaker_date_aware: bool = False
    # Read the loader's lexeme_df table instead of running ts_stat over every tsvector.
    stored_lexicon: bool = False
    # Below this many matching chunks, scan the filtered set exactly instead of using the HNSW index.
    exact_scan_max_rows: int = 0

    # --- E1 bake-off arms.
    collapse_span: int = OVERLAP_SPAN
    # P50g.
    set_pool: int | None = None
    # D1.
    debate_boost: int = 0
    debate_speeches: int = 3
    # B1.
    stock_penalty: float = 1.0
    # R1.
    rerank_text_mode: str = "sentence"
    # Which HNSW index the dense leg's ORDER BY is written for: "halfvec" or "vector" (full precision, what E1 froze
    # on).
    dense_index: str = "halfvec"
    filters: SearchFilters = field(default_factory=SearchFilters)

    def __post_init__(self) -> None:
        for name in ("dense_limit", "sparse_limit", "speaker_limit"):
            _int(getattr(self, name), name, 0, 1000)
        _int(self.fused_limit, "fused_limit", 1, 1000)
        _int(self.rrf_k, "rrf_k", 1, 10_000)
        _int(self.ef_search, "ef_search", 1, 1000)  # pgvector's own range; also the only integer that reaches SQL text
        _int(self.statement_timeout_ms, "statement_timeout_ms", 0, 600_000)
        _int(self.pool, "pool", 1, 500)
        _int(self.rerank_k, "rerank_k", 1, 10_000)
        _int(self.expand_radius, "expand_radius", 0, 50)
        _int(self.collapse_span, "collapse_span", 0, 10**9)
        _int(self.debate_boost, "debate_boost", 0, 1000)
        _int(self.debate_speeches, "debate_speeches", 1, 100)
        if self.set_pool is not None:
            _int(self.set_pool, "set_pool", 1, 500)
        if not (isinstance(self.stock_penalty, (int, float)) and math.isfinite(self.stock_penalty)
                and 0.0 <= self.stock_penalty <= 1.0):  # fmt: skip
            raise ValueError("stock_penalty must be a number from 0 to 1")
        if self.dense_index not in vector_index.FORMS:
            raise ValueError(f"dense_index must be one of {vector_index.FORMS}, not {self.dense_index!r}")
        if self.rerank_text_mode not in RERANK_TEXT_MODES:
            raise ValueError(f"rerank_text_mode must be one of {RERANK_TEXT_MODES}, "
                             f"not {self.rerank_text_mode!r}")  # fmt: skip
        _int(self.max_terms, "max_terms", 1, 200)
        _int(self.common_df, "common_df", 0, 10**9)
        _int(self.exact_scan_max_rows, "exact_scan_max_rows", 0, 10**9)
        if self.common_df_ratio is not None and not 0.0 < self.common_df_ratio <= 1.0:
            raise ValueError("common_df_ratio must be between 0 and 1")
        if not isinstance(self.filters, SearchFilters):
            raise TypeError("filters must be a SearchFilters")
        if self.rerank not in RERANK_MODES:
            raise ValueError(f"rerank must be one of {RERANK_MODES}, not {self.rerank!r}")
        if len(self.weights) != 3 or not all(isinstance(w, (int, float)) and math.isfinite(w) and w >= 0 for w in self.weights):
            raise ValueError("weights must be three non-negative numbers")
        if not 0.0 <= self.sparse_coverage <= 1.0:
            raise ValueError("sparse_coverage must be between 0 and 1")


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    speech_id: str
    debate_id: str
    seq: int
    sent_idx: int
    body: str
    parent_body: str
    speaker: str | None
    party: str | None
    office: str | None
    heading: str | None
    rrf: float
    dense_rank: int | None
    sparse_rank: int | None
    speaker_rank: int | None
    date: Date | None = None
    house: str | None = None
    column_ref: str | None = None
    url: str | None = None


@dataclass
class Hit:
    rank: int
    score: float  # the score the final order used: blended RRF, the reranker's score, or the fused RRF
    chunk_id: int
    speech_id: str
    debate_id: str
    seq: int
    sent_idx: int
    sentence: str  # the matched sentence
    passage: str  # its stored parent window
    speaker: str | None
    party: str | None
    office: str | None
    heading: str | None
    house: str | None
    date: Date | None
    column_ref: str | None
    url: str | None
    rrf: float
    rerank_score: float | None
    legs: dict[str, int | None]  # the rank in each leg, None where the leg did not return it
    sibling_chunk_ids: tuple[int, ...]  # neighbouring hits folded into this one
    also_in_speeches: tuple[str, ...]  # other speeches holding the identical passage
    context: str | None = None  # the surrounding sentences, joined; None when expansion is off
    # (seq, sent_idx, body, column_ref, chunk_id); older producers give the first four only
    context_sentences: tuple[tuple, ...] | None = None


@dataclass
class SearchResult:
    query: str
    searched: str  # the cleaned query actually used
    hits: list[Hit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- pure decisions
# ---------------------------------------------------------------------------


# What `rerank_text` may read.
RERANK_TEXT_MODES = ("sentence", "window")


def rerank_text(c: Candidate, mode: str = "sentence") -> str:
    """What the reranker reads: the matched text under a header naming who said it and in which debate."""
    if mode not in RERANK_TEXT_MODES:
        raise ValueError(f"mode must be one of {RERANK_TEXT_MODES}, not {mode!r}")
    who = c.speaker or ""
    if who and c.party:
        who += f" ({c.party})"
    if who and c.office:
        who += f", {c.office}"
    if c.heading:
        prefix = f"{who}, in the debate '{c.heading}': " if who else f"In the debate '{c.heading}': "
    else:
        prefix = f"{who}: " if who else ""
    body = (c.parent_body or c.body) if mode == "window" else c.body
    return prefix + body


class Ranked(NamedTuple):
    index: int  # position in the fused candidate list
    score: float | None  # the score the order used; None when nothing but fusion ordered it


def rank_candidates(n: int, scores: Sequence[float | None] | None, mode: str, k: int = 60) -> list[Ranked]:
    """The final order of `n` candidates given best first by fusion, and the reranker's scores (None where it could not
    score one, or no scores at all). "off": the fused order.
    """
    if mode not in RERANK_MODES:
        raise ValueError(f"mode must be one of {RERANK_MODES}, not {mode!r}")
    if scores is not None and len(scores) != n:
        raise ValueError(f"{len(scores)} scores for {n} candidates")
    if mode == "off" or scores is None:
        return [Ranked(i, None) for i in range(n)]
    usable = [i for i in range(n) if scores[i] is not None and math.isfinite(scores[i])]
    by_score = sorted(usable, key=lambda i: (-scores[i], i))
    if mode == "replace":
        unscored = [i for i in range(n) if i not in set(usable)]
        return [Ranked(i, float(scores[i])) for i in by_score] + [Ranked(i, None) for i in unscored]
    rerank_rank = {i: r for r, i in enumerate(by_score, start=1)}
    blended = {i: 1 / (k + i + 1) + (1 / (k + rerank_rank[i]) if i in rerank_rank else 0.0) for i in range(n)}
    return [Ranked(i, blended[i]) for i in sorted(range(n), key=lambda i: (-blended[i], i))]


def assemble_context(anchor_seq: int, lo: int, hi: int, rows: Sequence[tuple]) -> list[tuple]:
    """From (seq, sent_idx, body, column_ref, chunk_id) rows of ONE speech, the unbroken run of sentences around the
    anchor inside [lo, hi]. A gap (a boilerplate sentence, the start or end of the speech) ends the run.
    """
    by_seq = {r[0]: r for r in rows if lo <= r[0] <= hi}
    if anchor_seq not in by_seq:
        return []
    start = end = anchor_seq
    while start - 1 in by_seq:
        start -= 1
    while end + 1 in by_seq:
        end += 1
    return [by_seq[s] for s in range(start, end + 1)]


# --------------------------------------------------------------------------- SQL
# ---------------------------------------------------------------------------

# The dense leg, two ways.
_DENSE_HNSW = """
dense AS (
    SELECT id, row_number() OVER (ORDER BY dist, id) AS rnk
    FROM (
        SELECT c.id, c.embedding <=> %(qvec)s::vector AS dist
        FROM chunk c
        WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate{filter}
        ORDER BY {order}
        LIMIT %(dense_limit)s
    ) d
),"""

_DENSE_EXACT = """
scoped AS MATERIALIZED (
    SELECT c.id, c.embedding FROM chunk c
    WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate{filter}
),
dense AS (
    SELECT id, row_number() OVER (ORDER BY dist, id) AS rnk
    FROM (
        SELECT id, embedding <=> %(qvec)s::vector AS dist FROM scoped
        ORDER BY 2, id LIMIT %(dense_limit)s
    ) d
),"""

_FUSE_SQL = """
WITH{dense_leg}
sparse AS (
    SELECT id, row_number() OVER (ORDER BY score DESC, id) AS rnk
    FROM (
        SELECT c.id,
               sum(t.idf * (%(k1)s + 1) / (1 + %(k1)s * (1 - %(b)s + %(b)s * length(c.ts) / %(avg_len)s))) AS score
        FROM unnest(%(lexemes)s::text[], %(idfs)s::float8[]) AS t(lexeme, idf)
        JOIN chunk c ON c.ts @@ plainto_tsquery('simple', t.lexeme)
        WHERE NOT c.is_boilerplate{filter}
        GROUP BY c.id
        HAVING sum(t.idf) >= %(min_mass)s
        ORDER BY score DESC, c.id
        LIMIT %(sparse_limit)s
    ) s
),
member_chunks AS MATERIALIZED (
    SELECT c.id, c.speech_id, c.embedding <=> %(qvec)s::vector AS dist
    FROM speech sp
    JOIN chunk c ON c.speech_id = sp.id
    WHERE sp.mnis_id = ANY(%(members)s::int[]) AND NOT c.is_boilerplate AND c.embedding IS NOT NULL{filter}
),
speaker AS (
    SELECT id, row_number() OVER (ORDER BY dist, id) AS rnk
    FROM (
        SELECT id, dist FROM (
            SELECT DISTINCT ON (speech_id) id, speech_id, dist FROM member_chunks ORDER BY speech_id, dist, id
        ) best
        ORDER BY dist, id
        LIMIT %(speaker_limit)s
    ) s
),
fused AS (
    SELECT id, d.rnk AS dense_rank, s.rnk AS sparse_rank, p.rnk AS speaker_rank,
           coalesce(%(w_dense)s::float8 / (%(k)s + d.rnk), 0)
         + coalesce(%(w_sparse)s::float8 / (%(k)s + s.rnk), 0)
         + coalesce(%(w_speaker)s::float8 / (%(k)s + p.rnk), 0) AS rrf
    FROM dense d
    FULL OUTER JOIN sparse s USING (id)
    FULL OUTER JOIN speaker p USING (id)
)
SELECT f.id AS chunk_id, f.rrf, f.dense_rank, f.sparse_rank, f.speaker_rank,
       cc.speech_id, cc.debate_id, cc.seq, cc.sent_idx, cc.body, cc.parent_body,
       coalesce(cc.member_name, cc.speaker_name) AS speaker,
       coalesce(cc.party_on_date_base, party_base(cc.party_as_attributed)) AS party,
       cc.speaker_office AS office, cc.heading, cc.date, cc.house, cc.column_ref, cc.hansard_url AS url,
       (SELECT count(*) FROM dense) AS n_dense, (SELECT count(*) FROM sparse) AS n_sparse, (SELECT count(*) FROM speaker) AS n_speaker,
       current_setting('hnsw.ef_search', true) AS ef_search, current_setting('hnsw.iterative_scan', true) AS iterative_scan,
       current_setting('statement_timeout') AS statement_timeout
FROM (SELECT id, dense_rank, sparse_rank, speaker_rank, rrf FROM fused ORDER BY rrf DESC, id LIMIT %(fused_limit)s) f
JOIN chunk_context cc ON cc.chunk_id = f.id
ORDER BY f.rrf DESC, f.id
"""

# How many matching chunks there are, counted only up to the threshold that matters.
_SCOPE_COUNT = "SELECT count(*) FROM (SELECT 1 FROM chunk c WHERE NOT c.is_boilerplate{filter} LIMIT %(probe)s) t"


def fuse_sql(filters: "SearchFilters", exact: bool, dense_index: str = "halfvec") -> str:
    """The fused query for these filters. `dense_index` changes only the HNSW leg's ORDER BY; the exact leg needs no
    index.
    """
    predicate = filters.predicate("c")
    if exact:
        leg = _DENSE_EXACT.format(filter=predicate)
    else:
        leg = _DENSE_HNSW.format(filter=predicate, order=vector_index.order_expression(dense_index, alias="c"))
    return _FUSE_SQL.format(dense_leg=leg, filter=predicate)

_LEXEMES_SQL = "SELECT lexeme FROM unnest(to_tsvector('english', %s))"

# The names a member SPOKE under — full names Hansard printed for them at least twice (`printed_full_name`, migration
# 021).
_SPEAKER_ALIASES = """
    WITH printed AS (
        SELECT s.mnis_id, btrim(s.speaker_person) AS alias, count(*) AS n
        FROM speech s
        WHERE s.mnis_id IS NOT NULL AND printed_full_name(s.speaker_person)
        GROUP BY 1, 2
    ), ranked AS (
        SELECT mnis_id, alias, lower(alias) AS as_printed,
               lower(regexp_replace(alias, '^(Mr|Mrs|Ms|Miss|Mx|Dr|Sir|Dame)[.]?[[:space:]]+', '')) AS untitled,
               n, max(n) OVER (PARTITION BY alias) AS most
        FROM printed
    ), named AS (
        SELECT mnis_id, lower(name) AS name FROM member_id_map WHERE name IS NOT NULL
    )
    SELECT r.mnis_id, r.alias
    FROM ranked r
    WHERE r.n >= 2 AND r.n = r.most
      AND NOT EXISTS (SELECT 1 FROM named o WHERE o.name = r.as_printed AND o.mnis_id <> r.mnis_id)
      AND NOT EXISTS (SELECT 1 FROM named o WHERE o.name = r.untitled AND o.mnis_id <> r.mnis_id)
"""

_SPEAKERS_SQL = (
    "SELECT m.mnis_id, m.name, m.name_list_as, m.name_full_title FROM member_id_map m"
    " WHERE EXISTS (SELECT 1 FROM speech s WHERE s.mnis_id = m.mnis_id)"
    f" UNION ALL SELECT a.mnis_id, a.alias, NULL, NULL FROM ({_SPEAKER_ALIASES}) a"
    " JOIN member_id_map m ON m.mnis_id = a.mnis_id"
)

# The same index plus when each member was sitting, so a surname four people share can be narrowed by the date a query
# is filtered to.
_SPEAKERS_DATED_SQL = f"""
    WITH base AS (
        SELECT m.mnis_id, m.name, m.name_list_as, m.name_full_title,
               least(m.membership_start_date, spoke.first_day) AS serving_from,
               CASE WHEN m.membership_end_date IS NULL THEN NULL
                    ELSE greatest(m.membership_end_date, spoke.last_day) END AS serving_to
        FROM member_id_map m
        JOIN (
            SELECT s.mnis_id, min(d.date) AS first_day, max(d.date) AS last_day
            FROM speech s JOIN debate d ON d.id = s.debate_id
            WHERE s.mnis_id IS NOT NULL GROUP BY s.mnis_id
        ) spoke ON spoke.mnis_id = m.mnis_id
    )
    SELECT * FROM base
    UNION ALL  -- R-03: every alias carries its member's own service dates
    SELECT a.mnis_id, a.alias, NULL, NULL, b.serving_from, b.serving_to
    FROM ({_SPEAKER_ALIASES}) a JOIN base b ON b.mnis_id = a.mnis_id
"""

# c.id rides along: the cards used to show neighbours WITHOUT their ids, so three sentences in four that the generator
# read were ones it was forbidden to cite.
_EXPAND_SQL = """
SELECT w.k, c.seq, c.sent_idx, c.body, c.column_ref, c.id
FROM unnest(%(ks)s::int[], %(debates)s::text[], %(speeches)s::text[], %(los)s::int[], %(his)s::int[]) AS w(k, debate_id, speech_id, lo, hi)
JOIN chunk c ON c.debate_id = w.debate_id AND c.seq BETWEEN w.lo AND w.hi AND c.speech_id = w.speech_id
WHERE NOT c.is_boilerplate
ORDER BY w.k, c.seq
"""


# --------------------------------------------------------------------------- the searcher
# ---------------------------------------------------------------------------


class HybridSearcher:
    def __init__(self, pool: AsyncConnectionPool, gpu: Gpu, config: SearchConfig | None = None):
        self.pool, self.gpu, self.config = pool, gpu, config or SearchConfig()
        self._lexicon: Lexicon | None = None
        self._speakers: SpeakerIndex | None = None
        self._common_df: int | None = None
        # Arm B1's stock-sentence set.
        self._stock: frozenset[str] | None = None
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Reload the corpus statistics and the speaker index (after an ingest); searches also do this on a timer.
        """
        async with self._lock:
            await self._load()

    async def _load(self) -> None:
        cfg = self.config
        try:
            async with self.pool.connection() as conn:
                async with conn.transaction():
                    lexicon = await load_lexicon(conn, stored=cfg.stored_lexicon)
                    sql = _SPEAKERS_DATED_SQL if cfg.speaker_date_aware else _SPEAKERS_SQL
                    rows = await (await conn.execute(sql)).fetchall()
        except psycopg.Error as exc:
            raise RetrievalError(f"could not load the search statistics ({type(exc).__name__})") from exc
        threshold = common_df_threshold(lexicon.n_docs, absolute=cfg.common_df, ratio=cfg.common_df_ratio)
        self._lexicon = lexicon
        self._common_df = threshold
        self._speakers = SpeakerIndex.from_rows(rows, is_common=lambda word: lexicon.df.get(word, 0) >= threshold)
        self._loaded_at = time.monotonic()

    async def _statistics(self, config: SearchConfig) -> tuple[Lexicon, SpeakerIndex]:
        if self._lexicon is None or time.monotonic() - self._loaded_at >= config.lexicon_ttl_s:
            async with self._lock:
                if self._lexicon is None or time.monotonic() - self._loaded_at >= config.lexicon_ttl_s:
                    await self._load()
        return self._lexicon, self._speakers

    async def _embed(self, text: str) -> tuple[Vector | None, str | None]:
        try:
            return to_vector(await self.gpu.embed_query(text)), None
        except (GpuError, VectorError) as exc:
            return None, f"the query could not be embedded ({type(exc).__name__}): dense and speaker search skipped, keyword only"

    async def _lexemes(self, text: str) -> list[str]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(_LEXEMES_SQL, (text,))).fetchall()
        return [r[0] for r in rows]

    async def search(self, query: object, top_k: int = 10, config: SearchConfig | None = None) -> SearchResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}")
        cfg = config or self.config
        started = time.perf_counter()
        cleaned, truncated = clean_query(query)
        result = SearchResult(query=query if isinstance(query, str) else "", searched=cleaned)
        if not cleaned:
            result.warnings.append("the query has nothing to search for")
            return result
        if truncated:
            result.warnings.append(f"the query was longer than {MAX_QUERY_CHARS} characters and was shortened")
        diag = result.diagnostics
        diag["queries"] = 0
        timings = diag["timings_ms"] = {}
        try:
            lexicon, speakers = await self._statistics(cfg)
            on_date = cfg.filters.date_from if cfg.speaker_date_aware else None
            resolution = speakers.resolve(cleaned, on_date) if cfg.speaker else Resolution()
            keyword_text = strip_mentions(cleaned, resolution) if resolution else cleaned
            diag["speakers"] = {"member_ids": list(resolution.member_ids), "mentions": list(resolution.mentions), "ambiguous": resolution.ambiguous}
            diag["keyword_text"] = keyword_text
            embedding = asyncio.ensure_future(self._embed(cleaned)) if (cfg.dense or (cfg.speaker and resolution)) else None
            try:
                lexemes = await self._lexemes(keyword_text) if cfg.sparse else []
                diag["queries"] += 1 if cfg.sparse else 0
            except BaseException:
                if embedding:
                    embedding.cancel()
                raise
            vector, embed_warning = await embedding if embedding else (None, None)
            timings["embed_and_tokenise"] = round((time.perf_counter() - started) * 1000, 1)
            if embed_warning:
                result.warnings.append(embed_warning)
            terms = lexicon.select_terms(lexemes, cfg.max_terms)
            diag["terms"] = [t for t, _ in terms]
            diag["dense"] = "off" if not cfg.dense else ("ok" if vector is not None else "unavailable")
            if vector is None and not terms:
                if not embed_warning:
                    result.warnings.append("nothing to search with: no vector and no keyword terms")
                diag["legs"] = {"dense": 0, "sparse": 0, "speaker": 0}
                return result

            params = {
                "qvec": vector,
                "dense_limit": cfg.dense_limit if (cfg.dense and vector is not None) else 0,
                "k1": cfg.k1, "b": cfg.b, "avg_len": lexicon.avg_len,
                "lexemes": [t for t, _ in terms], "idfs": [i for _, i in terms], "min_mass": min_mass(terms, cfg.sparse_coverage),
                "sparse_limit": cfg.sparse_limit if cfg.sparse else 0,
                "members": list(resolution.member_ids) if (cfg.speaker and vector is not None) else [],
                "speaker_limit": cfg.speaker_limit,
                "w_dense": cfg.weights[0], "w_sparse": cfg.weights[1], "w_speaker": cfg.weights[2], "k": cfg.rrf_k,
                "fused_limit": cfg.fused_limit,
                **cfg.filters.params(),
            }
            if cfg.filters:
                diag["filters"] = cfg.filters.as_dict()
            diag["common_df"] = self._common_df
            step = time.perf_counter()
            rows = await self._fuse(params, cfg, diag)
            timings["fuse"] = round((time.perf_counter() - step) * 1000, 1)
            candidates = [_candidate(r) for r in rows]
            if rows:
                diag["legs"] = {"dense": rows[0]["n_dense"], "sparse": rows[0]["n_sparse"], "speaker": rows[0]["n_speaker"]}
                diag["gucs"] = {"ef_search": rows[0]["ef_search"], "iterative_scan": rows[0]["iterative_scan"], "statement_timeout": rows[0]["statement_timeout"]}
            else:
                diag["legs"] = {"dense": 0, "sparse": 0, "speaker": 0}
            diag["fused"] = len(candidates)

            # --- E1 arms, in pipeline order.
            set_question = arms.looks_like_a_set_question(cleaned)
            diag["set_question"] = set_question
            pool = cfg.set_pool if (set_question and cfg.set_pool is not None) else cfg.pool
            diag["pool"] = pool
            if cfg.debate_boost and set_question:
                candidates, promoted = arms.boost_debates(
                    candidates, top_debates=cfg.debate_boost, per_debate=cfg.debate_speeches)
                diag["debate_boost"] = promoted
            if cfg.stock_penalty < 1.0:
                candidates, penalised = arms.penalise_stock(
                    candidates, self._stock_sentences(), cfg.stock_penalty)
                diag["stock_penalised"] = penalised
            clusters = collapse(candidates, span=cfg.collapse_span, limit=pool)
            diag["clusters"] = len(clusters)
            step = time.perf_counter()
            scores = await self._rerank(cleaned, clusters, cfg, result)
            ranked = rank_candidates(len(clusters), scores, cfg.rerank, cfg.rerank_k)
            timings["rerank"] = round((time.perf_counter() - step) * 1000, 1)

            top = ranked[:top_k]
            step = time.perf_counter()
            contexts = await self._expand([clusters[r.index] for r in top], cfg.expand_radius, diag)
            timings["expand"] = round((time.perf_counter() - step) * 1000, 1)
            for position, (r, context) in enumerate(zip(top, contexts), start=1):
                result.hits.append(_hit(position, r, clusters[r.index], scores, context))
        except psycopg.Error as exc:
            raise RetrievalError(f"the database search failed ({type(exc).__name__})") from exc
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)
        return result

    async def _fuse(self, params: dict[str, Any], cfg: SearchConfig, diag: dict[str, Any]) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            if conn.info.transaction_status != pq.TransactionStatus.IDLE:
                await conn.rollback()  # inside an open transaction, conn.transaction is only a savepoint and SET LOCAL would outlive it
            async with conn.transaction():
                # cfg has validated these as integers, so formatting them into SET LOCAL (which cannot take parameters)
                # is safe
                await conn.execute(f"SET LOCAL statement_timeout = {cfg.statement_timeout_ms}")
                await conn.execute(f"SET LOCAL hnsw.ef_search = {cfg.ef_search}")
                await conn.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                exact = await self._scope_is_small(conn, cfg, params, diag)
                cursor = conn.cursor(row_factory=dict_row)
                await cursor.execute(fuse_sql(cfg.filters, exact, cfg.dense_index), params)
                rows = await cursor.fetchall()
        diag["queries"] += 4
        return rows

    async def _scope_is_small(self, conn, cfg: SearchConfig, params: dict[str, Any],
                              diag: dict[str, Any]) -> bool:
        """Should the dense leg scan the filtered set exactly instead of using HNSW?"""
        if not cfg.exact_scan_max_rows or not cfg.filters:
            return False
        probe = cfg.exact_scan_max_rows + 1
        row = await (await conn.execute(_SCOPE_COUNT.format(filter=cfg.filters.predicate("c")),
                                        {**params, "probe": probe})).fetchone()
        matching = int(row[0]) if row else probe
        diag["scope_rows"] = f"{matching}{'+' if matching >= probe else ''}"
        diag["queries"] += 1
        return matching <= cfg.exact_scan_max_rows

    def _stock_sentences(self) -> frozenset[str]:
        """The normalised text of every sentence that occurs often enough to be boilerplate (arm B1)."""
        if self._stock is None:
            raise RetrievalError(
                "stock_penalty is on but the stock-sentence table has not been loaded. Build it with "
                "`scripts/pipeline.py run session15 --only stock_sentences`, then call load_stock_sentences()."
            )
        return self._stock

    def load_stock_sentences(self, sentences: Iterable[str]) -> int:
        """Supply the stock-sentence set for arm B1. Returns how many were loaded."""
        self._stock = frozenset(arms.normalise_sentence(s) for s in sentences)
        return len(self._stock)

    async def _rerank(self, query: str, clusters: list[Cluster[Candidate]], cfg: SearchConfig, result: SearchResult) -> list[float | None] | None:
        if cfg.rerank == "off" or len(clusters) < 2:  # one candidate needs no ordering
            return None
        try:
            scores = await self.gpu.rerank(query, [rerank_text(c.anchor, cfg.rerank_text_mode) for c in clusters])
        except GpuError as exc:
            result.warnings.append(f"the reranker was unavailable ({type(exc).__name__}): results are in fused order")
            return None
        result.diagnostics["reranked"] = len(clusters)
        return scores

    async def _expand(self, clusters: list[Cluster[Candidate]], radius: int, diag: dict[str, Any]) -> list[tuple[str, tuple] | None]:
        if radius <= 0 or not clusters:
            return [None] * len(clusters)
        ks, debates, speeches, los, his = [], [], [], [], []
        for k, cluster in enumerate(clusters):
            seqs = [m.seq for m in cluster.members if m.speech_id == cluster.anchor.speech_id]
            ks.append(k)
            debates.append(cluster.anchor.debate_id)
            speeches.append(cluster.anchor.speech_id)
            los.append(min(seqs) - radius)
            his.append(max(seqs) + radius)
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(_EXPAND_SQL, {"ks": ks, "debates": debates, "speeches": speeches, "los": los, "his": his})).fetchall()
        diag["queries"] += 1
        by_k: dict[int, list[tuple]] = {k: [] for k in ks}
        for k, seq, sent_idx, body, column_ref, chunk_id in rows:
            by_k[k].append((seq, sent_idx, body, column_ref, chunk_id))
        out: list[tuple[str, tuple] | None] = []
        for k, cluster in enumerate(clusters):
            run_rows = assemble_context(cluster.anchor.seq, los[k], his[k], by_k[k])
            out.append((" ".join(r[2] for r in run_rows), tuple(run_rows)) if run_rows else None)
        return out


def _candidate(row: dict[str, Any]) -> Candidate:
    return Candidate(
        chunk_id=row["chunk_id"], speech_id=row["speech_id"], debate_id=row["debate_id"], seq=row["seq"], sent_idx=row["sent_idx"],
        body=row["body"], parent_body=row["parent_body"], speaker=row["speaker"], party=row["party"], office=row["office"],
        heading=row["heading"], rrf=float(row["rrf"]), dense_rank=row["dense_rank"], sparse_rank=row["sparse_rank"],
        speaker_rank=row["speaker_rank"], date=row["date"], house=row["house"], column_ref=row["column_ref"], url=row["url"],
    )


def _hit(position: int, ranked: Ranked, cluster: Cluster[Candidate], scores: list[float | None] | None, context: tuple[str, tuple] | None) -> Hit:
    a = cluster.anchor
    raw = scores[ranked.index] if scores is not None else None
    rerank_score = float(raw) if raw is not None and math.isfinite(raw) else None
    return Hit(
        rank=position, score=ranked.score if ranked.score is not None else a.rrf, chunk_id=a.chunk_id, speech_id=a.speech_id,
        debate_id=a.debate_id, seq=a.seq, sent_idx=a.sent_idx, sentence=a.body, passage=a.parent_body, speaker=a.speaker,
        party=a.party, office=a.office, heading=a.heading, house=a.house, date=a.date, column_ref=a.column_ref, url=a.url,
        rrf=a.rrf, rerank_score=rerank_score,
        legs={"dense": a.dense_rank, "sparse": a.sparse_rank, "speaker": a.speaker_rank},
        sibling_chunk_ids=tuple(m.chunk_id for m in cluster.siblings),
        also_in_speeches=tuple(dict.fromkeys(m.speech_id for m in cluster.siblings if m.speech_id != a.speech_id)),
        context=context[0] if context else None,
        context_sentences=context[1] if context else None,
    )
