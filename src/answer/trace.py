"""Persisting what happened, per request."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("answer.trace")

# Columns the table has.
COLUMNS = (
    "query_hash", "question", "route", "route_by", "route_why", "resolved_filters",
    "corpus_version", "retrieval_config_hash", "recipe", "prompt_version", "model_id", "served_model",
    "fingerprint", "verifier", "verifier_precision",
    "outcome", "abstention_reason", "cache_hit", "degraded", "degraded_reason", "failed_reason",
    "retrieval_hits", "cards", "card_sentences", "cards_dropped", "claims_generated", "claims_rejected",
    "claims_verified", "gold_evidence_retrieved", "gold_evidence_rank",
    "claims_dropped", "repairs", "rewrites",
    "retrieval_ms", "generation_ms", "verify_ms", "total_ms", "prompt_tokens", "completion_tokens",
    "cached_tokens", "cost_usd", "llm_calls",
    "run_id", "session_label", "gold_item_id", "gold_split", "gold_category", "is_evaluation",
)

# Fields stored as jsonb.
_JSON_FIELDS = frozenset({"resolved_filters"})


@dataclass
class Trace:
    """One request's story. Every field optional except the four the table requires."""

    query_hash: str
    route: str
    corpus_version: str
    retrieval_config_hash: str
    outcome: str = "answer"

    question: str | None = None
    route_by: str | None = None
    route_why: str | None = None
    resolved_filters: dict[str, Any] = field(default_factory=dict)
    recipe: str | None = None
    prompt_version: str | None = None
    model_id: str | None = None
    served_model: str | None = None
    fingerprint: str | None = None
    verifier: str | None = None
    verifier_precision: str | None = None
    abstention_reason: str | None = None
    cache_hit: bool = False
    degraded: bool = False
    degraded_reason: str | None = None
    failed_reason: str | None = None
    retrieval_hits: int | None = None
    cards: int | None = None
    # How many citable sentences those cards held.
    card_sentences: int | None = None
    # Evaluation only, and three-valued on purpose.
    gold_evidence_retrieved: bool | None = None
    # Where the best gold speech landed in the deck, 1-based.
    gold_evidence_rank: int | None = None
    cards_dropped: int | None = None
    claims_generated: int | None = None
    claims_rejected: int | None = None
    claims_verified: int | None = None
    claims_dropped: int | None = None
    repairs: int = 0
    rewrites: int = 0
    retrieval_ms: int | None = None
    generation_ms: int | None = None
    verify_ms: int | None = None
    total_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    llm_calls: int = 0
    run_id: str | None = None
    session_label: str | None = None
    gold_item_id: str | None = None
    gold_split: str | None = None
    gold_category: str | None = None
    is_evaluation: bool = False

    def as_row(self) -> dict[str, Any]:
        row = {k: v for k, v in asdict(self).items() if k in COLUMNS}
        for name in _JSON_FIELDS:
            row[name] = json.dumps(row.get(name) or {}, default=str)
        return row


async def record(conn, trace: Trace) -> int | None:
    """Write one trace row. Returns its id, or None if it could not be written."""
    row = trace.as_row()
    names = [n for n in COLUMNS if n in row]
    placeholders = ", ".join(["%s"] * len(names))
    try:
        cur = await conn.execute(
            f"INSERT INTO answer_trace ({', '.join(names)}) VALUES ({placeholders}) RETURNING id",
            [row[n] for n in names],
        )
        rows = await cur.fetchall()
        return int(rows[0][0]) if rows else None
    except Exception as exc:  # noqa: BLE001 - telemetry must not break the thing it is observing
        log.warning("could not write a trace row (%s: %s)", type(exc).__name__,
                    " ".join(str(exc).split())[:200])  # fmt: skip
        return None


async def summary(conn, run_id: str | None = None) -> dict[str, Any]:
    """The numbers asks to be able to see: outcomes, cache hits, degradation, latency and spend by route."""
    where, params = ("WHERE run_id = %s", (run_id,)) if run_id else ("", ())
    cur = await conn.execute(f"""
        SELECT count(*) AS requests,
               count(*) FILTER (WHERE outcome = 'answer')  AS answers,
               count(*) FILTER (WHERE outcome = 'clarify') AS clarifications,
               count(*) FILTER (WHERE outcome = 'abstain') AS abstentions,
               count(*) FILTER (WHERE outcome = 'error')   AS errors,
               count(*) FILTER (WHERE cache_hit)           AS cache_hits,
               count(*) FILTER (WHERE degraded)            AS degraded,
               count(*) FILTER (WHERE failed_reason IS NOT NULL) AS failed,
               COALESCE(sum(repairs), 0)   AS repairs,
               COALESCE(sum(rewrites), 0)  AS rewrites,
               COALESCE(sum(llm_calls), 0) AS llm_calls,
               COALESCE(sum(cost_usd), 0)  AS cost_usd,
               COALESCE(sum(prompt_tokens), 0)     AS prompt_tokens,
               COALESCE(sum(completion_tokens), 0) AS completion_tokens,
               COALESCE(sum(cached_tokens), 0)     AS cached_tokens,
               COALESCE(sum(claims_verified), 0)   AS claims_verified,
               COALESCE(sum(claims_dropped), 0)    AS claims_dropped,
               COALESCE(sum(claims_rejected), 0)   AS claims_rejected,
               percentile_disc(0.5)  WITHIN GROUP (ORDER BY total_ms) AS median_ms,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY total_ms) AS p95_ms
        FROM answer_trace {where}
    """, params)
    columns = [c.name for c in cur.description or []]
    rows = await cur.fetchall()
    overall = dict(zip(columns, rows[0], strict=True)) if rows else {}

    cur = await conn.execute(f"""
        SELECT route, outcome, count(*) AS n,
               percentile_disc(0.95) WITHIN GROUP (ORDER BY total_ms) AS p95_ms
        FROM answer_trace {where}
        GROUP BY route, outcome ORDER BY route, outcome
    """, params)
    columns = [c.name for c in cur.description or []]
    by_route = [dict(zip(columns, row, strict=True)) for row in await cur.fetchall()]
    return {"overall": overall, "by_route": by_route}

# Columns whose job is to carry a measurement, and the rows they could apply to.
_RETRIEVING = "route IN ('retrieval', 'overview', 'set')"
_GENERATING = "coalesce(llm_calls, 0) > 0"

DIAGNOSTIC_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("retrieval_hits", _RETRIEVING),
    ("cards", _RETRIEVING),
    ("card_sentences", _RETRIEVING),
    ("cards_dropped", _RETRIEVING),
    ("retrieval_ms", _RETRIEVING),
    ("claims_generated", _GENERATING),
    ("claims_verified", _GENERATING),
    ("claims_dropped", _GENERATING),
    ("generation_ms", _GENERATING),
    ("verify_ms", _GENERATING),
    ("prompt_tokens", _GENERATING),
    ("completion_tokens", _GENERATING),
    ("cached_tokens", _GENERATING),
    # These apply to every request, answered or not.
    ("total_ms", None),
    ("cost_usd", None),
    ("llm_calls", None),
)

# Columns that are legitimately constant in a run where the thing they count never happened.
MAY_BE_CONSTANT = frozenset({"rewrites", "repairs", "cached_tokens", "cards_dropped", "gold_evidence_retrieved",
                            "cost_usd", "llm_calls"})  # fmt: skip

# Below this many eligible rows, neither "dead" nor "constant" means anything — a column with one eligible row has one
# distinct value by arithmetic.
MIN_ELIGIBLE_ROWS = 5


async def liveness(conn, run_id: str) -> dict[str, Any]:
    """Which diagnostic columns actually carried a measurement in this run."""
    rows, dead, constant, not_applicable = {}, [], [], []
    for column, applies in DIAGNOSTIC_COLUMNS:
        where = f"run_id = %s AND {applies}" if applies else "run_id = %s"
        cur = await conn.execute(
            f"SELECT count(*), count({column}), count(DISTINCT {column}) FROM answer_trace WHERE {where}",
            (run_id,),
        )
        total, populated, distinct = await cur.fetchone()
        rows[column] = {"eligible_rows": total, "populated": populated, "distinct": distinct}
        if total < MIN_ELIGIBLE_ROWS:
            # Too few rows this column could apply to for either verdict to mean anything.
            not_applicable.append(column)
        elif not populated:
            dead.append(column)
        elif distinct <= 1 and column not in MAY_BE_CONSTANT:
            constant.append(column)
    return {"columns": rows, "dead": dead, "constant": constant, "not_applicable": not_applicable,
            "ok": not dead and not constant}  # fmt: skip
