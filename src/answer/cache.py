"""The answer cache: read and write, against migration 007's schema."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

from answer.contract import Answer
from answer.keys import CORPUS_VERSION, question_hash

log = logging.getLogger("answer.cache")

# The columns that together identify one answer.
KEY_COLUMNS = ("query_hash", "corpus_version", "retrieval_config_hash", "recipe", "prompt_version", "model_id",
               "code_version")


class CacheError(RuntimeError):
    """A refusal to store, always because the answer is not fit to be reused."""


@dataclass
class Key:
    """Everything that identifies one cached answer."""

    query_hash: str
    corpus_version: str
    retrieval_config_hash: str
    recipe: str
    prompt_version: str
    model_id: str
    code_version: str | None = None

    @classmethod
    def of(cls, answer: Answer, filters: dict[str, Any] | None = None) -> Key:
        provenance = answer.provenance
        return cls(
            query_hash=question_hash(answer.question, filters or provenance.resolved_filters, provenance.route),
            corpus_version=provenance.corpus_version,
            retrieval_config_hash=provenance.retrieval_config_hash,
            recipe=provenance.recipe,
            prompt_version=provenance.prompt_version,
            model_id=provenance.model_id,
            code_version=provenance.code_version,
        )  # fmt: skip

    def as_params(self) -> tuple:
        return (self.query_hash, self.corpus_version, self.retrieval_config_hash,
                self.recipe, self.prompt_version, self.model_id, self.code_version)  # fmt: skip


_WHERE = " AND ".join(f"{column} = %s" for column in KEY_COLUMNS)


async def lookup(conn, key: Key) -> dict[str, Any] | None:
    """The cached row, or None. Bumps the hit count on a hit."""
    cur = await conn.execute(f"""
        UPDATE answer_cache
           SET hit_count = hit_count + 1, last_hit_at = now()
         WHERE {_WHERE}
        RETURNING id, question, route, outcome, abstention_reason, abstention_detail, answer, answer_json,
                  coverage_note, structured_facts, cited_chunk_ids, cited_speech_ids, claim_count,
                  claims_verified, claims_dropped, verifier, verifier_precision, verified_at, created_at,
                  hit_count, degraded, degraded_reason, total_ms, cost_usd
    """, key.as_params())
    rows = await cur.fetchall()
    if not rows:
        return None
    columns = [c.name for c in cur.description or []]
    return dict(zip(columns, rows[0], strict=True))


def _ids(answer: Answer) -> tuple[list[int], list[str]]:
    chunks = sorted({i for claim in answer.claims for i in claim.cited_chunk_ids})
    speeches = sorted({s for claim in answer.claims for s in claim.cited_speech_ids})
    return chunks, speeches


async def store(conn, answer: Answer, *, filters: dict[str, Any] | None = None,
                verifier: str, verifier_precision: str = "", chunk_count: int | None = None,
                retrieved_chunk_ids: list[int] | None = None, claims_dropped: int = 0,
                degraded: bool = False, degraded_reason: str = "", total_ms: int | None = None,
                retrieval_ms: int | None = None, generation_ms: int | None = None,
                verify_ms: int | None = None, prompt_tokens: int | None = None,
                completion_tokens: int | None = None, cost_usd: float | None = None,
                gold_item_id: str | None = None, gold_split: str | None = None,
                gold_category: str | None = None, run_id: str | None = None,
                session_label: str | None = None, is_evaluation: bool = False,
                window_label: str | None = None) -> str:  # fmt: skip
    """Write one answer, or refuse."""
    if not verifier:
        raise CacheError("refusing to cache an answer with no verifier recorded")
    unverified = [c.id for c in answer.claims if not c.verification.showable]
    if unverified:
        raise CacheError(
            f"refusing to cache an answer whose claim(s) {unverified} did not pass verification; a cache outlives "
            "the process that failed to check them")

    key = Key.of(answer, filters)
    outcome = "abstain" if answer.abstention else ("clarify" if answer.facets.get("clarify") else "answer")
    chunks, speeches = _ids(answer)
    dates = [c.date for c in answer.claims if c.date]
    houses = sorted({c.house for c in answer.claims if c.house})
    sections = sorted({c.section for c in answer.claims if c.section})

    cur = await conn.execute("""
        INSERT INTO answer_cache (
            query_hash, corpus_version, retrieval_config_hash, recipe, prompt_version, model_id, code_version,
            system_fingerprint, chunk_count, question, question_norm, route, resolved_filters,
            asked_house, asked_section, asked_from, asked_to, window_label,
            outcome, abstention_reason, abstention_detail, coverage_note, answer, answer_json,
            structured_facts, facets,
            verified_at, verifier, verifier_precision, claim_count, claims_verified, claims_dropped,
            degraded, degraded_reason,
            retrieval_ms, generation_ms, verify_ms, total_ms, prompt_tokens, completion_tokens, cost_usd,
            retrieved_chunk_ids, cited_chunk_ids, cited_speech_ids, speaker_mnis_ids, retrieved_count,
            gold_item_id, gold_split, gold_category, run_id, session_label, is_evaluation
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s,
            now(), %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (query_hash, corpus_version, retrieval_config_hash, recipe, prompt_version, model_id, code_version)
        DO UPDATE SET
            -- A re-answer under identical provenance replaces the row rather than failing. Identical provenance
            -- means the same system answering the same question, so the newer run is the better record of it, and
            -- the hit count is deliberately preserved.
            outcome = EXCLUDED.outcome, abstention_reason = EXCLUDED.abstention_reason,
            abstention_detail = EXCLUDED.abstention_detail, coverage_note = EXCLUDED.coverage_note,
            answer = EXCLUDED.answer, answer_json = EXCLUDED.answer_json,
            verified_at = now(), verifier = EXCLUDED.verifier,
            verifier_precision = EXCLUDED.verifier_precision,
            claim_count = EXCLUDED.claim_count, claims_verified = EXCLUDED.claims_verified,
            claims_dropped = EXCLUDED.claims_dropped, degraded = EXCLUDED.degraded,
            degraded_reason = EXCLUDED.degraded_reason, total_ms = EXCLUDED.total_ms,
            cost_usd = EXCLUDED.cost_usd, cited_chunk_ids = EXCLUDED.cited_chunk_ids,
            cited_speech_ids = EXCLUDED.cited_speech_ids, updated_at = now()
        RETURNING query_hash
    """, (
        *key.as_params(),
        answer.provenance.system_fingerprint, chunk_count,
        answer.question, question_hash(answer.question, filters or {}, answer.provenance.route),
        answer.provenance.route, json.dumps(answer.provenance.resolved_filters, default=str),
        houses[0] if len(houses) == 1 else None, sections[0] if len(sections) == 1 else None,
        min(dates) if dates else None, max(dates) if dates else None, window_label,
        outcome, answer.abstention.value if answer.abstention else None,
        answer.abstention_detail or None, answer.coverage_note or None,
        answer.rendered or None, answer.model_dump_json(),
        json.dumps([f.model_dump() for f in answer.structured_facts], default=str),
        json.dumps(answer.facets, default=str),
        verifier, verifier_precision or None,
        len(answer.claims) + claims_dropped, len(answer.claims), claims_dropped,
        degraded, degraded_reason or None,
        retrieval_ms, generation_ms, verify_ms, total_ms, prompt_tokens, completion_tokens, cost_usd,
        sorted(retrieved_chunk_ids or answer.retrieved_chunk_ids), chunks, speeches,
        sorted({c.speaker_mnis_id for c in answer.claims if c.speaker_mnis_id}),
        len(retrieved_chunk_ids or answer.retrieved_chunk_ids),
        gold_item_id, gold_split, gold_category, run_id, session_label, is_evaluation,
    ))  # fmt: skip
    rows = await cur.fetchall()
    return str(rows[0][0]) if rows else key.query_hash


async def invalidate(conn, *, corpus_version: str | None = None, model_id: str | None = None,
                     prompt_version: str | None = None,
                     retrieval_config_hash: str | None = None) -> int:  # fmt: skip
    """Delete every entry that does NOT match the current system."""
    clauses, params = [], []
    for column, value in (("corpus_version", corpus_version), ("model_id", model_id),
                          ("prompt_version", prompt_version),
                          ("retrieval_config_hash", retrieval_config_hash)):  # fmt: skip
        if value is not None:
            clauses.append(f"{column} <> %s")
            params.append(value)
    if not clauses:
        raise CacheError("invalidate needs at least one current value, or it would delete the whole cache")
    cur = await conn.execute(f"DELETE FROM answer_cache WHERE {' OR '.join(clauses)}", tuple(params))
    return cur.rowcount or 0


async def stats(conn) -> dict[str, Any]:
    """What the cache holds, for the session report and for "most asked"."""
    cur = await conn.execute("""
        SELECT count(*) AS entries,
               count(*) FILTER (WHERE outcome = 'answer') AS answers,
               count(*) FILTER (WHERE outcome = 'abstain') AS abstentions,
               count(*) FILTER (WHERE outcome = 'clarify') AS clarifications,
               count(*) FILTER (WHERE degraded) AS degraded,
               COALESCE(sum(hit_count), 0) AS hits,
               COALESCE(sum(cost_usd), 0) AS cost_usd,
               count(DISTINCT model_id) AS models,
               count(DISTINCT prompt_version) AS prompts,
               count(DISTINCT corpus_version) AS corpora
        FROM answer_cache
    """)
    rows = await cur.fetchall()
    columns = [c.name for c in cur.description or []]
    return dict(zip(columns, rows[0], strict=True)) if rows else {}


async def most_asked(conn, limit: int = 10) -> list[dict[str, Any]]:
    cur = await conn.execute("""
        SELECT question, route, outcome, hit_count, last_hit_at
        FROM answer_cache WHERE hit_count > 0
        ORDER BY hit_count DESC, last_hit_at DESC NULLS LAST LIMIT %s
    """, (limit,))
    columns = [c.name for c in cur.description or []]
    return [dict(zip(columns, row, strict=True)) for row in await cur.fetchall()]


def default_corpus_version() -> str:
    """The corpus as served now: masked answers never share a key with unmasked ones."""
    from app import sealed

    return sealed.served_label(CORPUS_VERSION)


def today() -> Date:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date()
