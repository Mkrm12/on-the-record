"""The evaluation harness behind the Phase 2 gate: run the gold questions through named configurations of the search
engine, score them at speech level, aggregate per category, and compare configurations pairwise.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from retrieval.hybrid import SearchConfig
from retrieval.metrics import dedupe_speeches, found_at_k, ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class Variant:
    name: str
    config: SearchConfig
    description: str


def standard_variants(base: SearchConfig | None = None) -> list[Variant]:
    """The ladder: each rung changes one thing, so the effect of every component is visible on its own."""
    b = base or SearchConfig()
    return [
        Variant("dense", replace(b, sparse=False, speaker=False, rerank="off"), "Baseline: dense search only (no keyword leg, no reranker)"),
        Variant("hybrid-naive", replace(b, speaker=False, sparse_coverage=0.0, rerank="off"), "Hybrid as first specified: dense + keyword, equal-weight RRF, no coverage threshold"),
        Variant("hybrid", replace(b, speaker=False, rerank="off"), "Variant A: dense + keyword with the IDF-coverage threshold, fused with RRF"),
        Variant("hybrid+speaker", replace(b, rerank="off"), "Variant A plus the speaker leg (I-20)"),
        Variant("hybrid+rerank", replace(b, speaker=False), "Variant B: Variant A plus the reranker (blended with the fused order)"),
        Variant("full", b, "Full pipeline: hybrid + speaker + reranker (blend): the default configuration"),
        Variant("full-replace", replace(b, rerank="replace"), "Full pipeline with the reranker replacing the fused order"),
    ]


@dataclass
class QueryOutcome:
    qid: str
    category: str
    relevant: frozenset[str]
    ranked: list[str]  # speech ids, best first, each once
    abstain: bool
    top_rerank: float | None  # the highest reranker score among the hits
    timings: dict[str, float]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Scores:
    found: int
    n: int
    found5: int
    recall: float | None
    rr: float
    ndcg: float | None


def speech_ranking(hits: Sequence[Any]) -> list[str]:
    return dedupe_speeches(s for h in hits for s in (h.speech_id, *h.also_in_speeches))


def outcome_from_result(query: Any, result: Any) -> QueryOutcome:
    scored = [h.rerank_score for h in result.hits if h.rerank_score is not None]
    return QueryOutcome(
        qid=query.id,
        category=query.category,
        relevant=frozenset(r.speech_id for r in query.relevant),
        ranked=speech_ranking(result.hits),
        abstain=bool(query.expected_abstain),
        top_rerank=max(scored) if scored else None,
        timings=dict(result.diagnostics.get("timings_ms", {})),
        warnings=list(result.warnings),
    )


def score_outcome(o: QueryOutcome, k: int = 10) -> Scores:
    return Scores(
        found=found_at_k(o.relevant, o.ranked, k),
        n=len(o.relevant),
        found5=found_at_k(o.relevant, o.ranked, 5),
        recall=recall_at_k(o.relevant, o.ranked, k),
        rr=reciprocal_rank(o.relevant, o.ranked, k),
        ndcg=ndcg_at_k(o.relevant, o.ranked, k),
    )


@dataclass(frozen=True)
class Summary:
    found: int
    n: int
    found5: int
    queries: int
    mrr: float
    ndcg: float

    @property
    def recall(self) -> float:
        return self.found / self.n if self.n else 0.0


@dataclass(frozen=True)
class Table:
    by_category: dict[str, Summary]
    overall: Summary


def _summarise(scores: list[Scores]) -> Summary:
    q = len(scores)
    return Summary(
        found=sum(s.found for s in scores), n=sum(s.n for s in scores), found5=sum(s.found5 for s in scores), queries=q,
        mrr=sum(s.rr for s in scores) / q if q else 0.0, ndcg=sum(s.ndcg or 0.0 for s in scores) / q if q else 0.0,
    )


def aggregate(outcomes: Sequence[QueryOutcome], k: int = 10) -> Table:
    """Per-category and overall summaries over the answerable queries (an abstain question has nothing to find)."""
    groups: dict[str, list[Scores]] = defaultdict(list)
    for o in outcomes:
        if not o.abstain:
            groups[o.category].append(score_outcome(o, k))
    everything = [s for scores in groups.values() for s in scores]
    return Table({c: _summarise(s) for c, s in groups.items()}, _summarise(everything))


def paired_differences(base: Sequence[QueryOutcome], other: Sequence[QueryOutcome], metric: str, k: int = 10) -> list[float]:
    """Per-query (other minus base) on "rr", "recall" or "ndcg", over answerable queries, matched by query id."""
    if {o.qid for o in base} != {o.qid for o in other}:
        raise ValueError("the two runs cover different queries")
    theirs = {o.qid: o for o in other}
    diffs = []
    for o in base:
        if o.abstain:
            continue
        a, b = score_outcome(o, k), score_outcome(theirs[o.qid], k)
        diffs.append((getattr(b, metric) or 0.0) - (getattr(a, metric) or 0.0))
    return diffs


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """The chance that a random positive scores above a random negative (ties count half); None without both groups.
    """
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))] if ordered else math.nan


async def run_variant(searcher: Any, queries: Sequence[Any], variant: Variant, top_k: int = 30) -> list[QueryOutcome]:
    """Search every query with the variant's configuration. One failing query does not stop the run: it is recorded as
    an empty ranking with the reason, so a partial outage shows up in the numbers instead of hiding them.
    """
    outcomes = []
    for q in queries:
        try:
            result = await searcher.search(q.question, top_k=top_k, config=variant.config)
            outcomes.append(outcome_from_result(q, result))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed: the warning names the failure
            outcomes.append(
                QueryOutcome(q.id, q.category, frozenset(r.speech_id for r in q.relevant), [], bool(q.expected_abstain), None, {}, [f"search failed: {exc}"])
            )
    return outcomes
