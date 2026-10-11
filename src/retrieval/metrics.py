"""Retrieval metrics at speech level (the gold set labels speeches, not chunks). Pure, so they are unit tested against
hand-worked values. Every function takes the *ranked* speech ids best first (duplicates allowed: use dedupe_speeches
first when a speech can appear more than once) and the set of relevant speech ids.
"""

import math
import random
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass


def dedupe_speeches(ranked: Iterable[str]) -> list[str]:
    """Each speech at its best position: several sentences of one speech count as one result."""
    return list(dict.fromkeys(ranked))


def found_at_k(relevant: Collection[str], ranked: Sequence[str], k: int) -> int:
    return len(set(relevant) & set(ranked[:k]))


def recall_at_k(relevant: Collection[str], ranked: Sequence[str], k: int) -> float | None:
    """Share of the relevant speeches inside the top k; None when there are none (an abstain question)."""
    return found_at_k(relevant, ranked, k) / len(relevant) if relevant else None


def reciprocal_rank(relevant: Collection[str], ranked: Sequence[str], k: int) -> float:
    """1 / (position of the first relevant speech), 0 when none is within the top k."""
    wanted = set(relevant)
    for position, speech in enumerate(ranked[:k], start=1):
        if speech in wanted:
            return 1 / position
    return 0.0


def ndcg_at_k(relevant: Collection[str], ranked: Sequence[str], k: int) -> float | None:
    """Binary-gain nDCG: 1.0 when every relevant speech is at the top, lower the further down they sit."""
    wanted = set(relevant)
    if not wanted:
        return None
    seen: set[str] = set()
    dcg = 0.0
    for position, speech in enumerate(ranked[:k], start=1):
        if speech in wanted and speech not in seen:
            seen.add(speech)
            dcg += 1 / math.log2(position + 1)
    ideal = sum(1 / math.log2(p + 1) for p in range(1, min(len(wanted), k) + 1))
    return dcg / ideal


@dataclass(frozen=True)
class Bootstrap:
    mean: float
    low: float
    high: float
    wins: int
    losses: int
    ties: int


def bootstrap_paired(diffs: Sequence[float], *, iterations: int = 5000, seed: int = 0, confidence: float = 0.95) -> Bootstrap:
    """The mean of per-query differences (system B minus system A) with a percentile-bootstrap confidence interval.
    With a few dozen queries an interval that spans zero means the difference is not established, whatever the mean.
    """
    if not diffs:
        raise ValueError("no differences to bootstrap")
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(iterations))
    alpha = (1 - confidence) / 2
    low, high = means[int(alpha * iterations)], means[min(iterations - 1, int((1 - alpha) * iterations))]
    return Bootstrap(
        mean=sum(diffs) / n,
        low=low,
        high=high,
        wins=sum(d > 0 for d in diffs),
        losses=sum(d < 0 for d in diffs),
        ties=sum(d == 0 for d in diffs),
    )
