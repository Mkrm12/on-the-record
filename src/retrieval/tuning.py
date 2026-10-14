"""Knob tuning against the judged pool, with the adoption rule enforced in code."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from retrieval.metrics import Bootstrap, bootstrap_paired, found_at_k, ndcg_at_k, recall_at_k, reciprocal_rank

# A passage counts as relevant at this judged grade or above.
RELEVANT_AT = 2

# The primary metric's depth.
PRIMARY_K = 100

# The dev windows.
DEV_WINDOWS = ("W0", "W1", "W2", "W4")
HELDOUT_WINDOW = "W3"

CONFIDENCE = 0.95


class HeldOutViolation(RuntimeError):
    """An attempt to score a split that has not been unlocked. Raised, never warned."""


def guard_split(split: str, *, confirm: bool = False, final_exam: bool = False) -> None:
    """Refuse to score the test split without `--confirm` or the held-out window without `--final-exam`."""
    if split == "test" and not confirm:
        raise HeldOutViolation(
            "the test split confirms the final configuration once: pass --confirm, and log that you used it"
        )
    if split == "heldout" and not final_exam:
        raise HeldOutViolation(
            "W3 is the final exam and is scored ONCE: pass --final-exam. A second look burns it as a "
            "held-out set, and re-tuning on it is not allowed"
        )


class MixedCorpus(RuntimeError):
    """The embedded population is not all one recipe, so cosine across it is not meaningful."""


RECIPE_SQL = ("SELECT coalesce(embedding_recipe, '(untagged)'), count(*) FROM chunk"
              " WHERE embedding IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")


async def corpus_recipes(db: Any) -> dict[str, int]:
    """How many embedded chunks carry each embedding recipe. One key is what a sane corpus looks like."""
    async with db.connection() as conn:
        rows = await (await conn.execute(RECIPE_SQL)).fetchall()
    return {str(recipe): int(count) for recipe, count in rows}


def require_coherent_corpus(recipes: dict[str, int]) -> str:
    """The single recipe the corpus is embedded under, or a refusal naming the mixture."""
    if not recipes:
        raise MixedCorpus("no chunk has an embedding: there is nothing to search")
    if len(recipes) > 1:
        shown = ", ".join(f"{name} ({count:,})" for name, count in recipes.items())
        raise MixedCorpus(
            f"the corpus is embedded under {len(recipes)} recipes and cosine across them is meaningless: "
            f"{shown}. Finish the re-embed (cd gpu; uv run python embedder.py) before scoring anything."
        )
    return next(iter(recipes))


# Knobs that change how many results come back.
TRUNCATION_KNOBS = frozenset({"pool", "fused_limit", "dense_limit", "sparse_limit", "speaker_limit"})


def comparable_depth(returned_lengths: dict[Any, int]) -> int:
    """The deepest metric depth at which variants with these result-list lengths can be compared fairly."""
    return min(returned_lengths.values()) if returned_lengths else 0


def truncation_confound(knob: str, k: int, returned_lengths: dict[Any, int]) -> str:
    """Why comparing this knob at this depth is invalid, or "" if it is fine."""
    if knob not in TRUNCATION_KNOBS or not returned_lengths:
        return ""
    depth = comparable_depth(returned_lengths)
    if k <= depth:
        return ""
    shortest = min(returned_lengths, key=lambda v: returned_lengths[v])
    return (
        f"{knob} controls how many results come back ({shortest} returns "
        f"{returned_lengths[shortest]}), so a metric at depth {k} rewards the longer list rather than the "
        f"better one. Compare it at depth {depth} or shallower."
    )


# --- relevance from the judged pool ------------------------------------------------------------------


def relevant_speeches(item: Any, threshold: int = RELEVANT_AT) -> frozenset[str]:
    """Every speech judged relevant for this item, from the pool, plus the ones it was built from."""
    graded = getattr(item, "graded_relevance", None) or {}
    pooled = {speech for speech, grade in graded.items() if grade >= threshold}
    return frozenset(pooled | set(getattr(item, "relevant_speech_ids", ()) or ()))


def scorable(items: Sequence[Any]) -> list[Any]:
    """The items a retrieval number can honestly be computed over: retrieval-testable, with relevance."""
    return [i for i in items if getattr(i, "retrieval_testable", False) and relevant_speeches(i)]


@dataclass(frozen=True)
class GoldQuery:
    """The shape `retrieval.evaluation.run_variant` expects, built from a v2 gold item."""

    id: str
    question: str
    category: str
    relevant: tuple[Any, ...]
    expected_abstain: bool
    window: str
    split: str

    @staticmethod
    def of(item: Any, threshold: int = RELEVANT_AT) -> GoldQuery:
        speeches = sorted(relevant_speeches(item, threshold))
        return GoldQuery(
            id=item.id,
            question=item.question,
            category=item.category,
            relevant=tuple(_Rel(s) for s in speeches),
            expected_abstain=item.expected_behaviour.startswith("abstain"),
            window=item.window or "",
            split=item.split or "",
        )


@dataclass(frozen=True)
class _Rel:
    speech_id: str


# --- metrics over outcomes ---------------------------------------------------------------------------


def metric_of(outcome: Any, metric: str, k: int) -> float | None:
    """One metric for one query, or None where it is undefined (no relevant speeches)."""
    relevant, ranked = outcome.relevant, outcome.ranked
    if not relevant:
        return None
    if metric == "recall":
        return recall_at_k(relevant, ranked, k)
    if metric == "mrr":
        return reciprocal_rank(relevant, ranked, k)
    if metric == "ndcg":
        return ndcg_at_k(relevant, ranked, k)
    if metric == "found":
        return float(found_at_k(relevant, ranked, k))
    raise ValueError(f"unknown metric {metric!r}")


def mean_metric(outcomes: Sequence[Any], metric: str, k: int) -> float | None:
    values = [v for o in outcomes if (v := metric_of(o, metric, k)) is not None]
    return sum(values) / len(values) if values else None


def paired(base: Sequence[Any], other: Sequence[Any], metric: str, k: int) -> list[float]:
    """Per-query differences (other minus base) over the queries both scored and both define."""
    by_id = {o.qid: o for o in other}
    out: list[float] = []
    for b in base:
        o = by_id.get(b.qid)
        if o is None:
            continue
        a, c = metric_of(b, metric, k), metric_of(o, metric, k)
        if a is not None and c is not None:
            out.append(c - a)
    return out


# --- the adoption rule -------------------------------------------------------------------------------


def bootstrap_p(diffs: Sequence[float], *, iterations: int = 5000, seed: int = 0) -> float:
    """A two-sided bootstrap p-value for "the mean difference is zero"."""
    if not diffs:
        return 1.0
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(rng.choices(diffs, k=n)) / n for _ in range(iterations)]
    share_at_or_below = sum(1 for m in means if m <= 0) / iterations
    share_at_or_above = sum(1 for m in means if m >= 0) / iterations
    return min(1.0, 2 * min(share_at_or_below, share_at_or_above))


@dataclass
class Comparison:
    name: str
    metric: str
    k: int
    diffs: list[float]
    interval: Bootstrap
    p: float
    adjusted_alpha: float = 0.05
    established: bool = False
    holds_across_windows: bool | None = None
    per_window: dict[str, float] = field(default_factory=dict)
    # Set when the comparison is invalid because the knob controls the result-list length at this depth.
    confound: str = ""

    @property
    def mean(self) -> float:
        return self.interval.mean

    def line(self) -> str:
        sign = "+" if self.mean >= 0 else ""
        verdict = "ESTABLISHED" if self.established else "not established"
        across = "" if self.holds_across_windows is None else (
            ", sign holds across windows" if self.holds_across_windows else ", **sign flips across windows**"
        )
        return (
            f"{self.name}: {sign}{self.mean:.4f} {self.metric}@{self.k} "
            f"[{self.interval.low:+.4f}, {self.interval.high:+.4f}] p={self.p:.4f} "
            f"(alpha {self.adjusted_alpha:.4f}) -- {verdict}{across}"
            + (f"\n      CONFOUNDED: {self.confound}" if self.confound else "")
        )


def holm_bonferroni(comparisons: Sequence[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Mark which comparisons in a FAMILY survive the correction. Mutates and returns them."""
    ordered = sorted(comparisons, key=lambda c: c.p)
    m = len(ordered)
    still_rejecting = True
    for i, comparison in enumerate(ordered):
        comparison.adjusted_alpha = alpha / (m - i)
        # Holm is a step-down procedure: once one test fails, every larger p-value fails too.
        if still_rejecting and comparison.p <= comparison.adjusted_alpha:
            comparison.established = True
        else:
            still_rejecting = False
            comparison.established = False
    return list(comparisons)


def compare(
    base: Sequence[Any],
    other: Sequence[Any],
    name: str,
    *,
    metric: str = "recall",
    k: int = PRIMARY_K,
    seed: int = 7,
    iterations: int = 5_000,
) -> Comparison:
    """One paired comparison, with its interval, its p-value, and its per-window means."""
    diffs = paired(base, other, metric, k)
    interval = (bootstrap_paired(diffs, seed=seed, iterations=iterations) if diffs
                else Bootstrap(0.0, 0.0, 0.0, 0, 0, 0))  # fmt: skip
    return Comparison(
        name=name, metric=metric, k=k, diffs=diffs, interval=interval,
        p=bootstrap_p(diffs, seed=seed, iterations=iterations),
    )


def check_across_windows(
    comparison: Comparison,
    base: Sequence[Any],
    other: Sequence[Any],
    window_of: dict[str, str],
    *,
    windows: Sequence[str] = DEV_WINDOWS,
) -> Comparison:
    """Whether the difference has the same sign in every dev window that has queries."""
    means: dict[str, float] = {}
    for window in windows:
        ids = {qid for qid, w in window_of.items() if w == window}
        if not ids:
            continue
        b = [o for o in base if o.qid in ids]
        o = [x for x in other if x.qid in ids]
        diffs = paired(b, o, comparison.metric, comparison.k)
        if diffs:
            means[window] = sum(diffs) / len(diffs)
    comparison.per_window = means
    if len(means) < 2:
        comparison.holds_across_windows = None
    else:
        signs = {m > 0 for m in means.values() if m != 0}
        comparison.holds_across_windows = len(signs) <= 1
    return comparison


def adopt(comparison: Comparison) -> bool:
    """The pre-registered decision, applied mechanically. No judgement, so nobody can lean on it."""
    if comparison.confound:
        return False
    return bool(comparison.established and comparison.mean > 0 and comparison.holds_across_windows is not False)


# --- leave-one-window-out knob choice -----------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    held: str
    tune_on: tuple[str, ...]


def lowo_folds(windows: Sequence[str] = DEV_WINDOWS) -> list[Fold]:
    """Tune on all but one dev window, score on the one left out, rotate."""
    return [Fold(held=w, tune_on=tuple(x for x in windows if x != w)) for w in windows]


def plateau(scores: dict[Any, float], *, tolerance: float = 0.005) -> Any:
    """The most conservative value within `tolerance` of the best score."""
    if not scores:
        raise ValueError("no scores to choose from")
    best = max(scores.values())
    good = [value for value, score in scores.items() if score >= best - tolerance]
    return min(good, key=_sort_key)


def _sort_key(value: Any) -> Any:
    """Order knob values sensibly whether they are numbers or tuples of numbers."""
    if isinstance(value, (int, float)):
        return (value,)
    if isinstance(value, tuple):
        return (sum(value), value)
    return (math.inf,)


@dataclass
class KnobResult:
    knob: str
    per_fold_choice: dict[str, Any]
    held_out_scores: dict[str, float]
    stable: bool
    chosen: Any
    note: str = ""

    def line(self) -> str:
        choices = ", ".join(f"{fold}->{value}" for fold, value in self.per_fold_choice.items())
        return (
            f"{self.knob}: chose {self.chosen} "
            f"({'stable' if self.stable else 'UNSTABLE across folds'}; per fold {choices})"
            + (f" -- {self.note}" if self.note else "")
        )


def tune_knob(
    knob: str,
    scores_by_window: dict[Any, dict[str, float]],
    *,
    tolerance: float = 0.005,
    windows: Sequence[str] = DEV_WINDOWS,
) -> KnobResult:
    """Leave-one-window-out over one knob."""
    folds = lowo_folds(windows)
    per_fold: dict[str, Any] = {}
    held_out: dict[str, float] = {}
    for fold in folds:
        tuning_scores: dict[Any, float] = {}
        for value, by_window in scores_by_window.items():
            seen = [by_window[w] for w in fold.tune_on if w in by_window]
            if seen:
                tuning_scores[value] = sum(seen) / len(seen)
        if not tuning_scores:
            continue
        choice = plateau(tuning_scores, tolerance=tolerance)
        per_fold[fold.held] = choice
        if fold.held in scores_by_window.get(choice, {}):
            held_out[fold.held] = scores_by_window[choice][fold.held]

    distinct = {repr(v) for v in per_fold.values()}
    stable = len(distinct) <= 1
    # Every fold agreeing is the only case where the folds themselves pick the value.
    if stable and per_fold:
        chosen = next(iter(per_fold.values()))
        note = ""
    else:
        overall = {
            value: sum(by_window.values()) / len(by_window)
            for value, by_window in scores_by_window.items()
            if by_window
        }
        chosen = plateau(overall, tolerance=tolerance)
        note = "folds disagreed, so the value is the plateau over all dev windows; treat it as weakly supported"
    return KnobResult(knob=knob, per_fold_choice=per_fold, held_out_scores=held_out,
                      stable=stable, chosen=chosen, note=note)  # fmt: skip
