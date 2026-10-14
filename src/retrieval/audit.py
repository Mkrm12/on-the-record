"""Measurement repairs: the diagnostics that say whether a retrieval number can be believed."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from retrieval.metrics import recall_at_k, reciprocal_rank

# An arm is only scored when this share of its top-k is judged (E1 section 4.1 item 7).
JUDGED_GATE = 0.95


# --- is the metric's own ceiling below 1.0? -------------------------------------------------------------


def attainable_recall(relevant_count: int, k: int) -> float:
    """The best recall@k a perfect ranking could score for a query with this many relevant speeches."""
    if relevant_count <= 0:
        raise ValueError("a query with no relevant speeches has no attainable recall")
    return min(k, relevant_count) / relevant_count


def normalised_recall(relevant: Collection[str], ranked: Sequence[str], k: int) -> float | None:
    """recall@k as a share of what recall@k could reach. None when there is nothing relevant."""
    got = recall_at_k(relevant, ranked, k)
    if got is None:
        return None
    return got / attainable_recall(len(set(relevant)), k)


def r_precision(relevant: Collection[str], ranked: Sequence[str]) -> float | None:
    """Precision at depth |R|: the one depth that is fair to every query whatever the size of its answer."""
    wanted = set(relevant)
    if not wanted:
        return None
    cut = ranked[: len(wanted)]
    return len(wanted & set(cut)) / len(wanted)


# --- is the depth real? ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ListLengths:
    """The length distribution of the final list, and whether a depth can honestly be claimed."""

    minimum: int
    median: int
    maximum: int
    mean: float
    queries: int

    def honest_depth(self, claimed: int) -> int:
        """The deepest depth the shortest list can support. Claiming more measures list length."""
        return min(claimed, self.minimum)

    def label(self, claimed: int) -> str:
        """What the column should be called. `R@final` when the list never reaches the claimed depth."""
        return f"recall@{claimed}" if self.minimum >= claimed else "R@final"

    def note(self, claimed: int) -> str:
        if self.minimum >= claimed:
            return ""
        return (f"the final list is {self.minimum}-{self.maximum} long (median {self.median}), so a metric "
                f"at depth {claimed} is really at depth {self.honest_depth(claimed)}; report it as R@final")


def final_list_lengths(rankings: Iterable[Sequence[str]]) -> ListLengths | None:
    """The length distribution of what a configuration actually returned."""
    lengths = sorted(len(r) for r in rankings)
    if not lengths:
        return None
    return ListLengths(lengths[0], int(median(lengths)), lengths[-1],
                       sum(lengths) / len(lengths), len(lengths))  # fmt: skip


# --- are we grading an answer we never marked? ----------------------------------------------------------


def judged_at_k(judged: Collection[str], ranked: Sequence[str], k: int) -> float | None:
    """The share of the top k that carries any judgment. None when the list is empty."""
    top = ranked[:k]
    if not top:
        return None
    known = set(judged)
    return sum(1 for speech in top if speech in known) / len(top)


def scorable_at_gate(coverage: Mapping[str, float | None], gate: float = JUDGED_GATE) -> tuple[bool, str]:
    """Whether an arm may be scored at all, and why not if it may not."""
    values = [v for v in coverage.values() if v is not None]
    if not values:
        return False, "no query returned anything, so there is nothing to score"
    mean = sum(values) / len(values)
    if mean >= gate:
        return True, ""
    short = sorted(k for k, v in coverage.items() if v is not None and v < gate)
    return False, (
        f"judged@k is {mean:.3f}, below the {gate:.2f} gate: {len(short)} of {len(values)} queries return "
        f"results nobody has judged, and an unjudged result is scored as a miss. Judge the gap before "
        f"comparing this arm (worst: {', '.join(short[:5])})"
    )


def condense(judged: Collection[str], ranked: Sequence[str]) -> list[str]:
    """The ranking with unjudged documents removed (Sakai's condensed list)."""
    known = set(judged)
    return [speech for speech in ranked if speech in known]


def condensed_pair(judged: Collection[str], relevant: Collection[str], ranked: Sequence[str],
                   k: int) -> tuple[float | None, float | None]:  # fmt: skip
    """(recall@k, MRR@k) over the condensed list."""
    condensed = condense(set(judged) | set(relevant), ranked)
    return recall_at_k(relevant, condensed, k), reciprocal_rank(relevant, condensed, k)


# --- how many slots are wasted? -------------------------------------------------------------------------


def wasted_slots(speech_ids: Sequence[str], k: int = 10) -> int:
    """Top-k positions holding a speech that already appeared higher up."""
    seen: set[str] = set()
    wasted = 0
    for speech in speech_ids[:k]:
        if speech in seen:
            wasted += 1
        seen.add(speech)
    return wasted


def wasted_share(lists: Iterable[Sequence[str]], k: int = 10) -> tuple[float, int, int]:
    """(share, wasted, total) of top-k slots that repeat a speech, across queries."""
    wasted = total = 0
    for speech_ids in lists:
        top = speech_ids[:k]
        total += len(top)
        wasted += wasted_slots(top, k)
    return (wasted / total if total else 0.0), wasted, total


# --- the whole picture for one arm ----------------------------------------------------------------------


@dataclass(frozen=True)
class ArmAudit:
    """Everything section 4.1 requires about one configuration on one split."""

    arm: str
    split: str
    queries: int
    lengths: ListLengths | None
    judged_at_10: float | None
    judged_at_20: float | None
    scorable: bool
    refusal: str
    recall_at_10: float | None
    condensed_recall_at_10: float | None
    mrr_at_10: float | None
    condensed_mrr_at_10: float | None
    normalised_recall_at_10: float | None
    r_precision: float | None
    attainable_recall_at_10: float | None
    mean_relevant: float

    def line(self) -> str:
        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.4f}"

        label = self.lengths.label(100) if self.lengths else "R@final"
        return (f"{self.arm:16s} {self.split:8s} n={self.queries:3d} "
                f"judged@10 {fmt(self.judged_at_10)} R@10 {fmt(self.recall_at_10)} "
                f"(condensed {fmt(self.condensed_recall_at_10)}) MRR {fmt(self.mrr_at_10)} "
                f"norm {fmt(self.normalised_recall_at_10)} R-prec {fmt(self.r_precision)} [{label}]"
                + ("" if self.scorable else f"\n    NOT SCORABLE: {self.refusal}"))


def audit_arm(arm: str, split: str, outcomes: Sequence[object],
              judged_by_query: Mapping[str, Collection[str]]) -> ArmAudit:
    """Audit one arm on one split."""
    coverage10: dict[str, float | None] = {}
    lengths, recalls, condensed_recalls, mrrs, condensed_mrrs = [], [], [], [], []
    normalised, precisions, attainables, relevant_counts, coverage20 = [], [], [], [], []

    for outcome in outcomes:
        qid = getattr(outcome, "qid")
        relevant = set(getattr(outcome, "relevant") or ())
        ranked = list(getattr(outcome, "ranked") or ())
        judged = set(judged_by_query.get(qid, ())) | relevant
        lengths.append(ranked)
        coverage10[qid] = judged_at_k(judged, ranked, 10)
        if (c20 := judged_at_k(judged, ranked, 20)) is not None:
            coverage20.append(c20)
        if not relevant:
            continue
        relevant_counts.append(len(relevant))
        attainables.append(attainable_recall(len(relevant), 10))
        if (v := recall_at_k(relevant, ranked, 10)) is not None:
            recalls.append(v)
        mrrs.append(reciprocal_rank(relevant, ranked, 10))
        if (v := normalised_recall(relevant, ranked, 10)) is not None:
            normalised.append(v)
        if (v := r_precision(relevant, ranked)) is not None:
            precisions.append(v)
        cr, cm = condensed_pair(judged, relevant, ranked, 10)
        if cr is not None:
            condensed_recalls.append(cr)
        if cm is not None:
            condensed_mrrs.append(cm)

    def mean(values: Sequence[float]) -> float | None:
        return sum(values) / len(values) if values else None

    ok, refusal = scorable_at_gate(coverage10)
    return ArmAudit(
        arm=arm, split=split, queries=len(relevant_counts),
        lengths=final_list_lengths(lengths),
        judged_at_10=mean([v for v in coverage10.values() if v is not None]),
        judged_at_20=mean(coverage20), scorable=ok, refusal=refusal,
        recall_at_10=mean(recalls), condensed_recall_at_10=mean(condensed_recalls),
        mrr_at_10=mean(mrrs), condensed_mrr_at_10=mean(condensed_mrrs),
        normalised_recall_at_10=mean(normalised), r_precision=mean(precisions),
        attainable_recall_at_10=mean(attainables),
        mean_relevant=(sum(relevant_counts) / len(relevant_counts)) if relevant_counts else 0.0,
    )  # fmt: skip
