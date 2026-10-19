"""The pre-registered adoption rule for E1's bake-off, as code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Stage A.
MIN_GAIN = 0.01

# Stage A.
CATEGORY_TOLERANCE = 0.03
CATEGORY_MIN_ITEMS = 10

# Stage A.
WINDOWS_REQUIRED = 3
DEV_WINDOWS = ("W0", "W1", "W2", "W4")

# Stage A tier 1.
NOOP_EPSILON = 0.005

# Latency budget, milliseconds at p95, end to end on mains power.
LATENCY_P95_MS = 1200
LATENCY_HEADROOM_MS = 250

# E1's spend cap in US dollars, and the measured price of one judgment.
SPEND_CAP_USD = 4.00
COST_PER_JUDGMENT = 0.49 / 3959

# The metrics Stage C corrects across as one family.
CONFIRMATORY = ("recall@10", "mrr@10", "normalised_recall@10:multi_speech", "debate_recall@3:who_spoke")


# --- Stage A, tier 1: the free screen -------------------------------------------------------------------


@dataclass(frozen=True)
class NoOp:
    """Why an arm was dropped before any judging was paid for."""

    arm: str
    moved: dict[str, float]

    @property
    def reason(self) -> str:
        biggest = max(self.moved.values(), key=abs, default=0.0) if self.moved else 0.0
        return (f"every condensed metric moved less than {NOOP_EPSILON} (largest {biggest:+.4f}), so it "
                "cannot be moving a raw one either: the condensed list is a subsequence of the raw list")


def tier1_survivors(
    condensed: Mapping[str, Mapping[str, float | None]], baseline: Mapping[str, float | None]
) -> tuple[list[str], list[NoOp]]:
    """Which arms are worth paying to judge (Stage A tier 1)."""
    survivors: list[str] = []
    dropped: list[NoOp] = []
    for arm, metrics in condensed.items():
        moved = {
            name: value - base
            for name, value in metrics.items()
            if value is not None and (base := baseline.get(name)) is not None
        }
        if moved and all(abs(delta) < NOOP_EPSILON for delta in moved.values()):
            dropped.append(NoOp(arm, moved))
        else:
            survivors.append(arm)
    return survivors, dropped


# --- the budget ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What a judging run would cost, and whether it may go ahead."""

    pairs: int
    cost_usd: float
    cap_usd: float
    balance_usd: float | None

    @property
    def within_cap(self) -> bool:
        return self.cost_usd <= self.cap_usd

    @property
    def affordable(self) -> bool:
        return self.balance_usd is None or self.cost_usd <= self.balance_usd

    @property
    def ok(self) -> bool:
        return self.within_cap and self.affordable

    def refusal(self) -> str:
        """Why this run must not start, or "" if it may."""
        if not self.within_cap:
            return (f"{self.pairs:,} pairs would cost ${self.cost_usd:.2f}, over this run's ${self.cap_usd:.2f} "
                    f"cap. Screen fewer arms, or raise --cap deliberately.")
        if not self.affordable:
            return (f"{self.pairs:,} pairs would cost ${self.cost_usd:.2f} and the balance is "
                    f"${self.balance_usd:.2f}. Top up before running this.")
        return ""


def price(pairs: int, cap_usd: float = SPEND_CAP_USD, balance_usd: float | None = None) -> Budget:
    """Price a judging run from the measured rate, never from a guess."""
    if pairs < 0:
        raise ValueError("pairs cannot be negative")
    return Budget(pairs, round(pairs * COST_PER_JUDGMENT, 4), cap_usd, balance_usd)


# --- Stage A, tier 2: the four conditions --------------------------------------------------------------


@dataclass
class ArmVerdict:
    """One arm's Stage A result: the numbers, each condition, and the pass or fail."""

    arm: str
    gains: dict[str, float] = field(default_factory=dict)
    per_window: dict[str, float] = field(default_factory=dict)
    category_losses: dict[str, float] = field(default_factory=dict)
    judged_at_10: float | None = None
    p95_ms: float | None = None
    failures: list[str] = field(default_factory=list)
    # Set when the arm could not be scored at all (judged@10 below the gate, or it returned nothing).
    unscorable: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures and not self.unscorable

    @property
    def best_gain(self) -> float:
        return max(self.gains.values(), default=0.0)

    def line(self) -> str:
        if self.unscorable:
            return f"{self.arm:6s} NOT SCORABLE  {self.unscorable}"
        gains = "  ".join(f"{k} {v:+.4f}" for k, v in sorted(self.gains.items()))
        verdict = "PASS" if self.passed else "fail"
        why = "" if self.passed else "\n         " + "\n         ".join(self.failures)
        return f"{self.arm:6s} {verdict}  {gains}" + why


def sign_agreement(per_window: Mapping[str, float], required: int = WINDOWS_REQUIRED) -> tuple[bool, int, str]:
    """Whether the gain points the same way in enough dev windows."""
    values = [v for w, v in per_window.items() if w in DEV_WINDOWS]
    if not values:
        return False, 0, "no per-window numbers, so the sign cannot be checked"
    positive = sum(1 for v in values if v > 0)
    negative = sum(1 for v in values if v < 0)
    agreeing = max(positive, negative)
    if agreeing >= required:
        return True, agreeing, ""
    return False, agreeing, (f"the sign agrees in only {agreeing} of {len(values)} dev windows "
                             f"({required} required): {', '.join(f'{w} {v:+.4f}' for w, v in sorted(per_window.items()))}")


def screen_arm(
    arm: str,
    gains: Mapping[str, float],
    per_window: Mapping[str, float],
    category_deltas: Mapping[str, tuple[float, int]],
    *,
    judged_at_10: float | None = None,
    p95_ms: float | None = None,
    baseline_p95_ms: float | None = None,
    judged_gate: float = 0.95,
) -> ArmVerdict:
    """Apply Stage A's four conditions to one arm. All four must hold."""
    verdict = ArmVerdict(arm=arm, gains=dict(gains), per_window=dict(per_window),
                         judged_at_10=judged_at_10, p95_ms=p95_ms)  # fmt: skip

    if judged_at_10 is not None and judged_at_10 < judged_gate:
        verdict.unscorable = (
            f"judged@10 is {judged_at_10:.3f}, below the {judged_gate:.2f} gate. Every result nobody has "
            "judged is scored as a miss, so this arm would be measured on a curriculum it never saw."
        )
        return verdict

    if not gains:
        verdict.unscorable = "no metric could be computed for this arm"
        return verdict

    # 1. a real gain somewhere
    if verdict.best_gain < MIN_GAIN:
        verdict.failures.append(f"best gain {verdict.best_gain:+.4f} is under the +{MIN_GAIN} floor")

    # 2. no category of consequence goes backwards
    for category, (delta, items) in sorted(category_deltas.items()):
        if items >= CATEGORY_MIN_ITEMS and delta < -CATEGORY_TOLERANCE:
            verdict.category_losses[category] = delta
    if verdict.category_losses:
        worst = min(verdict.category_losses.items(), key=lambda kv: kv[1])
        verdict.failures.append(
            f"{worst[0]} loses {worst[1]:+.4f} R@10 on {category_deltas[worst[0]][1]} items, past the "
            f"{CATEGORY_TOLERANCE} tolerance"
        )

    # 3. the sign holds across windows
    holds, _agreeing, why = sign_agreement(per_window)
    if not holds:
        verdict.failures.append(why)

    # 4. latency
    if p95_ms is not None:
        if p95_ms > LATENCY_P95_MS:
            verdict.failures.append(f"p95 {p95_ms:.0f} ms is over the {LATENCY_P95_MS} ms budget")
        elif baseline_p95_ms is not None and p95_ms - baseline_p95_ms > LATENCY_HEADROOM_MS:
            verdict.failures.append(
                f"p95 adds {p95_ms - baseline_p95_ms:.0f} ms, past the {LATENCY_HEADROOM_MS} ms an arm may "
                "take without an established gain (Stage C decides that, so it cannot pass the screen)"
            )
    return verdict


# --- Stage B: the bundle -------------------------------------------------------------------------------


def pick_within_groups(
    passed: Sequence[str], groups: Mapping[str, Sequence[str]], scores: Mapping[str, float]
) -> tuple[list[str], dict[str, list[str]]]:
    """One arm per mutually exclusive group, the best on its own target metric."""
    in_group = {arm: name for name, members in groups.items() for arm in members}
    chosen: dict[str, str] = {}
    set_aside: dict[str, list[str]] = {}
    for arm in passed:
        name = in_group.get(arm)
        if name is None:
            continue
        current = chosen.get(name)
        if current is None or scores.get(arm, 0.0) > scores.get(current, 0.0):
            if current is not None:
                set_aside.setdefault(name, []).append(current)
            chosen[name] = arm
        else:
            set_aside.setdefault(name, []).append(arm)
    bundle = [arm for arm in passed if in_group.get(arm) is None or chosen.get(in_group[arm]) == arm]
    return bundle, {k: sorted(v) for k, v in set_aside.items()}


def drop_one(
    bundle_metrics: Mapping[str, float | None],
    without: Mapping[str, Mapping[str, float | None]],
    epsilon: float = NOOP_EPSILON,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Which arms the bundle does not need (Stage B's drop-one check)."""
    droppable: list[str] = []
    moved_by_arm: dict[str, dict[str, float]] = {}
    for arm, metrics in without.items():
        moved = {
            name: value - base
            for name, value in metrics.items()
            if value is not None and (base := bundle_metrics.get(name)) is not None
        }
        moved_by_arm[arm] = moved
        if moved and all(abs(delta) < epsilon for delta in moved.values()):
            droppable.append(arm)
    return sorted(droppable), moved_by_arm


# --- Stage C: what to adopt ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Adoption:
    """The decision, and the sentence that justifies it."""

    adopt: list[str]
    reason: str
    established: list[str]
    worse: list[str]

    @property
    def freezes(self) -> bool:
        """Retrieval is frozen either way -- what changes is whether anything was adopted first."""
        return True


def decide(
    bundle: Sequence[str],
    established: Mapping[str, bool],
    means: Mapping[str, float],
    significantly_worse: Sequence[str] = (),
) -> Adoption:
    """Stage C's table, applied mechanically."""
    worse = sorted(significantly_worse)
    if worse:
        return Adoption([], f"{', '.join(worse)} came out significantly worse on test, so nothing is adopted "
                            "and retrieval freezes as it is.", [], worse)  # fmt: skip

    won = sorted(name for name, ok in established.items() if ok)
    if won:
        return Adoption(list(bundle),
                        f"established on test: {', '.join(won)}. The bundle is adopted and retrieval freezes.",
                        won, [])

    negative = sorted(name for name, mean in means.items() if mean < 0)
    if negative:
        return Adoption([], "nothing was established on test and "
                            f"{', '.join(negative)} point downwards, so nothing is adopted.", [], [])

    s1 = ["S1"] if "S1" in bundle else []
    if s1:
        return Adoption(s1, "nothing was established on test, but every point estimate is at or above zero, "
                            "so S1 alone is adopted: merging duplicate sentences of one speech cannot lower a "
                            "speech-level metric. Logged as a decision taken on weak evidence.", [], [])
    return Adoption([], "nothing was established on test and S1 was not in the bundle, so nothing is adopted.",
                    [], [])


def floor_p(p: float, iterations: int) -> str:
    """A p-value the bootstrap can actually support."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    floor = 1.0 / iterations
    return f"p < {floor:.4g}" if p < floor else f"p = {p:.4g}"
