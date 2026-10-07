"""Answer-level scoring: severity-weighted quality, the harmful-error rate with its upper bound, and the gate."""

from __future__ import annotations

import math
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from goldset.schema import CRITICALITY_POINTS, HARMFUL_PENALTY, OUTCOME_CREDIT, OUTCOMES

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "scoring.toml"

ABSTAINING = frozenset({"correct_abstention", "over_abstention"})


@dataclass(frozen=True)
class ScoringConfig:
    points: dict[str, int] = field(default_factory=lambda: dict(CRITICALITY_POINTS))
    credit: dict[str, float] = field(default_factory=lambda: dict(OUTCOME_CREDIT))
    harmful_penalty: int = HARMFUL_PENALTY

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> ScoringConfig:
        if not path.exists():
            return cls()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return cls(
            points={**CRITICALITY_POINTS, **data.get("points", {})},
            credit={**OUTCOME_CREDIT, **data.get("credit", {})},
            harmful_penalty=int(data.get("harmful_penalty", HARMFUL_PENALTY)),
        )


@dataclass(frozen=True)
class Graded:
    """One answered question: what it was worth, and how the answer came out."""

    item_id: str
    criticality: str
    outcome: str
    attempted: bool = True


def wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """The upper end of a 95% Wilson interval."""
    if trials <= 0:
        return 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return min(1.0, (centre + spread) / denominator)


def score(graded: list[Graded], config: ScoringConfig | None = None) -> dict:
    """The three numbers, the gate, and enough breakdown to see where a score came from."""
    config = config or ScoringConfig()
    unknown = sorted({g.outcome for g in graded} - set(OUTCOMES))
    if unknown:
        raise ValueError(f"unknown outcome(s): {', '.join(unknown)}")

    available = sum(config.points[g.criticality] for g in graded)
    earned = 0.0
    for g in graded:
        if g.outcome == "harmful":
            earned += config.harmful_penalty
        else:
            earned += config.points[g.criticality] * config.credit[g.outcome]

    harmful = sum(1 for g in graded if g.outcome == "harmful")
    attempted = [g for g in graded if g.outcome not in ABSTAINING]
    correct = sum(1 for g in graded if g.outcome in ("correct", "correct_missing_okay_nuggets", "correct_abstention"))
    correct_attempted = sum(1 for g in attempted if g.outcome in ("correct", "correct_missing_okay_nuggets"))

    return {
        "n": len(graded),
        "harmful": {
            "count": harmful,
            "rate": harmful / len(graded) if graded else 0.0,
            "upper95": wilson_upper(harmful, len(graded)),
            "rule_of_three": 3 / len(graded) if graded and harmful == 0 else None,
        },
        "quality": {
            "points_available": available,
            "points_earned": round(earned, 4),
            # Can go negative: enough harmful answers should not bottom out at zero and look merely bad.
            "weighted": round(earned / available, 4) if available else 0.0,
        },
        "coverage": {
            "attempted": len(attempted),
            "not_attempted": len(graded) - len(attempted),
            "correct": correct,
            "correct_given_attempted": round(correct_attempted / len(attempted), 4) if attempted else 0.0,
            "over_abstention": sum(1 for g in graded if g.outcome == "over_abstention"),
        },
        "by_outcome": {o: c for o, c in sorted(Counter(g.outcome for g in graded).items())},
        # The gate is separate from the score on purpose: a system can score 82% and still be unreleasable.
        "gate": {"passed": harmful == 0, "reason": "" if harmful == 0 else f"{harmful} harmful outcome(s)"},
    }
