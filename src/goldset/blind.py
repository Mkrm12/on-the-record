"""Blind grading: withholding judge output, anonymising for a second grader, and comparing graders."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from goldset.judge import RELEVANT_AT, binary_agreement, cohen_kappa

# Any key whose name suggests a grade.
FORBIDDEN_KEYS = ("graded_relevance", "grade", "grades", "judge", "judged", "pooled_by", "deepseek", "baseline")


class BlindnessError(ValueError):
    """Something grade-shaped reached a worksheet that must not carry one."""


def assert_blind(payload: object, path: str = "worksheet") -> None:
    """Raise if anything in the payload is named like a grader's output. Belt and braces over the allow-list."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if any(bad in str(key).lower() for bad in FORBIDDEN_KEYS):
                raise BlindnessError(f"{path}.{key}: the worksheet must not carry judge output")
            assert_blind(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, value in enumerate(payload):
            assert_blind(value, f"{path}[{i}]")


def anonymise(worksheet: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, str]]:
    """Opaque ids and a shuffled order. Returns the worksheet and the key, which travel separately."""
    rng = random.Random(seed)
    pairs = list((worksheet.get("calibration_pairs") or {}).items())
    items = list((worksheet.get("items") or {}).items())
    rng.shuffle(pairs)
    rng.shuffle(items)

    key: dict[str, str] = {}
    out_pairs: dict[str, dict] = {}
    for n, (real, pair) in enumerate(pairs, start=1):
        opaque = f"p{n:04d}"
        key[opaque] = real
        out_pairs[opaque] = {k: v for k, v in pair.items() if k not in ("item_id", "speech_id")}

    out_items: dict[str, dict] = {}
    for n, (real, item) in enumerate(items, start=1):
        opaque = f"q{n:04d}"
        key[opaque] = real
        out_items[opaque] = {k: v for k, v in item.items() if k != "id"}

    anonymous = {
        "anonymised": True,
        "seed": seed,
        "counts": {"items": len(out_items), "calibration_pairs": len(out_pairs)},
        "calibration_pairs": out_pairs,
        "items": out_items,
    }
    assert_blind(anonymous)
    return anonymous, key


def load_grader_file(path: Path, key_path: Path) -> tuple[dict[str, int], dict[str, str], list[str]]:
    """De-anonymise a grader's file through the key it never saw."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    grades: dict[str, int] = {}
    why: dict[str, str] = {}
    rejected: list[str] = []
    for opaque, entry in (raw.get("grades") or {}).items():
        real = key.get(opaque)
        if real is None:
            rejected.append(f"{opaque}: not in the key")
            continue
        grade = entry.get("grade") if isinstance(entry, dict) else entry
        if not isinstance(grade, int) or isinstance(grade, bool) or not 0 <= grade <= 3:
            rejected.append(f"{opaque}: grade {grade!r} is not an integer 0-3")
            continue
        grades[real] = grade
        if isinstance(entry, dict):
            why[real] = str(entry.get("why", ""))
    return grades, why, rejected


def crosses_boundary(a: int, b: int, threshold: int = RELEVANT_AT) -> bool:
    """Whether two grades disagree about relevance, which is the only disagreement a metric can see."""
    return (a >= threshold) != (b >= threshold)


def agreement(a: dict[str, int], b: dict[str, int], label_a: str, label_b: str,
              threshold: int = RELEVANT_AT) -> dict[str, Any]:  # fmt: skip
    """Four-level and binarised agreement between two graders over the pairs both graded."""
    shared = sorted(set(a) & set(b))
    pa = {(k, ""): a[k] for k in shared}
    pb = {(k, ""): b[k] for k in shared}
    return {
        "pair": f"{label_a} vs {label_b}",
        "n": len(shared),
        "four_level": cohen_kappa(pa, pb),
        "binarised": binary_agreement(pa, pb, threshold),
        "crossing_disagreements": sum(1 for k in shared if crosses_boundary(a[k], b[k], threshold)),
        "non_crossing_disagreements": sum(
            1 for k in shared if a[k] != b[k] and not crosses_boundary(a[k], b[k], threshold)
        ),
    }
