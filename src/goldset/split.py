"""The dev/test split, made once and recorded by hash."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any

from goldset.schema import GoldItem

HELDOUT_WINDOW = "W3"
DEFAULT_SEED = 20260924


def assign(items: list[GoldItem], seed: int = DEFAULT_SEED, test_share: float = 0.5) -> list[GoldItem]:
    """Set `split` on every item, in place, and return them."""
    strata: dict[tuple[str, str, str], list[GoldItem]] = defaultdict(list)
    for item in items:
        if item.window == HELDOUT_WINDOW:
            item.split = "heldout"
        elif item.subcategory == "carried_from_v1":
            item.split = "dev"
        else:
            strata[(item.window or "none", item.house or "none", item.category)].append(item)

    rng = random.Random(seed)
    for key in sorted(strata):
        bucket = strata[key]
        rng.shuffle(bucket)
        # Alternate rather than slice: a stratum of one lands on a side that depends on the key, not always on dev, and
        # a stratum of three splits 2/1 instead of 3/0.
        offset = rng.random() < test_share
        for i, item in enumerate(bucket):
            item.split = "test" if (i % 2 == 0) == offset else "dev"
    return items


def split_hash(items: list[GoldItem]) -> str:
    """Identifies the assignment itself, independently of the questions' wording."""
    payload = sorted((i.id, i.split) for i in items)
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:16]


def report(items: list[GoldItem]) -> dict[str, Any]:
    """Enough to see at a glance whether the split is balanced where it needs to be."""
    counts = Counter(i.split for i in items)
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        per_category[item.category][item.split] += 1
    scorable = [i for i in items if i.split in ("dev", "test")]
    return {
        "hash": split_hash(items),
        "counts": dict(counts),
        "by_window": {
            window: dict(Counter(i.split for i in items if (i.window or "none") == window))
            for window in sorted({i.window or "none" for i in items})
        },
        "by_house": {
            house: dict(Counter(i.split for i in items if (i.house or "none") == house))
            for house in sorted({i.house or "none" for i in items})
        },
        "by_category": {k: dict(v) for k, v in sorted(per_category.items())},
        # The imbalance that would matter: a category with everything on one side cannot support a paired comparison on
        # that category at all.
        "single_sided_categories": sorted(
            category
            for category, splits in per_category.items()
            if {s for s, n in splits.items() if n and s in ("dev", "test")} in ({"dev"}, {"test"})
        ),
        "dev": sum(1 for i in scorable if i.split == "dev"),
        "test": sum(1 for i in scorable if i.split == "test"),
        "heldout": counts.get("heldout", 0),
    }
