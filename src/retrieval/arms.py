"""The E1 bake-off arms: cheap, isolated retrieval interventions, each one config flag."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

# `collapse`'s span for "a speech occupies exactly one slot" (arm S1).
MERGE_WHOLE_SPEECH = 10**9

# How often a normalised sentence must occur in the corpus before it counts as boilerplate (arm B1).
STOCK_OCCURRENCES = 20


# --- the set-question gate ------------------------------------------------------------------------------ Pre-
# registered before any arm ran, as a rule over the QUERY TEXT.

SET_QUESTION_PATTERNS: tuple[str, ...] = (
    r"\bwho\s+(spoke|took part|contributed|participated|was involved)",
    r"\bwho\s+(?:all\s+)?(?:has|have|had)\s+spoken",
    r"\bwhich\s+(members|mps|m\.?p\.?s|peers|lords|ministers|parties)\b",
    r"\bwhat\s+did\s+(mps|members|peers|the house)\s+say",
    r"\b(list|name)\s+(the\s+)?(members|mps|speakers|peers)\b",
    r"\bdebate\s+on\b",
    r"\b(views|opinions|positions|arguments)\s+(on|about|of)\b",
    r"\bcontributions?\s+(to|on|about)\b",
    r"\beveryone\s+who\b",
    r"\ball\s+(the\s+)?(members|mps|speakers|peers)\b",
)

_SET_GATE = re.compile("|".join(SET_QUESTION_PATTERNS), re.IGNORECASE)


def looks_like_a_set_question(query: str) -> bool:
    """Whether the query asks for a set of passages rather than one fact."""
    return bool(query and _SET_GATE.search(query))


# --- B1: stock sentences --------------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")
_NOT_WORD = re.compile(r"[^\w\s]")


def normalise_sentence(text: str | None) -> str:
    """The form two copies of the same procedural sentence share: case-folded, punctuation dropped, spaces collapsed.
    Hansard prints "Question put and agreed to." and "Question put, and agreed to."
    """
    if not text:
        return ""
    return _WHITESPACE.sub(" ", _NOT_WORD.sub(" ", text.casefold())).strip()


def penalise_stock(candidates: Sequence[Any], stock: frozenset[str], multiplier: float) -> tuple[list[Any], int]:
    """Re-score and re-sort candidates whose sentence is stock boilerplate (arm B1)."""
    if not 0.0 <= multiplier <= 1.0:
        raise ValueError("multiplier must be between 0 and 1")
    penalised = 0
    scored = []
    for position, candidate in enumerate(candidates):
        is_stock = normalise_sentence(getattr(candidate, "body", "")) in stock
        penalised += is_stock
        scored.append((getattr(candidate, "rrf", 0.0) * (multiplier if is_stock else 1.0), position, candidate))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored], penalised


# --- D1: the debate leg, MaxP ---------------------------------------------------------------------------


def rank_debates(candidates: Sequence[Any]) -> list[tuple[str, float]]:
    """Debates best first, scored MaxP: each debate takes its single best fused chunk."""
    best: dict[str, float] = {}
    order: list[str] = []
    for candidate in candidates:
        debate = getattr(candidate, "debate_id", None)
        if debate is None:
            continue
        score = float(getattr(candidate, "rrf", 0.0))
        if debate not in best:
            best[debate], _ = score, order.append(debate)
        elif score > best[debate]:
            best[debate] = score
    return sorted(((d, best[d]) for d in order), key=lambda row: (-row[1], order.index(row[0])))


def boost_debates(candidates: Sequence[Any], *, top_debates: int, per_debate: int) -> tuple[list[Any], int]:
    """Lift the best `per_debate` speeches of each of the top `top_debates` debates to the front (arm D1)."""
    if top_debates <= 0:
        return list(candidates), 0
    if per_debate <= 0:
        raise ValueError("per_debate must be at least 1")

    leading = [debate for debate, _ in rank_debates(candidates)[:top_debates]]
    rank_of = {debate: i for i, debate in enumerate(leading)}
    chosen: dict[str, int] = {}
    promoted: list[tuple[int, int, int, Any]] = []
    rest: list[tuple[int, Any]] = []

    for position, candidate in enumerate(candidates):
        debate = getattr(candidate, "debate_id", None)
        speech = getattr(candidate, "speech_id", None)
        if debate in rank_of and speech is not None and chosen.get(debate, 0) < per_debate:
            # One slot per SPEECH: a second sentence of a speech already promoted buys no coverage.
            if not any(getattr(c, "speech_id", None) == speech for *_, c in promoted):
                chosen[debate] = chosen.get(debate, 0) + 1
                promoted.append((rank_of[debate], chosen[debate], position, candidate))
                continue
        rest.append((position, candidate))

    promoted.sort(key=lambda row: (row[1], row[0], row[2]))  # round-robin: best of each debate, then seconds
    return [row[3] for row in promoted] + [c for _, c in rest], len(promoted)


# --- the arm registry ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One intervention: a name, the config change it makes, and what it is for."""

    key: str
    title: str
    why: str
    changes: dict[str, Any]
    # Arms in the same group are mutually exclusive; Stage B keeps the best one on its target metric.
    group: str = ""
    # The metric this arm is expected to move, used by Stage A's "target set metric" condition.
    target: str = "recall@10"
    # Non-empty when the arm cannot run without something built first.
    requires: str = ""

    def apply(self, config: Any) -> Any:
        """`config` with this arm's flags set. Frozen dataclass, so a new object."""
        return replace(config, **self.changes)


ARMS: tuple[Arm, ...] = (
    Arm("S1", "one slot per speech",
        "relevance is judged per speech, so a second sentence of the same speech wastes a slot; 5.67% of "
        "live top-10 slots were repeats",
        {"collapse_span": MERGE_WHOLE_SPEECH}),
    Arm("P30", "rerank pool 30", "0.1065 of first-stage relevance sits below the pool=20 cut on dev",
        {"pool": 30}, group="pool"),
    Arm("P50", "rerank pool 50",
        "as P30 but deeper: the dev ceiling at depth 100 is 0.7444 against 0.6379 delivered, so the "
        "relevance below the cut is worth more than one extra decile of pool",
        {"pool": 50}, group="pool"),
    Arm("P50g", "pool 50 for set questions only",
        "set questions need depth; single-fact questions need precision, and a deep pool costs both latency "
        "and precision on them",
        {"pool": 20, "set_pool": 50}, group="pool"),
    Arm("D1", "debate leg, MaxP",
        "for a set question the pool can spend every slot on one speech; this represents each leading debate "
        "by several distinct speeches",
        {"debate_boost": 5, "debate_speeches": 3}, group="debate", target="debate_recall@3"),
    Arm("B1", "stock-sentence penalty",
        "procedural boilerplate eats slots: 'Question put and agreed to' is not an answer to anything",
        {"stock_penalty": 0.5}, requires="the stock-sentence table (pipeline step `stock_sentences`)"),
    Arm("R1", "the reranker reads the parent window",
        "the most accurate component currently sees one sentence; the window was worse under the old "
        "embedding recipe and is worth retesting under the header recipe",
        {"rerank_text_mode": "window"}),
)

BY_KEY: dict[str, Arm] = {arm.key: arm for arm in ARMS}

# Kept in code so that `uplift.py --list` cannot claim a coverage the project does not have.
NOT_IMPLEMENTED: dict[str, str] = {
    "D2": "a debate-centroid table and an HNSW index over it (migration + build step)",
    "C1": "a speech-centroid table (~118.5k vectors) and an HNSW index over it (migration + build step)",
    "Q1": "a second dense leg in the fusion SQL, for the query rendered through the recipe renderer",
    "N1": "pg_trgm similarity in the speaker leg, with a non-word-token guard",
    "A1": "a curated alias table with a source for every line, built from Hansard's heading vocabulary",
    "O1": "an audit of what the production office-at-date resolver already does, before adding anything",
    "R2": "gte-reranker-modernbert-base installed in the gpu/ environment, with VRAM and p95 measured",
}


class UnknownArm(ValueError):
    """An arm key that does not exist, or exists only on paper."""


def resolve(keys: Sequence[str] | None) -> list[Arm]:
    """The arms named, in the order given. None or empty means every implemented arm."""
    if not keys:
        return list(ARMS)
    out = []
    for key in keys:
        if key in BY_KEY:
            out.append(BY_KEY[key])
            continue
        if key in NOT_IMPLEMENTED:
            raise UnknownArm(f"arm {key!r} is in the plan but not implemented: it needs "
                             f"{NOT_IMPLEMENTED[key]}")  # fmt: skip
        raise UnknownArm(f"unknown arm {key!r}; implemented: {', '.join(sorted(BY_KEY))}")
    return out


def bundle(arms: Iterable[Arm], config: Any) -> Any:
    """Every arm's flags applied to one config (Stage B)."""
    seen: dict[str, str] = {}
    out = config
    for arm in arms:
        if arm.group:
            if arm.group in seen:
                raise UnknownArm(f"arms {seen[arm.group]!r} and {arm.key!r} are mutually exclusive "
                                 f"(group {arm.group!r}); Stage B keeps one")  # fmt: skip
            seen[arm.group] = arm.key
        out = arm.apply(out)
    return out


def groups(arms: Iterable[Arm]) -> dict[str, list[str]]:
    """group -> the arm keys in it, for the Stage B exclusivity report."""
    out: dict[str, list[str]] = {}
    for arm in arms:
        if arm.group:
            out.setdefault(arm.group, []).append(arm.key)
    return out
