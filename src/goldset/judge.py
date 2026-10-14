"""Pooled relevance judging: build the pool from real retrieval, then grade it 0-3, UMBRELA-style."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from goldset.llm import Client, Estimate, estimate, parse_grade
from goldset.schema import GoldItem

POOL_DEPTH = 10

# Enough for a digit even when the model reasons its way to it first.
GRADE_MAX_TOKENS = 2048

# Where the pool and the raw grades are cached.
JUDGMENTS_PATH = Path(__file__).resolve().parents[2] / "data" / "review" / "judgments.json"
JUDGMENTS_VERSION = 1

# UMBRELA's four levels, written for this corpus.
GRADE_RUBRIC = """3 = Perfectly relevant: the passage is dedicated to the question and contains the answer.
2 = Highly relevant: the passage answers the question, but alongside other material or indirectly.
1 = Related: the passage is on the topic but does not answer the question.
0 = Irrelevant: the passage does not help answer the question at all."""

SYSTEM = (
    "You grade whether a passage from the UK Parliament's official record (Hansard) helps answer a "
    "question. You are strict: being about the same subject is not the same as answering. "
    "Reply with a single digit, 0, 1, 2 or 3, and nothing else."
)

PROMPT = """Question: {question}

Passage (spoken by {speaker}{party} in the {house}, {date}, debate: {heading}):
\"\"\"{passage}\"\"\"

{rubric}

Grade (0-3):"""


@dataclass(frozen=True)
class Pair:
    """One query-passage pair awaiting a grade."""

    item_id: str
    speech_id: str
    question: str
    passage: str
    speaker: str = ""
    party: str = ""
    house: str = ""
    date: str = ""
    heading: str = ""

    def prompt(self) -> str:
        return PROMPT.format(
            question=self.question,
            speaker=self.speaker or "an unnamed member",
            party=f", {self.party}" if self.party else "",
            house=self.house or "House",
            date=self.date or "an unrecorded date",
            heading=self.heading or "not recorded",
            passage=self.passage[:2000],
            rubric=GRADE_RUBRIC,
        )


@dataclass
class Pool:
    pairs: list[Pair] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    # item id -> variant -> the ranked speech ids that variant returned, so judged@10 is computable.
    retrieved: dict[str, dict[str, list[str]]] = field(default_factory=dict)


async def _passages(conn: AsyncConnection, speech_ids: list[str]) -> dict[str, dict]:
    """The text and metadata a judge needs to grade a speech. Truncated at the first ~1,600 characters: a judge reading
    eight pages to grade one sentence costs money and grades worse.
    """
    if not speech_ids:
        return {}
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT s.id, left(s.text, 1600) AS passage, s.speaker_name, s.speaker_party,
                   d.house, d.date, d.heading
              FROM speech s LEFT JOIN debate d ON d.id = s.debate_id
             WHERE s.id = ANY(%(ids)s)
            """,
            {"ids": speech_ids},
        )
        return {r["id"]: r for r in await cur.fetchall()}


async def build_pool(
    conn: AsyncConnection,
    searcher: Any,
    items: list[GoldItem],
    variants: list[Any],
    depth: int = POOL_DEPTH,
    top_k: int = 30,
) -> Pool:
    """Run every variant over every retrieval-testable item and collect the union of their top `depth`."""
    pool = Pool(variants=[v.name for v in variants])
    wanted: set[str] = set()
    per_item: dict[str, set[str]] = defaultdict(set)

    for item in items:
        if not item.retrieval_testable:
            continue
        pool.retrieved[item.id] = {}
        for variant in variants:
            try:
                result = await searcher.search(item.question, top_k=top_k, config=variant.config)
            except Exception as exc:  # noqa: BLE001 - a failed query is recorded, not fatal
                pool.retrieved[item.id][variant.name] = []
                pool.retrieved[item.id].setdefault("_errors", []).append(f"{variant.name}: {exc}")
                continue
            ranked: list[str] = []
            for hit in result.hits:
                speech_id = getattr(hit, "speech_id", None) or getattr(getattr(hit, "candidate", None), "speech_id", None)
                if speech_id and speech_id not in ranked:
                    ranked.append(speech_id)
            pool.retrieved[item.id][variant.name] = ranked
            per_item[item.id].update(ranked[:depth])
        per_item[item.id].update(item.relevant_speech_ids)
        wanted.update(per_item[item.id])

    passages = await _passages(conn, sorted(wanted))
    by_id = {i.id: i for i in items}
    for item_id, speech_ids in per_item.items():
        item = by_id[item_id]
        for speech_id in sorted(speech_ids):
            row = passages.get(speech_id)
            if not row:
                continue
            pool.pairs.append(
                Pair(
                    item_id=item_id,
                    speech_id=speech_id,
                    question=item.question,
                    passage=row["passage"] or "",
                    speaker=row["speaker_name"] or "",
                    party=row["speaker_party"] or "",
                    house=row["house"] or "",
                    date=str(row["date"] or ""),
                    heading=row["heading"] or "",
                )
            )
    return pool


# Measured on this judge, not assumed: with `reasoning_effort="none"` a grade is a single token; with reasoning on,
# deepseek-flash spent 37-55 and deepseek-v4-pro about 52 on these prompts.
OUTPUT_TOKENS_REASONING_OFF = 4
OUTPUT_TOKENS_REASONING_ON = 120


def expected_output_tokens(reasoning_effort: str) -> int:
    """What one grade really costs in output tokens under a given reasoning setting."""
    return OUTPUT_TOKENS_REASONING_OFF if reasoning_effort == "none" else OUTPUT_TOKENS_REASONING_ON


def estimate_judging(
    pool: Pool, model: str, reasoning_effort: str = "none", limit: int | None = None
) -> Estimate:
    """What grading this pool will cost. Called before anything is sent, every time."""
    pairs = pool.pairs[:limit] if limit is not None else pool.pairs
    return estimate(
        [p.prompt() for p in pairs], model, output_tokens_each=expected_output_tokens(reasoning_effort)
    )


SMOKE_PAIR = Pair(
    item_id="_smoke",
    speech_id="_smoke",
    question="What colour is the sky described as?",
    passage="The sky over the chamber appeared a deep blue that afternoon.",
    speaker="A Member",
    house="Commons",
    date="2026-01-01",
    heading="Smoke test",
)


async def smoke_test(client: Client, *, max_tokens: int = GRADE_MAX_TOKENS) -> tuple[bool, str]:
    """Send ONE real grading prompt and require a parsable grade back. Called before every batch."""
    try:
        reply = await client.complete(SMOKE_PAIR.prompt(), system=SYSTEM, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 - the reason is what the caller reports
        return False, f"the judge did not answer a test prompt: {exc}"
    if (grade := parse_grade(reply)) is None:
        return False, f"the judge's reply carried no grade: {reply[:80]!r}"
    return True, f"test prompt graded {grade}"


def persists(limit: int | None) -> bool:
    """Whether a judging run with this limit may write the durable artifacts."""
    return limit is None


@dataclass
class JudgeRun:
    """What a grading run produced AND what it lost. Both halves are needed to trust a number."""

    grades: dict[str, dict[str, int]] = field(default_factory=dict)
    offered: int = 0
    graded: int = 0
    no_reply: int = 0
    unparsable: int = 0

    @property
    def yield_rate(self) -> float:
        return round(self.graded / self.offered, 4) if self.offered else 0.0

    def line(self) -> str:
        return (
            f"{self.graded:,} of {self.offered:,} pairs graded ({self.yield_rate:.1%}); "
            f"{self.no_reply:,} with no reply, {self.unparsable:,} unparsable"
        )


async def judge_run(
    pool: Pool, client: Client, *, limit: int | None = None, max_tokens: int = GRADE_MAX_TOKENS
) -> JudgeRun:
    """Grade the pool and report the losses."""
    # `is not None`, not truthiness: a limit of 0 means grade nothing, not grade everything.
    pairs = pool.pairs[:limit] if limit is not None else pool.pairs
    replies = await client.batch([p.prompt() for p in pairs], system=SYSTEM, max_tokens=max_tokens)
    grades: dict[str, dict[str, int]] = defaultdict(dict)
    run = JudgeRun(offered=len(pairs))
    for pair, reply in zip(pairs, replies):
        if reply is None:
            run.no_reply += 1
            continue
        grade = parse_grade(reply)
        if grade is None:
            run.unparsable += 1
            continue
        grades[pair.item_id][pair.speech_id] = grade
        run.graded += 1
    run.grades = dict(grades)
    return run


async def judge(
    pool: Pool, client: Client, *, limit: int | None = None, max_tokens: int = GRADE_MAX_TOKENS
) -> dict[str, dict[str, int]]:
    """The grades alone, for callers that do not need the loss report."""
    return (await judge_run(pool, client, limit=limit, max_tokens=max_tokens)).grades


def apply_grades(items: list[GoldItem], grades: dict[str, dict[str, int]], variants: list[str]) -> None:
    """Write the pooled judgments onto the items, in place."""
    for item in items:
        graded = dict(grades.get(item.id, {}))
        for speech_id in item.relevant_speech_ids:
            graded[speech_id] = max(graded.get(speech_id, 0), 2)
        if graded:
            item.graded_relevance = graded
            item.pooled_by = tuple(variants)


def judged_at_k(pool: Pool, grades: dict[str, dict[str, int]], k: int = 10) -> dict[str, float]:
    """The share of each variant's top k that carries a grade. The honest caveat on every pooled number."""
    out: dict[str, float] = {}
    for variant in pool.variants:
        judged = total = 0
        for item_id, per_variant in pool.retrieved.items():
            for speech_id in (per_variant.get(variant) or [])[:k]:
                total += 1
                judged += speech_id in grades.get(item_id, {})
        out[variant] = round(judged / total, 4) if total else 0.0
    return out


def save_judgments(
    pool: Pool,
    grades: dict[str, dict[str, int]],
    *,
    model: str,
    path: Path = JUDGMENTS_PATH,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the pool and the RAW grades to disk."""
    payload = {
        "version": JUDGMENTS_VERSION,
        "model": model,
        "judged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "variants": list(pool.variants),
        "retrieved": pool.retrieved,
        "grades": {item: dict(per) for item, per in grades.items()},
        "pairs": [asdict(p) for p in pool.pairs],
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_judgments(path: Path = JUDGMENTS_PATH) -> tuple[Pool, dict[str, dict[str, int]], dict[str, Any]]:
    """Read back what `save_judgments` wrote: the pool, the raw grades, and the metadata."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != JUDGMENTS_VERSION:
        raise ValueError(f"{path.name} is version {data.get('version')!r}, not {JUDGMENTS_VERSION}: re-run --judge")
    fields = set(Pair.__dataclass_fields__)
    pool = Pool(
        pairs=[Pair(**{k: v for k, v in p.items() if k in fields}) for p in data.get("pairs", [])],
        variants=list(data.get("variants", [])),
        retrieved=data.get("retrieved", {}) or {},
    )
    grades = {
        item: {speech: int(g) for speech, g in per.items()}
        for item, per in (data.get("grades") or {}).items()
    }
    meta = {k: v for k, v in data.items() if k not in ("pairs", "grades", "retrieved")}
    return pool, grades, meta


def flatten(grades: dict[str, dict[str, int]]) -> dict[tuple[str, str], int]:
    """item -> speech -> grade, as the (item, speech) -> grade mapping `cohen_kappa` compares."""
    return {(item, speech): grade for item, per in grades.items() for speech, grade in per.items()}


def confusion(a: dict[tuple[str, str], int], b: dict[tuple[str, str], int]) -> dict[str, Any]:
    """The 4x4 grade-by-grade table over the pairs both graders scored, plus where they differ by how much."""
    shared = sorted(set(a) & set(b))
    table = {f"{i}": {f"{j}": 0 for j in range(4)} for i in range(4)}
    off_by = Counter()
    for key in shared:
        x, y = a[key], b[key]
        if 0 <= x <= 3 and 0 <= y <= 3:
            table[f"{x}"][f"{y}"] += 1
        off_by[abs(x - y)] += 1
    return {
        "n": len(shared),
        "table": table,
        "off_by": {str(k): v for k, v in sorted(off_by.items())},
        "exact": off_by.get(0, 0),
        "adjacent_or_better": off_by.get(0, 0) + off_by.get(1, 0),
        "severe": sum(v for k, v in off_by.items() if k >= 2),
    }


# The grade at which a passage counts as relevant.
RELEVANT_AT = 2

# Applied to BINARISED relevance, which is the only call recall, MRR and nDCG's gain actually make.
GATE = 0.6


def binary_agreement(
    a: dict[tuple[str, str], int], b: dict[tuple[str, str], int], threshold: int = RELEVANT_AT
) -> dict[str, Any]:
    """Agreement on the only call the ranking metrics make: relevant, or not, at `threshold`."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"kappa": None, "n": 0, "threshold": threshold, "note": "no pairs graded by both"}
    binary_a = {k: a[k] >= threshold for k in shared}
    binary_b = {k: b[k] >= threshold for k in shared}
    both = sum(1 for k in shared if binary_a[k] and binary_b[k])
    neither = sum(1 for k in shared if not binary_a[k] and not binary_b[k])
    only_a = sum(1 for k in shared if binary_a[k] and not binary_b[k])
    only_b = sum(1 for k in shared if not binary_a[k] and binary_b[k])
    n = len(shared)
    observed = (both + neither) / n
    pa, pb = (both + only_a) / n, (both + only_b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return {
        "kappa": round(kappa, 4),
        "n": n,
        "threshold": threshold,
        "agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "both_relevant": both,
        "neither_relevant": neither,
        "only_first_relevant": only_a,
        "only_second_relevant": only_b,
        "passes_gate": kappa >= 0.6,
    }


def gate_verdict(
    owner: dict[tuple[str, str], int],
    judge_grades: dict[tuple[str, str], int],
    *,
    threshold: int = RELEVANT_AT,
    gate: float = GATE,
) -> dict[str, Any]:
    """The whole calibration answer in one object: can this judge be used, and for what."""
    four = cohen_kappa(owner, judge_grades)
    binary = binary_agreement(owner, judge_grades, threshold)
    return {
        "gate": gate,
        "threshold": threshold,
        "four_level": four,
        "binarised": binary,
        "usable_for_ranking": bool(binary.get("kappa") is not None and binary["kappa"] >= gate),
        "usable_for_graded_ndcg": bool(four.get("kappa") is not None and four["kappa"] >= gate),
        "verdict": (
            "no pairs graded by both"
            if not binary.get("n")
            else "usable for recall, MRR and binary nDCG; NOT for graded nDCG"
            if binary["kappa"] >= gate > (four["kappa"] or 0)
            else "usable, including for graded nDCG"
            if binary["kappa"] >= gate
            else "not usable: it disagrees about which passages are answers at all"
        ),
    }


def cohen_kappa(a: dict[tuple[str, str], int], b: dict[tuple[str, str], int]) -> dict[str, Any]:
    """Cohen's kappa between two graders over the pairs they both graded."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"kappa": None, "n": 0, "agreement": None, "note": "no pairs graded by both"}
    observed = sum(1 for key in shared if a[key] == b[key]) / len(shared)
    levels = sorted({a[k] for k in shared} | {b[k] for k in shared})
    expected = sum(
        (sum(1 for k in shared if a[k] == level) / len(shared))
        * (sum(1 for k in shared if b[k] == level) / len(shared))
        for level in levels
    )
    kappa = 1.0 if expected == 1 else (observed - expected) / (1 - expected)
    return {
        "kappa": round(kappa, 4),
        "n": len(shared),
        "agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "passes_gate": kappa >= 0.6,
    }
