"""Gold set v2: the schema, the scoring configuration, and validation that refuses a malformed set."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# v3 adds what a reference point IS (`kind`, `side`) and how many of a broad item's vital points a right answer needs
# (`breadth`, `min_points`).
SCHEMA_VERSION = 3
SCHEMA_VERSIONS = (2, 3)
GOLD_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "gold_v2.json"

# `narrow` asks for one fact, one named member, one event or "who"; `broad` asks what was said, both sides, whether
# there was any argument.
BREADTHS = ("narrow", "broad")
# `fact` a figure, date, result or (when asked for) a name; `point` an argument, whoever made it; `example` one member's
# instance of a point, never required unless the question names that member or asks who; `scope` what the record does
# not show, worded about the record and never about "the passage".
NUGGET_KINDS = ("fact", "point", "example", "scope")
_V3_ITEM_FIELDS = ("breadth", "min_points")
_V3_NUGGET_FIELDS = ("kind", "side")

TO_VERIFY = "TO_VERIFY"

# --- the controlled vocabularies -------------------------------------------------------------------

SPLITS = ("dev", "test", "heldout")

CRITICALITY_POINTS = {"minor": 1, "major": 3, "critical": 5}

# The penalty is fixed, not proportional to the tier: a fabricated quote on a trivia question is exactly as
# disqualifying as one on a critical question.
HARMFUL_PENALTY = -8

OUTCOMES = (
    "correct",
    "correct_missing_okay_nuggets",
    "missing_vital_nugget",
    "correct_abstention",
    "over_abstention",
    "wrong_but_benign",
    "harmful",
)

# Which outcomes score what share of the tier's points.
OUTCOME_CREDIT = {
    "correct": 1.0,
    "correct_missing_okay_nuggets": 0.75,
    "missing_vital_nugget": 0.0,
    "correct_abstention": 1.0,
    "over_abstention": 0.0,
    "wrong_but_benign": 0.0,
}

EXPECTED_BEHAVIOURS = (
    "answer",
    "partial",
    "abstain_not_in_record",
    "abstain_out_of_window",
    "clarify_ambiguous",
    "correct_false_premise",
    "neutralise_loaded_framing",
)

# An item whose expected behaviour is one of these must NOT carry relevant speeches: the whole point is that the record
# does not support an answer.
NON_ANSWERING = frozenset({"abstain_not_in_record", "abstain_out_of_window"})

PROVENANCES = ("db_template", "passage_llm", "query_first", "adversarial", "owner_written")

STATUSES = ("provisional", "verified", "dropped")

# --- categories ------------------------------------------------------------------------------------- The value is the
# expected behaviour an item of that category must declare, or None where the category spans several.

CATEGORIES: dict[str, str | None] = {
    # --- retrieval-testable, human-meaningful ---
    "exact_term": "answer",
    "paraphrase": "answer",
    "numeric": "answer",
    "named_speaker": "answer",
    "office_at_date": None,
    "party_attribution": "answer",
    "multi_speech": "answer",
    "multi_hop": "answer",
    "temporal": "answer",
    "comparison": "answer",
    # `stance` and `colloquial` span behaviours.
    "stance": None,
    "scoping": "answer",
    "quote_verification": None,
    "colloquial": None,  # see `stance`
    "multi_part": None,
    "cross_window": "answer",
    "boilerplate_trap": None,
    # --- adversarial ---
    "misattribution": "correct_false_premise",
    "mention_vs_speaker": "correct_false_premise",
    "false_premise": "correct_false_premise",
    "numeric_perturbation": "correct_false_premise",
    "negation_flip": "correct_false_premise",
    "out_of_window": "abstain_out_of_window",
    "near_miss_abstention": "abstain_not_in_record",
    "ambiguous": "clarify_ambiguous",
    "loaded": "neutralise_loaded_framing",
    # --- structured fact, answered from SQL rather than from retrieved text ---
    "division_result": "answer",
    "division_vote": "answer",
    "division_tally": "answer",
    "party_at_date": "answer",
    "office_held_at_date": "answer",
    "bill_stage_date": "answer",
    "count": "answer",
    "who_spoke": "answer",
}

# Structured-fact categories are answered from the database, not from retrieved passages.
STRUCTURED_FACT = frozenset(
    {
        "division_result",
        "division_vote",
        "division_tally",
        "party_at_date",
        "office_held_at_date",
        "bill_stage_date",
        "count",
    }
)

_ID = re.compile(r"^[a-z]{1,4}\d{4}$")
_WORD = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    "a an and are as at be been being but by did do does for from had has have he her him his how i if in "
    "into is it its me my no not of on or our she should that the their them then there these they this "
    "those to was we were what when where which who whom why will with would you your".split()
)

MIN_QUESTION_LENGTH = 10
MIN_QUOTE_LENGTH = 20


@dataclass(frozen=True)
class Nugget:
    """One atomic fact an answer is expected to contain."""

    text: str
    vital: bool
    speech_ids: tuple[str, ...] = ()
    kind: str = ""  # v3: one of NUGGET_KINDS; "" on an item not yet migrated
    side: str = ""  # v3: the side of a "both sides" question this point belongs to; "" when sides do not apply


@dataclass(frozen=True)
class Evidence:
    speech_id: str
    quote: str


@dataclass
class GoldItem:
    id: str
    question: str
    category: str
    criticality: str
    expected_behaviour: str
    provenance: str
    status: str = "provisional"
    split: str = ""
    window: str = ""
    house: str = ""
    section: str = ""
    date: str = ""
    subcategory: str = ""
    evidence: tuple[Evidence, ...] = ()
    nuggets: tuple[Nugget, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    expected_structured_facts: dict[str, Any] = field(default_factory=dict)
    graded_relevance: dict[str, int] = field(default_factory=dict)
    pooled_by: tuple[str, ...] = ()
    reviewer: str = ""
    notes: str = ""
    source_item: str = ""
    breadth: str = ""  # v3: one of BREADTHS; "" on an item not yet migrated
    min_points: int | None = None  # v3: a broad item's k of n; None means every vital point is needed

    @property
    def relevant_speech_ids(self) -> tuple[str, ...]:
        """The speeches this item was BUILT from. Pooled judging adds others; these are never dropped."""
        return tuple(e.speech_id for e in self.evidence)

    @property
    def points(self) -> int:
        return CRITICALITY_POINTS[self.criticality]

    @property
    def is_structured_fact(self) -> bool:
        return self.category in STRUCTURED_FACT

    @property
    def retrieval_testable(self) -> bool:
        """Whether speech-level recall means anything for this item."""
        return not self.is_structured_fact and bool(self.evidence)


# --- text helpers used by the overlap filter and by validation ---------------------------------------


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens. Keeps digits: a numeric question's overlap is mostly its numbers."""
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _STOPWORDS}


def jaccard(a: str, b: str) -> float:
    """Token-Jaccard over content words. 0.0 when either side has no content words."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def longest_common_run(question: str, quote: str) -> int:
    """The longest run of consecutive tokens the question copies from the quote."""
    q, s = tokens(question), tokens(quote)
    if not q or not s:
        return 0
    # Standard O(n*m) longest-common-substring table, one row at a time.
    best = 0
    previous = [0] * (len(s) + 1)
    for i in range(1, len(q) + 1):
        current = [0] * (len(s) + 1)
        for j in range(1, len(s) + 1):
            if q[i - 1] == s[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def normalise_question(question: str) -> str:
    """The key duplicate detection compares. Punctuation and case are not meaningful differences."""
    return " ".join(sorted(content_tokens(question)))


# --- validation --------------------------------------------------------------------------------------


def _validate_item(  # noqa: C901 - one branch per rule; splitting it would hide the list
    item: dict, seen_ids: set[str], seen_questions: dict[str, str]
) -> list[str]:
    label = item.get("id") if isinstance(item.get("id"), str) else "<no id>"
    errors: list[str] = []

    def bad(message: str) -> None:
        errors.append(f"{label}: {message}")

    qid = item.get("id")
    if not isinstance(qid, str) or not _ID.match(qid):
        bad("id must be one to four lower-case letters and four digits, like db0001, adv0012 or old0025")
    elif qid in seen_ids:
        bad(f"duplicate id {qid}")
    else:
        seen_ids.add(qid)

    question = item.get("question")
    if not isinstance(question, str) or len(question.strip()) < MIN_QUESTION_LENGTH:
        bad("question is missing or too short")
    else:
        key = normalise_question(question)
        if key and key in seen_questions:
            bad(f"duplicate question (same content words as {seen_questions[key]})")
        elif key:
            seen_questions[key] = str(qid)

    category = item.get("category")
    if category not in CATEGORIES:
        bad(f"unknown category {category!r}")

    if item.get("criticality") not in CRITICALITY_POINTS:
        bad("criticality must be minor, major or critical")
    if item.get("provenance") not in PROVENANCES:
        bad(f"unknown provenance {item.get('provenance')!r}")
    status = item.get("status")
    if status not in STATUSES:
        bad(f"status must be one of {', '.join(STATUSES)}")
    if (split := item.get("split", "")) and split not in SPLITS:
        bad(f"unknown split {split!r}")

    behaviour = item.get("expected_behaviour")
    if behaviour not in EXPECTED_BEHAVIOURS:
        bad(f"unknown expected_behaviour {behaviour!r}")
    elif category in CATEGORIES and (required := CATEGORIES[category]) and behaviour != required:
        bad(f"category {category} must expect {required}, not {behaviour}")

    evidence = item.get("evidence")
    if not isinstance(evidence, list):
        bad("evidence must be a list")
        evidence = []
    else:
        seen_speeches: set[str] = set()
        for e in evidence:
            if not isinstance(e, dict):
                bad("every evidence entry must be an object")
                continue
            speech_id, quote = e.get("speech_id"), e.get("quote")
            if not isinstance(speech_id, str) or not speech_id.strip():
                bad("every evidence entry needs a speech_id")
            elif speech_id in seen_speeches:
                bad(f"duplicate evidence speech {speech_id}")
            else:
                seen_speeches.add(speech_id)
            if not isinstance(quote, str) or len(quote.strip()) < MIN_QUOTE_LENGTH:
                bad(f"evidence quote must be at least {MIN_QUOTE_LENGTH} characters")

    if behaviour in NON_ANSWERING and evidence:
        bad(f"{behaviour} must have no evidence: the point is that the record does not support an answer")
    if behaviour in ("answer", "partial") and not evidence and category not in STRUCTURED_FACT:
        bad("an answering item outside the structured-fact categories needs at least one evidence speech")
    if category in STRUCTURED_FACT and not item.get("expected_structured_facts"):
        bad("a structured-fact item needs expected_structured_facts")

    # An adversarial item that names no forbidden claim cannot fail a system that falls for it, which makes it
    # decoration rather than a test.
    if behaviour in ("correct_false_premise", "neutralise_loaded_framing") and not item.get("forbidden_claims"):
        bad(f"{behaviour} needs at least one forbidden_claim")

    nuggets = [n for n in (item.get("nuggets") or []) if isinstance(n, dict)]
    for nugget in item.get("nuggets") or []:
        if not isinstance(nugget, dict):
            bad("every nugget must be an object")
        elif not isinstance(nugget.get("text"), str) or not nugget["text"].strip():
            bad("every nugget needs text")
        elif not isinstance(nugget.get("vital"), bool):
            bad("every nugget needs vital true or false")
        elif "kind" in nugget and nugget["kind"] not in NUGGET_KINDS:
            bad(f"a reference point's kind must be one of {', '.join(NUGGET_KINDS)}, not {nugget['kind']!r}")
        elif "side" in nugget and (not isinstance(nugget["side"], str) or not nugget["side"].strip()):
            bad("a reference point's side, when given, must name the side")

    breadth = item.get("breadth")
    if "breadth" in item and breadth not in BREADTHS:
        bad(f"breadth must be one of {', '.join(BREADTHS)}, not {breadth!r}")
    vital_count = sum(1 for n in nuggets if n.get("vital") is True)
    if "min_points" in item and item["min_points"] is not None:
        k = item["min_points"]
        if breadth != "broad":
            bad("min_points belongs to a broad item only: a narrow item needs every vital point")
        elif not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= vital_count:
            bad(f"min_points must be a whole number from 1 to the item's {vital_count} vital points, not {k!r}")
    sides = {n["side"] for n in nuggets if n.get("vital") is True and n.get("side")}
    if any(n.get("side") for n in nuggets) and len(sides) < 2:
        bad("a reference point carries a side, but the vital points cover fewer than two sides: sides mean nothing "
            "unless a right answer can be held to one point from each")
    elif sides and isinstance(item.get("min_points"), int) and item["min_points"] < len(sides):
        bad(f"min_points {item['min_points']} is fewer than the {len(sides)} sides, but a right answer states one from "
            "each side")

    for speech_id, grade in (item.get("graded_relevance") or {}).items():
        if not isinstance(grade, int) or not 0 <= grade <= 3:
            bad(f"graded relevance for {speech_id} must be an integer 0 to 3")

    if status == "verified" and TO_VERIFY in json.dumps(item, default=str):
        bad("status is verified but the item still carries a TO_VERIFY placeholder")

    return errors


def validate(data: object) -> list[str]:
    """Every problem with a gold set, as a list. An empty list means it is usable."""
    if not isinstance(data, dict):
        return ["the gold set must be an object"]
    errors: list[str] = []
    version = data.get("schema_version")
    if version not in SCHEMA_VERSIONS:
        errors.append(f"schema_version must be one of {SCHEMA_VERSIONS}, not {version!r}")
    items = data.get("items")
    if not isinstance(items, list):
        return errors + ["items must be a list"]
    if version == 2:
        uses_v3 = [i.get("id") for i in items if isinstance(i, dict) and (
            any(f in i for f in _V3_ITEM_FIELDS)
            or any(isinstance(n, dict) and any(f in n for f in _V3_NUGGET_FIELDS) for n in i.get("nuggets") or []))]
        if uses_v3:
            errors.append(f"items {uses_v3[:5]} use v3 fields (breadth, min_points, kind, side), which need schema_version "
                          "3")

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            errors.append("every item must be an object")
            continue
        errors += _validate_item(item, seen_ids, seen_questions)

    live = [i for i in items if isinstance(i, dict) and i.get("status") != "dropped"]
    # A held-out item in the dev or test split would leak W3 into tuning, which is the one thing the whole window design
    # exists to prevent.
    for item in live:
        if item.get("window") == "W3" and item.get("split") not in ("", "heldout"):
            errors.append(f"{item.get('id')}: W3 items must be split 'heldout', not {item.get('split')!r}")
    return errors


def load(path: Path = GOLD_PATH) -> tuple[GoldItem, ...]:
    """Read and validate a gold set. Raises ValueError if it is unusable."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if errors := validate(data):
        raise ValueError(f"invalid gold set ({len(errors)} problems): " + "; ".join(errors[:5]))
    return tuple(from_dict(item) for item in data["items"] if item.get("status") != "dropped")


def from_dict(item: dict) -> GoldItem:
    known = {f for f in GoldItem.__dataclass_fields__}
    kwargs = {k: v for k, v in item.items() if k in known}
    kwargs["evidence"] = tuple(Evidence(e["speech_id"], e["quote"]) for e in item.get("evidence", []))
    kwargs["nuggets"] = tuple(
        Nugget(n["text"], n["vital"], tuple(n.get("speech_ids", [])), kind=n.get("kind", ""), side=n.get("side", ""))
        for n in item.get("nuggets", [])
    )
    kwargs["forbidden_claims"] = tuple(item.get("forbidden_claims", []))
    kwargs["pooled_by"] = tuple(item.get("pooled_by", []))
    return GoldItem(**kwargs)


def _nugget_dict(n: Nugget) -> dict:
    out: dict[str, Any] = {"text": n.text, "vital": n.vital, "speech_ids": list(n.speech_ids)}
    if n.kind:
        out["kind"] = n.kind
    if n.side:
        out["side"] = n.side
    return out


def to_dict(item: GoldItem) -> dict:
    """The item as stored. v3 fields are written only when set, so a v2 item serialises exactly as it always did."""
    out = asdict(item)
    out["evidence"] = [{"speech_id": e.speech_id, "quote": e.quote} for e in item.evidence]
    out["nuggets"] = [_nugget_dict(n) for n in item.nuggets]
    out["forbidden_claims"] = list(item.forbidden_claims)
    out["pooled_by"] = list(item.pooled_by)
    if not item.breadth:
        out.pop("breadth")
    if item.min_points is None:
        out.pop("min_points")
    return out


def vital_required(item: GoldItem) -> int:
    """How many vital points a right answer must state: a broad item's `min_points` (k of n), else every vital point.
    """
    vital = sum(1 for n in item.nuggets if n.vital)
    if item.breadth == "broad" and item.min_points is not None:
        return min(item.min_points, vital)
    return vital


def dump(items: list[GoldItem], path: Path, **meta: Any) -> dict:
    """Write a gold set, refusing to write one that does not validate."""
    data = {
        "schema_version": SCHEMA_VERSION,
        **meta,
        "items": [to_dict(i) for i in items],
    }
    if errors := validate(data):
        raise ValueError(f"refusing to write an invalid gold set ({len(errors)} problems): " + "; ".join(errors[:5]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def content_hash(items: list[GoldItem]) -> str:
    """A hash over the questions and their splits, so "gold set v2" names one exact instrument."""
    payload = sorted((i.id, i.question, i.split, i.category, i.expected_behaviour) for i in items)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def census(items: list[GoldItem]) -> dict[str, Any]:
    return {
        "items": len(items),
        "by_split": dict(Counter(i.split or "unassigned" for i in items)),
        "by_window": dict(Counter(i.window or "none" for i in items)),
        "by_category": dict(Counter(i.category for i in items)),
        "by_criticality": dict(Counter(i.criticality for i in items)),
        "by_provenance": dict(Counter(i.provenance for i in items)),
        "by_expected_behaviour": dict(Counter(i.expected_behaviour for i in items)),
        "by_house": dict(Counter(i.house or "none" for i in items)),
        "structured_fact": sum(1 for i in items if i.is_structured_fact),
        "retrieval_testable": sum(1 for i in items if i.retrieval_testable),
        "points_available": sum(i.points for i in items),
    }
