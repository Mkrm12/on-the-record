"""Gold set v2: the schema, the scoring configuration, and validation that refuses a malformed set."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
GOLD_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "gold_v2.json"

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

    for nugget in item.get("nuggets") or []:
        if not isinstance(nugget, dict):
            bad("every nugget must be an object")
        elif not isinstance(nugget.get("text"), str) or not nugget["text"].strip():
            bad("every nugget needs text")
        elif not isinstance(nugget.get("vital"), bool):
            bad("every nugget needs vital true or false")

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
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    items = data.get("items")
    if not isinstance(items, list):
        return errors + ["items must be a list"]

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
        Nugget(n["text"], n["vital"], tuple(n.get("speech_ids", []))) for n in item.get("nuggets", [])
    )
    kwargs["forbidden_claims"] = tuple(item.get("forbidden_claims", []))
    kwargs["pooled_by"] = tuple(item.get("pooled_by", []))
    return GoldItem(**kwargs)


def to_dict(item: GoldItem) -> dict:
    out = asdict(item)
    out["evidence"] = [{"speech_id": e.speech_id, "quote": e.quote} for e in item.evidence]
    out["nuggets"] = [{"text": n.text, "vital": n.vital, "speech_ids": list(n.speech_ids)} for n in item.nuggets]
    out["forbidden_claims"] = list(item.forbidden_claims)
    out["pooled_by"] = list(item.pooled_by)
    return out


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
