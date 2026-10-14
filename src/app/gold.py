"""Gold evaluation set: loading and validation."""

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

GOLD_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "gold_eval.json"
SCHEMA_VERSION = 1
MIN_QUERIES = 25
MIN_QUOTE_LENGTH = 20
MIN_PER_CATEGORY = 2
MAX_PARAPHRASE_OVERLAP = 0.3  # a "paraphrase" question may share at most this share of its words with the answer
STATUSES = ("provisional", "verified")

_ID = re.compile(r"^g\d{3}$")
_WORD = re.compile(r"[a-z]+")
_STOPWORDS = frozenset(
    "about after also been before being could does from have into made many more most much only other over said says "
    "should some such than that their them then there these they this those very were what when where which while will "
    "with would your".split()
)


@dataclass(frozen=True)
class Relevant:
    speech_id: str
    evidence_quote: str


@dataclass(frozen=True)
class GoldQuery:
    id: str
    question: str
    category: str
    status: str
    expected_abstain: bool
    relevant: tuple[Relevant, ...]
    notes: str = ""


def content_words(text: str) -> set[str]:
    """Lower-cased words of four or more letters, without stopwords."""
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def overlap(question: str, quote: str) -> float:
    """Share of the question's content words (lightly stemmed) that also appear in the quote."""
    q = {_stem(w) for w in content_words(question)}
    if not q:
        return 0.0
    return len(q & {_stem(w) for w in content_words(quote)}) / len(q)


def _validate_relevant(label: str, relevant: object) -> list[str]:
    if not isinstance(relevant, list):
        return [f"{label}: relevant must be a list"]
    errors, seen = [], set()
    for item in relevant:
        speech_id = item.get("speech_id") if isinstance(item, dict) else None
        quote = item.get("evidence_quote") if isinstance(item, dict) else None
        if not isinstance(speech_id, str) or not speech_id.strip():
            errors.append(f"{label}: every relevant entry needs a speech_id")
        elif speech_id in seen:
            errors.append(f"{label}: duplicate speech_id {speech_id}")
        else:
            seen.add(speech_id)
        if not isinstance(quote, str) or len(quote.strip()) < MIN_QUOTE_LENGTH:
            errors.append(f"{label}: evidence_quote must be at least {MIN_QUOTE_LENGTH} characters")
    return errors


def _validate_query(i: int, q: object, categories: dict, seen_ids: set, seen_questions: set) -> list[str]:
    if not isinstance(q, dict):
        return [f"query #{i}: must be an object"]
    qid = q.get("id")
    label = qid if isinstance(qid, str) and qid else f"query #{i}"
    errors = []
    if not isinstance(qid, str) or not _ID.match(qid):
        errors.append(f"{label}: id must look like g001")
    elif qid in seen_ids:
        errors.append(f"{label}: duplicate id {qid}")
    seen_ids.add(qid)

    question = q.get("question")
    if not isinstance(question, str) or len(question.strip()) < 10:
        errors.append(f"{label}: question is missing or too short")
    else:
        key = " ".join(question.lower().split())
        if key in seen_questions:
            errors.append(f"{label}: duplicate question")
        seen_questions.add(key)

    category = q.get("category")
    if category not in categories:
        errors.append(f"{label}: unknown category {category!r}")
    if q.get("status") not in STATUSES:
        errors.append(f"{label}: status must be one of {', '.join(STATUSES)}")

    abstain = q.get("expected_abstain")
    if not isinstance(abstain, bool):
        errors.append(f"{label}: expected_abstain must be true or false")
    elif abstain != (category == "abstain"):
        errors.append(f"{label}: abstain category and expected_abstain disagree")

    errors += _validate_relevant(label, q.get("relevant"))
    relevant = q.get("relevant") if isinstance(q.get("relevant"), list) else []
    if abstain is True and relevant:
        errors.append(f"{label}: an abstain query must have no relevant speeches")
    if abstain is False and not relevant:
        errors.append(f"{label}: an answerable query needs at least one relevant speech")

    if category == "paraphrase" and isinstance(question, str):
        for item in relevant:
            quote = item.get("evidence_quote") if isinstance(item, dict) else None
            if isinstance(quote, str) and (shared := overlap(question, quote)) > MAX_PARAPHRASE_OVERLAP:
                errors.append(
                    f"{label}: paraphrase question shares {shared:.0%} of its words with the evidence "
                    f"(limit {MAX_PARAPHRASE_OVERLAP:.0%}); reword it or change the category"
                )
    return errors


def validate(data: object) -> list[str]:
    """Every problem with a gold set, as readable strings. An empty list means the set is valid."""
    if not isinstance(data, dict):
        return ["the gold file must be a JSON object"]
    errors = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    categories = data.get("categories")
    if not isinstance(categories, dict) or not categories:
        errors.append("categories must be a non-empty object")
        categories = {}

    corpus = data.get("corpus") if isinstance(data.get("corpus"), dict) else {}
    try:
        if date.fromisoformat(corpus.get("first_sitting", "")) > date.fromisoformat(corpus.get("last_sitting", "")):
            errors.append("corpus: first_sitting is after last_sitting")
    except (ValueError, TypeError):
        errors.append("corpus: first_sitting and last_sitting must be ISO dates")

    queries = data.get("queries")
    if not isinstance(queries, list):
        return [*errors, "queries must be a list"]
    if len(queries) < MIN_QUERIES:
        errors.append(f"need at least {MIN_QUERIES} queries, found {len(queries)}")

    seen_ids: set = set()
    seen_questions: set = set()
    for i, q in enumerate(queries):
        errors += _validate_query(i, q, categories, seen_ids, seen_questions)

    counts = Counter(q.get("category") for q in queries if isinstance(q, dict))
    for name in categories:
        if counts[name] < MIN_PER_CATEGORY:
            errors.append(f"category {name!r} has {counts[name]} queries, needs at least {MIN_PER_CATEGORY}")
    return errors


def load(path: Path = GOLD_PATH, statuses: tuple[str, ...] = STATUSES) -> list[GoldQuery]:
    """The gold queries whose status is in `statuses`. Raises ValueError if the file is invalid."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if errors := validate(data):
        raise ValueError("invalid gold set: " + "; ".join(errors[:5]))
    return [
        GoldQuery(
            q["id"], q["question"], q["category"], q["status"], q["expected_abstain"],
            tuple(Relevant(r["speech_id"], r["evidence_quote"]) for r in q["relevant"]), q.get("notes", ""),
        )  # fmt: skip
        for q in data["queries"]
        if q["status"] in statuses
    ]
