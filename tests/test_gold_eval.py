"""The gold evaluation set (tests/data/gold_eval.json) and the rules that keep it honest."""

import copy
import json

import pytest

from app.gold import GOLD_PATH, MIN_QUERIES, GoldQuery, content_words, load, overlap, validate
from dbhelpers import require, scalar
from ingest.text import clean_text

CATEGORIES = {"exact_term": "x", "paraphrase": "x", "abstain": "x"}


def query(n, category="exact_term", **extra):
    q = {
        "id": f"g{n:03d}",
        "question": f"What did the Minister say about the widget scheme number {n}?",
        "category": category,
        "status": "provisional",
        "expected_abstain": category == "abstain",
        "relevant": [] if category == "abstain" else [{"speech_id": f"S{n}", "evidence_quote": "The widget scheme will be reviewed next spring."}],
        "notes": "",
    }
    q.update(extra)
    return q


def gold(queries=None, **extra):
    if queries is None:
        queries = [query(i, ("exact_term", "paraphrase", "abstain")[i % 3]) for i in range(MIN_QUERIES)]
        for q in queries:
            if q["category"] == "paraphrase":
                q["question"] = f"Which official promised a rethink of the gadget programme {q['id']}?"
    data = {
        "schema_version": 1,
        "description": "d",
        "corpus": {"first_sitting": "2026-09-01", "last_sitting": "2026-09-17"},
        "categories": dict(CATEGORIES),
        "queries": queries,
    }
    data.update(extra)
    return data


def errors_for(data):
    return validate(data)


def has_error(data, fragment):
    return any(fragment in e for e in errors_for(data))


# --------------------------------------------------------------------------- validate: the synthetic baseline is clean,
# and each rule fires on its own ---------------------------------------------------------------------------


def test_a_well_formed_set_has_no_errors():
    assert errors_for(gold()) == []


def test_wrong_schema_version():
    assert has_error(gold(schema_version=2), "schema_version")


def test_too_few_queries():
    data = gold()
    data["queries"] = data["queries"][: MIN_QUERIES - 1]
    assert has_error(data, "at least")


def test_more_than_the_minimum_is_fine():
    data = gold()
    data["queries"].append(query(99, "exact_term"))
    assert errors_for(data) == []


@pytest.mark.parametrize("bad_id", ["g1", "G001", "gold1", "", "g0001"])
def test_query_ids_must_be_g_and_three_digits(bad_id):
    data = gold()
    data["queries"][0]["id"] = bad_id
    assert has_error(data, "id")


def test_duplicate_ids():
    data = gold()
    data["queries"][1]["id"] = data["queries"][0]["id"]
    assert has_error(data, "duplicate id")


def test_duplicate_questions_ignoring_case_and_spacing():
    data = gold()
    data["queries"][1]["question"] = "  " + data["queries"][0]["question"].upper()
    assert has_error(data, "duplicate question")


def test_blank_or_tiny_question():
    data = gold()
    data["queries"][0]["question"] = "  hi "
    assert has_error(data, "question")


def test_unknown_category():
    data = gold()
    data["queries"][0]["category"] = "nonsense"
    assert has_error(data, "category")


def test_unknown_status():
    data = gold()
    data["queries"][0]["status"] = "approved"
    assert has_error(data, "status")


def test_abstain_flag_must_match_the_abstain_category():
    data = gold()
    abstain = next(q for q in data["queries"] if q["category"] == "abstain")
    abstain["expected_abstain"] = False
    assert has_error(data, "abstain")
    data = gold()
    other = next(q for q in data["queries"] if q["category"] != "abstain")
    other["expected_abstain"] = True
    assert has_error(data, "abstain")


def test_an_abstain_query_cannot_name_relevant_speeches():
    data = gold()
    abstain = next(q for q in data["queries"] if q["category"] == "abstain")
    abstain["relevant"] = [{"speech_id": "S1", "evidence_quote": "The widget scheme will be reviewed next spring."}]
    assert has_error(data, "abstain")


def test_an_answerable_query_needs_at_least_one_relevant_speech():
    data = gold()
    next(q for q in data["queries"] if q["category"] == "exact_term")["relevant"] = []
    assert has_error(data, "relevant")


def test_relevant_speech_ids_are_unique_within_a_query():
    data = gold()
    q = next(q for q in data["queries"] if q["category"] == "exact_term")
    q["relevant"] = [q["relevant"][0], dict(q["relevant"][0])]
    assert has_error(data, "duplicate speech_id")


def test_evidence_quote_must_be_substantial():
    data = gold()
    next(q for q in data["queries"] if q["category"] == "exact_term")["relevant"][0]["evidence_quote"] = "Too short."
    assert has_error(data, "evidence_quote")


def test_missing_relevant_fields():
    data = gold()
    q = next(q for q in data["queries"] if q["category"] == "exact_term")
    q["relevant"] = [{"speech_id": "S1"}]
    assert has_error(data, "evidence_quote")
    q["relevant"] = [{"evidence_quote": "The widget scheme will be reviewed next spring."}]
    assert has_error(data, "speech_id")


def test_every_declared_category_needs_at_least_two_queries():
    data = gold()
    data["categories"]["numeric_fact"] = "x"
    assert has_error(data, "numeric_fact")


def test_the_benchmark_size_is_25():
    assert MIN_QUERIES == 25


def test_a_category_with_a_single_query_is_rejected():
    data = gold()
    kept = [q for q in data["queries"] if q["category"] != "paraphrase"]
    kept.append(next(q for q in data["queries"] if q["category"] == "paraphrase"))
    data["queries"] = kept
    assert has_error(data, "'paraphrase' has 1 queries")


def test_a_paraphrase_query_may_not_reuse_the_source_wording():
    data = gold()
    q = next(q for q in data["queries"] if q["category"] == "paraphrase")
    q["question"] = "Will the widget scheme be reviewed next spring?"  # lifted from the evidence sentence
    assert has_error(data, "paraphrase")


def test_the_leak_check_only_applies_to_paraphrase_queries():
    data = gold()
    q = next(q for q in data["queries"] if q["category"] == "exact_term")
    q["question"] = "Will the widget scheme be reviewed next spring?"
    assert errors_for(data) == []


def test_bad_top_level_shapes_are_reported_not_raised():
    assert validate({}) != []
    assert validate({"schema_version": 1, "queries": "nope", "categories": {}, "corpus": {}}) != []


def test_corpus_window_must_be_ordered_dates():
    assert has_error(gold(corpus={"first_sitting": "2026-09-17", "last_sitting": "2026-09-01"}), "corpus")
    assert has_error(gold(corpus={"first_sitting": "soon", "last_sitting": "2026-09-01"}), "corpus")


def test_validate_does_not_mutate_its_input():
    data = gold()
    before = copy.deepcopy(data)
    validate(data)
    assert data == before


# --------------------------------------------------------------------------- overlap / content_words
# ---------------------------------------------------------------------------


def test_content_words_drop_stopwords_short_words_and_punctuation():
    assert content_words("What did the Minister say about NHS waiting-lists in 2026?") == {"minister", "waiting", "lists"}


def test_overlap_is_the_share_of_question_words_found_in_the_quote():
    assert overlap("Minister promised widget review", "The widget review was delayed") == pytest.approx(2 / 4)
    assert overlap("reviewed widgets", "the widget review") == 1.0  # light stemming: reviewed~review, widgets~widget
    assert overlap("", "anything at all") == 0.0
    assert overlap("a to of", "anything") == 0.0


# --------------------------------------------------------------------------- load: filters by status, refuses an
# invalid file ---------------------------------------------------------------------------


def write(tmp_path, data):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_returns_typed_queries(tmp_path):
    queries = load(write(tmp_path, gold()))
    assert len(queries) == MIN_QUERIES and isinstance(queries[0], GoldQuery)
    assert queries[0].relevant[0].speech_id == "S0"


def test_load_can_filter_to_verified_only(tmp_path):
    data = gold()
    data["queries"][0]["status"] = "verified"
    assert [q.id for q in load(write(tmp_path, data), statuses=("verified",))] == ["g000"]


def test_load_refuses_an_invalid_file(tmp_path):
    data = gold(schema_version=9)
    with pytest.raises(ValueError, match="schema_version"):
        load(write(tmp_path, data))


# --------------------------------------------------------------------------- The real gold set
# ---------------------------------------------------------------------------


def real():
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def test_the_real_gold_file_is_valid():
    assert validate(real()) == []


def test_the_real_gold_file_has_the_benchmark_size_and_mix():
    queries = real()["queries"]
    assert len(queries) >= MIN_QUERIES
    cats = {q["category"] for q in queries}
    assert {"exact_term", "paraphrase", "abstain"} <= cats


def test_every_real_relevant_speech_exists_and_contains_its_quote(db):
    require(db, "SELECT count(*) FROM speech", "no speeches loaded")
    for q in real()["queries"]:
        for rel in q["relevant"]:
            row = db.execute("SELECT text FROM speech WHERE id = %s", (rel["speech_id"],)).fetchone()
            assert row, f"{q['id']}: speech {rel['speech_id']} is not in the database"
            assert rel["evidence_quote"] in clean_text(row[0]), f"{q['id']}: quote not found in {rel['speech_id']}"


def test_every_real_relevant_speech_is_retrievable_and_inside_the_corpus_window(db):
    require(db, "SELECT count(*) FROM speech", "no speeches loaded")
    corpus = real()["corpus"]
    for q in real()["queries"]:
        for rel in q["relevant"]:
            sid = rel["speech_id"]
            embeddable = scalar(db, "SELECT count(*) FROM chunk WHERE speech_id = %s AND NOT is_boilerplate", sid)
            assert embeddable > 0, f"{q['id']}: {sid} has no embeddable chunk, so it can never be retrieved"
            day = str(scalar(db, "SELECT d.date FROM speech s JOIN debate d ON d.id = s.debate_id WHERE s.id = %s", sid))
            assert corpus["first_sitting"] <= day <= corpus["last_sitting"], f"{q['id']}: {sid} is dated {day}"
