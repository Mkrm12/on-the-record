"""These are tests on the shipped instrument, not on a fixture. A benchmark that penalises a correct answer, or that
asks for a thing which does not exist, measures itself rather than the system -- and the only way that stays fixed
is a test that reads `tests/data/gold_v2.json` and fails if the defect comes back.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

from goldset.dbgen import _STAGE_PHRASINGS_HOUSELESS, _stage_question
from goldset.schema import CATEGORIES, GOLD_PATH, content_hash, load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repair_gold import REPAIRS, check, houseless_question, repairs_for  # noqa: E402


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items() -> tuple:
    return load()


# An abstention that the record answers -------------------------------------------------------


def test_every_flagged_repair_is_in_place():
    """`scripts/repair_gold.py --check` is the audit; this makes it a test that has to keep passing."""
    assert check() == []


def test_a_later_note_after_a_repair_keeps_the_repair_in_place():
    import importlib.util
    from pathlib import Path as _P

    spec = importlib.util.spec_from_file_location("repair_gold_t", _P(__file__).resolve().parents[1] / "scripts" /
                                                  "repair_gold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.in_place({"notes": "[I-72] repaired. | [nuggets] later"}, "notes", "[I-72] repaired.")
    assert not module.in_place({"notes": "[nuggets] only"}, "notes", "[I-72] repaired.")
    assert not module.in_place({"category": "x"}, "category", "y")


def test_old0024_no_longer_forbids_an_answer_the_record_supports(items):
    """The old wording asked what was said about licensing the UK's first spaceport launch and forbade any answer at
    all. A correct, cited answer scored the -8 harmful-error penalty.
    """
    item = next(i for i in items if i.id == "old0024")
    assert item.category == "near_miss_abstention" and item.expected_behaviour == "abstain_not_in_record"
    assert "conditions" in item.question and "SaxaVord" in item.question
    assert "Any answer at all" not in " ".join(item.forbidden_claims), (
        "a blanket ban on answering is what made the old item wrong: the topic IS in the record"
    )
    assert "I-72" in item.notes, "the replacement has to carry its reason, or the next reader re-litigates it"


# A question scoped to a House the stage does not have -----------------------------------------


def test_no_item_scopes_a_house_the_answer_leaves_null(items):
    """Royal Assent is taken in neither House, so its structured answer carries `house: null`; a question that
    nonetheless asks for it "in the Commons" cannot be answered correctly. This is the same rule
    `baseline.check_structured_facts` applies during review.
    """
    houses = ("Commons", "Lords")
    offenders = [
        i.id for i in items
        if "house" in i.expected_structured_facts
        and i.expected_structured_facts["house"] is None
        and any(f"in the {h}" in i.question or f"the {h} take" in i.question for h in houses)
    ]
    assert offenders == [], f"these questions scope a House their own answer leaves null: {offenders}"


def test_the_generator_cannot_produce_the_defect_again():
    """The fix at source. `_stage_question` chooses its wording from the stage's House, so a houseless stage gets a
    houseless question instead of a default of "Commons".
    """
    rng = random.Random(0)
    for _ in range(20):
        assert "Commons" not in _stage_question("Finance Act 2026", "Royal Assent", None, rng)
        assert "Lords" not in _stage_question("Finance Act 2026", "Royal Assent", "", rng)
    assert all("{house}" not in phrasing for phrasing in _STAGE_PHRASINGS_HOUSELESS)
    with_house = _stage_question("Trade Act 2021", "Second Reading", "Lords", random.Random(1))
    assert "Lords" in with_house, "a stage that does have a House must still say which"


# A label that contradicts the item's own vital nugget -----------------------------------------


def test_stance_and_colloquial_may_be_partial():
    """`CATEGORIES` forced both to expect `answer`, so two items whose own vital nugget says the answer is absent had
    to be labelled as though it were present. Both categories now span behaviours.
    """
    assert CATEGORIES["stance"] is None and CATEGORIES["colloquial"] is None


@pytest.mark.parametrize("item_id", ["hum0067", "hum0089"])
def test_the_near_miss_items_are_typed_partial(items, item_id):
    item = next(i for i in items if i.id == item_id)
    assert item.expected_behaviour == "partial"
    assert item.nuggets and item.nuggets[0].vital, "the vital nugget is what a correct answer must say"
    # Still scoreable for retrieval: `partial` items keep their judged pool, unlike a true abstention.
    assert any(grade >= 2 for grade in item.graded_relevance.values())


# --- the instrument's identity --------------------------------------------------------------------------


def test_the_stored_hash_matches_the_repaired_questions(raw, items):
    """`content_hash` covers the question, split, category and behaviour, so repairing a question changes it. The
    stored hash has to move with it: a gold set whose recorded hash names an earlier wording is a set that cannot be
    cited.
    """
    assert raw["gold_hash"] == content_hash(list(items))
    assert raw.get("repaired", "").startswith("2026-09-26"), "a repaired set should say when and why"


REPAIRED = ("old0024", "hum0067", "hum0089", "db0176", "db0179", "db0182", "db0185", "db0188", "db0191",
            "db0194", "db0197", "db0200")  # fmt: skip


def test_the_repairs_are_declared_where_a_reader_will_look(raw):
    """Every repaired item carries its issue number in `notes`, so the next reader does not re-litigate it."""
    by_id = {i["id"]: i for i in raw["items"]}
    assert set(REPAIRS) == {"old0024", "hum0067", "hum0089"}, "the hand-listed repairs are the ones with no rule"
    for item_id in REPAIRED:
        assert any(tag in by_id[item_id]["notes"] for tag in ("I-72", "I-73", "I-74")), item_id


def test_a_repaired_set_needs_no_further_repair(raw):
    """Idempotence."""
    assert repairs_for(raw["items"]) == REPAIRS


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("When did the X Act have its Royal Assent in the Commons?", "When did the X Act have its Royal Assent?"),
        ("What was the date of the Royal Assent of the X Act in the Lords?",
         "What was the date of the Royal Assent of the X Act?"),
        ("On what date did the Commons take the Royal Assent of the X Act?",
         "On what date was the Royal Assent of the X Act?"),
    ],
)
def test_the_house_clause_is_deleted_and_nothing_else_is_rewritten(before, after):
    assert houseless_question(before) == after


@pytest.mark.parametrize("question", [
    "When did the X Act have its Second Reading?",
    "What was the date of the Commons Examiners of the Royal Albert Hall Bill [HL]?",
    "Who spoke in the Commons about fisheries?",
])
def test_a_question_with_no_house_clause_is_left_alone(question):
    """The stage name "Commons Examiners" contains the word and is not a scope; nor is a mid-sentence mention. Only the
    trailing clause the generator produces is stripped.
    """
    assert houseless_question(question) is None
