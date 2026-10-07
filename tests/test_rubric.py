"""One rubric for every grader of E3's answers: the maintainer's Gemini, the maintainer's calibration page, the scorer.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from goldset import rubric as R
from goldset.schema import EXPECTED_BEHAVIOURS

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_rubric_tests", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EA = _script("eval_answers")
GJ = _script("gemini_judge")


def test_the_rubric_grades_in_the_scorers_vocabulary():
    assert R.OUTCOMES == EA.OUTCOMES
    assert set(R.OUTCOME_DEFINITIONS) == set(R.OUTCOMES)


def test_every_behaviour_the_gold_set_can_expect_is_put_into_words():
    assert set(R.EXPECTED_IN_WORDS) == set(EXPECTED_BEHAVIOURS)


def test_an_item_without_a_behaviour_is_graded_as_one_to_answer():
    """The schema refuses an unknown behaviour, so this is the defensive path — and it must still say something."""
    from types import SimpleNamespace

    assert R.expected_text(SimpleNamespace()) == R.EXPECTED_IN_WORDS["answer"]
    assert R.expected_text(SimpleNamespace(expected_behaviour=None)) == R.EXPECTED_IN_WORDS["answer"]


def test_the_gemini_export_uses_the_rubric_and_nothing_of_its_own():
    assert GJ.EXPECTED_IN_WORDS is R.EXPECTED_IN_WORDS
    assert GJ.answer_text is R.answer_text
    assert GJ.OUTCOMES == R.OUTCOMES


def test_the_prompt_carries_every_definition_rule_behaviour_and_reply_field():
    prompt = R.judging_prompt()
    for name, definition in R.OUTCOME_DEFINITIONS.items():
        assert f'"{name}": {definition}' in prompt
    for rule in R.GRADING_RULES:
        assert rule in prompt
    for text in R.EXPECTED_IN_WORDS.values():
        assert text in prompt
    for field in R.REPLY_FIELDS + ("packet_id", "judge_model", "verdicts"):
        assert f'"{field}"' in prompt


def test_the_worked_example_is_a_verdict_the_importer_accepts():
    """The example teaches the reply's shape; if the importer would refuse it, Gemini is taught to be refused."""
    prompt = R.judging_prompt()
    raw = prompt[prompt.index('{"id": "J017"'):]
    depth, end = 0, 0
    for n, ch in enumerate(raw):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0:
            end = n + 1
            break
    verdict = json.loads(" ".join(raw[:end].split()))
    assert set(verdict) == set(R.REPLY_FIELDS) and verdict["confidence"] in R.CONFIDENCE
    key = {"packet_id": "p1", "map": {"J017": "hum0001"}, "vital_totals": {"J017": 2}}
    assert GJ.check({"packet_id": "p1", "verdicts": [verdict]}, key) == []


def test_the_prompt_at_the_foot_of_gemini_log_is_the_rubric_exactly():
    """Regenerate it with `uv run python scripts/gemini_judge.py prompt --write`."""
    log = (ROOT / "logs" / "GEMINI_LOG.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert GJ.PROMPT_START in log and GJ.PROMPT_END in log, "the prompt's markers are missing from GEMINI_LOG.md"
    between = log[log.index(GJ.PROMPT_START) + len(GJ.PROMPT_START):log.index(GJ.PROMPT_END)]
    fenced = re.search(r"```text\n(.*?)```", between, flags=re.S)
    assert fenced, "the prompt must sit in a ```text fence so it can be copied whole"
    assert fenced.group(1) == R.judging_prompt()


@pytest.mark.parametrize("row, shown", [
    ({"rendered": "Keir Mather said X. [2026-09-10]"}, "Keir Mather said X. [2026-09-10]"),
    ({"rendered": "", "abstention": "false_premise", "abstention_detail": "It was said in the Lords, not the Commons."},
     "(the system declined: It was said in the Lords, not the Commons.)"),
    ({"rendered": "", "abstention": "not_in_record", "abstention_detail": ""}, "(the system declined: no reason given)"),
])  # fmt: skip
def test_a_grader_sees_the_answer_a_reader_saw_declines_included(row, shown):
    assert R.answer_text(row) == shown
    assert R.is_decline(row) == (not row["rendered"])
