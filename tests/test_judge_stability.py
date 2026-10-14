"""The judge-stability instrument's own logic (scripts/judge_stability.py): what it re-judges and how it counts flips.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("judge_stability", ROOT / "scripts" / "judge_stability.py")
JS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(JS)


def row(i, outcome, by="groq", rendered="A short answer."):
    return {"id": f"x{i}", "outcome": outcome, "by": by, "rendered": rendered}


def test_only_the_judges_whole_answers_are_re_judged_and_rare_verdicts_first():
    rows = ([row(i, "correct") for i in range(10)] + [row(20, "harmful"), row(21, "wrong_but_benign")]
            + [row(30, "harmful", by="deterministic-structured"), row(31, "correct", rendered="x" * 600),
               row(32, "correct", rendered="   ")])
    chosen = JS.choose(rows, "groq", limit=5)
    assert [r["id"] for r in chosen[:2]] == ["x20", "x21"] and len(chosen) == 5
    assert all(r["by"] == "groq" and 0 < len(r["rendered"].strip()) < JS.STORED_WHOLE_BELOW for r in chosen)


def test_a_flip_is_any_disagreement_on_identical_input():
    got = JS.summarise({"a": ["correct", "correct", "correct"], "b": ["correct", "harmful", "correct"],
                        "c": ["harmful", "harmful", "harmful"]})
    assert (got["items"], got["flipped"], got["flip_rate"]) == (3, 1, 0.3333)
    assert (got["harmful_ever"], got["harmful_every_time"]) == (2, 1)
