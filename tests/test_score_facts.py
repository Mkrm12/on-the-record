"""The fact engine's dev score under the mask."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_facts_t", ROOT / "scripts" / "score_facts.py")
SF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SF)


def test_an_item_whose_gold_is_dated_in_the_window_is_counted_apart():
    assert SF.sealed_facts({"bill_id": 4090, "first_sitting": "2026-04-29", "sittings": 1})
    assert not SF.sealed_facts({"bill_id": 4129, "first_sitting": "2026-09-15", "sittings": 1})
    assert not SF.sealed_facts({"division_id": 2418, "ayes": 3})


def test_a_sealed_item_answered_with_a_sealed_date_is_a_breach():
    class Got:
        ok, value = True, {"first_sitting": "2026-04-29"}

    assert SF.breach(Got()) == ["2026-04-29"]
    Got.value = {"first_sitting": "2026-09-15"}
    assert SF.breach(Got()) == []
