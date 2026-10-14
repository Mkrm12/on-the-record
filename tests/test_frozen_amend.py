"""Amending frozen retrieval: a named change, a recomputed hash, and the history in the file."""

from __future__ import annotations

import json
import shutil

import pytest

from answer.pipeline import FROZEN_PATH, Frozen, PipelineError, amend_frozen, frozen_hash


@pytest.fixture
def frozen_copy(tmp_path):
    """A copy of the real file, complete: if it predates `dense_index`, the knob is named first, as amendment 1 did.
    """
    path = tmp_path / "retrieval_frozen.json"
    shutil.copy(FROZEN_PATH, path)
    if "dense_index" not in json.loads(path.read_text(encoding="utf-8"))["config"]:
        amend_frozen(path, {"dense_index": "vector"}, reason="name the form", evidence="pg_indexes", date="2026-10-05")
    return path


def test_the_hash_is_the_one_freeze_and_verify_compute(frozen_copy):
    data = json.loads(frozen_copy.read_text(encoding="utf-8"))
    assert frozen_hash(data["config"]) == data["config_hash"]
    assert Frozen.load(frozen_copy).verify_hash()


def test_an_amendment_sets_the_knob_moves_the_hash_and_records_the_change(frozen_copy):
    before = json.loads(frozen_copy.read_text(encoding="utf-8"))
    old = before["config"].get("ef_search")
    entry = amend_frozen(frozen_copy, {"ef_search": 120}, reason="a test", evidence="none needed", date="2026-10-05")
    after = json.loads(frozen_copy.read_text(encoding="utf-8"))
    assert after["config"]["ef_search"] == 120
    assert after["config_hash"] == frozen_hash(after["config"]) != before["config_hash"]
    assert Frozen.load(frozen_copy).verify_hash()
    assert after["amendments"][-1] == entry
    assert entry == {"date": "2026-10-05", "changed": {"ef_search": [old, 120]}, "previous_hash": before["config_hash"],
                     "config_hash": after["config_hash"], "reason": "a test", "evidence": "none needed"}
    untouched = {k: v for k, v in before.items() if k not in ("config", "config_hash", "amendments")}
    assert {k: after[k] for k in untouched} == untouched, "frozen_at, adopted_arms, reason, recipe, note stay as frozen"


def test_amendments_form_a_chain_of_hashes(frozen_copy):
    first = amend_frozen(frozen_copy, {"ef_search": 120}, reason="r1", evidence="e1", date="2026-10-05")
    second = amend_frozen(frozen_copy, {"ef_search": 100}, reason="r2", evidence="e2", date="2026-10-06")
    assert second["previous_hash"] == first["config_hash"]
    data = json.loads(frozen_copy.read_text(encoding="utf-8"))
    assert data["config_hash"] == second["config_hash"]


def test_a_knob_the_file_never_named_can_be_added_and_the_change_says_so(frozen_copy):
    data = json.loads(frozen_copy.read_text(encoding="utf-8"))
    data["config"].pop("dense_index", None)
    data["config_hash"] = frozen_hash(data["config"])
    frozen_copy.write_text(json.dumps(data), encoding="utf-8")
    entry = amend_frozen(frozen_copy, {"dense_index": "vector"}, reason="name it", evidence="pg_indexes",
                         date="2026-10-05")
    assert entry["changed"] == {"dense_index": [None, "vector"]}
    assert Frozen.load(frozen_copy).search_config().dense_index == "vector"


@pytest.mark.parametrize(("changes", "match"), [
    ({"no_such_knob": 1}, "no_such_knob"),
    ({"filters": {}}, "filters"),
    ({"dense_index": "bit"}, "dense_index"),  # SearchConfig itself refuses it
    ({"ef_search": 0}, "ef_search"),
    ({}, "nothing"),
])
def test_an_amendment_that_would_not_build_a_valid_config_is_refused_and_nothing_is_written(frozen_copy, changes,
                                                                                          match):
    before = frozen_copy.read_bytes()
    with pytest.raises((PipelineError, ValueError), match=match):
        amend_frozen(frozen_copy, changes, reason="r", evidence="e", date="2026-10-05")
    assert frozen_copy.read_bytes() == before


@pytest.mark.parametrize(("reason", "evidence"), [("", "e"), ("r", ""), ("  ", "e")])
def test_an_amendment_without_a_reason_and_its_evidence_is_refused(frozen_copy, reason, evidence):
    with pytest.raises(PipelineError, match="reason|evidence"):
        amend_frozen(frozen_copy, {"ef_search": 120}, reason=reason, evidence=evidence, date="2026-10-05")


def test_a_no_op_amendment_is_refused(frozen_copy):
    current = json.loads(frozen_copy.read_text(encoding="utf-8"))["config"]["ef_search"]
    with pytest.raises(PipelineError, match="nothing"):
        amend_frozen(frozen_copy, {"ef_search": current}, reason="r", evidence="e", date="2026-10-05")


def test_a_file_whose_hash_is_already_stale_is_not_amended_over(frozen_copy):
    """Amending on top of a hand edit would launder it: the new hash would cover a change nobody recorded."""
    data = json.loads(frozen_copy.read_text(encoding="utf-8"))
    data["config"]["pool"] = 7  # the hand edit, hash left alone
    frozen_copy.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PipelineError, match="hash"):
        amend_frozen(frozen_copy, {"ef_search": 120}, reason="r", evidence="e", date="2026-10-05")
