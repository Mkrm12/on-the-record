"""dev-A's migration to gold schema v3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from goldset.lint import member_named
from goldset.schema import GOLD_PATH, NUGGET_KINDS, content_hash, from_dict, load, validate, vital_required
from goldset.split import split_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("migrate_gold_v3_t", ROOT / "scripts" / "migrate_gold_v3.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

RAW = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
ITEMS = {i.id: i for i in load()}
DEV = [i for i in ITEMS.values() if i.split == "dev"]
OFFENDERS = json.loads((ROOT / "tests" / "data" / "gold_lint_offenders_2026-10-09.json").read_text(encoding="utf-8"))

# sha256 over every non-dev item as it stood before the migration.
NON_DEV_SHA = "f73684f546c4f59ab8daecbca2c9de68ba26eff7c8f9cd16d3f3b8302b4ddda3"


def _non_dev_sha(raw: dict) -> str:
    rows = [i for i in raw["items"] if i["split"] != "dev"]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def test_the_migration_is_in_place():
    assert M.check() == []


def test_the_gold_is_schema_v3_and_validates():
    assert RAW["schema_version"] == 3
    assert validate(RAW) == []


def test_the_hashes_name_the_migrated_instrument():
    items = [from_dict(i) for i in RAW["items"] if i.get("status") != "dropped"]
    assert RAW["gold_hash"] == content_hash(items) and RAW["split_hash"] == split_hash(items)
    assert "schema v3" in RAW["repaired"]


def test_every_other_split_is_untouched():
    assert _non_dev_sha(RAW) == NON_DEV_SHA


def test_every_dev_item_has_a_breadth_and_every_dev_nugget_a_kind():
    assert all(i.breadth in ("narrow", "broad") for i in DEV)
    assert all(n.kind in NUGGET_KINDS for i in DEV for n in i.nuggets)
    assert not any(i.breadth or any(n.kind for n in i.nuggets) for i in ITEMS.values() if i.split != "dev")


def test_the_32_frozen_offenders_are_all_rewritten():
    for entry in OFFENDERS["items"]:
        before, after = entry["item"], ITEMS[entry["item"]["id"]]
        old = [n["text"] for n in before["nuggets"] if n["vital"]]
        new = [n.text for n in after.nuggets if n.vital]
        assert old != new, f"{after.id} keeps the vital points the audit found rigid"
        assert after.breadth == entry["breadth"], after.id


def test_an_example_is_never_vital_and_names_its_speech():
    """An example is one member's instance of a point. The audit proves the name only if it can find the speech."""
    for item in DEV:
        for n in item.nuggets:
            if n.kind == "example":
                assert not n.vital, (item.id, n.text)
                if member_named(n.text) and not item.evidence:
                    assert n.speech_ids, (item.id, n.text)


def test_broad_items_are_scored_k_of_n():
    assert vital_required(ITEMS["hum0076"]) == 2 and sum(n.vital for n in ITEMS["hum0076"].nuggets) == 4
    assert vital_required(ITEMS["old0002"]) == 1 and sum(n.vital for n in ITEMS["old0002"].nuggets) == 2
    both = ITEMS["hum0046"]
    assert {n.side for n in both.nuggets if n.vital} == {"for", "against"} and vital_required(both) == 2


def test_d028_hum0083_says_the_bill_was_the_governments():
    item = ITEMS["hum0083"]
    texts = " ".join(n.text for n in item.nuggets if n.vital).lower()
    assert "private member" not in texts and "government" in texts and "458" in texts
    assert not any("abolished it" in f for f in item.forbidden_claims)  # the Bill did become an Act


def test_d062_db0237_names_the_substantive_opener_not_the_chair():
    item = ITEMS["db0237"]
    facts = item.expected_structured_facts
    assert (facts["mnis_id"], facts["opened_by"]) == (4051, "John Glen")
    assert item.evidence[0].speech_id == "70571D22-F0C3-4DCE-835D-805991EA0797"
    assert item.graded_relevance["70571D22-F0C3-4DCE-835D-805991EA0797"] == 2
    assert item.graded_relevance["000599A3-8F23-44F7-8BCF-A5944ADD8C3A"] == 0


def test_hum0095_answers_because_the_record_holds_the_governments_offer():
    item = ITEMS["hum0095"]
    assert item.expected_behaviour == "answer"
    assert any("210,000" in n.text for n in item.nuggets if n.vital)


def test_the_migration_refuses_to_touch_a_confirmation_split(tmp_path, monkeypatch):
    path = tmp_path / "gold.json"
    path.write_text(GOLD_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    test_id = next(i.id for i in ITEMS.values() if i.split == "test")
    monkeypatch.setitem(M.REWRITES, test_id, {"notes": "x", "_why": "a test that must be refused"})
    with pytest.raises(M.Refused, match="test"):
        M.apply(path)
    assert path.read_text(encoding="utf-8") == GOLD_PATH.read_text(encoding="utf-8")


def test_the_review_page_shows_every_change_before_and_after():
    page = M.review_page()
    for item_id in ("hum0083", "db0237", "hum0095", *(e["item"]["id"] for e in OFFENDERS["items"])):
        assert f"## {item_id}" in page, item_id
    assert page.count("**Before:**") == len(M.REWRITES)
    assert "provisional" in page and "Question (before):** What did peers say" in page


def test_the_dev_audit_is_clean_after_the_migration():
    """The objective audit over dev-A: nothing left but the items inside the sealed window, counted."""
    audit_spec = importlib.util.spec_from_file_location("audit_t", ROOT / "scripts" / "audit_gold_objective.py")
    A = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(A)
    try:
        sources = A.SqlSources.connect()
    except Exception as exc:  # noqa: BLE001 - no database is a skip, not a failure
        pytest.skip(f"hansard_v2 is not reachable ({type(exc).__name__})")
    try:
        findings = [f for item in DEV for f in A.audit_item(item, sources)]
    finally:
        sources.close()
    assert [f for f in findings if f.check != "sealed_window"] == []
    assert sorted({f.item for f in findings}) == ["db0176", "db0187"]  # Dated inside the sealed window
