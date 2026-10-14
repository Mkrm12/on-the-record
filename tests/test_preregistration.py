"""A hash-chained pre-registration ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import ledger as L

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def path(tmp_path):
    return tmp_path / "preregistration.jsonl"


def reg(path, label="dev s18 v3 re-judge", **kw):
    return L.register(path, kind=kw.pop("kind", "judged_run"), label=label,
                      plan=kw.pop("plan", "rubric v3 over dev s18; compare judged free text with 52/82"),
                      bars=kw.pop("bars", {"harmful": 0, "judged_vital": 0.75}),
                      gold=kw.pop("gold", {"content_hash": "ae2ee261bd61b0f6", "file_sha256": "f" * 64}),
                      rubric="v3", code="abc123", when=kw.pop("when", "2026-10-12T20:00:00+00:00"))  # fmt: skip


def test_an_entry_records_what_will_be_measured_and_chains_to_the_one_before(path):
    first, second = reg(path), reg(path, label="probes after S05")
    assert first["seq"] == 1 and first["prev"] == L.GENESIS and second["prev"] == first["hash"]
    assert {"gold", "rubric", "code", "plan", "bars", "kind", "label", "created"} <= set(first)
    assert L.verify(path) == []


@pytest.mark.parametrize("tamper", ["edit", "delete", "reorder", "append_forged"])
def test_any_change_after_the_fact_breaks_the_chain(path, tamper):
    for i in range(3):
        reg(path, label=f"run {i}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if tamper == "edit":
        entry = json.loads(lines[1])
        entry["bars"]["harmful"] = 3  # the goalposts moved after the run
        lines[1] = json.dumps(entry)
    elif tamper == "delete":
        del lines[1]
    elif tamper == "reorder":
        lines[0], lines[1] = lines[1], lines[0]
    else:
        forged = json.loads(lines[-1])
        forged.update(seq=4, label="forged", prev="0" * 64)
        lines.append(json.dumps(forged))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert L.verify(path), "tampering must be detected"


def test_a_report_s_citation_is_checked_against_the_ledger(path):
    entry = reg(path)
    assert L.cited(path, entry["hash"]) == entry
    assert L.cited(path, entry["hash"][:12]) == entry, "a prefix of 12 is enough to cite"
    with pytest.raises(KeyError):
        L.cited(path, "deadbeefdead")


def test_registering_needs_a_plan_and_bars():
    with pytest.raises(ValueError):
        L.register(Path("unused.jsonl"), kind="judged_run", label="x", plan="", bars={}, gold={}, rubric="v3",
                   code="c")  # fmt: skip


def test_the_committed_ledger_verifies():
    """The CI check: whatever has been registered so far still verifies."""
    real = ROOT / "reports" / "preregistration.jsonl"
    if real.exists():
        assert L.verify(real) == []


# --- QA pass 4: what the chain cannot see, and the receipt that can -----------------------------


def test_removing_the_last_entry_or_rehashing_onwards_passes_the_chain_and_fails_against_a_receipt(path):
    """A hash chain proves an entry in the middle was not edited. It cannot see the last entry removed, nor an edit re-
    hashed through to the end: only a head held outside the file can (the receipt `register` returns, and the
    committed file's history).
    """
    reg(path)
    receipt = reg(path, label="probes after S05")["hash"]
    assert L.head(path) == receipt
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0] + "\n", encoding="utf-8")  # the last entry removed
    assert L.verify(path) == [], "the chain alone cannot see it: said honestly in the docstring"
    assert L.verify(path, head=receipt), "the receipt can"
    # an edit re-hashed through to the end
    path.write_text("", encoding="utf-8")
    first = reg(path)
    receipt = reg(path, label="second")["hash"]
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    entries[0]["bars"] = {"harmful": 5}
    prev = L.GENESIS
    for e in entries:
        e["prev"], e["hash"] = prev, None
        e["hash"] = L._digest(e)
        prev = e["hash"]
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries), encoding="utf-8")
    assert L.verify(path) == [] and first["hash"] != entries[0]["hash"]
    assert L.verify(path, head=receipt)


def test_a_run_citing_a_registration_must_be_the_code_it_registered_and_come_after_it(path):
    entry = reg(path, when="2026-10-14T10:00:00+00:00")
    assert L.run_problems(entry, code="abc123", started="2026-10-14T11:00:00+00:00") == []
    assert any("code" in p for p in L.run_problems(entry, code="def456", started="2026-10-14T11:00:00+00:00"))
    assert any("before" in p for p in L.run_problems(entry, code="abc123", started="2026-10-14T09:00:00+00:00"))
