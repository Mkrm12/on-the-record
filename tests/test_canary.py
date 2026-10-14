"""The canary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval import canary as C


def asked(route="retrieval", outcome="answer", generated=3, verified=2, dropped=1, abstention=None, degraded="",
          failed=None):  # fmt: skip
    answer = SimpleNamespace(abstention=SimpleNamespace(value=abstention) if abstention else None)
    return SimpleNamespace(route=route, outcome=outcome, answer=answer, degraded_reason=degraded, failed_reason=failed,
                           claim_counts={"claims_generated": generated, "claims_verified": verified,
                                         "claims_dropped": dropped, "claims_rejected": 0})  # fmt: skip


def client(served=("deepseek-v4-flash",), prints=("fp_a",)):
    return {"requested_model": "deepseek-chat", "served_models": list(served), "system_fingerprints": list(prints)}


def record(served=("deepseek-v4-flash",), prints=("fp_a",), verifier="minicheck", rows=None, when="2026-10-12T20:00"):
    rows = rows if rows is not None else [C.row(i, asked()) for i in C.IDS]
    return C.summarise(rows, client(served, prints), verifier=verifier, spend_usd=0.004, when=when)


def test_the_canary_is_ten_fixed_dev_questions_outside_the_sealed_window():
    from app import sealed
    from goldset.schema import GOLD_PATH, load

    items = {i.id: i for i in load(GOLD_PATH)}
    assert len(C.IDS) == 10 and len(set(C.IDS)) == 10
    for item_id in C.IDS:
        item = items[item_id]
        assert item.split == "dev", f"{item_id} is not a dev question"
        assert not (item.date and sealed.hidden(item.date)), f"{item_id} is dated inside the sealed window"


def test_a_run_records_what_served_it_and_how_its_claims_verified():
    run = record()
    assert run["served_models"] == ["deepseek-v4-flash"] and run["fingerprints"] == ["fp_a"]
    assert run["verifier"] == "minicheck" and run["questions"] == 10
    assert run["claims_generated"] == 30 and run["claims_verified"] == 20
    assert run["verified_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert {r["id"] for r in run["rows"]} == set(C.IDS)


def test_the_first_run_has_nothing_to_compare_with():
    assert C.compare(None, record()) == []


def test_an_unchanged_run_reports_no_drift():
    assert C.compare(record(), record(when="2026-10-13T09:00")) == []


@pytest.mark.parametrize("change, expect", [
    ({"served": ("deepseek-v4-pro",)}, "served model"),
    ({"prints": ("fp_b",)}, "fingerprint"),
    ({"verifier": "minicheck-int8"}, "verifier"),
])  # fmt: skip
def test_a_changed_served_model_or_fingerprint_is_reported(change, expect):
    drift = C.compare(record(), record(**change))
    assert len(drift) == 1 and expect in drift[0], drift


def test_more_than_one_model_inside_one_run_is_reported_even_on_the_first_run():
    run = record(served=("deepseek-v4-flash", "deepseek-v4-pro"))
    assert any("more than one" in line for line in C.compare(None, run))


def test_a_falling_verification_rate_is_reported():
    worse = [C.row(i, asked(verified=1, dropped=2)) for i in C.IDS]  # 2/3 -> 1/3
    drift = C.compare(record(), record(rows=worse))
    assert any("verification rate" in line for line in drift), drift


def test_a_small_wobble_in_the_rate_is_not_drift():
    rows = [C.row(i, asked(verified=1 if n == 0 else 2)) for n, i in enumerate(C.IDS)]  # 20/30 -> 19/30
    assert C.compare(record(), record(rows=rows)) == []


def test_a_question_whose_outcome_or_route_changed_is_named():
    rows = [C.row(i, asked(outcome="abstain", abstention="no_evidence") if n == 3 else asked())
            for n, i in enumerate(C.IDS)]  # fmt: skip
    drift = C.compare(record(), record(rows=rows))
    assert any(C.IDS[3] in line and "answer" in line and "abstain" in line for line in drift), drift
    rows = [C.row(i, asked(route="structured") if n == 0 else asked()) for n, i in enumerate(C.IDS)]
    assert any(C.IDS[0] in line and "route" in line for line in C.compare(record(), record(rows=rows)))


@pytest.mark.parametrize("bad", [{"failed": "the generator timed out"}, {"degraded": "verifier unavailable"}])
def test_a_failed_or_degraded_answer_is_reported_on_any_run(bad):
    rows = [C.row(i, asked(**bad) if n == 0 else asked()) for n, i in enumerate(C.IDS)]
    assert any(C.IDS[0] in line for line in C.compare(None, record(rows=rows)))


def test_runs_over_different_questions_are_not_compared():
    other = record()
    other["ids"] = ["x"] * 10
    drift = C.compare(other, record())
    assert len(drift) == 1 and "not comparable" in drift[0]


def test_the_log_appends_and_reads_back_the_last_run(tmp_path):
    path = tmp_path / "canary.jsonl"
    assert C.last(path) is None
    C.append(path, record(when="2026-10-12T20:00"))
    C.append(path, record(prints=("fp_b",), when="2026-10-13T09:00"))
    assert C.last(path)["fingerprints"] == ["fp_b"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["created"] == "2026-10-12T20:00"


def test_a_run_stopped_at_its_cap_is_recorded_with_its_spend_and_reported():
    """QA pass 4 (L6): a canary stopped at its cap appended nothing, so what it spent went unrecorded."""
    current = C.summarise([C.row(C.IDS[0], asked())], client(), verifier="minicheck", spend_usd=0.05,
                          stopped="stopped at the cap ($0.050) after 1 question")  # fmt: skip
    assert current["stopped"].startswith("stopped at the cap") and current["spend_usd"] == 0.05
    assert any("stopped" in line for line in C.compare(None, current))
    assert any("stopped" in line for line in C.compare(record(), current))
