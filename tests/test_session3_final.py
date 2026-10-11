"""The closing report: pooling, the per-category regression check, and the disclosures it must not drop."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import session3_final as final  # noqa: E402


def outcome(qid, category, relevant, ranked):
    return final._Outcome(qid, category, frozenset(relevant), list(ranked))


def report_with(**splits) -> dict:
    return {
        "scored_at": "2026-09-26T03:22:02",
        "database": "hansard_v2",
        "relevant_at": 2,
        "corpus": {"embedding_recipe": "BAAI/bge-small-en-v1.5|full|v2", "embedded_chunks": 1_074_297},
        "splits": splits,
        "before": {},
        "w3_second_look": False,
    }


def split(queries, production, ceiling=None, categories=None) -> dict:
    return {
        "queries": queries,
        "production": production,
        "ceiling": ceiling or production,
        "by_category": categories or {},
        "failures": 0,
    }


# --- pooling -------------------------------------------------------------------------------------------


def test_pooling_weights_each_split_by_its_query_count():
    """A plain mean of the three split means would let three W3 queries outvote twenty dev ones."""
    got = final.pooled_categories(report_with(
        dev=split(20, {}, categories={"who_spoke": {"queries": 16, "recall@10": 0.625}}),
        test=split(16, {}, categories={"who_spoke": {"queries": 16, "recall@10": 0.2917}}),
        heldout=split(8, {}, categories={"who_spoke": {"queries": 8, "recall@10": 0.125}}),
    ))
    assert got["who_spoke"]["queries"] == 40
    assert got["who_spoke"]["recall@10"] == pytest.approx((0.625 * 16 + 0.2917 * 16 + 0.125 * 8) / 40)


def test_pooling_is_what_stops_a_dev_only_view_declaring_a_weakness_fixed():
    """`who_spoke` really does score 0.63 on dev and 0.29 and 0.13 on the other two splits. Pooled it is 0.39, which is
    the number that belongs in a report about what to fix next.
    """
    got = final.pooled_categories(report_with(
        dev=split(16, {}, categories={"who_spoke": {"queries": 16, "recall@10": 0.625}}),
        test=split(16, {}, categories={"who_spoke": {"queries": 16, "recall@10": 0.2917}}),
        heldout=split(8, {}, categories={"who_spoke": {"queries": 8, "recall@10": 0.125}}),
    ))
    assert got["who_spoke"]["recall@10"] < 0.45 < 0.625


def test_a_category_with_no_score_is_left_out_rather_than_counted_as_zero():
    got = final.pooled_categories(report_with(
        dev=split(2, {}, categories={"ambiguous": {"queries": 2, "recall@10": None}}),
    ))
    assert "ambiguous" not in got


# --- the per-category regression check -----------------------------------------------------------------


def test_category_change_reports_both_directions():
    before = [outcome("a", "who_spoke", ["s1"], ["s9"]), outcome("b", "temporal", ["s2"], ["s2"])]
    after = [outcome("a", "who_spoke", ["s1"], ["s1"]), outcome("b", "temporal", ["s2"], ["s9"])]
    change = final.category_change(after, before)
    assert change["who_spoke"]["delta"] == 1.0
    assert change["temporal"]["delta"] == -1.0


def test_only_a_fall_past_the_tolerance_counts_as_a_regression():
    """0.02 matches the bake-off's CATEGORY_TOLERANCE, so the rule reads the same whether it is vetoing a model or
    disclosing a trade-off in the model that was adopted.
    """
    change = {
        "steady": {"queries": 5, "before": 0.50, "after": 0.49, "delta": -0.01},
        "lost": {"queries": 5, "before": 0.50, "after": 0.40, "delta": -0.10},
        "gained": {"queries": 5, "before": 0.40, "after": 0.50, "delta": 0.10},
        "unknown": {"queries": 5, "before": None, "after": 0.50, "delta": None},
    }
    assert sorted(final.regressions(change)) == ["lost"]


def test_the_report_states_a_regression_instead_of_averaging_it_away():
    report = report_with(dev=split(5, {"recall@100": 0.6, "recall@10": 0.5, "mrr@10": 0.4, "ndcg@10": 0.4},
                                   categories={"comparison": {"queries": 5, "recall@10": 0.30}}))  # fmt: skip
    report["splits"]["dev"]["category_change"] = {
        "comparison": {"queries": 5, "before": 0.60, "after": 0.30, "delta": -0.30},
    }
    text = final.render(report)
    assert "Did the recipe cost anything?" in text
    assert "**Yes" in text and "comparison" in text and "-0.3000" in text


def test_the_report_says_so_plainly_when_nothing_regressed():
    report = report_with(dev=split(5, {"recall@10": 0.5}, categories={"exact_term": {"queries": 5, "recall@10": 0.9}}))
    report["splits"]["dev"]["category_change"] = {
        "exact_term": {"queries": 5, "before": 0.85, "after": 0.90, "delta": 0.05},
    }
    text = final.render(report)
    assert "No category lost more than 0.02" in text


# --- the disclosures the report must not drop -----------------------------------------------------------


def test_the_w3_second_look_is_disclosed_with_both_numbers():
    report = report_with(heldout=split(36, {"recall@100": 0.5559, "recall@10": 0.4953,
                                            "mrr@10": 0.5299, "ndcg@10": 0.4320}))  # fmt: skip
    report["w3_second_look"] = True
    report["before"] = {"heldout": {
        "source": "reports/session3_final_exam.json", "scored_at": "2026-09-25T18:57:10",
        "queries": 36,
        "production": {"recall@100": 0.4886, "recall@10": 0.4431, "mrr@10": 0.4056, "ndcg@10": 0.3649},
    }}
    text = final.render(report)
    assert "W3 has been scored twice" in text
    assert "0.4431" in text and "0.4953" in text, "both looks must be published, not just the better one"
    assert "a clean held-out result" in text and "*not*" in text
    assert "sequencing error" in text


def test_nothing_about_w3_appears_when_it_was_not_re_scored():
    text = final.render(report_with(dev=split(108, {"recall@10": 0.5681})))
    assert "scored twice" not in text and "second look" not in text
    assert "W3 is spent as a held-out set" in text, "the first exam still spent it"


def test_the_report_names_the_recipe_and_the_corpus_it_was_measured_on():
    text = final.render(report_with(dev=split(108, {"recall@10": 0.5681})))
    assert "BAAI/bge-small-en-v1.5|full|v2" in text
    assert "1,074,297" in text
    assert "Search configuration: unchanged" in text, "a reader must know the recipe is the only change"


def test_the_report_refuses_to_claim_a_graded_ndcg():
    """The judge fails its four-level calibration gate, so every nDCG in this project is binarised."""
    text = final.render(report_with(dev=split(108, {"ndcg@10": 0.4557})))
    assert "binarised" in text and "I-77" in text


def test_dev_is_not_offered_as_independent_evidence_for_the_choice():
    """The bake-off screened the recipes on the dev queries, so a dev gain confirms transfer, not selection."""
    report = report_with(dev=split(108, {"recall@100": 0.6379}))
    report["splits"]["dev"]["paired_comparison"] = {
        "metric": "recall@100", "mean": 0.0784, "low": 0.0054, "high": 0.1526, "p": 0.0384,
        "established": True,
    }
    text = final.render(report)
    assert "not independent evidence" in text
    assert "out-of-sample evidence is the test split" in text


def test_the_verdict_is_explicit():
    text = final.render(report_with(dev=split(108, {"recall@100": 0.6379, "recall@10": 0.5681},
                                              ceiling={"recall@100": 0.7444})))  # fmt: skip
    assert "Phase 2 is complete. Go for Phase 3." in text
    assert "25.6%" in text, "the first-stage miss rate is the number Phase 3 inherits"


# --- the "before" numbers ------------------------------------------------------------------------------


def test_the_before_numbers_come_from_the_files_that_recorded_them():
    """Read back rather than retyped: a number in a report that nothing on disk supports is not a result."""
    before = final.before_numbers()
    assert set(before) <= {"dev", "test", "heldout"}
    for split_name, entry in before.items():
        assert entry["source"].endswith(".json"), split_name
        assert entry["production"], split_name
    if "heldout" in before:
        assert before["heldout"]["production"]["recall@10"] == pytest.approx(0.4431, abs=5e-5)


def test_the_shipped_report_matches_the_shipped_json():
    """The rendered markdown is derived, so it must not drift from the JSON it was rendered from."""
    import json

    if not final.OUT_JSON.exists() or not final.OUT_MD.exists():
        pytest.skip("no scoring run on disk")
    if not (final.CACHE / "sweep_dev.json").exists():
        # The report's dev deltas are rendered from the local tuning cache (data/, never in git): without it the
        # rendering is legitimately different, which is not drift.
        pytest.skip("the local tuning cache the report was rendered from is not on disk")
    rendered = final.render(json.loads(final.OUT_JSON.read_text(encoding="utf-8")))
    assert rendered == final.OUT_MD.read_text(encoding="utf-8")


# --- the disclosure must survive a routine re-run --------------------------------------------------------


class _Args:
    def __init__(self, second_look_w3=False):
        self.second_look_w3 = second_look_w3


def test_a_dev_and_test_run_carries_a_previous_w3_result_forward(tmp_path, monkeypatch):
    """The pipeline scores dev and test only, on purpose. Without this, that routine run would overwrite the JSON and
    the rendered report would quietly lose the second-look disclosure -- which is worse than the sequencing error
    being disclosed.
    """
    import json

    out = tmp_path / "session3_final.json"
    out.write_text(json.dumps({
        "scored_at": "2026-09-26T03:22:02",
        "w3_second_look": True,
        "splits": {"heldout": split(36, {"recall@10": 0.4953})},
    }), encoding="utf-8")
    monkeypatch.setattr(final, "OUT_JSON", out)

    fresh = report_with(dev=split(108, {"recall@10": 0.5681}))
    final.carry_forward_heldout(fresh, _Args())
    assert fresh["splits"]["heldout"]["carried_forward_from"] == "2026-09-26T03:22:02"
    assert fresh["w3_second_look"] is True
    assert "W3 has been scored twice" in final.render(fresh)


def test_a_second_look_run_does_not_carry_anything_forward(tmp_path, monkeypatch):
    import json

    out = tmp_path / "session3_final.json"
    out.write_text(json.dumps({"scored_at": "old", "splits": {"heldout": split(36, {"recall@10": 0.1})}}),
                   encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(final, "OUT_JSON", out)
    fresh = report_with(dev=split(108, {}), heldout=split(36, {"recall@10": 0.4953}))
    final.carry_forward_heldout(fresh, _Args(second_look_w3=True))
    assert "carried_forward_from" not in fresh["splits"]["heldout"]
    assert fresh["splits"]["heldout"]["production"]["recall@10"] == 0.4953, "the fresh score wins"
