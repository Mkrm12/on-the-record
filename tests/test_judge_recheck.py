"""The judge re-check. Pure parts only: sampling, agreement breakdown, verdict."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import judge_recheck as jr  # noqa: E402
from goldset.judge import Pair, Pool  # noqa: E402


def pool_of(spec: list[tuple[str, str, int]]) -> tuple[Pool, dict[str, dict[str, int]]]:
    """(pool, grades) from (item, speech, grade) triples."""
    pairs = [Pair(item_id=i, speech_id=s, question=f"q for {i}", passage=f"passage {s}") for i, s, _ in spec]
    grades: dict[str, dict[str, int]] = {}
    for item, speech, grade in spec:
        grades.setdefault(item, {})[speech] = grade
    return Pool(pairs=pairs, variants=["dense"], retrieved={}), grades


# --- stratified sampling -------------------------------------------------------------------------------


def test_the_sample_is_spread_across_grades_not_proportional():
    """A proportional sample of a pool that is 90% grade 0 would give almost no pairs at 2 and 3 -- and those are the
    only ones the binary relevance gate turns on.
    """
    spec = [(f"i{n}", f"s{n}", 0) for n in range(90)]
    spec += [(f"j{n}", f"t{n}", 2) for n in range(5)]
    spec += [(f"k{n}", f"u{n}", 3) for n in range(5)]
    pool, grades = pool_of(spec)
    sample = jr.stratified(pool, grades, 12)
    census = jr._grade_census(sample, grades)
    assert len(sample) == 12
    assert census.get("2", 0) >= 4 and census.get("3", 0) >= 4, f"grades 2 and 3 under-sampled: {census}"


def test_sampling_is_deterministic_for_a_seed():
    pool, grades = pool_of([(f"i{n}", f"s{n}", n % 4) for n in range(40)])
    first = [(p.item_id, p.speech_id) for p in jr.stratified(pool, grades, 10, seed=7)]
    again = [(p.item_id, p.speech_id) for p in jr.stratified(pool, grades, 10, seed=7)]
    other = [(p.item_id, p.speech_id) for p in jr.stratified(pool, grades, 10, seed=8)]
    assert first == again and first != other


def test_an_unjudged_pair_is_never_sampled():
    """There is nothing for it to agree or disagree with, so including it would dilute the measurement."""
    pool, grades = pool_of([("i1", "s1", 2)])
    pool.pairs.append(Pair(item_id="i2", speech_id="s2", question="q", passage="p"))
    sample = jr.stratified(pool, grades, 10)
    assert [(p.item_id, p.speech_id) for p in sample] == [("i1", "s1")]


def test_asking_for_more_than_exists_returns_what_there_is():
    pool, grades = pool_of([("i1", "s1", 0), ("i2", "s2", 3)])
    assert len(jr.stratified(pool, grades, 50)) == 2


def test_a_zero_sample_is_a_bug():
    pool, grades = pool_of([("i1", "s1", 0)])
    with pytest.raises(ValueError, match="at least 1"):
        jr.stratified(pool, grades, 0)


def test_the_census_leaves_out_pairs_with_no_stored_grade():
    pool, grades = pool_of([("i1", "s1", 2)])
    extra = Pair(item_id="zz", speech_id="zz", question="q", passage="p")
    assert jr._grade_census([*pool.pairs, extra], grades) == {"2": 1}


# --- the agreement breakdown ---------------------------------------------------------------------------


def test_agreement_is_broken_down_by_the_stored_grade():
    """A judge that disagrees only at the 1/2 boundary has drifted in a way that moves every recall number; one that
    disagrees at 0 and 3 has a different problem. The headline kappa hides which.
    """
    old = {("i", "a"): 1, ("i", "b"): 1, ("i", "c"): 0, ("i", "d"): 3}
    new = {("i", "a"): 2, ("i", "b"): 1, ("i", "c"): 0, ("i", "d"): 3}
    got = jr.per_stratum_agreement(old, new)
    assert got["1"] == {"pairs": 2, "agreement": 0.5}
    assert got["0"]["agreement"] == 1.0 and got["3"]["agreement"] == 1.0


def test_a_pair_the_new_run_did_not_grade_is_left_out_of_the_breakdown():
    old = {("i", "a"): 2, ("i", "b"): 2}
    assert jr.per_stratum_agreement(old, {("i", "a"): 2})["2"]["pairs"] == 1


def test_crossings_report_the_direction_of_a_drift():
    """A judge that became generous inflates recall; one that became strict deflates it. Both give a low kappa and they
    need opposite responses.
    """
    old = {("i", "a"): 1, ("i", "b"): 2, ("i", "c"): 3}
    new = {("i", "a"): 2, ("i", "b"): 1, ("i", "c"): 3}
    assert jr.crossings(old, new) == {"became_relevant": 1, "became_irrelevant": 1}


def test_crossings_ignore_a_change_that_stays_on_one_side_of_the_line():
    """0 to 1 and 2 to 3 are four-level disagreements and change no binary label."""
    assert jr.crossings({("i", "a"): 0, ("i", "b"): 2}, {("i", "a"): 1, ("i", "b"): 3}) == {
        "became_relevant": 0, "became_irrelevant": 0}  # fmt: skip


# --- the pre-registered verdict ------------------------------------------------------------------------


def test_the_gate_is_the_binarised_kappa_not_the_four_level_one():
    """The bug this pins, with the real numbers from the run that hit it. Binarised kappa 0.9565, four-level 0.7679.
    """
    got, why = jr.verdict(binarised=0.9565, agreement=0.9797, four_level=0.7679)
    assert got == "stable", "a judge that agrees 0.96 on the relevance line has not drifted"
    assert "may be mixed" in why
    assert "0.7679" in why and "I-77" in why, "the four-level number is reported, not hidden"


def test_kappa_at_or_above_the_gate_allows_mixing():
    got, why = jr.verdict(0.80, 0.93)
    assert got == "stable" and "may be mixed" in why


def test_a_genuine_drift_in_the_relevance_line_still_fails():
    got, why = jr.verdict(0.72, 0.88, four_level=0.60)
    assert got == "drifted"
    assert "must NOT be mixed" in why and "re-judge the whole pool" in why


def test_a_low_four_level_kappa_alone_never_fails_the_gate():
    """It must not read as drift every time."""
    for four in (0.40, 0.55, 0.79):
        got, _ = jr.verdict(binarised=0.95, agreement=0.98, four_level=four)
        assert got == "stable", f"four-level {four} must not fail a binarised 0.95"


def test_no_comparable_pair_is_unknown_not_stable():
    """The failure this prevents: an empty comparison reading as a pass and licensing new judgments."""
    got, why = jr.verdict(None, None)
    assert got == "unknown" and "Do not add judgments" in why


def test_the_gate_is_the_one_the_plan_pre_registered():
    assert jr.MIX_GATE == 0.80 and jr.DEFAULT_SAMPLE == 150


# --- provenance, which is the real point of section 4.2 ------------------------------------------------


def test_the_client_records_which_model_actually_answered():
    """The stored judgments record only the alias `deepseek-flash`, so nothing on disk proves what graded them. Every
    new judgment must carry the served id.
    """
    from goldset.llm import Client

    client = Client(key="x", base="https://example", model="deepseek-flash")
    client._observe({"model": "deepseek-v4.1-flash", "system_fingerprint": "fp_abc"})
    got = client.provenance()
    assert got["requested_model"] == "deepseek-flash"
    assert got["served_models"] == ["deepseek-v4.1-flash"]
    assert got["system_fingerprints"] == ["fp_abc"]
    assert got["one_model"] is True


def test_two_models_answering_one_run_is_flagged_as_not_one_system():
    from goldset.llm import Client

    client = Client(key="x", base="https://example", model="deepseek-flash")
    client._observe({"model": "deepseek-v4.1-flash"})
    client._observe({"model": "deepseek-v4-pro"})
    assert client.provenance()["one_model"] is False


def test_provenance_survives_a_response_that_says_nothing_about_the_model():
    """Provenance is worth having and is not worth failing a judgment over."""
    from goldset.llm import Client

    client = Client(key="x", base="https://example", model="deepseek-flash")
    client._observe({})
    client._observe({"model": None, "system_fingerprint": ""})
    assert client.provenance()["served_models"] == []


# --- the cost, which the spend guard depends on -------------------------------------------------------


def test_the_estimate_is_read_off_the_dataclass_not_with_dict_get():
    """An earlier draft used `.get("usd", 0.0)` on an `Estimate` dataclass, which priced every run at $0 and made the
    spend check incapable of refusing anything. 150 pairs really cost about 2.6 cents.
    """
    from app.config import settings
    from goldset.judge import estimate_judging

    pool, grades = pool_of([(f"i{n}", f"s{n}", n % 4) for n in range(150)])
    estimate = estimate_judging(pool, settings.bulk_model, "", limit=None)
    assert not isinstance(estimate, dict)
    assert estimate.usd > 0.0 and estimate.calls == 150
    assert "rates" in estimate.line()
