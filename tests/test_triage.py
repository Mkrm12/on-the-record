"""The maintainer's triage of the judge's contested verdicts: blind first, strata fixed first, kappa still a kappa
(`goldset.triage`, `scripts/triage_review.py`).
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import pytest

from goldset import triage as T
from goldset.judge import cohen_kappa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import triage_review as TR  # noqa: E402

JUDGED = {f"J{n:03d}": {"outcome": "correct"} for n in range(1, 41)}


def test_the_strata_are_fixed_by_the_seed_and_draw_controls_only_from_the_rest():
    flagged, named, gold = ["J001", "J002"], ["J003"], ["J004", "J005"]
    a = T.design(JUDGED, flagged, named, gold, n_controls=4, seed=18)
    b = T.design(JUDGED, flagged, named, gold, n_controls=4, seed=18)
    assert a == b
    controls = a["strata"]["controls"]
    assert not set(controls["ids"]) & {*flagged, *named, *gold}
    assert controls["population"] == 40 - 5 and controls["weight"] == pytest.approx(35 / 4)
    assert a["strata"]["flagged"]["weight"] == 1 and a["kappa_population"] == 38
    assert a["excluded_from_kappa"] == gold and a["cards"] == 2 + 1 + 4 + 1


def test_the_owners_cap_and_a_doubled_or_unknown_id_are_refused():
    with pytest.raises(ValueError, match="cap"):
        T.design(JUDGED, [f"J{n:03d}" for n in range(1, 17)], [], [], n_controls=4, seed=18)
    with pytest.raises(ValueError, match="two strata"):
        T.design(JUDGED, ["J001"], ["J001"], [], n_controls=1, seed=18)
    with pytest.raises(ValueError, match="not in the judged run"):
        T.design(JUDGED, ["J999"], [], [], n_controls=1, seed=18)


def test_weighted_kappa_with_unit_weights_is_cohens_kappa():
    owner = ["correct", "harmful", "correct", "missing_vital_nugget", "correct", "over_abstention"]
    judge = ["correct", "correct", "correct", "missing_vital_nugget", "harmful", "over_abstention"]
    plain = cohen_kappa({(i,): o for i, o in enumerate(owner)}, {(i,): j for i, j in enumerate(judge)})
    weighted = T.weighted_kappa([(o, j, 1.0) for o, j in zip(owner, judge)])
    assert weighted["kappa"] == plain["kappa"] and weighted["agreement"] == plain["agreement"]


def test_weights_stand_rows_in_for_their_stratum():
    # one disagreement in a stratum of 1 (weight 1), two agreements standing for 10 rows each
    pairs = [("correct", "missing_vital_nugget", 1.0), ("correct", "correct", 10.0),
             ("missing_vital_nugget", "missing_vital_nugget", 10.0)]
    w = T.weighted_kappa(pairs)
    assert w["agreement"] == pytest.approx(20 / 21, abs=1e-4)
    assert w["kappa"] > T.weighted_kappa([(a, b, 1.0) for a, b, _ in pairs])["kappa"]
    assert T.weighted_kappa([("correct", "correct", 3.0)])["kappa"] == 1.0
    assert T.weighted_kappa([])["kappa"] is None


def key_for(cards: dict[str, tuple[str, str, int, int]], *, controls=(), population=10) -> dict:
    """A key: jid ->; vital_total 1."""
    def card(jid, j, c, jv, cv, stratum):
        return {"item_id": f"i{jid}", "stratum": stratum,
                "judge": {"outcome": j, "vital_present": jv, "vital_total": 1, "why": "w"},
                "claude": {"outcome": c, "vital_present": cv}}
    out = {jid: card(jid, *v, "flagged") for jid, v in cards.items()}
    out.update({jid: card(jid, "correct", "correct", 1, 1, "controls") for jid in controls})
    flagged = sorted(cards)
    return {"page": T.PAGE_VERSION, "packet_id": "p1", "seed": 18, "kappa_population": len(flagged) + population,
            "strata": {"flagged": {"ids": flagged, "population": len(flagged), "weight": 1.0},
                       "owner_named": {"ids": [], "population": 0, "weight": 0.0},
                       "controls": {"ids": sorted(controls), "population": population,
                                    "weight": population / len(controls) if controls else 0.0}},
            "cards": out, "gold_cards": {"divisions": {"ids": ["J090"], "item_ids": ["div1"]}}}


def label(blind, choice, final=None, final_vital=None, *, at=("2026-10-06T10:00:00.000Z", "2026-10-06T10:00:05.000Z")):
    return {"blind": blind, "blind_at": at[0], "revealed_at": at[1], "choice": choice, "final": final,
            "final_vital": final_vital}


KEY = key_for({"J001": ("over_abstention", "missing_vital_nugget", 0, 0),
               "J002": ("correct_missing_okay_nuggets", "correct", 1, 1),
               "J003": ("missing_vital_nugget", "correct", 0, 1)}, controls=("J010", "J011"))


def labels(**cards):
    return {"page": T.PAGE_VERSION, "packet_id": "p1", "cards": cards, "gold": {"divisions": {"choice": "B"}}}


def test_kappa_is_computed_from_the_blind_grade_not_the_one_chosen_after_the_reveal():
    lab = labels(J001=label("over_abstention", "B"), J002=label("correct", "B"), J003=label("correct", "B"),
                 J010=label("correct", "A"), J011=label("correct", "A"))
    out = T.score(lab, KEY, reps=200)
    assert out["disagreements_blind"] == ["J002", "J003"]
    assert out["reveal_changed_the_grade"]["cards"] == ["J001"]
    assert out["adjudicated"]["J001"] == {"outcome": "missing_vital_nugget", "vital_present": 0}
    assert out["kappa_on_sample"]["n"] == 5 and "passes_gate" not in out["kappa_on_sample"]
    assert out["complete"] and out["gold_decisions"] == {"divisions": "B"}


def test_no_harmful_card_means_recall_is_not_measured_and_a_missed_one_fails():
    clean = T.score(labels(J001=label("over_abstention", "A"), J002=label("correct_missing_okay_nuggets", "A"),
                           J003=label("missing_vital_nugget", "A"), J010=label("correct", "A"),
                           J011=label("correct", "A")), KEY, reps=200)
    assert clean["harmful"]["recall"] is None and "vacuously" in clean["harmful"]["note"]
    assert clean["passes"]
    missed = T.score(labels(J001=label("harmful", "C", "harmful", 0), J002=label("correct_missing_okay_nuggets", "A"),
                            J003=label("missing_vital_nugget", "A"), J010=label("correct", "A"),
                            J011=label("correct", "A")), KEY, reps=200)
    assert missed["harmful"]["recall"] == 0.0 and not missed["passes"]


def test_an_undecided_card_or_entry_leaves_it_incomplete_and_never_passing():
    lab = labels(J001=label("over_abstention", "A"), J002=label("correct_missing_okay_nuggets", None),
                 J003=label("missing_vital_nugget", "A"), J010=label("correct", "A"), J011=label("correct", "A"))
    out = T.score(lab, KEY, reps=200)
    assert not out["complete"] and not out["passes"]
    lab["cards"]["J002"]["choice"] = "A"
    lab["gold"] = {}
    assert not T.score(lab, KEY, reps=200)["complete"]


@pytest.mark.parametrize("card, problem", [
    (label("correct", "A", at=("2026-10-06T10:00:09.000Z", "2026-10-06T10:00:05.000Z")), "after the reveal"),
    ({"blind": None, "revealed_at": "2026-10-06T10:00:05.000Z", "choice": "A"}, "no blind grade"),
    (label("maybe", "A"), "not one of the outcomes"),
    (label("correct", "C", "missing_vital_nugget", None), "needs a vital count"),
    (label("correct", "C", "correct", 0), "needs every vital point"),
    (label("correct", "C", "missing_vital_nugget", 1), "needs a missing point"),
    (label("correct", "D"), "not A, B or C"),
])
def test_labels_that_are_not_blind_or_not_grades_are_refused(card, problem):
    assert any(problem in p for p in T.check_labels(labels(J001=card), KEY))
    with pytest.raises(ValueError):
        T.score(labels(J001=card), KEY, reps=10)


def test_labels_from_another_build_or_for_unknown_cards_are_refused():
    other = {**labels(), "packet_id": "p2"}
    assert any("not" in p for p in T.check_labels(other, KEY))
    assert any("not a card" in p for p in T.check_labels(labels(J777=label("correct", "A")), KEY))


def test_adjudication_applies_final_grades_and_the_entry_ruling_to_a_copy():
    rows = [{"id": "iJ001", "outcome": "over_abstention", "vital_present": 0, "vital_total": 2, "by": "gemini"},
            {"id": "div1", "outcome": "over_abstention", "vital_present": 0, "vital_total": 1, "by": "gemini"},
            {"id": "other", "outcome": "correct", "vital_present": 1, "vital_total": 1, "by": "deterministic"}]
    result = {"adjudicated": {"J001": {"outcome": "missing_vital_nugget", "vital_present": 0}},
              "gold_decisions": {"divisions": "B"}}
    out, clarify = T.adjudicate_rows(rows, KEY, result)
    assert out[0]["outcome"] == "missing_vital_nugget" and out[0]["by"] == "owner"
    assert out[1]["outcome"] == "correct_abstention" and out[1]["vital_present"] == 1 and clarify == ["div1"]
    assert out[2] == rows[2] and rows[0]["outcome"] == "over_abstention"  # the input is not mutated


TAKES_READY = all((ROOT / p).exists() for p in ("reports/triage_takes.json", "reports/e3_eval_dev_s18_final_judged.json",
                                                "reports/gemini/packet_dev_s18_final.json"))


@pytest.mark.skipif(not TAKES_READY, reason="the dev run's packet, verdicts and takes are local data")
def test_the_built_page_hides_every_verdict_and_take_until_its_card_is_revealed():
    takes = json.loads((ROOT / "reports" / "triage_takes.json").read_text(encoding="utf-8"))
    blind, reveal, gold, key = TR.assemble(takes)
    assert len(blind) + len(gold) <= T.CAP and len(blind) == len(key["cards"])
    assert all(set(card) == {"jid", "question", "expected", "reference_points", "vital_total", "must_not_claim",
                             "answer_is_a_decline", "answer"} for card in blind)  # nothing a grader produced
    page = TR.page_html(blind, reveal, gold, {"page": key["page"], "packet_id": key["packet_id"], "built_at": "t"})
    plain = re.sub(r'const REVEAL = "[A-Za-z0-9+/=]*";', "", page)
    for jid, r in reveal.items():
        assert r["judge"]["why"] not in plain and r["take"] not in plain, jid
    hidden = json.loads(base64.b64decode(re.search(r'const REVEAL = "([A-Za-z0-9+/=]*)";', page).group(1)))
    assert hidden == json.loads(json.dumps(reveal))
    assert "</script>" not in page.split("<script>", 1)[1].rsplit("</script>", 1)[0]


@pytest.mark.skipif(not TAKES_READY, reason="the dev run's packet, verdicts and takes are local data")
def test_a_take_that_no_longer_matches_the_seeded_draw_is_refused():
    takes = json.loads((ROOT / "reports" / "triage_takes.json").read_text(encoding="utf-8"))
    takes["seed"] = takes["seed"] + 1
    with pytest.raises(TR.BuildRefused, match="seeded draw"):
        TR.assemble(takes)
