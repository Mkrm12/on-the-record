"""MR-07: who held an office, either side of each change of holder."""

from __future__ import annotations

from datetime import date

import pytest

from answer.contract import Abstention, Answer, Provenance, StructuredFact
from eval.oracles import members as OM
from eval.probes import offices as PO

PROV = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                  model_id="none")  # fmt: skip
SITTINGS = [date(2025, 1, 14), date(2025, 1, 15), date(2025, 3, 4), date(2025, 3, 5), date(2025, 9, 9)]


def post(mid, name, start, end=None):
    return OM.Post(mnis_id=mid, name=name, start=start, end=end)


CHANCELLOR = [post(1, "Chancellor of the Exchequer", date(2024, 7, 5), date(2025, 3, 4)),
              post(2, "Chancellor of the Exchequer", date(2025, 3, 4))]


def test_each_change_is_asked_on_the_nearest_sitting_day_inside_each_tenure():
    trials = [t for t in PO.mr07_trials(CHANCELLOR, SITTINGS) if t.form == "canonical"]  # the qualified forms aside
    asked = sorted((t.subject["date"], t.subject["mnis_id"]) for t in trials)
    assert asked == [("2025-01-15", 1), ("2025-03-05", 2)], "the change day itself is neither side"
    assert all(t.question.startswith("Who held the office of Chancellor of the Exchequer on ") for t in trials)


def test_an_office_several_people_hold_at_once_is_not_asked():
    ministers = [post(1, "Minister of State (Home Office)", date(2024, 7, 5), date(2025, 9, 1)),
                 post(2, "Minister of State (Home Office)", date(2024, 7, 9))]
    assert PO.mr07_trials(ministers, SITTINGS) == []


def test_no_trial_inside_the_sealed_window():
    posts = [post(1, "Lord Chancellor", date(2025, 1, 1), date(2026, 5, 10)), post(2, "Lord Chancellor", date(2026, 5, 10))]
    days = [date(2026, 5, 6), date(2026, 5, 12), date(2026, 9, 8)]
    for t in PO.mr07_trials(posts, days):
        assert t.subject["date"] not in ("2026-05-06", "2026-05-12")


def answer(*facts, clarify=None, abstention=None):
    return Answer(question="q?", provenance=PROV, structured_facts=tuple(facts), rendered="x",
                  facets={"clarify": clarify} if clarify else {}, abstention=abstention,
                  abstention_detail="no holder is recorded" if abstention else "")  # fmt: skip


def holder(mid):
    return StructuredFact(template="office_held_at_date", params={"office_key": "k"},
                          value={"office": "Chancellor of the Exchequer", "mnis_id": mid, "holder": f"M{mid}"})


@pytest.fixture
def trial():
    return next(t for t in PO.mr07_trials(CHANCELLOR, SITTINGS) if t.subject.get("mnis_id") == 1)


def test_the_holder_is_a_pass_and_anyone_else_a_harm(trial):
    verdict = PO.verdict_for(())
    assert verdict(trial, answer(holder(1)))[0] == "pass"
    assert verdict(trial, answer(holder(2)))[0] == "harm"


def test_no_holder_said_of_a_held_office_is_a_harm_and_a_question_back_that_offers_the_holder_is_clarified(trial):
    verdict = PO.verdict_for(())
    assert verdict(trial, answer(abstention=Abstention.NOT_IN_RECORD))[0] == "harm"
    assert verdict(trial, answer(clarify=[{"mnis_id": 1}, {"mnis_id": 2}]))[0] == "clarified"
    assert verdict(trial, answer(clarify=[{"mnis_id": 3}]))[0] == "declined"


def test_the_posts_oracle_reads_government_and_opposition_posts_raw(tmp_path):
    import gzip
    import json

    folder = tmp_path / "members" / "7"
    folder.mkdir(parents=True)
    bio = {"value": {"governmentPosts": [{"name": "Chancellor of the Exchequer", "startDate": "2024-07-05T00:00:00",
                                          "endDate": None}],
                     "oppositionPosts": [{"name": "Shadow Chancellor of the Exchequer", "startDate": "2021-11-29T00:00:00",
                                          "endDate": "2024-07-05T00:00:00"}], "otherPosts": [{"name": "Chair, X"}]}}
    with gzip.open(folder / "Biography.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(bio, fh)
    names = sorted(p.name for p in OM.posts(tmp_path / "members"))
    assert names == ["Chancellor of the Exchequer", "Shadow Chancellor of the Exchequer"]


# --- the qualified forms' truth: relations are answered, so the oracle checks them --------------------

THREE = [post(1, "Chancellor of the Exchequer", date(2023, 1, 1), date(2024, 7, 5)),
         post(2, "Chancellor of the Exchequer", date(2024, 7, 5), date(2025, 3, 4)),
         post(3, "Chancellor of the Exchequer", date(2025, 3, 4))]


def _qualified(posts, form, **kw):
    return [t for t in PO.mr07_trials(posts, SITTINGS, **kw) if t.form == form]


def test_the_predecessor_and_successor_trials_carry_the_raw_posts_answer():
    (pred,) = _qualified(THREE, "predecessor")
    (succ,) = _qualified(THREE, "successor")
    assert pred.subject["not_mnis_id"] == 2 and pred.subject["relation_mnis_id"] == 1
    assert succ.subject["not_mnis_id"] == 2 and succ.subject["relation_mnis_id"] == 3


def test_with_no_earlier_holder_in_the_posts_the_predecessor_is_unknown_not_absent():
    (pred,) = _qualified(CHANCELLOR, "predecessor")
    assert pred.subject["relation_mnis_id"] is None


def _fact(mid):
    return StructuredFact(template="office_held_at_date", value={"mnis_id": mid, "holder": "X"})


def test_the_right_relation_passes_the_holder_is_a_harm_and_another_is_a_harm():
    verdict = PO.verdict_for(())
    (pred,) = _qualified(THREE, "predecessor")
    ok = Answer(question="q", provenance=PROV, structured_facts=(_fact(1),))
    assert verdict(pred, ok)[0] == "pass"
    assert verdict(pred, Answer(question="q", provenance=PROV, structured_facts=(_fact(2),)))[0] == "harm"
    assert verdict(pred, Answer(question="q", provenance=PROV, structured_facts=(_fact(3),)))[0] == "harm"
    refused = Answer(question="q", provenance=PROV, abstention=Abstention.INSUFFICIENT_EVIDENCE, abstention_detail="x")
    assert verdict(pred, refused)[0] == "declined", "the posts hold the predecessor: refusing is not a pass"
    (unknown,) = _qualified(CHANCELLOR, "predecessor")
    assert verdict(unknown, refused)[0] == "pass", "no earlier holder in the posts: refusing is right"


def test_the_party_asked_is_never_one_in_the_offices_name_nor_the_holders_own():
    snp = [post(1, "SNP Westminster Leader", date(2024, 7, 5), date(2025, 3, 4)),
           post(2, "SNP Westminster Leader", date(2025, 3, 4))]
    history = {1: [OM.Affiliation("Liberal Democrat", date(2015, 1, 1), None, 1)],
               2: [OM.Affiliation("Scottish National Party", date(2015, 1, 1), None, 1)]}
    for t in PO.mr07_trials(snp, SITTINGS, history=history):
        if t.form in ("party", "in_party", "for_party"):
            assert "SNP SNP" not in t.question and "the SNP" not in t.question.replace("the SNP Westminster", "")
            assert "Liberal Democrat" not in t.question, "the holder's own party that day is never the one asked"


def test_the_relation_truth_is_the_posts_as_the_mask_shows_them():
    """A post wholly inside the sealed window is not in the masked record, so it is never the predecessor or successor
    the record could name (S06's MR-07 run: the oracle named one, which the route rightly could not see). Nor is the
    holder after it: this test had certified 3 as 1's successor, which is false (2 held it between, unseen), and QA
    pass 9 (High 2) found the route asserting exactly that.
    """
    from app import sealed

    w = sealed.window()
    posts = [post(1, "Green Spokesperson", date(2024, 7, 4), w.start + (w.end - w.start) / 3),
             post(2, "Green Spokesperson", w.start + (w.end - w.start) / 3, w.end - (w.end - w.start) / 3),
             post(3, "Green Spokesperson", w.end + (w.end - w.start) / 3)]
    sittings = [date(2025, 1, 14), w.end + (w.end - w.start) / 2]
    (succ,) = [t for t in PO.mr07_trials(posts, sittings) if t.form == "successor"]
    assert succ.subject["not_mnis_id"] == 1 and succ.subject["relation_mnis_id"] is None, (
        "post 2 lies wholly inside the window: the masked record cannot say who came after 1")
    assert all(t.subject.get("relation_mnis_id") != 2 for t in PO.mr07_trials(posts, sittings)), "never the sealed one"


# --- a change of holder across the sealed window (QA pass 9, High 2)
# ---------------------------------------------------


@pytest.mark.parametrize("ended, began", [
    (date(2026, 5, 5), date(2026, 5, 6)),  # clipped on both sides by the mask
    (date(2026, 3, 20), date(2026, 8, 3)),  # clear of it on both sides, but the gap spans it
])
def test_a_relation_whose_change_touches_the_sealed_window_must_be_refused(ended, began):
    """The masked posts cannot show a post wholly inside the window, so a holder could sit between these two: the
    oracle's truth is "no relation the record can name", which only a refusal passes (it had certified the code's
    own blind spot by computing its truth on the same masked posts).
    """
    pair = [post(1, "Paymaster General", date(2025, 1, 1), ended), post(2, "Paymaster General", began)]
    (succ,) = _qualified(pair, "successor")  # one qualified set per office, on its first post
    assert succ.subject["not_mnis_id"] == 1 and succ.subject["relation_mnis_id"] is None, succ.question
    assert PO._neighbour(pair, pair[1], "predecessor") is None, "the other direction, on the later post"
    assert PO._neighbour(pair, pair[0], "successor") is None


def test_naming_a_relation_the_posts_do_not_hold_is_a_harm():
    verdict = PO.verdict_for(())
    (unknown,) = _qualified(CHANCELLOR, "predecessor")
    named = Answer(question="q", provenance=PROV, structured_facts=(_fact(7),))
    assert verdict(unknown, named)[0] == "harm"
