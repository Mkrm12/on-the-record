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
