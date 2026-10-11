"""Gold schema v3: what a reference point IS, and how many a broad question needs."""

from __future__ import annotations

import pytest

from goldset.schema import (
    BREADTHS,
    GOLD_PATH,
    NUGGET_KINDS,
    SCHEMA_VERSION,
    Evidence,
    GoldItem,
    Nugget,
    from_dict,
    load,
    to_dict,
    validate,
    vital_required,
)


def item(**kwargs) -> GoldItem:
    base = dict(id="hum0001", question="What was said about fracking and whether it is safe?", category="multi_speech",
                criticality="major", expected_behaviour="answer", provenance="passage_llm",
                evidence=(Evidence("SPEECH-1", "There are some 20 million people who live near fracking pads."),))
    return GoldItem(**(base | kwargs))


def wrap(*items: GoldItem, version: int = SCHEMA_VERSION) -> dict:
    return {"schema_version": version, "items": [to_dict(i) for i in items]}


BROAD = item(breadth="broad", min_points=2, nuggets=(
    Nugget("An argument that fracking can be done safely.", True, kind="point", side="for"),
    Nugget("An argument that it risks contaminating groundwater.", True, kind="point", side="against"),
    Nugget("Richard Tice pointed to millions living near US pads.", False, kind="example", side="for"),
))


def test_the_committed_gold_set_loads():
    items = load(GOLD_PATH)
    assert len(items) >= 488 - 5 and all(isinstance(i, GoldItem) for i in items)


def test_a_v2_item_has_no_v3_fields_and_serialises_exactly_as_before():
    plain = item(nuggets=(Nugget("A point.", True),))
    out = to_dict(plain)
    assert "breadth" not in out and "min_points" not in out
    assert out["nuggets"] == [{"text": "A point.", "vital": True, "speech_ids": []}]
    assert validate(wrap(plain, version=2)) == [] and validate(wrap(plain, version=3)) == []


def test_v3_fields_round_trip():
    again = from_dict(to_dict(BROAD))
    assert again == BROAD and again.nuggets[0].kind == "point" and again.nuggets[2].side == "for"
    assert validate(wrap(BROAD)) == []


def test_v3_fields_in_a_file_that_says_version_2_are_refused():
    errors = validate(wrap(BROAD, version=2))
    assert any("schema_version 3" in e for e in errors)


def test_the_schema_version_must_be_2_or_3():
    assert any("schema_version" in e for e in validate(wrap(item(), version=1)))
    assert any("schema_version" in e for e in validate(wrap(item(), version=4)))


@pytest.mark.parametrize("bad", ["Broad", "wide", "both"])
def test_breadth_comes_from_a_closed_list(bad):
    assert BREADTHS == ("narrow", "broad")
    assert any("breadth" in e for e in validate(wrap(item(breadth=bad))))


@pytest.mark.parametrize("bad", ["Point", "quote", "claim"])
def test_a_reference_point_kind_comes_from_a_closed_list(bad):
    assert NUGGET_KINDS == ("fact", "point", "example", "scope")
    assert any("kind" in e for e in validate(wrap(item(nuggets=(Nugget("x y z", True, kind=bad),)))))


@pytest.mark.parametrize("k", [0, 3, -1])
def test_min_points_is_a_count_from_one_to_the_vital_points(k):
    errors = validate(wrap(item(breadth="broad", min_points=k, nuggets=BROAD.nuggets)))
    assert any("min_points" in e for e in errors)


def test_min_points_belongs_to_broad_items_only():
    errors = validate(wrap(item(breadth="narrow", min_points=1, nuggets=BROAD.nuggets)))
    assert any("min_points" in e for e in errors)


def test_min_points_must_be_a_whole_number():
    data = wrap(BROAD)
    data["items"][0]["min_points"] = 1.5
    assert any("min_points" in e for e in validate(data))


def test_sides_are_meaningful_only_when_the_vital_points_cover_two_of_them():
    one_side = item(breadth="broad", min_points=1, nuggets=(Nugget("A for point.", True, kind="point", side="for"),
                                                             Nugget("Another for point.", True, kind="point",
                                                                    side="for")))
    assert any("side" in e for e in validate(wrap(one_side)))


def test_min_points_must_allow_one_point_from_each_side():
    """A both-sides item needs a point from each side, so a k below the number of sides could never be met as stated.
    """
    too_few = item(breadth="broad", min_points=1, nuggets=BROAD.nuggets)
    assert any("one from each side" in e for e in validate(wrap(too_few)))


def test_how_many_vital_points_a_right_answer_needs():
    assert vital_required(BROAD) == 2  # k of n on a broad item
    narrow = item(nuggets=(Nugget("A.", True), Nugget("B.", True), Nugget("C.", False)))
    assert vital_required(narrow) == 2  # every vital point, as in v2
    assert vital_required(item(nuggets=())) == 0
