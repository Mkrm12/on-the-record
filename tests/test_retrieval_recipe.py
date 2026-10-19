"""The embedding recipe, pinned to exact strings."""

from __future__ import annotations

import pytest

from retrieval.recipe import (
    HEADER_RECIPES,
    INCUMBENT_RECIPE,
    MEAN_SENTENCE_TOKENS,
    RECIPES_BY_KEY,
    RecipeError,
    resolve_recipes,
)

# A real row, in the shape `gpu/embedder.py`'s FETCH_SQL produces.
ROW = {
    "body": "In December, SaxaVord spaceport in the Shetland Islands became the UK’s first licensed "
            "vertical-launch spaceport, with more spaceports to follow.",
    "date": "2024-05-08",
    "house": "Commons",
    "heading": "Space Industry",
    "speaker_name": "Mr Jonathan Lord",
    "party": "Conservative",
}

# recipe key -> the exact text that must be embedded for ROW.
GOLDEN = {
    "sentence":
        "In December, SaxaVord spaceport in the Shetland Islands became the UK’s first licensed "
        "vertical-launch spaceport, with more spaceports to follow.",
    "heading":
        "[Space Industry] In December, SaxaVord spaceport in the Shetland Islands became the UK’s first "
        "licensed vertical-launch spaceport, with more spaceports to follow.",
    "heading_date":
        "[2024-05-08 | Space Industry] In December, SaxaVord spaceport in the Shetland Islands became the "
        "UK’s first licensed vertical-launch spaceport, with more spaceports to follow.",
    "speaker_heading":
        "[Mr Jonathan Lord | Space Industry] In December, SaxaVord spaceport in the Shetland Islands became "
        "the UK’s first licensed vertical-launch spaceport, with more spaceports to follow.",
    "full":
        "[2024-05-08 | Commons | Space Industry | Mr Jonathan Lord | Conservative] In December, SaxaVord "
        "spaceport in the Shetland Islands became the UK’s first licensed vertical-launch spaceport, "
        "with more spaceports to follow.",
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_the_rendered_text_is_exactly_this(key):
    assert RECIPES_BY_KEY[key].render(ROW) == GOLDEN[key]


def test_production_embeds_the_full_recipe():
    """The tag on every row in `hansard_v2` is `BAAI/bge-small-en-v1.5|full|v2`. If this assertion has to change, the
    corpus has to be re-embedded.
    """
    assert RECIPES_BY_KEY["full"].fields == ("date", "house", "heading", "speaker_name", "party")


def test_every_declared_recipe_has_a_golden_string():
    """A new recipe without a pinned string is a recipe nobody can verify was deployed as measured."""
    assert {r.key for r in HEADER_RECIPES} == set(GOLDEN)


def test_the_incumbent_is_the_bare_sentence_and_comes_first():
    """Everything is measured against it, so its identity and its position are both load-bearing."""
    assert INCUMBENT_RECIPE is HEADER_RECIPES[0]
    assert INCUMBENT_RECIPE.key == "sentence" and INCUMBENT_RECIPE.fields == ()


# --- the field rules ------------------------------------------------------------------------------------


def test_a_missing_field_is_dropped_rather_than_rendered_as_none():
    """The failure this prevents: embedding the literal text "None" into 1.07M vectors, where it becomes a token that
    matches every chunk with a missing field.
    """
    row = {**ROW, "house": None, "party": "", "speaker_name": "None"}
    assert RECIPES_BY_KEY["full"].render(row).startswith("[2024-05-08 | Space Industry] In December")


def test_a_row_with_no_metadata_at_all_renders_the_bare_body():
    row = {"body": "Royal Assent.", "date": None, "house": None, "heading": None, "speaker_name": None,
           "party": None}  # fmt: skip
    assert RECIPES_BY_KEY["full"].render(row) == "Royal Assent."


def test_the_date_is_truncated_to_ten_characters():
    assert RECIPES_BY_KEY["heading_date"].render(
        {**ROW, "date": "2024-05-08 00:00:00+00"}).startswith("[2024-05-08 | Space Industry]")


def test_whitespace_inside_a_header_field_is_collapsed_but_the_body_keeps_its_characters():
    row = {**ROW, "heading": "  Space\n\tIndustry  ", "body": "A  double  spaced body."}
    out = RECIPES_BY_KEY["heading"].render(row)
    assert out == "[Space Industry] A  double  spaced body."


def test_the_body_is_stripped_but_not_otherwise_touched():
    assert RECIPES_BY_KEY["sentence"].render({"body": "  Text.  "}) == "Text."


def test_a_pipe_in_a_field_does_not_forge_an_extra_header_slot():
    """A heading containing the separator would otherwise look like two fields to anything parsing the rendered text
    back. Nothing parses it today, so this pins the current behaviour rather than asserting a fix: if it ever
    matters, it will be visible here.
    """
    out = RECIPES_BY_KEY["heading"].render({**ROW, "heading": "Space | Industry"})
    assert out.startswith("[Space | Industry] ")


# --- the cost of a header -------------------------------------------------------------------------------


def test_the_overhead_ratio_reports_the_headers_share_of_the_embedded_text():
    """The full header is about 40% of a 29-token sentence's vector."""
    full = RECIPES_BY_KEY["full"].overhead_ratio()
    assert 0.35 < full < 0.45
    assert RECIPES_BY_KEY["sentence"].overhead_ratio() == 0.0
    assert MEAN_SENTENCE_TOKENS == 29


def test_overhead_rises_with_every_field_added():
    ratios = [RECIPES_BY_KEY[k].overhead_ratio() for k in ("sentence", "heading", "heading_date", "full")]
    assert ratios == sorted(ratios), "a longer header cannot cost less"


# --- resolution -----------------------------------------------------------------------------------------


def test_resolving_nothing_gives_every_recipe_in_declaration_order():
    assert resolve_recipes(None) == list(HEADER_RECIPES)
    assert resolve_recipes([]) == list(HEADER_RECIPES)


def test_resolving_a_subset_keeps_the_order_asked_for():
    assert [r.key for r in resolve_recipes(["full", "sentence"])] == ["full", "sentence"]


def test_an_unknown_recipe_is_refused_by_name():
    with pytest.raises(RecipeError, match="kitchen_sink"):
        resolve_recipes(["kitchen_sink"])
