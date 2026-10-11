"""The embedding recipe: the exact text that gets embedded for a chunk."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# The measured mean length of a Hansard sentence in tokens, used to express a header's cost as a share of the embedded
# text.
MEAN_SENTENCE_TOKENS = 29


class RecipeError(ValueError):
    """A recipe key that does not exist. `bakeoff.BakeoffError` subclasses this, so a bake-off preflight that catches
    BakeoffError still catches a bad recipe from here.
    """


# --- the recipes -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeaderRecipe:
    """How a chunk's text is built before it is embedded."""

    key: str
    fields: tuple[str, ...]
    expected_overhead_tokens: int
    note: str = ""

    def render(self, row: dict[str, Any]) -> str:
        """The exact text that gets embedded for this chunk."""
        body = (row.get("body") or "").strip()
        if not self.fields:
            return body
        parts = [_header_value(row, name) for name in self.fields]
        present = [p for p in parts if p]
        if not present:
            return body
        return f"[{' | '.join(present)}] {body}"

    def overhead_ratio(self, mean_sentence_tokens: int = MEAN_SENTENCE_TOKENS) -> float:
        """What share of the embedded text is header rather than speech."""
        total = mean_sentence_tokens + self.expected_overhead_tokens
        return self.expected_overhead_tokens / total if total else 0.0


def _header_value(row: dict[str, Any], name: str) -> str:
    """One header field, normalised. Absent is empty, never the string "None"."""
    value = row.get(name)
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    if name == "date":
        # The ISO date is what the corpus stores; it is also the form the questions use.
        return text[:10]
    return _collapse(text)


_WHITESPACE = re.compile(r"\s+")


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


# `sentence` is the incumbent and must stay first: everything is measured against it, and it is the only one already in
# production.
HEADER_RECIPES: tuple[HeaderRecipe, ...] = (
    HeaderRecipe("sentence", (), 0, "the incumbent: the sentence alone, exactly as production embeds it"),
    HeaderRecipe("heading", ("heading",), 6, "the debate heading only: the cheapest context there is"),
    HeaderRecipe("heading_date", ("date", "heading"), 11, "heading plus the sitting date, for temporal questions"),
    HeaderRecipe(
        "speaker_heading", ("speaker_name", "heading"), 12,
        "who said it and what the debate was: aimed at the attribution failures, not at temporal ones",
    ),
    HeaderRecipe(
        "full", ("date", "house", "heading", "speaker_name", "party"), 20,
        "everything the brief lists; the one most at risk of drowning a 29-token sentence",
    ),
)

RECIPES_BY_KEY: dict[str, HeaderRecipe] = {r.key: r for r in HEADER_RECIPES}
INCUMBENT_RECIPE = HEADER_RECIPES[0]


def resolve_recipes(keys: Sequence[str] | None) -> list[HeaderRecipe]:
    if not keys:
        return list(HEADER_RECIPES)
    out = []
    for key in keys:
        recipe = RECIPES_BY_KEY.get(key)
        if recipe is None:
            raise RecipeError(f"unknown header recipe {key!r}; known keys are {', '.join(sorted(RECIPES_BY_KEY))}")
        out.append(recipe)
    return out
