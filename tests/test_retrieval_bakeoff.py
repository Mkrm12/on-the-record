"""The bake-off's pure logic: candidate gating, header rendering, the plan, and the regression guard."""

from __future__ import annotations

import pytest
from retrieval import bakeoff


# --- candidate gating --------------------------------------------------------------------------------


def test_gated_weights_are_excluded_from_an_unattended_run():
    """An unattended job that stops on an authentication prompt three hours in is worse than one that never offered the
    model.
    """
    gated = [s for s in bakeoff.CANDIDATES if s.gated]
    assert gated, "embeddinggemma is gated and must be marked as such"
    assert not any(s.gated for s in bakeoff.default_candidates())


def test_a_gated_model_can_still_be_asked_for_explicitly():
    """Once the maintainer has accepted the licence and cached the weights it must be runnable."""
    [spec] = bakeoff.resolve_candidates(["embeddinggemma"])
    assert spec.gated and spec.key == "embeddinggemma"


def test_an_unknown_model_key_is_refused_with_the_known_ones_listed():
    with pytest.raises(bakeoff.BakeoffError, match="unknown model"):
        bakeoff.resolve_candidates(["bge-enormous"])


def test_a_model_that_cannot_fit_the_card_is_refused_before_the_run_not_during_it():
    with pytest.raises(bakeoff.BakeoffError, match="VRAM"):
        bakeoff.resolve_candidates(["qwen3-0.6b"], vram_mb=512)


def test_every_candidate_declares_a_licence_because_the_brief_requires_checking_them():
    for spec in bakeoff.CANDIDATES:
        assert spec.licence, f"{spec.key} has no licence recorded"


def test_the_incumbent_is_a_candidate_so_every_number_has_something_to_beat():
    assert bakeoff.INCUMBENT_KEY in bakeoff.BY_KEY
    incumbent = bakeoff.BY_KEY[bakeoff.INCUMBENT_KEY]
    assert incumbent.dimensions == 384, "production runs bge-small at 384 dimensions"
    assert incumbent.revision, "the incumbent's revision is known and must stay pinned"


def test_a_candidate_without_a_pinned_revision_is_allowed_but_declared():
    """Inventing a revision hash would be fabricating a fact. The worker resolves and records it instead."""
    unpinned = [s for s in bakeoff.CANDIDATES if not s.revision]
    assert unpinned, "the new candidates are not pinned; the manifest records what was actually used"


def test_the_vram_estimate_grows_with_the_model_and_leaves_headroom():
    small = bakeoff.BY_KEY["bge-small"].vram_estimate_mb
    large = bakeoff.BY_KEY["qwen3-0.6b"].vram_estimate_mb
    assert small < large
    assert large < 8192, "every default candidate must plausibly fit the 8 GB card"
    assert all(s.fits_in(8192) for s in bakeoff.default_candidates())


# --- header rendering --------------------------------------------------------------------------------


ROW = {
    "body": "The Government published their response to the consultation.",
    "date": "2026-09-02",
    "house": "Commons",
    "heading": "Local Government Reorganisation",
    "speaker_name": "A Member",
    "party": "Labour",
}


def test_the_incumbent_recipe_embeds_the_sentence_and_nothing_else():
    assert bakeoff.RECIPES_BY_KEY["sentence"].render(ROW) == ROW["body"]


def test_a_header_is_bracketed_and_pipe_separated_in_the_declared_field_order():
    assert bakeoff.RECIPES_BY_KEY["heading_date"].render(ROW) == (
        "[2026-09-02 | Local Government Reorganisation] "
        "The Government published their response to the consultation."
    )


def test_the_full_recipe_renders_every_field_the_brief_names():
    rendered = bakeoff.RECIPES_BY_KEY["full"].render(ROW)
    for value in ("2026-09-02", "Commons", "Local Government Reorganisation", "A Member", "Labour"):
        assert value in rendered
    assert rendered.startswith("[") and "] " in rendered


def test_a_missing_field_is_omitted_rather_than_rendered_as_none():
    """`str(None)` in an embedded header is a token of pure noise repeated 90,000 times."""
    rendered = bakeoff.RECIPES_BY_KEY["full"].render({**ROW, "speaker_name": None, "party": ""})
    assert "None" not in rendered and "|  |" not in rendered
    assert "Commons" in rendered


def test_the_literal_string_none_from_the_database_is_also_treated_as_absent():
    rendered = bakeoff.RECIPES_BY_KEY["heading"].render({**ROW, "heading": "none"})
    assert rendered == ROW["body"], "a header of nothing must not leave empty brackets"


def test_a_row_with_no_metadata_at_all_falls_back_to_the_bare_sentence():
    assert bakeoff.RECIPES_BY_KEY["full"].render({"body": "Just the words."}) == "Just the words."


def test_whitespace_in_a_header_field_is_collapsed():
    rendered = bakeoff.RECIPES_BY_KEY["heading"].render({**ROW, "heading": "Local\n  Government   Reform"})
    assert "[Local Government Reform] " in rendered


def test_the_date_is_truncated_to_the_day_not_the_timestamp():
    rendered = bakeoff.RECIPES_BY_KEY["heading_date"].render({**ROW, "date": "2026-09-02 00:00:00+00"})
    assert "[2026-09-02 | " in rendered


def test_the_body_is_stripped_so_a_header_does_not_get_double_spaced():
    assert bakeoff.RECIPES_BY_KEY["heading"].render({**ROW, "body": "  padded  "}).endswith("] padded")


def test_every_recipe_records_what_it_costs_the_sentence():
    """That has to be a number in the report, not a worry in a docstring."""
    assert bakeoff.RECIPES_BY_KEY["sentence"].overhead_ratio() == 0.0
    full = bakeoff.RECIPES_BY_KEY["full"].overhead_ratio()
    assert 0.3 < full < 0.5, f"a five-field header should be a third to a half of the text, got {full:.2f}"


def test_a_cheaper_header_costs_proportionally_less():
    ratios = [bakeoff.RECIPES_BY_KEY[k].overhead_ratio() for k in ("sentence", "heading", "heading_date", "full")]
    assert ratios == sorted(ratios), "overhead must rise monotonically with the number of fields"


# --- recipe identity ---------------------------------------------------------------------------------


def test_the_recipe_id_distinguishes_model_header_and_version():
    spec = bakeoff.BY_KEY["bge-base"]
    a = bakeoff.recipe_id(spec, bakeoff.RECIPES_BY_KEY["sentence"])
    b = bakeoff.recipe_id(spec, bakeoff.RECIPES_BY_KEY["full"])
    c = bakeoff.recipe_id(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    assert a != b != c and a != c
    assert all(x.count("|") == 2 for x in (a, b, c))


def test_the_recipe_id_matches_the_incumbents_format_so_the_queue_predicate_still_works():
    """`gpu/embedder.py` builds `name|recipe|version` and its PENDING predicate compares against it. A different shape
    here would make every production vector look pending.
    """
    got = bakeoff.recipe_id(bakeoff.BY_KEY["bge-small"], bakeoff.INCUMBENT_RECIPE, version="v1")
    assert got == "BAAI/bge-small-en-v1.5|sentence|v1"


def test_a_shard_stem_is_filesystem_safe_and_unique_per_combination():
    stems = {bakeoff.shard_stem(s, r) for s in bakeoff.CANDIDATES for r in bakeoff.HEADER_RECIPES}
    assert len(stems) == len(bakeoff.CANDIDATES) * len(bakeoff.HEADER_RECIPES)
    assert all(not set(stem) & set('/\\:*?"<>| ') for stem in stems)


# --- the plan ----------------------------------------------------------------------------------------


def test_the_plan_screens_headers_on_the_incumbent_first_then_the_other_models():
    trials = bakeoff.plan(bakeoff.default_candidates(), list(bakeoff.HEADER_RECIPES))
    header_stage = [t for t in trials if t.spec.key == bakeoff.INCUMBENT_KEY]
    assert len(header_stage) == len(bakeoff.HEADER_RECIPES)
    others = [t for t in trials if t.spec.key != bakeoff.INCUMBENT_KEY]
    assert all(t.recipe.key == "sentence" for t in others)
    assert len(trials) < len(bakeoff.default_candidates()) * len(bakeoff.HEADER_RECIPES)


def test_the_plan_has_no_duplicate_trials():
    trials = bakeoff.plan(bakeoff.default_candidates(), list(bakeoff.HEADER_RECIPES))
    assert len({t.stem for t in trials}) == len(trials)


def test_the_cross_product_is_available_for_a_deliberate_deeper_run():
    models = bakeoff.resolve_candidates(["bge-small", "bge-base"])
    recipes = bakeoff.resolve_recipes(["sentence", "full"])
    trials = bakeoff.plan(models, recipes, incumbent_only_headers=False)
    assert len(trials) == 4


def test_stage_two_puts_every_other_model_on_the_winning_header():
    winner = bakeoff.RECIPES_BY_KEY["heading"]
    trials = bakeoff.stage_two(bakeoff.default_candidates(), winner)
    assert trials and all(t.recipe.key == "heading" for t in trials)
    assert all(t.spec.key != bakeoff.INCUMBENT_KEY for t in trials)


def test_an_unknown_recipe_is_refused():
    """Raised by `retrieval.recipe` since E1 moved the renderer there (section 4.5). `BakeoffError` subclasses
    `RecipeError`, so one `except BakeoffError` in a bake-off preflight still covers a bad recipe key -- the catch
    has to be widened, not narrowed, and this pins that direction.
    """
    from retrieval.recipe import RecipeError

    with pytest.raises(RecipeError, match="unknown header recipe"):
        bakeoff.resolve_recipes(["kitchen_sink"])
    assert issubclass(bakeoff.BakeoffError, RecipeError)


def test_the_estimate_scales_with_model_size_and_corpus_size():
    small = bakeoff.plan([bakeoff.BY_KEY["bge-small"]], [bakeoff.INCUMBENT_RECIPE])
    big = bakeoff.plan([bakeoff.BY_KEY["qwen3-0.6b"]], [bakeoff.INCUMBENT_RECIPE], incumbent_only_headers=False)
    assert bakeoff.estimate_minutes(big, 90_040) > bakeoff.estimate_minutes(small, 90_040)
    assert bakeoff.estimate_minutes(small, 180_000) > bakeoff.estimate_minutes(small, 90_040)
    assert bakeoff.estimate_minutes(small, 90_040) > 0


# --- the regression guard ----------------------------------------------------------------------------


def result(label: str, **per_category: float) -> bakeoff.TrialResult:
    return bakeoff.TrialResult(
        label=label, stem=label, model_key="m", recipe_key="r",
        recall_at_100=0.6, recall_at_10=0.5, mrr_at_10=0.45, per_category=dict(per_category),
    )  # fmt: skip


def test_a_candidate_that_wins_overall_but_loses_multi_hop_is_flagged():
    """The maintainer's rule: better searches beat a better average. A mean that hides a collapse in the hard
    categories is exactly the number not to optimise.
    """
    base = result("incumbent", multi_hop=0.50, exact_term=0.60)
    other = result("candidate", multi_hop=0.30, exact_term=0.80)
    flagged = bakeoff.regressions(base, other)
    assert any("multi_hop" in f for f in flagged)
    assert not any("exact_term" in f for f in flagged), "an unprotected category is not a blocker"


def test_a_small_wobble_inside_the_tolerance_is_not_called_a_regression():
    base = result("incumbent", multi_hop=0.50)
    other = result("candidate", multi_hop=0.49)
    assert bakeoff.regressions(other=other, base=base) == []


def test_a_category_missing_from_either_side_is_skipped_rather_than_guessed():
    assert bakeoff.regressions(result("a", multi_hop=0.5), result("b")) == []
    assert bakeoff.regressions(result("a"), result("b", multi_hop=0.1)) == []


def test_the_protected_categories_cover_the_hard_question_types():
    for category in ("multi_hop", "multi_speech", "named_speaker", "temporal"):
        assert category in bakeoff.PROTECTED_CATEGORIES


def test_a_result_line_shows_the_header_cost_beside_the_recall():
    line = bakeoff.TrialResult(
        label="bge-base + full", stem="s", model_key="bge-base", recipe_key="full",
        recall_at_100=0.61, recall_at_10=0.52, mrr_at_10=0.47, dimensions=768,
        header_overhead_ratio=0.41,
    ).line()  # fmt: skip
    assert "0.6100" in line and "768" in line and "41%" in line


def test_a_missing_metric_prints_as_not_available_rather_than_zero():
    """A trial that failed to score must not look like a trial that scored zero."""
    assert "n/a" in bakeoff.TrialResult(label="x", stem="x", model_key="m", recipe_key="r").line()


# --- the adoption decision --------------------------------------------------------------------------- The rule is
# applied without judgement on purpose.


def _row(trial, model, recipe, recall, **cats):
    return {"trial": trial, "label": trial, "model_key": model, "recipe_key": recipe,
            "recall_at_100": recall, "recall_at_10": recall - 0.1, "mrr_at_10": recall - 0.15,
            "dimensions": 384, "per_category": dict(cats), "chunks": 120_000}  # fmt: skip


def _decide(rows):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bakeoff_decide", __import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "bakeoff_decide.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decide(rows)


def test_a_clear_win_with_no_regression_is_adopted():
    rows = [
        _row("bge-small__sentence", "bge-small", "sentence", 0.70, multi_hop=0.50, temporal=0.60),
        _row("bge-base__sentence", "bge-base", "sentence", 0.78, multi_hop=0.55, temporal=0.62),
    ]
    decision = _decide(rows)
    assert decision["adopt"] and decision["winner"] == "bge-base__sentence"
    assert decision["gain_vs_incumbent"] > 0


def test_a_win_inside_the_noise_floor_is_refused():
    """One query is worth about 0.009 of recall on 108 queries, so a 0.01 'win' is one query."""
    rows = [
        _row("bge-small__sentence", "bge-small", "sentence", 0.70, multi_hop=0.50),
        _row("bge-base__sentence", "bge-base", "sentence", 0.71, multi_hop=0.50),
    ]
    decision = _decide(rows)
    assert not decision["adopt"]
    assert any("noise floor" in r for c in decision["considered"] for r in c["reasons"])


def test_a_win_that_costs_multi_hop_is_refused_however_big_the_mean_gain():
    """The maintainer's standing rule: better searches beat a better average."""
    rows = [
        _row("bge-small__sentence", "bge-small", "sentence", 0.70, multi_hop=0.60, temporal=0.60),
        _row("qwen3-0.6b__sentence", "qwen3-0.6b", "sentence", 0.85, multi_hop=0.30, temporal=0.62),
    ]
    decision = _decide(rows)
    assert not decision["adopt"]
    assert any("regresses protected categories" in r for c in decision["considered"] for r in c["reasons"])


def test_without_the_incumbent_nothing_is_adopted_because_there_is_no_baseline():
    decision = _decide([_row("bge-base__sentence", "bge-base", "sentence", 0.9)])
    assert not decision["adopt"] and "not screened" in decision["reason"]


def test_an_empty_screen_adopts_nothing():
    assert not _decide([])["adopt"]


def test_the_best_eligible_candidate_wins_not_merely_the_first():
    rows = [
        _row("bge-small__sentence", "bge-small", "sentence", 0.70),
        _row("bge-base__sentence", "bge-base", "sentence", 0.75),
        _row("gte-modernbert__sentence", "gte-modernbert", "sentence", 0.80),
    ]
    assert _decide(rows)["winner"] == "gte-modernbert__sentence"


def test_every_decision_carries_its_caveats_so_a_screening_win_is_not_mistaken_for_a_pipeline_win():
    decision = _decide([_row("bge-small__sentence", "bge-small", "sentence", 0.70)])
    joined = " ".join(decision["caveats"])
    assert "optimistic" in joined and "dense-only" in joined


# --- candidates known not to work on this stack ------------------------------------------------------ arctic-m-v2 took
# three attempts, each failing further in: "please install xformers" (no Windows wheel for torch 2.11+cu128), then a
# TypeError when the flag was passed as a model kwarg instead of a config kwarg, then a CUDA device-side assert once the
# override took effect.


def test_a_candidate_known_not_to_work_is_out_of_the_default_set():
    arctic = bakeoff.BY_KEY["arctic-m-v2"]
    assert arctic.unsupported, "the incompatibility must be recorded, not just remembered"
    assert arctic not in bakeoff.default_candidates()


def test_the_recorded_reason_names_the_actual_failures():
    reason = bakeoff.BY_KEY["arctic-m-v2"].unsupported.lower()
    assert "xformers" in reason and "assert" in reason


def test_an_unsupported_candidate_can_still_be_demanded_explicitly():
    """Recorded, not removed: someone with a different stack may well get it working."""
    [spec] = bakeoff.resolve_candidates(["arctic-m-v2"])
    assert spec.key == "arctic-m-v2"


def test_the_default_set_is_the_four_that_actually_work():
    assert [s.key for s in bakeoff.default_candidates()] == [
        "bge-small", "bge-base", "gte-modernbert", "qwen3-0.6b",
    ]


def test_a_config_override_is_separate_from_the_attention_implementation():
    """They go to different places: config_kwargs vs model_kwargs. Passing a config field to the model constructor is
    what produced 'TypeError: unexpected keyword argument'.
    """
    arctic = bakeoff.BY_KEY["arctic-m-v2"]
    assert "use_memory_efficient_attention" in arctic.config_overrides
    assert arctic.attn_implementation == "sdpa"
    assert "attn_implementation" not in arctic.config_overrides


def test_a_plain_candidate_needs_no_overrides():
    for key in ("bge-small", "bge-base", "qwen3-0.6b"):
        assert bakeoff.BY_KEY[key].config_overrides == {}


def test_only_the_gated_candidate_is_marked_gated():
    assert [s.key for s in bakeoff.CANDIDATES if s.gated] == ["embeddinggemma"]
