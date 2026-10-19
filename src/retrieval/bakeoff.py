"""The embedder and header bake-off: candidate specs, text recipes, and the adoption decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from retrieval.recipe import RecipeError

# Bumped when the meaning of a recipe string changes, so old vectors become pending work rather than silently comparable
# to new ones.
RECIPE_VERSION = "v2"

# The production configuration, for reference and as the incumbent every candidate is measured against.
INCUMBENT_KEY = "bge-small"


class BakeoffError(RecipeError):
    """A candidate or recipe that must not be run. Raised in preflight, never mid-run."""


# --- candidate models ---------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """One embedder candidate."""

    key: str
    name: str
    dimensions: int
    max_seq: int
    # What we actually tokenise to, bounded by the model's own limit.
    working_seq: int = 512
    query_instruction: str = ""
    licence: str = ""
    approx_params_m: int = 0
    revision: str = ""
    trust_remote_code: bool = False
    # Config values forced before the weights load.
    config_overrides: dict[str, Any] = field(default_factory=dict)
    # Attention implementation to request.
    attn_implementation: str = ""
    # Weights behind a click-through licence.
    gated: bool = False
    # Non-empty when this candidate is known not to work on this stack, with the reason.
    unsupported: str = ""
    note: str = ""

    @property
    def effective_seq(self) -> int:
        """The sequence length this run will really use: our working length, capped by the model's."""
        return min(self.working_seq, self.max_seq)

    @property
    def vram_estimate_mb(self) -> int:
        """Rough working-set estimate in fp16: weights, plus activations at the length we actually use."""
        weights = self.approx_params_m * 2  # fp16: 2 bytes a parameter, MB because params are in millions
        working = int(weights * 1.5)  # activations, gradients-free inference, modest batches
        sequence = max(64, self.effective_seq // 4)  # a little more for a longer truncation length
        return int(weights + working + sequence + 512)

    def fits_in(self, vram_mb: int) -> bool:
        return self.vram_estimate_mb <= vram_mb


# Dimensions are expectations verified at smoke time.
CANDIDATES: tuple[ModelSpec, ...] = (
    ModelSpec(
        key=INCUMBENT_KEY,
        name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        dimensions=384,
        max_seq=512,
        query_instruction="Represent this sentence for searching relevant passages: ",
        licence="MIT",
        approx_params_m=33,
        note="the incumbent: what production runs today, and the baseline every candidate must beat",
    ),
    ModelSpec(
        key="bge-base",
        name="BAAI/bge-base-en-v1.5",
        dimensions=768,
        max_seq=512,
        query_instruction="Represent this sentence for searching relevant passages: ",
        licence="MIT",
        approx_params_m=109,
        note="the same family one size up: the cheapest way to find out whether capacity is the limit",
    ),
    ModelSpec(
        key="gte-modernbert",
        working_seq=512,
        name="Alibaba-NLP/gte-modernbert-base",
        dimensions=768,
        max_seq=8192,
        licence="Apache-2.0",
        approx_params_m=149,
        note="a modern encoder with a long context; no query instruction by design",
    ),
    ModelSpec(
        key="arctic-m-v2",
        name="Snowflake/snowflake-arctic-embed-m-v2.0",
        working_seq=512,
        dimensions=768,
        max_seq=8192,
        query_instruction="query: ",
        licence="Apache-2.0",
        approx_params_m=305,
        trust_remote_code=True,
        # Its custom modeling code hard-requires xformers when this flag is true, and xformers has no maintained wheel
        # for Windows + torch 2.11+cu128.
        config_overrides={"use_memory_efficient_attention": False, "unpad_inputs": False},
        attn_implementation="sdpa",
        # MEASURED INCOMPATIBILITY, three attempts, each failing further in: 1. as published -> "AssertionError: please
        # install xformers"; its custom code hard-requires xformers when use_memory_efficient_attention is true, and
        # there is no maintained xformers wheel for Windows + torch 2.11+cu128; 2. flag passed as a model kwarg ->
        # "TypeError: GteModel.__init__ got an unexpected keyword argument"; it is a config field, not a constructor
        # argument; 3. flag passed as a config kwarg -> loads, then "CUDA error: device-side assert triggered" at
        # inference.
        unsupported="needs xformers (no Windows wheel for torch 2.11+cu128); overriding the flag makes it "
        "load and then fail with a CUDA device-side assert at inference",
        note="strong on retrieval benchmarks, but it does not work on this stack: see `unsupported`",
    ),
    ModelSpec(
        key="qwen3-0.6b",
        working_seq=512,
        name="Qwen/Qwen3-Embedding-0.6B",
        dimensions=1024,
        max_seq=8192,
        query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        licence="Apache-2.0",
        approx_params_m=595,
        note="the largest candidate that fits 8 GB comfortably; slowest by a wide margin",
    ),
    ModelSpec(
        key="embeddinggemma",
        working_seq=512,
        name="google/embeddinggemma-300m",
        dimensions=768,
        max_seq=2048,
        query_instruction="task: search result | query: ",
        licence="Gemma Terms of Use (click-through, gated weights)",
        approx_params_m=308,
        gated=True,
        note="GATED: accept the licence on the hub and pre-download the weights, then enable explicitly",
    ),
)

BY_KEY: dict[str, ModelSpec] = {spec.key: spec for spec in CANDIDATES}


def default_candidates(vram_mb: int = 8192) -> list[ModelSpec]:
    """The candidates an unattended run may use: not gated, known to work, and inside the card."""
    return [s for s in CANDIDATES if not s.gated and not s.unsupported and s.fits_in(vram_mb)]


def resolve_candidates(keys: Sequence[str] | None, vram_mb: int = 8192) -> list[ModelSpec]:
    """Turn `--model` arguments into specs, refusing anything that cannot honestly be run."""
    if not keys:
        return default_candidates(vram_mb)
    chosen: list[ModelSpec] = []
    for key in keys:
        spec = BY_KEY.get(key)
        if spec is None:
            raise BakeoffError(f"unknown model {key!r}; known keys are {', '.join(sorted(BY_KEY))}")
        if not spec.fits_in(vram_mb):
            raise BakeoffError(
                f"{key} needs about {spec.vram_estimate_mb} MB of VRAM and only {vram_mb} MB is available"
            )
        if spec.unsupported:
            # Asked for explicitly, so allowed -- but the reason is stated rather than rediscovered.
            print(f"  ! {key} is known not to work on this stack: {spec.unsupported}")
        chosen.append(spec)
    return chosen


# --- header recipes: MOVED to retrieval.recipe --------------------------------------------------------- The renderer
# now lives in `retrieval.recipe`, because it defines the production embedding recipe and `gpu/embedder.py` and the
# loader need it without importing a bake-off (E1 section 4.5).

from retrieval.recipe import (  # noqa: E402,F401
    HEADER_RECIPES,
    INCUMBENT_RECIPE,
    MEAN_SENTENCE_TOKENS,
    RECIPES_BY_KEY,
    HeaderRecipe,
    resolve_recipes,
)


def recipe_id(spec: ModelSpec, recipe: HeaderRecipe, version: str = RECIPE_VERSION) -> str:
    """The string stored against a vector so two incomparable vectors can never look identical."""
    return f"{spec.name}|{recipe.key}|{version}"


def shard_stem(spec: ModelSpec, recipe: HeaderRecipe) -> str:
    """A filesystem-safe stem for this combination's vector shards and manifest."""
    return f"{spec.key}__{recipe.key}"


# --- the run plan -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One (model, header) combination to embed and screen."""

    spec: ModelSpec
    recipe: HeaderRecipe

    @property
    def stem(self) -> str:
        return shard_stem(self.spec, self.recipe)

    @property
    def label(self) -> str:
        return f"{self.spec.key} + {self.recipe.key}"


def plan(
    models: Sequence[ModelSpec],
    recipes: Sequence[HeaderRecipe],
    *,
    incumbent_only_headers: bool = True,
) -> list[Trial]:
    trials: list[Trial] = []
    seen: set[str] = set()

    def add(spec: ModelSpec, recipe: HeaderRecipe) -> None:
        trial = Trial(spec, recipe)
        if trial.stem not in seen:
            seen.add(trial.stem)
            trials.append(trial)

    incumbent = next((s for s in models if s.key == INCUMBENT_KEY), None)
    if not incumbent_only_headers:
        for spec in models:
            for recipe in recipes:
                add(spec, recipe)
        return trials

    # Stage 1: every header on the incumbent, so the header question is answered cheaply.
    header_host = incumbent or models[0]
    for recipe in recipes:
        add(header_host, recipe)
    # Stage 2: every other model at the incumbent recipe.
    for spec in models:
        if spec.key != header_host.key:
            add(spec, INCUMBENT_RECIPE)
    return trials


def stage_two(models: Sequence[ModelSpec], winner: HeaderRecipe, host_key: str = INCUMBENT_KEY) -> list[Trial]:
    """The model candidates at the header that won stage one."""
    return [Trial(spec, winner) for spec in models if spec.key != host_key]


def estimate_minutes(trials: Sequence[Trial], chunks: int, incumbent_chunks_per_second: float = 190.0) -> float:
    """How long the embedding will take, scaled from the incumbent's measured throughput."""
    total = 0.0
    incumbent = BY_KEY[INCUMBENT_KEY]
    for trial in trials:
        ratio = max(1.0, trial.spec.approx_params_m / max(1, incumbent.approx_params_m))
        rate = incumbent_chunks_per_second / ratio
        total += chunks / max(1.0, rate) / 60
    return round(total, 1)


# --- the decision -------------------------------------------------------------------------------------


@dataclass
class TrialResult:
    """What one trial scored. `per_category` is why a mean is not enough."""

    label: str
    stem: str
    model_key: str
    recipe_key: str
    recall_at_100: float | None = None
    recall_at_10: float | None = None
    mrr_at_10: float | None = None
    per_category: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    chunks: int = 0
    dimensions: int = 0
    header_overhead_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"{self.label:28s} R@100 {_fmt(self.recall_at_100)}  R@10 {_fmt(self.recall_at_10)}  "
            f"MRR {_fmt(self.mrr_at_10)}  dims {self.dimensions}  "
            f"header {self.header_overhead_ratio:.0%} of the text"
        )


def _fmt(value: float | None) -> str:
    return "  n/a " if value is None else f"{value:.4f}"


# Categories where a loss matters more than the mean.
PROTECTED_CATEGORIES = ("multi_hop", "multi_speech", "named_speaker", "party_attribution", "temporal", "comparison")


def regressions(base: TrialResult, other: TrialResult, tolerance: float = 0.02) -> list[str]:
    """Protected categories where `other` is materially worse than `base`."""
    out = []
    for category in PROTECTED_CATEGORIES:
        before, after = base.per_category.get(category), other.per_category.get(category)
        if before is None or after is None:
            continue
        if after < before - tolerance:
            out.append(f"{category} {before:.3f} -> {after:.3f}")
    return out
