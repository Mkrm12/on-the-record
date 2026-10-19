"""The bake-off worker: embed candidate models to disk, screen them by exact GPU search, unattended."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from retrieval.bakeoff import (  # noqa: E402
    BY_KEY,
    INCUMBENT_KEY,
    INCUMBENT_RECIPE,
    RECIPES_BY_KEY,
    BakeoffError,
    RecipeError,
    HeaderRecipe,
    ModelSpec,
    Trial,
    recipe_id,
)

OUT_DIR = REPO / "data" / "bakeoff"
DEFAULT_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/hansard_v2")

# Rows per shard.
DEFAULT_SHARD_ROWS = 5_000
DEFAULT_BATCH = 128
SMOKE_ROWS = 64

# The screening corpus: positives plus this many seeded distractors from the dev windows. 120k chunks is about ten
# minutes on the incumbent, so a nine-trial bake-off is an overnight run instead of a week.
DEFAULT_SCREEN_CHUNKS = 120_000
SCREEN_CORPUS_PATH = OUT_DIR / "screen_corpus.json"
SCREEN_SEED = 20260925

# W3 is the held-out window and never appears in a screening corpus, not even as a distractor.
HELDOUT_WINDOW = "W3"

# Leave this much VRAM unused when sizing a search tile.
VRAM_HEADROOM_MB = 1_536


class WorkerError(RuntimeError):
    """Something that must stop the run. The message always says what to do next."""


def say(message: str) -> None:
    print(message, flush=True)


# --- the corpus ---------------------------------------------------------------------------------------

# Every field any header recipe can use, plus the body.
ROWS_SQL = """
SELECT c.id,
       c.speech_id,
       c.body,
       c.debate_date::text AS date,
       c.house,
       d.heading,
       s.speaker_name,
       s.speaker_party AS party
  FROM chunk c
  LEFT JOIN speech s ON s.id = c.speech_id
  LEFT JOIN debate d ON d.id = c.debate_id
 WHERE NOT c.is_boilerplate
   AND c.body IS NOT NULL
 ORDER BY c.id
"""


def corpus_fingerprint(ids: Sequence[int]) -> str:
    """A digest of the exact chunk list, in order."""
    digest = hashlib.sha256()
    digest.update(str(len(ids)).encode())
    for chunk_id in ids:
        digest.update(b",")
        digest.update(str(chunk_id).encode())
    return digest.hexdigest()[:16]


@dataclass
class Corpus:
    ids: list[int]
    speech_ids: list[str]
    rows: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def fingerprint(self) -> str:
        return corpus_fingerprint(self.ids)


def count_embeddable(dsn: str) -> int:
    """How many chunks a full embed would have to cover. Measured, because the project's own notes said 90,040 -- which
    is W0 alone, from before the other four windows were loaded.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunk WHERE NOT is_boilerplate AND body IS NOT NULL")
            return int(cur.fetchone()[0])


def load_corpus(dsn: str, limit: int | None = None) -> Corpus:
    """Every embeddable chunk with the metadata a header needs. Read-only, by construction."""
    sql = ROWS_SQL + (f" LIMIT {int(limit)}" if limit else "")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.read_only = True
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    if not rows:
        raise WorkerError(
            f"no embeddable chunk found in {dsn.rsplit('/', 1)[-1]}. Is the corpus loaded and chunked?"
        )
    return Corpus(ids=[r["id"] for r in rows], speech_ids=[r["speech_id"] for r in rows], rows=rows)


def build_screen_corpus(
    dsn: str,
    *,
    target: int = DEFAULT_SCREEN_CHUNKS,
    seed: int = SCREEN_SEED,
    split: str = "dev",
    path: Path = SCREEN_CORPUS_PATH,
) -> dict[str, Any]:
    """Choose the screening corpus once, record it, and never choose it again."""
    import random

    queries = gold_queries(split)
    wanted = sorted({r.speech_id for q in queries for r in q.relevant})
    if not wanted:
        raise WorkerError(f"no relevant speech in the {split} split; run build_gold.py --judge first")

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM chunk WHERE NOT is_boilerplate AND body IS NOT NULL "
                "AND speech_id = ANY(%s) ORDER BY id",
                (wanted,),
            )
            positives = [r[0] for r in cur.fetchall()]
            cur.execute(
                "SELECT id FROM chunk WHERE NOT is_boilerplate AND body IS NOT NULL "
                "AND COALESCE(window_id,'') <> %s AND NOT (speech_id = ANY(%s)) ORDER BY id",
                (HELDOUT_WINDOW, wanted),
            )
            pool = [r[0] for r in cur.fetchall()]

    room = max(0, target - len(positives))
    rng = random.Random(seed)
    distractors = sorted(rng.sample(pool, min(room, len(pool))))
    ids = sorted(set(positives) | set(distractors))
    payload = {
        "seed": seed,
        "split": split,
        "target": target,
        "positives": len(positives),
        "relevant_speeches": len(wanted),
        "distractors": len(distractors),
        "distractor_pool": len(pool),
        "excluded_window": HELDOUT_WINDOW,
        "chunks": len(ids),
        "ids": ids,
        "note": ("absolute recall on this corpus is optimistic: it holds fewer distractors than production. "
                 "The comparison between candidates is fair because they all face this identical set, and "
                 "the winner's real number comes from the full embed in the adoption step."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    tmp.replace(path)
    say(f"screening corpus: {len(ids):,} chunks = {len(positives):,} positives "
        f"({len(wanted)} relevant speeches) + {len(distractors):,} seeded distractors, excluding {HELDOUT_WINDOW}")  # fmt: skip
    return {k: v for k, v in payload.items() if k != "ids"}


def load_screen_corpus(dsn: str, path: Path = SCREEN_CORPUS_PATH) -> Corpus:
    """The recorded screening corpus, in its recorded order. Refuses to invent one."""
    if not path.is_file():
        raise WorkerError(
            f"no screening corpus at {path.name}. Run `--build-screen-corpus` first: choosing it on the fly "
            "would let two trials face different corpora and make their recalls incomparable."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload["ids"]
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.read_only = True
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(ROWS_SQL.replace("WHERE NOT c.is_boilerplate", "WHERE c.id = ANY(%(ids)s) AND NOT c.is_boilerplate"),
                        {"ids": ids})  # fmt: skip
            rows = cur.fetchall()
    if len(rows) != len(ids):
        raise WorkerError(
            f"the screening corpus lists {len(ids):,} chunks but only {len(rows):,} are in the database. "
            "The corpus changed; rebuild it with --build-screen-corpus and re-embed every trial."
        )
    return Corpus(ids=[r["id"] for r in rows], speech_ids=[r["speech_id"] for r in rows], rows=rows)


# --- the model, and letting go of it ------------------------------------------------------------------


@dataclass
class Loaded:
    spec: ModelSpec
    model: Any
    resolved_revision: str = ""
    real_dimensions: int = 0


def vram_mb() -> tuple[float, float]:
    """(used, total) in MB, or (0, 0) with no card."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024**2, total / 1024**2
    except Exception:  # noqa: BLE001 - absence of a card is reported, not raised
        return 0.0, 0.0


def _embedding_dimension(model: Any) -> int:
    """The model's output width, across sentence-transformers renames."""
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # noqa: BLE001 - fall through to the probe
                continue
            if value:
                return int(value)
    probe = model.encode(["dimension probe"], convert_to_numpy=True, normalize_embeddings=True)
    return int(np.asarray(probe).shape[-1])


def raise_missing_dependency(spec: ModelSpec, exc: BaseException) -> None:
    """Turn "please install xformers" into something that says what to do instead of installing it."""
    raise WorkerError(
        f"{spec.key} could not load: {type(exc).__name__}: {exc}. This is usually a trust_remote_code "
        f"model demanding an optional dependency. Either add a config override to its ModelSpec in "
        f"src/retrieval/bakeoff.py to turn that code path off (as arctic-m-v2 does for xformers), or drop "
        f"the candidate with --skip {spec.key}. Do NOT install the dependency into the ROOT project: that "
        f"pulls a second, CPU-only torch into an environment that must never import torch."
    ) from exc


def hf_token() -> str:
    """The Hugging Face token, from the environment or.env. Returned, never printed."""
    # Conventional names first, then the obvious human spellings.
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
                 "HUGGINGFACE_TOKEN", "HUGGINGFACETOKEN"):  # fmt: skip
        if value := os.environ.get(name, "").strip():
            return value
    # .env is read directly as a fallback: the root settings object only knows the names it declares, and the token
    # lives in a file the GPU project does not otherwise load.
    env_file = REPO / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip().lower().replace("_", "") in {
                "hftoken", "huggingfacetoken", "huggingfacehubtoken", "huggingfacehubapitoken"
            } and value.strip():
                return value.strip()
    try:
        from app.config import settings

        return (settings.hf_token or "").strip()
    except Exception:  # noqa: BLE001 - absence of a token is a fact, not a failure
        return ""


def load_model(spec: ModelSpec) -> Loaded:
    """Load a candidate, verify its real output width, and record the revision actually used."""
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}
    if spec.revision:
        kwargs["revision"] = spec.revision
    token = hf_token()
    if spec.gated:
        if not token:
            raise WorkerError(
                f"{spec.key} is gated: its weights are only served to an account that has accepted the "
                f"licence ({spec.licence}). Accept it on https://huggingface.co/{spec.name} and put a "
                "read-scoped token in .env as HF_TOKEN, then re-run. Every other candidate is public."
            )
        kwargs["token"] = token
    elif token:
        # A token on a public model is harmless and lifts anonymous rate limits.
        kwargs["token"] = token

    # Config overrides and the attention implementation both travel as model_kwargs.
    model_kwargs: dict[str, Any] = {"torch_dtype": "float16"}
    if spec.attn_implementation:
        model_kwargs["attn_implementation"] = spec.attn_implementation
    config_kwargs: dict[str, Any] = dict(spec.config_overrides)
    if config_kwargs:
        kwargs["config_kwargs"] = config_kwargs
    extra = ", ".join(f"{k}={v}" for k, v in sorted(spec.config_overrides.items()))
    say(f"    loading {spec.name} (trust_remote_code={spec.trust_remote_code}"
        + (f", {extra}" if extra else "") + ")")  # fmt: skip
    try:
        model = SentenceTransformer(spec.name, device="cuda", model_kwargs=model_kwargs, **kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised below with something actionable
        message = str(exc)
        if spec.gated or "gated" in message.lower() or "403" in message:
            raise WorkerError(
                f"{spec.key} could not be downloaded: {type(exc).__name__}: {message[:200]}. For a gated "
                f"model, accept the licence at https://huggingface.co/{spec.name} with the SAME account "
                "the token belongs to -- a valid token does not bypass an unaccepted licence."
            ) from exc
        if not isinstance(exc, (AssertionError, ImportError)):
            raise
        raise_missing_dependency(spec, exc)
    except (AssertionError, ImportError) as exc:
        # The message a missing optional dependency gives is usually just its name, which is not enough to act on at
        # 3am.
        raise_missing_dependency(spec, exc)
    # Truncation is a correctness matter, not a speed one: an unbounded long-context model allocates for its longest row
    # and OOMs the card hours into a run.
    model.max_seq_length = spec.effective_seq
    real = int(_embedding_dimension(model))
    if real != spec.dimensions:
        unload(Loaded(spec, model))
        raise WorkerError(
            f"{spec.key}: the weights produce {real} dimensions but the spec declares {spec.dimensions}. "
            f"Fix `dimensions` in src/retrieval/bakeoff.py before embedding anything: vectors written at "
            f"the wrong width cannot be compared with anything."
        )
    revision = ""
    try:  # best effort: useful for reproducibility, never worth failing a run over
        revision = getattr(getattr(model, "_first_module")(), "auto_model").config._name_or_path or ""
    except Exception:  # noqa: BLE001
        revision = spec.revision
    return Loaded(spec=spec, model=model, resolved_revision=revision or spec.revision, real_dimensions=real)


def unload(loaded: Loaded | None) -> float:
    """Drop the model and give the VRAM back. Returns the megabytes released."""
    before, _ = vram_mb()
    if loaded is not None:
        try:
            loaded.model.to("cpu")
        except Exception:  # noqa: BLE001 - already gone, or never on the card
            pass
        loaded.model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    after, _ = vram_mb()
    return round(before - after, 1)


def encode(loaded: Loaded, texts: list[str], batch: int) -> np.ndarray:
    """Encode to unit-norm float32, then validate. Normalising here means cosine is a dot product later."""
    vectors = loaded.model.encode(
        texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != loaded.real_dimensions:
        raise WorkerError(f"{loaded.spec.key}: encoder returned shape {vectors.shape}, expected (*, {loaded.real_dimensions})")
    if not np.isfinite(vectors).all():
        raise WorkerError(f"{loaded.spec.key}: encoder produced non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        raise WorkerError(
            f"{loaded.spec.key}: vectors are not unit-norm (min {norms.min():.4f}, max {norms.max():.4f}); "
            "cosine screening assumes normalised vectors"
        )
    return vectors


# --- the manifest: what makes this resumable ----------------------------------------------------------


@dataclass
class Manifest:
    """One trial's state on disk. Written after every shard, so a crash costs one shard."""

    path: Path
    trial: str = ""
    model: str = ""
    model_key: str = ""
    recipe_key: str = ""
    recipe: str = ""
    dimensions: int = 0
    resolved_revision: str = ""
    effective_seq: int = 0
    shard_rows: int = DEFAULT_SHARD_ROWS
    total_rows: int = 0
    # Which corpus these shards describe.
    corpus_fingerprint: str = ""
    shards: dict[str, int] = field(default_factory=dict)  # filename -> rows
    seconds: float = 0.0
    complete: bool = False
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data.pop("path", None)
                return cls(path=path, **data)
            except (OSError, ValueError, TypeError):
                pass
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(self).items() if k != "path"}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        tmp.replace(self.path)

    @property
    def rows_done(self) -> int:
        return sum(self.shards.values())

    def shard_name(self, index: int) -> str:
        return f"{self.trial}.{index:04d}.npy"


def shard_plan(total: int, shard_rows: int) -> list[tuple[int, int, int]]:
    """(index, start, end) for every shard. Deterministic, so a resume lines up exactly."""
    return [(i, start, min(start + shard_rows, total)) for i, start in enumerate(range(0, total, shard_rows))]


# --- embedding one trial ------------------------------------------------------------------------------


def embed_trial(
    trial: Trial,
    corpus: Corpus,
    *,
    out_dir: Path = OUT_DIR,
    shard_rows: int = DEFAULT_SHARD_ROWS,
    batch: int = DEFAULT_BATCH,
    loaded: Loaded | None = None,
) -> Manifest:
    """Embed the whole corpus for one (model, header) trial, resumably."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(out_dir / f"{trial.stem}.manifest.json")
    manifest.trial = trial.stem
    manifest.model, manifest.model_key = trial.spec.name, trial.spec.key
    manifest.recipe_key, manifest.recipe = trial.recipe.key, recipe_id(trial.spec, trial.recipe)
    manifest.dimensions = trial.spec.dimensions
    manifest.effective_seq = trial.spec.effective_seq

    # A shard is only reusable if it was built from THIS corpus with THIS shard size.
    fingerprint = corpus.fingerprint
    stale = (
        (manifest.corpus_fingerprint and manifest.corpus_fingerprint != fingerprint)
        or (manifest.shards and manifest.shard_rows != shard_rows)
        or (manifest.total_rows and manifest.total_rows != len(corpus))
    )
    if stale:
        say(f"  {trial.label}: the cached shards were built from a different corpus "
            f"({manifest.total_rows:,} rows, fingerprint {manifest.corpus_fingerprint or 'unknown'}) than this "
            f"one ({len(corpus):,} rows, {fingerprint}). Discarding them: reusing them would score vectors "
            "computed for different chunks.")  # fmt: skip
        for name in list(manifest.shards):
            (out_dir / name).unlink(missing_ok=True)
        manifest.shards = {}
        manifest.seconds = 0.0
        manifest.complete = False
    manifest.corpus_fingerprint = fingerprint
    manifest.shard_rows = shard_rows
    manifest.total_rows = len(corpus)

    plan = shard_plan(len(corpus), shard_rows)
    todo = [(i, a, b) for i, a, b in plan if manifest.shard_name(i) not in manifest.shards
            or not (out_dir / manifest.shard_name(i)).is_file()]  # fmt: skip
    if not todo:
        manifest.complete = True
        manifest.save()
        say(f"  {trial.label}: already complete ({manifest.rows_done:,} rows)")
        return manifest

    say(f"  {trial.label}: {len(todo)} of {len(plan)} shards to do ({manifest.rows_done:,}/{len(corpus):,} rows done)")
    mine = loaded is None
    if mine:
        loaded = load_model(trial.spec)
    manifest.resolved_revision = loaded.resolved_revision
    started = time.monotonic()
    try:
        for index, start, end in todo:
            texts = [trial.recipe.render(row) for row in corpus.rows[start:end]]
            shard_started = time.monotonic()
            vectors = encode(loaded, texts, batch)
            shard_seconds = max(1e-6, time.monotonic() - shard_started)
            name = manifest.shard_name(index)
            # Through an open handle, not a path: np.save APPENDS.npy to a path that lacks it, so a ".npy.tmp" target
            # becomes ".npy.tmp.npy" and the atomic replace below fails on a file that was never written.
            tmp = out_dir / (name + ".tmp")
            with tmp.open("wb") as handle:
                np.save(handle, vectors.astype(np.float16))
            tmp.replace(out_dir / name)
            manifest.shards[name] = end - start
            manifest.seconds = round(manifest.seconds + (time.monotonic() - started), 1)
            started = time.monotonic()
            manifest.save()  # committed before the next shard begins
            # This shard's own rate, and the run's average.
            shard_rate = (end - start) / shard_seconds
            overall = manifest.rows_done / manifest.seconds if manifest.seconds else 0.0
            remaining = (len(corpus) - manifest.rows_done) / shard_rate / 60 if shard_rate else 0.0
            say(f"    shard {index + 1}/{len(plan)}  {manifest.rows_done:,}/{len(corpus):,} rows"
                f"  {shard_rate:,.0f}/s now, {overall:,.0f}/s avg"
                f"  ~{remaining:.0f} min left  vram {vram_mb()[0]:.0f}MB")  # fmt: skip
    finally:
        if mine:
            freed = unload(loaded)
            say(f"    released {freed:.0f} MB of VRAM")
    manifest.complete = manifest.rows_done >= len(corpus)
    manifest.save()
    return manifest


def load_vectors(manifest: Manifest, out_dir: Path = OUT_DIR, corpus: Corpus | None = None) -> np.ndarray:
    """Every shard, in order, as one array. Refuses a manifest whose shards do not add up."""
    if corpus is not None and manifest.corpus_fingerprint and manifest.corpus_fingerprint != corpus.fingerprint:
        raise WorkerError(
            f"{manifest.trial}: these shards were built from corpus {manifest.corpus_fingerprint} but the "
            f"corpus is now {corpus.fingerprint}. Re-embed this trial rather than scoring a mismatch."
        )
    plan = shard_plan(manifest.total_rows, manifest.shard_rows)
    parts = []
    for index, _start, _end in plan:
        path = out_dir / manifest.shard_name(index)
        if not path.is_file():
            raise WorkerError(f"{manifest.trial}: shard {path.name} is missing; re-run --embed to finish it")
        parts.append(np.load(path))
    matrix = np.concatenate(parts, axis=0)
    if matrix.shape[0] != manifest.total_rows:
        raise WorkerError(
            f"{manifest.trial}: shards hold {matrix.shape[0]} rows but the manifest expects "
            f"{manifest.total_rows}. Delete the shards and re-embed rather than screening a partial set."
        )
    return matrix


# --- exact search on the card -------------------------------------------------------------------------


def tile_rows(dimensions: int, free_mb: float) -> int:
    """How many document vectors fit in a search tile, leaving headroom."""
    usable = max(64.0, free_mb - VRAM_HEADROOM_MB)
    per_row_mb = dimensions * 2 / 1024**2
    return max(1_000, int(usable / max(per_row_mb, 1e-9) / 4))


def search_exact(docs: np.ndarray, queries: np.ndarray, top_k: int) -> np.ndarray:
    """Exact cosine top-k, on the GPU when there is one, in tiles. Returns document indices."""
    if docs.shape[1] != queries.shape[1]:
        raise WorkerError(f"dimension mismatch: documents are {docs.shape[1]}, queries are {queries.shape[1]}")
    top_k = min(top_k, docs.shape[0])
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is None or not torch.cuda.is_available():
        scores = queries.astype(np.float32) @ docs.astype(np.float32).T
        return np.argsort(-scores, axis=1)[:, :top_k]

    device = "cuda"
    q = torch.from_numpy(np.ascontiguousarray(queries)).to(device=device, dtype=torch.float16)
    free_mb = (torch.cuda.mem_get_info()[0]) / 1024**2
    step = tile_rows(docs.shape[1], free_mb)
    best_scores: torch.Tensor | None = None
    best_index: torch.Tensor | None = None
    for start in range(0, docs.shape[0], step):
        block = torch.from_numpy(np.ascontiguousarray(docs[start : start + step])).to(device=device, dtype=torch.float16)
        scores = q @ block.T  # (queries, block)
        k = min(top_k, scores.shape[1])
        values, indices = torch.topk(scores.float(), k=k, dim=1)
        indices = indices + start
        if best_scores is None:
            best_scores, best_index = values, indices
        else:
            best_scores = torch.cat([best_scores, values], dim=1)
            best_index = torch.cat([best_index, indices], dim=1)
            values, keep = torch.topk(best_scores, k=min(top_k, best_scores.shape[1]), dim=1)
            best_scores = values
            best_index = torch.gather(best_index, 1, keep)
        del block, scores
    assert best_index is not None
    out = best_index.cpu().numpy()
    del q, best_scores, best_index
    torch.cuda.empty_cache()
    return out


# --- screening: dense-only exact recall ---------------------------------------------------------------


def gold_queries(split: str = "dev") -> list[Any]:
    """The scorable gold queries for a split, with pooled binarised relevance."""
    from goldset.schema import load as load_gold
    from retrieval import tuning

    items = [i for i in load_gold() if i.split == split]
    return [tuning.GoldQuery.of(i) for i in tuning.scorable(items)]


def screen_trial(
    trial: Trial,
    corpus: Corpus,
    queries: list[Any],
    *,
    out_dir: Path = OUT_DIR,
    top_k: int = 100,
    batch: int = DEFAULT_BATCH,
    loaded: Loaded | None = None,
) -> dict[str, Any]:
    """Exact dense recall for one trial: embed the questions, search the shards, score at speech level."""
    from retrieval.metrics import dedupe_speeches, recall_at_k, reciprocal_rank

    manifest = Manifest.load(out_dir / f"{trial.stem}.manifest.json")
    if not manifest.complete:
        raise WorkerError(f"{trial.stem}: not fully embedded ({manifest.rows_done}/{manifest.total_rows}); run --embed")
    docs = load_vectors(manifest, out_dir, corpus)
    if docs.shape[0] != len(corpus):
        raise WorkerError(
            f"{trial.stem}: {docs.shape[0]} vectors for {len(corpus)} chunks. The corpus changed since "
            "embedding; delete the shards for this trial and re-embed rather than scoring a mismatch."
        )

    mine = loaded is None
    if mine:
        loaded = load_model(trial.spec)
    try:
        prefix = trial.spec.query_instruction or ""
        question_vectors = encode(loaded, [prefix + q.question for q in queries], batch)
    finally:
        if mine:
            unload(loaded)

    ranked = search_exact(docs, question_vectors, top_k)
    speech_of = corpus.speech_ids
    per_category: dict[str, list[float]] = {}
    recalls_100: list[float] = []
    recalls_10: list[float] = []
    rrs: list[float] = []
    for row, query in zip(ranked, queries):
        speeches = dedupe_speeches(speech_of[i] for i in row)
        relevant = {r.speech_id for r in query.relevant}
        if not relevant:
            continue
        r100 = recall_at_k(relevant, speeches, top_k)
        r10 = recall_at_k(relevant, speeches, 10)
        rr = reciprocal_rank(relevant, speeches, 10)
        recalls_100.append(r100)
        recalls_10.append(r10)
        rrs.append(rr)
        per_category.setdefault(query.category, []).append(r100)

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "trial": trial.stem,
        "label": trial.label,
        "model_key": trial.spec.key,
        "recipe_key": trial.recipe.key,
        "dimensions": manifest.dimensions,
        "resolved_revision": manifest.resolved_revision,
        "queries": len(recalls_100),
        "chunks": docs.shape[0],
        "recall_at_100": mean(recalls_100),
        "recall_at_10": mean(recalls_10),
        "mrr_at_10": mean(rrs),
        "per_category": {k: mean(v) for k, v in sorted(per_category.items())},
        "header_overhead_ratio": round(trial.recipe.overhead_ratio(), 3),
        "embed_minutes": round(manifest.seconds / 60, 1),
    }


def cmd_screen(args) -> int:
    """Score every embedded trial, print a table, and write the report the orchestrator reads."""
    from retrieval.bakeoff import TrialResult, regressions

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems = args.trial or [p.name.replace(".manifest.json", "") for p in sorted(OUT_DIR.glob("*.manifest.json"))]
    if not stems:
        say("nothing embedded yet: run --embed first")
        return 1
    corpus = load_corpus(args.dsn, limit=args.limit) if args.full_corpus else load_screen_corpus(args.dsn)
    queries = gold_queries(args.split)
    if not queries:
        raise WorkerError(f"no scorable {args.split} query has pooled relevance; run build_gold.py --judge first")
    say(f"screening {len(stems)} trial(s) against {len(queries)} {args.split} queries over {len(corpus):,} chunks")
    say("(dense only, exact brute-force search, no index and no reranker)\n")

    results = []
    for stem in stems:
        trial = parse_trial(stem)
        try:
            results.append(screen_trial(trial, corpus, queries, top_k=args.top_k, batch=args.batch))
            say("  " + results[-1]["label"].ljust(28)
                + f" R@100 {results[-1]['recall_at_100']}  R@10 {results[-1]['recall_at_10']}"
                + f"  MRR {results[-1]['mrr_at_10']}  dims {results[-1]['dimensions']}")  # fmt: skip
        except WorkerError as exc:
            say(f"  {trial.label:28s} SKIPPED: {exc}")

    if not results:
        say("\nnothing could be screened")
        return 1
    incumbent = next((r for r in results if r["model_key"] == "bge-small" and r["recipe_key"] == "sentence"), None)
    ranked = sorted(results, key=lambda r: (r["recall_at_100"] or 0), reverse=True)
    say("\nbest first:")
    for r in ranked:
        flag = ""
        if incumbent and r is not incumbent:
            base = TrialResult(label="incumbent", stem="i", model_key="bge-small", recipe_key="sentence",
                               per_category=incumbent["per_category"])  # fmt: skip
            other = TrialResult(label=r["label"], stem=r["trial"], model_key=r["model_key"],
                                recipe_key=r["recipe_key"], per_category=r["per_category"])  # fmt: skip
            hits = regressions(base, other)
            flag = ("  REGRESSES " + "; ".join(hits)) if hits else ""
        delta = ""
        if incumbent and r is not incumbent and r["recall_at_100"] and incumbent["recall_at_100"]:
            delta = f"  ({r['recall_at_100'] - incumbent['recall_at_100']:+.4f} vs incumbent)"
        say(f"  {r['label']:28s} R@100 {r['recall_at_100']}{delta}{flag}")

    report = {
        "split": args.split,
        "top_k": args.top_k,
        "queries": len(queries),
        "chunks": len(corpus),
        "incumbent": incumbent["trial"] if incumbent else None,
        "results": results,
    }
    path = Path(args.json) if args.json else (REPO / "reports" / "session3_bakeoff_screen.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    say(f"\nwrote {path}")
    return 0


# --- the smoke test ----------------------------------------------------------------------------------


def smoke(specs: list[ModelSpec], corpus: Corpus, recipes: list[HeaderRecipe], batch: int,
          corpus_size: int = 0) -> dict[str, Any]:  # fmt: skip
    """Prove every candidate works before the long run: shapes, norms, truncation, VRAM, header text."""
    _used, total = vram_mb()
    report: dict[str, Any] = {"vram_total_mb": round(total, 1), "candidates": {}, "ok": True}
    rows = corpus.rows[:SMOKE_ROWS]
    say(f"smoke test: {len(specs)} candidate(s) x {len(rows)} real chunks, card {total:.0f} MB")

    for spec in specs:
        entry: dict[str, Any] = {"model": spec.name, "declared_dimensions": spec.dimensions}
        loaded = None
        try:
            before, _ = vram_mb()
            loaded = load_model(spec)
            entry["resolved_revision"] = loaded.resolved_revision
            entry["real_dimensions"] = loaded.real_dimensions
            entry["max_seq_length"] = int(getattr(loaded.model, "max_seq_length", 0))
            if entry["max_seq_length"] > spec.effective_seq:
                raise WorkerError(
                    f"{spec.key}: max_seq_length is {entry['max_seq_length']} but must be truncated to "
                    f"{spec.effective_seq}; an unbounded length OOMs the card mid-run"
                )
            per_recipe = {}
            for recipe in recipes:
                texts = [recipe.render(row) for row in rows]
                vectors = encode(loaded, texts, batch)
                tokens = [len(loaded.model.tokenizer.tokenize(t)) for t in texts[:16]]
                body_tokens = [len(loaded.model.tokenizer.tokenize(r["body"] or "")) for r in rows[:16]]
                header = sum(tokens) - sum(body_tokens)
                per_recipe[recipe.key] = {
                    "shape": list(vectors.shape),
                    "mean_tokens": round(sum(tokens) / max(1, len(tokens)), 1),
                    "mean_body_tokens": round(sum(body_tokens) / max(1, len(body_tokens)), 1),
                    "measured_header_share": round(header / max(1, sum(tokens)), 3),
                    "expected_header_share": round(recipe.overhead_ratio(), 3),
                    "sample": texts[0][:160],
                }
            entry["recipes"] = per_recipe

            # Measured, not extrapolated.
            sample = [INCUMBENT_RECIPE.render(row) for row in rows]
            warm = time.monotonic()
            encode(loaded, sample, batch)  # warm the kernels first
            warm_seconds = time.monotonic() - warm
            timed = time.monotonic()
            encode(loaded, sample, batch)
            seconds = max(1e-6, time.monotonic() - timed)
            rate = len(sample) / seconds
            entry["measured_rows_per_second"] = round(rate, 1)
            entry["warmup_seconds"] = round(warm_seconds, 2)
            entry["projected_minutes_per_full_corpus"] = round(corpus_size / rate / 60, 1) if corpus_size else None
            entry["projected_minutes_per_screen_corpus"] = round(DEFAULT_SCREEN_CHUNKS / rate / 60, 1)
            peak, _ = vram_mb()
            entry["vram_peak_mb"] = round(peak, 1)
            entry["vram_headroom_mb"] = round(total - peak, 1)
            if total and peak > total - 512:
                raise WorkerError(f"{spec.key}: peaked at {peak:.0f} MB of {total:.0f} MB; not enough headroom")
            entry["ok"] = True
            say(f"  {spec.key:16s} ok  dims {loaded.real_dimensions}  peak {peak:.0f}MB  "
                f"headroom {total - peak:.0f}MB  {entry['measured_rows_per_second']:,.0f} rows/s  "
                f"-> {entry['projected_minutes_per_screen_corpus']:.0f} min screen"
                + (f" / {entry['projected_minutes_per_full_corpus']:.0f} min full"
                   if entry.get('projected_minutes_per_full_corpus') else ""))  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - every candidate is reported, one failure is not fatal
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["ok"] = False
            say(f"  {spec.key:16s} FAILED  {entry['error'][:140]}")
        finally:
            freed = unload(loaded)
            entry["vram_released_mb"] = freed
            after, _ = vram_mb()
            entry["vram_after_unload_mb"] = round(after, 1)
            # A leak here is the thing that kills the long run, so it is an error, not a note.
            if total and after > before + 512:
                entry["ok"] = False
                entry["error"] = (f"leaked {after - before:.0f} MB after unload; the next candidate would "
                                  "start with less VRAM than this one did")  # fmt: skip
                report["ok"] = False
                say(f"  {spec.key:16s} LEAK    {entry['error']}")
        report["candidates"][spec.key] = entry

    available = sorted(k for k, v in report["candidates"].items() if v.get("ok"))
    unavailable = {k: v.get("error", "unknown") for k, v in report["candidates"].items() if not v.get("ok")}
    report["available"] = available
    report["unavailable"] = unavailable
    incumbent_ok = INCUMBENT_KEY in available
    challengers = [k for k in available if k != INCUMBENT_KEY]

    # `ok` is about whether a bake-off can proceed, not about whether every candidate worked.
    report["ok"] = bool(incumbent_ok and challengers)
    report["incumbent_ok"] = incumbent_ok
    if not incumbent_ok:
        report["blocker"] = (f"the incumbent ({INCUMBENT_KEY}) failed its smoke test, so nothing has a "
                             "baseline to be measured against")  # fmt: skip
    elif not challengers:
        report["blocker"] = "every challenger failed its smoke test, so there is nothing to compare"
    return report


def smoke_isolated(specs: list[ModelSpec], args) -> dict[str, Any]:
    """Smoke-test each candidate in its own subprocess and collect the results."""
    import subprocess

    _used, total = vram_mb()
    report: dict[str, Any] = {"vram_total_mb": round(total, 1), "candidates": {}, "isolated": True}
    say(f"smoke test: {len(specs)} candidate(s), each in its own process, card {total:.0f} MB\n")

    for spec in specs:
        command = [sys.executable, str(Path(__file__).resolve()), "--smoke-one", spec.key,
                   "--dsn", args.dsn, "--batch", str(args.batch)]  # fmt: skip
        if args.limit:
            command += ["--limit", str(args.limit)]
        try:
            proc = subprocess.run(command, cwd=str(Path(__file__).resolve().parent), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=1_800)  # fmt: skip
            payload = None
            for line in (proc.stdout or "").splitlines():
                if line.startswith("@@RESULT@@"):
                    payload = json.loads(line[len("@@RESULT@@"):])
            if payload:
                report["candidates"].update(payload)
                entry = payload[spec.key]
                if entry.get("ok"):
                    say(f"  {spec.key:16s} ok  dims {entry.get('real_dimensions')}  "
                        f"peak {entry.get('vram_peak_mb', 0):.0f}MB  "
                        f"{entry.get('measured_rows_per_second', 0):,.0f} rows/s  "
                        f"-> {entry.get('projected_minutes_per_screen_corpus', 0):.0f} min screen / "
                        f"{entry.get('projected_minutes_per_full_corpus') or 0:.0f} min full")  # fmt: skip
                else:
                    say(f"  {spec.key:16s} FAILED  {str(entry.get('error'))[:150]}")
            else:
                # No result line at all: the child died before it could report, which is exactly the case isolation
                # exists for.
                tail = "\n".join((proc.stdout or "").strip().splitlines()[-3:])
                reason = (f"the candidate's process exited {proc.returncode} without reporting "
                          f"(a hard crash, an OOM or a CUDA abort). Last output: {tail[-300:]}")  # fmt: skip
                report["candidates"][spec.key] = {"model": spec.name, "ok": False, "error": reason}
                say(f"  {spec.key:16s} CRASHED  exit {proc.returncode} (isolated: other candidates unaffected)")
        except subprocess.TimeoutExpired:
            report["candidates"][spec.key] = {
                "model": spec.name, "ok": False,
                "error": "the candidate did not finish its smoke test within 30 minutes",
            }  # fmt: skip
            say(f"  {spec.key:16s} TIMEOUT")

    available = sorted(k for k, v in report["candidates"].items() if v.get("ok"))
    unavailable = {k: v.get("error", "unknown") for k, v in report["candidates"].items() if not v.get("ok")}
    challengers = [k for k in available if k != INCUMBENT_KEY]
    report["available"] = available
    report["unavailable"] = unavailable
    report["incumbent_ok"] = INCUMBENT_KEY in available
    report["ok"] = bool(report["incumbent_ok"] and challengers)
    if not report["incumbent_ok"]:
        report["blocker"] = (f"the incumbent ({INCUMBENT_KEY}) failed its smoke test, so nothing has a "
                             "baseline to be measured against")  # fmt: skip
    elif not challengers:
        report["blocker"] = "every challenger failed its smoke test, so there is nothing to compare"
    return report


# --- the CLI -----------------------------------------------------------------------------------------


def parse_trial(stem: str) -> Trial:
    if "__" not in stem:
        raise BakeoffError(f"trial must look like model__recipe, got {stem!r}")
    model_key, recipe_key = stem.split("__", 1)
    if model_key not in BY_KEY:
        raise BakeoffError(f"unknown model {model_key!r}")
    if recipe_key not in RECIPES_BY_KEY:
        raise BakeoffError(f"unknown recipe {recipe_key!r}")
    return Trial(BY_KEY[model_key], RECIPES_BY_KEY[recipe_key])


def cmd_status(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = sorted(OUT_DIR.glob("*.manifest.json"))
    if not manifests:
        say(f"nothing embedded yet in {OUT_DIR.relative_to(REPO)}")
        return 0
    say(f"{'trial':32s}{'rows':>12s}{'shards':>8s}{'dims':>6s}{'minutes':>9s}  state")
    for path in manifests:
        m = Manifest.load(path)
        say(f"{m.trial:32s}{m.rows_done:>12,}{len(m.shards):>8d}{m.dimensions:>6d}"
            f"{m.seconds / 60:>9.1f}  {'complete' if m.complete else 'PARTIAL'}")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="verify every candidate on a tiny slice, then stop")
    p.add_argument("--smoke-one", default="", help="internal: smoke-test ONE candidate in this process")
    p.add_argument("--screen", action="store_true", help="score every embedded trial by exact dense recall")
    p.add_argument("--split", default="dev", help="which gold split to screen against (dev only, by design)")
    p.add_argument("--top-k", type=int, default=100, help="screening depth: the candidate ceiling")
    p.add_argument("--embed", action="store_true", help="embed the corpus for --trial")
    p.add_argument("--status", action="store_true", help="what is done and what it cost")
    p.add_argument("--trial", action="append", default=[], help="model__recipe (repeatable)")
    p.add_argument("--model", action="append", default=[], help="restrict --smoke to these model keys")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--limit", type=int, default=None, help="only this many chunks (a dry run)")
    p.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--build-screen-corpus", action="store_true",
                   help="choose and record the screening corpus: positives plus seeded distractors")  # fmt: skip
    p.add_argument("--screen-chunks", type=int, default=DEFAULT_SCREEN_CHUNKS)
    p.add_argument("--strict", action="store_true",
                   help="fail the smoke test if ANY candidate is unavailable (default: exclude and continue)")  # fmt: skip
    p.add_argument("--skip", action="append", default=[], help="exclude these model keys entirely (repeatable)")
    p.add_argument("--full-corpus", action="store_true",
                   help="embed or screen the WHOLE corpus instead of the screening set (hours per trial)")  # fmt: skip
    p.add_argument("--json", default="", help="write the report here")
    args = p.parse_args(argv)

    if args.status:
        return cmd_status(args)

    if args.build_screen_corpus:
        summary = build_screen_corpus(args.dsn, target=args.screen_chunks)
        if args.json:
            Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
        say(json.dumps(summary, indent=2))
        return 0

    if args.smoke_one:
        # One candidate, in this process, printing its result as JSON.
        from retrieval.bakeoff import HEADER_RECIPES, BY_KEY

        spec = BY_KEY.get(args.smoke_one)
        if spec is None:
            raise WorkerError(f"unknown model {args.smoke_one!r}")
        corpus = load_corpus(args.dsn, limit=max(SMOKE_ROWS, args.limit or SMOKE_ROWS))
        report = smoke([spec], corpus, list(HEADER_RECIPES), args.batch,
                       corpus_size=count_embeddable(args.dsn))  # fmt: skip
        entry = report["candidates"][spec.key]
        print("@@RESULT@@" + json.dumps({spec.key: entry}))
        return 0 if entry.get("ok") else 1

    if args.smoke:
        from retrieval.bakeoff import default_candidates, resolve_candidates, HEADER_RECIPES

        _used, total = vram_mb()
        specs = resolve_candidates(args.model or None, int(total) or 8192) if args.model else default_candidates(int(total) or 8192)
        if args.skip:
            specs = [s for s in specs if s.key not in set(args.skip)]
            say(f"skipping on request: {', '.join(sorted(set(args.skip)))}")
        if not specs:
            raise WorkerError("every candidate was skipped; nothing to smoke-test")
        total_chunks = count_embeddable(args.dsn)
        say(f"corpus: {total_chunks:,} embeddable chunks in the database")
        report = smoke_isolated(specs, args)
        report["corpus_chunks"] = total_chunks
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
            say(f"wrote {args.json}")
        unavailable = report.get("unavailable") or {}
        if unavailable:
            say("")
            for key, reason in sorted(unavailable.items()):
                say(f"  EXCLUDED  {key}: {str(reason)[:200]}")
        if args.strict and unavailable:
            say("\n--strict: refusing to continue with any candidate unavailable.")
            return 1
        if not report["ok"]:
            say(f"\nSMOKE TEST FAILED: {report.get('blocker', 'no usable candidate')}")
            return 1
        say(f"\nsmoke test passed for {len(report['available'])} candidate(s): "
            f"{', '.join(report['available'])}. Each loads, produces its declared width, normalises, "
            "truncates and gives its VRAM back.")  # fmt: skip
        if unavailable:
            say(f"{len(unavailable)} candidate(s) excluded and recorded; the bake-off continues without them.")
        return 0

    if args.screen:
        if args.split != "dev":
            raise WorkerError(
                f"screening is dev-only by design; {args.split!r} would spend a locked split on a model "
                "search. The test split confirms the winner and W3 is the final exam."
            )
        return cmd_screen(args)

    if args.embed:
        if not args.trial:
            p.error("--embed needs at least one --trial model__recipe")
        trials = [parse_trial(t) for t in args.trial]
        corpus = load_corpus(args.dsn, limit=args.limit) if args.full_corpus else load_screen_corpus(args.dsn)
        say(f"corpus: {len(corpus):,} embeddable chunks")
        results = {}
        for trial in trials:
            manifest = embed_trial(trial, corpus, shard_rows=args.shard_rows, batch=args.batch)
            results[trial.stem] = {"rows": manifest.rows_done, "complete": manifest.complete,
                                   "minutes": round(manifest.seconds / 60, 1)}  # fmt: skip
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
        say("\n" + json.dumps(results, indent=2))
        return 0 if all(r["complete"] for r in results.values()) else 1

    p.print_help()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkerError, BakeoffError, RecipeError) as exc:
        say(f"\nFAILED: {exc}")
        raise SystemExit(1) from exc
