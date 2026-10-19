"""The bake-off worker's GPU-dependent parts: exact search, VRAM release, shard resumability."""

from __future__ import annotations

import json

import numpy as np
import pytest

import bakeoff


def unit(rows: int, dims: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((rows, dims)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


# --- exact search ------------------------------------------------------------------------------------


def test_exact_search_agrees_with_numpy_on_the_true_top_k():
    """The screening number rests entirely on this. If the GPU path and numpy disagree, every recall figure in the
    bake-off is a figure about a bug.
    """
    docs, queries = unit(2_000, 384, seed=1), unit(25, 384, seed=2)
    got = bakeoff.search_exact(docs, queries, top_k=10)
    want = np.argsort(-(queries @ docs.T), axis=1)[:, :10]
    # Ties aside, the retrieved SETS must match; order within equal scores is not meaningful.
    for row_got, row_want in zip(got, want):
        assert set(row_got.tolist()) == set(row_want.tolist())


def test_a_planted_duplicate_is_found_first():
    docs = unit(500, 256, seed=3)
    queries = docs[[7, 100, 499]].copy()
    got = bakeoff.search_exact(docs, queries, top_k=1)
    assert got[:, 0].tolist() == [7, 100, 499]


def test_tiling_does_not_change_the_answer(monkeypatch):
    """A tile boundary must not lose the true top hit. Forced to a tiny tile so many boundaries are crossed."""
    docs, queries = unit(3_000, 128, seed=4), unit(12, 128, seed=5)
    full = bakeoff.search_exact(docs, queries, top_k=20)
    monkeypatch.setattr(bakeoff, "tile_rows", lambda *_a, **_k: 137)
    tiled = bakeoff.search_exact(docs, queries, top_k=20)
    for row_full, row_tiled in zip(full, tiled):
        assert set(row_full.tolist()) == set(row_tiled.tolist())


def test_top_k_larger_than_the_corpus_is_clamped_not_an_error():
    docs, queries = unit(5, 64, seed=6), unit(2, 64, seed=7)
    assert bakeoff.search_exact(docs, queries, top_k=100).shape == (2, 5)


def test_a_dimension_mismatch_is_refused_rather_than_broadcast():
    with pytest.raises(bakeoff.WorkerError, match="dimension mismatch"):
        bakeoff.search_exact(unit(10, 384), unit(3, 768), top_k=5)


def test_a_tile_is_never_degenerate_however_little_vram_is_free():
    assert bakeoff.tile_rows(1024, free_mb=0) >= 1_000
    assert bakeoff.tile_rows(1024, free_mb=100_000) > bakeoff.tile_rows(1024, free_mb=2_000)


# --- VRAM discipline ---------------------------------------------------------------------------------


def test_unload_releases_vram_and_leaves_the_card_where_it_started():
    """The single most likely way the long run dies: a leaked model shrinking the card for the next one."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    before, total = bakeoff.vram_mb()
    spec = bakeoff.BY_KEY["bge-small"]
    loaded = bakeoff.load_model(spec)
    assert loaded.real_dimensions == spec.dimensions
    during, _ = bakeoff.vram_mb()
    assert during > before, "loading a model should occupy VRAM"
    freed = bakeoff.unload(loaded)
    after, _ = bakeoff.vram_mb()
    assert freed > 0
    assert after <= before + 256, f"leaked {after - before:.0f} MB (before {before:.0f}, after {after:.0f})"


def test_unload_of_nothing_is_safe():
    assert bakeoff.unload(None) >= -1  # no model, no crash, no negative surprise


def test_a_model_whose_real_width_disagrees_with_the_spec_is_refused(monkeypatch):
    """Vectors written at the wrong width cannot be compared with anything, so this must stop the run before a single
    shard is written.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from retrieval.bakeoff import ModelSpec

    lying = ModelSpec(key="liar", name="BAAI/bge-small-en-v1.5", dimensions=999, max_seq=512,
                      revision=bakeoff.BY_KEY["bge-small"].revision, approx_params_m=33)  # fmt: skip
    with pytest.raises(bakeoff.WorkerError, match="declares 999"):
        bakeoff.load_model(lying)


def test_encode_rejects_vectors_that_are_not_unit_norm():
    """Cosine screening assumes normalised vectors; an un-normalised encoder would silently rank by length."""

    class Bad:
        def encode(self, texts, **_kw):
            return np.full((len(texts), 4), 3.0, dtype=np.float32)

    loaded = bakeoff.Loaded(spec=bakeoff.BY_KEY["bge-small"], model=Bad(), real_dimensions=4)
    with pytest.raises(bakeoff.WorkerError, match="unit-norm"):
        bakeoff.encode(loaded, ["a", "b"], batch=2)


def test_encode_rejects_non_finite_output():
    class Nan:
        def encode(self, texts, **_kw):
            out = np.zeros((len(texts), 4), dtype=np.float32)
            out[0, 0] = np.nan
            return out

    loaded = bakeoff.Loaded(spec=bakeoff.BY_KEY["bge-small"], model=Nan(), real_dimensions=4)
    with pytest.raises(bakeoff.WorkerError, match="non-finite"):
        bakeoff.encode(loaded, ["a"], batch=1)


def test_encode_rejects_the_wrong_shape():
    class Wrong:
        def encode(self, texts, **_kw):
            return np.eye(len(texts), 9, dtype=np.float32)

    loaded = bakeoff.Loaded(spec=bakeoff.BY_KEY["bge-small"], model=Wrong(), real_dimensions=4)
    with pytest.raises(bakeoff.WorkerError, match="expected"):
        bakeoff.encode(loaded, ["a"], batch=1)


# --- resumability ------------------------------------------------------------------------------------


def corpus(rows: int) -> bakeoff.Corpus:
    return bakeoff.Corpus(
        ids=list(range(rows)),
        speech_ids=[f"s{i // 3}" for i in range(rows)],
        rows=[{"body": f"sentence number {i}.", "heading": "A Debate", "date": "2026-09-02",
               "house": "Commons", "speaker_name": "A Member", "party": "Labour"} for i in range(rows)],  # fmt: skip
    )


class Fake:
    """A deterministic stand-in encoder, so resumability is tested without a card."""

    def __init__(self, dims: int = 8):
        self.dims = dims
        self.calls = 0

    def encode(self, texts, **_kw):
        self.calls += 1
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, hash(text) % self.dims] = 1.0
        return out


def fake_loaded(dims: int = 8) -> bakeoff.Loaded:
    return bakeoff.Loaded(spec=bakeoff.BY_KEY["bge-small"], model=Fake(dims), real_dimensions=dims)


def test_a_trial_embeds_every_row_in_shards(tmp_path):
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    loaded = fake_loaded()
    manifest = bakeoff.embed_trial(trial, corpus(25), out_dir=tmp_path, shard_rows=10, loaded=loaded)
    assert manifest.complete and manifest.rows_done == 25
    assert len(manifest.shards) == 3
    assert bakeoff.load_vectors(manifest, tmp_path).shape == (25, 8)


def test_a_resumed_run_does_only_the_missing_shards(tmp_path):
    """Halting at 80% must cost one shard, not the run."""
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    first = fake_loaded()
    bakeoff.embed_trial(trial, corpus(30), out_dir=tmp_path, shard_rows=10, loaded=first)
    assert first.model.calls == 3

    # Lose the last shard, as a crash would.
    manifest = bakeoff.Manifest.load(tmp_path / f"{trial.stem}.manifest.json")
    last = manifest.shard_name(2)
    (tmp_path / last).unlink()
    manifest.shards.pop(last)
    manifest.complete = False
    manifest.save()

    second = fake_loaded()
    again = bakeoff.embed_trial(trial, corpus(30), out_dir=tmp_path, shard_rows=10, loaded=second)
    assert second.model.calls == 1, "a resume must re-encode only the missing shard"
    assert again.complete and again.rows_done == 30


def test_a_complete_trial_does_no_work_at_all_on_a_re_run(tmp_path):
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    bakeoff.embed_trial(trial, corpus(20), out_dir=tmp_path, shard_rows=10, loaded=fake_loaded())
    again = fake_loaded()
    bakeoff.embed_trial(trial, corpus(20), out_dir=tmp_path, shard_rows=10, loaded=again)
    assert again.model.calls == 0, "re-running a finished trial must not touch the GPU"


def test_a_manifest_claiming_a_shard_that_is_not_on_disk_is_redone(tmp_path):
    """The manifest is a claim about the filesystem; the filesystem wins."""
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    manifest = bakeoff.Manifest.load(tmp_path / f"{trial.stem}.manifest.json")
    manifest.trial = trial.stem
    manifest.total_rows, manifest.shard_rows = 10, 10
    manifest.shards = {manifest.shard_name(0): 10}
    manifest.complete = True
    manifest.save()
    loaded = fake_loaded()
    again = bakeoff.embed_trial(trial, corpus(10), out_dir=tmp_path, shard_rows=10, loaded=loaded)
    assert loaded.model.calls == 1, "a phantom shard must be re-encoded"
    assert again.complete


def test_screening_refuses_a_partial_shard_set_rather_than_scoring_it(tmp_path):
    """Screening a partial corpus would silently report recall over a fraction of it."""
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    bakeoff.embed_trial(trial, corpus(30), out_dir=tmp_path, shard_rows=10, loaded=fake_loaded())
    manifest = bakeoff.Manifest.load(tmp_path / f"{trial.stem}.manifest.json")
    (tmp_path / manifest.shard_name(1)).unlink()
    with pytest.raises(bakeoff.WorkerError, match="missing"):
        bakeoff.load_vectors(manifest, tmp_path)


def test_the_manifest_records_the_recipe_so_two_vector_sets_cannot_be_confused(tmp_path):
    for recipe_key in ("sentence", "full"):
        trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY[recipe_key])
        bakeoff.embed_trial(trial, corpus(10), out_dir=tmp_path, shard_rows=10, loaded=fake_loaded())
    a = json.loads((tmp_path / "bge-small__sentence.manifest.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "bge-small__full.manifest.json").read_text(encoding="utf-8"))
    assert a["recipe"] != b["recipe"] and a["recipe_key"] != b["recipe_key"]
    assert not set(a["shards"]) & set(b["shards"]), "two trials must not share shard filenames"


def test_a_manifest_written_atomically_survives_a_torn_write(tmp_path):
    """The manifest is rewritten after every shard; a torn write would lose the whole run's bookkeeping."""
    path = tmp_path / "t.manifest.json"
    manifest = bakeoff.Manifest.load(path)
    manifest.trial = "t"
    manifest.save()
    assert path.is_file() and not list(tmp_path.glob("*.tmp"))
    path.write_text("{ not json", encoding="utf-8", newline="\n")
    assert bakeoff.Manifest.load(path).trial == "", "unreadable state starts clean rather than crashing"


def test_the_trial_stem_round_trips_through_the_cli_parser():
    trial = bakeoff.parse_trial("bge-base__heading_date")
    assert trial.spec.key == "bge-base" and trial.recipe.key == "heading_date"
    assert trial.stem == "bge-base__heading_date"


@pytest.mark.parametrize("bad", ["nonsense", "bge-base", "nope__sentence", "bge-base__nope"])
def test_a_malformed_trial_is_refused(bad):
    with pytest.raises(Exception):
        bakeoff.parse_trial(bad)


# --- shards belong to a corpus, not just to an index ------------------------------------------------- Found by running
# the worker end to end.


def test_a_shard_set_from_a_different_corpus_is_discarded_not_reused(tmp_path):
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    bakeoff.embed_trial(trial, corpus(30), out_dir=tmp_path, shard_rows=10, loaded=fake_loaded())

    # A different corpus of a different size, as the screening corpus was.
    smaller = fake_loaded()
    again = bakeoff.embed_trial(trial, corpus(20), out_dir=tmp_path, shard_rows=10, loaded=smaller)
    assert again.total_rows == 20, "the manifest must describe the corpus actually embedded"
    assert again.rows_done == 20, "it must not inherit rows from the previous corpus"
    assert smaller.model.calls == 2, "every shard must be re-encoded against the new corpus"
    assert bakeoff.load_vectors(again, tmp_path).shape[0] == 20


def test_a_corpus_of_the_same_size_but_different_members_is_also_detected():
    """Row count alone cannot tell two corpora apart; the fingerprint is over the ordered ids."""
    a = bakeoff.Corpus(ids=[1, 2, 3], speech_ids=["s"] * 3, rows=[{}] * 3)
    b = bakeoff.Corpus(ids=[1, 2, 4], speech_ids=["s"] * 3, rows=[{}] * 3)
    reordered = bakeoff.Corpus(ids=[3, 2, 1], speech_ids=["s"] * 3, rows=[{}] * 3)
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint != reordered.fingerprint, "order matters: shard i indexes into this exact list"
    assert a.fingerprint == bakeoff.Corpus(ids=[1, 2, 3], speech_ids=["s"] * 3, rows=[{}] * 3).fingerprint


def test_screening_refuses_shards_that_belong_to_another_corpus(tmp_path):
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    bakeoff.embed_trial(trial, corpus(20), out_dir=tmp_path, shard_rows=10, loaded=fake_loaded())
    manifest = bakeoff.Manifest.load(tmp_path / f"{trial.stem}.manifest.json")
    with pytest.raises(bakeoff.WorkerError, match="(?i)re-embed"):
        bakeoff.load_vectors(manifest, tmp_path, corpus=corpus(21))


def test_changing_the_shard_size_also_invalidates_the_shards(tmp_path):
    """Shard 0 of a 10-row shard set is not shard 0 of a 5-row one."""
    trial = bakeoff.Trial(bakeoff.BY_KEY["bge-small"], bakeoff.RECIPES_BY_KEY["sentence"])
    bakeoff.embed_trial(trial, corpus(20), out_dir=tmp_path, shard_rows=10, loaded=fake_loaded())
    changed = fake_loaded()
    again = bakeoff.embed_trial(trial, corpus(20), out_dir=tmp_path, shard_rows=5, loaded=changed)
    assert changed.model.calls == 4 and again.rows_done == 20
