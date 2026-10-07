"""Mutants for the GPU embedding worker (gpu/embedder.py) and the CUDA pre-flight (gpu/check_cuda.py)."""

import sys
from pathlib import Path

from mutation_engine import ROOT, Mutant, Project

GPU_PYTHON = ROOT / "gpu" / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

PROJECT = Project(
    # The embedder takes the index's definition from src/retrieval/vector_index.py and the form to build from
    # config/retrieval_frozen.json.
    dirs=("gpu", "db", "src", "config"),
    cwd="gpu",
    python=str(GPU_PYTHON),
    pytest_args=("-k", "not test_gpu"),
)
TESTS = ["tests/test_embedder.py", "tests/test_check_cuda.py"]


def available() -> bool:
    return Path(GPU_PYTHON).exists()


_E = "gpu/embedder.py"
_C = "gpu/check_cuda.py"
_V = "src/retrieval/vector_index.py"

_RAW = [
    # --- text and vector helpers
    (_E, "text with no letter or digit is accepted", "return cleaned if any(ch.isalnum() for ch in cleaned) else None", "return cleaned or None"),
    (_E, "NUL bytes are kept", 'text.replace("\\x00", "").split()', "text.split()"),
    (_E, "lone surrogates are kept", 'text = text.encode("utf-8", "ignore").decode("utf-8")', "pass"),
    (_E, "a zero row becomes NaN", "where=norms > 0", "where=norms >= 0"),
    (_E, "wrong dimensions are accepted", "if v.ndim != 2 or v.shape[1] != DIMENSIONS:", "if v.ndim != 2:"),
    (_E, "NaN and inf are accepted", "if not np.isfinite(v).all():", "if False:"),
    (_E, "the unit-length check is far too loose", ".max() > tolerance", ".max() > 1.0"),
    # --- model loading
    (_E, "the local cache is not tried first", "device=device, local_files_only=True)", "device=device)"),
    (_E, "any load error triggers a download", "except OSError:", "except Exception:"),
    (_E, "the queue includes boilerplate", 'return (f"NOT {c}is_boilerplate AND ({c}embedding IS NULL"',
     'return (f"({c}embedding IS NULL"'),
    (_E, "a chunk embedded under another recipe is treated as finished (I-28)",
     'f" OR {c}embedding_recipe IS DISTINCT FROM %(recipe)s)")', 'f" AND %(recipe)s IS NOT NULL)")'),
    (_E, "the page cursor is inclusive", ' WHERE {_pending("c")} AND c.id > %(after)s', ' WHERE {_pending("c")} AND c.id >= %(after)s'),
    (_E, "ids and vectors may differ in number", "if len(ids) != len(vectors):", "if False:"),
    (_E, "an existing vector of the SAME recipe is overwritten",
     " AND (c.embedding IS NULL OR c.embedding_recipe IS DISTINCT FROM %s)", ""),
    (_E, "the recipe tag is never written", "embedding_recipe = %s, embedded_at = now()", "embedded_at = now()"),
    (_E, "embedded_at is not set", "embedded_at = now()", "embedded_at = NULL"),
    (_E, "the model name is recorded wrongly", "model: str = MODEL_NAME", 'model: str = "unknown"'),
    # --- the worker loop
    (_E, "the limit is ignored when paging", "min(page_size, target - result.embedded - result.skipped)", "page_size"),
    (_E, "pages are never committed", "work.commit()", "pass"),
    (_E, "unembeddable rows are sent to the database", "for (cid, _), v in zip(rows, vectors) if v is not None]", "for (cid, _), v in zip(rows, vectors) if True]"),
    (_E, "the page cursor never advances", "after_id = rows[-1][0]", "after_id = 0"),
    # --- index lifecycle
    (_E, "a run of exactly the threshold size is not managed", "target >= index_threshold", "target > index_threshold"),
    (_E, "the index is rebuilt even after Ctrl-C", 'if index_mode != "never" and not interrupted:', 'if index_mode != "never":'),
    (_E, "an existing index is rebuilt on every run", "if not index_exists(admin):", "if True:"),
    (_E, "VACUUM ANALYZE runs when nothing was written", "if vacuum and result.embedded > 0:", "if vacuum:"),
    (_V, "the index uses the wrong operator class", '"halfvec": "halfvec_cosine_ops"', '"halfvec": "halfvec_l2_ops"'),
    (_E, "the embedder builds the full-precision index whatever is frozen", "return _vector_index().frozen_form()",
     'return "vector"'),
    (_E, "the build is sized for the wrong form", "_vector_index().ROW_BYTES[form or index_form()]",
     '_vector_index().ROW_BYTES["vector"]'),
    (_E, "the embedder and the shared definition may disagree", "if ours != theirs:", "if False:"),
    (_E, "the index m parameter changes", "HNSW_M = 16", "HNSW_M = 8"),
    (_E, "the index ef_construction changes", "HNSW_EF_CONSTRUCTION = 64", "HNSW_EF_CONSTRUCTION = 100"),
    # --- CUDA pre-flight
    (_C, "a CPU-only torch build is not detected", 'if info["cuda_version"] is None:', "if False:"),
    (_C, "an unusable device is not detected", 'if not info["cuda_available"]:', "if False:"),
    (_C, "any GPU is accepted", 'if EXPECTED_GPU not in (info["device_name"] or ""):', "if False:"),
    (_C, "strict mode is ignored", "(errors if strict else warnings).append(message)", "warnings.append(message)"),
    (_C, "low free VRAM is never reported", 'if info["free_mib"] < LOW_VRAM_MIB:', 'if info["free_mib"] < 0:'),
    (_C, "the VRAM boundary is off by one", 'if info["free_mib"] < LOW_VRAM_MIB:', 'if info["free_mib"] <= LOW_VRAM_MIB:'),
    (_C, "running on battery is never reported", 'if info.get("on_battery"):', "if False:"),
    (_C, "battery and mains are swapped", "{0: True, 1: False}.get(ac_line_status)", "{0: False, 1: True}.get(ac_line_status)"),
    (_C, "an unknown power source is taken for mains", "{0: True, 1: False}.get(ac_line_status)", "{0: True, 1: False}.get(ac_line_status, False)"),
]  # fmt: skip

MUTANTS = [Mutant(*row) for row in _RAW]
