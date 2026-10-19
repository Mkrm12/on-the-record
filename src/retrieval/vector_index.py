"""The HNSW index the dense leg searches, and the ORDER BY that uses it — one coupled pair, defined once."""

from __future__ import annotations

import json
import re
from pathlib import Path

# The embedding model's dimension (BAAI/bge-small-en-v1.5).
DIM = 384
FORMS = ("vector", "halfvec")
INDEX_NAME = "chunk_embedding_idx"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

OPCLASS = {"vector": "vector_cosine_ops", "halfvec": "halfvec_cosine_ops"}
CAST = {"vector": "::vector", "halfvec": f"::halfvec({DIM})"}
# Bytes per vector the graph needs IN MEMORY while it is built — sizing maintenance_work_mem; when the graph outgrows
# it, pgvector inserts the rest one tuple at a time on disk, an order of magnitude slower.
ROW_BYTES = {"vector": 2_048, "halfvec": 1_600}

# How `pg_get_indexdef` prints each form, as a LIKE pattern, for checks that must run as SQL (the integrity gate).
INDEXDEF_LIKE = {"vector": "%USING hnsw (embedding vector_cosine_ops)%",
                 "halfvec": f"%USING hnsw (((embedding)::halfvec({DIM})) halfvec_cosine_ops)%"}

FROZEN_PATH = Path(__file__).resolve().parents[2] / "config" / "retrieval_frozen.json"
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _check(form: object) -> str:
    if form not in FORMS:
        raise ValueError(f"unknown index form {form!r}; the forms are {FORMS}")
    return form  # type: ignore[return-value]


def index_expression(form: str) -> str:
    """What the index is built over: the column itself, or its half-precision cast (parenthesised, as an expression
    index requires).
    """
    return "embedding" if _check(form) == "vector" else f"(embedding{CAST['halfvec']})"


def order_expression(form: str, alias: str = "c", param: str = "%(qvec)s") -> str:
    """The ORDER BY that the index of this form serves. It must repeat `index_expression` exactly."""
    column = f"{alias}.embedding" if alias else "embedding"
    if _check(form) == "vector":
        return f"{column} <=> {param}{CAST['vector']}"
    return f"{column}{CAST['halfvec']} <=> {param}{CAST['halfvec']}"


def index_sql(form: str, name: str = INDEX_NAME, *, concurrently: bool = False,
              m: int = HNSW_M, ef_construction: int = HNSW_EF_CONSTRUCTION) -> str:
    """CREATE INDEX for this form. `name` is formatted into SQL, so it must be a plain lower-case identifier."""
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(f"index name {name!r} is not a plain identifier")
    for value, label in ((m, "m"), (ef_construction, "ef_construction")):
        if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 1000:
            raise ValueError(f"{label} must be an integer from 2 to 1000")
    how = "CONCURRENTLY " if concurrently else ""
    return (f"CREATE INDEX {how}{name} ON chunk USING hnsw ({index_expression(form)} {OPCLASS[form]}) "
            f"WITH (m = {m}, ef_construction = {ef_construction})")


def form_of(indexdef: str | None) -> str | None:
    """The form of a live index, from `pg_indexes.indexdef`, or None when it is neither (another metric, another access
    method, no index). Postgres prints the halfvec expression as `((embedding)::halfvec(384))`.
    """
    text = indexdef or ""
    if "USING hnsw" not in text:
        return None
    if re.search(r"\(\(\(embedding\)::halfvec\(" + str(DIM) + r"\)\) halfvec_cosine_ops\)", text):
        return "halfvec"
    if re.search(r"\(embedding vector_cosine_ops\)", text):
        return "vector"
    return None


def frozen_form(path: Path = FROZEN_PATH) -> str:
    """The form frozen retrieval queries. A file frozen before the switch does not name one, and was measured on the
    full-precision index, so it reads as `vector`.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8")).get("config") or {}
    return _check(config.get("dense_index", "vector"))
