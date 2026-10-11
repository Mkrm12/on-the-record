"""The cache key, and the corpus it identifies."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# The corpus this project answers against: W0 through W5, 1,437,856 chunks.
CORPUS_VERSION = "v2-w0w5"


def normalise_question(question: str) -> str:
    """The form two spellings of one question share."""
    text = " ".join((question or "").split()).casefold()
    return text.rstrip("?").strip()


def question_hash(question: str, filters: dict[str, Any] | None = None, route: str = "") -> str:
    """The cache key's question half: the normalised question, its resolved filters and the route."""
    payload = json.dumps(
        {"q": normalise_question(question), "f": filters or {}, "r": route},
        sort_keys=True, separators=(",", ":"), default=str,
    )  # fmt: skip
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


_CODE_VERSION: str | None = None


def code_version() -> str:
    """A fingerprint of everything that decides an answer: the answer layer, the retrieval layer and the frozen
    retrieval config. Part of the cache key.
    """
    global _CODE_VERSION
    if _CODE_VERSION is None:
        import hashlib

        root = _root()
        digest = hashlib.sha256()
        for path in fingerprinted_files():
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(path.read_bytes())
        _CODE_VERSION = digest.hexdigest()[:12]
    return _CODE_VERSION


def _root():
    from pathlib import Path as _Path

    return _Path(__file__).resolve().parents[2]


def fingerprinted_files() -> list:
    """What `code_version` hashes: the answer and retrieval layers, the frozen config — and the migrations, because
    answers now depend on SQL functions defined there (`speaker_as_printed`, `printed_full_name`).
    """
    root = _root()
    files = sorted((root / "src" / "answer").glob("*.py")) + sorted((root / "src" / "retrieval").glob("*.py"))
    files.append(root / "config" / "retrieval_frozen.json")
    files += sorted((root / "db" / "migrations").glob("*.sql"))
    return files

