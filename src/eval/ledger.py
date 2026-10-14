"""The pre-registration ledger, hash-chained."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "reports" / "preregistration.jsonl"
GENESIS = "0" * 64
KINDS = frozenset({"judged_run", "probe_run", "exam", "confirmation"})
_HASHED = ("seq", "created", "kind", "label", "gold", "rubric", "code", "plan", "bars", "prev")


def _digest(entry: dict[str, Any]) -> str:
    body = {k: entry.get(k) for k in _HASHED}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def current() -> dict[str, Any]:
    """What the next registration measures today: the gold set's hashes, the rubric and the code."""
    from answer.keys import code_version
    from goldset import rubric
    from goldset.schema import GOLD_PATH, content_hash, load

    gold = {"content_hash": content_hash(list(load(GOLD_PATH))),
            "file_sha256": hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()}  # fmt: skip
    return {"gold": gold, "rubric": rubric.RUBRIC_VERSION, "code": code_version()}


def register(path: Path, *, kind: str, label: str, plan: str, bars: dict[str, Any], gold: dict[str, Any], rubric: str,
             code: str, when: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Append one registration. It needs a plan and bars: a registration that says nothing pre-registers nothing."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; one of {sorted(KINDS)}")
    if not plan.strip() or not bars:
        raise ValueError("a registration states its analysis plan and its bars")
    problems = verify(path)
    if problems:
        raise ValueError(f"the ledger does not verify, so nothing is added to it: {problems[:3]}")
    entries = _entries(path)
    entry = {"seq": len(entries) + 1, "created": when or datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "kind": kind, "label": label, "gold": gold, "rubric": rubric, "code": code, "plan": plan, "bars": bars,
             "prev": entries[-1]["hash"] if entries else GENESIS}  # fmt: skip
    entry["hash"] = _digest(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry


def head(path: Path) -> str:
    """The chain's head: the last entry's hash (GENESIS when empty). Recorded outside the file, it is a receipt."""
    entries = _entries(path)
    return entries[-1]["hash"] if entries else GENESIS


def verify(path: Path, head: str | None = None) -> list[str]:
    """Every break in the chain: a wrong hash (an edit), a gap or repeat in the sequence, a wrong link; and, given a
    `head` recorded earlier, its absence (the last entries removed, or an edit re-hashed through to the end).
    """
    problems = []
    prev = GENESIS
    entries = _entries(path)
    if head is not None and head != GENESIS and head not in {e.get("hash") for e in entries}:
        problems.append(f"the receipt {head[:12]} is not in the chain (entries removed, or re-hashed after it)")
    for i, entry in enumerate(entries, start=1):
        if entry.get("seq") != i:
            problems.append(f"entry {i}: sequence number {entry.get('seq')} (a removal or a reorder)")
        if entry.get("prev") != prev:
            problems.append(f"entry {i}: does not link to the entry before it")
        if entry.get("hash") != _digest(entry):
            problems.append(f"entry {i}: its contents no longer match its hash (edited after registration)")
        prev = entry.get("hash")
    return problems


def run_problems(entry: dict[str, Any], *, code: str, started: str) -> list[str]:
    """Why a run may not cite this registration: it ran other code than the one registered, or started before the
    registration was made.
    """
    problems = []
    if entry.get("code") != code:
        problems.append(f"the run's code {code!r} is not the registered code {entry.get('code')!r}")
    if str(started) < str(entry.get("created")):
        problems.append(f"the run started ({started}) before the registration was made ({entry.get('created')})")
    return problems


def cited(path: Path, citation: str) -> dict[str, Any]:
    """The registration a report cites, by a hash prefix of at least 12 characters, from a chain that verifies."""
    if len(citation) < 12:
        raise KeyError("cite a registration by at least 12 characters of its hash")
    if verify(path):
        raise KeyError("the ledger does not verify")
    hits = [e for e in _entries(path) if e.get("hash", "").startswith(citation)]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} registrations match {citation!r}")
    return hits[0]
