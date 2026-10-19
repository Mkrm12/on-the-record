"""The canary, ten fixed dev questions at the start of each paid session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "reports" / "canary.jsonl"

# Ten dev questions, fixed: evenly spaced through the dev split's retrieval questions in id order, none dated inside the
# sealed window.
IDS = ("adv0001", "adv0020", "adv0042", "hum0001", "hum0023", "hum0044", "hum0065", "hum0081", "old0001", "old0009")

RATE_TOLERANCE = 0.15
_COUNTS = ("claims_generated", "claims_verified", "claims_dropped")


def row(item_id: str, asked: Any) -> dict[str, Any]:
    """One question's line: its route, outcome and claim counts, and why it failed or degraded, if it did."""
    answer = getattr(asked, "answer", None)
    abstention = getattr(answer, "abstention", None) if answer is not None else None
    counts = getattr(asked, "claim_counts", None) or {}
    return {"id": item_id, "route": asked.route, "outcome": asked.outcome,
            "abstention": getattr(abstention, "value", None) if abstention else None,
            **{k: counts.get(k) for k in _COUNTS},
            "failed": asked.failed_reason or None, "degraded": asked.degraded_reason or None}  # fmt: skip


def summarise(rows: list[dict[str, Any]], provenance: dict[str, Any], *, verifier: str | None, spend_usd: float,
              when: str | None = None, stopped: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The run's record: what served it, the verifier, the totals and every row."""
    from datetime import datetime, timezone

    total = {k: sum(r.get(k) or 0 for r in rows) for k in _COUNTS}
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {"created": when or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ids": [r["id"] for r in rows], "questions": len(rows),
            "requested_model": provenance.get("requested_model"),
            "served_models": sorted(provenance.get("served_models") or []),
            "fingerprints": sorted(provenance.get("system_fingerprints") or []),
            "verifier": verifier or None, "outcomes": dict(sorted(outcomes.items())), **total,
            "verified_rate": (round(total["claims_verified"] / total["claims_generated"], 4)
                              if total["claims_generated"] else None),
            "spend_usd": round(spend_usd, 6), "rows": rows, "stopped": stopped}  # fmt: skip


def compare(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """Every way this run differs from the last one that matters, as sentences; when nothing moved."""
    drift = []
    if current.get("stopped"):
        # a partial run is recorded with its spend, and is never a clean baseline (QA pass 4, L6)
        drift.append(f"the run {current['stopped']}; it is not comparable as a whole")
    for key, what in (("served_models", "served model"), ("fingerprints", "fingerprint")):
        if len(current.get(key) or []) > 1:
            drift.append(f"more than one {what} inside this run: {current[key]}")
    for r in current["rows"]:
        if r.get("failed"):
            drift.append(f"{r['id']}: the request failed ({r['failed']})")
        elif r.get("degraded"):
            drift.append(f"{r['id']}: answered degraded ({r['degraded']})")
    if previous is None:
        return drift
    if list(previous.get("ids") or []) != list(current["ids"]):
        return [*drift, "the runs asked different questions, so they are not comparable"]
    for key, what in (("served_models", "served model"), ("fingerprints", "fingerprint"),
                      ("requested_model", "requested model"), ("verifier", "verifier")):  # fmt: skip
        if previous.get(key) != current.get(key):
            drift.append(f"the {what} changed: {previous.get(key)} -> {current.get(key)}")
    before, now = previous.get("verified_rate"), current.get("verified_rate")
    if (before is None) != (now is None) or (before is not None and abs(now - before) >= RATE_TOLERANCE):
        drift.append(f"the verification rate moved: {before} -> {now} "
                     f"({previous.get('claims_verified')}/{previous.get('claims_generated')} -> "
                     f"{current.get('claims_verified')}/{current.get('claims_generated')})")  # fmt: skip
    earlier = {r["id"]: r for r in previous.get("rows") or []}
    for r in current["rows"]:
        old = earlier.get(r["id"])
        if old is None:
            continue
        if old.get("route") != r.get("route"):
            drift.append(f"{r['id']}: the route changed: {old.get('route')} -> {r.get('route')}")
        if old.get("outcome") != r.get("outcome"):
            drift.append(f"{r['id']}: the outcome changed: {old.get('outcome')} -> {r.get('outcome')}"
                         + (f" ({r['abstention']})" if r.get("abstention") else ""))  # fmt: skip
    return drift


def last(path: Path = PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def append(path: Path, run: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(run, ensure_ascii=False, sort_keys=True) + "\n")
