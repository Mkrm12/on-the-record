"""Extract the blind review worksheet: everything a reviewer needs, and nothing the judge said."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goldset.blind import anonymise, assert_blind  # noqa: E402
from goldset.review import PACK_DIR  # noqa: E402

PACK = PACK_DIR / "pack.json"
DEFAULT_OUT = PACK_DIR / "blind_worksheet.json"

# Fields the reviewer is allowed to see, per kind.
KAPPA_FIELDS = ("item_id", "speech_id", "question", "passage", "speaker", "party", "house", "date", "heading")
ITEM_FIELDS = (
    "id", "question", "category", "subcategory", "criticality", "expected_behaviour", "provenance",
    "status", "split", "window", "house", "section", "date", "nuggets", "forbidden_claims",
    "expected_structured_facts", "evidence", "findings", "absence_candidates", "notes",
)  # fmt: skip
EVIDENCE_FIELDS = ("speech_id", "quote", "context", "speaker", "party", "office", "house", "section", "date", "heading")

def _pick(source: dict, fields: tuple[str, ...]) -> dict:
    return {f: source[f] for f in fields if f in source}


def build(pack: dict[str, Any]) -> dict[str, Any]:
    """The worksheet, ordered exactly as the review queue is, so the reviewer works one pass front to back."""
    items, kappa = pack.get("items", {}), pack.get("kappa", {})
    out_items: dict[str, dict] = {}
    out_kappa: dict[str, dict] = {}
    queue: list[dict] = []

    for entry in pack.get("queue", []):
        phase = entry.get("phase", "")
        if phase.startswith("optional"):
            continue
        if entry["kind"] == "kappa":
            pair = kappa.get(entry["id"])
            if pair is None:
                continue
            out_kappa[entry["id"]] = _pick(pair, KAPPA_FIELDS)
        else:
            item = items.get(entry["id"])
            if item is None:
                continue
            picked = _pick(item, ITEM_FIELDS)
            picked["evidence"] = [_pick(e, EVIDENCE_FIELDS) for e in item.get("evidence", [])]
            out_items[entry["id"]] = picked
        queue.append({"id": entry["id"], "kind": entry["kind"], "phase": phase})

    worksheet = {
        "gold_hash": pack.get("gold_hash", ""),
        "split_hash": pack.get("split_hash", ""),
        "built_at": pack.get("built_at", ""),
        "queue": queue,
        "items": out_items,
        "calibration_pairs": out_kappa,
        "counts": {"items": len(out_items), "calibration_pairs": len(out_kappa), "queue": len(queue)},
    }
    assert_blind(worksheet)
    return worksheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default=str(PACK), help="the review pack to extract from")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the worksheet")
    parser.add_argument("--anonymise", action="store_true",
                        help="opaque ids and a shuffled order, for an independent second grader")  # fmt: skip
    parser.add_argument("--shuffle-seed", type=int, default=20260925, help="seed for --anonymise")
    parser.add_argument("--key", default="", help="where to write the id mapping (never give this to the grader)")
    args = parser.parse_args(argv)

    source = Path(args.pack)
    if not source.exists():
        print(f"No pack at {source}. Run scripts/build_gold.py first.")
        return 1
    raw = source.read_bytes()
    pack = json.loads(raw)
    worksheet = build(pack)
    # The digest of the SOURCE pack, so the baseline can name the exact bytes it was graded from.
    worksheet["source_pack_sha256"] = hashlib.sha256(raw).hexdigest()

    key: dict[str, str] = {}
    if args.anonymise:
        worksheet, key = anonymise(worksheet, args.shuffle_seed)
        worksheet["source_pack_sha256"] = hashlib.sha256(raw).hexdigest()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(worksheet, ensure_ascii=False, indent=1), encoding="utf-8")
    if key:
        key_path = Path(args.key) if args.key else out.with_name(out.stem + "_key.json")
        key_path.write_text(json.dumps(key, indent=1), encoding="utf-8")
        print(f"Wrote the id mapping to {key_path} -- do NOT give this to the grader.")
    counts = worksheet["counts"]
    print(f"Wrote {out} -- {counts['items']} items and {counts['calibration_pairs']} calibration pairs, no grades.")
    print(f"Source pack sha256 {worksheet['source_pack_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
