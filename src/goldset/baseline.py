"""The reviewer's baseline: authored decisions, resolved against the pack and checked before they count."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "tools" / "review"
CALIBRATION_PATH = BASELINE_DIR / "baseline_calibration.json"
ITEMS_PATH = BASELINE_DIR / "baseline_items.json"

DECISIONS = ("keep", "edit", "drop", "verified")
GRADES = (0, 1, 2, 3)
MARK_KINDS = ("answer", "near_miss", "none")

# Characters that differ between what a reviewer types and what Hansard stores.
_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", "­": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ", "﻿": "",
    "​": "", "‌": "", "‍": "",
}  # fmt: skip


class BaselineError(ValueError):
    """A baseline that cannot be trusted. Always names the annotation that failed."""


def _fold(text: str) -> tuple[str, list[int]]:
    """Fold typography and collapse whitespace. Returns the folded text and, for each folded character, the index of
    the original character it came from -- which is what makes offsets recoverable.
    """
    out: list[str] = []
    origin: list[int] = []
    previous_space = False
    for i, char in enumerate(unicodedata.normalize("NFC", text)):
        folded = _FOLD.get(char, char)
        if folded == "":
            continue
        if folded.isspace():
            if previous_space:
                continue
            folded, previous_space = " ", True
        else:
            previous_space = False
        out.append(folded.lower())
        origin.append(i)
    return "".join(out), origin


def locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Where `needle` occurs in `haystack`, ignoring typography and whitespace runs, or None."""
    if not needle or not needle.strip():
        return None
    folded_hay, origin = _fold(haystack)
    folded_needle, _ = _fold(needle)
    if not folded_needle:
        return None
    at = folded_hay.find(folded_needle.strip())
    if at < 0:
        return None
    start = origin[at]
    last = origin[min(at + len(folded_needle.strip()), len(origin)) - 1]
    return start, last + 1


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, non-overlapping spans. Overlapping marks would nest `<mark>` tags and render as a mess."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass
class Annotation:
    """One authored judgment, before it is matched to the pack."""

    n: int
    kind: str  # "kappa" or "item"
    grade: int | None = None
    decision: str = ""
    summary: str = ""
    why: str = ""
    marks: tuple[str, ...] = ()
    mark_kind: str = "none"
    item: str = ""
    sid: str = ""
    checks: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        bad: list[str] = []
        where = f"annotation {self.n} ({self.item or '?'})"
        if self.kind == "kappa":
            if self.grade not in GRADES:
                bad.append(f"{where}: grade must be 0, 1, 2 or 3, not {self.grade!r}")
        elif self.decision not in DECISIONS:
            bad.append(f"{where}: decision must be one of {', '.join(DECISIONS)}, not {self.decision!r}")
        if not self.summary.strip():
            bad.append(f"{where}: needs a plain-English summary")
        if not self.why.strip():
            bad.append(f"{where}: needs a one-sentence reason")
        if self.mark_kind not in MARK_KINDS:
            bad.append(f"{where}: mark_kind must be one of {', '.join(MARK_KINDS)}")
        if self.mark_kind == "none" and self.marks:
            bad.append(f"{where}: mark_kind is 'none' but spans were given")
        if self.mark_kind != "none" and not self.marks:
            bad.append(f"{where}: mark_kind is {self.mark_kind!r} but no span was given")
        return bad


def _annotation(raw: dict, kind: str) -> Annotation:
    return Annotation(
        n=int(raw.get("n", 0)),
        kind=kind,
        grade=raw.get("grade"),
        decision=str(raw.get("decision", "")),
        summary=str(raw.get("summary", "")),
        why=str(raw.get("why", "")),
        marks=tuple(raw.get("marks", []) or ()),
        mark_kind=str(raw.get("mark_kind", "none")),
        item=str(raw.get("item", "")),
        sid=str(raw.get("sid", "")),
        checks=tuple(raw.get("checks", []) or ()),
    )


def load(path: Path, kind: str) -> list[Annotation]:
    """Read an annotation file. Returns if it does not exist: a baseline is optional, a broken one is not."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [_annotation(raw, kind) for raw in data.get("annotations", [])]


@dataclass
class Resolved:
    """A baseline matched to a pack: ready to embed, compare and export."""

    kappa: dict[str, dict[str, Any]] = field(default_factory=dict)
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _mark_spans(text: str, marks: tuple[str, ...], where: str, problems: list[str]) -> list[list[int]]:
    spans: list[tuple[int, int]] = []
    for mark in marks:
        found = locate(text, mark)
        if found is None:
            problems.append(f"{where}: highlight not found in the passage: {mark[:60]!r}")
            continue
        spans.append(found)
    return [[a, b] for a, b in merge_spans(spans)]


def resolve(
    pack: dict[str, Any],
    calibration: list[Annotation],
    items: list[Annotation],
    *,
    reviewer: str = "claude",
) -> Resolved:
    """Match annotations to the pack's queue by position, verify their checksums, resolve their highlights."""
    out = Resolved()
    queue = pack.get("queue", [])
    kappa_order = [e["id"] for e in queue if e["kind"] == "kappa"]
    item_order = [e["id"] for e in queue if e["kind"] == "item" and not e.get("phase", "").startswith("optional")]

    for annotation in calibration:
        out.problems.extend(annotation.validate())
        index = annotation.n - 1
        if not 0 <= index < len(kappa_order):
            out.problems.append(f"annotation {annotation.n}: there is no calibration pair at that position")
            continue
        key = kappa_order[index]
        pair = pack["kappa"][key]
        if annotation.item and pair["item_id"] != annotation.item:
            out.problems.append(
                f"annotation {annotation.n}: expected item {annotation.item}, the pack has {pair['item_id']}"
            )
            continue
        sid = annotation.sid.split("-placeholder")[0]
        if sid and not pair["speech_id"].startswith(sid):
            out.problems.append(
                f"annotation {annotation.n} ({annotation.item}): expected speech {sid}..., "
                f"the pack has {pair['speech_id'][:8]}..."
            )
            continue
        where = f"annotation {annotation.n} ({annotation.item})"
        out.kappa[key] = {
            "reviewer": reviewer,
            "grade": annotation.grade,
            "summary": annotation.summary,
            "why": annotation.why,
            "mark_kind": annotation.mark_kind,
            "spans": _mark_spans(pair.get("passage", ""), annotation.marks, where, out.problems),
            "item_id": pair["item_id"],
            "speech_id": pair["speech_id"],
        }

    for annotation in items:
        out.problems.extend(annotation.validate())
        index = annotation.n - 1
        if not 0 <= index < len(item_order):
            out.problems.append(f"item annotation {annotation.n}: there is no item at that position")
            continue
        key = item_order[index]
        if annotation.item and key != annotation.item:
            out.problems.append(f"item annotation {annotation.n}: expected {annotation.item}, the queue has {key}")
            continue
        record = pack["items"].get(key, {})
        where = f"item annotation {annotation.n} ({key})"
        contexts = [(piece.get("context") or piece.get("quote") or "") for piece in record.get("evidence", [])]
        # A span is looked for in EVERY evidence passage, not just the first: a multi-speech item's judgment often turns
        # on the second one, and "not found" there would be a false alarm.
        found: list[list[tuple[int, int]]] = [[] for _ in contexts]
        for mark in annotation.marks:
            for i, context in enumerate(contexts):
                if (at := locate(context, mark)) is not None:
                    found[i].append(at)
                    break
            else:
                out.problems.append(f"{where}: highlight not found in any evidence passage: {mark[:60]!r}")
        evidence_spans = [[[a, b] for a, b in merge_spans(spans)] for spans in found]
        out.items[key] = {
            "reviewer": reviewer,
            "decision": annotation.decision,
            "summary": annotation.summary,
            "why": annotation.why,
            "mark_kind": annotation.mark_kind,
            "checks": list(annotation.checks),
            "evidence_spans": evidence_spans,
        }
    return out


# --- the well-formedness checks the database spot-check phase actually needs -------------------------


_PLACEHOLDER = re.compile(r"TO_VERIFY|TODO|FIXME|XXX", re.IGNORECASE)

# Fields whose emptiness is a fact about Parliament, not a hole in the data.
NULLABLE = {
    "party": "the Speaker and some peers sit for no party",
    "house": "some stages, Royal Assent among them, are not taken in a House",
    "office_key": "a normalisation helper, not part of the answer",
    "proxy_name": "only a proxy vote has one",
    "office": "not every member holds an office",
}

_SCOPES_HOUSE = re.compile(r"\bin the (Commons|Lords|House of (?:Commons|Lords))\b", re.IGNORECASE)


def check_structured_facts(item: dict) -> list[str]:
    """What is wrong with a database-derived item's SQL answer, as plain sentences."""
    problems: list[str] = []
    facts = item.get("expected_structured_facts") or {}
    if not facts:
        problems.append("the item carries no expected_structured_facts at all")
        return problems
    if _PLACEHOLDER.search(json.dumps(facts, default=str)):
        problems.append("a placeholder is still in the answer")
    empty = [
        k
        for k, v in facts.items()
        if k not in NULLABLE and (v is None or (isinstance(v, (list, dict, str)) and len(v) == 0))
    ]
    if empty:
        problems.append(f"empty value for {', '.join(sorted(empty))}")

    # A question that says "in the Commons" about a stage the data attributes to no House asserts a scope its own answer
    # cannot support.
    if "house" in facts and facts.get("house") in (None, "") and _SCOPES_HOUSE.search(item.get("question", "")):
        problems.append("the question scopes an answer to a House that the answer itself leaves null")

    if not item.get("nuggets"):
        problems.append("no nuggets, so nothing can be scored against the answer")
    elif not any(n.get("vital") for n in item["nuggets"]):
        problems.append("no vital nugget, so a wrong answer cannot be distinguished from an incomplete one")
    return problems


def auto_item_annotations(pack: dict[str, Any], phase_prefix: str = "3 of 3") -> list[dict]:
    """Baseline entries for the database spot-check phase, from the mechanical checks."""
    out: list[dict] = []
    order = [e for e in pack.get("queue", []) if e["kind"] == "item" and not e.get("phase", "").startswith("optional")]
    for position, entry in enumerate(order, start=1):
        if not entry.get("phase", "").startswith(phase_prefix):
            continue
        item = pack["items"].get(entry["id"], {})
        problems = check_structured_facts(item)
        facts = item.get("expected_structured_facts") or {}
        shape = ", ".join(f"{k}" for k in sorted(facts)) or "nothing"
        out.append({
            "n": position,
            "item": entry["id"],
            "decision": "edit" if problems else "verified",
            "summary": f"A database-derived {item.get('category', 'item')} whose answer comes straight from SQL "
                       f"and carries {shape}.",
            "why": ("Automatic checks flagged: " + "; ".join(problems)) if problems
                   else "The SQL answer is present, non-empty, free of placeholders and has a vital nugget to score against.",
            "marks": [],
            "mark_kind": "none",
            "checks": problems or ["non-empty", "no placeholder", "has a vital nugget"],
        })  # fmt: skip
    return out


# --- agreement ---------------------------------------------------------------------------------------


def baseline_grades(resolved: Resolved) -> dict[tuple[str, str], int]:
    """The reviewer's grades as (item, speech) -> grade, the shape `judge.cohen_kappa` compares."""
    return {
        (v["item_id"], v["speech_id"]): int(v["grade"])
        for v in resolved.kappa.values()
        if v.get("grade") is not None
    }


def export(
    resolved: Resolved,
    pack: dict[str, Any],
    *,
    reviewer: str = "claude",
    agreement: dict[str, Any] | None = None,
    source: str = "",
) -> dict[str, Any]:
    return {
        "gold_hash": pack.get("gold_hash", ""),
        "split_hash": pack.get("split_hash", ""),
        "reviewed_at": "",
        "reviewer": reviewer,
        "source": source,
        "baseline": True,
        "blindness": {
            "how": "the grades were written from a review pack that carried no judge output: "
                   "scripts/blind_worksheet.py extracts the queue through a field allow-list and "
                   "re-checks its own output for any grade-shaped key before writing it",
            "authored_before_judging": True,
            "grades_unchanged_since": "the annotations in tools/review/baseline_*.json were not edited "
                                      "after the judge's grades existed",
            "what_this_does_not_prove": "the baseline and the judge's rubric were written by the same "
                                        "assistant, so agreement between them is not independent "
                                        "evidence that either is right. The owner's grades supersede "
                                        "the baseline wherever they are given, and the gate in the "
                                        "review tool is recomputed from the owner's grades alone.",
        },  # fmt: skip
        "decisions": {k: v["decision"] for k, v in resolved.items.items() if v.get("decision")},
        "owner_grades": {k: v["grade"] for k, v in resolved.kappa.items() if v.get("grade") is not None},
        "edits": {},
        "rationales": {
            **{k: {"summary": v["summary"], "why": v["why"]} for k, v in resolved.items.items()},
            **{k: {"summary": v["summary"], "why": v["why"]} for k, v in resolved.kappa.items()},
        },
        "agreement": agreement or {},
        "counts": {
            "decided": len(resolved.items),
            "graded": len(resolved.kappa),
            "queue": len(pack.get("queue", [])),
        },
    }
