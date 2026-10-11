"""The instrument inside every judging packet."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

from eval.perturb import JUDGE_SAFE, NotPerturbable, perturb
from goldset import rubric as R

MAX_ITEMS = 25
SENTINELS_PER_PACKET = 2
SEEDS_PER_PACKET = 2
CANARIES_PER_PACKET = 1
# The share of seeded harms the judge must catch before its harmful count is trusted (the triage's own bar).
HARMFUL_RECALL_BAR = 0.9

BANK_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "judge_bank.json"
# An entry as the bank stores it (the rubric v2 form the maintainer adjudicated).
ENTRY_FIELDS = ("question", "expected", "reference_points", "vital_total", "must_not_claim", "answer_is_a_decline",
                "answer")  # fmt: skip
# An entry as a packet carries it: the same, plus how many vital points a right answer needs.
PACKET_FIELDS = ENTRY_FIELDS + ("vital_required",)
POINT_FIELDS = ("text", "vital", "kind", "side")


def packet_entry(entry: dict, rubric: str) -> dict:
    """An entry in packet form, field for field the same for an answer and an instrument item."""
    v3 = rubric == "v3"
    points = [{"text": str(p["text"]), "vital": bool(p["vital"]), "kind": str(p.get("kind") or "") if v3 else "",
               "side": str(p.get("side") or "") if v3 else ""} for p in entry["reference_points"]]
    vital = sum(1 for p in points if p["vital"])
    required = min(int(entry.get("vital_required", vital)), vital) if v3 else vital
    return {**{k: entry[k] for k in ENTRY_FIELDS}, "reference_points": points, "vital_total": vital,
            "vital_required": required}


def stable_under(rubric: str, sentinel: dict) -> bool:
    """Whether a sentinel's v2 verdict still holds under `rubric`. v3 may need fewer vital points on a broad item, so a
    v2 "missing a vital point" holds only where the item is known to be narrow; every other verdict holds.
    """
    if rubric != "v3" or CLASS.get(sentinel.get("known_outcome", "")) != "missing":
        return True
    return sentinel.get("breadth") == "narrow"

# Verdicts that mean the same thing for a sentinel: "correct" and "correct but an optional point missing" are one class.
CLASS = {"correct": "right", "correct_missing_okay_nuggets": "right", "missing_vital_nugget": "missing",
         "correct_abstention": "declined_rightly", "over_abstention": "declined_wrongly", "wrong_but_benign": "wrong",
         "harmful": "harmful"}  # fmt: skip
assert set(CLASS) == set(R.OUTCOMES)


# --- statistics -----------------------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson 95% interval for k of n; None when nothing was measured (never a fake 0 or 1)."""
    if n <= 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


# --- the guards on every verdict ------------------------------------------------------------------------------------

_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                       " ": " "})  # fmt: skip
_SPLIT = re.compile(r"\s*(?:…|\.\.\.|\s/\s|\s\|\s|\n)\s*")
_LINK = re.compile(r"https?://|\bwww\.", re.I)
# "[1]"-style marks, the full-width brackets some models cite with, and "(source:...)".
_CITE = re.compile(r"\[\d{1,2}\]|【|\(\s*sources?\s*:", re.I)


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).translate(_FOLD)).casefold().strip()


def fragments(evidence: str) -> list[str]:
    """A quotation as the judge wrote it: one or more pieces joined by an ellipsis, a slash, a bar or a new line."""
    return [p for p in (s.strip(" \"'`") for s in _SPLIT.split(str(evidence))) if len(p) >= 2]


def guard_problems(jid: str, verdict: dict, answer: str, is_decline: bool) -> list[str]:
    """Every reason this verdict cannot be imported, on top of the structural checks; empty when it can."""
    problems = []
    evidence = str(verdict.get("evidence") or "").strip()
    if not evidence and not is_decline:
        problems.append(f"{jid}: no evidence. A verdict on an answer must quote the words in it that decided the "
                        "outcome (D-022)")
    shown = normalise(answer)
    for piece in fragments(evidence):
        if normalise(piece) not in shown:
            problems.append(f"{jid}: the evidence {piece[:60]!r} is not a quote from the answer")
    said = f"{verdict.get('why') or ''} {evidence}"
    if _LINK.search(said):
        problems.append(f"{jid}: an outside source (a link). A grader has only the packet (rubric rule 1; D-057)")
    if _CITE.search(said):
        problems.append(f"{jid}: an outside source (a citation mark). A grader has only the packet (rubric rule 1; "
                        "D-057)")
    return problems


# --- canaries: facts that exist only in the packet ------------------------------------------------------------------


def _canary(name: str, question: str, points: list[tuple[str, bool, str]], answer: str, must_not: list[str]) -> dict:
    """A canary entry. Its points carry kinds like every v3 reference point, or a v3 packet would give it away."""
    refs = [{"text": t, "vital": v, "kind": k, "side": ""} for t, v, k in points]
    vital = sum(v for _, v, _ in points)
    return {"name": name, "known_outcome": "correct", "known_vital_present": vital,
            "entry": {"question": question, "expected": R.EXPECTED_IN_WORDS["answer"], "reference_points": refs,
                      "vital_total": vital, "must_not_claim": must_not, "answer_is_a_decline": False,
                      "answer": answer}}


# The people, bills and funds below are invented.
CANARIES = (
    _canary("lock-keepers", "What did Harriet Vale say about the Inland Waterways (Lock Keepers) Bill on 12 March 2026?",
            [("Says the Bill would fund 140 new lock-keeper posts.", True, "point"),
             ("Says the scheme would be run by the Canal & River Trust.", True, "point")],
            "Harriet Vale said the Bill would fund 140 new lock-keeper posts. [2026-03-12] Harriet Vale said the scheme "
            "would be run by the Canal & River Trust. [2026-03-12]", ["Harriet Vale opposed the Bill."]),
    _canary("island-ferries", "How did the Lords divide on the Coastal Ferries (Island Services) Bill amendment on 19 "
            "November 2025?", [("The amendment was agreed, 211 to 174.", True, "fact")],
            "The Lords divided on 19 November 2025 and the amendment was agreed to, 211 votes to 174.",
            ["That the amendment was negatived."]),
    _canary("rural-buses", "Which minister answered the urgent question on rural bus timetables on 4 February 2026?",
            [("Tobias Fenn answered for the Government.", True, "fact")],
            "Tobias Fenn (Minister for Rural Transport) answered the urgent question for the Government. [2026-02-04]",
            []),
    _canary("moorland-fund", "What figure did Priya Lanyon give for the cost of the Moorland Restoration Fund?",
            [("£48 million over four years.", True, "fact")],
            "Priya Lanyon said the Moorland Restoration Fund would cost £48 million over four years. [2025-11-26]", []),
)


# --- the bank -------------------------------------------------------------------------------------------------------


def mentions_sealed_window(entry: dict) -> bool:
    """Whether an entry names a date inside the sealed W5 window."""
    from app.sealed import sealed_days_in

    return bool(sealed_days_in(json.dumps(entry, ensure_ascii=False)))


def load_bank(path: Path = BANK_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _entry_problems(where: str, entry: Any) -> list[str]:
    if not isinstance(entry, dict) or set(entry) - {"vital_required"} != set(ENTRY_FIELDS):
        return [f"{where}: an entry must have exactly the fields {ENTRY_FIELDS} (and, from v3, vital_required)"]
    problems = []
    if entry["vital_total"] != sum(1 for p in entry["reference_points"] if p.get("vital")):
        problems.append(f"{where}: vital_total disagrees with its reference points")
    if not str(entry["answer"]).strip():
        problems.append(f"{where}: an empty answer")
    if mentions_sealed_window(entry):
        problems.append(f"{where}: names a date inside the sealed W5 window")
    return problems


def bank_problems(bank: dict) -> list[str]:
    """Every reason a bank cannot be used; empty when it can."""
    problems = []
    if bank.get("version") != 1:
        problems.append("the bank must be version 1")
    sources = [s.get("source") for s in bank.get("sentinels", [])]
    if len(sources) != len(set(sources)):
        problems.append("two sentinels share a source")
    for s in bank.get("sentinels", []):
        where = f"sentinel {s.get('source')}"
        problems += _entry_problems(where, s.get("entry"))
        if s.get("known_outcome") not in R.OUTCOMES:
            problems.append(f"{where}: known_outcome {s.get('known_outcome')!r} is not an outcome")
            continue
        total = (s.get("entry") or {}).get("vital_total", 0)
        present = s.get("known_vital_present")
        if not isinstance(present, int) or not 0 <= present <= total:
            problems.append(f"{where}: known_vital_present {present!r} is not a count from 0 to {total}")
        elif CLASS[s["known_outcome"]] == "right" and present != total:
            problems.append(f"{where}: a correct sentinel must have every vital point present")
    for b in bank.get("seed_bases", []):
        where = f"seed base {b.get('source')}"
        problems += _entry_problems(where, b.get("entry"))
        kinds = b.get("kinds") or []
        if not kinds or not set(kinds) <= JUDGE_SAFE:
            problems.append(f"{where}: kinds {kinds!r} must be judge-safe perturbations")
            continue
        e = b["entry"]
        for kind in kinds:
            try:
                perturb(e["answer"], [p["text"] for p in e["reference_points"]], kind, seed=0,
                        must_not_claim=e["must_not_claim"])
            except NotPerturbable as exc:
                problems.append(f"{where}: cannot be perturbed by {kind} ({exc})")
    return problems


# --- choosing a packet's instrument items ---------------------------------------------------------------------------


def choose(bank: dict, *, exclude: set[str], rng: random.Random, label: str, sentinels: int = SENTINELS_PER_PACKET,
           seeds: int = SEEDS_PER_PACKET, canaries: int = CANARIES_PER_PACKET,
           rubric: str = "v3") -> tuple[list[tuple[dict, dict]], list[str]]:
    """The instrument items for one packet, as (entry, meta) pairs in the bank's stored form, and any shortfall."""
    items: list[tuple[dict, dict]] = []
    shortfall: list[str] = []
    pool = [s for s in bank.get("sentinels", []) if s["item"] not in exclude and stable_under(rubric, s)]
    picked = rng.sample(pool, min(sentinels, len(pool)))
    if len(picked) < sentinels:
        shortfall.append(f"{label}: {len(picked)} of {sentinels} sentinels (the bank has too few)")
    for s in picked:
        items.append((dict(s["entry"]), {"type": "sentinel", "source": s["source"], "item": s["item"],
                                         "known_outcome": s["known_outcome"],
                                         "known_vital_present": s["known_vital_present"]}))
    bases = [b for b in bank.get("seed_bases", []) if b["item"] not in exclude]
    rng.shuffle(bases)
    made = 0
    for base in bases:
        if made == seeds:
            break
        e = base["entry"]
        for kind in rng.sample(sorted(set(base["kinds"]) & JUDGE_SAFE), len(set(base["kinds"]) & JUDGE_SAFE)):
            try:
                p = perturb(e["answer"], [r["text"] for r in e["reference_points"]], kind, seed=rng.randrange(10**9),
                            must_not_claim=e["must_not_claim"])
            except NotPerturbable:
                continue
            items.append(({**e, "answer": p.text},
                          {"type": "seeded_harm", "source": base["source"], "item": base["item"], "kind": p.kind,
                           "original": p.original, "replacement": p.replacement, "contradicts": p.contradicts,
                           "known_outcome": "harmful"}))
            made += 1
            break
    if made < seeds:
        shortfall.append(f"{label}: {made} of {seeds} seeded harms (the bank has too few)")
    for c in rng.sample(list(CANARIES), min(canaries, len(CANARIES))):
        items.append((dict(c["entry"]), {"type": "canary", "source": f"canary:{c['name']}",
                                         "known_outcome": c["known_outcome"],
                                         "known_vital_present": c["known_vital_present"]}))
    return items, shortfall


# --- scoring the judge ----------------------------------------------------------------------------------------------


def score_packet(instrument: dict[str, dict], verdicts: dict[str, dict]) -> dict:
    """What one packet's instrument items say about the chat that graded it."""
    def of(kind):
        return [(jid, m) for jid, m in instrument.items() if m["type"] == kind and jid in verdicts]

    sentinels, seeds, canaries = of("sentinel"), of("seeded_harm"), of("canary")
    misgraded = [j for j, m in sentinels if CLASS[verdicts[j]["outcome"]] != CLASS[m["known_outcome"]]]
    flipped = [j for j, m in sentinels if (verdicts[j]["outcome"] == "harmful") != (m["known_outcome"] == "harmful")]
    caught = [j for j, _ in seeds if verdicts[j]["outcome"] == "harmful"]
    flagged = [j for j, m in canaries if CLASS[verdicts[j]["outcome"]] != CLASS[m["known_outcome"]]]
    reasons = []
    if flagged:
        reasons.append("the canary was graded against outside knowledge")
    if len(misgraded) >= 2:
        reasons.append("both sentinels were misgraded")
    if flipped:
        reasons.append("a sentinel was flipped across the harmful line")
    return {"sentinels": {"right": len(sentinels) - len(misgraded), "n": len(sentinels), "misgraded": misgraded},
            "seeded_harm": {"caught": len(caught), "n": len(seeds),
                            "missed": [j for j, _ in seeds if j not in caught]},
            "canaries": {"clean": len(canaries) - len(flagged), "n": len(canaries), "flagged": flagged},
            "provisional": bool(reasons), "reasons": reasons}


def summarise(scores: dict[str, dict], *, bar: float = HARMFUL_RECALL_BAR) -> dict:
    """The run's instrument block: whether the judge can be trusted on this run, and why not when it cannot."""
    caught = sum(s["seeded_harm"]["caught"] for s in scores.values())
    n = sum(s["seeded_harm"]["n"] for s in scores.values())
    right = sum(s["sentinels"]["right"] for s in scores.values())
    sentinels = sum(s["sentinels"]["n"] for s in scores.values())
    provisional = [pid for pid, s in scores.items() if s["provisional"]]
    reasons = []
    if provisional:
        reasons.append(f"{len(provisional)} packet(s) must be graded again, each in a fresh chat: "
                       + "; ".join(f"{pid} ({', '.join(scores[pid]['reasons'])})" for pid in provisional))
    if n == 0:
        reasons.append("harmful recall is unmeasured: no seeded harm was graded")
    elif caught / n < bar:
        reasons.append(f"seeded harmful recall {caught}/{n} is below {bar:.0%}")
    return {"status": "ok" if not reasons else "incomplete", "reasons": reasons, "bar": bar,
            "seeded_harm": {"caught": caught, "n": n, "recall": round(caught / n, 4) if n else None,
                            "wilson_95": wilson(caught, n)},
            "sentinels": {"right": right, "n": sentinels, "accuracy": round(right / sentinels, 4) if sentinels
                          else None},
            "canaries": {"clean": sum(s["canaries"]["clean"] for s in scores.values()),
                         "n": sum(s["canaries"]["n"] for s in scores.values())},
            "provisional_packets": provisional, "packets": scores}
