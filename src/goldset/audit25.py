"""Audit the original 25 gold questions against live metadata, and carry them into schema v2 as dev data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from goldset.schema import Evidence, GoldItem, Nugget

OLD_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "gold_eval.json"

# The old categories, mapped onto the v2 vocabulary.
CATEGORY_MAP = {
    "exact_term": "exact_term",
    "paraphrase": "paraphrase",
    "numeric_fact": "numeric",
    "named_speaker": "named_speaker",
    "multi_speech": "multi_speech",
    "abstain": "near_miss_abstention",
}

# Each is a question for the maintainer, not a verdict: the point of writing them here is that review cannot quietly
# skip them.
FLAGS: dict[str, str] = {
    "g001": "Does 'while trying to save children' appear anywhere in the cited speech? The question asserts it.",
    "g002": "Was the speaker in opposition at the date? And the 35bn is a claim made in debate, not a fact.",
    "g003": "Is the speaker actually the Minister? 'He will know...' may be addressed TO a Minister, not BY one. "
            "The quote also does not mention deporting foreign national offenders.",
    "g006": "Loaded wording ('hostile protests') is embedded in the question itself.",
    "g007": "The source says 'holiday tax' and 'Prime Minister'; 'seasonal', 'tourists' and 'premier' are invented.",
    "g010": "Does the speaker hold a government post at the date? The question turns on it.",
    "g012": "The passage says 'the council' and never names the authority, so the question may be unanswerable.",
    "g015": "The temporal anchor ('since the Government set its ambition to halve it') is not in the quote, and "
            "11% is a ministerial claim with no stated baseline.",
    "g016": "Depends on a heading synonym ('Business questions' vs 'Business of the House'): this is I-38.",
    "g019": "Mention-vs-speaker: the speech NAMES Kerry McCarthy but she is not the speaker.",
    "g020": "Ambiguous about which contract is meant.",
    "g021": "Are both cited speeches actually on the same topic?",
    "g023": "Zero lexical presence, so trivially abstained. Keep one of g023-g025 as a sanity check and "
            "replace the others with near-miss abstentions.",
    "g024": "Zero lexical presence: see g023.",
    "g025": "Zero lexical presence: see g023.",
}

# Criticality for the carried-over items.
CRITICALITY = {"abstain": "critical"}


@dataclass(frozen=True)
class AuditFinding:
    item_id: str
    kind: str
    message: str


async def audit(conn: AsyncConnection, path: Path = OLD_PATH) -> tuple[list[dict], list[AuditFinding]]:
    """Re-check all 25 against the database. Returns the raw items and every finding."""
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["queries"]
    findings: list[AuditFinding] = []

    speech_ids = sorted({r["speech_id"] for q in items for r in q["relevant"]})
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT s.id, s.text, s.speaker_name, s.speaker_office, s.speaker_party, s.mnis_id,
                   s.window_id, d.date, d.house, d.section, d.heading
              FROM speech s LEFT JOIN debate d ON d.id = s.debate_id
             WHERE s.id = ANY(%(ids)s)
            """,
            {"ids": speech_ids},
        )
        speeches = {r["id"]: r for r in await cur.fetchall()}

    for item in items:
        qid = item["id"]
        for relevant in item["relevant"]:
            speech = speeches.get(relevant["speech_id"])
            if speech is None:
                findings.append(AuditFinding(qid, "missing", f"speech {relevant['speech_id']} is not in hansard_v2"))
                continue
            quote = " ".join(relevant["evidence_quote"].split())
            if quote.lower() not in " ".join((speech["text"] or "").split()).lower():
                findings.append(AuditFinding(qid, "quote", "the evidence quote is not verbatim in the speech"))
            # Questions that assert a role the metadata does not support.
            if "Minister" in item["question"] and not (speech["speaker_office"] or ""):
                findings.append(
                    AuditFinding(qid, "role", f"the question says 'Minister' but {speech['speaker_name']!r} "
                                              f"has no recorded office on {speech['date']}")  # fmt: skip
                )
            if speech["window_id"] != "W0":
                findings.append(AuditFinding(qid, "window", f"speech is in {speech['window_id']}, not W0"))
        if flag := FLAGS.get(qid):
            findings.append(AuditFinding(qid, "brief", flag))
    return items, findings


async def migrate(conn: AsyncConnection, path: Path = OLD_PATH) -> tuple[list[GoldItem], list[AuditFinding]]:
    """The 25 as schema-v2 dev items, with every audit finding carried in their notes."""
    items, findings = await audit(conn, path)
    by_item: dict[str, list[str]] = {}
    for finding in findings:
        by_item.setdefault(finding.item_id, []).append(f"[{finding.kind}] {finding.message}")

    speech_ids = sorted({r["speech_id"] for q in items for r in q["relevant"]})
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT s.id, s.window_id, d.house, d.section, d.date FROM speech s "
            "LEFT JOIN debate d ON d.id = s.debate_id WHERE s.id = ANY(%(ids)s)",
            {"ids": speech_ids},
        )
        context = {r["id"]: r for r in await cur.fetchall()}

    out: list[GoldItem] = []
    for item in items:
        category = CATEGORY_MAP[item["category"]]
        abstaining = item["expected_abstain"]
        first = context.get(item["relevant"][0]["speech_id"]) if item["relevant"] else None
        out.append(
            GoldItem(
                id=f"old{int(item['id'][1:]):04d}",
                question=item["question"],
                category=category,
                criticality=CRITICALITY.get(item["category"], "major"),
                expected_behaviour="abstain_not_in_record" if abstaining else "answer",
                provenance="owner_written",
                status="provisional",
                split="dev",  # permanently: every parameter in the system was tuned on these
                window=(first or {}).get("window_id") or "W0",
                house=(first or {}).get("house") or "",
                section=(first or {}).get("section") or "",
                date=str((first or {}).get("date") or ""),
                subcategory="carried_from_v1",
                evidence=tuple(Evidence(r["speech_id"], r["evidence_quote"]) for r in item["relevant"]),
                nuggets=() if abstaining else (Nugget("TO_VERIFY: nuggets were not recorded in v1.", True),),
                forbidden_claims=("Any answer at all: the topic is not in the record.",) if abstaining else (),
                notes=" | ".join(by_item.get(item["id"], [])) or "Carried from gold set v1; no audit finding.",
            )
        )
    return out, findings


def audit_report(findings: list[AuditFinding]) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
    return {
        "findings": len(findings),
        "by_kind": by_kind,
        "items_flagged": sorted({f.item_id for f in findings}),
        "detail": [{"item": f.item_id, "kind": f.kind, "message": f.message} for f in findings],
    }
