"""Build the maintainer's review pack and render the local review tool."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from goldset import baseline as baseline_mod
from goldset.judge import Pool
from goldset.schema import GoldItem, content_hash, to_dict
from goldset.split import split_hash

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "tools" / "review" / "review_template.html"
PACK_DIR = ROOT / "data" / "review"
DECISIONS_PATH = ROOT / "tests" / "data" / "gold_v2_review.json"

# Measured on this pack's own content: a critical authored item runs to a few hundred words of evidence plus nuggets and
# forbidden claims. 45 s is a working pace with the keyboard, not a skim.
SECONDS_PER_ITEM = 45
SECONDS_PER_KAPPA = 20

KAPPA_SAMPLE = 100
DB_SPOT_CHECK = 30


@dataclass(frozen=True)
class Plan:
    critical: int
    kappa: int
    spot_check: int

    @property
    def total(self) -> int:
        return self.critical + self.kappa + self.spot_check

    def minutes(self) -> int:
        return round((self.critical * SECONDS_PER_ITEM + self.kappa * SECONDS_PER_KAPPA
                      + self.spot_check * SECONDS_PER_ITEM) / 60)  # fmt: skip


async def _evidence_context(conn: AsyncConnection, speech_ids: list[str]) -> dict[str, dict]:
    """The full speech around each quote, with who said it and in what capacity."""
    if not speech_ids:
        return {}
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT s.id, left(s.text, 2400) AS context, s.speaker_name, s.speaker_party, s.speaker_office,
                   d.house, d.section, d.date, d.heading
              FROM speech s LEFT JOIN debate d ON d.id = s.debate_id
             WHERE s.id = ANY(%(ids)s)
            """,
            {"ids": speech_ids},
        )
        return {r["id"]: r for r in await cur.fetchall()}


def _findings_by_item(validation: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for finding in validation.get("findings", []):
        out.setdefault(finding["item"], []).append(f"{finding['severity']}: {finding['check']} -- {finding['message']}")
    return out


async def build_pack(
    conn: AsyncConnection,
    items: list[GoldItem],
    *,
    pool: Pool | None = None,
    validation: dict[str, Any] | None = None,
    seed: int = 20260924,
    kappa_sample: int = KAPPA_SAMPLE,
    spot_check: int = DB_SPOT_CHECK,
    judge_grades: dict[str, dict[str, int]] | None = None,
    judge_model: str = "",
    with_baseline: bool = True,
) -> dict[str, Any]:
    """Everything the review tool needs, in one JSON-serialisable dict."""
    validation = validation or {}
    findings = _findings_by_item(validation)
    absence = validation.get("absence_candidates", {})

    speech_ids = sorted({e.speech_id for i in items for e in i.evidence})
    context = await _evidence_context(conn, speech_ids)

    packed: dict[str, dict] = {}
    for item in items:
        data = to_dict(item)
        data["evidence"] = [
            {
                "speech_id": e.speech_id,
                "quote": e.quote,
                "context": (context.get(e.speech_id) or {}).get("context") or e.quote,
                "speaker": (context.get(e.speech_id) or {}).get("speaker_name") or "",
                "party": (context.get(e.speech_id) or {}).get("speaker_party") or "",
                "office": (context.get(e.speech_id) or {}).get("speaker_office") or "",
                "house": (context.get(e.speech_id) or {}).get("house") or "",
                "section": (context.get(e.speech_id) or {}).get("section") or "",
                "date": str((context.get(e.speech_id) or {}).get("date") or ""),
                "heading": (context.get(e.speech_id) or {}).get("heading") or "",
            }
            for e in item.evidence
        ]
        data["findings"] = findings.get(item.id, [])
        data["absence_candidates"] = absence.get(item.id, [])
        packed[item.id] = data

    rng = random.Random(seed)

    # 1.
    critical = [i for i in items if i.criticality == "critical" and i.provenance != "db_template"]
    critical.sort(key=lambda i: (i.split != "dev", i.category, i.id))

    # 2.
    kappa: dict[str, dict] = {}
    if pool is not None and pool.pairs:
        chosen = rng.sample(pool.pairs, min(kappa_sample, len(pool.pairs)))
        for pair in chosen:
            kappa[f"k:{pair.item_id}:{pair.speech_id}"] = {
                "item_id": pair.item_id,
                "speech_id": pair.speech_id,
                "question": pair.question,
                "passage": pair.passage,
                "speaker": pair.speaker,
                "party": pair.party,
                "house": pair.house,
                "date": pair.date,
                "heading": pair.heading,
            }

    # 3.
    db_items = [i for i in items if i.provenance == "db_template"]
    spot = rng.sample(db_items, min(spot_check, len(db_items))) if db_items else []

    # 4.
    reviewed = {i.id for i in critical} | {i.id for i in spot}
    rest = [i for i in items if i.provenance != "db_template" and i.id not in reviewed]
    rest.sort(key=lambda i: (i.split != "dev", i.criticality != "major", i.category, i.id))

    queue = (
        [{"id": i.id, "kind": "item", "phase": "1 of 3 - critical items"} for i in critical]
        + [{"id": k, "kind": "kappa", "phase": "2 of 3 - judge calibration"} for k in kappa]
        + [{"id": i.id, "kind": "item", "phase": "3 of 3 - database spot-check"} for i in spot]
        + [{"id": i.id, "kind": "item", "phase": "optional - the rest of the authored items"} for i in rest]
    )
    plan = Plan(len(critical), len(kappa), len(spot))

    pack: dict[str, Any] = {
        "gold_hash": content_hash(items),
        "split_hash": split_hash(items),
        "built_at": validation.get("built_at", ""),
        "items": packed,
        "kappa": kappa,
        "queue": queue,
        "seconds_per_item": SECONDS_PER_ITEM,
        "seconds_per_kappa": SECONDS_PER_KAPPA,
        "plan": {"critical": plan.critical, "kappa": plan.kappa, "spot_check": plan.spot_check,
                 "total": plan.total, "estimated_minutes": plan.minutes()},  # fmt: skip
        "filters": [
            ["all", f"Everything ({len(queue)})"],
            ["todo", "Not yet decided"],
            ["disagreement", "Where the two graders differ"],
            ["1 of 3 - critical items", f"Critical items ({plan.critical})"],
            ["2 of 3 - judge calibration", f"Judge calibration ({plan.kappa})"],
            ["3 of 3 - database spot-check", f"Database spot-check ({plan.spot_check})"],
            ["optional - the rest of the authored items", f"The rest, optional ({len(rest)})"],
        ],
    }

    # The judge's raw grade per calibration pair, so the tool can put it beside the reviewer's.
    pack["judge"] = {
        "model": judge_model,
        "grades": {
            key: grade
            for key, pair in kappa.items()
            if (grade := (judge_grades or {}).get(pair["item_id"], {}).get(pair["speech_id"])) is not None
        },
    }

    # The authored baseline, resolved against THIS pack.
    pack["baseline"] = {"kappa": {}, "items": {}, "problems": [], "reviewer": ""}
    if with_baseline:
        resolved = baseline_mod.resolve(
            pack,
            baseline_mod.load(baseline_mod.CALIBRATION_PATH, "kappa"),
            baseline_mod.load(baseline_mod.ITEMS_PATH, "item"),
        )
        pack["baseline"] = {
            "kappa": resolved.kappa if resolved.ok else {},
            "items": resolved.items if resolved.ok else {},
            "problems": resolved.problems,
            "reviewer": "claude",
        }
    return pack


def render(pack: dict[str, Any], out_dir: Path = PACK_DIR) -> Path:
    """Write the self-contained page. Returns the path to open."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pack.json").write_text(json.dumps(pack, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    template = TEMPLATE.read_text(encoding="utf-8")
    # `</script>` inside the embedded JSON would close the host script tag early; escaping the slash is the standard fix
    # and leaves the JSON itself valid.
    blob = json.dumps(pack, ensure_ascii=False).replace("</", "<\\/")
    page = out_dir / "gold_v2_review.html"
    page.write_text(template.replace("__PACK__", blob), encoding="utf-8", newline="\n")
    return page
