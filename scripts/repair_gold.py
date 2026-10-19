"""Each repair is a defect in the instrument, not a judgement call about difficulty. A benchmark that penalises a
correct answer, or asks for a thing that does not exist, measures the benchmark.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goldset.schema import GOLD_PATH, content_hash, from_dict, validate  # noqa: E402
from goldset.split import split_hash  # noqa: E402

# The House clause, in each shape the generator's three phrasings put it.
_HOUSE_CLAUSE = (
    (re.compile(r"\s+in the (?:Commons|Lords)\?$"), "?"),
    (re.compile(r"^On what date did the (?:Commons|Lords) take the "), "On what date was the "),
)


def houseless_question(question: str) -> str | None:
    """`question` with the House clause removed, or None if it names no House."""
    for pattern, replacement in _HOUSE_CLAUSE:
        if pattern.search(question):
            return pattern.sub(replacement, question)
    return None


def scopes_a_house_it_has_not_got(item: dict) -> bool:
    """The question names a House while the item's own expected facts leave it null."""
    facts = item.get("expected_structured_facts") or {}
    return ("house" in facts and facts["house"] is None
            and houseless_question(item.get("question", "")) is not None)  # fmt: skip


def house_repairs(items: list[dict]) -> dict[str, dict]:
    """Computed, so it cannot miss one."""
    out = {}
    for item in items:
        if not scopes_a_house_it_has_not_got(item):
            continue
        stage = (item.get("expected_structured_facts") or {}).get("stage", "this stage")
        out[item["id"]] = {
            "question": houseless_question(item["question"]),
            "notes": f"[I-73] The House scope was removed 2026-09-26. The question asked for {stage} 'in the "
                     "Commons'; this item's own expected_structured_facts carries house: null, because the "
                     "stage is taken in neither House. Only the House clause changed -- the date, the bill "
                     "and the expected facts are untouched.",
            "_why": "I-73: a House the stage has not got",
        }
    return out


# id -> the fields to set, and why.
REPAIRS: dict[str, dict] = {
    "old0024": {
        "question": "What conditions did the regulator attach to SaxaVord's vertical-launch licence?",
        "forbidden_claims": [
            "Any statement of the licence conditions: the record does not give them.",
            "Offering the sentence about SaxaVord becoming the first licensed vertical-launch spaceport as "
            "though it answered what the conditions were.",
        ],
        "notes": "[I-72] Replaced 2026-09-26. The previous wording ('What was said about licensing the UK's "
                 "first spaceport launch?') was answered outright by a Commons sentence of 2024-05-08, so the "
                 "abstention was false and a correct cited answer would have scored -8. The new wording keeps "
                 "the near-miss purpose and is verified unanswerable: hansard_v2 has 7 SaxaVord chunks and 20 "
                 "spaceport chunks, and no chunk at all mentions a condition on a spaceflight or spaceport "
                 "licence. Still provisional: the owner has not confirmed the replacement.",
        "_why": "I-72: the abstention was false; the record answers the old question",
    },
    "hum0067": {
        "expected_behaviour": "partial",
        "notes": "[I-74] Re-typed from 'answer' to 'partial' on 2026-09-26. The vital nugget says the quoted "
                 "speech is a backbencher's own position and not the Government's, so a full answer to 'does "
                 "the government support...' is not in the record. The correct behaviour is to report what the "
                 "record does support and say plainly what it does not.",
        "_why": "I-74: typed `answer` while its own vital nugget says the answer is absent",
    },
    "hum0089": {
        "expected_behaviour": "partial",
        "notes": "[I-74] Re-typed from 'answer' to 'partial' on 2026-09-26. The record mentions the Chagos "
                 "islands only in passing, in a spending argument, and gives no cost. A system should say so "
                 "rather than either inventing a figure or abstaining on a question the record touches.",
        "_why": "I-74: typed `answer` while its own vital nugget says the answer is absent",
    },
}


def repairs_for(items: list[dict]) -> dict[str, dict]:
    return {**REPAIRS, **house_repairs(items)}


def check(path: Path = GOLD_PATH) -> list[str]:
    """What is still wrong. Empty means every repair is in place."""
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["items"]}
    problems = []
    for item_id, repair in repairs_for(data["items"]).items():
        item = by_id.get(item_id)
        if item is None:
            problems.append(f"{item_id} is missing from the gold set")
            continue
        for field, want in repair.items():
            if field.startswith("_"):
                continue
            if not in_place(item, field, want):
                problems.append(f"{item_id}.{field} is not repaired ({repair['_why']})")
    return problems


def in_place(item: dict, field: str, want) -> bool:
    """A repaired field holds its repair. `notes` may carry LATER notes after the repair's own, so for notes the repair
    is in place when they begin with it.
    """
    if field == "notes":
        return str(item.get(field) or "").startswith(str(want))
    return item.get(field) == want


def apply(path: Path = GOLD_PATH) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in data["items"]}
    changed = 0
    for item_id, repair in repairs_for(data["items"]).items():
        item = by_id.get(item_id)
        if item is None:
            print(f"  ! {item_id} is not in the gold set; skipped")
            continue
        for field, want in repair.items():
            if field.startswith("_") or in_place(item, field, want):
                continue
            print(f"  {item_id}.{field}: repaired ({repair['_why']})")
            item[field] = want
            changed += 1
    if not changed:
        print("nothing to repair")
        return 0

    if errors := validate(data):
        print(f"REFUSING to write: the repaired set does not validate ({len(errors)}): {errors[:3]}")
        return 1
    items = [from_dict(i) for i in data["items"] if i.get("status") != "dropped"]
    before = data.get("gold_hash", "")
    data["gold_hash"] = content_hash(items)
    data["split_hash"] = split_hash(items)
    data["repaired"] = "2026-09-26: I-72 (old0024), I-73 (db0191, db0194), I-74 (hum0067, hum0089)"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"\n{changed} field(s) repaired in {len(data['items'])} items")
    print(f"gold_hash {before} -> {data['gold_hash']}  (a repaired question is a different instrument)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--path", type=Path, default=GOLD_PATH)
    args = p.parse_args(argv)
    if args.apply:
        return apply(args.path)
    problems = check(args.path)
    for problem in problems:
        print(f"  {problem}")
    print(f"{len(problems)} repair(s) outstanding" if problems else "all repairs are in place")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
