"""Compare an independent grader's blind reading of the W5 items against mine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


REPORTS = ROOT / "reports"
OUT_MD = REPORTS / "e1_w5_blind_comparison.md"
OUT_JSON = REPORTS / "e1_w5_blind_comparison.json"

# For these families I claim the passage does NOT support the question as asked.
EXPECT_UNSUPPORTED = frozenset({"false_premise", "misattribution", "mention_vs_speaker", "negation_flip",
                                "numeric_perturbation", "near_miss_abstention"})  # fmt: skip

# A loaded item may well be answerable; what matters is whether the grader noticed the framing.
EXPECT_LOADED = frozenset({"loaded"})


def compare(key: list[dict], grades: dict[str, dict]) -> list[dict]:
    rows = []
    for item in key:
        ref = item["ref"]
        got = grades.get(ref)
        category = item["category"]
        expected_unsupported = category in EXPECT_UNSUPPORTED
        if got is None:
            rows.append({**item, "graded": False, "verdict": "not graded",
                         "agrees": None, "note": "the grader returned nothing for this ref"})  # fmt: skip
            continue

        supports = str(got.get("supports_answer", "")).strip().lower()
        contradiction = (got.get("contradiction") or "").strip()
        loaded = (got.get("loaded") or "").strip()

        if expected_unsupported:
            agrees = supports in ("no", "partly")
            verdict = ("agrees: the grader also found the question unsupported as asked" if agrees
                       else "DISAGREES: the grader says the passage answers the question as asked")
            spotted = bool(contradiction)
        elif category in EXPECT_LOADED:
            agrees = bool(loaded)
            verdict = ("agrees: the grader noticed the framing" if agrees
                       else "DISAGREES: the grader saw nothing loaded in the question")
            spotted = bool(loaded)
        else:
            agrees = supports in ("yes", "partly")
            verdict = ("agrees: answerable" if agrees
                       else "DISAGREES: the grader says the passage does not answer it")
            spotted = True

        rows.append({**item, "graded": True, "supports_answer": supports,
                     "grader_contradiction": contradiction, "grader_loaded": loaded,
                     "grader_reply": (got.get("careful_reply") or "").strip(),
                     "agrees": agrees, "described_the_problem": spotted, "verdict": verdict})  # fmt: skip
    return rows


def render(rows: list[dict], grader: str) -> str:
    graded = [r for r in rows if r["graded"]]
    disagreements = [r for r in graded if r["agrees"] is False]
    described = [r for r in graded if r.get("described_the_problem")]
    L = [
        f"# E1: blind comparison of the W5 items ({grader})",
        "",
        f"{len(rows)} adversarial items, graded {time.strftime('%Y-%m-%d %H:%M')}. The grader saw the question "
        "and the passage and nothing else — no nugget, no forbidden claims, no category, no hint that any item "
        "was adversarial. Agreement is therefore evidence about the items rather than an echo of my reading.",
        "",
        f"- **{len(graded)} of {len(rows)} graded**",
        f"- **{len(graded) - len(disagreements)} agree** with the item's own reading",
        f"- **{len(disagreements)} disagree** — listed below, and the only ones worth the owner's time",
        f"- **{len(described)}** independently described the problem in the question",
        "",
    ]
    if disagreements:
        L += ["## Disagreements", "",
              "One of us is wrong about each of these. My reading is in `my_reading`; the grader's is beside it.",
              ""]
        for r in disagreements:
            L += [f"### {r['ref']} — {r['category']} ({r['criticality']})", "",
                  f"**Question:** {r['question']}", "",
                  f"- **Grader:** supports_answer = `{r['supports_answer']}`"
                  + (f"; contradiction: {r['grader_contradiction']}" if r["grader_contradiction"] else ""),
                  f"- **Grader would reply:** {r['grader_reply']}",
                  f"- **My reading:** {' / '.join(r['my_reading']) or '(none recorded)'}",
                  ""]
    else:
        L += ["## Disagreements", "", "**None.** The grader independently reached the same reading on every "
              "item, without seeing any of mine.", ""]

    L += ["## Agreements, for the record", "", "| ref | category | grader | described the problem |",
          "|---|---|---|---|"]
    for r in graded:
        if r["agrees"]:
            L.append(f"| {r['ref']} | {r['category']} | `{r['supports_answer']}` | "
                     f"{'yes' if r.get('described_the_problem') else 'no'} |")
    L += ["", "A grader that marks an item unsupported but cannot say why has noticed something without "
              "diagnosing it; that is weaker agreement than one that describes the contradiction, and the "
              "column is there so the difference is visible."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grader", default="data/review/w5_blind_sonnet.json")
    p.add_argument("--key", default="data/review/w5_blind_key.json")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    key_path, grader_path = ROOT / args.key, ROOT / args.grader
    if not key_path.exists():
        raise SystemExit(f"no key at {key_path}: run scripts/w5_blind_worksheet.py first")
    if not grader_path.exists():
        raise SystemExit(f"no grades at {grader_path}: the grader has not written them yet")

    key = json.loads(key_path.read_text(encoding="utf-8"))["key"]
    payload = json.loads(grader_path.read_text(encoding="utf-8"))
    grades = {row["ref"]: row for row in payload.get("items", [])}
    rows = compare(key, grades)

    graded = [r for r in rows if r["graded"]]
    disagreements = [r for r in graded if r["agrees"] is False]
    report = {
        "compared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grader": payload.get("grader", grader_path.stem),
        "items": len(rows), "graded": len(graded),
        "agree": len(graded) - len(disagreements), "disagree": len(disagreements),
        "described_the_problem": sum(1 for r in graded if r.get("described_the_problem")),
        "rows": rows,
    }
    REPORTS.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    OUT_MD.write_text(render(rows, report["grader"]), encoding="utf-8", newline="\n")

    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    else:
        print(f"{report['graded']} of {report['items']} graded: {report['agree']} agree, "
              f"{report['disagree']} disagree; {report['described_the_problem']} described the problem")
        for r in disagreements:
            print(f"  DISAGREES {r['ref']} ({r['category']}): grader says '{r['supports_answer']}'")
            print(f"            Q: {r['question'][:100]}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    # Not a gate: a disagreement is information for the maintainer, not a reason to fail a build.
    return 0


if __name__ == "__main__":
    sys.exit(main())
