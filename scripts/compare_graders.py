"""Compare independent graders on the calibration sample and surface only where they differ."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goldset import baseline, judge as judging  # noqa: E402
from goldset.blind import agreement, crosses_boundary, load_grader_file  # noqa: E402
from goldset.review import PACK_DIR  # noqa: E402

REPORTS = ROOT / "reports"
RELEVANT_AT = judging.RELEVANT_AT


def load_run1(pack: dict) -> tuple[dict[str, int], dict[str, dict]]:
    """The authored baseline, keyed by the pack's real pair id, with its reasoning."""
    resolved = baseline.resolve(
        pack,
        baseline.load(baseline.CALIBRATION_PATH, "kappa"),
        baseline.load(baseline.ITEMS_PATH, "item"),
    )
    if not resolved.ok:
        raise SystemExit("the baseline does not resolve against this pack:\n  " + "\n  ".join(resolved.problems[:5]))
    grades = {key: v["grade"] for key, v in resolved.kappa.items() if v.get("grade") is not None}
    return grades, resolved.kappa


def load_run2(path: Path, key_path: Path) -> tuple[dict[str, int], dict[str, str]]:
    """An independent grader's file, de-anonymised through the key it was never given."""
    grades, why, rejected = load_grader_file(path, key_path)
    for reason in rejected[:5]:
        print(f"  ! dropped {reason}")
    if len(rejected) > 5:
        print(f"  ! and {len(rejected) - 5} more")
    return grades, why


def load_judge(pack: dict) -> dict[str, int]:
    """The judge's RAW grade per calibration pair, not the floored value written onto the items."""
    try:
        _pool, raw, _meta = judging.load_judgments()
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ! no cached judgments ({exc}); the judge column will be empty")
        return {}
    flat = judging.flatten(raw)
    return {
        key: grade
        for key, pair in pack["kappa"].items()
        if (grade := flat.get((pair["item_id"], pair["speech_id"]))) is not None
    }


def agreement_block(a: dict[str, int], b: dict[str, int], label_a: str, label_b: str) -> dict:
    """Four-level and binarised agreement, from the package so the tests share one implementation."""
    return agreement(a, b, label_a, label_b, RELEVANT_AT)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run2", required=True, help="the second grader's JSON, keyed by opaque ids")
    p.add_argument("--key", required=True, help="the opaque-id mapping the grader never saw")
    p.add_argument("--run2-label", default="run2")
    p.add_argument("--pack", default=str(PACK_DIR / "pack.json"))
    p.add_argument("--json", default=str(REPORTS / "session2_grader_comparison.json"))
    p.add_argument("--show", type=int, default=0, help="print the N worst disagreements in full")
    args = p.parse_args(argv)

    pack = json.loads(Path(args.pack).read_text(encoding="utf-8"))
    run1, run1_notes = load_run1(pack)
    run2, run2_why = load_run2(Path(args.run2), Path(args.key))
    judge = load_judge(pack)
    label2 = args.run2_label

    print(f"graders: run1 (claude/opus) {len(run1)}, {label2} {len(run2)}, judge {len(judge)}\n")
    for label, grades in (("run1", run1), (label2, run2), ("judge", judge)):
        dist = Counter(grades.values())
        rel = sum(1 for g in grades.values() if g >= RELEVANT_AT)
        print(f"  {label:8s} distribution {dict(sorted(dist.items()))}  "
              f"relevant(>= {RELEVANT_AT}) {rel}/{len(grades)}")  # fmt: skip

    blocks = [
        agreement_block(run1, run2, "run1", label2),
        agreement_block(run1, judge, "run1", "judge"),
        agreement_block(run2, judge, label2, "judge"),
    ]
    print("\nagreement (four-level kappa / binarised kappa at the relevance boundary):")
    for b in blocks:
        four, binary = b["four_level"], b["binarised"]
        print(f"  {b['pair']:22s} n={b['n']:3d}  four {four['kappa']}  binarised {binary['kappa']}"
              f"  ({'PASSES' if binary.get('passes_gate') else 'fails'} the 0.6 gate)")  # fmt: skip
        print(f"      {b['crossing_disagreements']:3d} disagreements cross the relevance boundary "
              f"(these can change a metric), {b['non_crossing_disagreements']:3d} do not")  # fmt: skip

    # --- the disagreements, which is the point of the exercise -----------------------------------------
    shared = sorted(set(run1) & set(run2))
    rows = []
    for key in shared:
        g1, g2 = run1[key], run2[key]
        gj = judge.get(key)
        crossing = crosses_boundary(g1, g2, RELEVANT_AT)
        if g1 == g2 and (gj is None or gj == g1):
            continue
        pair = pack["kappa"][key]
        rows.append({
            "key": key, "item_id": pair["item_id"], "speech_id": pair["speech_id"],
            "question": pair["question"], "heading": pair.get("heading", ""),
            "speaker": pair.get("speaker", ""), "date": pair.get("date", ""),
            "run1": g1, "run2": g2, "judge": gj,
            "run1_why": (run1_notes.get(key) or {}).get("why", ""),
            "run1_summary": (run1_notes.get(key) or {}).get("summary", ""),
            "run2_why": run2_why.get(key, ""),
            "spread": max([g for g in (g1, g2, gj) if g is not None]) - min([g for g in (g1, g2, gj) if g is not None]),
            "two_runs_disagree": g1 != g2,
            "crossing": crossing,
            "runs_agree_judge_differs": g1 == g2 and gj is not None and gj != g1,
            "judge_crosses_against_both": (
                g1 == g2 and gj is not None and crosses_boundary(gj, g1, RELEVANT_AT)
            ),
        })  # fmt: skip
    rows.sort(key=lambda r: (-r["spread"], not r["crossing"], r["item_id"]))

    both_runs = [r for r in rows if r["two_runs_disagree"]]
    runs_v_judge = [r for r in rows if r["runs_agree_judge_differs"]]
    print(f"\n{len(rows)} pairs where any two graders differ:")
    print(f"  {len(both_runs)} where the two blind runs disagree with each other "
          f"({sum(1 for r in both_runs if r['crossing'])} cross the relevance boundary)")  # fmt: skip
    print(f"  {len(runs_v_judge)} where both blind runs agree and the judge differs "
          f"({sum(1 for r in runs_v_judge if r['judge_crosses_against_both'])} cross the boundary)")  # fmt: skip

    if args.show:
        print(f"\nthe {min(args.show, len(rows))} widest disagreements:")
        for r in rows[: args.show]:
            print(f"\n  [{r['item_id']}] {r['question']}")
            print(f"    {r['speaker']} | {r['date']} | {r['heading']}")
            print(f"    run1 {r['run1']}  run2 {r['run2']}  judge {r['judge']}"
                  f"{'   << crosses the relevance boundary' if r['crossing'] else ''}")  # fmt: skip
            if r["run1_why"]:
                print(f"    run1: {r['run1_why']}")
            if r["run2_why"]:
                print(f"    run2: {r['run2_why']}")

    report = {
        "relevant_at": RELEVANT_AT,
        "graders": {"run1": "claude-opus (authored baseline)", "run2": label2, "judge": "deepseek-flash"},
        "counts": {"run1": len(run1), "run2": len(run2), "judge": len(judge)},
        "distributions": {
            "run1": {str(g): sum(1 for v in run1.values() if v == g) for g in range(4)},
            "run2": {str(g): sum(1 for v in run2.values() if v == g) for g in range(4)},
            "judge": {str(g): sum(1 for v in judge.values() if v == g) for g in range(4)},
        },
        "agreement": blocks,
        "disagreements": rows,
        "summary": {
            "any_difference": len(rows),
            "blind_runs_disagree": len(both_runs),
            "blind_runs_disagree_crossing": sum(1 for r in both_runs if r["crossing"]),
            "runs_agree_judge_differs": len(runs_v_judge),
            "judge_crosses_against_both": sum(1 for r in runs_v_judge if r["judge_crosses_against_both"]),
        },
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    # A --json given as a relative path is already readable as it stands; only an absolute one under the repository is
    # worth shortening, and one outside it cannot be.
    try:
        shown = out.relative_to(ROOT) if out.is_absolute() else out
    except ValueError:
        shown = out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
