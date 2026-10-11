"""Open the gold set v2 review tool."""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goldset.judge import JUDGMENTS_PATH  # noqa: E402
from goldset.review import DECISIONS_PATH, PACK_DIR  # noqa: E402
from goldset.schema import GOLD_PATH  # noqa: E402

PAGE = PACK_DIR / "gold_v2_review.html"
BASELINE = ROOT / "tools" / "review"


def stale() -> str:
    """Why the pack needs rebuilding, or '' if it does not."""
    if not PAGE.exists():
        return "the review pack has not been built yet"
    if not GOLD_PATH.exists():
        return "the gold set is missing"
    built = PAGE.stat().st_mtime
    if GOLD_PATH.stat().st_mtime > built:
        return "the gold set is newer than the review pack"
    # An edited baseline must reach the page, or the maintainer reviews against yesterday's rationales.
    for path in sorted(BASELINE.glob("baseline_*.json")):
        if path.stat().st_mtime > built:
            return f"{path.name} is newer than the review pack"
    if (BASELINE / "review_template.html").stat().st_mtime > built:
        return "the review template is newer than the review pack"
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the pack even if it looks current")
    parser.add_argument("--no-open", action="store_true", help="print the path instead of opening a browser")
    parser.add_argument("--rejudge", action="store_true",
                        help="re-run the LLM judge instead of reusing the cached grades (this spends money)")  # fmt: skip
    args = parser.parse_args(argv)

    if reason := ("rebuild was asked for" if args.rebuild else stale()):
        print(f"Rebuilding the review pack: {reason}.")
        # Default to the cache: re-rendering the page must never be a reason to re-run the judge.
        mode = "--judge" if args.rejudge else "--reuse-judgments"
        if mode == "--reuse-judgments" and not JUDGMENTS_PATH.exists():
            print(f"  no cached judgments at {JUDGMENTS_PATH.relative_to(ROOT)}; "
                  "the page will show the baseline with no judge column. "
                  "Run scripts/build_gold.py --judge once to grade the pool.")  # fmt: skip
        result = subprocess.run(  # noqa: S603 - a fixed command in this repository
            [sys.executable, str(ROOT / "scripts" / "build_gold.py"), mode], cwd=ROOT, check=False
        )
        if result.returncode != 0 or not PAGE.exists():
            print("The build did not produce a review pack. Run scripts/build_gold.py and read its output.")
            return 1

    print(f"\nReview tool: {PAGE}")
    print("  1-4 decide, 0-3 grade, a accepts the baseline for one item, arrows move, e edits, s saves.")
    print("  'Approve all' accepts every baseline decision you have not already made yourself.")
    print(f"  'Save decisions' downloads gold_v2_review.json; put it in {DECISIONS_PATH.parent.relative_to(ROOT)}/.")
    if args.no_open:
        return 0
    webbrowser.open(PAGE.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
