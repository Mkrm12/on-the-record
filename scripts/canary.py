"""Run the canary, ten fixed dev questions through the whole paid pipeline (about 1p), and report drift."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import generate as G  # noqa: E402
from answer import pipeline as PL  # noqa: E402
from eval import canary as C  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run  # noqa: E402

# Per question; ten of them cost about 0.4p at the E3 rates.
CAP_PER_QUESTION_USD = 0.01


async def ask_all(stamp: str, cap: float) -> dict:
    items = {i.id: i for i in load_gold()}
    runtime, pool = await PL.build_runtime(with_verifier=True)
    try:
        if runtime.verifier is None:
            raise SystemExit("the verifier is unavailable; the canary does not run without it (nothing was spent)")
        rows, stopped = [], None
        async with pool.connection() as conn:
            for item_id in C.IDS:
                if runtime.spend_usd >= cap:
                    # recorded, with what it spent, never dropped (QA pass 4, L6)
                    stopped = f"stopped at the cap (${cap:.3f}) after {len(rows)} questions"
                    break
                item = items[item_id]
                asked = await PL.ask(runtime, conn, item.question, use_cache=False, store=False,
                                     budget=G.Budget(usd=CAP_PER_QUESTION_USD), run_id=f"canary-{stamp}",
                                     session_label="A-15 canary", gold_item_id=item.id, gold_split=item.split,
                                     gold_category=item.category, keep_question=True)  # fmt: skip
                rows.append(C.row(item_id, asked))
                print(f"  {item_id:8} {asked.route:10} {asked.outcome:8} {rows[-1]['claims_verified']}/"
                      f"{rows[-1]['claims_generated']} verified")  # fmt: skip
        return C.summarise(rows, runtime.client.provenance(), verifier=runtime.verifier_name,
                           spend_usd=runtime.spend_usd, stopped=stopped)  # fmt: skip
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cap", type=float, default=0.05, help="stop the run once it has spent this much (USD)")
    args = p.parse_args(argv)
    previous = C.last(C.PATH)
    if args.dry_run:
        print(f"would ask {len(C.IDS)} questions: {', '.join(C.IDS)}")
        print(f"cap ${args.cap:.3f}; at most ${CAP_PER_QUESTION_USD:.3f} a question")
        print("last run: " + (f"{previous['created']}, ${previous['spend_usd']:.4f}, served {previous['served_models']}"
                              if previous else "none (this would be the first)"))  # fmt: skip
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    current = run(ask_all(stamp, args.cap))
    drift = C.compare(previous, current)
    C.append(C.PATH, current | {"drift": drift})
    print(f"served {current['served_models']} {current['fingerprints']}; verifier {current['verifier']}; "
          f"verified {current['claims_verified']}/{current['claims_generated']} ({current['verified_rate']}); "
          f"outcomes {current['outcomes']}; spent ${current['spend_usd']:.4f}")  # fmt: skip
    if previous is None:
        print("the first canary: nothing to compare with" + ("" if not drift else ", but:"))
    print("\n".join(f"DRIFT: {line}" for line in drift) if drift else "no drift")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
