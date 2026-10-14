"""Run the deterministic probes against the corpus. No model, no GPU, no spend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import safety, sealed  # noqa: E402
from app.config import settings  # noqa: E402
from eval.oracles import divisions as O  # noqa: E402
from eval.probes import core  # noqa: E402
from eval.oracles import bills as OB  # noqa: E402
from eval.oracles import members as OM  # noqa: E402
from eval.oracles import openers as OO  # noqa: E402
from eval.probes import bills as PB  # noqa: E402
from eval.probes import divisions as PD  # noqa: E402
from eval.probes import members as PM  # noqa: E402
from eval.probes import offices as PF  # noqa: E402
from eval.probes import openers as PO  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402


def _verdict(truths, debates):
    by_division, by_debate, by_stage = PD.verdict_for(truths), PO.verdict_for(debates), PB.verdict_for(())
    by_party, by_office = PM.verdict_for(()), PF.verdict_for(())

    def verdict(trial, answer):
        if trial.cls == "MR-08":
            return by_debate(trial, answer)
        if trial.cls == "MR-07":
            return by_office(trial, answer)
        if trial.cls == "ST-bill_stage_date":
            return by_stage(trial, answer)
        if trial.cls == "ST-party_at_date":
            return by_party(trial, answer)
        return by_division(trial, answer)

    return verdict


async def _run(trials, truths, debates) -> list[core.Outcome]:
    import psycopg

    dsn = safety.working_dsn(settings.pg_dsn)
    async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=5,
                                                     options="-c default_transaction_read_only=on") as conn:  # fmt: skip
        await conn.execute(sealed.serving_sql())  # production's own connection: the served views
        await conn.commit()
        return await core.run(conn, trials, _verdict(truths, debates))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--forms", default="canonical,variant,paraphrase,short_peer,address,display,compound,run_on,possessive,"
                                     "relational,party_subset,agree_to,carried,party_tally")  # QA pass 4 and 5's forms
    p.add_argument("--categories", default=",".join(PD.CATEGORIES))
    p.add_argument("--limit", type=int, default=0, help="only the first N divisions, by id")
    p.add_argument("--classes", default="st,stages,party,mr02,mr04,mr06,mr07,mr08,mr10,mr11,mr15",
                   help="st (every division, by name), mr02 (a division the record lacks), mr06 (a day's, unnamed), "
                        "mr08 (every debate the Chair opens: who opened it?)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--registration", help="the A-25 registration this run answers to (a hash prefix of 12 or more)")
    p.add_argument("--out", type=Path, default=None,
                   help="default reports/probes_<date>_<time>_<code version>.json: a run never overwrites another's "
                        "report, and the name says what code it measured (QA pass 3)")
    args = p.parse_args(argv)

    truths = sorted(O.truths(), key=lambda t: t.division_id)
    if args.limit:
        truths = truths[: args.limit]
    wanted = set(args.classes.split(","))
    registration = None
    if args.registration:
        from eval import ledger as L

        registration = L.cited(L.PATH, args.registration)  # refuses a citation the chain does not hold
        from datetime import datetime, timezone

        from answer.keys import code_version

        mismatch = L.run_problems(registration, code=code_version(),
                                  started=datetime.now(timezone.utc).isoformat(timespec="seconds"))  # fmt: skip
        if mismatch:
            raise SystemExit("REFUSED: this run cannot answer to that registration: " + "; ".join(mismatch))
    trials = []
    forms = tuple(args.forms.split(","))
    profiles = OM.profiles() if "st" in wanted and set(forms) & set(PD.OTHER_NAMES) else None
    if "st" in wanted:
        trials += PD.st_trials(truths, forms=forms, categories=tuple(args.categories.split(",")), profiles=profiles)
    if "mr02" in wanted:
        trials += PD.mr02_trials(truths)
    if "party" in wanted:
        trials += PM.party_trials(truths, history=OM.history(), seats=OM.seats())  # seats: QA pass 4
    if "stages" in wanted:
        trials += PB.st_trials(OB.truths())
    if "mr04" in wanted:
        trials += PD.mr04_trials(truths)
    if "mr06" in wanted:
        trials += PD.mr06_trials(truths)
    if "mr07" in wanted:
        trials += PF.mr07_trials(OM.posts(), {t.date for t in truths})
    if "mr10" in wanted:
        trials += PD.mr10_trials(truths)
    if "mr11" in wanted:
        trials += PD.mr11_trials(truths)
    if "mr15" in wanted:
        trials += PD.mr15_trials(PD.st_trials(truths, forms=("canonical",)))
    debates = list(OO.truths()) if "mr08" in wanted else []
    if "mr08" in wanted:
        trials += PO.mr08_trials(debates)
    per_class = {}
    for t in trials:
        per_class[t.cls] = per_class.get(t.cls, 0) + 1
    untitled = sum(1 for t in truths if not t.title)
    print(f"{len(truths)} divisions outside the sealed window ({untitled} with no title, so not askable by name); "
          f"{len(trials)} trials: {per_class}")
    if args.dry_run:
        return 0

    started = time.perf_counter()
    outcomes = run_async(_run(trials, truths, debates))
    seconds = round(time.perf_counter() - started, 1)
    summary = core.summarise(outcomes)
    from answer.keys import code_version

    code = code_version()
    if args.out is None:
        args.out = ROOT / "reports" / f"probes_{date.today().isoformat()}_{time.strftime('%H%M%S')}_{code[:8]}.json"
    report = {"report": "probes", "date": date.today().isoformat(), "code_version": code,
              "truth": "data/raw/corpus_v2 (Votes API payloads; Hansard's printed figures beside them), outside the "
                       "sealed window", "path": "router rules -> structured_answer -> reader_ready -> sealed guard",
              "seconds": seconds, "divisions": len(truths), "relations": PD.relations(outcomes),
              "registration": None if registration is None else {k: registration[k] for k in ("seq", "hash", "label")},
              **summary}  # fmt: skip
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    for cls, block in summary["by_class_and_form"].items():
        print(f"{cls:38} {block['trials']:5} trials  pass {block['pass_rate']}  acceptable {block['acceptable_rate']}  "
              f"harm {block['harm']} + mismatch {block['mismatch']} / {block['answered']} answered "
              f"(95% upper {'n/a, nothing answered' if block['vacuous'] else block['harm_upper_95']}; per subject "
              f"{block['harm_upper_95_by_cluster']} over {block['clusters']})  {block['outcomes']}")
    every = summary["all"]
    print(f"ALL: {every['trials']} trials in {seconds}s; pass {every['pass']} ({every['pass_rate']}); acceptable "
          f"{every['acceptable']} ({every['acceptable_rate']}); harm {every['harm']} + mismatch {every['mismatch']} in "
          f"{every['answered']} answered "
          f"(95% upper {every['harm_upper_95']}); by subject (division, debate or day) {every['harm_clusters']} of "
          f"{every['clusters']} "
          f"(95% upper {every['harm_upper_95_by_cluster']}); disputed apart: {summary['disputed']['trials']}")
    for tier, block in summary["by_tier"].items():  # Canonical forms show little; quote the adversarial bound
        print(f"{tier.upper():12} {block['trials']:5} trials; harm {block['harm']} + mismatch {block['mismatch']} in "
              f"{block['answered']} answered (95% upper {block['harm_upper_95']}; by subject "
              f"{block['harm_upper_95_by_cluster']})")
    print(f"relations: {report['relations']}")
    print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 1 if every["harm"] or every["mismatch"] else 0


if __name__ == "__main__":
    sys.exit(main())
