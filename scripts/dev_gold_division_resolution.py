"""Dev only: does every dev division item still resolve to its gold division, and every vote item to its member?"""

import json
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

import psycopg  # noqa: E402

from answer import resolve as RS  # noqa: E402
from answer import router as RO  # noqa: E402
from app import safety, sealed  # noqa: E402
from app.config import settings  # noqa: E402
from ingest.runtime import run  # noqa: E402

gold = [i for i in json.load(open(__import__("pathlib").Path(__file__).resolve().parents[1] / "tests" / "data" / "gold_v2.json", encoding="utf-8"))["items"] if i["split"] == "dev"
        and i["category"] in ("division_result", "division_tally", "division_vote")]


async def main():
    out = Counter()
    misses = []
    async with await psycopg.AsyncConnection.connect(safety.working_dsn(settings.pg_dsn),
                                                     options="-c default_transaction_read_only=on") as conn:
        await conn.execute(sealed.serving_sql()); await conn.commit()
        for item in gold:
            expected = (item.get("expected_structured_facts") or {}).get("division_id")
            d = RO.route(item["question"])
            params, notes = await RS.parameters(conn, item["category"], item["question"], d.filters)
            got = params.get("division_id")
            if item["category"] == "division_vote":
                exp_m = (item.get("expected_structured_facts") or {}).get("mnis_id")
                out["vote member same" if params.get("mnis_id") == exp_m else "vote member differs"] += 1
            key = "same division" if got == expected else ("unresolved" if got is None else "a different division")
            out[key] += 1
            if got != expected:
                misses.append((item["id"], key, notes[0][:120] if notes else ""))
    print(f"{len(gold)} dev division items:", dict(out))
    for m in misses:
        print("  ", m)


run(main())
