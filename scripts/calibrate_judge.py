"""Calibrate the answer judge against the maintainer — the one E3 step that needs a person."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goldset import calibration as CAL  # noqa: E402
from goldset import rubric as R  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402

PAGE = ROOT / "tools" / "review" / "calibration.html"
KEY = ROOT / "reports" / "calibration_key.json"
RESULT = ROOT / "reports" / "calibration_result.json"

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge Calibration</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6; --accent:#1f5f8b; --warn:#9b2c2c; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161615; --card:#1f1f1e; --ink:#ececea;
  --muted:#a3a39c; --line:#34342f; --accent:#7fb3d9; --warn:#e08b8b; } }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
header { position:sticky; top:0; background:var(--bg); padding:12px 0; border-bottom:1px solid var(--line); z-index:1; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:16px 0; }
.q { font-weight:600; } .muted { color:var(--muted); font-size:13px; } .vital { color:var(--accent); font-weight:600; }
.wrong { color:var(--warn); } .answer { white-space:pre-wrap; border-left:3px solid var(--line); padding-left:12px; }
label { display:block; padding:4px 0; cursor:pointer; } textarea { width:100%; box-sizing:border-box; min-height:40px; }
button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 14px; cursor:pointer; }
</style></head><body><main>
<header><strong>Judge calibration</strong> — label each answer against its reference points only.
<span id="progress" class="muted"></span> <button id="download">Download labels</button></header>
<details class="muted" open><summary>The outcomes and the rules — the judge's, word for word</summary>
<dl>__DEFINITIONS__</dl><ol>__RULES__</ol></details>
<div id="items"></div></main>
<script>
const ITEMS = __ITEMS__;
const OUTCOMES = __OUTCOMES__;
const STORE = "calibration-labels-v1";
let saved = {labels:{}, notes:{}};
try { saved = JSON.parse(localStorage.getItem(STORE)) || saved; } catch (e) {}
function persist() { try { localStorage.setItem(STORE, JSON.stringify(saved)); } catch (e) {} update(); }
function update() { document.getElementById("progress").textContent =
  ` ${Object.keys(saved.labels).length} of ${ITEMS.length} labelled`; }
const root = document.getElementById("items");
for (const it of ITEMS) {
  const card = document.createElement("div"); card.className = "card";
  const refs = it.reference_points.map(r => `<li class="${r.vital ? "vital" : ""}">${r.vital ? "VITAL: " : ""}${r.text}</li>`).join("");
  const wrong = it.wrong_if_asserted.length ? `<p class="muted">Wrong if asserted:</p><ul class="wrong">${it.wrong_if_asserted.map(w => `<li>${w}</li>`).join("")}</ul>` : "";
  card.innerHTML = `<div class="muted">${it.cid}</div><p class="q">${it.question}</p>
    <p class="muted">A right response: ${it.expected}</p>
    <p class="muted">Reference points:</p><ul>${refs}</ul>${wrong}
    <p class="muted">The answer:</p><div class="answer">${it.answer || "<i>(nothing was shown)</i>"}</div>
    <fieldset style="border:0;padding:0;margin-top:12px">${OUTCOMES.map(([v, t]) =>
      `<label><input type="radio" name="${it.cid}" value="${v}"> ${t}</label>`).join("")}</fieldset>
    <textarea placeholder="optional note" data-cid="${it.cid}"></textarea>`;
  root.appendChild(card);
  const chosen = saved.labels[it.cid];
  if (chosen) card.querySelector(`input[value="${chosen}"]`).checked = true;
  card.querySelector("textarea").value = saved.notes[it.cid] || "";
  card.querySelectorAll("input").forEach(i => i.addEventListener("change", e => { saved.labels[it.cid] = e.target.value; persist(); }));
  card.querySelector("textarea").addEventListener("input", e => { saved.notes[it.cid] = e.target.value; persist(); });
}
document.getElementById("download").onclick = () => {
  const blob = new Blob([JSON.stringify(saved, null, 1)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "calibration_labels.json"; a.click();
};
update();
</script></body></html>
"""


def _escape(items: list[dict]) -> list[dict]:
    """Every string the page shows, HTML-escaped: answers and questions are data, never markup."""
    def esc(value):
        if isinstance(value, str):
            return html.escape(value)
        if isinstance(value, list):
            return [esc(v) for v in value]
        if isinstance(value, dict):
            return {k: esc(v) for k, v in value.items()}
        return value
    return [esc(i) for i in items]


# The radio buttons' words; the definitions beside them are the rubric's own.
LABELS = {"correct": "correct", "correct_missing_okay_nuggets": "correct, optional points missing",
          "missing_vital_nugget": "missing a vital point", "correct_abstention": "right to decline (or ask which)",
          "over_abstention": "declined, should have answered", "wrong_but_benign": "wrong but benign",
          "harmful": "harmful"}  # fmt: skip


def page_html(items: list[dict]) -> str:
    """The labelling page: the judge's outcomes, definitions and rules (`goldset.rubric`), word for word."""
    definitions = "".join(f"<dt><b>{html.escape(LABELS[o])}</b></dt><dd>{html.escape(R.OUTCOME_DEFINITIONS[o], quote=False)}"
                          f"</dd>" for o in R.OUTCOMES)  # fmt: skip
    rules = "".join(f"<li>{html.escape(rule, quote=False)}</li>" for rule in R.GRADING_RULES)
    outcomes = json.dumps([[o, LABELS[o]] for o in R.OUTCOMES])
    return (_TEMPLATE.replace("__DEFINITIONS__", definitions).replace("__RULES__", rules)
            .replace("__OUTCOMES__", outcomes).replace("__ITEMS__", json.dumps(_escape(items), ensure_ascii=False)))


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build(report_path: Path, n: int, seed: int) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gold = {i.id: i for i in load_gold(ROOT / "tests" / "data" / "gold_v2.json")}
    judged = CAL.judged_rows(report["rows"])
    chosen = CAL.select(judged, n=n, seed=seed)
    items, key = CAL.worksheet(chosen, gold, seed=seed)
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page_html(items), encoding="utf-8", newline="\n")
    judge_harmful = sum(1 for r in judged if r["outcome"] == "harmful")
    # Harmful with no judge involved (a forbidden claim): never in the packet, and part of the run's harmful count.
    judged_ids = {r["id"] for r in judged}
    deterministic_harmful = sorted(r["id"] for r in report["rows"]
                                   if r.get("outcome") == "harmful" and r["id"] not in judged_ids)  # fmt: skip
    KEY.write_text(json.dumps({"report": report_path.name, "run_id": report.get("run_id"), "seed": seed,
                               "judge_harmful_in_run": judge_harmful, "deterministic_harmful": deterministic_harmful,
                               "key": key}, indent=1) + "\n", encoding="utf-8", newline="\n")  # fmt: skip
    strata = {o: sum(1 for r in chosen if r["outcome"] == o) for o in CAL.OWNER_OUTCOMES}
    print(f"{len(items)} answers from {report_path.name} ({len(judged)} judged in the run): {strata}")
    if deterministic_harmful:
        print(f"{len(deterministic_harmful)} answer(s) harmful without a judge (a forbidden claim; they stand): "
              f"{deterministic_harmful}")
    print(f"wrote {_shown(PAGE)} (open it in a browser) and {_shown(KEY)} (do not open while labelling)")
    return 0


def score(labels_path: Path) -> int:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    stored = json.loads(KEY.read_text(encoding="utf-8"))
    result = CAL.score(labels.get("labels", labels), stored["key"], stored.get("judge_harmful_in_run"),
                       deterministic_harmful=stored.get("deterministic_harmful", ()))
    RESULT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    k, h = result["kappa_outcomes"], result["harmful"]
    print(f"n {result['n']}: kappa (seven outcomes) {k['kappa']} (agreement {k['agreement']}); "
          f"harmful-or-not kappa {result['kappa_harmful_or_not']['kappa']}")
    print(f"harmful: owner {h['owner']}, judge {h['judge']}, caught {h['caught']} -> recall {h['recall']}, "
          f"precision {h['precision']}")
    print(f"verdict: {'PASSES' if result['passes'] else 'DOES NOT PASS'} (bars: kappa >= {CAL.KAPPA_BAR}, harmful "
          f"recall >= {CAL.HARMFUL_RECALL_BAR}). {result['note']}")
    adjudicated = result["adjudicated"]
    if adjudicated["every_judge_harmful_answer_labelled"]:
        after = adjudicated["harmful_after_owner_review"]
        print(f"harmful after the owner's review: {len(after)} {after} — the run's adjudicated harmful count")
    else:
        print(f"no adjudicated harmful count: {adjudicated['note']}")
    print(f"{len(result['disagreements'])} disagreement(s) in {RESULT.relative_to(ROOT)}")
    return 0 if result["passes"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--report", default=str(ROOT / "reports" / "e3_eval_dev.json"))
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--seed", type=int, default=18)
    s = sub.add_parser("score")
    s.add_argument("--labels", required=True)
    args = p.parse_args(argv)
    if args.command == "build":
        return build(Path(args.report).resolve(), args.n, args.seed)
    return score(Path(args.labels).resolve())


if __name__ == "__main__":
    sys.exit(main())
