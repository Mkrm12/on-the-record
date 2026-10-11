"""The maintainer's adjudication of the judge's contested verdicts: at most 20 cards, each graded blind before the
reveal.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import html
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from goldset import rubric as R  # noqa: E402
from goldset import triage as T  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402

import calibrate_judge as CJ  # noqa: E402
import e3_gate_report as EGR  # noqa: E402

TAKES = ROOT / "reports" / "triage_takes.json"
PAGE = ROOT / "tools" / "review" / "triage.html"
KEY = ROOT / "reports" / "triage_key.json"
RESULT = ROOT / "reports" / "triage_result.json"
STRATUM_LABEL = {"flagged": "contested in Claude's pre-read", "owner_named": "your named example",
                 "controls": "random control"}


class BuildRefused(Exception):
    """The inputs disagree with each other; nothing was written."""


def _read(path: Path | str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _consistent(outcome: str, present: int, total: int) -> bool:
    if outcome in ("correct", "correct_missing_okay_nuggets"):
        return present == total
    if outcome == "missing_vital_nugget":
        return present < total
    return 0 <= present <= total


def assemble(takes: dict) -> tuple[list[dict], dict, dict, dict]:
    """(blind cards in display order, reveal payload by id, the gold card(s), the key) — or BuildRefused."""
    packet, verdicts, pkey, report = (_read(takes[k]) for k in ("packet", "verdicts", "packet_key", "report"))
    ids = {takes["packet_id"], packet["packet_id"], verdicts["packet_id"], pkey["packet_id"],
           report["judged_offline"]["packet_id"]}
    if len(ids) != 1:
        raise BuildRefused(f"the takes, packet, verdicts, key and judged run name different packets: {sorted(ids)}")
    entries = {e["id"]: e for e in packet["entries"]}
    verdict = {v["id"]: v for v in verdicts["verdicts"]}
    rows = {r["id"]: r for r in report["rows"]}
    for jid, v in verdict.items():  # the run holds exactly what Gemini said (the import is the only writer)
        row = rows[pkey["map"][jid]]
        if row.get("by") != "gemini" or row.get("outcome") != v["outcome"]:
            raise BuildRefused(f"{jid}: the judged run says {row.get('outcome')!r} by {row.get('by')!r}, the verdicts "
                               f"say {v['outcome']!r}")
    gold_cards = takes["gold_cards"]
    gold_ids = [j for g in gold_cards.values() for j in g["ids"]]
    design = T.design(verdict, list(takes["flagged"]), list(takes["owner_named"]), gold_ids,
                      n_controls=takes["n_controls"], seed=takes["seed"], gold_cards=len(gold_cards))
    drawn = design["strata"]["controls"]["ids"]
    if drawn != sorted(takes["controls"]):
        raise BuildRefused(f"the seeded draw gives controls {drawn}; the takes were written for "
                           f"{sorted(takes['controls'])}")
    blind, reveal, key_cards = [], {}, {}
    for stratum in T.STRATA:
        source = takes[stratum]
        for jid in design["strata"][stratum]["ids"]:
            v, e, row = verdict[jid], entries[jid], rows[pkey["map"][jid]]
            judge = {"outcome": v["outcome"], "vital_present": v["vital_present"], "vital_total": v["vital_total"],
                     "why": v["why"]}
            claude = source[jid].get("claude") or {"outcome": v["outcome"], "vital_present": v["vital_present"]}
            if claude["outcome"] not in R.OUTCOMES or not _consistent(claude["outcome"], claude["vital_present"],
                                                                     v["vital_total"]):
                raise BuildRefused(f"{jid}: Claude's grade {claude} is not a consistent grade out of {v['vital_total']}")
            blind.append({k: e[k] for k in ("question", "expected", "reference_points", "vital_total",
                                            "must_not_claim", "answer_is_a_decline", "answer")} | {"jid": jid})
            reveal[jid] = {"item_id": row["id"], "category": row.get("category"), "criticality": row.get("criticality"),
                           "stratum": STRATUM_LABEL[stratum], "judge": judge, "claude": claude,
                           **{k: source[jid][k] for k in ("take", "record", "entry", "effect") if source[jid].get(k)}}
            key_cards[jid] = {"item_id": row["id"], "stratum": stratum, "judge": judge, "claude": claude}
    random.Random(takes["seed"] + 1).shuffle(blind)
    gold = []
    for name, card in gold_cards.items():
        items = [{"jid": j, "question": entries[j]["question"], "answer": entries[j]["answer"],
                  "wants": [p["text"] for p in entries[j]["reference_points"] if p["vital"]],
                  "record": card["record"][j], "judge": f"{verdict[j]['outcome']} ({verdict[j]['vital_present']} of "
                                                         f"{verdict[j]['vital_total']})"} for j in card["ids"]]  # fmt: skip
        gold.append({"name": name, "title": card["title"], "take": card["take"], "options": card["options"],
                     "effect": card["effect"], "items": items})
    key = {"page": T.PAGE_VERSION, "packet_id": packet["packet_id"], "report": Path(takes["report"]).name,
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": takes["seed"],
           **{k: design[k] for k in ("strata", "kappa_population", "excluded_from_kappa")},
           "cards": key_cards,
           "gold_cards": {name: {"ids": card["ids"], "item_ids": [pkey["map"][j] for j in card["ids"]]}
                          for name, card in gold_cards.items()}}
    return blind, reveal, gold, key


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge Triage</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6; --accent:#1f5f8b; --warn:#9b2c2c;
  --ok:#2f6b3a; --soft:#eef3f7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#161615; --card:#1f1f1e; --ink:#ececea;
  --muted:#a3a39c; --line:#34342f; --accent:#7fb3d9; --warn:#e08b8b; --ok:#8fcf9a; --soft:#22303a; } }
:root[data-theme="dark"] { --bg:#161615; --card:#1f1f1e; --ink:#ececea; --muted:#a3a39c; --line:#34342f;
  --accent:#7fb3d9; --warn:#e08b8b; --ok:#8fcf9a; --soft:#22303a; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui, sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
header { position:sticky; top:0; background:var(--bg); padding:10px 0; border-bottom:1px solid var(--line); z-index:1;
  display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:16px 0; }
.q { font-weight:600; font-size:16px; margin:6px 0; } .muted { color:var(--muted); font-size:13px; }
.vital { color:var(--accent); font-weight:600; } .wrong { color:var(--warn); } .ok { color:var(--ok); }
.answer { white-space:pre-wrap; border-left:3px solid var(--line); padding-left:12px; overflow-wrap:anywhere; }
.step { border-top:1px dashed var(--line); margin-top:14px; padding-top:12px; }
.reveal { background:var(--soft); border-radius:6px; padding:12px; margin-top:12px; }
.tag { display:inline-block; font-size:12px; border:1px solid var(--line); border-radius:999px; padding:0 8px;
  margin:0 6px 4px 0; }
label { display:block; padding:3px 0; cursor:pointer; } textarea { width:100%; box-sizing:border-box; min-height:38px;
  font:inherit; } select { font:inherit; max-width:100%; }
button { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:8px 14px; cursor:pointer;
  font:inherit; } button:disabled { opacity:.45; cursor:not-allowed; }
</style></head><body><main>
<header><strong>Judge triage</strong><span id="progress" class="muted"></span>
<button id="download" type="button">Download labels</button></header>
<section class="card"><p><strong>__N__ cards, about a minute each.</strong> Each card shows exactly what Gemini saw.</p>
<ol><li><b>Grade it yourself first</b>, against the entry as written (rule 1), even if you think the entry is wrong.
This blind grade is what kappa is computed from, and it locks when you reveal.</li>
<li><b>Reveal</b> to see Gemini's verdict and Claude's take, with anything read from the record.</li>
<li><b>Choose</b>: [A] Gemini, [B] Claude, or your own grade. This decides the scorecard. Tick "the entry needs
fixing" when the gold entry itself is wrong.</li></ol>
<p class="muted">The last card is one decision about five gold entries, not about the judge, so it has no blind step and
is kept out of kappa. Your progress is kept in this browser; download the labels when you are done.</p></section>
<details class="muted"><summary>The outcomes and the rules — the judge's, word for word</summary>
<dl>__DEFINITIONS__</dl><ol>__RULES__</ol></details>
<div id="cards"></div></main>
<script>
const CARDS = __CARDS__;
const GOLD = __GOLD__;
const OUTCOMES = __OUTCOMES__;
const META = __META__;
const REVEAL = "__REVEAL__";
const STORE = "triage-" + META.packet_id;
let state = {cards: {}, gold: {}};
try { state = JSON.parse(localStorage.getItem(STORE)) || state; } catch (e) {}
state.cards = state.cards || {}; state.gold = state.gold || {};
function save() { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) {} progress(); }
let opened = null;
function reveal(jid) {
  if (!opened) opened = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(REVEAL), c => c.charCodeAt(0))));
  return opened[jid];
}
const label = v => (OUTCOMES.find(o => o[0] === v) || [v, v])[1];
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined) e.append(k);
  return e;
}
function para(strong, text, cls) { return el("p", cls ? {class: cls} : {}, el("strong", {text: strong}), " " + text); }
function radio(name, value, text, checked, onchange, disabled) {
  const i = el("input", {type: "radio", name: name, value: value});
  i.checked = !!checked; i.disabled = !!disabled; i.addEventListener("change", onchange);
  return el("label", {}, i, " " + text);
}
function progress() {
  const blind = CARDS.filter(c => (state.cards[c.jid] || {}).revealed_at).length;
  const decided = CARDS.filter(c => (state.cards[c.jid] || {}).choice).length;
  const gold = GOLD.filter(g => (state.gold[g.name] || {}).choice).length;
  const done = blind === CARDS.length && decided === CARDS.length && gold === GOLD.length;
  const p = document.getElementById("progress");
  p.textContent = `${blind} of ${CARDS.length} graded blind · ${decided} of ${CARDS.length} decided · ` +
    `${gold} of ${GOLD.length} entry decision` + (done ? " · all done, download the labels" : "");
  p.className = done ? "ok" : "muted";
}
function verdictStep(it, c) {
  const r = reveal(it.jid), j = r.judge, k = r.claude, total = j.vital_total;
  const same = k.outcome === j.outcome && k.vital_present === j.vital_present;
  const box = el("div", {class: "reveal"});
  box.append(el("div", {}, ...[r.item_id, r.category, r.criticality, r.stratum].filter(Boolean)
    .map(t => el("span", {class: "tag", text: t}))));
  box.append(para("Gemini:", `${label(j.outcome)} · ${j.vital_present} of ${total} vital. ${j.why}`));
  box.append(para("Claude's take:", r.take));
  if (r.record) box.append(para("From the record:", r.record, "muted"));
  if (r.entry) box.append(para("The entry:", r.entry, "muted"));
  if (r.effect) box.append(para(same ? "Effect:" : "If you choose [B]:", r.effect, "muted"));
  const step = el("div", {class: "step"}, el("strong", {text: "3 · Your verdict"}));
  const name = "final-" + it.jid;
  const pick = (choice, outcome, vital) => () => { c.choice = choice; c.final = outcome; c.final_vital = vital; save(); };
  step.append(radio(name, "A", same ? `[A] Gemini and Claude agree: ${label(j.outcome)} (${j.vital_present} of ${total})`
    : `[A] Gemini: ${label(j.outcome)} (${j.vital_present} of ${total})`, c.choice === "A",
    pick("A", j.outcome, j.vital_present)));
  if (!same) step.append(radio(name, "B", `[B] Claude: ${label(k.outcome)} (${k.vital_present} of ${total})`,
    c.choice === "B", pick("B", k.outcome, k.vital_present)));
  const out = el("select", {}), vit = el("select", {});
  for (const [v, t] of OUTCOMES) out.append(el("option", {value: v, text: t}));
  for (let n = 0; n <= total; n++) vit.append(el("option", {value: String(n), text: `${n} of ${total} vital`}));
  out.value = c.choice === "C" ? c.final : (c.blind || j.outcome);
  vit.value = String(c.choice === "C" && Number.isInteger(c.final_vital) ? c.final_vital : j.vital_present);
  const own = () => { c.choice = "C"; c.final = out.value; c.final_vital = Number(vit.value); save(); };
  const own_radio = radio(name, "C", same ? "[B] Neither — my grade:" : "[C] Neither — my grade:", c.choice === "C", own);
  own_radio.append(" ", out, " ", vit);
  out.addEventListener("change", () => { own_radio.querySelector("input").checked = true; own(); });
  vit.addEventListener("change", () => { own_radio.querySelector("input").checked = true; own(); });
  step.append(own_radio);
  const wrong = el("input", {type: "checkbox"}); wrong.checked = !!c.entry_wrong;
  wrong.addEventListener("change", () => { c.entry_wrong = wrong.checked; save(); });
  const note = el("textarea", {placeholder: "optional note"}); note.value = c.note || "";
  note.addEventListener("input", () => { c.note = note.value; save(); });
  step.append(el("label", {}, wrong, " The entry itself needs fixing"), note);
  box.append(step);
  return box;
}
function card(it, n) {
  const c = state.cards[it.jid] || (state.cards[it.jid] = {});
  const box = el("section", {class: "card", id: it.jid});
  box.append(el("div", {class: "muted", text: `Card ${n} of ${CARDS.length + GOLD.length} · ${it.jid}`}),
    el("p", {class: "q", text: it.question}), el("p", {class: "muted", text: "A right response: " + it.expected}));
  const refs = el("ul");
  for (const p of it.reference_points) refs.append(el("li", {class: p.vital ? "vital" : "",
    text: (p.vital ? "VITAL: " : "optional: ") + p.text}));
  box.append(el("p", {class: "muted", text: `Reference points (vital_total ${it.vital_total}):`}), refs);
  if (it.must_not_claim.length) {
    const w = el("ul", {class: "wrong"});
    for (const m of it.must_not_claim) w.append(el("li", {text: m}));
    box.append(el("p", {class: "muted", text: "Wrong if asserted:"}), w);
  }
  box.append(el("p", {class: "muted", text: it.answer_is_a_decline ? "What the reader saw (a decline):"
    : "What the reader saw:"}), el("div", {class: "answer", text: it.answer}));
  const step = el("div", {class: "step"}, el("strong", {text: "1 · Your grade, blind "}),
    el("span", {class: "muted", text: "(against the entry as written)"}));
  const opts = el("div");
  for (const [v, t] of OUTCOMES) opts.append(radio("blind-" + it.jid, v, t, c.blind === v,
    () => { c.blind = v; c.blind_at = new Date().toISOString(); btn.disabled = false; save(); }, !!c.revealed_at));
  const btn = el("button", {type: "button", text: c.revealed_at ? "Locked" : "2 · Lock my grade and reveal"});
  btn.disabled = !c.blind || !!c.revealed_at;
  const slot = el("div");
  btn.addEventListener("click", () => {
    if (!c.blind || c.revealed_at) return;
    c.revealed_at = new Date().toISOString();
    opts.querySelectorAll("input").forEach(i => { i.disabled = true; });
    btn.disabled = true; btn.textContent = "Locked"; save(); slot.replaceChildren(verdictStep(it, c));
  });
  step.append(opts, btn); box.append(step, slot);
  if (c.revealed_at) slot.append(verdictStep(it, c));
  return box;
}
function goldCard(g, n) {
  const d = state.gold[g.name] || (state.gold[g.name] = {});
  const box = el("section", {class: "card", id: "gold-" + g.name});
  box.append(el("div", {class: "muted", text: `Card ${n} of ${CARDS.length + GOLD.length} · about the entries, ` +
    "not the judge (kept out of kappa)"}), el("p", {class: "q", text: g.title}), para("Claude's take:", g.take));
  for (const it of g.items) {
    const sub = el("div", {class: "step"});
    sub.append(el("p", {}, el("strong", {text: `${it.jid} · `}), it.question),
      para("The entry wants:", it.wants.join("; "), "muted"), para("The record that day:", it.record, "muted"),
      para("Gemini:", it.judge, "muted"), el("div", {class: "answer muted", text: it.answer}));
    box.append(sub);
  }
  const step = el("div", {class: "step"}, el("strong", {text: "Your decision"}));
  for (const [k, t] of Object.entries(g.options)) step.append(radio("gold-" + g.name, k, `[${k}] ${t}`,
    d.choice === k, () => { d.choice = k; save(); }));
  const note = el("textarea", {placeholder: "optional note"}); note.value = d.note || "";
  note.addEventListener("input", () => { d.note = note.value; save(); });
  step.append(el("p", {class: "muted", text: g.effect}), note);
  box.append(step);
  return box;
}
const root = document.getElementById("cards");
CARDS.forEach((it, i) => root.append(card(it, i + 1)));
GOLD.forEach((g, i) => root.append(goldCard(g, CARDS.length + i + 1)));
document.getElementById("download").onclick = () => {
  const out = {page: META.page, packet_id: META.packet_id, built_at: META.built_at,
    downloaded_at: new Date().toISOString(), cards: state.cards, gold: state.gold};
  const a = el("a", {href: URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"})),
    download: "triage_labels.json"});
  document.body.append(a); a.click(); a.remove();
};
progress();
</script></body></html>
"""


def _script_json(value) -> str:
    """JSON safe inside a <script> element: no '</' can close it early."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def page_html(blind: list[dict], reveal: dict, gold: list[dict], meta: dict) -> str:
    """The page."""
    definitions = "".join(f"<dt><b>{html.escape(CJ.LABELS[o])}</b></dt><dd>{html.escape(R.OUTCOME_DEFINITIONS[o], quote=False)}"
                          f"</dd>" for o in R.OUTCOMES)  # fmt: skip
    rules = "".join(f"<li>{html.escape(rule, quote=False)}</li>" for rule in R.GRADING_RULES)
    encoded = base64.b64encode(json.dumps(reveal, ensure_ascii=False).encode("utf-8")).decode("ascii")
    out = _TEMPLATE
    for token, value in (("__DEFINITIONS__", definitions), ("__RULES__", rules), ("__N__", str(len(blind) + len(gold))),
                         ("__OUTCOMES__", _script_json([[o, CJ.LABELS[o]] for o in R.OUTCOMES])),
                         ("__META__", _script_json(meta)), ("__GOLD__", _script_json(gold)),
                         ("__REVEAL__", encoded), ("__CARDS__", _script_json(blind))):
        out = out.replace(token, value)
    return out


def build(takes_path: Path = TAKES) -> int:
    raw = takes_path.read_bytes()
    takes = json.loads(raw.decode("utf-8"))
    try:
        blind, reveal, gold, key = assemble(takes)
    except (BuildRefused, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    key["takes_sha256"] = hashlib.sha256(raw).hexdigest()[:16]  # which takes this build argued from
    meta = {"page": key["page"], "packet_id": key["packet_id"], "built_at": key["built_at"]}
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page_html(blind, reveal, gold, meta), encoding="utf-8")
    KEY.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    s = key["strata"]
    print(f"{len(blind) + len(gold)} cards: {len(s['flagged']['ids'])} contested, {len(s['owner_named']['ids'])} you "
          f"named, {len(s['controls']['ids'])} random controls (of {s['controls']['population']}, weight "
          f"{s['controls']['weight']}), {len(gold)} entry decision ({len(key['excluded_from_kappa'])} rows, out of kappa)")
    print(f"wrote {PAGE.relative_to(ROOT)} (open it in a browser) and {KEY.relative_to(ROOT)}")
    return 0


def scorecard(result: dict, key: dict) -> list[tuple[str, str, str, str]]:
    """The bars with the maintainer's decisions applied (`e3_gate_report.bars` on an adjudicated copy of the run).
    """
    report = json.loads((ROOT / "reports" / key["report"]).read_text(encoding="utf-8"))
    gold = {i.id: i for i in load_gold(ROOT / "tests" / "data" / "gold_v2.json")}
    rows, clarify = T.adjudicate_rows(report["rows"], key, result)
    for item_id in clarify:
        gold[item_id] = dataclasses.replace(gold[item_id], expected_behaviour="clarify_ambiguous")
    return EGR.bars({**report, "rows": rows}, gold)


def score(labels_path: Path) -> int:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    key = json.loads(KEY.read_text(encoding="utf-8"))
    try:
        result = T.score(labels, key)
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 1
    table = scorecard(result, key) if result["complete"] else []
    RESULT.write_text(json.dumps({**result, "scorecard": table, "labels": labels_path.name}, indent=1,
                                 ensure_ascii=False) + "\n", encoding="utf-8")  # fmt: skip
    k, w = result["kappa_on_sample"], result["kappa_weighted_to_run"]
    print(f"{result['cards_graded_blind']} of {result['cards']} cards graded blind, {result['cards_decided']} decided"
          f"{'' if result['complete'] else ' — INCOMPLETE: no verdict and no scorecard until every card is decided'}")
    print(f"kappa on the sample (n {k['n']}, chosen for disagreement): {k['kappa']} (agreement {k['agreement']})")
    print(f"kappa weighted to the run ({w['population']} judged rows): {w['kappa']} (agreement {w['agreement']}), "
          f"95% interval {w['interval_95']}")
    h = result["harmful"]
    print(f"harmful: you {len(h['owner_blind'])}, Gemini {len(h['judge'])}, caught {len(h['caught'])} -> recall "
          f"{h['recall']}. {h['note']}")
    flips = result["reveal_changed_the_grade"]
    print(f"the reveal changed your grade on {len(flips['cards'])} card(s) {flips['cards']}")
    for name, target, value, verdict in table:
        print(f"  {name:<50} {target:<22} {verdict:<16} {value}")
    verdict = "PASSES" if result["passes"] else "DOES NOT PASS"
    print(f"calibration: {verdict} (kappa weighted >= {T.CAL.KAPPA_BAR}, harmful recall >= "
          f"{T.CAL.HARMFUL_RECALL_BAR}); written to {RESULT.relative_to(ROOT)}")
    return 0 if result["passes"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    s = sub.add_parser("score")
    s.add_argument("--labels", required=True)
    args = p.parse_args(argv)
    return build() if args.command == "build" else score(Path(args.labels).resolve())


if __name__ == "__main__":
    sys.exit(main())
