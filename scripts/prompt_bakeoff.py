"""Compare the three answer prompts on a dev sample, and build a page for the maintainer to pick one."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import generate as G  # noqa: E402
from answer import pipeline as PL  # noqa: E402
from answer import prompts as P  # noqa: E402
from goldset.llm import estimate  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402

REPORTS = ROOT / "reports"
PAGE = ROOT / "tools" / "review" / "prompt_bakeoff.html"
DATA = REPORTS / "e3_prompt_bakeoff.json"

# Categories worth comparing register on: they produce prose rather than a single figure.
PROSE_CATEGORIES = ("paraphrase", "multi_speech", "stance", "comparison", "exact_term", "colloquial",
                    "named_speaker", "false_premise", "loaded")  # fmt: skip


def sample(limit: int) -> list:
    """One item per category, in a fixed order, so two runs compare the same questions."""
    items = [i for i in load_gold() if i.split == "dev" and i.category in PROSE_CATEGORIES]
    by_category: dict[str, list] = {}
    for item in sorted(items, key=lambda i: i.id):
        by_category.setdefault(i_cat(item), []).append(item)
    picked: list = []
    for category in PROSE_CATEGORIES:
        if category in by_category and len(picked) < limit:
            picked.append(by_category[category][0])
    for category in PROSE_CATEGORIES:  # a second pass if one per category was not enough
        for item in by_category.get(category, [])[1:]:
            if len(picked) >= limit:
                break
            picked.append(item)
    return picked[:limit]


def i_cat(item) -> str:
    return getattr(item, "category", "")


async def run(items: list, warm: bool) -> dict:
    runtime, pool = await PL.build_runtime(with_verifier=True)
    rows: list[dict] = []
    try:
        if warm:
            # One throwaway pair so MiniCheck is resident before anything is timed.
            try:
                started = time.perf_counter()
                await runtime.verifier.verify([("The House rose at six.", "The House rose at six.")])
                print(f"verifier warmed in {time.perf_counter() - started:.1f}s")
            except Exception as exc:  # noqa: BLE001
                print(f"could not warm the verifier ({type(exc).__name__}); continuing degraded")

        for index, item in enumerate(items, start=1):
            # One item's retrieval failing must not cost the whole run and the money already spent on it.
            deck = result = None
            for attempt in (1, 2):
                try:
                    deck, result = await PL.search_deck(runtime, item.question)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[{index}/{len(items)}] {item.id} retrieval attempt {attempt} failed: "
                          f"{type(exc).__name__}")  # fmt: skip
            if deck is None:
                rows.append({"id": item.id, "category": i_cat(item), "question": item.question,
                             "cards": 0, "hits": 0, "evidence": [], "variants": {},
                             "error": "retrieval failed twice"})  # fmt: skip
                continue
            print(f"[{index}/{len(items)}] {item.id} {i_cat(item):14s} {len(deck.cards)} cards  {item.question[:64]}")
            per_variant: dict[str, dict] = {}
            for variant in P.VARIANTS:
                generated = await G.answer_question(
                    item.question, deck, runtime.client,
                    runtime.provenance("retrieval", {}, variant.version),
                    variant_key=variant.key, score_batch=runtime.score_batch(),
                    budget=G.Budget(), verifier_name=runtime.verifier_name,
                    precision=runtime.verifier_precision,
                )  # fmt: skip
                answer = generated.answer
                per_variant[variant.key] = {
                    "name": variant.name, "blurb": variant.blurb,
                    "abstained": answer.abstention.value if answer.abstention else None,
                    "abstention_detail": answer.abstention_detail,
                    "rendered": answer.rendered,
                    "coverage_note": answer.coverage_note,
                    "claims": [
                        {"kind": c.kind, "text": c.text, "speaker": c.speaker_name,
                         "date": str(c.date) if c.date else None,
                         "quote": c.citations[0].quote, "chunk_ids": list(c.citations[0].chunk_ids),
                         "entailment": c.verification.entailment_score}  # fmt: skip
                        for c in answer.claims
                    ],
                    "generated": generated.claims_generated, "rejected": generated.claims_rejected,
                    "verified": generated.claims_verified, "dropped": generated.claims_dropped,
                    "repaired": generated.repaired, "fell_back": generated.fell_back,
                    "degraded": generated.degraded,
                    "usd": round(generated.budget.spent_usd, 6) if generated.budget else 0.0,
                    "seconds": round(generated.budget.elapsed, 2) if generated.budget else 0.0,
                    "rejections": [r.why for r in generated.rejections[:4]],
                    "drops": [{"text": text, "why": why} for text, why in generated.drops[:8]],
                }
                print(f"      {variant.key:7s} verified {generated.claims_verified} "
                      f"dropped {generated.claims_dropped} rej {generated.claims_rejected} "
                      f"{'ABSTAIN' if answer.abstention else 'answer '} "
                      f"${per_variant[variant.key]['usd']:.5f}")
            rows.append({
                "id": item.id, "category": i_cat(item), "question": item.question,
                "cards": len(deck.cards), "hits": len(result.hits),
                "evidence": [{"speech_id": c.speech_id, "speaker": c.speaker,
                              "date": str(c.date) if c.date else None, "heading": c.heading,
                              "text": c.text}  # fmt: skip
                             for c in deck.cards[:6]],
                "variants": per_variant,
            })
    finally:
        await pool.__aexit__(None, None, None)

    answered = [r for r in rows if r.get("variants")]
    totals = {
        v.key: {
            "verified": sum(r["variants"][v.key]["verified"] for r in answered),
            "dropped": sum(r["variants"][v.key]["dropped"] for r in answered),
            "rejected": sum(r["variants"][v.key]["rejected"] for r in answered),
            "abstained": sum(1 for r in answered if r["variants"][v.key]["abstained"]),
            "fell_back": sum(1 for r in answered if r["variants"][v.key]["fell_back"]),
            "usd": round(sum(r["variants"][v.key]["usd"] for r in answered), 5),
            "seconds": round(sum(r["variants"][v.key]["seconds"] for r in answered) / max(1, len(answered)), 2),
            "words": round(sum(len((r["variants"][v.key]["rendered"] or "").split()) for r in answered)
                           / max(1, len(answered)), 1),  # fmt: skip
        }
        for v in P.VARIANTS
    }
    return {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": "dev", "items": len(answered),
        "failed_items": [r["id"] for r in rows if not r.get("variants")],
        "prompt_version": P.PROMPT_VERSION, "model": runtime.model_id,
        "verifier": runtime.verifier_name, "precision": runtime.verifier_precision,
        "retrieval_config_hash": runtime.frozen.config_hash,
        "totals": totals, "rows": rows,
        "total_usd": round(sum(t["usd"] for t in totals.values()), 5),
    }


# --- the page -------------------------------------------------------------------------------------------

_CSS = """
:root { --ink:#1a1a1a; --muted:#666; --line:#e2e2e2; --bg:#fafafa; --card:#fff; --pick:#0a7d32; }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
       background:var(--bg); }
header { padding:18px 22px; background:var(--card); border-bottom:1px solid var(--line); position:sticky; top:0;
         z-index:5; }
h1 { margin:0 0 4px; font-size:19px; }
.meta { color:var(--muted); font-size:13px; }
main { padding:22px; max-width:1500px; margin:0 auto; }
.q { background:var(--card); border:1px solid var(--line); border-radius:10px; margin-bottom:26px;
     overflow:hidden; }
.qh { padding:14px 18px; border-bottom:1px solid var(--line); }
.qh .cat { display:inline-block; font-size:11px; text-transform:uppercase; letter-spacing:.06em;
           color:var(--muted); border:1px solid var(--line); border-radius:20px; padding:2px 9px;
           margin-right:8px; }
.qh .txt { font-weight:600; font-size:16px; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:0; }
.v { padding:16px 18px; border-right:1px solid var(--line); }
.v:last-child { border-right:0; }
.v h3 { margin:0 0 2px; font-size:14px; }
.v .blurb { color:var(--muted); font-size:12px; margin-bottom:10px; }
.answer { white-space:pre-wrap; }
.abstain { color:#8a4b00; background:#fff6e8; border:1px solid #f0d9b5; border-radius:8px; padding:10px; }
.stats { margin-top:12px; font-size:12px; color:var(--muted); }
.stats b { color:var(--ink); }
.pickrow { padding:12px 18px; border-top:1px solid var(--line); background:#fcfcfc; }
label.pick { display:inline-flex; align-items:center; gap:6px; margin-right:18px; font-size:14px;
             cursor:pointer; }
label.pick.on { color:var(--pick); font-weight:600; }
details { margin-top:12px; }
summary { cursor:pointer; color:var(--muted); font-size:13px; }
.ev { font-size:13px; border-top:1px solid var(--line); padding:10px 0; }
.ev .who { color:var(--muted); font-size:12px; margin-bottom:3px; }
.tally { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px;
         margin-bottom:22px; }
table { border-collapse:collapse; font-size:13px; width:100%; }
th,td { text-align:left; padding:5px 12px 5px 0; border-bottom:1px solid var(--line); }
#out { position:fixed; right:18px; bottom:18px; background:var(--card); border:1px solid var(--line);
       border-radius:10px; padding:12px 16px; box-shadow:0 2px 14px rgba(0,0,0,.09); font-size:13px; }
#out b { font-size:15px; }
button { font:inherit; padding:7px 13px; border:1px solid var(--line); border-radius:7px; background:#fff;
         cursor:pointer; }
@media (max-width:1000px) { .grid { grid-template-columns:1fr; } .v { border-right:0;
                            border-bottom:1px solid var(--line); } }
"""

_JS = """
const picks = {};
function choose(id, key, el) {
  picks[id] = key;
  el.closest('.pickrow').querySelectorAll('label.pick').forEach(l => l.classList.remove('on'));
  el.closest('label').classList.add('on');
  render();
}
function render() {
  const counts = {};
  Object.values(picks).forEach(k => counts[k] = (counts[k] || 0) + 1);
  const total = Object.keys(picks).length;
  const parts = Object.entries(counts).sort((a,b) => b[1]-a[1]).map(([k,n]) => k + ' ' + n);
  document.getElementById('tally').innerHTML =
    total ? '<b>' + parts.join(' &middot; ') + '</b><br>' + total + ' of ' + QUESTIONS + ' chosen'
          : 'pick the clearest answer for each question';
}
function copy() {
  const winner = Object.entries(picks).reduce((acc,[,k]) => (acc[k]=(acc[k]||0)+1, acc), {});
  navigator.clipboard.writeText(JSON.stringify({picks, winner}, null, 2));
  document.getElementById('copied').textContent = 'copied';
}
"""


def page(data: dict) -> str:
    rows = data["rows"]
    out: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Prompt bake-off</title><style>", _CSS, "</style></head><body>",
        "<header><h1>Which answer is clearest?</h1>",
        f"<div class='meta'>{data['items']} dev questions &middot; model <code>{html.escape(data['model'])}"
        f"</code> &middot; prompt <code>{html.escape(data['prompt_version'])}</code> &middot; "
        f"retrieval <code>{html.escape(data['retrieval_config_hash'])}</code> &middot; "
        f"${data['total_usd']:.4f} &middot; {html.escape(data['ran_at'])}</div>",
        "<div class='meta'>All three read the <b>same</b> evidence and are checked by the same verifier. "
        "Only the writing style differs. Pick whichever you would actually want to read.</div></header><main>",
    ]

    out.append("<div class='tally'><table><tr><th>variant</th><th>verified claims</th><th>dropped</th>"
               "<th>rejected</th><th>abstained</th><th>fell back</th><th>avg words</th><th>avg s</th>"
               "<th>cost</th></tr>")
    for variant in P.VARIANTS:
        t = data["totals"][variant.key]
        out.append(f"<tr><td><b>{html.escape(variant.name)}</b></td><td>{t['verified']}</td>"
                   f"<td>{t['dropped']}</td><td>{t['rejected']}</td><td>{t['abstained']}</td>"
                   f"<td>{t['fell_back']}</td><td>{t['words']}</td><td>{t['seconds']}</td>"
                   f"<td>${t['usd']:.4f}</td></tr>")
    out.append("</table><div class='meta' style='margin-top:8px'>These numbers say whether a prompt produces "
               "<i>checkable</i> output. They cannot say whether it is readable — that is the judgement below."
               "</div></div>")

    for row in rows:
        if not row.get('variants'):
            continue
        out.append("<div class='q'><div class='qh'>"
                   f"<span class='cat'>{html.escape(row['category'])}</span>"
                   f"<span class='txt'>{html.escape(row['question'])}</span>"
                   f"<div class='meta' style='margin-top:6px'>{row['cards']} evidence cards from "
                   f"{row['hits']} hits</div></div><div class='grid'>")
        for variant in P.VARIANTS:
            v = row["variants"][variant.key]
            out.append(f"<div class='v'><h3>{html.escape(variant.name)}</h3>"
                       f"<div class='blurb'>{html.escape(variant.blurb)}</div>")
            if v["abstained"]:
                out.append(f"<div class='abstain'><b>Did not answer</b> — {html.escape(v['abstained'])}<br>"
                           f"{html.escape(v['abstention_detail'] or '')}</div>")
            else:
                out.append(f"<div class='answer'>{html.escape(v['rendered'] or '(nothing)')}</div>")
            entail = [c["entailment"] for c in v["claims"] if c["entailment"] is not None]
            out.append(f"<div class='stats'>verified <b>{v['verified']}</b> &middot; dropped {v['dropped']}"
                       f" &middot; rejected {v['rejected']}"
                       + (" &middot; <b>repaired</b>" if v["repaired"] else "")
                       + (" &middot; <b>quotes only</b>" if v["fell_back"] else "")
                       + (f" &middot; entailment {min(entail):.2f}–{max(entail):.2f}" if entail else "")
                       + f" &middot; {v['seconds']}s</div>")
            if v["claims"]:
                out.append("<details><summary>claims and quotes</summary>")
                for claim in v["claims"]:
                    out.append(f"<div class='ev'><div class='who'>{html.escape(claim['kind'])} &middot; "
                               f"{html.escape(claim['speaker'] or '?')} &middot; "
                               f"{html.escape(claim['date'] or '')} &middot; "
                               f"chunks {claim['chunk_ids']}</div>"
                               f"{html.escape(claim['text'])}<br>"
                               f"<i>&ldquo;{html.escape(claim['quote'])}&rdquo;</i></div>")
                out.append("</details>")
            if v.get("drops"):
                out.append("<details><summary>dropped by the verifier</summary>"
                           + "".join(f"<div class='ev'><div class='who'>{html.escape(d['why'])}</div>"
                                     f"{html.escape(d['text'])}</div>" for d in v["drops"])
                           + "</details>")
            if v["rejections"]:
                out.append("<details><summary>rejected by the contract</summary>"
                           + "".join(f"<div class='ev'>{html.escape(r)}</div>" for r in v["rejections"])
                           + "</details>")
            out.append("</div>")
        out.append("</div><div class='pickrow'>Clearest: ")
        for variant in P.VARIANTS:
            out.append(f"<label class='pick'><input type='radio' name='p-{html.escape(row['id'])}' "
                       f"onchange=\"choose('{html.escape(row['id'])}','{variant.key}',this)\">"
                       f"{html.escape(variant.name)}</label>")
        out.append("<label class='pick'><input type='radio' name='p-" + html.escape(row["id"]) +
                   f"' onchange=\"choose('{html.escape(row['id'])}','none',this)\">none of them</label>")
        out.append("</div><details style='padding:0 18px 14px'><summary>the evidence all three read</summary>")
        for card in row["evidence"]:
            out.append(f"<div class='ev'><div class='who'>{html.escape(card['speaker'] or '?')} &middot; "
                       f"{html.escape(card['date'] or '')} &middot; {html.escape(card['heading'] or '')}</div>"
                       f"{html.escape(card['text'][:700])}</div>")
        out.append("</details></div>")

    out.append("</main><div id='out'><div id='tally'>pick the clearest answer for each question</div>"
               "<div style='margin-top:8px'><button onclick='copy()'>copy my picks</button> "
               "<span id='copied' class='meta'></span></div></div>")
    out.append(f"<script>const QUESTIONS={len(rows)};{_JS}</script></body></html>")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=int, default=6)
    p.add_argument("--dry-run", action="store_true", help="price the run and write nothing")
    p.add_argument("--open", action="store_true", help="rebuild the page from the last run, spending nothing")
    p.add_argument("--no-warm", action="store_true", help="skip the verifier warm-up")
    args = p.parse_args(argv)

    if args.open:
        if not DATA.exists():
            p.error(f"no previous run at {DATA}")
        PAGE.parent.mkdir(parents=True, exist_ok=True)
        PAGE.write_text(page(json.loads(DATA.read_text(encoding="utf-8"))), encoding="utf-8", newline="\n")
        print(f"wrote {PAGE.relative_to(ROOT)}")
        return 0

    items = sample(args.items)
    if not items:
        p.error("no dev items matched the prose categories")
    calls = len(items) * len(P.VARIANTS)
    priced = estimate(["x" * 12000] * calls, "deepseek-flash", output_tokens_each=350)
    print(f"{len(items)} questions x {len(P.VARIANTS)} variants = {calls} calls")
    print(f"estimated ${priced.usd:.4f} (GBP {priced.gbp:.4f}) — {priced.basis}")
    for item in items:
        print(f"  {item.id} {i_cat(item):14s} {item.question[:72]}")
    if args.dry_run:
        print("dry run: nothing was spent")
        return 0

    data = run_async(run(items, warm=not args.no_warm))
    REPORTS.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page(data), encoding="utf-8", newline="\n")

    print()
    print(f"{'variant':20s} {'verified':>9s} {'dropped':>8s} {'rejected':>9s} {'abstain':>8s} "
          f"{'fellback':>9s} {'words':>6s} {'cost':>9s}")
    for variant in P.VARIANTS:
        t = data["totals"][variant.key]
        print(f"{variant.name:20s} {t['verified']:9d} {t['dropped']:8d} {t['rejected']:9d} "
              f"{t['abstained']:8d} {t['fell_back']:9d} {t['words']:6.1f} ${t['usd']:8.5f}")
    print(f"\ntotal spend ${data['total_usd']:.5f}")
    print(f"wrote {DATA.relative_to(ROOT)}")
    print(f"wrote {PAGE.relative_to(ROOT)}  <-- open this and pick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
