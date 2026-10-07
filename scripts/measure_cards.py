"""What the generator is SHOWN: for every dev item whose author cited evidence, where does that evidence end up?"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from answer import pipeline as PL  # noqa: E402
from answer import router as RO  # noqa: E402
from answer import routes as RT  # noqa: E402
from goldset.schema import load as load_gold  # noqa: E402
from ingest.runtime import run as run_async  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import eval_answers as EA  # noqa: E402

ORDER = ("citable", "visible_uncitable", "outside_window", "dropped_by_budget", "not_retrieved")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(_QUOTES)).strip().casefold()


def overlaps(sentence: str, quote: str) -> bool:
    """A sentence carries (part of) a quote: the quote lies within it, or it lies within a multi-sentence quote."""
    return bool(sentence) and bool(quote) and (quote in sentence or (len(sentence) > 20 and sentence in quote))


def classify(evidence, deck, hit_speeches: set[str]) -> str:
    quote = norm(evidence.quote)
    cards = [c for c in deck.cards if c.speech_id == evidence.speech_id]
    if not cards:
        return "dropped_by_budget" if evidence.speech_id in hit_speeches else "not_retrieved"
    best = "outside_window"
    for card in cards:
        if quote and quote not in norm(card.text) and not any(overlaps(norm(s.text), quote) for s in card.sentences):
            continue
        carrying = [s for s in card.sentences if overlaps(norm(s.text), quote)] or card.sentences
        if any(s.citable for s in carrying):
            return "citable"
        best = "visible_uncitable"
    return best


async def integrity(conn, deck) -> collections.Counter:
    """Every card against the database. Counts problems; an empty counter is a clean deck."""
    problems: collections.Counter = collections.Counter()
    ids = [s.chunk_id for c in deck.cards for s in c.sentences if s.citable]
    stored = dict(await (await conn.execute("SELECT id, body FROM chunk WHERE id = ANY(%s)", (ids,))).fetchall()) if ids else {}
    speeches = {c.speech_id for c in deck.cards}
    # The card names the speaker as `speaker_as_printed` does: the member's name, or the name printed at the time for a
    # member in another House now, and Hansard's label where the speech resolves to no member — "Baroness Merron", not
    # "Baroness Merron (Lab)".
    rows = await (await conn.execute(
        "SELECT s.id, speaker_as_printed(s.id), d.date, d.house FROM speech s "
        "JOIN debate d ON d.id = s.debate_id WHERE s.id = ANY(%s)", (list(speeches),))).fetchall()
    meta = {r[0]: r[1:] for r in rows}
    for card in deck.cards:
        for s in card.sentences:
            if s.citable and stored.get(s.chunk_id) is None:
                problems["citable id not in the database"] += 1
            elif s.citable and norm(stored[s.chunk_id]) != norm(s.text):
                problems["citable id's text differs from the stored sentence"] += 1
            if not (s.text or "").strip():
                problems["empty sentence"] += 1
        speaker, day, house = meta.get(card.speech_id, (None, None, None))
        if card.speech_id not in meta:
            problems["card speech not in the database"] += 1
            continue
        if card.date != day:
            problems["header date differs from the debate's"] += 1
        if card.house != house:
            problems["header House differs from the debate's"] += 1
        if (card.speaker or "") != (speaker or ""):
            problems["header speaker differs from the speech's"] += 1
        if not card.heading:
            problems["no debate heading"] += 1
        if not card.speaker:
            problems["no speaker"] += 1
        if not any(s.citable for s in card.sentences):
            problems["card with no citable sentence"] += 1
    return problems


async def main(split: str, label: str, limit: int | None, retrieval_top_k: int | None = None,
               dense_index: str | None = None, overrides: dict | None = None) -> int:
    if split != "dev":
        raise SystemExit("dev only: cards are diagnosed and changed on dev, never on a confirmation split")
    items = [i for i in load_gold(ROOT / "tests" / "data" / "gold_v2.json") if i.split == split and i.evidence]
    items = items[:limit] if limit else items
    runtime, pool = await PL.build_runtime(with_verifier=False)
    frozen_hash = runtime.frozen.config_hash
    if dense_index:
        # A MEASUREMENT override: the frozen file is untouched and the report says so; only this run's searches are
        # written for `dense_index`, so the two HNSW forms can be compared on the production deck path.
        runtime.frozen = dataclasses.replace(runtime.frozen, config={**runtime.frozen.config, "dense_index": dense_index})
    if overrides:
        # The same, for any frozen knob: measured here, adopted only by `uplift.py amend`.
        runtime.frozen = dataclasses.replace(runtime.frozen, config={**runtime.frozen.config, **overrides})
        runtime.frozen.search_config()  # an unknown knob or an invalid value is refused before anything is measured
    per_item, per_evidence = [], collections.Counter()
    problems: collections.Counter = collections.Counter()
    shape = collections.Counter()
    try:
        async with pool.connection() as conn:
            for n, item in enumerate(items, start=1):
                decision = await RO.route_with_model(item.question, None)  # the words alone, as production
                if decision.route not in ("retrieval", "overview"):
                    per_item.append({"id": item.id, "route": decision.route, "status": "not_a_card_route"})
                    continue
                scope = PL.retrieval_scope(decision.filters)

                def size(dated: bool) -> tuple[int, int]:
                    if retrieval_top_k is None:  # the pipeline's own policy
                        return PL.deck_size(decision.route, dated=dated)
                    if decision.route == "overview":  # an experiment varies the retrieval route only
                        return RT.OVERVIEW_POOL, PL.OVERVIEW_TOKEN_BUDGET
                    return retrieval_top_k, PL.C.DEFAULT_TOKEN_BUDGET

                top, budget = size(bool(scope))
                deck, result = await PL.search_deck(runtime, item.question, top, token_budget=budget,
                                                    filters=scope or None)  # fmt: skip
                widened = False
                if scope and len(result.hits) < PL.WIDEN_BELOW:
                    top, budget = size(False)
                    deck, result = await PL.search_deck(runtime, item.question, top, token_budget=budget)
                    widened = True
                hit_speeches = {h.speech_id for h in result.hits}
                statuses = [classify(e, deck, hit_speeches) for e in item.evidence]
                bearing = EA.gold_speech_ids(item)
                other_bearing_in_deck = any(c.speech_id in bearing for c in deck.cards)
                # For a quote outside the card's window: is it inside retrieval's own (untrimmed) expansion?
                full = {h.speech_id: norm(h.context or h.sentence) for h in result.hits}
                in_full_expansion = [bool(norm(e.quote) and norm(e.quote) in full.get(e.speech_id, ""))
                                     for e, st in zip(item.evidence, statuses) if st == "outside_window"]
                per_evidence.update(statuses)
                best = min(statuses, key=ORDER.index)
                problems.update(await integrity(conn, deck))
                shape["decks"] += 1
                shape["cards"] += len(deck.cards)
                shape["sentences"] += sum(len(c.sentences) for c in deck.cards)
                shape["citable"] += len(deck.chunk_ids)
                shape["dropped"] += deck.dropped
                shape["tokens"] += deck.used_tokens
                per_item.append({"id": item.id, "category": item.category, "route": decision.route,
                                 "widened": widened, "status": best, "statuses": statuses,
                                 "answer_bearing_in_deck": other_bearing_in_deck,
                                 "hit_ids": [h.chunk_id for h in result.hits],
                                 "outside_window_in_full_expansion": in_full_expansion,
                                 "cards": len(deck.cards), "dropped": deck.dropped, "tokens": deck.used_tokens})
                if n % 20 == 0:
                    print(f"  [{n}/{len(items)}]")
    finally:
        await pool.close()

    carded = [r for r in per_item if r["status"] != "not_a_card_route"]
    by_item = collections.Counter(r["status"] for r in carded)
    report = {
        "split": split, "label": label, "items": len(items), "items_on_a_card_route": len(carded),
        "frozen_hash": frozen_hash, "dense_index": runtime.frozen.config.get("dense_index"),
        "dense_index_override": dense_index, "overrides": overrides or {},
        "item_best_status": {k: by_item.get(k, 0) for k in ORDER},
        "evidence_status": {k: per_evidence.get(k, 0) for k in ORDER},
        "deck_shape": {"decks": shape["decks"], "cards_per_deck": round(shape["cards"] / max(shape["decks"], 1), 2),
                       "sentences_per_card": round(shape["sentences"] / max(shape["cards"], 1), 2),
                       "citable_per_card": round(shape["citable"] / max(shape["cards"], 1), 2),
                       "cards_dropped_total": shape["dropped"],
                       "tokens_per_deck": round(shape["tokens"] / max(shape["decks"], 1))},
        "integrity_problems": dict(problems),
        "per_item": per_item,
    }
    out = ROOT / "reports" / f"cards_{split}_{label}.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"\n{len(carded)} of {len(items)} evidence-bearing dev items take a card route")
    print("best status per item:  " + "  ".join(f"{k} {by_item.get(k, 0)}" for k in ORDER))
    print("per evidence quote:    " + "  ".join(f"{k} {per_evidence.get(k, 0)}" for k in ORDER))
    missed = [r for r in carded if r["status"] == "not_retrieved"]
    rescued = sum(1 for r in missed if r["answer_bearing_in_deck"])
    print(f"of {len(missed)} items whose CITED speech was not retrieved, {rescued} still had another answer-bearing "
          f"speech (graded >= 2) in the deck; {len(missed) - rescued} had none")
    widening = [x for r in carded for x in r["outside_window_in_full_expansion"]]
    print(f"outside-window quotes inside retrieval's full expansion: {sum(widening)} of {len(widening)}")
    report["cited_not_retrieved_but_other_evidence_in_deck"] = rescued
    report["cited_not_retrieved_and_no_evidence_in_deck"] = len(missed) - rescued
    print(f"deck shape: {report['deck_shape']}")
    print(f"integrity problems: {dict(problems) or 'none'}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--label", default="before")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None,
                        help="an experiment: this many cards on the retrieval route (default: the pipeline's deck_size)")
    parser.add_argument("--dense-index", choices=("vector", "halfvec"), default=None,
                        help="an experiment (T3): search through this HNSW form instead of the frozen one; the "
                             "frozen file is not changed and the report records the override")
    parser.add_argument("--set", action="append", metavar="KNOB=VALUE",
                        help="an experiment: override any frozen SearchConfig knob for this run only (e.g. "
                             "ef_search=200); recorded in the report, never written to the frozen file")
    args = parser.parse_args()
    overrides = {}
    for item in args.set or ():
        knob, _, raw = item.partition("=")
        try:
            overrides[knob.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[knob.strip()] = raw
    sys.exit(run_async(main(args.split, args.label, args.limit, args.top_k, args.dense_index, overrides)))
