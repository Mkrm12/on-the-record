"""Adversarial discrepancy triage: the maintainer adjudicates at most 20 contested judge verdicts."""

from __future__ import annotations

import random
from typing import Any

from goldset import calibration as CAL
from goldset import rubric as R
from goldset.judge import cohen_kappa

CAP = 20  # the maintainer's strict cap on cards
PAGE_VERSION = "triage-v1"
CHOICES = ("A", "B", "C")
STRATA = ("flagged", "owner_named", "controls")


def design(judged: dict[str, dict], flagged: list[str], owner_named: list[str], gold_ids: list[str], *,
           n_controls: int, seed: int, gold_cards: int = 1) -> dict[str, Any]:
    """The strata, fixed before anything is labelled: each stratum's ids, the judged rows it stands for, its weight.
    """
    named = [*flagged, *owner_named, *gold_ids]
    unknown = sorted(set(named) - set(judged))
    repeated = sorted({j for j in named if named.count(j) > 1})
    if unknown or repeated:
        raise ValueError(f"ids not in the judged run {unknown} or in two strata {repeated}")
    pool = sorted(set(judged) - set(named))
    if n_controls > len(pool):
        raise ValueError(f"{n_controls} controls asked for, {len(pool)} rows to draw from")
    controls = sorted(random.Random(seed).sample(pool, n_controls))
    cards = len(flagged) + len(owner_named) + len(controls) + gold_cards
    if cards > CAP:
        raise ValueError(f"{cards} cards: the owner's cap is {CAP}")
    strata = {
        "flagged": {"ids": sorted(flagged), "population": len(flagged)},
        "owner_named": {"ids": sorted(owner_named), "population": len(owner_named)},
        "controls": {"ids": controls, "population": len(pool)},
    }
    for s in strata.values():
        s["weight"] = round(s["population"] / len(s["ids"]), 6) if s["ids"] else 0.0
    return {"seed": seed, "cards": cards, "strata": strata, "kappa_population": len(judged) - len(gold_ids),
            "excluded_from_kappa": sorted(gold_ids)}


def weighted_kappa(pairs: list[tuple[str, str, float]]) -> dict[str, Any]:
    """Cohen's kappa with each (maintainer, judge, weight) pair standing for `weight` rows of the run: observed and
    chance agreement are both taken over the weighted rows. All weights 1 gives `goldset.judge.cohen_kappa` exactly.
    """
    total = sum(w for *_, w in pairs)
    if not pairs or total <= 0:
        return {"kappa": None, "agreement": None, "expected_agreement": None, "weight": 0}
    observed = sum(w for a, b, w in pairs if a == b) / total
    levels = {a for a, _, _ in pairs} | {b for _, b, _ in pairs}
    expected = sum((sum(w for a, _, w in pairs if a == lv) / total) * (sum(w for _, b, w in pairs if b == lv) / total)
                   for lv in levels)  # fmt: skip
    kappa = 1.0 if expected >= 1 else (observed - expected) / (1 - expected)
    return {"kappa": round(kappa, 4), "agreement": round(observed, 4), "expected_agreement": round(expected, 4),
            "weight": round(total, 4)}


def bootstrap_interval(by_stratum: dict[str, list[tuple[str, str, float]]], *, reps: int = 2000,
                       seed: int = 18) -> list[float] | None:
    """A 95% percentile interval for the weighted kappa: rows resampled within each stratum, as they were drawn."""
    filled = [pairs for pairs in by_stratum.values() if pairs]
    if not filled:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [rng.choice(pairs) for pairs in filled for _ in pairs]
        k = weighted_kappa(sample)["kappa"]
        if k is not None:
            values.append(k)
    values.sort()
    return [round(values[int(0.025 * (len(values) - 1))], 4), round(values[int(0.975 * (len(values) - 1))], 4)]


def check_labels(labels: dict, key: dict) -> list[str]:
    """Every reason the maintainer's file cannot be scored; empty when it can. Blindness is checked, not trusted: a
    card whose blind grade was set after its reveal is refused.
    """
    problems = []
    if labels.get("page") != PAGE_VERSION or labels.get("packet_id") != key["packet_id"]:
        problems.append(f"the labels are for {labels.get('page')!r}/{labels.get('packet_id')!r}, not "
                        f"{PAGE_VERSION!r}/{key['packet_id']!r}")
    cards = labels.get("cards") or {}
    for jid in sorted(set(cards) - set(key["cards"])):
        problems.append(f"{jid}: not a card in this build")
    for jid, card in cards.items():
        if jid not in key["cards"]:
            continue
        if card.get("blind") not in R.OUTCOMES:
            problems.append(f"{jid}: blind grade {card.get('blind')!r} is not one of the outcomes")
        if card.get("final") is not None and card.get("final") not in R.OUTCOMES:
            problems.append(f"{jid}: final grade {card.get('final')!r} is not one of the outcomes")
        if card.get("choice") is not None and card.get("choice") not in CHOICES:
            problems.append(f"{jid}: choice {card.get('choice')!r} is not A, B or C")
        if card.get("revealed_at") and not card.get("blind_at"):
            problems.append(f"{jid}: revealed with no blind grade on record")
        elif card.get("revealed_at") and card["blind_at"] > card["revealed_at"]:
            problems.append(f"{jid}: the blind grade was set after the reveal — it is not blind")
        total = key["cards"][jid]["judge"]["vital_total"]
        fv = card.get("final_vital")
        if fv is not None and (isinstance(fv, bool) or not isinstance(fv, int) or not 0 <= fv <= total):
            problems.append(f"{jid}: final vital count {fv!r} is not 0..{total}")
        elif card.get("choice") == "C":
            # the maintainer's own grade carries its own count, held to the judge's consistency rules
            # (gemini_judge.check)
            if fv is None:
                problems.append(f"{jid}: 'neither' needs a vital count")
            elif card.get("final") in ("correct", "correct_missing_okay_nuggets") and fv < total:
                problems.append(f"{jid}: {card['final']!r} needs every vital point ({fv} of {total})")
            elif card.get("final") == "missing_vital_nugget" and fv >= total:
                problems.append(f"{jid}: 'missing_vital_nugget' needs a missing point ({fv} of {total})")
    for name, decision in (labels.get("gold") or {}).items():
        if name not in key["gold_cards"]:
            problems.append(f"gold card {name!r} is not in this build")
        elif decision.get("choice") not in ("A", "B", None):
            problems.append(f"gold card {name!r}: choice {decision.get('choice')!r} is not A or B")
    return problems


def final_grade(card: dict, meta: dict) -> tuple[str, int] | None:
    """The adjudicated (outcome, vital_present) for a card, or None while it is undecided."""
    choice = card.get("choice")
    if choice == "A":
        return meta["judge"]["outcome"], meta["judge"]["vital_present"]
    if choice == "B":
        return meta["claude"]["outcome"], meta["claude"]["vital_present"]
    if choice == "C" and card.get("final") in R.OUTCOMES and isinstance(card.get("final_vital"), int):
        return card["final"], card["final_vital"]
    return None


def score(labels: dict, key: dict, *, reps: int = 2000) -> dict[str, Any]:
    """The maintainer's cards against the judge's verdicts, through the key. Kappa uses the BLIND grades only."""
    problems = check_labels(labels, key)
    if problems:
        raise ValueError(f"{len(problems)} problem(s): " + "; ".join(problems[:8]))
    cards = labels.get("cards") or {}
    weight = {jid: s["weight"] for s in key["strata"].values() for jid in s["ids"]}
    stratum_of = {jid: name for name, s in key["strata"].items() for jid in s["ids"]}
    blind = {jid: c["blind"] for jid, c in cards.items() if c.get("blind")}
    judge = {jid: key["cards"][jid]["judge"]["outcome"] for jid in blind}
    sample = cohen_kappa({(j,): blind[j] for j in blind}, {(j,): judge[j] for j in blind})
    sample.pop("passes_gate", None)  # a sample chosen for disagreement is not the gate's figure; the weighted one is
    pairs = [(blind[j], judge[j], weight[j]) for j in sorted(blind)]
    by_stratum = {s: [(blind[j], judge[j], weight[j]) for j in sorted(blind) if stratum_of[j] == s] for s in STRATA}
    weighted = weighted_kappa(pairs)
    weighted["interval_95"] = bootstrap_interval(by_stratum, reps=reps, seed=key.get("seed", 18))
    finals = {jid: final_grade(c, key["cards"][jid]) for jid, c in cards.items()}
    decided = {jid: g for jid, g in finals.items() if g}
    flips = sorted(j for j, g in decided.items() if blind.get(j) and g[0] != blind[j])
    owner_harmful = sorted(j for j in blind if blind[j] == "harmful")
    judge_harmful = sorted(j for j in blind if judge[j] == "harmful")
    caught = [j for j in owner_harmful if judge[j] == "harmful"]
    recall = len(caught) / len(owner_harmful) if owner_harmful else None
    graded = len(key["cards"])
    complete = len(blind) == graded and len(decided) == graded and all(
        (labels.get("gold") or {}).get(g, {}).get("choice") for g in key["gold_cards"])
    k = weighted["kappa"]
    passes = bool(complete and k is not None and k >= CAL.KAPPA_BAR
                  and (recall is None or recall >= CAL.HARMFUL_RECALL_BAR))
    return {
        "complete": complete,
        "cards_graded_blind": len(blind), "cards_decided": len(decided), "cards": graded,
        "kappa_on_sample": sample,
        "kappa_weighted_to_run": {**weighted, "population": key["kappa_population"],
                                  "note": "design-weighted: each card stands for the judged rows of its stratum"},
        "bars": {"kappa": CAL.KAPPA_BAR, "harmful_recall": CAL.HARMFUL_RECALL_BAR},
        "harmful": {"owner_blind": owner_harmful, "judge": judge_harmful, "caught": caught,
                    "recall": None if recall is None else round(recall, 4),
                    "note": "" if owner_harmful else "no card was graded harmful by the owner: harmful recall is not "
                                                     "measurable from this triage (the bar is met vacuously)"},
        "reveal_changed_the_grade": {"cards": flips, "rate": round(len(flips) / len(decided), 4) if decided else None},
        "disagreements_blind": sorted(j for j in blind if blind[j] != judge[j]),
        "passes": passes,
        "adjudicated": {j: {"outcome": g[0], "vital_present": g[1]} for j, g in sorted(decided.items())},
        "gold_decisions": {g: (labels.get("gold") or {}).get(g, {}).get("choice") for g in key["gold_cards"]},
        "entries_marked_wrong": sorted(j for j, c in cards.items() if c.get("entry_wrong")),
    }


def adjudicate_rows(rows: list[dict], key: dict, result: dict) -> tuple[list[dict], list[str]]:
    """The run's rows with the maintainer's final grades applied: (rows, item ids whose gold entry the maintainer ruled
    should ask which — the division card's B). Rows not on a card are untouched; a copy is returned, never the
    input.
    """
    by_item = {meta["item_id"]: jid for jid, meta in key["cards"].items()}
    out, to_clarify = [], []
    for card, choice in result["gold_decisions"].items():
        if choice == "B":
            to_clarify += key["gold_cards"][card]["item_ids"]
    for row in rows:
        row = dict(row)
        jid = by_item.get(row["id"])
        if jid and jid in result["adjudicated"]:
            row.update(result["adjudicated"][jid], by="owner")
        elif row["id"] in to_clarify:
            # the entry now says "ask which": the reply named every division and asked — that point is stated
            row.update(outcome="correct_abstention", vital_present=row.get("vital_total"), by="owner")
        out.append(row)
    return out, sorted(to_clarify)
