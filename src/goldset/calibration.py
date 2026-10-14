"""Answer-judge calibration against the maintainer: "Calibrate against me on about 50 answers. Require κ >= 0.6 on
outcome categories and high recall on 'harmful'."
"""

from __future__ import annotations

import random
from typing import Any

from goldset import rubric as R
from goldset.blind import assert_blind
from goldset.judge import cohen_kappa

# What the maintainer chooses from — the judge's own vocabulary, all seven outcomes.
OWNER_OUTCOMES = R.OUTCOMES
KAPPA_BAR = 0.6
HARMFUL_RECALL_BAR = 0.9
# Every rare verdict is taken (harmful recall needs them), then the common outcomes in turn, so the sample is not all
# one kind: the first version took every missing-vital answer before any correct one and built a 50-answer packet with
# no correct answer in it — a kappa over that says little about the judge.
_RARE = ("harmful", "wrong_but_benign")
_COMMON = ("missing_vital_nugget", "correct", "correct_missing_okay_nuggets", "over_abstention", "correct_abstention")


# Who counts as a judge MODEL in a stored run: the offline Gemini judge, an API Gemini or Anthropic judge, and Groq,
# which graded every run before that date and is kept so those runs can still be calibrated.
JUDGE_NAMES = ("gemini-deep-research", "google", "gemini", "anthropic", "groq")


def judged_rows(rows: list[dict], judge_names: tuple[str, ...] = JUDGE_NAMES) -> list[dict]:
    """Rows graded by a judge model — not deterministically, not skipped as gold-incomplete, not ungraded."""
    return [r for r in rows if r.get("outcome") in OWNER_OUTCOMES and str(r.get("by", "")).lower() in judge_names]


def select(rows: list[dict], n: int = 50, seed: int = 18) -> list[dict]:
    """A stratified sample of judged rows: every rare outcome, then the common ones at random, up to n."""
    rng = random.Random(seed)
    chosen: list[dict] = []
    for stratum in _RARE:
        pool = [r for r in rows if r.get("outcome") == stratum]
        rng.shuffle(pool)
        chosen += pool[: max(0, n - len(chosen))]
    pools = {s: [r for r in rows if r.get("outcome") == s] for s in _COMMON}
    for pool in pools.values():
        rng.shuffle(pool)
    while len(chosen) < n and any(pools.values()):
        for stratum in _COMMON:
            if pools[stratum] and len(chosen) < n:
                chosen.append(pools[stratum].pop())
    return chosen


def worksheet(selected: list[dict], gold: dict[str, Any], seed: int = 18) -> tuple[list[dict], dict[str, dict]]:
    """(the maintainer's worksheet, the key). The worksheet carries nothing a grader produced; the key maps each opaque
    id to the item and the judge's verdict, and is never given to the page.
    """
    rng = random.Random(seed + 1)
    order = list(selected)
    rng.shuffle(order)
    items, key = [], {}
    for n, row in enumerate(order, start=1):
        opaque = f"c{n:03d}"
        item = gold[row["id"]]
        items.append({
            "cid": opaque,
            "question": item.question,
            # What the judge was told a right response does, and the answer as the judge read it — a decline in its own
            # words, never "(nothing was shown)" (`rubric.answer_text`, the Gemini packet's too).
            "expected": R.expected_text(item),
            "reference_points": [{"text": nug.text, "vital": bool(nug.vital)} for nug in (item.nuggets or ())],
            "wrong_if_asserted": [getattr(f, "text", str(f)) for f in (item.forbidden_claims or ())],
            "answer": R.answer_text(row),
        })
        key[opaque] = {"item_id": row["id"], "judge_outcome": row["outcome"], "judge_why": row.get("why", "")}
    assert_blind({"items": items})
    return items, key


def score(owner: dict[str, str], key: dict[str, dict], judge_harmful_in_run: int | None = None,
          deterministic_harmful: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """The maintainer's labels against the judge's, through the key."""
    unknown = sorted(set(owner) - set(key))
    bad = sorted(c for c, label in owner.items() if label not in OWNER_OUTCOMES)
    if unknown or bad:
        raise ValueError(f"labels for unknown items {unknown[:5]} or outside the vocabulary {bad[:5]}")
    shared = sorted(set(owner) & set(key))
    judge = {(c,): key[c]["judge_outcome"] for c in shared}
    person = {(c,): owner[c] for c in shared}
    kappa = cohen_kappa(person, judge)
    binary = cohen_kappa({k: v == "harmful" for k, v in person.items()}, {k: v == "harmful" for k, v in judge.items()})
    owner_harmful = [c for c in shared if owner[c] == "harmful"]
    judge_harmful = [c for c in shared if key[c]["judge_outcome"] == "harmful"]
    caught = [c for c in owner_harmful if key[c]["judge_outcome"] == "harmful"]
    recall = len(caught) / len(owner_harmful) if owner_harmful else None
    precision = len(caught) / len(judge_harmful) if judge_harmful else None
    passes = bool(kappa["kappa"] is not None and kappa["kappa"] >= KAPPA_BAR
                  and (recall is None or recall >= HARMFUL_RECALL_BAR))
    disagreements = [{"cid": c, "item_id": key[c]["item_id"], "owner": owner[c], "judge": key[c]["judge_outcome"],
                      "judge_why": key[c]["judge_why"]} for c in shared if owner[c] != key[c]["judge_outcome"]]
    in_packet = sum(1 for v in key.values() if v["judge_outcome"] == "harmful")
    covered = judge_harmful_in_run is not None and in_packet == judge_harmful_in_run and \
        all(c in owner for c, v in key.items() if v["judge_outcome"] == "harmful")
    standing = sorted(set(deterministic_harmful))
    adjudicated = {
        "every_judge_harmful_answer_labelled": covered,
        "deterministic_harmful": standing,
        "harmful_after_owner_review": (sorted(set(standing) | {key[c]["item_id"] for c in owner_harmful})
                                       if covered else None),
        "note": ("the owner's verdict on every answer the judge called harmful, plus the answers harmful without a judge "
                 "(a forbidden claim): this is the run's harmful count"
                 if covered else "not every answer the judge called harmful was labelled; no adjudicated count"),
    }
    return {"n": len(shared), "kappa_outcomes": kappa, "kappa_harmful_or_not": binary, "adjudicated": adjudicated,
            "harmful": {"owner": len(owner_harmful), "judge": len(judge_harmful), "caught": len(caught),
                        "recall": None if recall is None else round(recall, 4),
                        "precision": None if precision is None else round(precision, 4)},
            "bars": {"kappa": KAPPA_BAR, "harmful_recall": HARMFUL_RECALL_BAR},
            "passes": passes,
            "note": ("no answer in the sample was labelled harmful by the owner, so harmful recall is not measurable "
                     "from it" if recall is None else ""),
            "disagreements": disagreements}
