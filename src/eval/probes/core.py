"""The probe runner and its arithmetic: a closed outcome vocabulary, the production path, and exact bounds."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from psycopg import pq

from goldset.instrument import wilson

OUTCOMES = ("harm", "error", "misroute", "fallthrough", "declined", "clarified", "aggregate", "substituted",
            "mismatch", "incomplete", "pass")
# The outcomes a reader is well served by: `pass` always; `clarified`, `aggregate` and `substituted` only for a truly
# ambiguous question, and a clarification only when it offers the subject asked about.
ACCEPTABLE_ALWAYS = frozenset({"pass"})
ACCEPTABLE_WHEN_AMBIGUOUS = frozenset({"pass", "clarified", "aggregate", "substituted"})
# The outcomes in which the answer asserted (or offered) something.
ANSWERED = frozenset({"harm", "aggregate", "substituted", "mismatch", "incomplete", "pass"})
# Counted in the harm bound: a harm, and an answer the oracle could not confirm, until a person triages it.
HARMFUL = frozenset({"harm", "mismatch"})
# Forms whose truth uses the route's own source and rule, so they test the code, not the rule: reported apart.
SHARED_RULE = ("no_vote_day", "history_change", "seat_gap", "future")
POPULATION = ("generated questions: the forms these generators write (canonical, everyday, paraphrased, by other names, "
              "compound, mutated) about the record's own divisions, debates, stages and members outside the sealed "
              "window. The bound holds for those forms; it is not a bound on every question a person could ask. "
              "Every truth is read, by code of its own, from the same official payloads the database was loaded "
              "from (the shared source): the probes test the answer path (resolution, routing, wording, the "
              "mask), not the sources.")


def asserts_absence(answer: Any) -> bool:
    """Whether the answer states that the record does not hold what was asked: a claim, judged true or false."""
    abstention = getattr(answer, "abstention", None)
    return abstention is not None and getattr(abstention, "value", abstention) == "not_in_record"


class ProbeError(RuntimeError):
    """A generator produced a trial it must never produce (one dated inside the sealed window)."""


@dataclass(frozen=True)
class Trial:
    trial_id: str
    cls: str  # the probe class, e.g.
    category: str  # the structured fact the question asks for (or "opener" for the set route)
    question: str
    form: str  # "canonical": a form the router is specified to read; "variant": everyday wording
    subject: Mapping[str, Any]  # what the oracle needs (division_id, mnis_id, expected values)
    ambiguous: bool = False  # the record holds more than one answer to the question as worded
    disputed: bool = False  # the sources disagree about the subject
    route: str = "structured"  # the route production must choose: "structured", or "set" for who spoke

    @property
    def cluster(self) -> tuple:
        """What the trial's outcome is correlated with: its division, or for a day's question its House and day. Trials
        on one division share its resolution, so harm is also bounded per cluster, not only per trial.
        """
        for key, kind in (("division_id", "division"), ("debate_id", "debate"), ("stage_id", "stage"),
                          ("mnis_id", "member")):  # fmt: skip
            if self.subject.get(key) is not None and (key != "mnis_id" or "division_id" not in self.subject):
                return (kind, self.subject[key])
        return ("day", self.subject.get("house"), self.subject.get("date"))


@dataclass(frozen=True)
class Outcome:
    trial: Trial
    outcome: str
    why: str = ""
    route: str | None = None
    category: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")

    @property
    def acceptable(self) -> bool:
        allowed = ACCEPTABLE_WHEN_AMBIGUOUS if self.trial.ambiguous else ACCEPTABLE_ALWAYS
        return self.outcome in allowed


# (trial, answer) -> (outcome, why).
Verdict = Callable[[Trial, Any], tuple[str, str]]


def _provenance():
    from answer.contract import Provenance

    return Provenance(route="structured", corpus_version="probe", retrieval_config_hash="probe", recipe="probe",
                      prompt_version="probe", model_id="none")  # fmt: skip


async def run_trial(conn, trial: Trial, verdict: Verdict) -> Outcome:
    """One trial on the production path. Raises `ProbeError` for a trial dated inside the sealed window."""
    from answer import pipeline as P
    from answer import router as RO
    from answer import sealing as SG

    decision = RO.route(trial.question)
    if SG.question_is_sealed(decision.filters):
        raise ProbeError(f"{trial.trial_id} is dated inside the sealed window; no generator may produce it")
    seen = {"route": decision.route, "category": decision.category}
    if trial.route == "set":
        if decision.route != "set":
            return Outcome(trial, "misroute", f"routed to {decision.route}/{decision.category}: {decision.why}"[:300],
                           **seen)  # fmt: skip
        from answer import routes as RT

        answer = (await RT.set_route(conn, trial.question, _provenance(), filters=decision.filters)).answer
    else:
        if decision.route != "structured" or decision.category != trial.category:
            return Outcome(trial, "misroute", f"routed to {decision.route}/{decision.category}: {decision.why}"[:300],
                           **seen)  # fmt: skip
        answer, notes = await P.structured_answer(conn, trial.question, decision, _provenance())
        if answer is None:
            return Outcome(trial, "fallthrough", "; ".join(notes)[:300], **seen)
    answer = P.reader_ready(answer)
    answer, withheld = await SG.guard(conn, answer)
    if withheld:
        return Outcome(trial, "error", f"the sealed-window guard withheld an answer outside it: {withheld[:3]}",
                       **seen)  # fmt: skip
    outcome, why = verdict(trial, answer)
    return Outcome(trial, outcome, why, **seen)


async def run(conn, trials: Iterable[Trial], verdict: Verdict) -> list[Outcome]:
    """Every trial, in order, on one connection. Read-only: the structured route writes nothing."""
    out = []
    for trial in trials:
        out.append(await run_trial(conn, trial, verdict))
        if conn.info.transaction_status != pq.TransactionStatus.IDLE:  # no trial's transaction carries into the next
            await conn.rollback()
    return out


# --- the arithmetic --------------------------------------------------------------------------------------


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    logp, logq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    return min(1.0, sum(math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) + i * logp + (n - i) * logq)
                        for i in range(k + 1)))  # fmt: skip


def harm_upper(k: int, n: int, alpha: float = 0.05) -> float | None:
    """The exact (Clopper-Pearson) one-sided upper bound on a rate seen k times in n: the p at which seeing k or fewer
    has probability alpha. At k = 0 it is 1 - alpha^(1/n) (about 3/n, the rule of three).
    """
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = k / n, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _block(outcomes: list[Outcome]) -> dict[str, Any]:
    """One block of the report. Harm is bounded twice: over the answered trials (the only ones that could harm), and
    over the clusters they fall in (trials on one division are correlated, so the cluster bound is the honest one).
    """
    n = len(outcomes)
    counts = Counter(o.outcome for o in outcomes)
    harms = counts["harm"]
    harmful = sum(counts[k] for k in HARMFUL)
    answered = [o for o in outcomes if o.outcome in ANSWERED]
    clusters = {o.trial.cluster for o in answered}
    harm_clusters = {o.trial.cluster for o in answered if o.outcome in HARMFUL}
    ok = sum(1 for o in outcomes if o.acceptable)
    interval = wilson(ok, n) if n else None

    def bound(k: int, m: int) -> float | None:
        return None if not m else round(harm_upper(k, m), 6)

    return {"trials": n, "outcomes": {k: counts[k] for k in OUTCOMES if counts[k]},
            "pass": counts["pass"], "pass_rate": None if not n else round(counts["pass"] / n, 4),
            # the bound counts a mismatch with the harms; nothing answered bounds nothing (`vacuous`)
            "answered": len(answered), "vacuous": not answered, "harm": harms, "mismatch": counts["mismatch"],
            "harm_upper_95": bound(harmful, len(answered)),
            "clusters": len(clusters), "harm_clusters": len(harm_clusters),
            "harm_upper_95_by_cluster": bound(len(harm_clusters), len(clusters)),
            "acceptable": ok, "acceptable_rate": None if not n else round(ok / n, 4),
            "acceptable_wilson_95": None if interval is None else [round(interval[0], 4), round(interval[1], 4)]}


def summarise(outcomes: list[Outcome], *, examples: int = 5) -> dict[str, Any]:
    """Per class, then per class and form: every outcome counted, the exact harm bound, the acceptable rate."""
    judged = [o for o in outcomes if not o.trial.disputed]
    by_class: dict[str, list[Outcome]] = defaultdict(list)
    by_form: dict[str, list[Outcome]] = defaultdict(list)
    for o in judged:
        by_class[o.trial.cls].append(o)
        by_form[f"{o.trial.cls} [{o.trial.form}]"].append(o)
    shown: dict[str, list[dict[str, Any]]] = {}
    for o in judged:
        if o.outcome == "pass":
            continue
        bucket = shown.setdefault(o.outcome, [])
        if o.outcome == "harm" or len(bucket) < examples:
            bucket.append({"trial": o.trial.trial_id, "question": o.trial.question, "why": o.why})
    disputed = [o for o in outcomes if o.trial.disputed]
    return {
        "population": POPULATION,
        "all": _block(judged),
        "by_class": {k: _block(v) for k, v in sorted(by_class.items())},
        "by_class_and_form": {k: _block(v) for k, v in sorted(by_form.items())},
        # The canonical forms ask in the record's own words, which the route reads, so they show little; the bound that
        # measures understanding is the adversarial forms' (other names, compounds, mutations, run-on dates, qualified
        # offices, longer names, unreached stages...).
        "by_tier": {"canonical": _block([o for o in judged if o.trial.form == "canonical"]),
                    "adversarial": _block([o for o in judged if o.trial.form not in ("canonical", *SHARED_RULE)]),
                    "shared_rule": _block([o for o in judged if o.trial.form in SHARED_RULE])},
        "examples": {k: shown[k] for k in OUTCOMES if k in shown},
        "disputed": {"trials": len(disputed), "outcomes": dict(Counter(o.outcome for o in disputed)),
                     "list": [{"trial": o.trial.trial_id, "outcome": o.outcome, "why": o.why} for o in disputed]},
    }


@dataclass
class Tally:
    """A running count for the CLI's progress line."""

    counts: Counter = field(default_factory=Counter)

    def add(self, outcome: Outcome) -> None:
        self.counts[outcome.outcome] += 1

    def line(self) -> str:
        return ", ".join(f"{k} {self.counts[k]}" for k in OUTCOMES if self.counts[k]) or "nothing yet"
