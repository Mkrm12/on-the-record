"""The set and overview routes, and the corrective loop."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from answer import cards as C
from answer import generate as G
from answer import roster as R
from answer.contract import Abstention, Answer, Provenance
from answer.verify import SUPPORT_THRESHOLD

log = logging.getLogger("answer.routes")

# : a wide pool for an overview, and the top debates to actually generate over.
OVERVIEW_POOL = 60
OVERVIEW_DEBATES = 8
# Per-debate generation runs concurrently, capped so a single overview cannot saturate the provider or the GPU.
OVERVIEW_CONCURRENCY = 3
# 's ceiling on corrective query rewrites.
MAX_REWRITES = 2

# Words that appear in questions and never in Hansard, with the register the record actually uses.
COLLOQUIAL: dict[str, str] = {
    "tories": "Conservative",
    "tory": "Conservative",
    "labour party": "Labour",
    "lib dems": "Liberal Democrat",
    "lib dem": "Liberal Democrat",
    "u-turn": "reversed its position",
    "u turn": "reversed its position",
    "uturn": "reversed its position",
    "backtrack": "reversed its position",
    "climbdown": "reversed its position",
    "slammed": "criticised",
    "blasted": "criticised",
    "hit out at": "criticised",
    "row over": "disagreement about",
    "cash": "funding",
    "cuts to": "reductions in",
    "scrapped": "abolished",
    "axed": "abolished",
    "pm": "Prime Minister",
    "chancellor": "Chancellor of the Exchequer",
    "nhs": "National Health Service",
}


def rewrite_query(question: str) -> tuple[str, str]:
    """A deterministic rewrite of the question into the record's own register."""
    text = question
    changed: list[str] = []
    for slang, formal in sorted(COLLOQUIAL.items(), key=lambda kv: -len(kv[0])):
        pattern = re.compile(rf"\b{re.escape(slang)}\b", re.I)
        if pattern.search(text):
            text = pattern.sub(formal, text)
            changed.append(f"{slang} -> {formal}")
    return (text, "; ".join(changed))


# --- the set route --------------------------------------------------------------------------------------


@dataclass
class SetAnswer:
    """A roster, or a request to clarify which debate was meant."""

    answer: Answer
    debate_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    clarify: tuple[dict[str, Any], ...] = ()
    speakers: int = 0


async def set_route(conn, question: str, provenance: Provenance, *, filters: Any = None,
                    candidates: list[R.DebateCandidate] | None = None) -> SetAnswer:  # fmt: skip
    """Who spoke: identify the debate, then pull the SQL roster. No model is called at all."""
    house = getattr(filters, "house", None)
    when = getattr(filters, "on_date", None)
    found = candidates
    if found is None:
        found = await R.candidates_in_question(conn, question, house, when)
    choice = R.pick_debate(found)

    if choice.needs_clarification:
        offered = tuple({"debate_id": c.debate_id, "title": c.title,
                         "date": c.date.isoformat() if c.date else None, "house": c.house}  # fmt: skip
                        for c in choice.clarify)
        return SetAnswer(
            answer=Answer(
                question=question, provenance=provenance,
                facets={"clarify": list(offered), "reason": choice.reason},
                coverage_note="Several debates match this description; pick one and the list of speakers follows.",
            ),
            confidence=choice.confidence, clarify=offered,
        )  # fmt: skip

    if not choice.ok:
        return SetAnswer(answer=Answer(
            question=question, provenance=provenance, abstention=Abstention.NOT_IN_RECORD,
            abstention_detail=choice.reason or "no debate in the record matches this description",
        ))  # fmt: skip

    built = await R.roster_with_gaps(conn, [choice.debate_id], confidence_score=choice.confidence, query=question)
    roster = built["roster"]
    if not roster.entries:
        return SetAnswer(answer=Answer(
            question=question, provenance=provenance, abstention=Abstention.NOT_IN_RECORD,
            abstention_detail=f"debate {choice.debate_id} records no contributions",
        ), debate_ids=(choice.debate_id,), confidence=choice.confidence)  # fmt: skip

    note = built["caveat"]
    # A roster is stated in words.
    chosen = next((c for c in (found or ()) if c.debate_id == choice.debate_id), None)
    names = ", ".join(e.speaker_name for e in roster.entries)
    where = (f"the debate on {chosen.title} ({chosen.house}, {chosen.date.isoformat()})"
             if chosen and chosen.date else f"debate {choice.debate_id}")
    rendered = f"Speakers in {where}, in order: {names}." + (f" {note}" if note else "")
    if asks_for_opener(question):
        # The question is who spoke FIRST, and a roster in speaking order does not say so: "Speakers: A, B, C" left the
        # vital point for the reader to infer.
        opener = await R.first_speaker(conn, choice.debate_id)
        if not opener.get("missing"):
            party = f" ({opener['party']})" if opener.get("party") else ""
            who = opener.get("display_name") or opener["speaker_name"]
            # The question asked who OPENED it, so that is the answer; the count says how many followed.
            count = len(roster.entries)
            rendered = (f"{who}{party} opened the debate on {opener['heading']} in the {opener['house']} on "
                        f"{opener['date'].isoformat()}. {count} member{'s' if count != 1 else ''} spoke in all.")
            if note:
                rendered += f" {note}"
    return SetAnswer(
        answer=Answer(question=question, provenance=provenance, roster=roster,
                      coverage_note=note, rendered=rendered),  # fmt: skip
        debate_ids=(choice.debate_id,), confidence=choice.confidence, speakers=len(roster.entries),
    )  # fmt: skip


# The ways a question asks who spoke FIRST.
_OPENER = re.compile(
    r"\b(?:spoke|speak|speaking)\s+first\b|\bfirst\s+(?:to\s+speak|speaker|contribution|speech)\b"
    r"|\b(?:led|leads?|lead|kicked|kicks?)\s+off\b|\bopening\s+(?:speech|speaker|contribution)\b"
    r"|\bwho\s+(?:opened|opens|began|begins|started|starts)\b", re.I)


def asks_for_opener(question: str) -> bool:
    return bool(_OPENER.search(question or ""))


# --- the overview route ---------------------------------------------------------------------------------


async def facets(conn, chunk_ids: list[int]) -> dict[str, Any]:
    """Counts over the retrieved pool, from SQL, labelled as what they are."""
    if not chunk_ids:
        return {"label": "passages found by search", "total": 0}
    cur = await conn.execute("""
        SELECT c.house, d.section, sp.speaker_party AS party, c.debate_date, sp.mnis_id,
               min(sp.speaker_name) AS speaker_name
        FROM chunk c
        JOIN speech sp ON sp.id = c.speech_id
        JOIN debate d ON d.id = c.debate_id
        WHERE c.id = ANY(%s)
        GROUP BY c.house, d.section, sp.speaker_party, c.debate_date, sp.mnis_id
    """, (list(chunk_ids),))
    rows = await cur.fetchall()
    columns = [c.name for c in cur.description or []]
    records = [dict(zip(columns, row, strict=True)) for row in rows]

    def tally(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            value = record.get(key)
            if value:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    dates = sorted(r["debate_date"] for r in records if r.get("debate_date"))
    return {
        "label": "passages found by search, not everything said on the subject",
        "total": len(chunk_ids),
        "by_house": tally("house"),
        "by_section": tally("section"),
        "by_party": tally("party"),
        "speakers": len({r["mnis_id"] for r in records if r.get("mnis_id")}),
        "first_date": dates[0].isoformat() if dates else None,
        "last_date": dates[-1].isoformat() if dates else None,
    }


def cluster_by_debate(hits: list[Any], limit: int = OVERVIEW_DEBATES) -> list[tuple[str, list[Any]]]:
    """Group hits by debate, best first, then spread across time, House and party."""
    groups: dict[str, list[Any]] = {}
    for hit in hits:
        groups.setdefault(hit.debate_id, []).append(hit)
    ordered = sorted(groups.items(), key=lambda kv: min(h.rank for h in kv[1]))

    picked: list[tuple[str, list[Any]]] = []
    seen_dates: set = set()
    seen_houses: set = set()
    seen_parties: set = set()
    remaining = list(ordered)
    while remaining and len(picked) < limit:
        def novelty(entry: tuple[str, list[Any]]) -> tuple:
            members = entry[1]
            dates = {h.date for h in members if h.date}
            houses = {h.house for h in members if h.house}
            parties = {h.party for h in members if h.party}
            adds = ((1 if dates - seen_dates else 0) + (1 if houses - seen_houses else 0)
                    + (1 if parties - seen_parties else 0))  # fmt: skip
            return (-adds, min(h.rank for h in members))

        best = min(remaining, key=novelty) if picked else remaining[0]
        remaining.remove(best)
        picked.append(best)
        for hit in best[1]:
            if hit.date:
                seen_dates.add(hit.date)
            if hit.house:
                seen_houses.add(hit.house)
            if hit.party:
                seen_parties.add(hit.party)
    return picked


# The reduction prompt.
REDUCE_PROMPT = """\
Below are claims that have already been checked against the parliamentary record and found accurate. Each is
attributed and carries an id.

Write a short overall summary of what these claims show, as a list of summary claims.

RULES:
- Every summary claim must follow from the claims below. Do not add anything they do not say.
- Do not introduce a name, a number, a date or a position that is not in the claims below.
- Attribute a position to a person, never to a party or to "MPs" as a group.
- One sentence per claim.
- For each summary claim, list the ids of the claims it draws on.

CLAIMS:
{claims}

Return JSON only:
{{"summary": [{{"text": "one sentence", "from": ["c1", "c2"]}}]}}
"""


@dataclass
class Overview:
    """Per-debate verified claims, a reduced summary, and the facets."""

    answer: Answer
    debates: int = 0
    per_debate_claims: int = 0
    # What the debates' generators wrote and the verifier dropped, summed: the trace's claims columns read only the
    # retrieval route's `Generated`, so they were empty for every overview.
    claims_generated: int = 0
    claims_dropped: int = 0
    # Nothing verified while a debate's generator failed (a refused key, an outage): the decline was never established.
    generator_failed: str = ""
    failed_debates: int = 0
    # A debate answered without the full verifier — the first reason, "" when none was.
    degraded_reason: str = ""
    # Each debate generator's own decline reason, counted: "outside_window" from most debates read declines the
    # overview; the rest is diagnosis — a debate holding nothing on the subject declines "not_in_record".
    debate_declines: dict[str, int] = field(default_factory=dict)
    summary_claims: int = 0
    summary_dropped: int = 0
    remaining_debates: tuple[dict[str, Any], ...] = ()
    calls: int = 0
    cost_usd: float = 0.0


async def overview_route(conn, question: str, hits: list[Any], client: Any, provenance: Provenance, *,
                         score_batch: Any = None, budget: G.Budget | None = None,
                         verifier_name: str = "deterministic", precision: str = "",
                         variant_key: str | None = None) -> Overview:  # fmt: skip
    """Cluster, generate per debate, verify, then reduce to summary claims entailed by what survived."""
    budget = budget or G.Budget()
    clustered = cluster_by_debate(hits)
    if not clustered:
        return Overview(answer=Answer(question=question, provenance=provenance,
                                      abstention=Abstention.NOT_IN_RECORD,
                                      abstention_detail="no debate matched this subject"))  # fmt: skip

    import asyncio

    semaphore = asyncio.Semaphore(OVERVIEW_CONCURRENCY)
    # Every id the per-debate decks offered the model.
    shown: set[int] = set()

    async def for_debate(debate_id: str, members: list[Any]) -> G.Generated | None:
        async with semaphore:
            if budget.exhausted():
                return None
            deck = C.build_deck(members, budget_tokens=3_000)
            shown.update(deck.chunk_ids)
            return await G.answer_question(
                question, deck, client, provenance, variant_key=variant_key,
                score_batch=score_batch, budget=budget, verifier_name=verifier_name, precision=precision,
            )  # fmt: skip

    produced = await asyncio.gather(*(for_debate(d, m) for d, m in clustered), return_exceptions=True)

    verified: list = []
    generated_total = dropped_total = 0
    failures: list[str] = []
    degraded: list[str] = []
    declines: Counter[str] = Counter()
    for result in produced:
        if isinstance(result, BaseException) or result is None:
            if isinstance(result, BaseException):
                log.warning("an overview debate failed (%s)", type(result).__name__)
                failures.append(f"{type(result).__name__}: {' '.join(str(result).split())[:160]}")
            continue
        if getattr(result, "generator_failed", ""):
            failures.append(result.generator_failed)
            continue  # never read: neither its claims (it has none) nor its decline (the system's, not the model's)
        if getattr(result, "degraded", False):
            degraded.append(getattr(result, "degraded_reason", "") or "degraded (no reason recorded)")
        reason = getattr(result.answer, "abstention", None)
        if reason is not None:
            declines[reason.value] += 1
        verified.extend(result.answer.claims)
        generated_total += getattr(result, "claims_generated", 0) or 0
        dropped_total += getattr(result, "claims_dropped", 0) or 0
    # A debate whose generator was never reached was never READ.
    any_failed = failures[0] if failures else ""
    attempted = sum(1 for result in produced if result is not None)
    read = attempted - len(failures)

    # Whether a question falls outside the record's periods is a property of the QUESTION.
    if read and declines.get(Abstention.OUTSIDE_WINDOW.value, 0) * 2 > read:
        return Overview(
            answer=Answer(question=question, provenance=provenance, abstention=Abstention.OUTSIDE_WINDOW,
                          abstention_detail=G.READER_REASONS[Abstention.OUTSIDE_WINDOW]),  # fmt: skip
            debates=len(clustered), claims_generated=generated_total, claims_dropped=dropped_total,
            failed_debates=len(failures), degraded_reason=degraded[0] if degraded else "",
            debate_declines=dict(declines), calls=budget.calls, cost_usd=budget.spent_usd,
        )  # fmt: skip

    all_chunks = [i for hit in hits for i in (hit.chunk_id, *(hit.sibling_chunk_ids or ()))]
    facet_block = await facets(conn, all_chunks)

    if not verified:
        # A DECLINE, in the system's own words: this used to return facet counts and a note with no claims and no
        # abstention, so the reader was shown an empty answer.
        return Overview(
            answer=Answer(question=question, provenance=provenance, abstention=Abstention.INSUFFICIENT_EVIDENCE,
                          abstention_detail=(f"{facet_block['total']} passages were found by search and none produced "
                                             "a claim that could be verified against the record.")),  # fmt: skip
            debates=len(clustered), claims_generated=generated_total, claims_dropped=dropped_total,
            generator_failed=any_failed, failed_debates=len(failures), degraded_reason=degraded[0] if degraded else "",
            debate_declines=dict(declines), calls=budget.calls, cost_usd=budget.spent_usd,
        )  # fmt: skip

    # --- the reduction ------------------------------------------------------------------------------
    numbered = {f"c{i + 1}": claim for i, claim in enumerate(verified)}
    listing = "\n".join(
        f"{key}: {claim.speaker_name or 'a member'} ({claim.date}) — {claim.text}"
        for key, claim in numbered.items()
    )
    summary_claims: list = []
    dropped = 0
    # A summary sentence is the model's paraphrase of verified claims, so it is shown only once the verifier has checked
    # it against them.
    if score_batch is not None and not budget.exhausted():
        try:
            reply = await client.chat(REDUCE_PROMPT.format(claims=listing), max_tokens=900,
                                     temperature=0.0, json_object=True)  # fmt: skip
            budget.record(reply)
            data = json.loads(reply.content)
            for entry in (data.get("summary") or [])[:6]:
                text = str(entry.get("text") or "").strip()
                sources = [numbered[k] for k in (entry.get("from") or []) if k in numbered]
                if not text or not sources:
                    dropped += 1
                    continue
                # The rule that makes an overview trustworthy: the summary's evidence is the verified claims, not the
                # corpus.
                supporting = " ".join(c.text for c in sources)
                import inspect

                produced_scores = score_batch([supporting], [text])
                scores = list(await produced_scores if inspect.isawaitable(produced_scores)
                              else produced_scores)  # fmt: skip
                # No score is not a pass.
                if not scores or scores[0] < SUPPORT_THRESHOLD:
                    dropped += 1
                    continue
                summary_claims.append({"text": text, "from": [k for k in (entry.get("from") or [])
                                                              if k in numbered]})  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - a failed reduction leaves the per-debate claims standing
            log.warning("the overview reduction failed (%s: %s)", type(exc).__name__,
                        " ".join(str(exc).split())[:160])  # fmt: skip

    listed = tuple({"debate_id": d, "hits": len(m)} for d, m in clustered[OVERVIEW_DEBATES:])
    unread = (f" {len(failures)} of the {attempted} debates could not be read (the generator could not be reached), so "
              f"this overview is drawn from the rest." if failures else "")  # fmt: skip
    fields = dict(
        question=question, provenance=provenance, claims=tuple(verified),
        retrieved_chunk_ids=frozenset(shown) | frozenset(i for i in all_chunks if i is not None and i >= 0),
        facets={**facet_block, "summary": summary_claims, "other_debates": list(listed)},
        coverage_note=(f"Drawn from {len(clustered)} debates out of {facet_block['total']} passages found by "
                       f"search. This is not everything said on the subject.{unread}"),  # fmt: skip
    )
    answer = Answer(**fields, rendered=overview_text(Answer(**fields), summary_claims))
    return Overview(
        answer=answer, debates=len(clustered), per_debate_claims=len(verified),
        claims_generated=generated_total, claims_dropped=dropped_total, failed_debates=len(failures),
        degraded_reason=degraded[0] if degraded else "", debate_declines=dict(declines),
        summary_claims=len(summary_claims), summary_dropped=dropped,
        remaining_debates=listed, calls=budget.calls, cost_usd=budget.spent_usd,
    )  # fmt: skip


def overview_text(answer: Answer, summary: list[dict[str, Any]]) -> str:
    """What an overview's reader is shown: the checked summary sentences, then every verified per-debate claim with its
    speaker and date, then the coverage note — which is never dropped.
    """
    lead = " ".join(str(s.get("text") or "").strip() for s in summary if str(s.get("text") or "").strip())
    body = answer.render()
    return f"{lead} {body}".strip() if lead else body


# --- the corrective loop --------------------------------------------------------------------------------


@dataclass
class Attempt:
    """One pass of the corrective loop, kept so the trace can say what was tried."""

    query: str
    rewritten_from: str | None = None
    what_changed: str = ""
    hits: int = 0
    cards: int = 0
    outcome: str = ""


@dataclass
class Corrected:
    """The final answer and the attempts it took."""

    generated: G.Generated
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def rewrites(self) -> int:
        return sum(1 for a in self.attempts if a.rewritten_from)


async def with_corrections(search, answer_fn, question: str, *, budget: G.Budget,
                           max_rewrites: int = MAX_REWRITES) -> Corrected:  # fmt: skip
    """Retrieve, answer, and on a miss rewrite the query once and try again."""
    attempts: list[Attempt] = []
    deck = await search(question)
    result = await answer_fn(question, deck)
    attempts.append(Attempt(query=question, hits=len(deck.cards), cards=len(deck.cards),
                            outcome=(result.answer.abstention.value if result.answer.abstention
                                     else "answer")))  # fmt: skip

    retryable = {Abstention.NOT_IN_RECORD, Abstention.INSUFFICIENT_EVIDENCE}
    used = 0
    while (used < max_rewrites
           and result.answer.abstention in retryable
           and not budget.exhausted()):  # fmt: skip
        rewritten, changed = rewrite_query(attempts[-1].query)
        if not changed or rewritten == attempts[-1].query:
            break
        used += 1
        deck = await search(rewritten)
        retried = await answer_fn(rewritten, deck)
        attempts.append(Attempt(query=rewritten, rewritten_from=attempts[-1].query, what_changed=changed,
                                hits=len(deck.cards), cards=len(deck.cards),
                                outcome=(retried.answer.abstention.value if retried.answer.abstention
                                         else "answer")))  # fmt: skip
        # Kept only if it actually did better.
        if not retried.answer.abstention:
            result = retried
            break
    return Corrected(generated=result, attempts=attempts)
