"""Nothing a reader is shown is dated inside the sealed window."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, datetime
from typing import Any

from answer.contract import Abstention, Answer, Provenance
from app import sealed

log = logging.getLogger(__name__)

# At most this many reasons are kept: one is enough to withhold, and the log needs only the first few.
MAX_REASONS = 12


def question_is_sealed(filters: Any) -> bool:
    """Whether the question's own date (a day, or a range wholly inside the window) is sealed, with the mask on."""
    if not sealed.masked() or not filters:
        return False
    get = (lambda k: filters.get(k)) if isinstance(filters, Mapping) else (lambda k: getattr(filters, k, None))
    on_date = get("on_date")
    if on_date:
        return sealed.is_sealed(on_date)
    return sealed.inside(get("from_date"), get("to_date"))


def scope_answer(question: str, provenance: Provenance, *, detail: str = "") -> Answer:
    """The honest answer to a question about the sealed window: the record holds it, and it is not yet available."""
    return Answer(question=question, provenance=provenance, abstention=Abstention.OUTSIDE_WINDOW,
                  abstention_detail=detail or sealed.SCOPE_STATEMENT, coverage_note=sealed.SCOPE_STATEMENT,
                  rendered=sealed.SCOPE_STATEMENT)  # fmt: skip


# --- what an answer holds, and when each thing is dated ------------------------------------------------------------


def _ids(answer: Answer) -> tuple[set[str], set[str], set[int]]:
    speeches = {c.speech_id for claim in answer.claims for c in claim.citations}
    debates = set(answer.roster.debate_ids) if answer.roster else set()
    chunks = {int(i) for claim in answer.claims for c in claim.citations for i in c.chunk_ids}
    chunks |= {int(i) for i in answer.retrieved_chunk_ids}
    return speeches, debates, chunks


async def record_dates(conn: Any, answer: Answer) -> dict[str, dict[Any, date]]:
    """The record's own date for every speech an answer cites, debate it lists and chunk it read or cites."""
    speeches, debates, chunks = _ids(answer)
    out: dict[str, dict[Any, date]] = {"speech": {}, "debate": {}, "chunk": {}}
    if speeches:
        # `public.`, always: on a serving connection the served views hide exactly the rows this must see.
        cur = await conn.execute("SELECT s.id, d.date FROM public.speech s JOIN public.debate d ON d.id = s.debate_id "
                                 "WHERE s.id = ANY(%s)", (sorted(speeches),))
        out["speech"] = {r[0]: r[1] for r in await cur.fetchall()}
    if debates:
        cur = await conn.execute("SELECT id, date FROM public.debate WHERE id = ANY(%s)", (sorted(debates),))
        out["debate"] = {r[0]: r[1] for r in await cur.fetchall()}
    if chunks:
        cur = await conn.execute("SELECT id, debate_date FROM public.chunk WHERE id = ANY(%s)", (sorted(chunks),))
        out["chunk"] = {int(r[0]): r[1] for r in await cur.fetchall()}
    return out


def _dates_within(value: Any) -> Iterator[tuple[str, date]]:
    """Every date held anywhere inside a nested value: date objects, ISO strings, and day-level dates in text."""
    if isinstance(value, (date, datetime)):
        day = value.date() if isinstance(value, datetime) else value
        yield day.isoformat(), day
    elif isinstance(value, str):
        if sealed.is_sealed(value):
            yield value[:10], date.fromisoformat(value[:10])
        for day in sealed.sealed_days_in(value):
            yield day.isoformat(), day
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _dates_within(v)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for v in value:
            yield from _dates_within(v)


def reasons(answer: Answer, dates: Mapping[str, Mapping[Any, Any]]) -> list[str]:
    """Why an answer must be withheld: each thing it holds that is dated inside the window. Empty when it is clean, and
    always empty while the exam has lifted the mask.
    """
    if not sealed.masked():
        return []
    out: list[str] = []

    def add(reason: str) -> None:
        if reason not in out:
            out.append(reason)

    for claim in answer.claims:
        if sealed.is_sealed(claim.date):
            add(f"claim {claim.id} is dated {claim.date}")
        for citation in claim.citations:
            day = dates.get("speech", {}).get(citation.speech_id)
            if sealed.is_sealed(day):
                add(f"claim {claim.id} cites speech {citation.speech_id}, dated {day}")
            for chunk in citation.chunk_ids:
                day = dates.get("chunk", {}).get(int(chunk))
                if sealed.is_sealed(day):
                    add(f"claim {claim.id} cites chunk {chunk}, dated {day}")
    for chunk in sorted(int(i) for i in answer.retrieved_chunk_ids):
        day = dates.get("chunk", {}).get(chunk)
        if sealed.is_sealed(day):
            add(f"retrieved chunk {chunk}, dated {day}")
    if answer.roster:
        for debate_id in answer.roster.debate_ids:
            day = dates.get("debate", {}).get(debate_id)
            if sealed.is_sealed(day):
                add(f"the roster lists debate {debate_id}, dated {day}")
    for fact in answer.structured_facts:
        for text, day in _dates_within([fact.value, fact.params]):
            if sealed.is_sealed(day):
                add(f"structured fact {fact.template} holds {text}")
    for text, day in _dates_within([answer.facets]):
        if sealed.is_sealed(day):
            add(f"facets hold {text}")
    for day in sealed.sealed_days_in(" ".join((answer.rendered, answer.coverage_note, answer.abstention_detail))):
        add(f"the text names {day.isoformat()}")
    return out[:MAX_REASONS]


async def guard(conn: Any, answer: Answer | None) -> tuple[Answer | None, list[str]]:
    """The exit check. Returns the answer unchanged with no reasons, or the scope answer with the reasons it replaced
    the original for.
    """
    if answer is None or not sealed.masked():
        return answer, []
    try:
        dates = await record_dates(conn, answer)
    except Exception as exc:  # noqa: BLE001 - an answer that cannot be checked is withheld, never shown on trust
        why = [f"could not read the record's dates to check the answer ({type(exc).__name__})"]
        log.error("the sealed-window guard withheld an answer: %s", why[0])
        return scope_answer(answer.question, answer.provenance), why
    why = reasons(answer, dates)
    if not why:
        return answer, []
    log.error("sealed-window mask breach caught at the exit (an inner layer failed): %s", "; ".join(why[:3]))
    return scope_answer(answer.question, answer.provenance), why
