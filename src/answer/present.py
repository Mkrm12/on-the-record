"""Present-tense questions: answered as of the record's last sitting, and stamped so."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from answer.contract import Answer

# The categories that are a member's state on a day, so can be asked about "now".
STATE_CATEGORIES = frozenset({"office_held_at_date", "party_at_date"})

# Words that put a question in the present whatever its verb.
PRESENT_MARKER = re.compile(r"\b(?:now|nowadays|currently|current|presently|at\s+present|today|these\s+days)\b", re.I)
_PRESENT_VERB = re.compile(r"(?:\b(?:is|are|holds|occupies|fills|serves|belongs|represents|sits|does|do)\b"
                           r"|\b(?:who|what|which|where)['’]s\b)", re.I)  # fmt: skip
_PAST_VERB = re.compile(r"\b(?:was|were|did|had|held|occupied|filled|served|belonged|represented|sat|used\s+to|"
                        r"former|formerly|previous|previously|then)\b", re.I)  # fmt: skip


def _dated(filters: Any) -> bool:
    if not filters:
        return False
    get = (lambda k: filters.get(k)) if isinstance(filters, Mapping) else (lambda k: getattr(filters, k, None))
    return any(get(k) for k in ("on_date", "from_date", "to_date", "year"))


def asks_present(question: str, filters: Any = None) -> bool:
    """Whether an undated question asks about the present."""
    if _dated(filters):
        return False
    text = question or ""
    if _PAST_VERB.search(text):
        return False
    return bool(PRESENT_MARKER.search(text) or _PRESENT_VERB.search(text))


def _day(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def stamp(found: Answer, as_of: date) -> Answer:
    """The "As of" stamp, leading the answer, and the note on what it cannot know. Idempotent."""
    if found.facets.get("as_of"):
        return found
    lead = f"As of the record up to {_day(as_of)}, its most recent sitting:"
    note = (f"The record runs to {_day(as_of)}, its most recent sitting; it cannot say what has changed since, and it "
            f"reports what was recorded, not what is true today.")  # fmt: skip
    update: dict[str, Any] = {"coverage_note": " ".join(x for x in (note, found.coverage_note) if x)}
    if found.abstention is None:
        update["facets"] = {**found.facets, "as_of": as_of.isoformat()}
        # an answer whose text is built at the exit (a clarification) is built here, so the stamp never stands alone:
        # `reader_ready` fills only an empty text (review pass 1, S06)
        body = found.rendered or found.render()
        update["rendered"] = f"{lead} {body}".strip()
    return found.model_copy(update=update)
