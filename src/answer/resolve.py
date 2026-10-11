"""Turning a question's words into the parameters a structured fact needs."""

from __future__ import annotations

import logging
import re
from datetime import date as Date
from typing import Any

log = logging.getLogger("answer.resolve")

# Words that carry no signal when matching a division title.
_NOISE = frozenset("""
the a an of on in at to for and or was were is are be been being that this those these it its
motion question division vote voted carried agreed rejected negatived passed
lords commons house did was what were who when which how many number
""".split())

_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*", re.I)
# "New Clause 6", "Amendment 12", "No. 2" — the fragments that actually distinguish two divisions on one day.
_DISTINCTIVE = re.compile(
    r"\b(new\s+clause|new\s+schedule|clause|amendment|schedule|ways\s+and\s+means\s+motion|motion|"
    r"standing\s+order(?:\s+no\.?)?|instruction|la|no\.?)\s*(\d+)\b", re.I)  # fmt: skip


def _marks(text: str) -> dict[str, set[str]]:
    """Distinctive fragments grouped BY KIND: {"new clause": {"3"}, "no": {"2"}}."""
    found: dict[str, set[str]] = {}
    for match in _DISTINCTIVE.finditer(text or ""):
        kind = re.sub(r"[\s.]+", " ", match.group(1) or "").strip().lower().rstrip(".")
        # `LA62` in a title and "Lords Amendment 62" in a question are the same reference, so they must land in the same
        # bucket or the comparison reads a contradiction where there is agreement.
        if kind == "la":
            kind = "amendment"
        # "Standing Order No. 24" and "Standing Order 24" likewise.
        elif kind.startswith("standing order"):
            kind = "standing order"
        found.setdefault(kind, set()).add(match.group(2))
    return found


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in _NOISE and len(w) > 2}


def title_score(question: str, title: str | None) -> float:
    """How well a division title matches the question: the share of the title's own terms that the question uses."""
    title_terms = _terms(title or "")
    if not title_terms:
        return 0.0
    asked = _terms(question)
    overlap = len(title_terms & asked) / len(title_terms)

    title_marks = _marks(title or "")
    asked_marks = _marks(question)
    if title_marks:
        shared = set(title_marks) & set(asked_marks)
        agrees = any(title_marks[k] & asked_marks[k] for k in shared)
        conflicts = any(not (title_marks[k] & asked_marks[k]) for k in shared)
        if conflicts:
            # A CONTRADICTION, not a gap.
            overlap -= 0.60
        elif agrees:
            overlap += 0.5
        elif not shared:
            overlap -= 0.35
    return max(0.0, overlap)


# Courtesy titles that `member_id_map` stores and a question omits.
_HONORIFIC = r"^(?:The\s+)?(?:Rt\s+Hon\s+)?(?:Dr|Sir|Dame|Mr|Mrs|Ms|Miss|Professor|Prof)\.?\s+"


# A resolution must beat the runner-up by this much to be used.
MIN_MARGIN = 0.15
# And it must clear this on its own, or the question is not about any of them.
MIN_SCORE = 0.30


async def _all(conn, sql: str, params: tuple = ()) -> list[dict]:
    cur = await conn.execute(sql, params)
    columns = [c.name for c in cur.description or []]
    return [dict(zip(columns, row, strict=True)) for row in await cur.fetchall()]


async def division_id(conn, question: str, *, house: str | None = None, when: Date | None = None,
                      number: int | None = None) -> tuple[int | None, str]:  # fmt: skip
    """The division the question means, or None and a reason."""
    if when is None:
        return None, "no date in the question, and a division cannot be identified by title alone"
    rows = await _all(conn, """
        SELECT id, house, number, title, division_date
        FROM division
        WHERE division_date = %s AND (%s::text IS NULL OR house = %s)
        ORDER BY number NULLS LAST, id
    """, (when, house, house))
    if not rows:
        return None, f"no division is recorded on {when}" + (f" in the {house}" if house else "")

    if number is not None:
        exact = [r for r in rows if r["number"] == number]
        if len(exact) == 1:
            return exact[0]["id"], f"division {number} on {when}"
    if len(rows) == 1:
        return rows[0]["id"], f"the only division on {when}" + (f" in the {house}" if house else "")

    scored = sorted(((title_score(question, r["title"]), r) for r in rows), key=lambda pair: -pair[0])
    best, runner = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
    if best[0] < MIN_SCORE:
        return None, (f"{len(rows)} divisions on {when} and the question matches none of them well "
                      f"(best {best[0]:.2f})")  # fmt: skip
    if best[0] - runner[0] < MIN_MARGIN:
        return None, (f"{len(rows)} divisions on {when} and two match equally well "
                      f"({best[0]:.2f} vs {runner[0]:.2f}); the question does not say which")  # fmt: skip
    return best[1]["id"], f"matched {best[1]['title']!r} on {when} (score {best[0]:.2f})"


async def mnis_id(conn, question: str) -> tuple[int | None, str]:
    """The member the question names, or None."""
    # The fourth clause strips a leading courtesy title from the STORED name.
    candidates = await _all(conn, """
        SELECT mnis_id, name, name_list_as, name_full_title,
               regexp_replace(name, %s, '', 'i') AS name_bare
        FROM member_id_map
        WHERE %s ILIKE '%%' || name || '%%'
           OR (name_list_as IS NOT NULL AND %s ILIKE '%%' || name_list_as || '%%')
           OR (name_full_title IS NOT NULL AND %s ILIKE '%%' || name_full_title || '%%')
           OR %s ILIKE '%%' || regexp_replace(name, %s, '', 'i') || '%%'
    """, (_HONORIFIC, question, question, question, question, _HONORIFIC))
    if not candidates:
        return None, "no member named in the question matches the member record"
    # Longest matched name wins: "Baroness Featherstone" over "Featherstone", and it prevents a short name that is a
    # substring of a longer one ("Ali" inside "Alison") from taking the match.
    def matched_length(row: dict) -> int:
        """The length of the name as the QUESTION spells it, which is what the tie-break is really about."""
        asked = question.lower()
        return max((len(n) for n in (row.get("name"), row.get("name_bare"), row.get("name_list_as"),
                                     row.get("name_full_title")) if n and n.lower() in asked), default=0)  # fmt: skip

    best = max(candidates, key=matched_length)
    same = [r for r in candidates if matched_length(r) == matched_length(best) and r["mnis_id"] != best["mnis_id"]]
    same = [best, *same]
    if len(same) > 1:
        return None, f"{len(same)} members match that name; the question does not say which"
    return best["mnis_id"], f"matched {best['name']}"


async def office_key(conn, question: str) -> tuple[str | None, str]:
    """The office the question names, as `member_office.office_key`."""
    # `length(office)` is in the select list because this is a SELECT DISTINCT: Postgres refuses to order a DISTINCT by
    # an expression it cannot see, and the omission failed every single `office_held_at_date` item — twelve of them on
    # dev — silently, until the trace table recorded the reason.
    rows = await _all(conn, """
        SELECT DISTINCT office_key, office, length(office) AS office_length
        FROM member_office
        WHERE office IS NOT NULL AND %s ILIKE '%%' || office || '%%'
        ORDER BY office_length DESC, office LIMIT 5
    """, (question,))
    if not rows:
        return None, "no office named in the question matches one in the record"
    return rows[0]["office_key"], f"matched {rows[0]['office']!r}"


async def bill_candidates(conn, question: str) -> list[dict]:
    """Every bill whose title the question contains, longest title first."""
    return await _all(conn, """
        SELECT id, title FROM bill
        WHERE title IS NOT NULL AND %s ILIKE '%%' || title || '%%'
        ORDER BY length(title) DESC, id LIMIT 8
    """, (question,))


async def bill_id(conn, question: str) -> tuple[int | None, str]:
    """The bill the question names, matched on title within the corpus's 429 bills."""
    rows = await bill_candidates(conn, question)
    if rows:
        return rows[0]["id"], f"matched {rows[0]['title']!r}"
    scored = sorted(((title_score(question, r["title"]), r)
                     for r in await _all(conn, "SELECT id, title FROM bill")), key=lambda p: -p[0])  # fmt: skip
    if scored and scored[0][0] >= 0.6 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= MIN_MARGIN):
        return scored[0][1]["id"], f"matched {scored[0][1]['title']!r} (score {scored[0][0]:.2f})"
    return None, "no bill title in the record matches the question closely enough"


_STAGE = re.compile(
    r"\b(second reading|third reading|first reading|committee stage|report stage|"
    r"order of commitment discharged|consideration of commons amendments|royal assent|"
    r"legislative grand committee|money resolution|ways and means)\b", re.I)  # fmt: skip

_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
            "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}  # fmt: skip


def division_number(question: str) -> int | None:
    """An explicit division number, or an ordinal like "the third division"."""
    explicit = re.search(r"\bdivision\s*(?:no\.?|number)?\s*(\d{1,4})\b", question, re.I)
    if explicit:
        return int(explicit.group(1))
    ordinal = re.search(r"\b(" + "|".join(_ORDINAL) + r")\s+division\b", question, re.I)
    return _ORDINAL[ordinal.group(1).lower()] if ordinal else None


# "second reading" written by a person, "2nd reading" recorded by Parliament.
_ORDINALS = {"first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
             "1st": "1st", "2nd": "2nd", "3rd": "3rd", "4th": "4th", "5th": "5th"}  # fmt: skip


# Filler in a question ABOUT a stage.
_STAGE_NOISE = frozenset({
    "what", "was", "the", "when", "did", "date", "took", "take", "of", "on", "in", "for", "its", "it", "bill",
    "act", "and", "which", "that", "this", "have", "has", "had", "does", "do", "how", "who", "reach", "reached",
})

_ORDINAL_FORMS = frozenset({"1st", "2nd", "3rd", "4th", "5th"})


def _stage_terms(text: str) -> set[str]:
    """Words that identify a stage, with ordinals normalised and only stage-question filler dropped."""
    out = {_ORDINALS.get(w, w) for w in _WORD.findall((text or "").lower())}
    return {w for w in out
            if w not in _STAGE_NOISE and (len(w) > 2 or w in _ORDINAL_FORMS)}  # fmt: skip


def stage_name_offline(question: str) -> str | None:
    """The regex form, kept only for the pure tests and for a caller with no connection."""
    found = _STAGE.search(question)
    return found.group(1).title() if found else None


async def stage_name(conn, question: str, *, bill_id_: int | None = None) -> tuple[str | None, str]:
    """The stage the question names, as `bill_stage.stage_name` spells it."""
    rows = await _all(conn, """
        SELECT DISTINCT stage_name FROM bill_stage
         WHERE stage_name IS NOT NULL AND (%s::int IS NULL OR bill_id = %s)
    """, (bill_id_, bill_id_))
    if not rows:
        return None, ("this bill has no stages recorded" if bill_id_ else "no stages are recorded")

    asked = _stage_terms(question)
    if not asked:
        return None, "the question names no stage"

    def score(name: str) -> float:
        terms = _stage_terms(name)
        if not terms:
            return 0.0
        # Normalised on the STAGE's terms, so a long question cannot out-score a short one, and a stage whose every word
        # the question uses wins over a longer stage that merely shares a word.
        return len(terms & asked) / len(terms)

    scored = sorted(((score(r["stage_name"]), r["stage_name"]) for r in rows), key=lambda pair: (-pair[0], pair[1]))
    best = scored[0]
    runner = scored[1] if len(scored) > 1 else (0.0, None)
    if best[0] < 0.5:
        return None, f"the question names no stage this bill reached (best {best[1]!r} at {best[0]:.2f})"
    if best[0] - runner[0] < 0.01 and runner[1] is not None:
        return None, (f"the question matches {best[1]!r} and {runner[1]!r} equally well; it does not say which")
    return best[1], f"matched {best[1]!r} (score {best[0]:.2f})"


async def parameters(conn, category: str, question: str, filters: Any) -> tuple[dict[str, Any], list[str]]:
    """The parameters `answer.facts` needs for this category, and a note per resolution."""
    house = getattr(filters, "house", None)
    when = getattr(filters, "on_date", None)
    from_date = getattr(filters, "from_date", None)
    notes: list[str] = []

    if category in ("division_result", "division_tally", "division_vote", "division_debate"):
        found, why = await division_id(conn, question, house=house, when=when,
                                       number=division_number(question))  # fmt: skip
        notes.append(f"division_id: {why}")
        if found is None:
            return {}, notes
        params: dict[str, Any] = {"division_id": found}
        if category == "division_vote":
            member, why = await mnis_id(conn, question)
            notes.append(f"mnis_id: {why}")
            if member is None:
                return {}, notes
            params["mnis_id"] = member
        return params, notes

    if category == "party_at_date":
        member, why = await mnis_id(conn, question)
        notes.append(f"mnis_id: {why}")
        if member is None or when is None:
            notes.append("a date is required for party at a date" if when is None else "")
            return {}, notes
        return {"mnis_id": member, "date": when}, notes

    if category == "office_held_at_date":
        key, why = await office_key(conn, question)
        notes.append(f"office_key: {why}")
        if key is None or when is None:
            notes.append("a date is required for office at a date" if when is None else "")
            return {}, notes
        return {"office_key": key, "date": when}, notes

    if category == "bill_stage_date":
        # Bill and stage are resolved TOGETHER, because the question constrains both: "the Order of Commitment
        # discharged of X" names a bill that has one, and 13 titles here belong to more than one bill.
        candidates = await bill_candidates(conn, question)
        notes.append(f"bill: {len(candidates)} title match(es)")
        bill, stage, stage_why = None, None, "no bill matched"
        holders: list[tuple[int, str]] = []
        for candidate in candidates:
            found, why_stage = await stage_name(conn, question, bill_id_=candidate["id"])
            if found is not None:
                holders.append((candidate["id"], found))
            elif bill is None:
                bill, stage_why = candidate["id"], why_stage
        if len(holders) == 1:
            bill, stage = holders[0]
            stage_why = f"matched {stage!r} on bill {bill}"
        elif len(holders) > 1:
            # Several same-titled bills hold the stage named.
            bill, stage = None, None
            stage_why = (f"{len(holders)} bills share this title and each reached "
                         f"{holders[0][1]!r}; the question does not say which")
        notes.append(f"stage: {stage_why}")
        if bill is None or stage is None:
            return {}, notes
        return {"bill_id": bill, "stage": stage, "house": house}, notes

    if category == "count":
        # `count_any` routes on which parameters are present, so the resolver's job is to pass through what the question
        # actually said and let it refuse if that is not enough.
        to_date = getattr(filters, "to_date", None)
        params: dict[str, Any] = {}
        if house:
            params["house"] = house
        member, why = await mnis_id(conn, question)
        if member is not None:
            params["mnis_id"] = member
            notes.append(f"mnis_id: {why}")
        if member is not None and from_date and to_date:
            # A named member with a span is a contributions count, and it takes precedence: a range plus a person is
            # never a question about how many divisions there were.
            params["from"], params["to"] = from_date, to_date
        elif from_date and to_date and (to_date - from_date).days > 20:
            params["month"] = from_date
        elif when:
            params["date"] = when
        elif from_date:
            params["month"] = from_date
        notes.append(f"count parameters: {sorted(params)}")
        return (params if len(params) > (1 if "house" in params else 0) else {}), notes

    return {}, [f"no resolver for category {category!r}"]

async def phrase_speakers(conn, phrase: str, limit: int = 5000) -> int:
    """How many distinct members the record shows using these exact words. The chunk text index, a phrase query, and a
    cap on the rows read: the answer only has to clear a small threshold, never be exact at 467.
    """
    cur = await conn.execute("""
        SELECT count(DISTINCT coalesce(c.mnis_id::text, s.speaker_name))
        FROM (SELECT speech_id, mnis_id FROM chunk WHERE ts @@ phraseto_tsquery('english', %s) LIMIT %s) c
        JOIN speech s ON s.id = c.speech_id
    """, (phrase, limit))
    row = await cur.fetchone()
    return int(row[0] or 0)


async def division_candidates(conn, question: str, *, house: str | None = None, when: Date | None = None,
                              number: int | None = None) -> tuple[list[int], str]:  # fmt: skip
    """Every division the question could mean, best-matching first."""
    if when is None:
        return [], "no date in the question, and a division cannot be identified by title alone"
    rows = await _all(conn, """
        SELECT id, house, number, title FROM division
        WHERE division_date = %s AND (%s::text IS NULL OR house = %s)
        ORDER BY number NULLS LAST, id
    """, (when, house, house))
    if not rows:
        return [], f"no division is recorded on {when}" + (f" in the {house}" if house else "")
    if number is not None:
        exact = [r for r in rows if r["number"] == number]
        if len(exact) == 1:
            return [exact[0]["id"]], f"division {number} on {when}"
    if len(rows) == 1:
        return [rows[0]["id"]], f"the only division on {when}" + (f" in the {house}" if house else "")

    scored = sorted(((title_score(question, r["title"]), r) for r in rows), key=lambda pair: -pair[0])
    best = scored[0][0]
    if best < MIN_SCORE:
        return [], f"{len(rows)} divisions on {when} and the question matches none of them well (best {best:.2f})"
    tied = [r["id"] for value, r in scored if best - value < MIN_MARGIN]
    return tied, (f"{len(tied)} division(s) on {when} match the question equally well (score {best:.2f})"
                  if len(tied) > 1 else f"matched {scored[0][1]['title']!r} on {when} (score {best:.2f})")  # fmt: skip

