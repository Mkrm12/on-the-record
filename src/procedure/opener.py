"""Who opens a debate: never the Chair, never a point of order raised before the business."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date as Date

# The roles the classifier gives the Chair (`ingest.rules`).
PRESIDING_ROLES = ("speaker", "deputy_speaker", "chair", "lord_speaker", "lords_deputy")

# The Chair's printed attribution, matched against the attribution with every space and full stop removed and folded to
# lower case ("Madam Deput y Speaker (Caroline Nokes)" -> "madamdeputyspeaker(carolinenokes)").
_PRINTED = re.compile(
    r"^(?:mr|madam)(?:deputy)?speaker(?:-elect)?(?:\(|$)"
    r"|^the(?:senior)?(?:deputy)?speaker(?:\(|$)"
    r"|^thelordspeaker(?:\(|$)"
    r"|^the(?:principal)?(?:deputy)?chairmanofcommittees(?:\(|$)"
    r"|^(?:the)?(?:first|second|temporary)?(?:deputy)?chair(?:man)?ofwaysandmeans(?:\(|$)"
    r"|^the(?:temporary|(?:first|second)?deputy)?chair(?:man)?(?:\(|$)"
    r"|\(inthechair\)")

_FOLD = re.compile(r"[\s.]+")

# An attribution to no one in particular ("Hon.
_COLLECTIVE = re.compile(r"^(?:several\s+|some\s+|an?\s+)?(?:right\s+)?(?:hon(?:ourable)?\.?|noble)\s+(?:members?|lords?|"
                         r"friends?|ladies|gentlemen)\b|^(?:members|lords)\b", re.I)

# How a point of order is raised: at the very start of the contribution.
_POINT_OF_ORDER = re.compile(r"^\W*(?:i\s+rise\s+)?on\s+a\s+(?:further\s+|separate\s+|brief\s+)?point\s+of\s+order\b"
                             r"|^\W*further\s+to\s+(?:[\w\u2019'.-]+\s+){0,6}?point\s+of\s+order\b", re.I)
_POINT_OF_ORDER_BUSINESS = re.compile(r"\bpoints?\s+of\s+order\b", re.I)


def presiding(printed: str | None, role: str | None) -> bool:
    """Whether a contribution is the Chair's: its role, or the Chair's printed attribution in any spacing."""
    if (role or "") in PRESIDING_ROLES:
        return True
    folded = _FOLD.sub("", (printed or "").casefold()).replace("\u2019", "'")
    return bool(folded) and bool(_PRINTED.search(folded))


# Which presiding role a printed form is, in the classifier's vocabulary (`ingest.rules`), first match wins.
_ROLE_OF = ((re.compile(r"^(?:mr|madam)speaker(?:-elect)?(?:\(|$)|^thespeaker(?:\(|$)"), "speaker"),
            (re.compile(r"^thelordspeaker(?:\(|$)"), "lord_speaker"),
            (re.compile(r"^the(?:senior|second|third|principal)?deputy(?:speaker|chairmanofcommittees)(?:\(|$)"
                        r"|^thechairmanofcommittees(?:\(|$)"), "lords_deputy"),
            (re.compile(r"^(?:mr|madam)deputyspeaker(?:\(|$)|waysandmeans(?:\(|$)"
                        r"|^the(?:first|second|third)deputychair(?:man)?(?:\(|$)"), "deputy_speaker"))


def presiding_role(printed: str | None) -> str | None:
    """The presiding role a printed attribution names, or None when it is not the Chair's."""
    if not presiding(printed, None):
        return None
    folded = _FOLD.sub("", (printed or "").casefold()).replace("\u2019", "'")
    return next((role for pattern, role in _ROLE_OF if pattern.search(folded)), "chair")


def collective(printed: str | None, mnis_id: int | None) -> bool:
    return mnis_id is None and bool(_COLLECTIVE.search((printed or "").strip()))


def point_of_order(text: str | None) -> bool:
    """Whether a contribution raises a point of order (or speaks further to one), judged by how it opens."""
    return bool(_POINT_OF_ORDER.search(text or ""))


def business_is_point_of_order(heading: str | None, debate_type: str | None) -> bool:
    return debate_type == "point_of_order" or bool(_POINT_OF_ORDER_BUSINESS.search(heading or ""))


# A heading that names the presiding Chair's own statement.
_SPEAKER_IN_HEADING = r"(?:the\s+)?(?:(?:mr|madam)\s+)?(?:(?:lord|senior|deputy|first|second|principal)\s+){0,2}speaker"
_CHAIR_OF_IN_HEADING = (r"(?:the\s+)?(?:(?:first|second|deputy|principal|temporary)\s+){0,2}chair(?:man)?\s+of\s+"
                        r"(?:ways\s+and\s+means|committees)")
_CHAIR_IN_HEADING = r"(?:the\s+)?(?:(?:first|second|deputy|principal|temporary)\s+){0,2}chair(?:man)?"
_ITS_STATEMENT = r"(?:['\u2019]s|s['\u2019]|['\u2019]|s)?\s*:?\s*statements?"
_CHAIR_BUSINESS = re.compile(
    rf"^(?:{_SPEAKER_IN_HEADING}|{_CHAIR_OF_IN_HEADING}){_ITS_STATEMENT}\b"
    rf"|^{_CHAIR_IN_HEADING}{_ITS_STATEMENT}\s*$"
    rf"|^statements?\s+(?:by|from)\s+(?:{_SPEAKER_IN_HEADING}|{_CHAIR_OF_IN_HEADING}|{_CHAIR_IN_HEADING})\s*$", re.I)


def business_of_the_chair(heading: str | None, debate_type: str | None) -> bool:
    """Whether the business is the Chair's own statement, by the classifier's type or by its heading."""
    return debate_type == "speaker_statement" or bool(_CHAIR_BUSINESS.search(" ".join((heading or "").split())))


# Markup as HTML reads it: a comment first (it may hold ">"), a declaration, or a tag, which starts "<" or "</" and a
# letter (a bare "<" is text, and "</ em>" is no end tag); and an italic tag, open or close, in which Hansard prints a
# stage direction.
_MARKUP = re.compile(r"<!--.*?-->|<![^>]*>|</?[A-Za-z][^>]*>", re.S)
_ITALIC_TAG = re.compile(r"<(/?)(?:em|i)\b[^>]*>", re.I)
# The acts that open business, however printed: moving it, asking a question ("Asked by", "To ask His Majesty's
# Government"), presenting a bill or a petition.
_OPENING_ACT = re.compile(r"\b(?:mov(?:e|ed|es|ing)|ask(?:s|ed|ing)?|present(?:s|ed|ing)?)\b", re.I)
# A stage direction is a phrase ("King's consent signified." is three words); a longer run of italics is a passage (a
# quotation, words spoken in Welsh) and is read as a speech.
STAGE_DIRECTION_WORDS = 12
# A note in square brackets is the reporter's ("[Inaudible.]", "[Interruption.]"), innermost first; one never closed
# runs to the end of the contribution.
_NOTE = re.compile(r"\[[^\[\]]*\]")


def _only_notes(said: str) -> bool:
    """Whether every word of `said` is inside the reporter's square-bracketed notes. Such a note, standing alone in a
    member's place, stands where the member's words were: they had the floor, so it is a speech.
    """
    rest, before = said, None
    while rest != before:
        before, rest = rest, _NOTE.sub(" ", rest)
    rest = rest.split("[", 1)[0]
    return any(ch.isalpha() for ch in said) and not any(ch.isalnum() for ch in rest)


def _words(pieces: list[str]) -> str:
    return " ".join(html.unescape(" ".join(pieces)).split())


def _italic_split(markup: str) -> tuple[str, str]:
    """(the words outside italics, the words inside them). Every tag is a word boundary (QA pass 7: runs of italics
    glued together lost "Moved by"); an italic close of either kind closes either; words after an italic tag that is
    never closed are not shown to be italic, so they count as outside.
    """
    plain: list[str] = []
    italic: list[str] = []
    pending: list[str] = []
    depth, at = 0, 0
    for tag in _MARKUP.finditer(markup):
        (pending if depth else plain).append(markup[at:tag.start()])
        at = tag.end()
        italic_tag = _ITALIC_TAG.fullmatch(tag.group(0))
        if italic_tag and italic_tag.group(1):
            if depth:
                depth -= 1
                if not depth:
                    italic.extend(pending)
                    pending = []
        elif italic_tag:
            depth += 1
        (pending if depth else plain).append(" ")
    (pending if depth else plain).append(markup[at:])
    return _words(plain + pending), _words(italic)


# A contribution's three states: words the member said; nothing the member said; or the floor with the words not
# recorded (a gap in the transcript, noted by the reporter where the words were).
SPEECH, NOTHING, UNRECORDED = "speech", "nothing", "unrecorded"
# The reporter's notes are printed in square brackets; one standing alone may be in round or full-width ones too.
_FULL_WIDTH = str.maketrans({"［": "[", "］": "]", "（": "(", "）": ")"})
_ROUND_NOTE = re.compile(r"\(([^()]*)\)")
# A note that says the member's words were not captured: the transcript has a gap where they spoke.
_TRANSCRIPT_GAP = re.compile(r"\b(?:inaudible|indistinct|unclear|no\s+audio|audio\s+(?:lost|unavailable|cut|failed|"
                             r"failure|dropped)|not\s+(?:recorded|audible|heard|caught|transcribed)|connection\s+(?:lost|"
                             r"failed|dropped|broken)|technical\s+(?:difficult\w*|problems?|faults?|issues?))\b",
                             re.I)  # fmt: skip
# A note that reports someone speaking ("Hon.
_SPEAKER_LABEL = re.compile(r"^\s*[\w.'’ -]{1,40}:\s*[\"“‘']")


def _notes_alone(text: str, glue: bool = False) -> list[str] | None:
    """The notes' words when the whole contribution, markup aside, is the reporter's notes and nothing else (square
    brackets, or one note in round or full-width ones); None when it holds anything more. Tags are word boundaries,
    or with `glue` nothing at all: markup cannot say whether two runs are one word ("<em>In</em><em>audible</em>")
    or two ("<em>Moved</em><em>by</em>", QA pass 7), so the gap test reads both (QA pass 9, Med 5).
    """
    whole = _words([_MARKUP.sub("" if glue else " ", text)]).translate(_FULL_WIDTH).strip()
    round_note = _ROUND_NOTE.fullmatch(whole)
    if round_note:
        return [round_note.group(1)]
    if whole and _only_notes(whole):
        return [part for part in re.split(r"[\[\]]", whole) if part.strip()]
    return None


def contribution_state(text: str | None) -> str:
    """What a whole contribution is. The reporter's notes alone are judged by what they say: a gap in the transcript
    ("[Inaudible.]", "(Indistinct)", "[Connection lost.]") is the floor with the words not recorded, which counts as
    speaking; any other note alone ("[Laughter.]", "[Hon.
    """
    notes = _notes_alone(text or "")
    if notes is not None:
        read_both_ways = notes + (_notes_alone(text or "", glue=True) or [])
        gap = any(_TRANSCRIPT_GAP.search(note) and not _SPEAKER_LABEL.match(note) for note in read_both_ways)
        return UNRECORDED if gap else NOTHING
    plain, said = _italic_split(text or "")
    if plain:
        return SPEECH  # words outside italics, or in italics never closed: a speech
    stage = not (len(said.split()) > STAGE_DIRECTION_WORDS or _OPENING_ACT.search(said) or point_of_order(said))
    return NOTHING if stage else SPEECH


def non_speech(text: str | None) -> bool:
    """Whether a whole contribution says nothing a member said (`contribution_state`): the floor with the words not
    recorded is not nothing, the member had the floor.
    """
    return contribution_state(text) == NOTHING


@dataclass(frozen=True)
class Speech:
    debate_id: str
    seq: int
    speech_id: str
    mnis_id: int | None
    speaker_name: str | None
    speaker_role: str | None
    is_boilerplate: bool
    opening: str | None  # the contribution's first words: enough to tell a point of order
    whole: str | None = None  # the whole contribution where it is short enough to read whole (a stage direction is)

    @property
    def is_chair(self) -> bool:
        return presiding(self.speaker_name, self.speaker_role)

    @property
    def says_nothing(self) -> bool:
        """Empty, or stage directions alone. Only a contribution read whole can be judged so: a long one is a speech.
        """
        return self.whole is not None and non_speech(self.whole)

    @property
    def words_unrecorded(self) -> bool:
        """The floor with the words not recorded: the reporter's note of a gap alone."""
        return self.whole is not None and contribution_state(self.whole) == UNRECORDED


@dataclass(frozen=True)
class Found:
    """A debate's opener, or None, and what the record shows before it."""

    speech: Speech | None
    chair_first: bool = False  # the record's first contribution is the Chair's
    points_of_order_before: int = 0  # points of order raised before the business, passed over
    chair_business: bool = False  # the business is the Chair's own statement: no member opened it
    statement: Speech | None = None  # that statement, where the record holds it


def choose(speeches: list[Speech], *, business_is_point_of_order: bool, business_of_the_chair: bool = False) -> Found:
    """The opener of one record: its first contribution by an individual who is not presiding and that says something,
    passing over points of order raised before the business unless the business is one (or the record holds nothing
    else). Where the business is the Chair's own statement, nobody opened it.
    """
    ordered = sorted(speeches, key=lambda s: s.seq)
    chair_first = bool(ordered) and ordered[0].is_chair
    if business_of_the_chair:
        return Found(None, chair_first, chair_business=True, statement=next((s for s in ordered if s.is_chair), None))
    eligible = [s for s in ordered if not s.is_chair and not collective(s.speaker_name, s.mnis_id)
                and not s.says_nothing]  # fmt: skip
    if not eligible:
        return Found(None, chair_first)
    if business_is_point_of_order:
        return Found(eligible[0], chair_first)
    for passed, speech in enumerate(eligible):
        if not point_of_order(speech.opening):
            return Found(speech, chair_first, passed)
    return Found(eligible[0], chair_first)


def members_who_spoke(speeches: list[Speech]) -> int:
    """How many members contributed: never the Chair, never a collective attribution, never a boilerplate line, never a
    contribution that says nothing.
    """
    return len({s.mnis_id for s in speeches
                if s.mnis_id is not None and not s.is_chair and not s.is_boilerplate and not s.says_nothing})


# --- the database --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One debate record and its speeches, in the record's order."""

    debate_id: str
    house: str
    date: Date
    heading: str | None
    debate_type: str | None
    section: str | None
    section_order: int | None
    speeches: tuple[Speech, ...]

    @property
    def found(self) -> Found:
        return choose(list(self.speeches),
                      business_is_point_of_order=business_is_point_of_order(self.heading, self.debate_type),
                      business_of_the_chair=business_of_the_chair(self.heading, self.debate_type))

    @property
    def members(self) -> int:
        return members_who_spoke(list(self.speeches))


# Only a contribution's opening is read: a point of order is told by its first words.
OPENING_CHARS = 160
WHOLE_CHARS = 400

_SQL = f"""
    SELECT d.id, d.house, d.date, d.heading, d.debate_type, d.section, d.section_order,
           s.seq, s.id, s.mnis_id, s.speaker_name, s.speaker_role, coalesce(s.is_boilerplate, false),
           left(s.text, {OPENING_CHARS}),
           CASE WHEN length(coalesce(s.text, '')) <= {WHOLE_CHARS} THEN coalesce(s.text, '') END
    FROM debate d
    LEFT JOIN speech s ON s.debate_id = d.id
    WHERE (%(all)s OR d.id = ANY(%(ids)s))
    ORDER BY d.id, s.seq
"""


async def records(conn, debate_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, Record]:
    """Each debate's record with its speeches; every debate the connection can see when `debate_ids` is None (the
    invariant gate). On a served connection that is the masked record.
    """
    cur = await conn.execute(_SQL, {"all": debate_ids is None, "ids": list(debate_ids or ())})
    return _assemble(await cur.fetchall())


def records_sync(conn, debate_ids: list[str] | tuple[str, ...] | None = None) -> dict[str, Record]:
    """`records` on a synchronous connection (the invariant gate)."""
    return _assemble(conn.execute(_SQL, {"all": debate_ids is None, "ids": list(debate_ids or ())}).fetchall())


def _assemble(rows) -> dict[str, Record]:
    meta: dict[str, tuple] = {}
    speeches: dict[str, list[Speech]] = {}
    for (did, house, day, heading, kind, section, order, seq, sid, mnis, name, role, boiler, opening, whole) in rows:
        meta.setdefault(did, (house, day, heading, kind, section, order))
        bucket = speeches.setdefault(did, [])
        if sid is not None:
            bucket.append(Speech(did, seq if seq is not None else 0, sid, mnis, name, role, bool(boiler), opening,
                                 whole))
    return {did: Record(did, *meta[did], tuple(speeches[did])) for did in meta}


async def openers(conn, debate_ids: list[str] | tuple[str, ...]) -> dict[str, Found]:
    return {did: rec.found for did, rec in (await records(conn, debate_ids)).items()}
