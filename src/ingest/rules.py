"""The procedure rules engine."""

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from app.db import pool
from ingest.text import clean_text, split_sentences
from ingest.runtime import run

console = Console()

MAX_DESCRIPTION = 2000
MAX_PHRASE = 200

# --------------------------------------------------------------------------- Speaker attribution
# ---------------------------------------------------------------------------

PARTIES = frozenset({
    "Lab", "Con", "LD", "CB", "Lab/Co-op", "Lab Co-op", "DUP", "Non-Afl", "Ind", "GP", "Green",
    "Reform", "SNP", "PC", "UUP", "SDLP", "Alliance", "Your Party", "Bp", "Ref", "TUV", "Res",
    "Independent", "Ind Lab", "Ind Con", "Spk",
})  # fmt: skip

_HONORIFIC = re.compile(r"^(?:The )?(?:Lord|Baroness|Lady|Earl|Countess|Viscount|Baron|Duke|Duchess|Marquess|Sir|Dame|Dr|Mr|Mrs|Ms|Miss|Rt Hon)\b")
_PEER_PERSON = re.compile(r"^The (?:Earl|Countess|Viscount|Baron|Duke|Duchess|Marquess|Baroness|Lord Bishop|Bishop|Archbishop|Lord) ")
# A peer's name.
_LORD_OFFICE_WORD = (r"(?:Chancellor|President|Privy|Speaker|Chair|Chairman|Commissioner|Commissioners|Advocate|Chief|"
                     r"High|Great|Steward|Chamberlain|Lieutenant|Mayor|Justice|in\sWaiting)(?![\w-])")
_PEER_LEAD = re.compile(rf"^(?:(?:Lord|Baroness|Baron|Viscount|Viscountess|Earl|Countess)\s(?!{_LORD_OFFICE_WORD})"
                        r"|The\s(?:Earl|Countess|Duke|Duchess|Marquess|Marchioness|Viscount|Viscountess)\s)")  # fmt: skip
_LORDS_OFFICE = re.compile(r"^(?:The )?Lord (?:Speaker|Chancellor|Privy Seal|President|Chair(?:man)? of Committees|"
                           r"Commissioner|Advocate)\b")  # fmt: skip
_KNOWN_CHAIR_OFFICE = re.compile(r"^(?:Mr Speaker|Madam Speaker|(?:Mr|Madam) Deputy Speaker|Madam Chair(?:man)?|Mr Chairman)$")
_MINISTER_WORDS = re.compile(
    r"\b(?:Minister|Secretary of State|Chancellor|Attorney General|Solicitor General|Advocate General|"
    r"Under-Secretary|Secretary to the Treasury|(?:Economic|Exchequer|Financial|Chief|Parliamentary) Secretary|Whip|Lord in Waiting|Leader of the House|Paymaster General|Comptroller|Treasurer|"
    r"Vice-Chamberlain|Parliamentary Secretary|Lord Privy Seal|Lord President|Captain of the)\b"
)  # fmt: skip

CHAIR_ROLES = frozenset({"speaker", "deputy_speaker", "chair", "lord_speaker", "lords_deputy"})


class SpeakerInfo(NamedTuple):
    office: str | None  # "The Minister for Housing and Planning", "Mr Speaker"
    person: str | None  # "Matthew Pennycook"
    constituency: str | None  # Commons seat as printed
    party: str | None  # as printed: "Lab", "Lab/Co-op", "Non-Afl"
    role: str  # speaker|deputy_speaker|chair|lord_speaker|lords_deputy|minister|shadow_minister|office_holder|member|unknown
    is_chair: bool
    remote: bool  # attended remotely or virtually ("[R]", "[V]")
    flags: tuple[str, ...] = ()  # remote | virtual | maiden_speech | valedictory_speech


_UNKNOWN = SpeakerInfo(None, None, None, None, "unknown", False, False)
_TRAILING_GROUP = re.compile(r"\s*\(([^()]*)\)$")
_REMOTE = re.compile(r"\s*\[([A-Z])\]$")
_REMOTE_FLAGS = {"R": "remote", "V": "virtual"}
_SPEECH_FLAG = re.compile(r"^(Maiden|Valedictory) Speech$", re.I)


def _office_role(office: str) -> str:
    if re.match(r"^(?:Mr|Madam) Speaker$", office):
        return "speaker"
    if _KNOWN_CHAIR_OFFICE.match(office):
        return "deputy_speaker"
    if office == "The Lord Speaker":
        return "lord_speaker"
    if re.match(r"^The (?:First |Second |Third )?Deputy Chairman(?: of Ways and Means)?$", office):
        return "deputy_speaker"  # Commons: the Chairman of Ways and Means's deputies
    if re.match(r"^The Temporary Chair(?:man)? of Ways and Means$", office):
        return "deputy_speaker"  # QA pass 3: the temporary chair presides as the deputies do
    # "Principal Deputy Chairman of Committees" is the Lords' title before 2016, inside the ten-year span (QA pass 3)
    if re.match(r"^The (?:Senior |Second |Third |Principal )?Deputy (?:Speaker|Chairman of Committees|Chair)\b|^The Chairman of Committees\b", office):
        return "lords_deputy"
    if re.search(r"\bShadow\b", office):
        return "shadow_minister"
    if _MINISTER_WORDS.search(office):
        return "minister"
    return "office_holder"


@lru_cache(maxsize=8192)  # ~1,600 distinct attribution strings across 100k+ calls
def parse_speaker(name: str | None) -> SpeakerInfo:
    text = re.sub(r"\s+", " ", name or "").strip()  # \s covers the en-spaces (U+2002) Hansard uses
    if not text:
        return _UNKNOWN
    flags: list[str] = []
    if m := _REMOTE.search(text):
        flags.append(_REMOTE_FLAGS.get(m.group(1), "remote"))
        text = text[: m.start()]

    groups: list[str] = []
    while m := _TRAILING_GROUP.search(text):
        groups.insert(0, m.group(1).strip())
        text = text[: m.start()].strip()
    while groups and (f := _SPEECH_FLAG.match(groups[-1])):  # "(Maiden Speech)" is a flag, not a party
        flags.append(f"{f.group(1).lower()}_speech")
        groups.pop()
    lead = text

    party = None
    last = groups[-1] if groups else None
    if last is not None and not _HONORIFIC.match(last):
        # Decided by the SHAPE Hansard gives a label, not by a list of parties seen in 2020 and 2024-26: the list read
        # 2019's "(Independent Group for Change)" as the seat and lost the real one, and "Lord Pearson of Rannoch
        # (UKIP)" as a peer sitting for a constituency called UKIP.
        if last in PARTIES or len(groups) >= 2:
            party = groups.pop()  # "Name (Seat) (Party)": with two brackets the last is the party, whatever its name
        elif _PEER_LEAD.match(lead):
            party = groups.pop()  # a peer has no constituency, so a peer's one bracket is a party
    in_chair = bool(groups) and groups[-1].lower() == "in the chair"
    if in_chair:
        groups.pop()
    inner = groups[-1] if groups else None

    office = person = constituency = None
    if in_chair:
        person, role = lead, "chair"
    elif _KNOWN_CHAIR_OFFICE.match(lead) or _LORDS_OFFICE.match(lead):
        office, person = lead, inner
        role = _office_role(lead)
    elif inner is None and _PEER_PERSON.match(lead):
        person, role = lead, "member"  # "The Earl of Courtown": a name, not an office
    elif lead.startswith("The ") or (inner is not None and _HONORIFIC.match(inner)):
        office, person = lead, inner
        role = _office_role(lead)
    else:
        person, constituency, role = lead, inner, "member"

    remote = bool({"remote", "virtual"} & set(flags))
    return SpeakerInfo(office, person, constituency, party, role, role in CHAIR_ROLES, remote, tuple(flags))


# --------------------------------------------------------------------------- Motions and debates
# ---------------------------------------------------------------------------

_NEUTRAL_MOTION = re.compile(r"^\s*that\s+(?:this|the)\s+(?:house|committee|grand committee)\s+(?:has\s+considered|takes?\s+note|notes?\s+the\s+matter)\b", re.I)


def classify_motion(text: str | None) -> str | None:
    """'neutral' for general-debate motions ("has considered", "takes note") where the stance panel should show
    positions only; 'substantive' for everything else.
    """
    if not text or not text.strip():
        return None
    return "neutral" if _NEUTRAL_MOTION.match(text) else "substantive"


_DELEGATED = re.compile(r"^Draft\b|\b(?:Regulations|Order|Rules|Directions|Scheme|Code of Practice)\s+(?:20\d\d|\(No\.)|^Delegated Legislation$|\(Amendment\)|\(Amendment of", re.I)
_BILL_HEADING = re.compile(r"\bBill(?:\s*\[(?:HL|Lords|Commons)\])?\s*$")
_SPECIAL_HEADINGS = (
    (re.compile(r"^Engagements$", re.I), "prime_ministers_questions"),
    (re.compile(r"^Topical Questions$|^Oral Questions$", re.I), "topical_questions"),
    (re.compile(r"^(?:Business of the House|Arrangement of Business|Business Statement)$", re.I), "business_statement"),
    (re.compile(r"^(?:Deputy |Lord )?Speaker[’']s Statement", re.I), "speaker_statement"),
    (re.compile(r"^Points? of Order$", re.I), "point_of_order"),
    (re.compile(r"^(?:Deaths? of|Retirement of|Tributes? to|Introduction:)", re.I), "tributes"),
    (re.compile(r"^Humble Address", re.I), "humble_address"),
    # A new Parliament opening (W1) and pandemic proceedings (W3): headings that exist in no 13-day window, so without
    # them the most distinctive days in the corpus all fall through to "other" and the load report's rules coverage says
    # nothing about them.
    (re.compile(r"^Election of (?:a |the )?Speaker|^Speaker[- ]Elect", re.I), "speaker_election"),
    (re.compile(r"^Members? Sworn|^Oaths? and Affirmations?|^Swearing[- ]in", re.I), "members_sworn"),
    (re.compile(r"^Debate on the Address|^(?:King|Queen)[’']s Speech|^Address in Reply"
                r"|^(?:His|Her) Majesty[’']s Most Gracious Speech", re.I), "address_debate"),
    (re.compile(r"^Prorogation$|^Dissolution$|^State Opening", re.I), "opening_or_closing"),
    (re.compile(r"^Remote Division", re.I), "remote_division"),
    (re.compile(r"^(?:Hybrid|Virtual|Remote)[ \t]+(?:Substantive[ \t]+|Scrutiny[ \t]+)?Proceedings"
                r"|^Remote Voting$|Hybrid Scrutiny Proceedings$", re.I), "proceedings_arrangement"),
)  # fmt: skip
_EVENT_DEBATE_TYPES = (
    ("adjournment_motion", "adjournment"),
    ("urgent_question", "urgent_question"),
    ("leave_to_bring_in_bill", "ten_minute_rule"),
    ("statement_begin", "statement"),
    ("petition_presented", "petition"),
    ("question_asked", "oral_questions"),
    ("third_reading", "third_reading"),
    ("second_reading", "second_reading"),
)
_SECTION_DEBATE_TYPES = {
    "Westminster Hall": "westminster_hall_debate",
    "Grand Committee": "grand_committee",
    "General Committees": "general_committee",
    "Public Bill Committees": "public_bill_committee",
}


def classify_debate(heading: str | None, section: str | None, house: str | None, events: set[str] | frozenset[str]) -> str:
    """debate_type. Special headings win, then procedural events, then heading shape, then section."""
    heading = re.sub(r"\s+", " ", heading or "").strip()
    for pattern, kind in _SPECIAL_HEADINGS:
        if pattern.search(heading):
            return kind
    for event_type, kind in _EVENT_DEBATE_TYPES:
        if event_type in events:
            return kind
    if _DELEGATED.search(heading):
        return "delegated_legislation"
    if re.search(r"Committee Report\)$", heading):
        return "committee_report"
    if _BILL_HEADING.search(heading):
        return "bill"
    if "motion_moved" in events or "motion_stated" in events:
        return "general_debate"
    return _SECTION_DEBATE_TYPES.get(section or "", "other")


# --------------------------------------------------------------------------- Events
# ---------------------------------------------------------------------------


class Event(NamedTuple):
    event_type: str
    rule: str
    char_start: int  # offset into clean_text(speech.text)
    raw_phrase: str
    description: str | None
    motion_type: str | None


@dataclass(frozen=True)
class EventRule:
    name: str
    event_type: str
    pattern: re.Pattern
    group: str | None = None  # capture group whose text is the description
    chair_only: bool = False
    motion: bool = False  # description is motion text: classify it
    not_within: str | None = None  # skip if inside a match of this rule name


def _r(name, event_type, pattern, group=None, chair_only=False, motion=False, not_within=None, flags=re.M):
    return EventRule(name, event_type, re.compile(pattern, flags), group, chair_only, motion, not_within)


# Line-start anchors use (?m)^: the text is cleaned so each paragraph is one line.
_MOVE = r"(?:My Lords,[ \t]*)?I beg to move"
_THAT = r"(?:That|that)"
_BILL = r"(?:(?:the|this)[ \t]+)?Bill"
_STAGE_PREFIX = rf"(?:(?:{_MOVE}),?[ \t]*(?:{_THAT}[ \t]+)?|{_THAT}[ \t]+)?"

EVENT_RULES: tuple[EventRule, ...] = (
    # --- motions ------------------------------------------------------------
    _r("beg_to_move", "motion_moved",
       rf"^{_MOVE}(?:,?[ \t]*\n?[ \t]*(?P<motion>{_THAT}\b[^\n]*)|[.,]?[ \t]*$)", "motion", motion=True),
    _r("leave_to_bring_in", "leave_to_bring_in_bill",
       rf"^(?:{_MOVE},?[ \t]*\n?[ \t]*)?(?P<motion>That leave be given to bring in a Bill[^\n]*)", "motion", motion=True),
    _r("motion_stated", "motion_stated",
       r"^(?P<motion>That (?:this|the) (?:House|Committee|Grand Committee)\b[^\n]*)", "motion", motion=True, not_within="beg_to_move"),
    _r("amendment_moved", "amendment_proposed",
       rf"^{_MOVE}[ \t]+(?P<amend>(?:new clause|new schedule|amendment|clause)\b[^\n]*)", "amend"),
    _r("amendment_moving", "amendment_proposed", r"\bin moving (?P<amend>Amendment [0-9]+[A-Z]*)", "amend"),
    _r("amendment_stage_direction", "amendment_proposed", r"^Amendment proposed\b[:.]?[ \t]*(?P<amend>[^\n]*)", "amend"),
    # --- question stages ----------------------------------------------------
    _r("question_proposed", "question_proposed",
       r"^(?:Motion made, and )?Question proposed\b(?:[ \t]*[,.:]?[ \t]*(?P<q>That\b[^\n]*))?", "q"),
    _r("question_is", "question_proposed", r"^The Question is,?[ \t]+(?P<q>[Tt]hat\b[^\n]*)", "q", chair_only=True),
    _r("question_put", "question_put",
       r"^Question put\b(?![ \t]+and[ \t]+(?:agreed|negatived))[,.:]?[ \t]*(?P<q>[^\n]*)", "q"),
    _r("put_and_agreed", "agreed_without_division", r"^Question put and agreed to\b"),
    _r("put_and_negatived", "negatived_without_division", r"^Question put and negatived\b"),
    _r("accordingly_agreed", "question_agreed",
       r"^(?:Question accordingly (?:agreed to|resolved in the affirmative)|(?:On Question, )?Motion agreed)\b"),
    _r("accordingly_negatived", "question_negatived", r"^Question accordingly (?:negatived|resolved in the negative)\b"),
    # --- divisions ----------------------------------------------------------
    _r("house_divided", "division_called",
       r"^The (?:House|Committee|Grand Committee) (?:divided[:.]|proceeded to a Division\b)|^Division called\b"),
    _r("test_opinion", "division_called",
       r"(?:^|(?<=[.!?] ))I wish to test the opinion of the (?:House|Committee)\b"),
    _r("division_tally", "division_result",
       r"\b(?:Ayes\s+\d+,?\s+Noes\s+\d+|Contents?\s+\d+,?\s+Not[- ]Contents?\s+\d+)", flags=re.M | re.I),
    _r("division_no", "division_number", r"\bDivision No\.?[ \t]*(?P<n>\d+)", "n", flags=re.M | re.I),
    # --- resolutions --------------------------------------------------------
    _r("resolved", "resolved", r"^Resolved,?(?:[ \t]+|[ \t]*\n[ \t]*)(?P<t>[^\n]+)", "t"),
    _r("ordered", "ordered", r"^Ordered,?(?:[ \t]+|[ \t]*\n[ \t]*)(?P<t>[^\n]+)", "t"),
    # --- bill stages --------------------------------------------------------
    _r("second_reading", "second_reading",
       rf"^{_STAGE_PREFIX}{_BILL}[ \t]+(?:(?:be|is)[ \t]+)?(?:now[ \t]+)?read a second time", flags=re.M | re.I),
    _r("third_reading", "third_reading",
       rf"^{_STAGE_PREFIX}{_BILL}[ \t]+(?:(?:be|do|is)[ \t]+)?(?:now[ \t]+)?(?:read the third time|pass)\b", flags=re.M | re.I),
    _r("bill_committed", "bill_committed",
       rf"^(?:(?:{_MOVE},?[ \t]*)?{_THAT}[ \t]+the[ \t]+)?Bill[ \t]+(?:be[ \t]+)?committed\b", flags=re.M | re.I),
    _r("amendment_result", "amendment_result",
       r"^Amendments?(?:[ \t]+[0-9]+[A-Z]*)?[ \t]+(?P<r>agreed to|agreed|negatived|made|withdrawn|disagreed to)\b", "r", flags=re.M | re.I),
    _r("amendment_withdrawn", "amendment_withdrawn",
       r"\b(?:beg leave to withdraw|I withdraw)[ \t]+(?:the|my|this)[ \t]+amendment\b", flags=re.I),
    _r("clause_result", "clause_result",
       r"^(?:New Clause|Clause|Schedule|New Schedule)[ \t]+[0-9A-Z]+(?:\([0-9a-z]+\))*[ \t]+"
       r"(?P<r>agreed to|negatived|disagreed to|ordered to stand part of the Bill|added to the Bill)\b", "r", flags=re.M | re.I),
    _r("closure", "closure", rf"^(?:{_MOVE},?[ \t]*)?(?:That )?the Question be now put\.?[ \t]*$", flags=re.M | re.I),
    # --- chair / sitting ----------------------------------------------------
    _r("royal_assent", "royal_assent",
       r"(?:signified|notif\w*)[^\n.]{0,120}\bRoyal Assent\b|\bRoyal Assent Act 1967\b|^Royal Assent$", chair_only=True, flags=re.M | re.I),
    _r("point_of_order", "point_of_order",
       r"^(?:On|Further to|I rise on|I raise|May I raise|I wish to raise)\b[^.\n]{0,40}?\bpoints? of order\b", flags=re.M | re.I),
    _r("adjourn_motion", "adjournment_motion",
       rf"^(?:{_MOVE},?[ \t]*)?(?:That )?this House do now adjourn\b", flags=re.M | re.I),
    _r("sitting_adjourned", "sitting_adjourned",
       r"^(?:(?:The )?(?:House|Committee|Sitting|Debate)[ \t]+(?:was |stood )?(?:adjourned|suspended)\b|Adjourned\b|Sitting suspended\b)", flags=re.M | re.I),
    _r("urgent_question", "urgent_question", r"\bto ask (?:an )?urgent question\b", chair_only=True, flags=re.I),
    _r("prorogation", "prorogation", r"^Prorogation\b"),
    # --- statements, questions, interests -----------------------------------
    _r("statement_commons", "statement_begin",
       r"^With permission,[ \t]*(?:(?:Mr|Madam)[ \t]+(?:Deputy[ \t]+)?Speaker,[ \t]*)?(?:I|may I)[ \t]+(?:will|wish to|would like to|shall)[ \t]+(?:now[ \t]+)?"
       r"make a statement(?:[ \t]+(?:on|about|regarding|concerning)[ \t]+(?P<topic>[^.\n]+))?", "topic", flags=re.M | re.I),
    _r("statement_update", "statement_begin",
       r"^With permission,[ \t]*(?:(?:Mr|Madam)[ \t]+(?:Deputy[ \t]+)?Speaker,[ \t]*)?(?:I|may I)[ \t]+(?:will|wish to|would like to|shall)[ \t]+(?:now[ \t]+)?"
       r"(?:provide|update|inform|brief|give) the House\b(?:[^.\n]*?[ \t]+(?:on|about|regarding|concerning)[ \t]+(?P<topic>[^.\n]+)|[^.\n]*)",
       "topic", flags=re.M | re.I),
    _r("statement_lords", "statement_begin",
       r"^(?:My Lords,[ \t]*)?with the leave of the House,[ \t]*I (?:shall|will) (?:now )?repeat a Statement(?:[ \t]+(?P<topic>[^.\n]+))?", "topic", flags=re.M | re.I),
    _r("question_asked", "question_asked",
       r"^(?P<q>To ask (?:His Majesty[’']s Government|the [A-Z][^\n]*?)\b[^\n]*)", "q"),
    _r("question_asked_commons", "question_asked",
       r"^(?P<q>T?\d+\.(?:[ \t]+|[ \t]*\n[ \t]*)(?:If (?:he|she|the)|What|Whether|Will|How|Who|When|Which)\b[^\n]*)", "q"),
    _r("urgent_question_opener", "urgent_question", r"^\(Urgent Question\)[ \t]*:?[ \t]*(?P<q>[^\n]+)", "q", flags=re.M | re.I),
    _r("petition", "petition_presented",
       r"(?P<p>\bI (?:rise to |have the honour to |have the privilege to |wish to )?present (?:a|this|the|an e-)?[ \t]*petition\b[^.\n]*)", "p", flags=re.I),
    _r("declaration", "declaration_of_interest",
       r"(?P<d>\bI (?:declare|refer|draw attention|wish to declare|should declare)\b(?:"
       r"[^.\n]{0,100}?\bmy\b[^.\n]{0,60}?\binterests?\b|"
       r"[^.\n]{0,100}?\bRegister of\b[^.\n]{0,40}Interests|"
       r"[^.\n]{0,40}?\ban interest\b)[^.\n]*)", "d", flags=re.I),
    _r("maiden_speech", "maiden_speech", r"\bmy maiden speech\b", flags=re.I),
    # --- a new Parliament opening ------------------------------------------- W1 contains a dissolution and a State
    # Opening, so it holds events no 13-day window could: the election of a Speaker, Members being sworn, and the Debate
    # on the Address.
    _r("oath_taken", "oath_or_affirmation",
       r"(?P<who>(?:The following Members?|The Speaker[^\n]{0,40}?)?[ \t]*"
       r"(?:took and subscribed the Oath|made and subscribed the (?:solemn )?[Aa]ffirmation)[^\n]{0,120})",
       "who", flags=re.M),
    _r("oath_words", "oath_or_affirmation",
       r"\b(?:I swear by Almighty God that I will be faithful|"
       r"I do solemnly, sincerely and truly declare and affirm)\b", flags=re.I),
    _r("speaker_elected", "speaker_elected",
       r"\b(?:take the Chair as Speaker[- ]Elect|called to the Chair as Speaker|"
       r"do take the Chair of this House as Speaker)\b", flags=re.I),
    _r("speaker_election_question", "speaker_elected",
       r"^(?P<q>That [^\n]{0,80}do take the Chair of this House as Speaker[^\n]*)", "q", motion=True),
    _r("humble_address_moved", "humble_address",
       r"(?P<m>That an [Hh]umble Address be presented to (?:His|Her) Majesty[^\n]*)", "m", motion=True),
    _r("gracious_speech", "address_debate",
       r"\b(?:most gracious Speech|(?:King|Queen)[’']s Speech)\b[^\n]{0,80}", flags=re.I),
    # W3's remote divisions are announced in the text as well as recorded in the Votes API (`division.is_remote`,
    # migration 006).
    _r("remote_division_result", "remote_division_result",
       r"^Remote Division result[:.]?[ \t]*(?P<r>[^\n]*)", "r", flags=re.M | re.I),
)  # fmt: skip

_RULES_BY_NAME = {r.name: r for r in EVENT_RULES}


def _trim(value: str | None, limit: int) -> str | None:
    value = " ".join(value.split()) if value else None
    return value[:limit] if value else None


def find_events(text: str | None, speaker_name: str | None = None) -> list[Event]:
    """Procedural events in one speech, sorted by position. `char_start` indexes clean_text(text)."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    is_chair = parse_speaker(speaker_name).is_chair
    spans: dict[str, list[tuple[int, int]]] = {}
    events: list[Event] = []
    for rule in EVENT_RULES:
        if rule.chair_only and not is_chair:
            continue
        for m in rule.pattern.finditer(cleaned):
            spans.setdefault(rule.name, []).append(m.span())
            if rule.not_within and any(a <= m.start() < b for a, b in spans.get(rule.not_within, ())):
                continue
            description = _trim(m.group(rule.group), MAX_DESCRIPTION) if rule.group and m.group(rule.group) else None
            if rule.event_type == "division_result":
                description = _trim(m.group(0), MAX_DESCRIPTION)
            events.append(Event(
                rule.event_type, rule.name, m.start(), _trim(m.group(0), MAX_PHRASE) or "",
                description, classify_motion(description) if rule.motion and description else None,
            ))  # fmt: skip
    events.sort(key=lambda e: (e.char_start, e.event_type))
    return events


# --------------------------------------------------------------------------- Boilerplate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Formula:
    reason: str
    pattern: re.Pattern  # must match the WHOLE sentence
    chair_only: bool = False


def _f(reason, pattern, chair_only=False, flags=re.I):
    return Formula(reason, re.compile(pattern, flags), chair_only)


_DASH = "—–-"
FORMULAS: tuple[Formula, ...] = (
    _f("order_call", r"(?:order[.,!]?\s*)+", chair_only=True),
    _f("chair_call", r"(?:[^.!?]{0,80}, )?I call\b[^.!?]*[.!?]?", chair_only=True),
    _f("chair_housekeeping",
       r"(?:We have|There will be|The time limit|Time limit|May I remind|I remind|Can I remind|Please|That brings us to|"
       r"(?:People|Colleagues|Everyone|Anyone)\s+(?:should|must|need|may)|"
       r"(?:Hon\. )?Members (?:should|must|are))\b.{0,200}", chair_only=True),
    _f("give_way",
       r"(?:on that point,\s*|briefly,\s*)?(?:will|would|might|may|could|can)\s+(?:the|my|I|he|she)\b.{0,70}?\bgive way"
       r"(?:\s+(?:again|on that point|briefly|for a moment))?\s*[?.!]?"),
    _f("give_way", r"(?:I (?:will|shall) |I am (?:delighted|happy|glad|pleased) to )?give way(?: to .{1,60})?[.!]"),
    _f("stage_direction", r"[\[(]?(?:Interruption|Laughter|Applause)\.?[\])]?"),
    _f("stage_direction", rf"(?:rose|indicated (?:assent|dissent|agreement)|nodded)[.{_DASH}]*"),
    _f("stage_direction", rf"My Lords?\s*[{_DASH}]+"),
    _f("question_formula", r"The Question is as on the Order Paper\.?"),
    _f("question_formula", r"The Question is,?\s+(?:that|whether)\b.*", chair_only=True),
    _f("question_formula", r"Question put(?:,\s+(?:That|whether)\b.*| and (?:agreed to|negatived))?\.?", flags=0),
    _f("question_formula", r"Question accordingly (?:agreed to|negatived)\.?", flags=0),
    _f("question_formula", r"(?:On Question, )?Motion agreed\.?", flags=0),
    _f("question_formula", r"(?:Motion made, and )?Question proposed(?:,?\s+That\b.*)?\.?", flags=0),
    _f("question_formula", r"The (?:House|Committee|Grand Committee) divided:?.*", flags=0),
    _f("question_formula", r"Ayes\s+\d+.*Noes\s+\d+.*"),
    _f("question_formula", r"Division No\.?\s*\d+\]?\.?"),
    _f("resolution", r"(?:Resolved|Ordered),?(?:\s+That\b.+)?", flags=0),
    _f("formal_motion",
       r"(?:(?:my lords,\s+)?i beg to move,?\s+)?that\s+(?:the|this)\s+(?:bill|clause|new clause\s+\S+|amendment|question|schedule)\b"
       r".{0,60}?\b(?:be|do)\s+(?:now\s+)?(?:read\s+(?:a|the)\s+(?:second|third)\s+time|pass|made|agreed to|put|considered|"
       r"committed\b.*|added to the bill)\.?"),
    _f("formal_motion", r"(?:my lords,\s+)?i beg to move[.,]?"),
    _f("formal_motion", r"i wish to test the opinion of the (?:house|committee)\.?"),
    _f("formal_motion", r"(?:(?:in that case|accordingly|therefore),?\s+)?(?:i (?:therefore )?)?(?:beg leave to withdraw|withdraw) the amendment\.?"),
    _f("formal_motion", r"(?:i beg to move,?\s+)?that (?:the question be now put|this house do now adjourn)\.?"),
    _f("division_formula", r"(?:not[- ])?contents?[.?!]?", chair_only=True),
)  # fmt: skip


def sentence_boilerplate_reason(sentence: str, speaker_name: str | None = None) -> str | None:
    """The reason this single sentence is pure procedure, or None. Used per chunk."""
    sentence = " ".join((sentence or "").split())
    if not sentence:
        return None
    is_chair = parse_speaker(speaker_name).is_chair
    for formula in FORMULAS:
        if (is_chair or not formula.chair_only) and formula.pattern.fullmatch(sentence):
            return formula.reason
    return None


def boilerplate_reason(text: str | None, speaker_name: str | None = None) -> str | None:
    """'empty' for a speech with no words, else the first sentence's formula name if EVERY sentence is a formula, else
    None.
    """
    sentences = split_sentences(clean_text(text))
    if not sentences:
        return "empty"
    reasons = [sentence_boilerplate_reason(s, speaker_name) for s in sentences]
    return reasons[0] if all(reasons) else None


# --------------------------------------------------------------------------- Database
# ---------------------------------------------------------------------------

# One WINDOW at a time.
SQL_WINDOWS = "SELECT DISTINCT window_id FROM debate ORDER BY window_id"
SQL_SPEECHES = """
    SELECT s.id, s.debate_id, s.seq, s.speaker_name, s.text FROM speech s JOIN debate d ON d.id = s.debate_id
     WHERE d.window_id = %s ORDER BY s.debate_id, s.seq
"""
SQL_CHUNKS = """
    SELECT c.id, c.body, s.speaker_name FROM chunk c JOIN speech s ON s.id = c.speech_id JOIN debate d ON d.id = s.debate_id
     WHERE d.window_id = %s
"""
SQL_DEBATES = "SELECT id, heading, section, house FROM debate WHERE window_id = %s"
# The window's speeches, for scoping a write to the window.
_IN_WINDOW = "SELECT s.id FROM speech s JOIN debate d ON d.id = s.debate_id WHERE d.window_id = %s"

# Every write skips rows that already hold the computed values, so a re-run is a no-op.
SQL_UPDATE_SPEECH = """
    UPDATE speech s SET
        is_boilerplate = u.is_boilerplate, boilerplate_reason = u.reason,
        speaker_office = u.office, speaker_person = u.person, speaker_constituency = u.constituency,
        speaker_party = u.party, speaker_role = u.role, speaker_flags = string_to_array(u.flags, ',')
    FROM unnest(%s::text[], %s::bool[], %s::text[], %s::text[], %s::text[], %s::text[], %s::text[], %s::text[], %s::text[])
         AS u(id, is_boilerplate, reason, office, person, constituency, party, role, flags)
    WHERE s.id = u.id
      AND (s.is_boilerplate, s.boilerplate_reason, s.speaker_office, s.speaker_person, s.speaker_constituency,
           s.speaker_party, s.speaker_role, s.speaker_flags)
          IS DISTINCT FROM
          (u.is_boilerplate, u.reason, u.office, u.person, u.constituency, u.party, u.role, string_to_array(u.flags, ','))
"""
SQL_UPSERT_EVENT = """
    INSERT INTO procedure_event (debate_id, speech_id, seq, event_type, raw_phrase, description, motion_type, rule, char_start)
    VALUES (%s, %s, %s, %s, %s, %s, %s::motion_type, %s, %s)
    ON CONFLICT (speech_id, event_type, char_start) DO UPDATE SET
        debate_id = EXCLUDED.debate_id, seq = EXCLUDED.seq, raw_phrase = EXCLUDED.raw_phrase,
        description = EXCLUDED.description, motion_type = EXCLUDED.motion_type, rule = EXCLUDED.rule
    WHERE (procedure_event.debate_id, procedure_event.seq, procedure_event.raw_phrase, procedure_event.description,
           procedure_event.motion_type, procedure_event.rule)
          IS DISTINCT FROM (EXCLUDED.debate_id, EXCLUDED.seq, EXCLUDED.raw_phrase, EXCLUDED.description,
                            EXCLUDED.motion_type, EXCLUDED.rule)
"""
# Rule-made events of THIS window's speeches that no longer match (a rule changed) are removed; rows with rule IS NULL
# are never touched.
SQL_DELETE_STALE_EVENTS = f"""
    DELETE FROM procedure_event pe
    WHERE pe.rule IS NOT NULL
      AND pe.speech_id IN ({_IN_WINDOW})
      AND NOT EXISTS (
          SELECT 1 FROM unnest(%s::text[], %s::text[], %s::int[]) AS k(speech_id, event_type, char_start)
          WHERE k.speech_id = pe.speech_id AND k.event_type = pe.event_type AND k.char_start = pe.char_start)
"""
SQL_FLAG_CHUNKS = """
    UPDATE chunk c SET is_boilerplate = true
    FROM unnest(%s::bigint[]) AS u(id) WHERE c.id = u.id AND NOT c.is_boilerplate
"""
SQL_UNFLAG_CHUNKS = f"""
    UPDATE chunk c SET is_boilerplate = false
    WHERE c.is_boilerplate AND c.speech_id IN ({_IN_WINDOW})
      AND NOT EXISTS (SELECT 1 FROM unnest(%s::bigint[]) AS u(id) WHERE u.id = c.id)
"""
SQL_UPDATE_DEBATE_TYPES = """
    UPDATE debate d SET debate_type = u.t
    FROM unnest(%s::text[], %s::text[]) AS u(id, t)
    WHERE d.id = u.id AND d.debate_type IS DISTINCT FROM u.t
"""


@dataclass
class RulesResult:
    speeches: int = 0
    speech_rows_changed: int = 0
    events: Counter = None
    reasons: Counter = None
    roles: Counter = None
    debate_types: Counter = None
    chunks_flagged: int = 0
    chunks_total: int = 0
    stale_events_removed: int = 0
    chunk_flags_changed: int = 0
    debate_types_changed: int = 0
    windows: list = None
    dry_run: bool = False


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
        TimeElapsedColumn(), console=console, transient=True,
    )  # fmt: skip


async def classify_window(conn, window_id: str, result: RulesResult, dry_run: bool, progress: Progress) -> None:
    """Every rule, for one window, written in one transaction. The caller's connection is in autocommit, so the
    `conn.transaction` here is a real BEGIN/COMMIT: a failure in a later window never costs this one (gotcha 37).
    """
    async with conn.transaction():
        speeches = await (await conn.execute(SQL_SPEECHES, (window_id,))).fetchall()
        columns: list[list] = [[] for _ in range(9)]
        event_rows: list[tuple] = []
        debate_events: dict[str, set[str]] = {}
        task = progress.add_task(f"{window_id}: classifying speeches", total=len(speeches))
        for speech_id, debate_id, seq, name, text in speeches:
            who = parse_speaker(name)
            reason = boilerplate_reason(text, name)
            for i, value in enumerate((speech_id, reason is not None, reason, who.office, who.person,
                                       who.constituency, who.party, who.role, ",".join(who.flags))):
                columns[i].append(value)
            result.reasons[reason or "-"] += 1
            result.roles[who.role] += 1
            for e in find_events(text, name):
                result.events[e.event_type] += 1
                debate_events.setdefault(debate_id, set()).add(e.event_type)
                event_rows.append((debate_id, speech_id, seq, e.event_type, e.raw_phrase, e.description,
                                   e.motion_type, e.rule, e.char_start))  # fmt: skip
            progress.advance(task)
        progress.remove_task(task)
        result.speeches += len(speeches)
        del speeches

        # Streamed through a server-side cursor: a window's chunk bodies are never all in memory at once.
        flagged_ids: list[int] = []
        seen = 0
        task = progress.add_task(f"{window_id}: flagging chunks", total=None)
        async with conn.cursor(name=f"rules_chunks_{window_id}") as cur:
            cur.itersize = 20_000
            await cur.execute(SQL_CHUNKS, (window_id,))
            async for chunk_id, body, name in cur:
                seen += 1
                if sentence_boilerplate_reason(body, name):
                    flagged_ids.append(chunk_id)
                if seen % 20_000 == 0:
                    progress.update(task, completed=seen)
        progress.remove_task(task)
        result.chunks_total += seen
        result.chunks_flagged += len(flagged_ids)

        debates = await (await conn.execute(SQL_DEBATES, (window_id,))).fetchall()
        debate_ids, debate_types = [], []
        for debate_id, heading, section, house in debates:
            kind = classify_debate(heading, section, house, debate_events.get(debate_id, set()))
            debate_ids.append(debate_id)
            debate_types.append(kind)
            result.debate_types[kind] += 1

        if dry_run:
            return
        cur = await conn.execute(SQL_UPDATE_SPEECH, columns)
        result.speech_rows_changed += cur.rowcount
        async with conn.cursor() as cur:
            await cur.executemany(SQL_UPSERT_EVENT, event_rows)
            await cur.execute(SQL_DELETE_STALE_EVENTS, (window_id, [r[1] for r in event_rows],
                                                        [r[3] for r in event_rows], [r[8] for r in event_rows]))
            result.stale_events_removed += cur.rowcount
        result.chunk_flags_changed += (await conn.execute(SQL_FLAG_CHUNKS, (flagged_ids,))).rowcount
        result.chunk_flags_changed += (await conn.execute(SQL_UNFLAG_CHUNKS, (window_id, flagged_ids))).rowcount
        result.debate_types_changed += (await conn.execute(SQL_UPDATE_DEBATE_TYPES, (debate_ids, debate_types))).rowcount


async def run_rules(dry_run: bool = False, windows: list[str] | None = None) -> RulesResult:
    """Classify every window (or the named ones), one committed window at a time. Idempotent: a re-run changes nothing,
    and an interrupted run is resumed by running again.
    """
    from app import safety

    safety.refuse_protected(pool.conninfo)
    result = RulesResult(events=Counter(), reasons=Counter(), roles=Counter(), debate_types=Counter(),
                         windows=[], dry_run=dry_run)  # fmt: skip
    await pool.open(wait=True)
    try:
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            try:
                known = [r[0] for r in await (await conn.execute(SQL_WINDOWS)).fetchall()]
                unknown = sorted(set(windows or ()) - set(known))
                if unknown:
                    raise SystemExit(f"no debates in window(s) {unknown}; known: {known}")
                with _progress() as progress:
                    for window_id in (windows or known):
                        await classify_window(conn, window_id, result, dry_run, progress)
                        result.windows.append(window_id)
                        console.print(f"  {window_id}: done ({result.speeches:,} speeches, "
                                      f"{result.chunks_total:,} chunks so far)")
            finally:
                await conn.set_autocommit(False)
    finally:
        await pool.close()
    return result


def print_summary(r: RulesResult) -> None:
    def table(title, counter, total=None):
        t = Table(title=title)
        t.add_column("")
        t.add_column("Count", justify="right")
        for key, n in counter.most_common():
            t.add_row(str(key), f"{n:,}")
        console.print(t)

    table(f"Procedure events ({sum(r.events.values()):,})", r.events)
    reasons = Counter({k: v for k, v in r.reasons.items() if k != "-"})
    table(f"Boilerplate speeches ({sum(reasons.values()):,} of {r.speeches:,})", reasons)
    table("Speaker roles", r.roles)
    table("Debate types", r.debate_types)
    console.print(
        f"Chunks flagged boilerplate: [bold]{r.chunks_flagged:,}[/bold] of {r.chunks_total:,} | "
        f"speech rows changed: {r.speech_rows_changed:,} | stale events removed: {r.stale_events_removed:,} | "
        f"chunk flags changed: {r.chunk_flags_changed:,} | debate types changed: {r.debate_types_changed:,} | "
        f"windows: {', '.join(r.windows or [])}"
        + (" [yellow](dry run: nothing written)[/yellow]" if r.dry_run else "")
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Flag boilerplate and extract procedure events.")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    parser.add_argument("--window", action="append", default=None,
                        help="classify only this window (repeatable); default every window, one at a time")
    args = parser.parse_args(argv)
    print_summary(run(run_rules(args.dry_run, args.window)))


if __name__ == "__main__":
    main()
