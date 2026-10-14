"""The router: which route answers this question, and what filters it carries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

# 's route names.
ROUTES = ("structured", "set", "retrieval", "overview", "multi_hop", "timeline", "clarify", "abstain")

# Gold categories that are structured-fact questions, answered from SQL by `answer.facts` with no model at all.
STRUCTURED_CATEGORIES = frozenset({
    "division_result", "division_tally", "division_vote", "division_debate", "party_at_date",
    "office_held_at_date", "bill_stage_date", "count",
})

# "Which debate was division 149 in the Commons on 8 May 2024 taken during?" — a question about the `division.debate_id`
# column, which no amount of searching speech text can answer. 36 gold items (13 dev, 15 test, 8 heldout) ask it and
# every one was routed to retrieval; the generator was handed ten speech cards, said correctly that none of them
# recorded a division, and abstained.
_VOTE_WORD = re.compile(r"\b(?:division|divided|divide|divides|vote|voted|votes|voting)\b", re.I)
_ASKS_FOR_DEBATE = re.compile(
    r"\b(?:which|what)\s+debate\b"
    r"|\bwhat\s+(?:was|were)\s+(?:being\s+)?(?:debated|discussed)\b"
    r"|\bwhat\s+was\s+under\s+(?:debate|discussion)\b"
    r"|\b(?:find|name|identify|give|show)\s+(?:me\s+)?the\s+debate\b"
    r"|\bthe\s+debate\s+(?:that|which)\s+(?:led|gave\s+rise)\s+to\b"
    # "during which debate" asks for it; a bare "taken during" only describes the division
    r"|\bduring\s+(?:which|what)\s+(?:debate|discussion|proceedings)\b", re.I)


def _asks_for_debate_behind_division(text: str) -> bool:
    """Both halves, anywhere in the text: a division, and a request for its debate. Two linear searches: the same test
    as one pattern of two `.*` look-aheads was quadratic, 1.25 s on 8 KB and more than 300 s on 100 KB (QA pass 3).
    """
    return bool(_VOTE_WORD.search(text) and _ASKS_FOR_DEBATE.search(text))


# The longest question read.
MAX_QUESTION_CHARS = 1000

# WHICH fact a question asks for, from its words alone.
_NAME = r"(?:[\w'’.-]+\s+){1,6}?"
# The words a question uses for a House deciding something, ONE list for the router's result shape and the resolver's
# fate questions (QA pass 5: the resolver's own list had "pass" where the router read "agree", "carried", "lost"...).
RESULT_VERBS = (r"agree(?:\s+to)?|approve|pass|reject|accept|carry|back|support|endorse|defeat|throw\s+out|"
                r"vote\s+(?:for|against|down|through)")
RESULT_PARTICIPLES = (r"carried|agreed(?:\s+to)?|passed|rejected|defeated|negatived|approved|lost|accepted|backed|"
                      r"supported|endorsed|thrown\s+out|voted\s+(?:down|through)")
# A part of the House: a party, a bench, a group of members.
PART_OF_HOUSE = re.compile(
    r"\b(?:conservatives?|tory|tories|labour|liberal\s+democrats?|lib\s*dems?|snp|scottish\s+national|greens?|"
    r"green\s+party|reform(?:\s+uk)?|dup|democratic\s+unionists?|plaid(?:\s+cymru)?|sinn\s+f[eé]in|sdlp|alliance|"
    r"independents?|cross[\s-]?bench(?:ers?)?|bishops?|lords\s+spiritual|rebels?|backbench(?:ers?)?|frontbench(?:ers?)?|"
    r"women|men|female|male|government\s+(?:mps|members|peers)|opposition\s+(?:mps|members|peers))\b"
    r"|\bamong\s+(?:the\s+)?\w+", re.I)
_VOTE_CONTEXT = r"(?:division|divided|vote|voted|votes|voting|motion|amendment|clause|bill|act|regulations?|order|reading)"
_FACT_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("division_tally", re.compile(
        r"\bayes?\s+and\s+(?:the\s+)?noes?\b|\bcontents?\s+and\s+(?:the\s+)?not[\s-]contents?\b"
        r"|\bvoting\s+(?:figures|numbers|tally)\b|\bvote\s+count\b|\btally\b"
        # Only the whole House between "how many" and the verb: "how many Conservative MPs voted against" is a count of
        # part of it, and was answered with the House's 360 to 211.
        r"|\bhow\s+many\s+(?:(?:hon(?:ourable)?\.?\s+)?(?:mps|members|peers|lords|people|parliamentarians)\s+"
        r"(?:of\s+(?:the\s+)?(?:house|commons|lords)\s+)?|of\s+them\s+)?(?:voted|were\s+(?:for|against))\b"
        r"|\bby\s+how\s+many\s+votes?\b"
        r"|\bwhat\s+(?:was|were)\s+the\s+(?:margin|majority|numbers|figures|split)\b"
        r"|\bnumbers\s+on\s+(?:each|both)\s+sides?\b", re.I)),
    ("division_vote", re.compile(
        rf"\bhow\s+did\s+(?!the\s+(?:house|commons|lords|government|opposition)\b){_NAME}vote\b"
        rf"|\b(?:was|were)\s+{_NAME}an?\s+(?:aye|no|content|not[\s-]content)\b"
        r"|\bwhich\s+lobby\s+did\b"
        rf"|\bdid\s+(?!the\s+(?:house|commons|lords|government|opposition)\b){_NAME}"
        rf"(?:vote|support|oppose|back|rebel)\b(?=.*\b{_VOTE_CONTEXT}\b)", re.I | re.S)),
    ("division_result", re.compile(
        r"\bpassed\s+or\s+(?:rejected|defeated|failed)\b|\bcarried\s+or\s+(?:not|rejected|defeated|lost)\b"
        r"|\b(?:was|were)\s+(?:the\s+)?[^?]{0,140}?\b(?:motion|amendment|clause|bill|regulations?|order|instrument)\b"
        rf"[^?]{{0,90}}?\b(?:{RESULT_PARTICIPLES})\b"
        r"|\bdid\s+the\s+(?:house(?:\s+of\s+(?:commons|lords))?|commons|lords)\s+"
        rf"(?:{RESULT_VERBS})\b"
        r"|\bdid\s+the\s+government\s+(?:win|lose)\b"
        r"|\b(?:result|outcome)\s+of\s+the\s+(?:[\w'’-]+\s+){0,3}(?:division|vote)\b"
        r"|\bwhat\s+was\s+the\s+(?:result|outcome)\b(?=[^?]*\b(?:division|divided|vote)\b)", re.I)),
    ("count", re.compile(
        r"\bhow\s+many\s+(?:separate\s+)?(?:divisions|debates|sittings|speeches|contributions)\b"
        r"|\bhow\s+many\s+times\s+(?:did|has|had|was|were)\b", re.I)),
    ("bill_stage_date", re.compile(
        r"\bwhen\s+(?:did|was|were)\b[^?]{0,160}?\b(?:bill|act)\b[^?]{0,120}?"
        r"\b(?:reading|stage|royal\s+assent|consideration|examin\w+|commitment)\b"
        r"|\bwhen\s+(?:did|was|were)\b[^?]{0,120}?\b(?:reading|stage|royal\s+assent|consideration|examin\w+)\b"
        r"[^?]{0,160}?\b(?:bill|act)\b"
        r"|\b(?:on\s+)?what\s+(?:was\s+the\s+)?date\b[^?]{0,160}?"
        r"\b(?:reading|stage|royal\s+assent|consideration|examin\w+|commitment|committee)\b", re.I)),
    ("party_at_date", re.compile(
        r"\b(?:which|what)\s+party\s+(?:was|is|were|did|does)\b|\b[\w'’.-]+['’]s\s+party\b"
        r"|\bparty\s+(?:affiliation|membership)\b", re.I)),
    ("office_held_at_date", re.compile(
        r"\bwho\s+(?:held|occupied|filled)\s+the\s+(?:post|office|role|position|job)\s+of\b"
        r"|\bwho\s+was\s+(?:serving|acting)\s+as\b"
        r"|\bwho\s+(?:was|were)\s+the\s+(?:[\w'’()-]+\s+){0,6}?(?:secretary|minister|chancellor|whip|spokesperson|"
        r"leader\s+of\s+the\s+(?:house|opposition)|chamberlain|attorney\s+general|solicitor\s+general|"
        r"paymaster\s+general|advocate\s+general|(?<!first\s)(?<!opening\s)speaker)\b", re.I)),
)
# A question about WHAT was said, or why, is text — even when it mentions a vote ("Why did X vote against it?", "What
# did MPs say about the result?").
_NOT_A_FACT = re.compile(r"\b(?:why|reasons?|because|explain\w*|justif\w*|argu\w*|say|said|says|saying|mean|means|"
                         r"think|thought)\b", re.I)
# "My MP" is a member the record cannot identify.
_UNNAMED_MEMBER = re.compile(r"\b(?:my|our)\s+(?:local\s+)?(?:mp|member(?:\s+of\s+parliament)?|representative|peer)\b",
                             re.I)


# The words a "who said X?" question quotes.
_QUOTED = (
    re.compile(r"[\"“‘']([^\"”’']{12,})[\"”’']"),
    re.compile(r"^\s*(?:which\s+(?:mp|member|peer|minister)\s+|who\s+)(?:said|says|used\s+the\s+(?:phrase|words))\s*"
               r"[:,]?\s*(?:that\s+)?(.+?)\s*\??\s*$", re.I),
    re.compile(r"\b(?:where|when)\s+(?:they|he|she|someone|members?|mps?|peers?)\s+(?:say|says|said)\s+(?:that\s+)?"
               r"(.+?)\s*\??\s*$", re.I),
)
QUOTED_MIN_WORDS = 4


def quoted_phrase(question: str) -> str | None:
    """The words a question asks the author of, or None."""
    text = " ".join((question or "").split())
    for pattern in _QUOTED:
        match = pattern.search(text)
        if match and len(match.group(1).split()) >= QUOTED_MIN_WORDS:
            return match.group(1).strip(" .,:;!?")
    return None


# A tally's qualifier, whatever the party: "the Ulster Unionist Party tally", "the Alba ayes and noes".
_TALLY_QUALIFIED = re.compile(
    r"\bthe\s+(?P<q>(?:[\w'’-]+\s+){1,4}?)(?:tally|ayes\s+and\s+(?:the\s+)?noes|vote\s+count|voting\s+(?:figures|numbers)|"
    r"contents\s+and\s+(?:the\s+)?not[\s-]contents)\b", re.I)
_WHOLE_HOUSE = frozenset("final overall total full official exact whole house house's commons lords division division's "
                         "published recorded".split())


def _qualified_tally(text: str) -> bool:
    m = _TALLY_QUALIFIED.search(text)
    return bool(m) and any(w.lower() not in _WHOLE_HOUSE for w in m.group("q").split())


def structured_category(question: str) -> str | None:
    """The fact category a question asks for, from its words, or None when it is not a fact lookup."""
    text = " ".join((question or "").split())
    # The "why" guard reads the question with Parliament's message names blanked: "Lords Reason 120B" in a division's
    # title names a message between the Houses, not a request for reasons.
    if not text or _NOT_A_FACT.search(_HOUSE_IN_TITLE.sub(" ", text)):
        return None
    for category, pattern in _FACT_SHAPES:
        if pattern.search(text):
            blanked = _HOUSE_IN_TITLE.sub(" ", text)
            if category == "division_tally" and (PART_OF_HOUSE.search(blanked) or _qualified_tally(blanked)):
                continue
            return category
    return None


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}  # fmt: skip
_MONTH_ALT = "|".join(MONTHS)

# --- the patterns ---------------------------------------------------------------------------------------

_HOUSE = (
    (re.compile(r"\b(?:the\s+)?(?:house\s+of\s+)?commons\b|\bmps?\b|\bmember\s+of\s+parliament\b", re.I), "Commons"),
    (re.compile(r"\b(?:the\s+)?(?:house\s+of\s+)?lords\b|\bpeers?\b|\bmy\s+lords\b|\bbaroness\b|\bviscount\b", re.I), "Lords"),
)

_SECTION = (
    (re.compile(r"\bwestminster\s+hall\b", re.I), "Westminster Hall"),
    (re.compile(r"\bpublic\s+bill\s+committee", re.I), "Public Bill Committees"),
    (re.compile(r"\bgeneral\s+committee", re.I), "General Committees"),
    (re.compile(r"\bgrand\s+committee\b", re.I), "Grand Committee"),
)

# Ordered: the first intent whose pattern matches wins, so the more specific patterns come first.
_INTENTS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("set", re.compile(
        r"\bwho\s+(?:all\s+)?(?:spoke|contributed|took\s+part|participated|opened|closed)\b"
        # Every everyday form of the question, not only the gold's.
        r"|\bwho\s+(?:led|leads|kicked|kicks)\s+off\b|\b(?:spoke|speak)\s+first\b"
        r"|\bfirst\s+(?:to\s+speak|speaker)\b|\bopening\s+speech\b"
        r"|\bwho\s+(?:began|started|wound\s+up)\s+(?:the\s+)?(?:debate|discussion|proceedings)\b"
        r"|\bwhich\s+members?\s+(?:spoke|contributed|took\s+part|participated)\b"
        r"|\blist\s+(?:the\s+)?(?:members?|speakers?|peers?|mps?)\b"
        r"|\bwho\s+(?:else\s+)?(?:was|were)\s+(?:in|at|present)\b", re.I), "asks for the set of people who spoke"),
    ("timeline", re.compile(
        r"\b(?:timeline|chronolog\w+|sequence\s+of\s+events)\b"
        r"|\bwhat\s+happened\s+(?:to|with)\s+the\s+[\w\s]+bill\b"
        r"|\bhow\s+did\s+the\s+[\w\s]+bill\s+progress\b"
        r"|\bstage\s+by\s+stage\b", re.I), "asks how something unfolded over time"),
    ("overview", re.compile(
        r"\b(?:overview|summar\w+|main\s+(?:points|arguments|themes)|key\s+(?:points|arguments|themes))\b"
        r"|\bwhat\s+(?:did|do)\s+(?:mps|peers|members|parties)\s+say\s+about\b"
        r"|\b(?:competing|different|opposing|both)\s+(?:views?|sides?|arguments?)\b"
        r"|\bwhat\s+are\s+the\s+arguments?\b"
        r"|\bwhere\s+do\s+the\s+parties\s+stand\b", re.I), "asks for the shape of a debate, not one passage"),
    ("multi_hop", re.compile(
        r"\band\s+(?:then\s+)?(?:how|did)\s+(?:did\s+)?(?:they|he|she|the\s+same)\b"
        r"|\bwho\s+(?:also|then)\b"
        r"|\bafter\s+(?:saying|arguing|voting)\b"
        r"|\bboth\s+spoke\s+and\s+voted\b", re.I), "needs two linked lookups"),
)

_OFFICES = re.compile(
    r"\bthe\s+(prime\s+minister|chancellor(?:\s+of\s+the\s+exchequer)?|home\s+secretary|foreign\s+secretary|"
    r"health\s+secretary|defence\s+secretary|education\s+secretary|justice\s+secretary|"
    r"secretary\s+of\s+state(?:\s+for\s+[\w\s]+?)?|leader\s+of\s+the\s+opposition|speaker|"
    r"deputy\s+prime\s+minister|attorney\s+general|lord\s+chancellor)\b", re.I)  # fmt: skip

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})\s+(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(19\d{2}|20\d{2})\b")
# Relative expressions.
_RELATIVE = re.compile(r"\b(yesterday|today|this\s+(?:week|month|year)|last\s+(?:week|month|year|session)|"
                       r"recently|lately|the\s+other\s+day)\b", re.I)  # fmt: skip


@dataclass
class Filters:
    """What the question said about where to look. Every field is evidence from the text, never a default."""

    house: str | None = None
    section: str | None = None
    on_date: Date | None = None
    from_date: Date | None = None
    to_date: Date | None = None
    office: str | None = None
    relative: str | None = None
    year: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Only what was found. An empty filter must not appear in the cache key as a null, or two questions that
        differ in nothing would hash differently depending on which fields the extractor happened to set.
        """
        out: dict[str, Any] = {}
        for name in ("house", "section", "office", "relative", "year"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        for name in ("on_date", "from_date", "to_date"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value.isoformat()
        return out


@dataclass
class Decision:
    """The route, the filters, and the reason — which is the part that makes a misroute diagnosable."""

    route: str
    filters: Filters = field(default_factory=Filters)
    why: str = ""
    by: str = "rules"
    confidence: float = 1.0
    clarify: tuple[str, ...] = ()
    category: str | None = None

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"unknown route {self.route!r}; known: {', '.join(ROUTES)}")


def _month_end(year: int, month: int) -> Date:
    return Date(year, 12, 31) if month == 12 else Date(year, month + 1, 1).fromordinal(
        Date(year, month + 1, 1).toordinal() - 1)


# "between 1 May 2024 and 31 July 2024", "from 1 October to 31 December 2025".
_RANGE = re.compile(r"\b(?:between|from)\b(?P<first>.{1,40}?)\b(?:and|to|until|through)\b"
                    r"(?P<second>.{1,40}?)(?=[?.,;]|$)", re.I | re.S)  # fmt: skip


def _one_date(text: str) -> Date | None:
    """The first date in a fragment, in any of the forms a question might use."""
    for pattern, order in ((_ISO, "ymd"), (_DMY, "dmy"), (_MDY, "mdy")):
        match = pattern.search(text)
        if not match:
            continue
        try:
            if order == "ymd":
                return Date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            if order == "dmy":
                return Date(int(match.group(3)), MONTHS[match.group(2).lower()], int(match.group(1)))
            return Date(int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2)))
        except (ValueError, KeyError):
            continue
    return None


def extract_range(question: str) -> tuple[Date | None, Date | None]:
    """(from, to) when the question states a span, else (None, None)."""
    found = _RANGE.search(question)
    if not found:
        return (None, None)
    first, second = _one_date(found.group("first")), _one_date(found.group("second"))
    # A year on only one side is the usual shorthand: "from 1 October to 31 December 2025".
    if first and second and first > second:
        return (None, None)
    if first and second:
        return (first, second)
    if second and not first:
        # "between the 1st and 31 July 2024" — borrow the month and year from the side that has them.
        day = re.search(r"\b(\d{1,2})\b", found.group("first"))
        if day:
            try:
                return (Date(second.year, second.month, int(day.group(1))), second)
            except ValueError:
                return (None, None)
    return (None, None)


def extract_dates(question: str) -> tuple[Date | None, Date | None, Date | None, int | None, str | None]:
    """(on_date, from_date, to_date, year, relative) from the question text."""
    # A range first: a date inside "between X and Y" is a bound, not the day the question is about.
    span_from, span_to = extract_range(question)
    if span_from and span_to:
        return None, span_from, span_to, span_from.year, None

    for pattern, order in ((_ISO, "ymd"), (_DMY, "dmy"), (_MDY, "mdy")):
        match = pattern.search(question)
        if match:
            try:
                if order == "ymd":
                    found = Date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                elif order == "dmy":
                    found = Date(int(match.group(3)), MONTHS[match.group(2).lower()], int(match.group(1)))
                else:
                    found = Date(int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2)))
            except ValueError:
                continue  # "31 February 2026" is not a date; treat it as no date rather than crash
            return found, None, None, found.year, None

    month = _MONTH_YEAR.search(question)
    if month:
        year, number = int(month.group(2)), MONTHS[month.group(1).lower()]
        return None, Date(year, number, 1), _month_end(year, number), year, None

    relative = _RELATIVE.search(question)
    year_only = _YEAR.search(question)
    if year_only:
        year = int(year_only.group(1))
        return None, Date(year, 1, 1), Date(year, 12, 31), year, (relative.group(1).lower() if relative else None)
    return None, None, None, None, (relative.group(1).lower() if relative else None)


# A House named INSIDE an amendment reference.
_HOUSE_IN_TITLE = re.compile(
    r"\b(?:the\s+)?(?:lords|commons)\b\s*(?:in\s+their\s+)?"
    r"(?:amendments?|reasons?|message)\b(?:\s*\d+[a-z]*)?"
    r"|\[\s*(?:lords|hl)\s*\]", re.I)  # fmt: skip


def extract_filters(question: str) -> Filters:
    """House, section, dates and office, from the text alone."""
    filters = Filters()
    # Read the House from the question with amendment references blanked out, so "Lords Amendment 7" inside a quoted
    # title cannot decide which House was sitting.
    without_titles = _HOUSE_IN_TITLE.sub(" ", question)
    for pattern, house in _HOUSE:
        if pattern.search(without_titles):
            filters.house = house
            break
    for pattern, section in _SECTION:
        if pattern.search(question):
            filters.section = section
            # A named section implies its House, which the section names unambiguously and the question may not.
            filters.house = "Lords" if section == "Grand Committee" else filters.house or "Commons"
            break
    office = _OFFICES.search(question)
    if office:
        filters.office = " ".join(office.group(1).split()).lower()
    on_date, from_date, to_date, year, relative = extract_dates(question)
    filters.on_date, filters.from_date, filters.to_date = on_date, from_date, to_date
    filters.year, filters.relative = year, relative
    return filters


def route(question: str, category: str | None = None) -> Decision:
    """The route for this question, by rules alone, from its TEXT alone. `by == "rules"` means no model was consulted.
    """
    del category  # deliberately unused: see the docstring
    if len(question or "") > 4 * MAX_QUESTION_CHARS:
        # Measured on the RAW question, before whitespace collapses: padding then 400,000 characters passed a cap read
        # on the first 4,000, and the resolvers then read all of it.
        return Decision("clarify", why=f"the question is {len(question):,} characters; at most "
                                       f"{MAX_QUESTION_CHARS:,} are read",
                        clarify=(f"Could you ask that more briefly? Questions shorter than {MAX_QUESTION_CHARS:,} "
                                 "characters are read.",), confidence=0.0)  # fmt: skip
    text = " ".join((question or "").split())
    if not text:
        return Decision("abstain", why="the question is empty", confidence=1.0)
    if len(text) > MAX_QUESTION_CHARS:
        return Decision("clarify", why=f"the question is {len(text):,} characters; at most {MAX_QUESTION_CHARS:,} are read",
                        clarify=(f"Could you ask that more briefly? Questions shorter than {MAX_QUESTION_CHARS:,} "
                                 "characters are read.",), confidence=0.0)  # fmt: skip

    filters = extract_filters(text)

    # A member the question does not name cannot be looked up, searched for or guessed: "my MP" is someone only the
    # asker knows.
    unnamed = _UNNAMED_MEMBER.search(text)
    if unnamed:
        return Decision("clarify", filters, f"the question does not say which member ({unnamed.group(0)!r})",
                        clarify=("which MP or peer do you mean — a name or a constituency?",), confidence=0.0)

    # Shape before label.
    if _asks_for_debate_behind_division(text):
        return Decision("structured", filters,
                        "the question asks which debate a division was taken during, which is a column join",
                        category="division_debate")  # fmt: skip

    # An office with no date at all.
    if filters.office and not (filters.on_date or filters.from_date or filters.year):
        return Decision(
            "clarify", filters,
            f"'the {filters.office}' names different people across the corpus and the question gives no date",
            clarify=("which date or period do you mean?",), confidence=0.0,
        )  # fmt: skip

    # A speech AND a vote is two lookups, not one fact: "what did X say, and how did he vote?"
    multi_hop = next(p for name, p, _ in _INTENTS if name == "multi_hop")
    fact = None if multi_hop.search(text) else structured_category(text)
    if fact:
        return Decision("structured", filters, f"a {fact.replace('_', ' ')} question, answered from SQL",
                        category=fact)  # fmt: skip

    for name, pattern, why in _INTENTS:
        match = pattern.search(text)
        if match:
            return Decision(name, filters, f"{why} (matched {match.group(0)!r})")

    return Decision("retrieval", filters, "no structured, set or overview signal; searching the text")


# The prompt for the LLM fallback.
LLM_ROUTER_PROMPT = """\
Classify this question about the UK Parliamentary record into exactly one route.

structured — one exact answer held in a database: a vote result, a tally, how a named person voted, someone's party
             or office on a date, a bill stage date, or a count.
set        — asks which people spoke or took part.
overview   — asks for the shape of a debate: main arguments, competing views, what parties said about a topic.
timeline   — asks how something unfolded over time.
multi_hop  — needs two linked lookups, e.g. what someone said AND how they then voted.
retrieval  — anything else answerable by finding passages of text.
clarify    — cannot be answered as asked because it is ambiguous (for example an office with no date).

QUESTION: {question}

Return JSON only: {{"route": "...", "why": "one short sentence"}}
"""


async def route_with_model(question: str, client: Any, category: str | None = None) -> Decision:
    """Rules first; the model only for what the rules left as plain `retrieval`."""
    decided = route(question)
    if decided.route != "retrieval" or client is None:
        return decided
    try:
        reply = await client.chat(LLM_ROUTER_PROMPT.format(question=question), max_tokens=120,
                                 temperature=0.0, json_object=True)  # fmt: skip
        import json

        data = json.loads(reply.content)
        proposed = str(data.get("route", "")).strip().lower()
        # Never "structured": the rules already found no fact shape, and a structured decision without a category is one
        # the pipeline cannot answer.
        if proposed in ROUTES and proposed not in ("retrieval", "structured"):
            return Decision(proposed, decided.filters,
                            f"the rules found no signal; the classifier chose {proposed}: "
                            f"{str(data.get('why', ''))[:160]}",
                            by="llm", confidence=0.7)  # fmt: skip
    except Exception:  # noqa: BLE001 - the rules' answer stands
        return decided
    return decided
