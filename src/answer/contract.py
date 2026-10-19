"""The answer contract: the interface between the generator (E3) and the verifier (E2)."""

from __future__ import annotations

import re
import unicodedata
from datetime import date as Date
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- normalisation --------------------------------------------------------------------------------------

# Characters Hansard uses that a copied quote will not.
_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "­": "",  # soft hyphen: invisible, and breaks a substring match
    "​": "", "﻿": "",
}
_FOLD_RE = re.compile("|".join(map(re.escape, _FOLD)))
_SPACE = re.compile(r"\s+")

# The ellipsis a quote may use to skip material.
ELLIPSES = ("…", "...")


def normalise(text: str | None) -> str:
    """The form two spellings of the same sentence share."""
    if not text:
        return ""
    folded = _FOLD_RE.sub(lambda m: _FOLD[m.group()], unicodedata.normalize("NFKC", text))
    return _SPACE.sub(" ", folded).strip().casefold()


# A member in another House now is shown as printed then, with the present name after it — "Michael Gove (now Lord
# Gove)".
_NOW_NAME = re.compile(r"\(now ([^()]+)\)\s*$")


def attribution_names(speaker: str | None) -> tuple[str, ...]:
    """Every name a claim may lead with for this speaker: the card's whole form, the name before any bracket (Hansard's
    "Lincoln Jopp (Spelthorne) (Con)" is attributed as "Lincoln Jopp"), and the present name of a member shown "(now
    …)". A claim that names its speaker by any of them is attributed, and must not be attributed twice.
    """
    if not speaker or not speaker.strip():
        return ()
    names = [speaker.strip(), speaker.split("(")[0].strip()]
    now = _NOW_NAME.search(speaker)
    if now:
        names.append(now.group(1).strip())
    return tuple(dict.fromkeys(n for n in names if n))


# The words that open an attributed claim — "X said that …".
REPORTING_VERBS = (
    "said that", "said", "argued that", "argued", "told", "added that", "added", "stated that", "stated",
    "claimed that", "claimed", "asked whether", "asked", "warned that", "warned", "noted that", "noted",
    "explained that", "explained", "confirmed that", "confirmed", "replied that", "replied",
    "welcomed", "opposed", "supported", "criticised", "described",
)
# What a printed name may open with, which a claim may also put before the surname ("Mr Mather said").
_HONORIFICS = ("Mr", "Mrs", "Ms", "Miss", "Dr", "Sir", "Dame", "Professor")
# A last word that makes a printed name an OFFICE, not a person's: "Mr Speaker" is not named by "Speaker said".
_OFFICE_WORDS = frozenset({"speaker", "chancellor", "chair", "chairman", "chairwoman", "chairperson", "minister",
                           "secretary", "leader", "president", "general", "advocate", "whip", "commissioner", "clerk",
                           "member", "members"})  # fmt: skip
# A peer's printed name: the title, the name, and usually a territorial designation ("Baroness Smith of Malvern").
_PEER_TITLE = re.compile(r"^(?:Lord|Lady|Baroness|Baron|Viscount|Viscountess|Earl|Countess|Duke|Duchess|Marquess|"
                         r"Marchioness)\s")


def short_names(speaker: str | None) -> tuple[str, ...]:
    """Shorter names a claim may use for its own speaker, beside `attribution_names`' full forms."""
    forms: list[str] = []
    for name in attribution_names(speaker):
        if "(" in name or name.startswith("The "):
            continue
        words = name.split()
        if len(words) < 2:
            continue
        if _PEER_TITLE.match(name):
            if " of " in name:
                forms.append(name.split(" of ")[0].strip())
            continue
        if words[-1].casefold() in _OFFICE_WORDS:
            continue
        for start in range(1, len(words)):
            tail = " ".join(words[start:])
            if words[start] in _HONORIFICS:
                continue
            forms.append(tail)
            forms += [f"{h} {tail}" for h in _HONORIFICS]
    full = {normalise(n) for n in attribution_names(speaker)}
    return tuple(dict.fromkeys(f for f in forms if f and normalise(f) not in full))


def _spelled_at_start(body: str, name: str) -> int | None:
    """How many characters at the start of `body` spell `name` — typography, case and spacing folded as `normalise`
    folds them, and ending on a word boundary ("Matheson" does not start with "Mather") — or None.
    """
    target = normalise(name)
    if not target or not normalise(body).startswith(target):
        return None
    for end in range(1, len(body) + 1):
        spelled = normalise(body[:end])
        if spelled == target:
            if end < len(body) and (body[end].isalnum() or body[end] == "_"):
                return None
            return end
        if len(spelled) > len(target):
            return None
    return None


def _reporting_verb_follows(rest: str) -> bool:
    following = rest.lstrip(" ,:").casefold()
    return any(following.startswith(verb) and (len(following) == len(verb) or not following[len(verb)].isalnum())
               for verb in REPORTING_VERBS)  # fmt: skip


def speaker_lead(text: str, speaker: str | None) -> str | None:
    """The words at the start of a claim that name its own speaker, exactly as the claim wrote them, or None."""
    if not text or not speaker or not speaker.strip():
        return None
    body = text.lstrip()
    for name in sorted(attribution_names(speaker), key=len, reverse=True):
        end = _spelled_at_start(body, name)
        if end is not None:
            return body[:end]
    for name in sorted(short_names(speaker), key=len, reverse=True):
        end = _spelled_at_start(body, name)
        if end is not None and _reporting_verb_follows(body[end:]):
            return body[:end]
    return None


def quote_segments(quote: str) -> list[str]:
    """A quote split on its ellipses, empty parts dropped."""
    text = quote
    for mark in ELLIPSES:
        text = text.replace(mark, "…")
    return [part.strip() for part in text.split("…") if part.strip()]


def quote_matches(quote: str, source: str) -> bool:
    """Whether `quote` appears in `source`, allowing for typography and ordered ellipsis segments."""
    haystack = normalise(source)
    if not haystack:
        return False
    position = 0
    for segment in quote_segments(quote) or [quote]:
        needle = normalise(segment)
        if not needle:
            continue
        found = haystack.find(needle, position)
        if found < 0:
            return False
        position = found + len(needle)
    return True


# --- sentences ------------------------------------------------------------------------------------------

# Abbreviations that end in a full stop mid-sentence.
_ABBREVIATIONS = frozenset({
    "no", "nos", "hon", "rt", "mr", "mrs", "ms", "dr", "sir", "st", "prof", "col", "capt", "gen", "sgt",
    "etc", "vs", "cf", "pp", "p", "vol", "ch", "para", "approx", "dept", "govt", "jr", "sr", "inc", "ltd",
    "e.g", "i.e", "viz", "ibid", "op", "cit", "al", "fig", "ref", "sec", "art", "cl",
})

_BREAK_RE = re.compile(r"([A-Za-z0-9.']*)([.!?])([\"')\]]?)\s+(?=[A-Z\"'(])")


def _is_break(match: re.Match[str]) -> bool:
    """Whether this stop really ends a sentence."""
    if match.group(2) in "!?":
        return True
    word = match.group(1).lower().rstrip(".")
    if not word:
        return True
    if word in _ABBREVIATIONS:
        return False
    return len(word) > 1  # a single letter is an initial: "A.


def sentence_breaks(text: str) -> list[int]:
    """Positions of genuine sentence boundaries, ignoring a trailing stop."""
    body = text.rstrip()
    if body.endswith((".", "!", "?")):
        body = body[:-1]
    return [m.end(2) for m in _BREAK_RE.finditer(body) if _is_break(m)]


def is_one_sentence(text: str) -> bool:
    return not sentence_breaks(text)


# --- numbers --------------------------------------------------------------------------------------------

# Written number words, as tokens for the compound parser rather than as standalone numbers.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}  # fmt: skip
_MULTIPLIERS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
                "trillion": 1_000_000_000_000, "dozen": 12}  # fmt: skip
_NUMBER_WORDS = {**_UNITS, **_TENS, **_MULTIPLIERS}
# Longest first, so the alternation cannot match "six" inside "sixteen" and leave "teen" behind.
_WORDS_ALT = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

# Suffixes that scale a digit-written number.
_SCALES = {"bn": 1e9, "billion": 1e9, "m": 1e6, "million": 1e6, "k": 1e3, "thousand": 1e3,
           "tn": 1e12, "trillion": 1e12}  # fmt: skip

# Written numbers below ten are not checked.
WORD_NUMBER_FLOOR = 10

_NUMBER_RE = re.compile(
    r"(?P<currency>[£$€])?\s*"
    r"(?P<value>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<scale>bn|tn|billion|million|trillion|thousand|k|m)\b)?"
    r"(?:\s*(?P<percent>%|per cent\b|percent\b))?",
    re.IGNORECASE,
)
# A run of number words, optionally joined by "and" or hyphens: "one hundred and fifty", "twenty-five".
_WORD_RUN_RE = re.compile(
    r"\b(?:" + _WORDS_ALT + r")(?:[\s-]+(?:and[\s-]+)?(?:" + _WORDS_ALT + r"))*\b",
    re.IGNORECASE,
)


def _evaluate_words(run: str) -> float | None:
    """Evaluate a run of number words: "one hundred and fifty" → 150, "twenty-five" → 25, "dozen" → 12."""
    tokens = [t for t in re.split(r"[\s-]+", run.lower()) if t and t != "and"]
    if not tokens:
        return None
    total = current = 0.0
    for token in tokens:
        if token in _UNITS:
            current += _UNITS[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token in ("hundred", "dozen"):
            current = (current or 1) * _MULTIPLIERS[token]
        else:  # thousand and above close off the group before them
            total += (current or 1) * _MULTIPLIERS[token]
            current = 0.0
    return total + current


def _token(value: float) -> str:
    """The comparable spelling of a number."""
    if value.is_integer() and abs(value) < 1e18:
        return str(int(value))
    return f"{value:.10g}"


def numbers_in(text: str | None) -> set[str]:
    """Every number in `text`, normalised so that £35 billion, £35bn and 35000000000 all compare equal."""
    body = text or ""
    found: set[str] = set()
    spans: list[tuple[int, int]] = []
    for match in _NUMBER_RE.finditer(body):
        spans.append(match.span())
        raw = match.group("value").replace(",", "")
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - the pattern cannot produce this
            continue
        scale = (match.group("scale") or "").lower()
        if scale:
            value *= _SCALES[scale]
        percent = bool(match.group("percent"))
        # A bare four-digit integer in the year range is a year, not a magnitude.
        is_year = (not scale and not percent and not match.group("currency")
                   and raw.isdigit() and len(raw) == 4 and 1000 <= value <= 2999)  # fmt: skip
        token = _token(value)
        found.add(f"{token}%" if percent else (f"year:{token}" if is_year else token))
    for match in _WORD_RUN_RE.finditer(body):
        # A scale word already consumed as a suffix is not a second number: "£35 billion" is one figure, and counting
        # "billion" again put 1000000000 into the claim that no quote could contain.
        if any(start < match.end() and match.start() < stop for start, stop in spans):
            continue
        value = _evaluate_words(match.group())
        if value is not None and value >= WORD_NUMBER_FLOOR:
            found.add(_token(value))
    return found


def missing_numbers(claim: str, evidence: str) -> set[str]:
    """Numbers the claim states that the evidence does not. Empty means the claim invents no figure."""
    return numbers_in(claim) - numbers_in(evidence)


# --- the contract ---------------------------------------------------------------------------------------


class Abstention(str, Enum):
    """Why no answer was given. A closed set: "I could not find it" is not a reason a user can act on."""

    NOT_IN_RECORD = "not_in_record"
    OUTSIDE_WINDOW = "outside_window"
    AMBIGUOUS = "ambiguous"
    FALSE_PREMISE = "false_premise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Citation(BaseModel):
    """One cited speech, the sentences within it, and the verbatim quote taken from them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    speech_id: str = Field(min_length=1)
    chunk_ids: tuple[int, ...] = Field(min_length=1, description="sentence ids; citations are sentence-level")
    hansard_url: str | None = None
    quote: str = Field(min_length=1)

    @field_validator("chunk_ids")
    @classmethod
    def _distinct(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(value)) != len(value):
            raise ValueError("chunk_ids must be distinct: a sentence cited twice inflates the evidence")
        return value


class Verification(BaseModel):
    """What each check concluded. Passing is a conjunction, not a vote."""

    model_config = ConfigDict(extra="forbid")

    quote_ok: bool | None = None
    numbers_ok: bool | None = None
    attribution_ok: bool | None = None
    citation_ok: bool | None = None
    window_ok: bool | None = None
    entailment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    entailment_verdict: Literal["supported", "unsupported", "unavailable"] | None = None
    checked_by: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def deterministic_passed(self) -> bool:
        """Every deterministic check that ran said yes, and at least one ran."""
        results = [self.quote_ok, self.numbers_ok, self.attribution_ok, self.citation_ok, self.window_ok]
        ran = [r for r in results if r is not None]
        return bool(ran) and all(ran)

    @property
    def showable(self) -> bool:
        """Whether this claim may be shown to a user at all (truth rule 3)."""
        return self.deterministic_passed and self.entailment_verdict != "unsupported"


class Claim(BaseModel):
    """One atomic, cited, checkable statement. One sentence, never two."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["said", "summary"]
    text: str = Field(min_length=1)

    speaker_mnis_id: int | None = None
    speaker_name: str | None = None
    party_at_date: str | None = None
    office_at_date: str | None = None

    date: Date | None = None
    house: str | None = None
    section: str | None = None

    citations: tuple[Citation, ...] = Field(min_length=1)
    verification: Verification = Field(default_factory=Verification)

    @field_validator("text")
    @classmethod
    def _atomic(cls, value: str) -> str:
        """One sentence. An atomic claim is the unit the verifier can pass or fail; two sentences joined by a full stop
        can be half true, and half true is what gets shown.
        """
        text = value.strip()
        if not text:
            raise ValueError("a claim needs text")
        if not is_one_sentence(text):
            raise ValueError(
                f"a claim must be ONE sentence and this is more than one: {text[:80]!r}. Split it, so each "
                "half can be verified or dropped on its own."
            )
        return text

    @model_validator(mode="after")
    def _numbers_are_in_the_quotes(self) -> Claim:
        """Every number the claim states must appear in the text it cites."""
        evidence = " ".join(c.quote for c in self.citations)
        absent = missing_numbers(self.text, evidence)
        if absent:
            raise ValueError(
                f"claim {self.id} states number(s) {sorted(absent)} that its own citations do not: "
                f"{self.text[:80]!r}"
            )
        return self

    @property
    def numbers(self) -> set[str]:
        return numbers_in(self.text)

    @property
    def cited_chunk_ids(self) -> set[int]:
        return {chunk for citation in self.citations for chunk in citation.chunk_ids}

    @property
    def cited_speech_ids(self) -> set[str]:
        return {citation.speech_id for citation in self.citations}


class StructuredFact(BaseModel):
    """A fact from SQL. `source` is fixed: a model may never supply one (truth rule 1)."""

    model_config = ConfigDict(extra="forbid")

    template: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    value: dict[str, Any] | None = None
    source: Literal["sql"] = "sql"
    caveat: str = ""


class RosterEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mnis_id: int | None = None
    speaker_name: str = Field(min_length=1)
    party_at_date: str | None = None
    office_at_date: str | None = None
    contributions: int = Field(ge=1)
    best_chunk_id: int | None = None
    # False when the member had the floor but every contribution is the reporter's note of a gap in the transcript
    # ("[Inaudible.]"): listed, and the reader told the words are not recorded.
    words_recorded: bool = True


class Roster(BaseModel):
    """The set route's output: who spoke, from SQL, once retrieval has identified the debate."""

    model_config = ConfigDict(extra="forbid")

    debate_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    entries: tuple[RosterEntry, ...] = ()


class Provenance(BaseModel):
    """What produced this answer. Without it a cached answer from before a freeze is indistinguishable from one taken
    after it, which is the whole reason the retrieval config is hashed.
    """

    model_config = ConfigDict(extra="forbid")

    route: str = Field(min_length=1)
    resolved_filters: dict[str, Any] = Field(default_factory=dict)
    corpus_version: str = Field(min_length=1)
    retrieval_config_hash: str = Field(min_length=1)
    recipe: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    # A fingerprint of the code that produced the answer (`answer.keys.code_version`).
    code_version: str | None = None
    system_fingerprint: str | None = None


class Answer(BaseModel):
    """Everything the system says, and everything needed to check it."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    provenance: Provenance
    claims: tuple[Claim, ...] = ()
    structured_facts: tuple[StructuredFact, ...] = ()
    roster: Roster | None = None
    facets: dict[str, Any] = Field(default_factory=dict)
    abstention: Abstention | None = None
    abstention_detail: str = ""
    coverage_note: str = ""
    rendered: str = ""
    # Every chunk id the retrieval handed over.
    retrieved_chunk_ids: frozenset[int] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def _answers_or_abstains(self) -> Answer:
        has_content = bool(self.claims or self.structured_facts or self.roster or self.facets)
        if self.abstention and has_content:
            raise ValueError(
                f"an answer that abstains ({self.abstention.value}) must not also assert claims or facts: "
                "a user cannot act on both at once"
            )
        if not self.abstention and not has_content:
            raise ValueError(
                "an answer with nothing in it must say why it abstained. 'No results' is not an answer; "
                "pick a reason from Abstention so the user knows whether to rephrase or give up."
            )
        return self

    @model_validator(mode="after")
    def _citations_were_retrieved(self) -> Answer:
        """'s citation check, at construction. A cited id that was never retrieved is fabricated."""
        if not self.retrieved_chunk_ids:
            return self  # nothing to check against; the verifier reports this separately
        for claim in self.claims:
            outside = sorted(claim.cited_chunk_ids - set(self.retrieved_chunk_ids))
            if outside:
                raise ValueError(
                    f"claim {claim.id} cites chunk(s) {outside} that were never retrieved for this question"
                )
        return self

    @model_validator(mode="after")
    def _rendered_text_is_verified(self) -> Answer:
        """Unverified text is never shown (truth rule 3)."""
        if not self.rendered:
            return self
        unshowable = [c.id for c in self.claims if not c.verification.showable]
        if unshowable:
            raise ValueError(
                f"rendered text is set while claim(s) {unshowable} have not passed verification. Verify first, "
                "then render from what passed."
            )
        return self

    @property
    def showable_claims(self) -> tuple[Claim, ...]:
        return tuple(c for c in self.claims if c.verification.showable)

    def render(self) -> str:
        """The answer text, built deterministically from verified claims and structured facts."""
        parts: list[str] = []
        for fact in self.structured_facts:
            if fact.value is not None:
                body = ", ".join(f"{k} {v}" for k, v in fact.value.items() if v is not None)
                parts.append(f"{body}.{(' ' + fact.caveat) if fact.caveat else ''}")
        for claim in self.showable_claims:
            # Measured on the first real answer this system produced.
            text = claim.text.strip()
            who = claim.speaker_name or "A member"
            when = f" [{claim.date}]" if claim.date else ""
            lead = speaker_lead(text, claim.speaker_name)
            if lead is None:
                parts.append(f"{who} said: {text}{when}")
            elif normalise(lead) in {normalise(n) for n in attribution_names(claim.speaker_name)}:
                parts.append(f"{text}{when}")
            else:
                # Named by a shorter form ("Mather said that …") — the printed name takes its place, rather than a
                # second attribution in front of it.
                parts.append(f"{who}{text[len(lead):]}{when}")
        if self.roster and self.roster.entries:
            names = ", ".join(e.speaker_name for e in self.roster.entries)
            parts.append(f"Speakers: {names}.")
        if self.coverage_note:
            parts.append(self.coverage_note)
        return " ".join(parts).strip()


Answered = Annotated[Answer, Field(description="a complete, checkable answer or a reasoned abstention")]
