"""Turning a question's words into the parameters a structured fact needs."""

from __future__ import annotations

import logging
import re
import weakref
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

from answer import present as PRESENT
from procedure import eras as ERAS

log = logging.getLogger("answer.resolve")

# Words that carry no signal when matching a division title.
_NOISE = frozenset("""
the a an of on in at to for and or was were is are be been being that this those these it its
motion question division divisions vote voted carried agreed rejected negatived passed
lords commons house did was what were who when which how many number
house's lords' commons'
""".split())

_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*", re.I)
# The references that tell two divisions on one day apart: a KIND ("new clause", "amendment", "no.") and its references,
# which the record prints singly ("New Clause 6"), as lists and ranges ("Amendments 23 and 106 to 120", "LA23 and LA106
# to LA120"), with letters for later rounds ("Lords Amendment 1B", "120C, 120D and 120E", "LA72D to LA72H"), as lettered
# amendments ("Amendment A1", "amendments (a) to (f)") and, for Lords amendments, as `LA`.
_KIND = (r"new\s+clauses?|new\s+schedules?|clauses?|amendments?|schedules?|ways\s+and\s+means\s+motions?|motions?|"
         r"standing\s+orders?(?:\s+no\.?)?|instructions?|no\.?|"
         # the record's shorthand for Government amendments: "to propose Gov (a) to (f) in lieu of LA23" (QA pass 4)
         r"gov(?:t|ernment)?\.?(?=\s*\()")
# One reference.
_REF = (r"(?:(?-i:LA)?\d+(?-i:[A-Z]{0,2})(?!\w)|(?-i:[A-Z]{1,2})\d+(?-i:[A-Z]?)(?!\w)|\([a-z]{1,2}\)"
        r"|(?-i:[A-Z])(?![\w(]))")
_SEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+or\s+|\s+to\s+)"
_DISTINCTIVE = re.compile(rf"(?:\b(?P<kind>{_KIND})\s*|(?=\b(?-i:LA)\d))(?P<refs>{_REF}(?:{_SEP}{_REF})*)", re.I)
# "in clause 5, page 3", "after clause 2", "leave out clause 9": where an amendment goes, not what a division is on.
_LOCATION = re.compile(r"\b(?:in|after|before|into|to|leave\s+out)\s+$", re.I)
# "moved amendment 7A, as an amendment to amendment 4": 4 is the target, not what the division decided (QA pass 4: the
# division on 7A was answered as amendment 4's).
_TARGET = re.compile(r"\bas\s+an\s+amendment\s+to\s+(?:the\s+)?$", re.I)
# QA pass 5 (H2): the other ways the record reaches one amendment through another.
_AMENDING_CLAUSE = re.compile(r"\bas\s+(?:an\s+)?amendments?\s+to\b(?P<clause>[^,;.\u201c\u201d\"]*)$", re.I)
_LETTERED_TO = re.compile(r"(?:\)|\b\d+[A-Z]{0,2})\s+to\s+(?:the\s+)?(?:lords|commons)\s+$", re.I)
# A lettered reference in a title: where one is left unread, the title names business the reader cannot see.
_LETTERED = re.compile(r"\((?-i:[a-z]{1,2})\)")
# "the amendment I asked about": the pronoun, after a determiner and before a lower-case word, is no reference.
_DETERMINER = re.compile(r"\b(?:the|that|this|which|an?)\s+$", re.I)
# Vocabulary that names a reference: where the record uses it and the reader parses nothing, the division might be on
# anything of that kind, so no absence is claimed for its day.
_VOCAB = re.compile(r"\b(?:new\s+clauses?|new\s+schedules?|clauses?|amendments?|schedules?)\b|(?-i:\bLA)(?=\d)", re.I)
# ... and "moved amendment 7A, as an amendment to amendment 4" describes the relation, not another reference.
_DESCRIPTIVE = re.compile(r"\b(?:insert|add)\s+(?:the\s+)?(?:following\s+)?new\s+(?:clause|schedule)\b"
                          r"|\bas\s+an\s+amendment\s+to\b", re.I)
# The longest range expanded ("Amendments 106 to 120"); a longer one keeps its two ends.
_MAX_RANGE = 300


def _kind_name(raw: str) -> str:
    kind = re.sub(r"[\s.]+", " ", raw or "amendment").strip().lower()
    if kind.startswith("standing order"):
        return "standing order"  # "Standing Order No. 24" and "Standing Order 24"
    if kind.startswith("ways and means"):
        return "ways and means motion"
    if kind.startswith("gov"):
        return "amendment"
    return {"clauses": "clause", "new clauses": "new clause", "amendments": "amendment", "schedules": "schedule",
            "new schedules": "new schedule", "motions": "motion", "instructions": "instruction"}.get(kind, kind)


def _ref(token: str) -> str:
    token = token.strip()
    if token.startswith("("):
        return token.lower()  # "(D)" is "(d)": the record letters them in lower case (QA pass 5, H6)
    return token[2:] if token[:2] == "LA" else token


def _expand(lo: str, hi: str) -> list[str]:
    """A range's references: "106 to 120", "72D to 72H", "(a) to (f)"; its ends alone when it is no simple range."""
    if lo.isdigit() and hi.isdigit() and 0 < int(hi) - int(lo) <= _MAX_RANGE:
        return [str(n) for n in range(int(lo), int(hi) + 1)]
    a, b = re.fullmatch(r"(\d*)([A-Z]?)", lo), re.fullmatch(r"(\d*)([A-Z]?)", hi)
    if a and b and a.group(1) == b.group(1) and a.group(2) and b.group(2) and a.group(2) < b.group(2):
        return [a.group(1) + chr(c) for c in range(ord(a.group(2)), ord(b.group(2)) + 1)]
    pa, pb = re.fullmatch(r"\(([a-z])\)", lo), re.fullmatch(r"\(([a-z])\)", hi)
    if pa and pb and pa.group(1) < pb.group(1):
        return [f"({chr(c)})" for c in range(ord(pa.group(1)), ord(pb.group(1)) + 1)]
    return [lo, hi]


def _references(text: str):
    """Each reference the text makes: (kind, references, span), location phrases included (kind "location")."""
    for m in _DISTINCTIVE.finditer(text or ""):
        kind = _kind_name(m.group("kind") or "")
        before = text[max(0, m.start() - 30):m.start()]
        if (m.group("refs").strip() == "I" and _DETERMINER.search(before)
                and re.match(r"\s+(?-i:[a-z])", text[m.end():])):  # fmt: skip
            continue  # the pronoun (QA pass 4, L5)
        if kind in ("clause", "schedule") and _LOCATION.search(text[max(0, m.start() - 12):m.start()]):
            kind = "location"
        elif (_TARGET.search(before) or _AMENDING_CLAUSE.search(text[max(0, m.start() - 200):m.start()])
              or _LETTERED_TO.search(text[max(0, m.start() - 20):m.start()]) or _quoted(text, m.start())):
            kind = f"target:{kind}"  # what the division's amendment is about, of its own kind
        parts = re.split(rf"({_SEP})", m.group("refs"))
        refs: list[str] = []
        for i in range(0, len(parts), 2):
            token = _ref(parts[i])
            if i >= 2 and parts[i - 1].strip() == "to" and refs:
                refs[-1:] = _expand(refs[-1], token)
            else:
                refs.append(token)
        yield kind, set(refs), m.span()


def _quoted(text: str, at: int) -> bool:
    """Whether position `at` lies inside quoted words (the record's curly quotes, or straight ones)."""
    head = text[:at]
    return head.count("\u201c") > head.count("\u201d") or head.count('"') % 2 == 1


def _marks(text: str, *, targets: bool = False) -> dict[str, set[str]]:
    """References grouped BY KIND: {"new clause": {"3"}, "no": {"2"}, "amendment": {"23", "106",...}}."""
    found: dict[str, set[str]] = {}
    for kind, refs, _ in _references(text):
        if kind.startswith("target:") and targets:
            kind = kind.split(":", 1)[1]  # a question naming "(a) to Lords Amendment 2" names both, each of its kind
        if kind != "location" and not kind.startswith("target:"):
            found.setdefault(kind, set()).update(refs)
    return found


def _base(ref: str) -> str:
    """The amendment a lettered round belongs to: "48B" is a later round of 48."""
    digits = re.match(r"\d+", ref)
    return digits.group(0) if digits else ref


# The marks that name one piece of business: a reference to one of these names one division at most.
_CLAUSE_KINDS = frozenset({"new clause", "new schedule", "amendment", "clause", "schedule", "motion"})


def _held(row: dict) -> dict[str, set[str]]:
    return _marks(f"{row.get('title') or ''} {row.get('motion_text') or ''}")


def _targets(row: dict) -> dict[str, set[str]]:
    """What the division's amendment amends ("as an amendment to amendment 297"), by kind."""
    found: dict[str, set[str]] = {}
    for kind, refs, _ in _references(f"{row.get('title') or ''} {row.get('motion_text') or ''}"):
        if kind.startswith("target:"):
            found.setdefault(kind.split(":", 1)[1], set()).update(refs)
    return found


def _rounds(row: dict) -> dict[str, set[str]]:
    """The held amendments that have ROUNDS: those exchanged between the Houses ("Lords Amendment 48B", "LA48B",
    "Commons Amendment 120C"), whose letters are successive rounds of one amendment. A lettered amendment moved at a
    stage ("amendment 387A") is an amendment of its own.
    """
    text = f"{row.get('title') or ''} {row.get('motion_text') or ''}"
    found: set[str] = set()
    for m in _DISTINCTIVE.finditer(text):
        kind = _kind_name(m.group("kind") or "")
        house = re.search(r"\b(?:lords|commons)['\u2019]?\s+$", text[max(0, m.start() - 12):m.start()], re.I)
        if kind == "amendment" and (house or "LA" in m.group("refs")):
            found |= {_ref(t) for t in re.split(_SEP, m.group("refs")) if t.strip()}
    held = _held(row).get("amendment", set())
    return {"amendment": held & found} if held & found else {}


def _a_round_of(asked: dict[str, set[str]], row: dict) -> bool:
    """Whether the division carries another round of what is asked ("48B" for 48), and is not merely an amendment TO
    it: "297AA, as an amendment to amendment 297" is not a round of 297.
    """
    return _holds(asked, _rounds(row), exact=False) and not _holds(asked, _targets(row), exact=True)


def _holds(asked: dict[str, set[str]], held: dict[str, set[str]], *, exact: bool, every: bool = False) -> bool:
    """Whether `held` carries what `asked` names: any of it, or with `every`, all of it (each reference, of each kind).
    Exactly, or by the amendment each round belongs to.
    """
    def norm(refs: set[str]) -> set[str]:
        return refs if exact else {_base(r) for r in refs}

    if every:
        return all(norm(refs) <= norm(held.get(kind, set())) for kind, refs in asked.items())
    return any(norm(refs) & norm(held.get(kind, set())) for kind, refs in asked.items())


def _asked_marks(question: str) -> dict[str, set[str]]:
    """The clause, schedule and amendment references a question makes, its dates taken out first ("on New Clause 2 and
    13 May 2020" names New Clause 2 only).
    """
    return {k: v for k, v in _marks(_undated(question), targets=True).items() if k in _CLAUSE_KINDS}


def _carrying(question: str, rows: list[dict]) -> list[dict] | None:
    """When the question names a clause, schedule or amendment: the divisions that carry it, in their title or motion
    (the Lords print amendment numbers in the motion); failing an exact reference, those carrying another round of
    it ("LA48B" for "Lords Amendment 48"). None when it names none, so the title score alone decides.
    """
    asked = _asked_marks(question)
    if not asked:
        return None
    # every reference named, not any: "amendment (a) to Lords Amendment 900" is not a division on (a) to Lords Amendment
    # 1
    exact = [r for r in rows if _carries(asked, r, exact=True)]
    return exact or [r for r in rows if _carries(asked, r, exact=False)]


def _carries(asked: dict[str, set[str]], row: dict, *, exact: bool) -> bool:
    """Whether the division decided what the question names: every reference named is held or is what the held one
    amends ("(a) to Lords Amendment 2" names both), and at least one is held itself, so a division whose amendment
    merely amends the one asked (387B for 387, Motion 31A for Commons Amendment 31) does not carry it (QA pass 5).
    """
    held, aimed = _held(row), _targets(row)
    union = {k: held.get(k, set()) | aimed.get(k, set()) for k in set(held) | set(aimed)}
    if not _holds(asked, union, exact=exact, every=True):
        return False
    # exactly, any held reference; by round, only an amendment exchanged between the Houses has rounds
    if not _holds(asked, held if exact else _rounds(row), exact=exact):
        return False
    # a later round counts ("48B" for 48), an amendment TO the one asked does not ("297AA" for 297)
    return exact or not _holds(asked, aimed, exact=True)


# A title that is a bill's name and nothing more: with no motion, nothing says what the division was on.
_BARE_BILL = re.compile(r"^[^:]*\bBill\b(?:\s*\[(?:HL|Lords)\])?\s*\.?$", re.I)


def _readable(row: dict) -> bool:
    """Whether everything a division is on can be read from its title and motion: no reference word goes unparsed and,
    for a bill, something beyond the bill's name says which business it was.
    """
    text = f"{row.get('title') or ''} {row.get('motion_text') or ''}"
    covered = [span for *_, span in _references(text)] + [m.span() for m in _DESCRIPTIVE.finditer(text)]
    for m in _VOCAB.finditer(text):
        if not any(a <= m.start() < b for a, b in covered):
            return False
    # A lettered reference the title leaves unread ("propose (a) and (b) in lieu"): the title only, because a Lords
    # motion quotes the amendment's own text, whose "(a)" is a paragraph, not a reference.
    title = row.get("title") or ""
    for m in _LETTERED.finditer(title):
        if not any(a <= m.start() < b for a, b in covered):
            return False
    return bool((row.get("motion_text") or "").strip()) or not _BARE_BILL.match((row.get("title") or "").strip())


def _absence_claimable(asked: dict[str, set[str]], rows: list[dict]) -> bool:
    """Whether "no division that day is on X" is what the record shows: no division carries X or another round of it,
    and every division can be read. Otherwise the absence is not claimed.
    """
    if any(_holds(asked, _held(r), exact=True) or _a_round_of(asked, r) or _holds(asked, _targets(r), exact=False)
           for r in rows):  # fmt: skip
        return False  # held, a round of it held, or mentioned (amended, quoted): "no division was on it" is not said
    return all(_readable(r) for r in rows)


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in _NOISE and len(w) > 2}


def _undated(question: str) -> str:
    """The question with its full date expressions taken out (ISO, "20 May 2020", "May 20, 2020"): the day the filter
    chose. A month or a year stays, since titles name them ("Spending Review (June 2025)", "...
    """
    from answer import router as RO

    text = question or ""
    for pattern in (RO._ISO, RO._DMY, RO._MDY):
        text = pattern.sub(" ", text)
    return text


# Where a question names a bill: the words just before "Bill" ("the Rwanda Bill", "Finance (No. 2) Bill's"), read back
# to the first word that cannot be part of a bill's name.
_BILL_WORD = re.compile(r"\bbill(?:['’]s)?\b", re.I)
_NOT_A_BILL_NAME = frozenset("""
the a an on to in at about over with by from for did does do was were is are vote voted votes voting division divided
how which what when who sir mr mrs ms miss dr dame lord lady baroness and or this that its their his her
""".split())
_TOKEN = re.compile(r"[\w'’()\[\].,-]+")
# The stages a question, a title or a stage link can name, normalised ("second reading" and "2nd Reading" are one
# stage): the House's own names for them.
_STAGE_FORMS = ((re.compile(r"\b(?:first|1st)\s+reading\b|\bread\s+a\s+first\s+time\b", re.I), "1st reading"),
                (re.compile(r"\b(?:second|2nd)\s+reading\b|\bread\s+a\s+second\s+time\b", re.I), "2nd reading"),
                (re.compile(r"\b(?:third|3rd)\s+reading\b|\bread\s+a\s+third\s+time\b", re.I), "3rd reading"),
                (re.compile(r"\breport\s+stage\b|\b(?:on|at)\s+report\b(?!\s+(?:of|from|on|by)\b)", re.I), "report stage"),
                (re.compile(r"\bcommittee\s+stage\b|\bcommittee\s+of\s+the\s+whole\s+house\b|\bgrand\s+committee\b"
                            r"|\bin\s+committee\b(?!\s+(?:room|on|of)\b)", re.I), "committee stage"),
                (re.compile(r"\bconsideration\s+of\s+(?:the\s+)?(?:commons|lords)(?:['\u2019])?\s+(?:amendments|message|reasons)\b",
                            re.I), "consideration"))  # fmt: skip


def linked_stage(stage_name: str | None) -> str | None:
    """The stage the Bills API links a division to (`bill_stage.stage_name`), in the same vocabulary, or None."""
    found = _stages(stage_name or "")
    if not found and re.search(r"\bcommittee\b", stage_name or "", re.I):
        found = {"committee stage"}  # "Committee stage", "Committee of the whole House", "Grand Committee"
    return next(iter(sorted(found)), None)


def _bill_names(text: str) -> list[set[str]]:
    """The terms of each bill name the text gives before the word "Bill"; empty sets are left out. The apostrophe in
    either printing, and a possessive, are one form ("Children's", "Children’s", "Childrens"; QA pass 8, Med 4).
    """
    out = []
    for m in _BILL_WORD.finditer(text or ""):
        words = _TOKEN.findall(text[max(0, m.start() - 120):m.start()])
        name: list[str] = []
        for word in reversed(words[-8:]):
            if word.lower().strip("(),.[]") in _NOT_A_BILL_NAME and word.lower() not in ("and", "of"):
                break
            name.append(word)
        terms = _terms(" ".join(_name_words(" ".join(reversed(name)))))
        if terms:
            out.append(terms)
    return out


# --- which bill, by the record's own bills (S05's close, QA pass 8, H1)
# ------------------------------------------------

# An Act's citation, the number of a second Act of a year before the year ("Finance Act (No. 2) 2024"): it was the Bill
# of the same name and number ("Finance (No. 2) Bill").
_ACT_TITLE = re.compile(r"^(?P<name>.+?)\s+Act(?P<no>\s*\(No\.?\s*\d+\))?\s+\d{4}$", re.I)


def _name_words(text: str | None) -> list[str]:
    """A bill's name as its words in order: case folded, the [HL] marker dropped, the apostrophe in either printing and
    a possessive taken off, and a final "s" folded on both sides of every comparison, so "Children's", "Children’s"
    and "Childrens" are one word (a word ending "ss", "Business", keeps it).
    """
    folded = (text or "").casefold().replace("’", "'").replace("‘", "'").replace("[hl]", " ")
    folded = re.sub(r"(\w)'s\b", r"\1", folded)
    folded = re.sub(r"(\w)s'(?!\w)", r"\1s", folded)
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z0-9]+", folded)]


@dataclass(frozen=True)
class BillNames:
    """The record's bills by name (`served.bill`): each title, and an Act's name as the Bill it was ("Sentencing Act
    2026" is the Sentencing Bill), as word sequences.
    """

    names: tuple[tuple[tuple[str, ...], int], ...]

    @classmethod
    def from_rows(cls, rows) -> BillNames:
        names = set()
        for row in rows:
            for alias in (row.get("title"), row.get("former_short_title")):
                if not alias:
                    continue
                act = _ACT_TITLE.match(alias.strip())
                for form in (alias, f"{act.group('name')}{act.group('no') or ''} Bill" if act else None):
                    words = tuple(_name_words(form))
                    if form and words:
                        names.add((words, int(row["id"])))
        return cls(tuple(sorted(names)))

    def named(self, text: str | None) -> frozenset[int]:
        """The bills a text names: every name found whole and in order, except one lying inside a longer name found
        there ("Sentencing Bill" inside "Sentencing (Women) Bill" is not named; two bills of one name both are).
        """
        words = _name_words(text)
        found = [(i, i + len(name), bid) for name, bid in self.names
                 for i in range(len(words) - len(name) + 1) if tuple(words[i:i + len(name)]) == name]  # fmt: skip
        return frozenset(bid for a, b, bid in found
                         if not any(a2 <= a and b <= b2 and b2 - a2 > b - a for a2, b2, _ in found))


def annotate_bills(names: BillNames, question: str, rows: list[dict]) -> list[dict]:
    """Each candidate division with the bills the question names and the bills the division is on: its stage's bill
    (`bill_id`) and the bills its title names, or its motion's where the title names none.
    """
    asked = names.named(_undated(question))
    for row in rows:
        own = set(names.named(row.get("title")))
        if row.get("bill_id") is not None:
            own.add(int(row["bill_id"]))
        if not own:
            own = set(names.named(row.get("motion_text")))
        row["bills"] = (asked, frozenset(own))
    return rows


_NAMES_BY_CONN: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def with_bills(conn, question: str, rows: list[dict]) -> list[dict]:
    """`annotate_bills` with the record's bills, read once per connection (the masked `served.bill` on a reader's).
    """
    names = _NAMES_BY_CONN.get(conn)
    if names is None:
        names = BillNames.from_rows(await _all(conn, "SELECT id, title, former_short_title FROM bill"))
        _NAMES_BY_CONN[conn] = names
    return annotate_bills(names, question, rows)


def _stages(text: str) -> set[str]:
    return {stage for pattern, stage in _STAGE_FORMS if pattern.search(text or "")}


# A question asking whether the House passed, approved or rejected the business itself (a bill, an Act, an instrument),
# rather than asking about a division.
_PASSAGE = re.compile(
    r"\b(?:did|has|had|have|would)\s+(?:the\s+)?(?:house(?:\s+of\s+(?:commons|lords))?|commons|lords|mps|peers|"
    r"parliament)\s+(?:pass|approve|reject|back|support|accept|endorse|throw\s+out|vote\s+(?:for|against|down|through))\b"
    r"|\b(?:was|were)\s+(?:the\s+)?[^?]{0,160}?\b(?:bill|act|regulations?|order|rules|instrument|measure)\b(?:\s*\[hl\])?"
    r"\s+(?:passed|approved|rejected|backed|accepted|defeated|thrown\s+out|voted\s+(?:down|through))\b", re.I)
# A division on an amendment, or on a motion to amend, not on the business itself.
_AMENDMENT_LIKE = re.compile(
    r"\bamendments?\b|\bnew\s+(?:clause|schedule)s?\b|\bclauses?\s+\d|\bschedules?\s+\d|\bmotion\s+(?-i:[A-Z])\d*\b"
    r"|\bleave\s+out\b|\binsert\b|\bdeclines?\b|\bregret|\breasons?\s+\d|\binsist|\bdisagree|\bin\s+lieu\b"
    r"|\bstand\s+part\b|\bas\s+an\s+amendment\b", re.I)


def _fate_object(question: str) -> str | None:
    """What a FATE question asks the fate of (whether it was agreed, passed, carried, lost …), in the router's words
    for a House deciding (`router.RESULT_VERBS`, one list: QA pass 5, H1): "did the Lords agree to the X
    Regulations", "was the X Bill carried", "were the X Regulations agreed". None for a question about a division
    ("what was the result of the division on X").
    """
    from answer import router as RO

    text = " ".join(_undated(question).split())
    m = re.search(r"\b(?:did|has|had|have|would)\s+(?:the\s+)?(?:house(?:\s+of\s+(?:commons|lords))?|commons|lords|mps|"
                  rf"peers|members|parliament)\s+(?:{RO.RESULT_VERBS})\b(?P<o>[^?]*)", text, re.I)  # fmt: skip
    if m:
        return m.group("o")
    m = re.search(rf"\b(?:was|were)\s+(?P<o>[^?]{{0,220}}?)\s+(?:{RO.RESULT_PARTICIPLES})\b", text, re.I)
    return m.group("o") if m else None


_BUSINESS = re.compile(r"\b(?:bill|act|regulations?|order|rules|instrument|measure|scheme)\b|\bmotion\s+to\s+approve\b"
                       r"|\bmotion\s+that\s+this\s+house\s+approve\b", re.I)


def asks_passage(question: str) -> bool:
    """Whether the question asks the fate of the business itself (a bill, an Act, an instrument, "Business of the
    House", "the motion to approve X": anything it names that is no reference), naming no amendment, clause or
    motion: whatever the verb.
    """
    obj = _fate_object(question)
    named = obj and (_BUSINESS.search(obj) or re.search(r"\b(?!(?:Lords|Commons|House|It|They)\b)[A-Z][a-z]", obj))
    if not named:
        return False
    if re.search(r"\bmotion\b", obj, re.I) and not re.search(r"\bmotion\s+(?:to|that\s+this\s+house)\s+approve\b", obj,
                                                             re.I):  # fmt: skip
        return False  # "was the motion on X carried" asks about the division's own question
    return not _asked_marks(question) and not _kinds_unnumbered(question)


def _amendment_fate(question: str) -> dict[str, set[str]]:
    """The references whose fate the question asks ("was Lords Amendment 1B agreed", "did the Lords agree to Commons
    Amendment 31"), or {}.
    """
    obj = _fate_object(question)
    if not obj:
        return {}
    # a question that names the motion about the amendment ("motion to disagree with Lords Amendment 5 passed") asks the
    # motion's fate, which the division on that motion answers
    named_motion = _motioned({"title": obj, "motion_text": ""})
    return {k: refs - named_motion.get(k, set()) for k, refs in _asked_marks(obj).items()
            if refs - named_motion.get(k, set())}


# The record's ways of putting a motion ABOUT an amendment to a division: to agree, disagree or insist, or something in
# lieu of it.
_MOTION_ABOUT = re.compile(r"\b(?:dis)?agree(?:ment)?\b|\binsist|\bin\s+lieu\b|\bnot\s+(?:to\s+)?insist", re.I)


def _motioned(row: dict) -> dict[str, set[str]]:
    """The amendments the division's question was a motion about (QA pass 5, M3), as distinct from the one it moved.
    """
    text = f"{row.get('title') or ''} {row.get('motion_text') or ''}"
    found: dict[str, set[str]] = {}
    for kind, refs, (start, _) in _references(text):
        if kind in ("amendment", "target:amendment") and _MOTION_ABOUT.search(text[max(0, start - 60):start]):
            found.setdefault("amendment", set()).update(refs)
    return found


def decides_passage(question: str, row: dict) -> bool:
    """Whether the division decided the business's own fate (QA pass 5, M1: "Instruction (No. 3)" answered "did the
    Commons pass the Bill"): a reading stage, "do now pass", or an instrument's own approval. To PASS a bill, asked
    with no stage named, is its Third Reading (or "do now pass"), not its Second.
    """
    if amendment_like(row):
        return False
    text = f"{row.get('title') or ''} {row.get('motion_text') or ''}"
    stages = _stages(text)
    if re.search(r"\bdo\s+now\s+pass\b", text, re.I):
        stages.add("3rd reading")
    instrument = bool(re.search(r"\b(?:regulations?|order|rules|scheme|measure)\b", row.get("title") or "", re.I)) and \
        not re.search(r"\bbill\b", row.get("title") or "", re.I)  # fmt: skip
    obj = _fate_object(question) or ""
    # the same KIND of business: a bill's reading never decides an instrument, nor an instrument's approval a bill
    asked_instrument = bool(re.search(r"\b(?:regulations?|order|rules|scheme|measure)\b", obj, re.I)) and \
        not re.search(r"\bbill\b", obj, re.I)  # fmt: skip
    if asked_instrument != instrument and (asked_instrument or re.search(r"\bbill\b", obj, re.I)):
        return False
    if instrument:
        return True
    passing = re.search(r"\bpass(?:ed)?\b", question, re.I) and not _stages(question)
    if passing:
        return "3rd reading" in stages
    return bool(stages & {"2nd reading", "3rd reading"}) or bool(obj and _stages(obj) & stages)


# "Amendment" as part of a name: "(Fees Amendment)", "(Amendment) Act", "Amendment Regulations".
_NAMED_AMENDMENT = re.compile(r"\([^()]*\bamendments?\b[^()]*\)|\bamendments?\s+(?:\(no\.?\s*\d+\)\s+)?"
                              r"(?:regulations?|order|rules|act|bill|scheme)\b", re.I)


def amendment_like(row: dict) -> bool:
    """Whether the division was on an amendment, or a motion to amend: never because the instrument's or the bill's own
    name says Amendment ("Draft Medical Devices (Fees Amendment) Regulations 2026"; pass 1 on QA pass 5's fixes).
    """
    text = _NAMED_AMENDMENT.sub(" ", f"{row.get('title') or ''} {row.get('motion_text') or ''}")
    return bool(_AMENDMENT_LIKE.search(text))


def passage_divisions(question: str, rows: list[dict]) -> list[dict]:
    """For a passage question ("did the Lords pass the X Bill"), the day's divisions on that business when every one of
    them was on an amendment: the reader is shown them, named, with no result asserted. Empty when the question is
    no passage question, or when a division on the business itself exists (that one answers it).
    """
    if not asks_passage(question):
        return []
    on_business = [r for r in rows if not _contradicts_business(question, r)]
    # Listed whenever none of them is shown deciding passage: on an amendment, or on a motion the record does not
    # describe (a bare Lords title), the result would be an answer to another question.
    if not on_business or any(decides_passage(question, r) for r in on_business):
        return []
    return on_business


def contradicts(question: str, row: dict) -> bool:
    """Whether the division cannot be what the question asks about (see `_contradicts_business`), or answers a FATE
    question with a division that did not decide that fate: an amendment's for the business, a motion that is no
    passage for a bill (QA pass 5, M1), or a motion about an amendment, or an amendment to it, for the amendment
    itself (QA pass 5, H2 and M3).
    """
    if asks_passage(question) and not decides_passage(question, row):
        # "Did the Lords pass the Tobacco and Vapes Bill" is not answered by the division on Lord Murray's amendment 1,
        # and "did the Lords approve the Regulations" not by an amendment to decline to approve them, whose defeat means
        # they WERE approved.
        return True
    fate = _amendment_fate(question)
    if fate:
        held = _held(row)
        about = {k: _motioned(row).get(k, set()) | _targets(row).get(k, set()) for k in fate}
        decided = {k: held.get(k, set()) - about.get(k, set()) for k in fate}
        if _holds(fate, about, exact=False) and not _holds(fate, decided, exact=True):
            # "Was Lords Amendment 1B agreed": the division was on the motion to insist on disagreeing with it
            return True
    return _contradicts_business(question, row)


def _contradicts_business(question: str, row: dict) -> bool:
    """Whether the division cannot be what the question asks about: the question names a bill the division's bills do
    not match, a stage the division does not show, or a reference its title contradicts (QA pass 3, finding 1). "The
    Rwanda Bill" is not the Post Office (Horizon System) Offences Bill however few divisions there were that day,
    and "Second Reading" is not the Third.
    """
    title, motion = row.get("title") or "", row.get("motion_text") or ""
    # Which bill is decided by the record's own bills where it can be (`annotate_bills`): a bill the question names
    # whole must be one the division is on, and a division on no bill at all is never the one a question naming a Bill
    # asks about (QA pass 8, H1: 55 of 98 real bill-and-day pairs were answered from another bill's division, or an
    # order's).
    asked_bills, own_bills = row.get("bills") or (frozenset(), frozenset())
    names_a_bill = bool(_BILL_WORD.search(f"{title} {motion}"))
    if asked_bills and own_bills:
        if not asked_bills & own_bills:
            return True
    elif asked_bills and not names_a_bill:
        return True
    else:
        # A name the record does not hold whole is compared by its words, ALL of them in the division's bill: a short
        # name ("the Rwanda Bill") is the bill's own words, and a word in common is not the same bill ("Victims and
        # Courts" is not "Courts and Tribunals": MR-11).
        asked = _bill_names(_undated(question))
        own = _bill_names(title) + _bill_names(motion)
        if asked and not own:
            return True
        # The division's own printed bill name, found whole in the question, is the bill asked about, whatever the words
        # read before it ("Did the Lords pass Bus Services (No. 2) Bill [HL]"; a title printing "Bill" twice).
        printed = re.match(r"\s*(.+?\bBill)\b", title, re.I)
        named_whole = bool(printed) and f" {' '.join(_name_words(printed.group(1)))} " in \
            f" {' '.join(_name_words(_undated(question)))} "  # fmt: skip
        if not named_whole and any(not any(name <= bill for bill in own) for name in asked):
            return True
    # A stage the question names must be one the division shows (its title, or the Lords' motion, "read a second time").
    asked = _stages(question)
    shown = _stages(f"{title} {motion}") | ({linked} if (linked := linked_stage(row.get("stage_name"))) else set())
    if asked and not (asked & shown):
        return True
    # A reference of a kind both name, with no number in common, is a CONTRADICTION, not a gap: the question said New
    # Clause 3 and this title says New Clause 1.
    title_marks, asked_marks = _marks(title), _marks(_undated(question))
    return any(not ({_base(r) for r in title_marks[k]} & {_base(r) for r in asked_marks[k]})
               for k in set(title_marks) & set(asked_marks))  # fmt: skip


# Words a question uses to ask about a division rather than to name its business.
_ASKING = frozenset("""
result results outcome outcomes ayes noes aye noe tally votes voting majority government win won lose lost defeated pass passed
fail failed members mps peers split counted tellers recorded there they them against favour favor support supported
oppose opposed many much take took place held happened went into lobby lobbies margin figures numbers each side sides
both contents content not-content not-contents whole tell explain about with from vote's way back backed rebel
""".split())


def _business(question: str) -> set[str]:
    """The terms a question uses to name the business it asks about: no date, no asking words, no voter."""
    text = _undated(question)
    subject = vote_subject(text)
    if subject:
        text = text.replace(subject, " ")
    return _terms(text) - _ASKING


def names_business(question: str) -> bool:
    """Whether the question names any business at all, beyond its House, its date, its voter and its asking words.
    """
    return bool(_business(question))


def fits_only_division(question: str, row: dict) -> bool:
    """Whether the only division on the asked day can be the one asked about: nothing in the question contradicts it,
    and the business the question names, if any, is mostly in its title or motion (QA pass 3, finding 1). "The only
    division that day" used to be taken whatever the question named; 27 House-days outside the window hold one.
    """
    if contradicts(question, row):
        return False
    business = _business(question)
    if not business:
        return True  # the date and House alone chose it
    held = _terms(f"{row.get('title') or ''} {row.get('motion_text') or ''}")
    return len(business & held) / len(business) >= 0.5


_KIND_WORD = re.compile(r"\b(new\s+clauses?|new\s+schedules?|amendments?|clauses?|schedules?)\b", re.I)


def _kinds_unnumbered(question: str) -> set[str]:
    """The kinds a question names without a reference ("the amendment", "Labour's new clause")."""
    text = _NAMED_AMENDMENT.sub(" ", _undated(question))  # "(Amendment) Regulations" names the instrument
    numbered = [span for kind, _, span in _references(text) if kind != "location"]
    return {_kind_name(m.group(1)) for m in _KIND_WORD.finditer(text)
            if not any(a <= m.start() < b for a, b in numbered)
            # "passed without amendment" says how a bill passed, not which division it was
            and not text[max(0, m.start() - 8):m.start()].lower().endswith("without ")}  # fmt: skip


def _of_kind(row: dict, kind: str) -> bool:
    text = f"{row.get('title') or ''} {row.get('motion_text') or ''}"
    return bool(re.search(r"\b" + kind.replace(" ", r"\s+") + r"s?\b", text, re.I))


def _narrow(question: str, rows: list[dict]) -> tuple[list[dict], str]:
    """The divisions the question could mean, before any is scored: none it contradicts (a bill, stage or reference it
    names that the division's is not), and, when it names a kind without its number ("the amendment"), only those of
    that kind. Returns the rows left and, when none are, why.
    """
    kept = [r for r in rows if not contradicts(question, r)]
    if not kept:
        return [], "every division that day is on a bill, stage or reference other than the one the question names"
    for kind in sorted(_kinds_unnumbered(question)):
        of_kind = [r for r in kept if _of_kind(r, kind)]
        if not of_kind:
            return [], f"no division that day is on a {kind}"
        kept = of_kind
    return kept, ""


def _covered(question: str, row: dict) -> float:
    """The share of the business the question names that the division's title and motion hold."""
    business = _business(question)
    if not business:
        return 1.0
    return len(business & _terms(f"{row.get('title') or ''} {row.get('motion_text') or ''}")) / len(business)


def _equally_covered(question: str, best: dict, rows: list[dict]) -> list[dict]:
    """The other divisions that hold everything the question names as fully as `best` does: the question does not
    choose between them, however much shorter one title is. Empty when the question names no business, or names
    every word of `best`'s title, which is naming it.
    """
    business = _business(question)
    if not business or _terms(best.get("title") or "") <= business:
        return []  # no business named, or `best` named in full
    mine = _covered(question, best)
    return [r for r in rows if r is not best and r.get("id") != best.get("id") and _covered(question, r) >= mine]


def title_score(question: str, title: str | None, motion: str | None = None, *, row: dict | None = None) -> float:
    """How well a division title matches the question: the share of the title's own terms that the question uses."""
    title_terms = _terms(title or "")
    if not title_terms:
        return 0.0
    # the division's own row where the caller has it, so the bills it is on are judged by identity as the filter judged
    # them (`annotate_bills`), never by words alone
    if contradicts(question, row if row is not None else {"title": title, "motion_text": motion}):
        return 0.0  # a bill, stage or reference the title contradicts (QA pass 3)
    # The question's own date chose the day already; its words are not title evidence: "on 20 May 2020" matched
    # "Business of the House (20 May)" and one division was given as the day's four.
    question = _undated(question)
    asked = _terms(question)
    overlap = len(title_terms & asked) / len(title_terms)

    # A reference both name scores; one the title names and the question does not is a gap, and costs.
    title_marks = _marks(title or "")
    asked_marks = _marks(question)
    if title_marks:
        shared = set(title_marks) & set(asked_marks)
        if any(title_marks[k] & asked_marks[k] for k in shared):
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


def _only(when: Date, house: str | None, day_total: int) -> str:
    """The note for a division chosen as the only one left: "the only division that day" only when it was (QA pass 4,
    L1: the note said so when 5 or 9 divisions were held and the others had been ruled out).
    """
    where = f" in the {house}" if house else ""
    if day_total == 1:
        return f"the only division on {when}{where}"
    return f"the only one of the {day_total} divisions on {when}{where} left by what the question names"


# A division is on a calendar day when that is its sitting's date, or when it was held after midnight in the sitting of
# the day before: Hansard dates it by the sitting, and a question may date it by the clock (the Lords divided at 00:40
# on 19 March 2026 in the sitting of the 18th).
ON_DAY = ("(d.division_date = %s OR (d.division_date = %s::date - 1 AND d.held_at::date = %s "
          "AND d.held_at::time <> time '00:00'))")


_COUNTED = re.compile(r"\bhow\s+many\s+(?:separate\s+|different\s+)?(\w+)", re.I)
_DIVIDED = re.compile(r"\b(?:divid\w*|divisions?|vot(?:e|es|ed|ing))\b", re.I)


def count_subject(question: str) -> str | None:
    """What a "how many" question counts: "divisions" or "debates", or None when it names neither (the count then keeps
    the shape its span gives it). Read from the noun after "how many", and for "how many times" from the verb ("did
    the Lords divide", "did the House vote").
    """
    found = _COUNTED.search(question or "")
    if not found:
        return None
    noun = found.group(1).lower()
    # "how many votes" is the members' votes (a tally), never a count of divisions
    if noun.startswith("division"):
        return "divisions"
    if noun.startswith("debate"):
        return "debates"
    if noun == "times" and _DIVIDED.search(question[found.end():]):
        return "divisions"
    return None


def numbered(rows: list[dict], number: int, when: Date) -> list[dict]:
    """The rows numbered `number`, the asked day's own sitting's first: numbers restart each sitting, so on a day that
    also holds the sitting before's divisions after midnight, "division 1 on 19 March" is the 19th's division 1. A
    late division is found by its number only when the day's own sitting has none of it.
    """
    exact = [r for r in rows if r["number"] == number]
    own = [r for r in exact if r.get("division_date") == when]
    return own or exact


async def division_id(conn, question: str, *, house: str | None = None, when: Date | None = None,
                      number: int | None = None) -> tuple[int | None, str]:  # fmt: skip
    """The division the question means, or None and a reason."""
    if when is None:
        return None, "no date in the question, and a division cannot be identified by title alone"
    rows = await _all(conn, """
        SELECT d.id, d.house, d.number, d.title, d.motion_text, d.division_date, bs.stage_name, bs.bill_id
        FROM division d LEFT JOIN bill_stage bs ON bs.id = d.bill_stage_id
        WHERE """ + ON_DAY + """ AND (%s::text IS NULL OR d.house = %s)
        ORDER BY d.number NULLS LAST, d.id
    """, (when, when, when, house, house))
    rows = await with_bills(conn, question, rows)
    if not rows:
        return None, f"no division is recorded on {when}" + (f" in the {house}" if house else "")
    day_total = len(rows)
    held = _carrying(question, rows)
    if held is not None:
        if not held:
            return None, f"no division on {when} carries the clause, schedule or amendment the question names"
        rows = held

    if number is not None:
        exact = numbered(rows, number, when)
        if len(exact) == 1:
            if contradicts(question, exact[0]):
                return None, f"division {number} on {when} is {exact[0]['title']!r}, which the question contradicts"
            return exact[0]["id"], f"division {number} on {when}"
    if len(rows) > 1:
        rows, why = _narrow(question, rows)
        if not rows:
            return None, f"{why} ({when})"
    if len(rows) == 1:
        if not fits_only_division(question, rows[0]):
            return None, (f"{_only(when, house, day_total)} is {rows[0]['title']!r}, which is not the business the "
                          f"question names")  # fmt: skip
        return rows[0]["id"], _only(when, house, day_total)

    scored = sorted(((title_score(question, r["title"], r.get("motion_text"), row=r), r) for r in rows),
                    key=lambda pair: -pair[0])
    best, runner = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
    if best[0] < MIN_SCORE:
        return None, (f"{len(rows)} divisions on {when} and the question matches none of them well "
                      f"(best {best[0]:.2f})")  # fmt: skip
    if best[0] - runner[0] < MIN_MARGIN:
        return None, (f"{len(rows)} divisions on {when} and two match equally well "
                      f"({best[0]:.2f} vs {runner[0]:.2f}); the question does not say which")  # fmt: skip
    also = _equally_covered(question, best[1], rows)
    if also:
        return None, (f"{len(also) + 1} divisions on {when} hold everything the question names "
                      f"({best[1]['title']!r} among them); the question does not say which")  # fmt: skip
    return best[1]["id"], f"matched {best[1]['title']!r} on {when} (score {best[0]:.2f})"


# Where a vote question names the member it asks about: the subject of its verb.
_SUBJECT = r"(?P<s>(?:(?!\b(?:did|how|was|were|which|and\s+how)\b)[^,;?])+?)"
# The shapes, most specific first: "how did X vote" names its voter more surely than "did X back".
_VOTE_SUBJECTS = tuple(re.compile(pattern.replace("SUBJECT", _SUBJECT), re.I) for pattern in (
    r"\b(?:how|which\s+way)\s+did\s+SUBJECT\s+vote\b",
    r"\bwhich\s+lobby\s+did\s+SUBJECT\s+(?:go|walk|vote|enter)\b",
    r"\b(?:was|were)\s+SUBJECT\s+an?\s+(?:aye|no|content|not[\s-]content)\b",
    r"\bdid\s+SUBJECT\s+(?:vote|support|oppose|back|rebel)\b",
))  # fmt: skip


# Where a party question names its member (QA pass 5, H5): "which party was SUBJECT in", "which party did SUBJECT sit
# for", "what was SUBJECT's party", "which party was SUBJECT a member of".
_PARTY_SUBJECTS = tuple(re.compile(p, re.I) for p in (
    r"\bwhich\s+party\s+(?:was|is|were)\s+(?P<s>.+?)\s+(?:a\s+member\s+of|in|with|for)\b",
    r"\bwhich\s+party\s+did\s+(?P<s>.+?)\s+(?:sit|belong|represent|stand)\b",
    r"\bwhat\s+(?:was|is)\s+(?P<s>.+?)['\u2019]s?\s+party\b",
    r"\bwhat\s+party\s+(?:was|is|did)\s+(?P<s>.+?)\s+(?:a\s+member\s+of|in|with|for|sit|belong|represent)\b",
))  # fmt: skip


def party_subject(question: str) -> str | None:
    """The words a party question uses for its member, or None when it has no such shape."""
    text = " ".join(_undated(question or "").split())
    for pattern in _PARTY_SUBJECTS:
        m = pattern.search(text)
        if m:
            return m.group("s").strip(" ,") or None
    return None


def vote_subject(question: str) -> str | None:
    """The words a vote question uses for the member it asks about, or None when it has no such shape."""
    text = " ".join((question or "").split())
    for pattern in _VOTE_SUBJECTS:
        m = pattern.search(text)
        if m:
            return m.group("s").strip() or None
    return None


def without_titles(question: str, titles: list[str | None]) -> str:
    """The question with each title removed wherever it is quoted, ignoring case and spacing."""
    text = " ".join((question or "").split())
    for title in sorted((t for t in titles if t), key=len, reverse=True):
        pattern = r"\s+".join(re.escape(word) for word in title.split())
        text = re.sub(pattern, " ", text, flags=re.I)
    return " ".join(text.split())


# What follows a peer's name when it is a longer peer's: "Lord Moynihan" inside "Lord Moynihan of Chelsea".
_DESIGNATION = re.compile(r"\s+of\s+\w", re.I)
# Every dash and curly apostrophe, one character for one (so spans stay aligned): "Lord Booth–Smith" is not Lord Booth,
# and "Neil O’Brien" is Neil O'Brien.
_FOLD = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"} | {"\u2019": "'", "\u2018": "'"})
# "Keir Starmer's deputy" names someone else; "Keir Starmer's vote" names him.
_ATTRIBUTES = frozenset("""
vote votes voting ballot lobby lobbies party parties position positions record stance view views name
""".split())
_POSSESSIVE = re.compile(r"'s?\s+(\w+)")
# A member named as the object of a relation is not the member asked about: "the MP who succeeded Theresa May", "the
# deputy of Keir Starmer".
_RELATIONAL = re.compile(
    r"\b(?:succeeded|replaced|defeated|beat|unseated|ousted|preceded|followed|(?:successor|predecessor|deputy|replacement|"
    r"opponent|rival|challenger|spouse|wife|husband|son|daughter|pps|aide|adviser|advisor|parliamentary\s+private\s+"
    r"secretary)\s+(?:to|of|for))\s+(?:the\s+)?$", re.I)
# The words a vote question may put around a member's name and still be about that member: titles, address and party.
_ADDRESS = frozenset("""
the rt right hon hon. honourable member mp mr mrs ms miss mx dr sir dame lord lady baroness baron viscount viscountess
earl countess marquess marchioness duke duchess prof professor kc qc noble gallant learned friend my
conservative labour liberal democrat democrats lib dem snp scottish national green reform uk independent crossbench
crossbencher plaid cymru dup sdlp alliance sinn fein féin co-op workers ulster unionist
""".split())


def _name_span(text: str, forms: tuple[str | None, ...]) -> tuple[int, int] | None:
    """Where the longest of a member's names stands in the text as whole words, or None."""
    return _name_spans(text, forms)[0]


def _describes(text: str, forms: tuple[str | None, ...]) -> bool:
    """Whether the text names this member only to describe someone else ("Keir Starmer's deputy", "the MP who succeeded
    Theresa May"): a reference to another person, so a question that also names a member is not about one member
    alone (QA pass 4: "Neil Other's whip, and how did Alan Mak" must not be answered for Alan Mak).
    """
    span, described = _name_spans(text, forms)
    return span is None and described


def _name_spans(text: str, forms: tuple[str | None, ...]) -> tuple[tuple[int, int] | None, bool]:
    """(the longest whole-word span of a member's names, whether a match was set aside as describing someone else).
    """
    lowered = text.translate(_FOLD).lower()
    best: tuple[int, int] | None = None
    described = False
    for f in forms:
        f = f.translate(_FOLD) if f else f
        if not f or f.lower() not in lowered:
            continue
        # a hyphen continues a name: "Lord Booth" is not inside "Lord Booth-Smith"; "Alan Mak's vote" still holds "Alan
        # Mak", and "Alan Mak's deputy" does not (QA pass 4)
        for m in re.finditer(r"(?<![\w-])" + re.escape(f.lower()) + r"(?!\w|-\w)", lowered):
            if " of " not in f.lower() and _DESIGNATION.match(lowered, m.end()):
                continue
            owned = _POSSESSIVE.match(lowered, m.end())
            if (owned and owned.group(1) not in _ATTRIBUTES) or _RELATIONAL.search(lowered[max(0, m.start() - 60):m.start()]):
                described = True
                continue
            if best is None or m.end() - m.start() > best[1] - best[0]:
                best = m.span()
            break
    return best, described


def _named(text: str, forms: tuple[str | None, ...]) -> int:
    """The length of the longest of a member's names the text holds as whole words, or 0."""
    span = _name_span(text, forms)
    return span[1] - span[0] if span else 0


def _people(spans: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """The members named in their own right: a match inside a longer member's match is part of that name ("Ian Paisley"
    inside "Ian Paisley Jnr").
    """
    return {mid: (a, b) for mid, (a, b) in spans.items()
            if not any(o != mid and oa <= a and b <= ob and ob - oa > b - a for o, (oa, ob) in spans.items())}


def _apart(spans: dict[int, tuple[int, int]]) -> bool:
    """Whether the text names two members at different places (two people), not one name two members share."""
    places = set(spans.values())
    return any(a2 >= b1 or a1 >= b2 for (a1, b1) in places for (a2, b2) in places if (a1, b1) != (a2, b2))


def _forms(row: dict) -> tuple[str | None, ...]:
    return (row.get("printed"), row.get("name"), row.get("name_bare"), row.get("name_list_as"), row.get("name_full_title"))


def _distinct_people(text: str, rows: list[dict]) -> set[int]:
    """The members of `rows` the text names in their own right, as whole words."""
    spans = {r["mnis_id"]: span for r in rows if (span := _name_span(text, _forms(r)))}
    return set(_people(spans))


def _only_address(text: str, span: tuple[int, int]) -> bool:
    """Whether everything in `text` outside the name at `span` is a title, an address word or a party."""
    rest = (text[: span[0]] + " " + text[span[1]:]).translate(_FOLD).lower()
    return all(w.strip(".,'") in _ADDRESS or not w.strip(".,'") for w in re.findall(r"[\w'.,-]+", rest))


def _not_only_the_member(text: str) -> str:
    return (f"the question's voter ({text!r}) says more than a member's name, so it may be someone else; ask about "
            f"the member by name")


def _pick_listed(rows: list[dict], text: str, *, whole: bool = False) -> tuple[int | None, str | None]:
    """The one member of a division's list the text names, a refusal when it names several equally, or (None, None)
    when it names none of them. With `whole` (the text is a vote's subject), the name must be all the subject says
    beyond titles, address and party: "the MP who succeeded Theresa May" is not Theresa May.
    """
    spans: dict[int, tuple[int, int]] = {}
    for r in rows:
        span = _name_span(text, _forms(r))
        if span and (r["mnis_id"] not in spans or span[1] - span[0] > spans[r["mnis_id"]][1] - spans[r["mnis_id"]][0]):
            spans[r["mnis_id"]] = span
    if not spans:
        return None, None
    people = _people(spans)
    others = [r for r in rows if r["mnis_id"] not in people and _describes(text, _forms(r))]
    if others:
        return None, ("the question names more than one member (one of them only to describe someone else); ask "
                      "about one member by name")  # fmt: skip
    if _apart(people):
        # "Did Keir Starmer and Alan Mak vote the same way": two voters, and answering for one is answering a question
        # that was not asked (QA pass 3)
        names = sorted({text[a:b] for a, b in people.values()})
        return None, f"the question names more than one member ({' and '.join(names)}); ask about one at a time"
    scored = {mid: b - a for mid, (a, b) in people.items()}
    best = max(scored.values())
    top = [mid for mid, length in scored.items() if length == best]
    if len(top) > 1:
        return None, f"{len(top)} members of the division's list match that name; the question does not say which"
    if whole and not _only_address(text, people[top[0]]):
        return None, _not_only_the_member(text)
    printed = next((r.get("printed") or r.get("name") for r in rows if r["mnis_id"] == top[0]), "")
    return top[0], f"matched {printed} in the division's own list"


# A peer as people name one: the title and name before the territorial designation ("Lord Ahmad" for "Lord Ahmad of
# Wimbledon").
_PEER_SHORT = re.compile(r"^((?:The\s+)?(?:Lord|Baroness|Lady|Viscount|Viscountess|Earl|Countess|Marquess|Duke)\s+"
                         r"(?!Bishop\b)[\w'’.-]+(?:\s+[\w'’.-]+)?)\s+of\s+\S", re.I)


def peer_short(name: str | None) -> str | None:
    m = _PEER_SHORT.match(name or "")
    return m.group(1) if m else None


async def _short_peer(conn, rows: list[dict], text: str, *, whole: bool = False) -> tuple[int | None, str | None]:
    """A listed peer the text names by short form only. Accepted only when exactly one member of the whole record
    carries that short form: one in the list is not enough, or "Lord Smith" could be answered for the wrong Lord
    Smith.
    """
    hits: dict[int, str] = {}
    for r in rows:
        short = peer_short(r.get("printed") or r.get("name"))
        if short and _named(text, (short,)):
            hits[r["mnis_id"]] = short
    if not hits:
        return None, None
    shorts = set(hits.values())
    if len(hits) > 1:
        return None, f"{len(hits)} members of the division's list are called {' or '.join(sorted(shorts))}"
    (mid, short), = hits.items()
    if whole and not _only_address(text, _name_span(text, (short,)) or (0, 0)):
        return None, _not_only_the_member(text)
    carriers = await _all(conn, "SELECT mnis_id FROM member_id_map WHERE lower(regexp_replace(name, %s, '', 'i')) = "
                                "lower(%s)", (r"\s+of\s+.*$", short))  # fmt: skip
    others = {c["mnis_id"] for c in carriers} - {mid}
    if others:
        return None, f"{len(others) + 1} members of the record are called {short}; the question does not say which"
    return mid, f"matched {short} in the division's own list (the only member so called)"


async def mnis_id(conn, question: str, *, division_ids: tuple[int, ...] | list[int] = (),
                  on_date: Date | None = None) -> tuple[int | None, str]:  # fmt: skip
    """The member the question names, or None."""
    listed: list[dict] = []
    stripped = " ".join((question or "").split())
    if division_ids:
        ids = [int(i) for i in division_ids]
        titles = [r["title"] for r in await _all(conn, "SELECT title FROM division WHERE id = ANY(%s)", (ids,))]
        stripped = without_titles(question, titles)
        listed = await _all(conn, """
            SELECT DISTINCT v.mnis_id, v.member_name AS printed, m.name, m.name_list_as, m.name_full_title,
                   regexp_replace(m.name, %s, '', 'i') AS name_bare
              FROM division_vote v LEFT JOIN member_id_map m ON m.mnis_id = v.mnis_id
             WHERE v.division_id = ANY(%s)
        """, (_HONORIFIC, ids))
    subject = vote_subject(stripped) or party_subject(stripped)
    # The subject, when the question has one, is the only place the member is read from: falling back to the whole
    # question let a listed name elsewhere (a mover, "on Neil Parish's new clause") answer for an absent member.
    texts = [subject] if subject else [stripped]
    whole = bool(subject)
    for text in texts if listed else ():
        found, why = _pick_listed(listed, text, whole=whole)
        if why:
            return found, why
        found, why = await _short_peer(conn, listed, text, whole=whole)
        if why:
            return found, why
    refusal = "no member named in the question matches the member record"
    day = on_date or await _one_day(conn, division_ids)
    for text in texts:
        found, why = await _member_record(conn, text, whole=whole, day=day)
        if found is not None or "members match" in why:
            return found, why
        refusal = why
    return None, refusal


async def _member_record(conn, question: str, *, whole: bool = False, day: Date | None = None) -> tuple[int | None, str]:
    """The member `member_id_map` holds for a name the text spells out: the longest match; a tie is broken by the one
    member seated on `day`, else refused.
    """
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
    # The SQL finds names as substrings; a candidate counts only where its name stands in the text as whole words, under
    # the same rules as the division's list: a hyphen continues a name, and a name followed by "of <place>" is part of a
    # longer peer's.
    spans = {r["mnis_id"]: span for r in candidates if (span := _name_span(question, _forms(r)))}
    described = [r for r in candidates if r["mnis_id"] not in spans and _describes(question, _forms(r))]
    candidates = [r for r in candidates if r["mnis_id"] in spans]
    if not candidates:
        return None, "no member named in the question matches the member record"
    if described:
        # a member named to describe someone else is a second person (QA pass 4)
        return None, ("the question names more than one member (one of them only to describe someone else); ask "
                      "about one member by name")  # fmt: skip

    # Longest matched name wins: "Baroness Featherstone" over "Featherstone", and it prevents a short name that is a
    # substring of a longer one ("Ali" inside "Alison") from taking the match.
    def matched_length(row: dict) -> int:
        """The length of the name as the QUESTION spells it, which is what the tie-break is really about."""
        a, b = spans[row["mnis_id"]]
        return b - a

    people = _people(spans)
    if _apart(people):
        names = sorted({question[a:b] for a, b in people.values()})
        return None, f"the question names more than one member ({' and '.join(names)}); ask about one at a time"
    best = max(candidates, key=matched_length)
    same = [r for r in candidates if matched_length(r) == matched_length(best) and r["mnis_id"] != best["mnis_id"]]
    same = [best, *same]
    if len(same) > 1:
        seated = [r for r in same if day is not None and await _seated(conn, r["mnis_id"], day)]
        if len(seated) == 1:
            return seated[0]["mnis_id"], (f"matched {seated[0]['name']}: of {len(same)} members of that name, the only "
                                          f"one seated on {day}")  # fmt: skip
        return None, f"{len(same)} members match that name; the question does not say which"
    if whole and not _only_address(question, spans[best["mnis_id"]]):
        return None, _not_only_the_member(question)
    return best["mnis_id"], f"matched {best['name']}"


async def _one_day(conn, division_ids) -> Date | None:
    """The day the named divisions share, when they share one."""
    if not division_ids:
        return None
    days = {r["division_date"] for r in await _all(conn, "SELECT division_date FROM division WHERE id = ANY(%s)",
                                                    ([int(i) for i in division_ids],))}  # fmt: skip
    return days.pop() if len(days) == 1 else None


async def _seated(conn, mnis: int, day: Date) -> bool:
    """Whether the Members API's dated seats put the member in either House on `day` (`answer.facts.house_sat`)."""
    from answer import facts as F

    return await F.house_sat(conn, mnis, day) is not None


# What may stand before an office the question names in full: a question's opener ("who was", "who held the office of",
# "serving as"), then at most "the".
_OFFICE_OPENER = re.compile(
    r"(?:^|\b(?:who|whom|was|were|is|held|holds|hold|as|occupied|occupies|occupy|filled|name|identify))\s*"
    r"(?:(?:the\s+)?(?:office|post|position|role|job)\s+of\s+)?(?:the\s+)?(?:(?:current|present)\s+)?$", re.I)


def _office_named_in_full(question: str, office: str) -> bool:
    """Whether the question names `office` itself: nothing of its own qualifies the match ("'s deputy", "(Home Office)"
    after it; a party, a relation or "former" before it).
    """
    name = re.sub(r"^the\s+", "", " ".join((office or "").split()), flags=re.I)
    if not name:
        return False
    text = " ".join((question or "").split()).translate(_FOLD)
    for m in re.finditer(r"(?<![\w-])" + re.escape(name.translate(_FOLD)) + r"(?![\w-])", text, re.I):
        # Nothing of the office's phrase may follow it before the date or the end: "the Prime Minister, First Lord of
        # the Treasury, … and Minister for the Union's deputy" holds "Prime Minister" inside a longer office, ended by a
        # possessive; "Speaker of the House of Commons" is not "Speaker".
        tail = re.split(r"\s+(?:on|in|at|during|from|between|as\s+of)\s+|[?.!]", text[m.end():], maxsplit=1)[0]
        rest = tail.strip(" ,;:")
        if rest and not PRESENT.PRESENT_MARKER.fullmatch(rest):  # "… now?" says when
            continue
        if _OFFICE_OPENER.search(text[: m.start()]) and _office_residue_is_filler(text, m.span()):
            return True
    return False


# What an office question may hold besides its opener, its office and its date (QA pass 5, H4: "in the SNP", "at the
# Liberal Democrats", "for the Liberal Democrats" after the date, "In the SNP," before the opener).
_OFFICE_FILLER = frozenset("""
who whom was were is held holds hold the office post position role job of on serving served as occupied occupies
occupy filled name identify which minister person member mp peer tell me please can you a an
fills serves current currently now nowadays presently present at today these days
""".split())


def _office_residue_is_filler(text: str, span: tuple[int, int]) -> bool:
    from answer import router as RO

    rest = _undated(text[: span[0]] + " " + text[span[1]:])
    words = [w.lower() for w in re.findall(r"[A-Za-z][\w'-]*", rest)]
    return all(w in _OFFICE_FILLER or w in RO.MONTHS for w in words)


@dataclass(frozen=True)
class OfficeCatalogue:
    """The record's offices, as every question on a connection matches them (S06)."""

    ranked: tuple[dict, ...]
    printed: tuple[str, ...]


_OFFICES_BY_CONN: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def offices(conn) -> OfficeCatalogue:
    """The catalogue, read once per connection, as `with_bills` reads the bills. A scan of `member_office` with two
    regular expressions had run for every question, twice for a qualified office (`office_key`, then
    `office_relations.qualified`): about 60 ms each, nine tenths of such an answer (S06's timing).
    """
    found = _OFFICES_BY_CONN.get(conn)
    if found is None:
        rows = await _all(conn, """
            SELECT office_key, office, printed, length(office) AS office_length,
                   regexp_replace(office, '^the\\s+', '', 'i') AS bare
              FROM (SELECT DISTINCT office_key, office AS printed,
                           regexp_replace(btrim(office), '\\s+', ' ', 'g') AS office
                      FROM member_office WHERE office IS NOT NULL) o
             ORDER BY length(office) DESC, office, office_key, printed
        """)
        ranked: dict[tuple[str, str], dict] = {}
        for row in rows:
            ranked.setdefault((row["office_key"], row["office"]), {
                "office_key": row["office_key"], "office": row["office"], "office_length": row["office_length"],
                "folded": row["bare"].lower()})  # fmt: skip
        found = OfficeCatalogue(tuple(ranked.values()), tuple(dict.fromkeys(row["printed"] for row in rows)))
        _OFFICES_BY_CONN[conn] = found
    return found


async def office_candidates(conn, question: str) -> list[dict]:
    """Every office the record holds whose name (without a leading "the") the question contains, longest first, at most
    twenty.
    """
    text = " ".join((question or "").split()).lower()
    named = [r for r in (await offices(conn)).ranked if r["folded"] and r["folded"] in text][:20]
    return [{"office_key": r["office_key"], "office": r["office"], "office_length": r["office_length"]} for r in named]


async def office_key(conn, question: str) -> tuple[str | None, str]:
    """The office the question names, as `member_office.office_key`."""
    rows = await office_candidates(conn, question)
    if not rows:
        return None, "no office named in the question matches one in the record"
    named = [r for r in rows if _office_named_in_full(question, r["office"])]
    if not named:
        return None, (f"the question qualifies {rows[0]['office']!r} (a party, a relation or another office around it), "
                      f"so it names a different office")  # fmt: skip
    return named[0]["office_key"], f"matched {named[0]['office']!r}"


_HL = re.compile(r"\s*\[(?:HL|Lords)\]\s*$", re.I)


def _bill_named_in_full(question: str, title: str) -> bool:
    """Whether the question names this bill whole: its title (without " [HL]") stands in the question with no word of a
    longer bill name before it. "the Social Housing Bill" is not the Housing Bill, nor "the Health and Care Bill"
    the Care Bill.
    """
    name = _HL.sub("", title or "").strip()
    if not name:
        return False
    text = (question or "").translate(_FOLD)
    for m in re.finditer(r"(?<![\w-])" + re.escape(name.translate(_FOLD)) + r"(?![\w-])", text, re.I):
        words = _TOKEN.findall(text[max(0, m.start() - 60):m.start()])
        prev = words[-1].lower().strip("(),.[]'") if words else ""
        if not prev or (prev in _NOT_A_BILL_NAME and prev not in ("and", "of")):
            return True
    return False


async def bill_candidates(conn, question: str) -> list[dict]:
    """Every bill whose title the question contains, longest title first."""
    rows = await _all(conn, """
        SELECT id, title FROM bill
        WHERE title IS NOT NULL AND %s ILIKE '%%' || regexp_replace(title, '\\s*\\[(HL|Lords)\\]\\s*$', '', 'i') || '%%'
        ORDER BY length(title) DESC, id LIMIT 20
    """, (question,))
    return [r for r in rows if _bill_named_in_full(question, r["title"])][:8]


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
    from answer import router as RO

    # Dates out first, and never a number a month follows: "the Lords division 5 November 2025" is division 2, not 5.
    text = _undated(question)
    explicit = re.search(r"\bdivision\s*(?:no\.?|number)?\s*(\d{1,4})\b(?!(?:st|nd|rd|th)?\s+(?:" + RO._MONTH_ALT
                         + r")\b)", text, re.I)  # fmt: skip
    if explicit:
        return int(explicit.group(1))
    ordinal = re.search(r"\b(" + "|".join(_ORDINAL) + r")\s+division\b", text, re.I)
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
# The kinds of stage Parliament's procedure has.
_STAGE_KINDS = frozenset({"reading", "committee", "report", "consideration", "assent", "examiners", "resolution",
                          "commitment", "allocation", "carry-over", "instruction", "message", "revive"})


def _stage_terms(text: str) -> set[str]:
    """Words that identify a stage, with ordinals normalised and only stage-question filler dropped."""
    out = {_ORDINALS.get(w, w) for w in _WORD.findall((text or "").lower())}
    return {w for w in out
            if w not in _STAGE_NOISE and (len(w) > 2 or w in _ORDINAL_FORMS)}  # fmt: skip


def _stage_score(asked: set[str], name: str) -> float:
    """How well a stage's name matches the question's stage terms: the share of the stage's terms the question uses,
    and 0 for another ordinal ("1st reading" when the 2nd is asked: half its terms matched, and it was answered, QA
    pass 4) or another kind of stage.
    """
    terms = _stage_terms(name)
    if not terms:
        return 0.0
    asked_ordinals, ordinals = asked & _ORDINAL_FORMS, terms & _ORDINAL_FORMS
    if ordinals and asked_ordinals and not (ordinals & asked_ordinals):
        return 0.0
    if (asked & _STAGE_KINDS) - (terms & _STAGE_KINDS):
        return 0.0
    # Normalised on the STAGE's terms, so a long question cannot out-score a short one, and a stage whose every word the
    # question uses wins over a longer stage that merely shares a word.
    return len(terms & asked) / len(terms)


def stage_question(question: str, titles: list[str | None]) -> str:
    """The question with the bill's own title taken out, so its words are never read as a stage's ("International
    Committee of the Red Cross (Status) Bill" named no Committee stage: QA pass 5, M2).
    """
    names = [t for t in titles if t] + [_HL.sub("", t).strip() for t in titles if t]
    return without_titles(question, names)


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

    scored = sorted(((_stage_score(asked, r["stage_name"]), r["stage_name"]) for r in rows),
                    key=lambda pair: (-pair[0], pair[1]))
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
            member, why = await mnis_id(conn, question, division_ids=(found,))
            notes.append(f"mnis_id: {why}")
            if member is None:
                return {}, notes
            params["mnis_id"] = member
        return params, notes

    if category == "party_at_date":
        # The date names the lists whose printed names apply: a member since renamed or raised to the peerage is found
        # by the name that day's division lists printed.
        that_day = [r["id"] for r in await _all(conn, "SELECT id FROM division WHERE division_date = %s",
                                                 (when,))] if when else []  # fmt: skip
        member, why = await mnis_id(conn, question, division_ids=that_day, on_date=when)
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
            found, why_stage = await stage_name(conn, stage_question(question, [candidate["title"]]),
                                                bill_id_=candidate["id"])  # fmt: skip
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
        subject = count_subject(question)
        if subject and "mnis_id" not in params:
            # what is counted is what the question counts; the span only says over when (QA pass 9, High 3: a day's
            # divisions were counted as its debates because a day had meant debates)
            params["counts"] = subject
        notes.append(f"count parameters: {sorted(params)}")
        return (params if len(params) > (1 if "house" in params else 0) else {}), notes

    return {}, [f"no resolver for category {category!r}"]

def describe(title: str | None, motion: str | None, limit: int = 110) -> str | None:
    """How a reader tells a division apart: its motion where that says more than its title (a Lords division's title is
    usually just the bill's name), cut at a word, else its title.
    """
    text = " ".join((motion or "").split())
    # compared with the title's spacing normalised too: "Amendment 39" in a title is the motion's "Amendment 39"
    if text and text.rstrip(".") != " ".join((title or "").split()).rstrip("."):
        return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return title


_MARK_NAMES = {"new clause": "New Clause", "new schedule": "New Schedule", "amendment": "Amendment",
               "clause": "Clause", "schedule": "Schedule"}


async def unheld_mark(conn, question: str, *, house: str | None, when: Date | None) -> str | None:
    """When the question names a clause, schedule or amendment that no division that day carries, in its title or its
    motion (the Lords print amendment numbers in the motion), the sentence saying so with the day's divisions; else
    None. That is MR-02's right answer: the record holds no such division, and these are the ones it holds.
    """
    asked = {k: v for k, v in _asked_marks(question).items() if k in _MARK_NAMES}
    if when is None or not asked:
        return None
    rows = await _all(conn, """
        SELECT d.number, d.title, d.motion_text, d.house, d.division_date FROM division d
         WHERE """ + ON_DAY + """ AND (%s::text IS NULL OR d.house = %s) ORDER BY d.number NULLS LAST, d.id
    """, (when, when, when, house, house))
    if not rows or not _absence_claimable(asked, rows):
        return None
    named = ", ".join(f"{_MARK_NAMES[k]} {n}" for k in sorted(asked) for n in sorted(asked[k]))
    where = f"the {house}" if house else "either House"
    def one(r: dict) -> str:
        # a division of the sitting before, held after midnight, is placed in its sitting
        sitting = ERAS.other_sitting(r["division_date"], when)
        placed = f" ({sitting})" if sitting else ""
        if r["number"] is None:
            return f"{describe(r['title'], r['motion_text'])}{placed}"
        return f"division {r['number']}{placed}: {describe(r['title'], r['motion_text'])}"

    listed = "; ".join(one(r) for r in rows[:12])
    more = f"; and {len(rows) - 12} more" if len(rows) > 12 else ""
    return (f"No division in {where} on {when.day} {when:%B %Y} is recorded on {named}. The divisions recorded that "
            f"day were: {listed}{more}.")


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
        SELECT d.id, d.house, d.number, d.title, d.motion_text, d.division_date, bs.stage_name, bs.bill_id
        FROM division d LEFT JOIN bill_stage bs ON bs.id = d.bill_stage_id
        WHERE """ + ON_DAY + """ AND (%s::text IS NULL OR d.house = %s)
        ORDER BY d.number NULLS LAST, d.id
    """, (when, when, when, house, house))
    rows = await with_bills(conn, question, rows)
    if not rows:
        return [], f"no division is recorded on {when}" + (f" in the {house}" if house else "")
    day_total = len(rows)
    held = _carrying(question, rows)
    if held is not None:
        if not held:
            return [], f"no division on {when} carries the clause, schedule or amendment the question names"
        rows = held
    if number is not None:
        exact = numbered(rows, number, when)
        if len(exact) == 1:
            if contradicts(question, exact[0]):
                return [], f"division {number} on {when} is {exact[0]['title']!r}, which the question contradicts"
            return [exact[0]["id"]], f"division {number} on {when}"
    if len(rows) > 1:
        rows, why = _narrow(question, rows)
        if not rows:
            return [], f"{why} ({when})"
    if len(rows) == 1:
        if not fits_only_division(question, rows[0]):
            return [], (f"{_only(when, house, day_total)} is {rows[0]['title']!r}, which is not the business the "
                        f"question names")  # fmt: skip
        return [rows[0]["id"]], _only(when, house, day_total)

    if not _business(question) and held is None and not _kinds_unnumbered(question):
        # "What was the result of the Commons division on 20 May 2020?" names no business, so it asks about every
        # division that day: each is answered, never one given as "the" division and never a search.
        return [r["id"] for r in rows], (f"the question names no business, so every one of the {len(rows)} divisions "
                                         f"on {when} is meant")
    scored = sorted(((title_score(question, r["title"], r.get("motion_text"), row=r), r) for r in rows),
                    key=lambda pair: -pair[0])
    best = scored[0][0]
    if best < MIN_SCORE:
        return [], f"{len(rows)} divisions on {when} and the question matches none of them well (best {best:.2f})"
    tied = [r["id"] for value, r in scored if best - value < MIN_MARGIN]
    tied += [r["id"] for r in _equally_covered(question, scored[0][1], rows) if r["id"] not in tied]
    return tied, (f"{len(tied)} division(s) on {when} match the question equally well (score {best:.2f})"
                  if len(tied) > 1 else f"matched {scored[0][1]['title']!r} on {when} (score {best:.2f})")  # fmt: skip

