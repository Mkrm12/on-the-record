"""The gold lint: a reference point must test what its question asks, and nothing more."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from goldset.schema import GoldItem

CODES = ("L0", "L1", "L2", "L3", "L4", "L5")


@dataclass(frozen=True)
class Finding:
    item: str
    code: str
    message: str


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_TITLE = r"(?:Lord|Baroness|Lady|Sir|Dame|Dr|Mr|Mrs|Ms|Miss|Viscount|Earl|Bishop)"
_NOT_A_NAME = r"(?!(?:The|A|An|This|That|These|Those|Both|It|Its|They|Their|He|She|His|Her|No|Yes|Neither)\b)"
# A name word: a capital, then a lower-case letter or an apostrophe, then letters of either case (McKee, MacDonald,
# O'Brien), so an acronym (UEFA, BBC) is never a name.
_WORD = r"[A-Z](?:[a-z]|['’][A-Z])[A-Za-z'’\-]*"
_NAME = (rf"(?:{_TITLE}\s+{_WORD}(?:\s+(?:of\s+)?{_WORD}){{0,3}}"
         rf"|{_NOT_A_NAME}{_WORD}(?:\s+{_WORD}){{1,3}})")
_VERBS = (r"(?:said|says|argued|argues|arguing|raised|raises|asked|made|pointed|responded|complained|claimed|opposed|"
          r"supported|called|warned|told|moved|replied|welcomed|criticised|noted|objected|acknowledged|spoke|did)")
# A short lead-in ("Yes:", "Only slightly:", "UEFA:") and "The record shows", then a name, then a verb or a comma.
_MEMBER = re.compile(rf"^(?:[A-Z][\w '’]{{0,24}}:\s*)?(?:The record shows\s+)?(?P<name>{_NAME})"
                     rf"(?=\s*[,.(]|\s+{_VERBS}\b)")
_FULL_DATE = re.compile(rf"\b\d{{1,2}} (?:{_MONTHS}) \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")
_ANY_DATE = re.compile(rf"\b(?:{_MONTHS})\b|\b(?:19|20)\d{{2}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b(?:last|this) (?:week|"
                       r"month|year|session)\b", re.I)

_WHO = re.compile(r"\b(?:who|whom|whose|whos|which (?:member|members|mp|mps|peer|peers|minister|lord|baroness|"
                  r"politician|person|speaker|benches|party))\b|\bname the\b", re.I)
_OFFICE = re.compile(r"\b(?:minister|secretary|chancellor|leader of the house|speaker|spokesperson|shadow|whip|"
                     r"attorney general|solicitor general)\b", re.I)
_DEFINITE = re.compile(r"\bthe (?:mp|peer|member|lord|baroness|backbencher)\b(?=\s+(?:who|that|defending|for|from)\b)",
                       re.I)
_BOTH_SIDES = re.compile(r"\b(?:both sides|for and against|competing views|each side|supporters and opponents|"
                         r"arguments for and against)\b", re.I)

_PASSAGE = re.compile(
    r"\b(?:the|this|that|a|either) (?:(?:cited|quoted|given|authoring) )?(?:passages?|quotations?|extracts?)\b"
    r"|\bquoted (?:speech|speeches|passage|passages|contributions?)\b|\b(?:speech(?:es)?|contributions?) quoted\b"
    r"|\bthe (?:Conservative|Labour|Liberal Democrat|Reform|SNP|Government|Opposition|backbench|quoted|cited) speech\b"
    r"|\bfrom them alone\b|\bany comparison\b|\bimpressionistic\b|\ba count question\b|\bnot a retrieval\b"
    r"|\bthe whole record\b", re.I)
_CITER = re.compile(
    r"\b(?:as )?(?:cited|claimed|stated|quoted|given|reported) by (?:the )?[A-Z]"
    r"|\baccording to (?:the )?(?:" + _TITLE + r"\s+)?[A-Z][a-z]+(?: [A-Z][a-z]+)+"
    r"|^[^:]{1,60}:\s*" + _NAME + r"(?: \([^)]*\))?,? (?:[^,.]{0,60}, )?(?:said|says|called|argued|asked|told|warned|"
    r"claimed)\b"
    r"|\((?:" + _TITLE + r"\s+)?[A-Z][a-z'’\-]+ [A-Z][a-z'’\-]+,[^)]*\)")


def member_named(text: str) -> str | None:
    """The member a reference point leads with, by name, or None."""
    m = _MEMBER.search(text.strip())
    return m.group("name") if m else None


def asks_who(question: str) -> bool:
    """Whether the question asks for a member: who, whose, which member or peer, name the..."""
    return _WHO.search(question) is not None


def _surname(name: str) -> str:
    words = [w for w in re.split(r"\s+", name) if w and w[0].isupper() and not re.fullmatch(_TITLE, w)]
    return words[-1] if words else name


def _question_allows(question: str, name: str | None) -> bool:
    """Whether the question itself makes naming this member right: it asks who, names the member or the office, or
    points at one person.
    """
    if _WHO.search(question) or _OFFICE.search(question) or _DEFINITE.search(question):
        return True
    return bool(name) and re.search(rf"\b{re.escape(_surname(name))}\b", question, re.I) is not None


def lint(item: GoldItem) -> list[Finding]:
    """Every rule this item breaks, in CODES order."""
    found: list[Finding] = []

    def flag(code: str, message: str) -> None:
        found.append(Finding(item.id, code, message))

    if not item.breadth:
        flag("L0", "no breadth: say whether the question is narrow (one fact, member, event or 'who') or broad")
        return found
    vital = [n for n in item.nuggets if n.vital]
    if item.breadth == "broad":
        for n in vital:
            name = member_named(n.text)
            if (name or n.kind == "example") and not _question_allows(item.question, name):
                who = f"names {name}" if name else "is an example"
                flag("L1", f"a vital point {who} on a broad question that does not ask for a member: make the point "
                           f"vital and the member an optional example ({n.text[:60]!r})")
            elif _FULL_DATE.search(n.text) and not _ANY_DATE.search(item.question):
                flag("L1", f"a vital point demands one dated instance on a broad, undated question ({n.text[:60]!r})")
    for n in vital:
        if _PASSAGE.search(n.text):
            flag("L2", f"a vital point is worded about the passage or the method, not the record ({n.text[:60]!r})")
    for n in vital:
        if n.kind == "example":
            continue
        m = _CITER.search(n.text.strip())
        if m and not _question_allows(item.question, member_named(n.text) or _citer_name(m.group(0))):
            flag("L3", f"a vital fact is bundled with a citer the question did not ask about: split the fact (vital) "
                       f"from the citer (an optional example) ({n.text[:60]!r})")
    if _BOTH_SIDES.search(item.question) and len({n.side for n in vital if n.side}) < 2:
        flag("L4", "the question asks for both sides, but its vital points do not cover two sides")
    if item.breadth == "broad" and item.min_points is None:
        flag("L5", "a broad item needs min_points: how many of its vital points a right answer must state")
    return found


def _citer_name(span: str) -> str | None:
    m = re.search(rf"(?:by|to) (?:the )?(?P<n>(?:{_TITLE}\s+)?[A-Z][a-z]+(?: [A-Z][a-z]+)+)", span)
    return m.group("n") if m else None


def lint_all(items: Iterable[GoldItem]) -> list[Finding]:
    return [f for i in items for f in lint(i)]
