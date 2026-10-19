"""Turn a correct answer into a wrong one by changing exactly one fact the entry itself states."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from app import sealed as _sealed  # noqa: E402 - the one definition of the window (H-45, D-061)

KINDS = ("number", "date", "result", "negation")
JUDGE_SAFE = frozenset({"number", "date", "result"})
# The sealed W5 window, from `config/sealed.toml` (never a copy of the dates).
SEALED = (_sealed.window().start, _sealed.window().end)

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")  # fmt: skip
_LONG_DATE = re.compile(r"\b(\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})\b")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# A number as printed: digits with optional thousands commas and one decimal part.
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w,]|\.\d)")

# Result words and their opposites, longest first so "agreed to" wins over "agreed".
_RESULTS = (
    ("voted Not Content", "voted Content"), ("voted Content", "voted Not Content"),
    ("voted Aye", "voted No"), ("voted No", "voted Aye"),
    ("Ayes have it", "Noes have it"), ("Noes have it", "Ayes have it"),
    ("was agreed to", "was negatived"), ("was negatived", "was agreed to"),
    ("agreed to", "negatived"), ("was agreed", "was negatived"),
    ("carried", "defeated"), ("defeated", "carried"),
)  # fmt: skip
_NEGATIONS = (
    ("would not", "would"), ("will not", "will"), ("did not", "did"), ("does not", "does"), ("is not", "is"),
    ("opposed", "supported"), ("supported", "opposed"),
)  # fmt: skip


class NotPerturbable(ValueError):
    """The answer holds no fact of this kind that the entry also states; a seed built from it would prove nothing.
    """


@dataclass(frozen=True)
class Perturbation:
    kind: str
    text: str  # the answer with one fact changed
    original: str  # the span that was replaced
    replacement: str
    contradicts: str  # the reference point (or must-not-claim) the changed answer now contradicts
    judge_safe: bool


def _digits(s: str) -> str:
    return s.replace(",", "")


def _date_spans(text: str) -> list[tuple[int, int, date, str]]:
    spans = []
    for m in _LONG_DATE.finditer(text):
        try:
            spans.append((m.start(), m.end(), date(int(m.group(3)), _MONTHS.index(m.group(2)) + 1, int(m.group(1))),
                          "long"))
        except ValueError:
            continue
    for m in _ISO_DATE.finditer(text):
        try:
            spans.append((m.start(), m.end(), date(int(m.group(1)), int(m.group(2)), int(m.group(3))), "iso"))
        except ValueError:
            continue
    return spans


def _dates_in(texts: Sequence[str]) -> set[date]:
    return {d for t in texts for (_, _, d, _) in _date_spans(t)}


def _fmt_date(d: date, style: str) -> str:
    return d.isoformat() if style == "iso" else f"{d.day} {_MONTHS[d.month - 1]} {d.year}"


def _sealed(d: date) -> bool:
    return SEALED[0] <= d <= SEALED[1]


def _numbers_in(texts: Sequence[str]) -> set[str]:
    found = set()
    for t in texts:
        masked = _mask_dates(t)
        found |= {_digits(m.group(1)) + (m.group(2) or "") for m in _NUMBER.finditer(masked)}
    return found


def _mask_dates(text: str) -> str:
    """The text with every date's characters replaced, so a day or a year is never taken for a figure."""
    chars = list(text)
    for start, end, _, _ in _date_spans(text):
        chars[start:end] = "#" * (end - start)
    return "".join(chars)


def _new_number(old: str, taken: set[str], rng: random.Random) -> str:
    whole, _, frac = old.partition(".")
    value = int(_digits(whole))
    if frac:
        step = rng.choice((3, 4, 5, 6, 7)) / (10 ** len(frac))
        candidates = [round(float(_digits(old)) + s * step, len(frac)) for s in (1, -1, 2)]
        options = [f"{c:.{len(frac)}f}" for c in candidates if c > 0]
    else:
        delta = max(2, round(value * rng.choice((0.12, 0.17, 0.23))))
        options = [str(value + delta), str(value - delta) if value - delta > 0 else str(value + 2 * delta),
                   str(value + delta + 1)]
    for opt in options:
        if opt not in taken and opt != _digits(old):
            if "," in whole and "." not in opt:
                opt = f"{int(opt):,}"
            return opt
    raise NotPerturbable(f"no replacement for {old!r} avoids every figure the entry states")


def _choose(rng: random.Random, items: list):
    return items[rng.randrange(len(items))]


def perturbable_kinds(answer: str, reference_points: Sequence[str], must_not_claim: Sequence[str] = ()) -> list[str]:
    """The kinds `perturb` can apply to this answer, in KINDS order."""
    kinds = []
    for kind in KINDS:
        try:
            perturb(answer, reference_points, kind, seed=0, must_not_claim=must_not_claim)
        except NotPerturbable:
            continue
        kinds.append(kind)
    return kinds


def perturb(answer: str, reference_points: Sequence[str], kind: str, *, seed: int | str,
            must_not_claim: Sequence[str] = ()) -> Perturbation:
    """Change one `kind` of fact in `answer` that `reference_points` also state. Deterministic for a given seed."""
    if kind not in KINDS:
        raise ValueError(f"unknown perturbation kind {kind!r}; expected one of {KINDS}")
    if not (answer or "").strip():
        raise NotPerturbable("an empty answer (or a decline) has no fact to change")
    rng = random.Random(f"{seed}|{kind}")
    refs = [r for r in reference_points if r]
    if kind == "number":
        ref_numbers = _numbers_in(refs)
        masked = _mask_dates(answer)
        candidates = [m for m in _NUMBER.finditer(masked) if _digits(m.group(1)) + (m.group(2) or "") in ref_numbers]
        if not candidates:
            raise NotPerturbable("no figure in the answer is one the reference points state")
        m = _choose(rng, candidates)
        old = m.group(0)
        new = _new_number(old, ref_numbers | _numbers_in([answer]), rng)
        target = next(r for r in refs if _digits(m.group(1)) + (m.group(2) or "") in _numbers_in([r]))
        text = answer[:m.start()] + new + answer[m.end():]
    elif kind == "date":
        ref_dates = _dates_in(refs)
        spans = [s for s in _date_spans(answer) if s[2] in ref_dates]
        if not spans:
            raise NotPerturbable("no date in the answer is one the reference points state")
        start, end, old_date, style = _choose(rng, spans)
        taken = ref_dates | _dates_in([answer])
        shifts = [timedelta(days=d) for d in (rng.choice((3, 5, 9, 12)), -rng.choice((4, 6, 10)), 365, -365, 31)]
        new_date = next((old_date + s for s in shifts if old_date + s not in taken and not _sealed(old_date + s)),
                        None)  # fmt: skip
        if new_date is None:
            raise NotPerturbable("no shifted date avoids the entry's dates and the sealed window")
        old = answer[start:end]
        new = _fmt_date(new_date, style)
        target = next(r for r in refs if old_date in _dates_in([r]))
        text = answer[:start] + new + answer[end:]
    else:
        pairs = _RESULTS if kind == "result" else _NEGATIONS
        hits = []
        for original, opposite in pairs:
            for m in re.finditer(rf"\b{re.escape(original)}\b", answer, flags=re.I):
                if any(re.search(rf"\b{re.escape(_root(original))}", r, flags=re.I) for r in refs):
                    target = next(r for r in refs if re.search(rf"\b{re.escape(_root(original))}", r, flags=re.I))
                elif any(re.search(rf"\b{re.escape(_root(opposite))}", c, flags=re.I) for c in must_not_claim):
                    target = next(c for c in must_not_claim if re.search(rf"\b{re.escape(_root(opposite))}", c,
                                                                          flags=re.I))  # fmt: skip
                else:
                    continue
                hits.append((m, original, opposite, target))
            if hits:
                break  # the longest matching pair wins; never a shorter one inside it
        if not hits:
            raise NotPerturbable(f"no {kind} word in the answer is one the entry states")
        m, original, opposite, target = _choose(rng, hits)
        old = m.group(0)
        new = _match_case(old, opposite)
        text = answer[:m.start()] + new + answer[m.end():]
    if text == answer:
        raise NotPerturbable("the change would leave the answer as it was")
    return Perturbation(kind=kind, text=text, original=old, replacement=new, contradicts=target,
                        judge_safe=kind in JUDGE_SAFE)


def _root(phrase: str) -> str:
    """The word a reference point would use: "voted Aye" -> "Aye", "was agreed to" -> "agreed"."""
    words = [w for w in phrase.split() if w.lower() not in {"voted", "was", "to", "have", "it"}]
    return words[0] if words else phrase


def _match_case(old: str, new: str) -> str:
    return new[0].upper() + new[1:] if old[:1].isupper() else new
