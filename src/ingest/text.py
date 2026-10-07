"""Text primitives shared by chunking, the rules engine and the member parser."""

import html
import re
from bisect import bisect_left, bisect_right
from typing import NamedTuple

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

# --------------------------------------------------------------------------- Cleaning
# ---------------------------------------------------------------------------

# Hansard wraps written questions in these; they mark paragraph boundaries.
_BLOCK_TAG = re.compile(r"</?(?:p|br|td|th|tr|Question|QuestionText)\b[^>]*>", re.I)
# QNum is a label, not a boundary: committee transcripts render "<strong>Q</strong> <QNum/> question text" inline, and a
# break here strands a lone "Q" chunk.
_LABEL_TAG = re.compile(r"</?QNum\b[^>]*>", re.I)
# A tag must start "<x" or "</x" so a bare "a < b" or "3 <4" is left alone.
_ANY_TAG = re.compile(r"</?[A-Za-z][^>]*>")


def clean_text(raw: str | None) -> str:
    """Strip Hansard HTML and normalise whitespace; paragraphs become newlines."""
    if not raw:
        return ""
    text = _BLOCK_TAG.sub("\n", raw)
    text = _LABEL_TAG.sub(" ", text)
    text = _ANY_TAG.sub("", text)  # inline tags: span column markers, em, strong
    text = html.unescape(text)  # after tag removal, so "&lt;p&gt;" stays text
    text = re.sub(r"[^\S\n]+", " ", text)  # any whitespace except newline, incl. nbsp
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------- Column markers
# ---------------------------------------------------------------------------


class Marker(NamedTuple):
    offset: int  # index in the cleaned text of the first character of the new column
    ref: str  # as printed: "1341", "278GC", "45WH"


_MARKER_TAG = re.compile(r'<span\b[^>]*\bclass="[^"]*\bcolumn-number\b[^"]*"[^>]*>\s*</span>', re.I)
_MARKER_VALUE = re.compile(r'\bdata-column-number="([^"]+)"', re.I)
_OPEN, _CLOSE = "", ""  # private-use sentinels: they survive cleaning, unlike a <span>
_SENTINEL = re.compile(f"{_OPEN}([^{_CLOSE}]*){_CLOSE}")


def _to_sentinel(match: re.Match) -> str:
    value = _MARKER_VALUE.search(match.group(0))
    return f"{_OPEN}{value.group(1)}{_CLOSE}" if value else ""  # a marker with no value is dropped, not guessed


def clean_text_with_markers(raw: str | None) -> tuple[str, list[Marker]]:
    """clean_text(raw) plus the column markers. The text is identical to clean_text(raw) (a test pins that), so offsets
    are valid for everything that reads the cleaned text.
    """
    if not raw:
        return "", []
    cleaned = clean_text(_MARKER_TAG.sub(_to_sentinel, raw))
    if _OPEN not in cleaned:
        return cleaned, []
    out, cursor, markers = "", 0, []
    for m in _SENTINEL.finditer(cleaned):
        out += cleaned[cursor : m.start()]
        cursor = m.end()
        # The marker was hiding an adjacency from clean_text's whitespace rules.
        while True:
            left, right = out[-1:], cleaned[cursor : cursor + 1]
            if left == " " and right == " ":
                cursor += 1
            elif left == " " and right == "\n":
                out = out[:-1]
            elif left == "\n" and right in (" ", "\n"):
                cursor += 1
            else:
                break
        markers.append(Marker(len(out), m.group(1)))
    out += cleaned[cursor:]
    lead = len(out) - len(out.lstrip())
    out = out.strip()
    return out, [Marker(min(max(mk.offset - lead, 0), len(out)), mk.ref) for mk in markers]


_REF = re.compile(r"^(\d+)(\D*)$")


def column_number(ref: str | None) -> int | None:
    m = _REF.match(ref or "")
    return int(m.group(1)) if m else None


def previous_ref(ref: str | None) -> str | None:
    """The column before `ref`, keeping its suffix ("278GC" -> "277GC"); None if it cannot be derived."""
    m = _REF.match(ref or "")
    if not m or int(m.group(1)) <= 1:
        return None
    return f"{int(m.group(1)) - 1}{m.group(2)}"


# --------------------------------------------------------------------------- Sentences
# ---------------------------------------------------------------------------

# Untrained Punkt plus an explicit abbreviation list: no `punkt_tab` download, and it fixes the dominant Hansard
# failure.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "rev", "fr", "st", "sr", "jr",
    "hon", "rt", "gen", "col", "lt", "capt", "cdr", "cllr", "ald", "messrs",
    "no", "nos", "para", "paras", "cl", "cmd", "vol", "ch",
    "vs", "e.g", "i.e", "u.s", "u.k",
}  # fmt: skip


def _make_tokenizer() -> PunktSentenceTokenizer:
    params = PunktParameters()
    params.abbrev_types = set(_ABBREVIATIONS)
    return PunktSentenceTokenizer(params)


_TOKENIZER = _make_tokenizer()
_HAS_LETTER = re.compile(r"[^\W\d_]")
# Ends a sentence: terminal punctuation, then any closing quotes/brackets.
_TERMINAL = re.compile(r"[.!?…][\"'’”)\]]*$")


class Span(NamedTuple):
    text: str
    start: int  # offsets into the text that was split
    end: int


def split_sentence_spans(text: str | None) -> list[Span]:
    """Split cleaned text into sentences, with where each one is."""
    spans: list[Span] = []
    offset = 0
    for paragraph in (text or "").split("\n"):
        stripped = paragraph.strip()
        if stripped:
            base = offset + len(paragraph) - len(paragraph.lstrip())
            for i, (a, b) in enumerate(_TOKENIZER.span_tokenize(stripped)):
                piece = stripped[a:b]
                sentence = piece.strip()
                if not _HAS_LETTER.search(sentence):
                    continue
                start = base + a + len(piece) - len(piece.lstrip())
                end = start + len(sentence)
                if spans and sentence[0].islower() and (i > 0 or not _TERMINAL.search(spans[-1].text)):
                    spans[-1] = Span(spans[-1].text + " " + sentence, spans[-1].start, end)
                else:
                    spans.append(Span(sentence, start, end))
        offset += len(paragraph) + 1
    return spans


def split_sentences(text: str | None) -> list[str]:
    return [s.text for s in split_sentence_spans(text)]


# --------------------------------------------------------------------------- Column assignment
# ---------------------------------------------------------------------------


class ColumnInfo(NamedTuple):
    ref: str | None  # column the sentence STARTS in
    number: int | None
    end_ref: str | None  # set only when the sentence runs into another column
    basis: str | None  # marker (read in this speech) | carried (from an earlier speech) | inferred


def assign_columns(spans: list[Span], markers: list[Marker], carry: tuple[str, str] | None = None) -> list[ColumnInfo]:
    """`carry` is (ref, basis) for text that precedes every marker in this speech."""
    offsets = [m.offset for m in markers]
    out = []
    for span in spans:
        i = bisect_right(offsets, span.start) - 1  # last marker at or before the sentence start
        if i >= 0:
            ref, basis = markers[i].ref, "marker"
        elif carry:
            ref, basis = carry
        else:
            ref = basis = None
        j = bisect_left(offsets, span.end) - 1  # last marker strictly inside the sentence
        end_ref = markers[j].ref if j >= 0 and offsets[j] > span.start and markers[j].ref != ref else None
        out.append(ColumnInfo(ref, column_number(ref), end_ref, basis))
    return out
