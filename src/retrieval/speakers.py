"""Resolve MPs and peers named in a question to member ids. Pure: build the index once from member rows."""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

MAX_MEMBERS = 8  # a question naming more people than this is not about who spoke
MAX_AMBIGUOUS = 4  # a surname shared by more members than this is too vague to steer a search
_MAX_NAME_TOKENS = 8

# Stripped from the ends of names when forming keys.
_TITLE_WORDS = frozenset(
    "the rt hon right honourable honorable mr mrs ms miss mx dr sir dame lord lady baroness baron viscount earl reverend rev "
    "professor prof cllr bishop archbishop".split()
)
_TITLES_BEFORE_SURNAME = _TITLE_WORDS - {"the"}
_POST_NOMINALS = frozenset("mp msp mbe obe cbe kbe dbe kc qc dl frs cvo cb cmg dsc mc".split())
_ALWAYS_COMMON = frozenset("may will can must shall might could would should".split())  # stopwords: corpus counts cannot flag them

_UNIT = re.compile(r"[^\W_]+(?:['’`][^\W_]+)*(?:-[^\W_]+(?:['’`][^\W_]+)*)*")  # a word, keeping its hyphens and apostrophes
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)

Key = tuple[str, ...]


def _tokens(text: str) -> list[str]:
    """Lower-cased, accent-free words; apostrophes and full stops vanish (O'Brien -> obrien), other punctuation splits.
    """
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)).casefold()
    return re.findall(r"[^\W_]+", re.sub(r"['’`´.]", "", text))


def _strip_titles(tokens: list[str]) -> list[str]:
    start, end = 0, len(tokens)
    while start < end and tokens[start] in _TITLE_WORDS:
        start += 1
    while end > start and tokens[end - 1] in _POST_NOMINALS:
        end -= 1
    return tokens[start:end]


def name_keys(name: str | None, list_as: str | None, full_title: str | None) -> tuple[set[Key], Key | None]:
    """(full-name keys of two or more words, the surname key or None) for one member. `name` is kept with and without
    its title (Ms Nusrat Ghani, Nusrat Ghani); the full title form (Rt Hon Graham Stuart MP) is kept stripped.
    """
    full: set[Key] = set()
    if name:
        tokens = _tokens(name)
        full.add(tuple(tokens))
        full.add(tuple(_strip_titles(tokens)))
    if full_title:
        full.add(tuple(_strip_titles(_tokens(full_title))))
    full = {k for k in full if len(k) >= 2 and len(k) <= _MAX_NAME_TOKENS}
    surname: Key | None = None
    if list_as and "," in list_as:
        head = _tokens(list_as.split(",", 1)[0])
        surname = tuple(head) if head else None
    return full, surname


@dataclass(frozen=True)
class Resolution:
    member_ids: tuple[int, ...] = ()  # in order of first mention, at most MAX_MEMBERS
    mentions: tuple[str, ...] = ()  # the words matched, with any title and post-nominal
    spans: tuple[tuple[int, int], ...] = ()  # character spans of the mentions in the query
    ambiguous: bool = False  # a surname that more than one member holds

    def __bool__(self) -> bool:
        return bool(self.member_ids)


def strip_mentions(query: str, resolution: Resolution) -> str:
    """The query without the names: they are not in the text, and a name's rare words would swamp the keyword leg.
    """
    for start, end in sorted(resolution.spans, reverse=True):
        query = f"{query[:start]} {query[end:]}"
    return " ".join(query.split())


def common_df_threshold(n_docs: int, *, absolute: int, ratio: float | None) -> int:
    """How many chunks a word must appear in before it counts as an ordinary word."""
    if ratio is None:
        return absolute
    return max(1, round(ratio * n_docs))


class SpeakerIndex:
    def __init__(self, full: dict[Key, set[int]], surnames: dict[Key, set[int]],
                 is_common: Callable[[str], bool], members: set[int],
                 service: dict[int, tuple[object, object]] | None = None):
        self._full, self._surnames, self._is_common, self._members = full, surnames, is_common, members
        self._service = service or {}
        self._max_full = max((len(k) for k in full), default=0)
        self._max_surname = max((len(k) for k in surnames), default=0)

    @classmethod
    def from_rows(cls, rows: Iterable[Sequence], is_common: Callable[[str], bool] | None = None) -> "SpeakerIndex":
        """rows: (mnis_id, name, name_list_as, name_full_title[, start_date, end_date])."""
        full: dict[Key, set[int]] = defaultdict(set)
        surnames: dict[Key, set[int]] = defaultdict(set)
        members: set[int] = set()
        service: dict[int, tuple[object, object]] = {}
        for row in rows:
            mnis_id, name, list_as, full_title = row[0], row[1], row[2], row[3]
            keys, surname = name_keys(name, list_as, full_title)
            if not keys and not surname:
                continue
            members.add(mnis_id)
            if len(row) >= 6:
                service[mnis_id] = (row[4], row[5])
            for key in keys:
                full[key].add(mnis_id)
            if surname:
                surnames[surname].add(mnis_id)
        return cls(dict(full), dict(surnames), is_common or (lambda _word: False), members, service)

    def __len__(self) -> int:
        return len(self._members)

    def serving(self, mnis_id: int, on_date) -> bool:
        """Was this member serving on that date? Unknown dates mean yes: a member whose service dates the API does not
        give must not vanish from every dated query.
        """
        start, end = self._service.get(mnis_id, (None, None))
        if start is not None and on_date < start:
            return False
        return not (end is not None and on_date > end)

    def resolve(self, query: object, on_date=None) -> Resolution:
        if not isinstance(query, str) or not query.strip() or not self._members:
            return Resolution()
        units = [(m.start(), m.end(), m.group()) for m in _UNIT.finditer(query)]
        flat: list[tuple[str, int]] = [(t, u) for u, (_, _, text) in enumerate(units) for t in _tokens(_POSSESSIVE.sub("", text))]
        tokens = [t for t, _ in flat]
        consumed = [False] * len(flat)
        found: list[tuple[int, int, set[int]]] = []  # (first flat index, one past the last, member ids)

        i = 0  # pass 1: full names, longest first
        while i < len(flat):
            for n in range(min(self._max_full, len(flat) - i), 1, -1):
                ids = self._full.get(tuple(tokens[i : i + n]))
                if ids:
                    found.append((i, i + n, ids))
                    consumed[i : i + n] = [True] * n
                    i += n
                    break
            else:
                i += 1

        i = 0  # pass 2: bare surnames, under the guard rails
        while i < len(flat):
            for n in range(min(self._max_surname, len(flat) - i), 0, -1):
                ids = self._surnames.get(tuple(tokens[i : i + n]))
                if not ids or any(consumed[i : i + n]) or len(ids) > MAX_AMBIGUOUS:
                    continue
                span = tokens[i : i + n]
                titled = i > 0 and not consumed[i - 1] and tokens[i - 1] in _TITLES_BEFORE_SURNAME
                first_unit = flat[i][1]
                capitalised = first_unit > 0 and units[first_unit][2][:1].isupper()
                common = all(t in _ALWAYS_COMMON or self._is_common(t) for t in span)
                if titled or (capitalised and not common):
                    found.append((i, i + n, ids))
                    consumed[i : i + n] = [True] * n
                    i += n
                    break
            else:
                i += 1

        if not found:
            return Resolution()
        found.sort(key=lambda f: f[0])
        member_ids: list[int] = []
        ambiguous = False
        spans: list[tuple[int, int]] = []
        for start, end, ids in found:
            if on_date is not None:
                # A date narrows a shared surname to the people who were actually sitting.
                serving = {m for m in ids if self.serving(m, on_date)}
                if serving:
                    ids = serving
            ambiguous = ambiguous or len(ids) > 1
            member_ids.extend(m for m in sorted(ids) if m not in member_ids)
            first, last = flat[start][1], flat[end - 1][1]
            while first > 0 and set(_tokens(units[first - 1][2])) <= _TITLE_WORDS and _tokens(units[first - 1][2]):
                first -= 1  # take the title with the name: "Rt Hon" goes, so does "Ms"
            while last + 1 < len(units) and _tokens(units[last + 1][2]) and set(_tokens(units[last + 1][2])) <= _POST_NOMINALS:
                last += 1  # and a trailing "MP" or "KC"
            spans.append((units[first][0], units[last][1]))
        return Resolution(
            member_ids=tuple(member_ids[:MAX_MEMBERS]),
            mentions=tuple(query[a:b] for a, b in spans),
            spans=tuple(spans),
            ambiguous=ambiguous,
        )
