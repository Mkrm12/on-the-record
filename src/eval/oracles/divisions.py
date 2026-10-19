"""The division oracle (ST-*): every result, tally and member vote, from the Votes API payloads as fetched."""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app import sealed

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "corpus_v2"
log = logging.getLogger(__name__)

# The corpus numbers Lords divisions from here, so they never collide with the Commons ids.
LORDS_OFFSET = 1_000_000


@dataclass(frozen=True)
class Vote:
    mnis_id: int
    lobby: str  # aye | no | no_vote | conflict
    is_teller: bool
    proxy_name: str | None
    party: str | None
    name: str | None = None  # as the division list printed it


@dataclass(frozen=True)
class DivisionTruth:
    division_id: int
    house: str
    date: date
    number: int | None
    title: str
    ayes: int | None
    noes: int | None
    ayes_listed: int
    noes_listed: int
    ayes_named: int
    noes_named: int
    tally_reconciled: bool | None
    votes: tuple[Vote, ...]
    printed: tuple[int, int] | None = None  # Hansard's printed figures (ayes, noes), where Hansard records the division
    is_deferred: bool = False
    # English Votes for English Laws (2015-21): the second count, of the MPs concerned, when the payload has one.
    double_majority: tuple[int | None, int | None] | None = None
    # The motion as the payload prints it (the Lords' `amendmentMotionNotes`, tags removed): where the Lords name the
    # amendment a division was on.
    motion: str = ""
    # The Commons payload's own description of the division (`FriendlyDescription`): for a tie, how the casting vote
    # went ("Tied vote: casting vote with the Ayes").
    described: str = ""
    # When the vote was taken, by the payload's clock: the Commons `Date`, the Lords `remoteVotingStart` (the Lords
    # `date` is the sitting's).
    held_at: datetime | None = None

    @property
    def casting(self) -> str | None:
        """The outcome the record's own words give a tie (the casting vote with the Ayes, or the Noes), else None.
        """
        found = re.search(r"\bcasting\s+vote\s+(?:was\s+)?(?:given\s+)?with\s+the\s+(ayes|noes|contents|not[- ]contents)\b",
                          self.described or "", re.I)
        if not found:
            return None
        return "agreed" if found.group(1).lower() in ("ayes", "contents") else "negatived"

    @property
    def result(self) -> str | None:
        """The record's result: None when a figure is unrecorded, or an EVEL division's two majorities disagree or one
        is unrecorded (it passed only with both); a tie when the counts are equal (the Chair's casting vote).
        """
        if self.ayes is None or self.noes is None:
            return None
        whole = "agreed" if self.ayes > self.noes else "negatived" if self.noes > self.ayes else "tie"
        if self.double_majority is not None:
            a, n = self.double_majority
            if a is None or n is None:
                return None
            subset = "agreed" if a > n else "negatived" if n > a else "tie"
            if subset != whole:
                return None
        return whole


@dataclass(frozen=True)
class Disagreement:
    division_id: int
    check: str
    mnis_id: int | None
    oracle: Any
    database: Any


@dataclass(frozen=True)
class SourceConflict:
    """The Votes API's figures and Hansard's printed ones disagree: a defect in a source, not the database."""

    division_id: int
    votes_api: tuple[int, int]
    hansard: tuple[int, int]


def _day(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def _clock(value: Any) -> datetime | None:
    """A payload timestamp as a time of day; None when the payload has none."""
    return datetime.fromisoformat(str(value)) if value else None


def _votes(lobbies: list[tuple[str, bool, list[Mapping[str, Any]]]], *, key: str, party: str,
           proxy: str | None, name: str) -> tuple[Vote, ...]:  # fmt: skip
    """Each member's vote, decided from every listing at once so that list order cannot change it: named in both the
    aye and the no lobby, a conflict; in one, that lobby (a "no vote recorded" line beside a vote does not undo the
    vote); in neither, no vote. A teller if any listing was a teller's.
    """
    listings: dict[int, list[tuple[str, bool, Mapping[str, Any]]]] = {}
    for lobby, teller, members in lobbies:
        for m in members:
            if m.get(key) is None:
                continue  # an entry naming no member counts for no one, as the loader reports it
            listings.setdefault(int(m[key]), []).append((lobby, teller, m))
    out = []
    for mid, items in listings.items():
        voted = {lobby for lobby, _, _ in items if lobby in ("aye", "no")}
        lobby = "conflict" if voted == {"aye", "no"} else next(iter(voted)) if voted else "no_vote"
        mine = [i for i in items if i[0] == lobby] or items
        entry = next((m for l_, t, m in mine if not t), mine[0][2])
        out.append(Vote(mid, lobby, any(t for _, t, _ in items), (entry.get(proxy) or None) if proxy else None,
                        entry.get(party) or None, entry.get(name) or None))  # fmt: skip
    return tuple(sorted(out, key=lambda v: v.mnis_id))


def _listed(votes: tuple[Vote, ...], lobby: str) -> int:
    """The members recorded in a lobby: each once, tellers included, a conflict in neither."""
    return sum(1 for v in votes if v.lobby == lobby)


def _figure(value: Any) -> int | None:
    return None if value is None else int(value)


def _ids(members: list[Mapping[str, Any]], key: str) -> set[int]:
    return {int(m[key]) for m in members if m.get(key) is not None}


def parse_commons(payload: Mapping[str, Any]) -> DivisionTruth:
    ayes, noes = payload.get("Ayes") or [], payload.get("Noes") or []
    aye_tellers, no_tellers = payload.get("AyeTellers") or [], payload.get("NoTellers") or []
    votes = _votes([("aye", False, ayes), ("no", False, noes), ("aye", True, aye_tellers), ("no", True, no_tellers),
                    ("no_vote", False, payload.get("NoVoteRecorded") or [])],
                   key="MemberId", party="Party", proxy="ProxyName", name="Name")  # fmt: skip
    ayes_named = len(_ids(ayes, "MemberId") - _ids(aye_tellers, "MemberId"))
    noes_named = len(_ids(noes, "MemberId") - _ids(no_tellers, "MemberId"))
    figures = _figure(payload.get("AyeCount")), _figure(payload.get("NoCount"))
    evel = None
    if payload.get("DoubleMajorityAyeCount") is not None or payload.get("DoubleMajorityNoCount") is not None:
        evel = _figure(payload.get("DoubleMajorityAyeCount")), _figure(payload.get("DoubleMajorityNoCount"))
    return DivisionTruth(division_id=int(payload["DivisionId"]), house="Commons", date=_day(payload["Date"]),
                         held_at=_clock(payload["Date"]),
                         number=payload.get("Number"), title=str(payload.get("Title") or "").strip(),
                         ayes=figures[0], noes=figures[1],
                         ayes_listed=_listed(votes, "aye"), noes_listed=_listed(votes, "no"),
                         ayes_named=ayes_named, noes_named=noes_named,
                         tally_reconciled=None if None in figures else (ayes_named, noes_named) == figures,
                         votes=votes, is_deferred=bool(payload.get("IsDeferred")), double_majority=evel,
                         described=_text(payload.get("FriendlyDescription")))  # fmt: skip


def _text(html: Any) -> str:
    """Markup as plain text: tags removed, entities decoded, spacing collapsed (`html` here is the payload's field).
    """
    import html as _html

    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", str(html or ""))).split())


def parse_lords(payload: Mapping[str, Any]) -> DivisionTruth:
    contents, not_contents = payload.get("contents") or [], payload.get("notContents") or []
    content_tellers, not_content_tellers = payload.get("contentTellers") or [], payload.get("notContentTellers") or []
    votes = _votes([("aye", False, contents), ("no", False, not_contents), ("aye", True, content_tellers),
                    ("no", True, not_content_tellers)], key="memberId", party="party", proxy=None,
                   name="name")  # fmt: skip
    ayes_named = len(_ids(contents, "memberId") | _ids(content_tellers, "memberId"))
    noes_named = len(_ids(not_contents, "memberId") | _ids(not_content_tellers, "memberId"))
    figures = _figure(payload.get("authoritativeContentCount")), _figure(payload.get("authoritativeNotContentCount"))
    return DivisionTruth(division_id=LORDS_OFFSET + int(payload["divisionId"]), house="Lords",
                         date=_day(payload["date"]), held_at=_clock(payload.get("remoteVotingStart")),
                         number=payload.get("number"),
                         title=str(payload.get("title") or "").strip(), ayes=figures[0], noes=figures[1],
                         ayes_listed=_listed(votes, "aye"), noes_listed=_listed(votes, "no"),
                         ayes_named=ayes_named, noes_named=noes_named,
                         tally_reconciled=None if None in figures else (ayes_named, noes_named) == figures,
                         votes=votes, motion=_text(payload.get("amendmentMotionNotes")))  # fmt: skip


def _read(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _windows(folder: Path) -> list[Path]:
    """The per-window folders, never the sealed one."""
    sealed_id = sealed.window().window_id
    return sorted(p for p in folder.iterdir() if p.is_dir() and p.name != "detail"
                  and not (sealed.masked() and p.name == sealed_id))  # fmt: skip


def printed_figures(raw: Path = RAW) -> dict[tuple[str, date, str], tuple[int, int]]:
    """Hansard's printed figures, (ayes, noes), by (House, date, division number), outside the sealed window."""
    folder = raw / "hansard_divisions"
    seen: dict[tuple[str, date, str], set[tuple[int, int]]] = {}
    if not folder.exists():
        return {}
    for window in _windows(folder):
        for house in ("Commons", "Lords"):
            for index in sorted((window / house).glob("index_*.json.gz")):
                for r in _read(index).get("Results") or []:
                    if sealed.hidden(r.get("Date")) or r.get("AyesCount") is None or r.get("NoesCount") is None:
                        continue
                    key = (house, _day(r["Date"]), str(r.get("Number")).strip())
                    seen.setdefault(key, set()).add((int(r["AyesCount"]), int(r["NoesCount"])))
    # Records that disagree under one key are left out, never picked: which one is right is not the oracle's to say.
    split = sorted(k for k, figures in seen.items() if len(figures) > 1)
    if split:
        log.warning("%d Hansard division key(s) hold disagreeing records and are left out: %s", len(split), split[:5])
    return {k: next(iter(figures)) for k, figures in seen.items() if len(figures) == 1}


def truths(raw: Path = RAW) -> Iterator[DivisionTruth]:
    """Every division the raw cache holds outside the sealed window, Commons then Lords, each with Hansard's printed
    figures where Hansard records it.
    """
    printed = printed_figures(raw)
    for t in _payloads(raw):
        yield replace(t, printed=printed.get((t.house, t.date, str(t.number).strip())))


def source_conflicts(truth: DivisionTruth) -> list[SourceConflict]:
    """The division's figures where the Votes API and Hansard's printed record disagree."""
    if truth.printed is None or truth.printed == (truth.ayes, truth.noes):
        return []
    return [SourceConflict(truth.division_id, (truth.ayes, truth.noes), truth.printed)]


def _payloads(raw: Path) -> Iterator[DivisionTruth]:
    commons = raw / "commons_divisions"
    if commons.exists():
        for window in _windows(commons):
            for index in sorted(window.glob("index_*.json.gz")):
                for entry in _read(index):
                    if sealed.hidden(entry.get("Date")):
                        continue  # dated inside the window, wherever it was filed: its detail file is never opened
                    detail = commons / "detail" / f"{int(entry['DivisionId'])}.json.gz"
                    if detail.exists():
                        payload = _read(detail)
                        if not sealed.hidden(payload.get("Date")):
                            yield parse_commons(payload)
    lords = raw / "lords_divisions"
    if lords.exists():
        for window in _windows(lords):
            for index in sorted(window.glob("index_*.json.gz")):
                for entry in _read(index):
                    if not sealed.hidden(entry.get("date")):
                        yield parse_lords(entry)


# The database's division columns the oracle checks, by its own name for each.
DIVISION_FIELDS = (("house", "house"), ("date", "division_date"), ("number", "number"), ("result", "result"),
                   ("ayes", "ayes"), ("noes", "noes"), ("ayes_listed", "ayes_listed"),
                   ("noes_listed", "noes_listed"), ("ayes_named", "ayes_named"), ("noes_named", "noes_named"),
                   ("tally_reconciled", "tally_reconciled"), ("title", "title"), ("is_deferred", "is_deferred"))  # fmt: skip


def compare(truth: DivisionTruth, record: Mapping[str, Any] | None,
            votes: Mapping[int, Mapping[str, Any]]) -> list[Disagreement]:  # fmt: skip
    """Every way the database disagrees with the payload: the division's own fields, then each member's vote."""
    did = truth.division_id
    if record is None:
        return [Disagreement(did, "division.missing", None, "in the payload", "not in the database")]
    out = []
    for name, column in DIVISION_FIELDS:
        mine, theirs = getattr(truth, name), record.get(column)
        if name == "result" and mine == "tie":
            continue  # the counts cannot say how the Chair voted
        if name == "title":
            mine, theirs = mine or None, (theirs or "").strip() or None
        if column not in record and name in ("title", "is_deferred"):
            continue  # a caller that did not read the column is not told it disagrees
        if mine != theirs:
            out.append(Disagreement(did, f"division.{name}", None, mine, theirs))
    expected = {v.mnis_id: v for v in truth.votes}
    for mid, vote in expected.items():
        row = votes.get(mid)
        if row is None:
            out.append(Disagreement(did, "vote.missing", mid, vote.lobby, None))
            continue
        if row.get("lobby") != vote.lobby:
            out.append(Disagreement(did, "vote.lobby", mid, vote.lobby, row.get("lobby")))
        if bool(row.get("is_teller")) != vote.is_teller:
            out.append(Disagreement(did, "vote.is_teller", mid, vote.is_teller, row.get("is_teller")))
        if (row.get("proxy_name") or None) != vote.proxy_name:
            out.append(Disagreement(did, "vote.proxy_name", mid, vote.proxy_name, row.get("proxy_name")))
        if "party" in row and (row.get("party") or None) != vote.party:
            out.append(Disagreement(did, "vote.party", mid, vote.party, row.get("party")))
    for mid in sorted(set(votes) - set(expected)):
        out.append(Disagreement(did, "vote.extra", mid, None, votes[mid].get("lobby")))
    return out


# --- what a division was on, read from the payload with code of the oracle's own ----------

# A division on an amendment, or on a motion to amend, rather than on the business itself: Parliament's own words for
# amending ("moved amendment 1", "moved Motion A1", "as an amendment to", "insist", "disagree", "in lieu", "leave out",
# "decline to approve").
_AMENDING = re.compile(r"\bamendments?\b|\bnew\s+clauses?\b|\bnew\s+schedules?\b|\bclause\s+\d|\bschedule\s+\d"
                       r"|\bmoved\s+motion\b|\bmotion\s+(?-i:[A-Z])\d*\b|\binsist|\bdisagree|\bin\s+lieu\b|\breasoned\b"
                       r"|\bdeclines?\b|\bregret|\bleave\s+out\b|\binsert\b|\bstand\s+part\b", re.I)


# The oracle's own reading of "Amendment" in a name: "(Fees Amendment) Regulations", "(Amendment) Act".
_IN_A_NAME = re.compile(r"\([^()]*amendment[^()]*\)|\bamendments?\s+(?:regulations?|order|rules|act|bill|scheme)\b",
                        re.I)


def on_an_amendment(t: DivisionTruth) -> bool:
    """Whether the division decided an amendment or a motion to amend, not the business itself; "Amendment" in the
    instrument's or the bill's own name does not count.
    """
    return bool(_AMENDING.search(_IN_A_NAME.sub(" ", f"{t.title or ''} {t.motion or ''}")))


def passage_misread(t: DivisionTruth) -> bool:
    """Whether "did the House pass <its title>" would ask about something the division did not decide: it decided an
    amendment, and its title names only the business (a Lords title is the bill's name; the amendment is in the
    motion). A title that names the amendment itself ("...: New Clause 2") makes the question about the amendment,
    and the division's result is then its answer.
    """
    return on_an_amendment(t) and not _AMENDING.search(_IN_A_NAME.sub(" ", t.title or ""))


_LETTER_RUN = re.compile(r"\b(?:gov(?:ernment|t)?\.?\s+(?:amendments?\s+)?|amendments?\s+)"
                         r"(\([a-z]\)(?:\s*(?:,|and|to)\s*\([a-z]\))*)", re.I)


def letters(title: str) -> set[str]:
    """The lettered amendments a title names: "Gov (a) to (f)", "Government amendments (a) and (b)", "Gov (a)"."""
    out: set[str] = set()
    for m in _LETTER_RUN.finditer(title or ""):
        parts = re.findall(r"\(([a-z])\)|\b(to)\b", m.group(1), re.I)
        seq = [x for x, _ in parts if x]
        out |= {f"({x.lower()})" for x in seq}
        tokens = [x or y for x, y in parts]
        for i, tok in enumerate(tokens):
            if tok.lower() == "to" and 0 < i < len(tokens) - 1:
                lo, hi = tokens[i - 1].lower(), tokens[i + 1].lower()
                out |= {f"({chr(c)})" for c in range(ord(lo), ord(hi) + 1)}
    return out


_AMENDED = re.compile(r"\bmoved\s+(?:amendment|motion)\s+(\d*)([A-Z0-9]+),?\s+as\s+an\s+amendment\s+to\s+(?:replace\s+)?"
                      r"(?:the\s+motion\s+that\b[^,]*?\bin\s+their\s+)?(?:commons\s+|lords\s+)?amendment\s+(\d+[A-Z]*)\b",
                      re.I)  # QA pass 5 (H2): every amending relation the record prints, not one literal
_MOVED = re.compile(r"\bmoved\s+amendment\s+(\d+[A-Z]*)\b")


def amendment_to(t: DivisionTruth) -> tuple[str, str] | None:
    """(the amendment moved, the amendment it amends) when a Lords motion moves an amendment to another: "moved
    amendment 7A, as an amendment to amendment 4".
    """
    # its own number is its place in the marshalled list, not its target's: 7A amends amendment 4
    m = _AMENDED.search(t.motion or "")
    return (m.group(1) + m.group(2), m.group(3)) if m else None


def moved(t: DivisionTruth) -> str | None:
    """The amendment a Lords motion moves ("moved amendment 3, in clause 2" moves 3)."""
    m = _MOVED.search(t.motion or "")
    return m.group(1) if m else None
