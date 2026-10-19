"""The rebuild oracle: does `hansard_v2`, built from the cache alone, reproduce `hansard`?"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.conninfo import make_conninfo
from rich.console import Console
from rich.table import Table

from app.config import settings

console = Console()

# W0's scope, as the original ingest defined it.
W0_START, W0_END = "2026-09-01", "2026-09-17"

# Rows that are in `hansard` and not in `hansard_v2`, each with why.
EXPLAINED_ONLY_OLD = {
    "afccb16f-e0c8-4082-b297-d426404234f7":
        "A debate with no speeches. The API served 26 General Committee contributions of "
        "2026-09-09 under BOTH this id and 4c458239-4668-4241-9013-751a4b8e7c79, two "
        "records of one sitting that took a draft Code of Practice and draft Regulations "
        "together. The old ingest read its debate rows from the un-deduplicated pages and "
        "so created this empty shell; both databases attach all 26 contributions to "
        "4c458239. hansard_v2 is right not to have it.",
    "00914F62-97DA-49F2-BA23-22734B65B808":
        "A contribution (2026-09-09, Westminster Hall, 'Pica Eating Disorder: SEND', "
        "Margaret Mullane) that the API listed when the original ingest ran and no longer "
        "serves: the day now lists 754 and serves 727. This is I-17's 'stored speeches the "
        "API does not return', seen from the other side. It accounts for exactly 1 speech, "
        "9 chunks, 2 boilerplate chunks and 3 procedure events -- which is the whole of "
        "every remaining count difference.",
}

# Counts that are expected to grow as the ruleset grows, so a difference here is reported rather than failed.
INFORMATIONAL = frozenset({"procedure events (all types)"})

# The count differences those two rows explain, as arithmetic rather than assertion.
EXPLAINED_DELTAS = {
    "debates": 1, "speeches": 1, "speeches with an MNIS id": 1,
    "procedure events (types `hansard` also has)": 3,
    "chunks": 9, "boilerplate chunks": 2, "debate ids": 1, "speech ids": 1,
    "speech text, speaker, flags": 1, "debate heading, section, type": 1,
    "chunks per speech": 1, "procedure event types": 3,
}


@dataclass
class Check:
    name: str
    old: Any
    new: Any
    detail: str = ""
    only_old: list[str] = field(default_factory=list)
    only_new: list[str] = field(default_factory=list)
    # A count that is EXPECTED to grow, so a difference is information rather than a failure.
    informational: bool = False

    @property
    def same(self) -> bool:
        return self.old == self.new and not self.only_old and not self.only_new

    @property
    def explained(self) -> bool:
        """Same, or different only by rows and counts already accounted for."""
        if self.same or self.informational:
            return True
        if self.only_new:
            return False
        if any(i.split(":")[-1] not in EXPLAINED_ONLY_OLD for i in self.only_old):
            return False
        expected = EXPLAINED_DELTAS.get(self.name)
        if expected is None:
            return False
        old, new = self.old, self.new
        if isinstance(old, dict) and isinstance(new, dict):
            old, new = sum(old.values()), sum(new.values())
        if isinstance(old, int) and isinstance(new, int):
            return old - new == expected
        return False

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "hansard": self.old, "hansard_v2": self.new,
                "same": self.same, "explained": self.explained, "detail": self.detail,
                "informational": self.informational,
                "only_hansard": self.only_old[:20], "only_hansard_v2": self.only_new[:20]}


SCOPE = f"d.date BETWEEN '{W0_START}' AND '{W0_END}'"

COUNTS = {
    "debates": f"SELECT count(*) FROM debate d WHERE {SCOPE}",
    "debates with at least one speech":
        f"SELECT count(*) FROM debate d WHERE {SCOPE} AND EXISTS (SELECT 1 FROM speech s WHERE s.debate_id = d.id)",
    "speeches": f"SELECT count(*) FROM speech s JOIN debate d ON d.id = s.debate_id WHERE {SCOPE}",
    "speeches with an MNIS id":
        f"SELECT count(*) FROM speech s JOIN debate d ON d.id = s.debate_id WHERE {SCOPE} AND s.mnis_id IS NOT NULL",
    "boilerplate speeches":
        f"SELECT count(*) FROM speech s JOIN debate d ON d.id = s.debate_id WHERE {SCOPE} AND s.is_boilerplate",
    # The like-for-like count is the "types `hansard` also has" check.
    "procedure events (all types)":
        f"SELECT count(*) FROM procedure_event p JOIN debate d ON d.id = p.debate_id WHERE {SCOPE}",
    "chunks": f"SELECT count(*) FROM chunk c JOIN debate d ON d.id = c.debate_id WHERE {SCOPE}",
    "boilerplate chunks":
        f"SELECT count(*) FROM chunk c JOIN debate d ON d.id = c.debate_id WHERE {SCOPE} AND c.is_boilerplate",
    "divisions": "SELECT count(*) FROM division WHERE division_date BETWEEN %(start)s AND %(end)s",
    "division votes":
        "SELECT count(*) FROM division_vote v JOIN division x ON x.id = v.division_id "
        "WHERE x.division_date BETWEEN %(start)s AND %(end)s",
}

ID_SETS = {
    "debate ids": f"SELECT lower(d.id) FROM debate d WHERE {SCOPE}",
    "speech ids": f"SELECT s.id FROM speech s JOIN debate d ON d.id = s.debate_id WHERE {SCOPE}",
}

# One row per speech: everything that has to be identical, hashed so a mismatch is visible without shipping 8,986 texts
# around.
SPEECH_FINGERPRINT = f"""
    SELECT s.id,
           md5(coalesce(s.text, '')) AS text_md5,
           coalesce(s.speaker_name, '')  AS speaker_name,
           coalesce(s.mnis_id, -1)       AS mnis_id,
           s.is_boilerplate,
           coalesce(s.boilerplate_reason, '') AS reason,
           coalesce(s.speaker_office, '') AS office,
           coalesce(s.speaker_person, '') AS person,
           coalesce(s.speaker_party, '')  AS party,
           coalesce(s.speaker_role, '')   AS role
    FROM speech s JOIN debate d ON d.id = s.debate_id
    WHERE {SCOPE}
"""

DEBATE_FINGERPRINT = f"""
    SELECT lower(d.id), coalesce(d.heading, ''), coalesce(d.section, ''), coalesce(d.house, ''),
           d.date::text, coalesce(d.debate_type, '')
    FROM debate d WHERE {SCOPE}
"""

EVENT_TYPES = f"""
    SELECT p.event_type, count(*) FROM procedure_event p JOIN debate d ON d.id = p.debate_id
    WHERE {SCOPE} GROUP BY 1
"""

CHUNKS_PER_SPEECH = f"""
    SELECT c.speech_id, count(*) FROM chunk c JOIN debate d ON d.id = c.debate_id
    WHERE {SCOPE} GROUP BY 1
"""

DIVISION_FINGERPRINT = """
    SELECT id, house, number, division_date::text, coalesce(ayes, -1), coalesce(noes, -1),
           coalesce(ayes_listed, -1), coalesce(noes_listed, -1)
    FROM division WHERE division_date BETWEEN %(start)s AND %(end)s
"""


def _rows(conn, sql: str) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, {"start": W0_START, "end": W0_END})
        return cur.fetchall()


def _scalar(conn, sql: str) -> int:
    return _rows(conn, sql)[0][0]


def _set_check(name: str, old: list[tuple], new: list[tuple]) -> Check:
    a = {r[0] for r in old}
    b = {r[0] for r in new}
    return Check(name, len(a), len(b), only_old=sorted(str(x) for x in a - b),
                 only_new=sorted(str(x) for x in b - a))


def _map_check(name: str, old: list[tuple], new: list[tuple], key_cols: int = 1) -> Check:
    """Compare row-by-row on a key, reporting how many keys differ in their other columns."""
    a = {r[:key_cols]: r[key_cols:] for r in old}
    b = {r[:key_cols]: r[key_cols:] for r in new}
    shared = a.keys() & b.keys()
    differing = sorted(str(k[0]) for k in shared if a[k] != b[k])
    return Check(name, len(a), len(b), detail=f"{len(differing)} of {len(shared)} shared rows differ",
                 only_old=sorted(str(k[0]) for k in a.keys() - b.keys()),
                 only_new=sorted(str(k[0]) for k in b.keys() - a.keys()) + [f"DIFFERS:{d}" for d in differing[:20]])


def compare(old_dsn: str, new_dsn: str) -> list[Check]:
    checks: list[Check] = []
    with psycopg.connect(old_dsn) as old, psycopg.connect(new_dsn) as new:
        for name, sql in COUNTS.items():
            checks.append(Check(name, _scalar(old, sql), _scalar(new, sql),
                                informational=name in INFORMATIONAL))
        for name, sql in ID_SETS.items():
            checks.append(_set_check(name, _rows(old, sql), _rows(new, sql)))
        checks.append(_map_check("speech text, speaker, flags", _rows(old, SPEECH_FINGERPRINT),
                                 _rows(new, SPEECH_FINGERPRINT)))
        checks.append(_map_check("debate heading, section, type", _rows(old, DEBATE_FINGERPRINT),
                                 _rows(new, DEBATE_FINGERPRINT)))
        checks.append(_map_check("chunks per speech", _rows(old, CHUNKS_PER_SPEECH),
                                 _rows(new, CHUNKS_PER_SPEECH)))
        checks.append(_map_check("divisions and tallies", _rows(old, DIVISION_FINGERPRINT),
                                 _rows(new, DIVISION_FINGERPRINT)))
        old_events, new_events = Counter(dict(_rows(old, EVENT_TYPES))), Counter(dict(_rows(new, EVENT_TYPES)))
        # Compare only the event types `hansard` could produce.
        added = {k: n for k, n in new_events.items() if k not in old_events}
        comparable = Counter({k: n for k, n in new_events.items() if k in old_events})
        differing = sorted(k for k in old_events.keys() | comparable.keys()
                           if old_events[k] != comparable[k])
        detail = ", ".join(f"{k}: {old_events[k]} -> {comparable[k]}" for k in differing) or "identical"
        if added:
            detail += f"  [new event types in hansard_v2 only, not a difference: {added}]"
        checks.append(Check("procedure event types", dict(old_events), dict(comparable), detail=detail))
        checks.append(Check("procedure events (types `hansard` also has)",
                            sum(old_events.values()), sum(comparable.values()),
                            detail=f"{sum(added.values())} event(s) of {len(added)} new type(s) excluded"))
    return checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--old", default="hansard", help="the database to reproduce (default: %(default)s)")
    p.add_argument("--new", default="hansard_v2")
    p.add_argument("--json", help="also write the result here")
    args = p.parse_args()

    old_dsn = make_conninfo(settings.pg_dsn, dbname=args.old)
    new_dsn = make_conninfo(settings.pg_dsn, dbname=args.new)
    checks = compare(old_dsn, new_dsn)

    table = Table(title=f"W0 rebuild oracle: {args.new} against {args.old} ({W0_START} .. {W0_END})")
    table.add_column("Check")
    table.add_column(args.old, justify="right")
    table.add_column(args.new, justify="right")
    table.add_column("")
    for check in checks:
        old = check.old if not isinstance(check.old, dict) else f"{sum(check.old.values()):,}"
        new = check.new if not isinstance(check.new, dict) else f"{sum(check.new.values()):,}"
        mark = ("[green]same[/green]" if check.same else
                "[blue]expected to grow[/blue]" if check.informational else
                "[cyan]explained[/cyan]" if check.explained else "[bold red]UNEXPLAINED[/bold red]")
        table.add_row(check.name, f"{old:,}" if isinstance(old, int) else str(old),
                      f"{new:,}" if isinstance(new, int) else str(new), mark)
    console.print(table)
    for check in checks:
        if check.same:
            continue
        console.print(f"[bold]{check.name}[/bold]: {check.detail or ''}"
                      f"{'' if check.explained else '  [red](not explained)[/red]'}")
        if check.only_old:
            console.print(f"  only in {args.old}: {check.only_old[:10]}")
        if check.only_new:
            console.print(f"  only in {args.new}: {check.only_new[:10]}")
    unexplained = [c.name for c in checks if not c.explained]
    if not unexplained:
        console.print("\n[bold]Every difference is accounted for:[/bold]")
        for ident, why in EXPLAINED_ONLY_OLD.items():
            console.print(f"  [cyan]{ident}[/cyan]: {why}")

    payload = {"old": args.old, "new": args.new, "scope": [W0_START, W0_END],
               "checks": [c.as_dict() for c in checks],
               "same": sum(c.same for c in checks),
               "explained": sum(c.explained for c in checks),
               "unexplained": unexplained,
               "explanations": EXPLAINED_ONLY_OLD, "total": len(checks)}
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8", newline="\n")
        console.print(f"written to {path}")
    console.print(f"[bold]{payload['same']} of {payload['total']} checks identical, "
                  f"{payload['explained']} identical or explained, {len(unexplained)} unexplained[/bold]")
    sys.exit(1 if unexplained else 0)


if __name__ == "__main__":
    main()
