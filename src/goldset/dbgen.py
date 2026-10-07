"""The database-derived half of the gold set: ~300 items whose answers come straight from SQL."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from goldset.schema import Evidence, GoldItem, Nugget

# Items drawn from it are born `heldout`; the splitter never reassigns them.
HELDOUT_WINDOWS = frozenset({"W3", "W5"})
HELDOUT_WINDOW = "W3"  # kept as an alias; prefer HELDOUT_WINDOWS

MIN_EVIDENCE_CHARS = 40


def _split_for(window: str) -> str:
    return "heldout" if window in HELDOUT_WINDOWS else ""


def _pretty_date(value: Any) -> str:
    """'12 January 2026'. %-d / %#d are platform-specific, so strip the zero by hand."""
    return f"{value.day} {value:%B %Y}"


@dataclass
class Budget:
    """How many items each template should try to produce."""

    division_result: int = 40
    division_tally: int = 30
    division_vote: int = 45
    party_at_date: int = 28
    office_held_at_date: int = 30
    bill_stage_date: int = 28
    count: int = 24
    who_spoke: int = 40
    debate_of_division: int = 35

    def total(self) -> int:
        return sum(getattr(self, f) for f in self.__dataclass_fields__)


class _Ids:
    """Sequential ids with a per-prefix counter, so ids stay stable when a template's budget changes."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, prefix: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}{self._counts[prefix]:04d}"


async def _rows(conn: AsyncConnection, sql: str, params: dict | None = None) -> list[dict]:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, params or {})
        return list(await cur.fetchall())


# When set, generation is restricted to these windows.
ONLY_WINDOWS: frozenset[str] | None = None


def _restrict(rows: list[dict]) -> list[dict]:
    """Rows from the windows generation is restricted to."""
    if ONLY_WINDOWS is None:
        return rows
    return [r for r in rows if "window_id" not in r or r.get("window_id") in ONLY_WINDOWS]


def _stratify(rows: list[dict], budget: int, rng: random.Random, key: str = "window_id") -> list[dict]:
    """Take `budget` rows spread as evenly as the data allows over the values of `key`."""
    rows = _restrict(rows)
    groups: dict[Any, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups, key=lambda k: (k is None, str(k)))
    taken: list[dict] = []
    while len(taken) < budget and any(groups[k] for k in order):
        for k in order:
            if groups[k] and len(taken) < budget:
                taken.append(groups[k].pop())
    return taken


# --- templates --------------------------------------------------------------------------------------- Each returns
# items whose `expected_structured_facts` carry the exact values an answer must reproduce.


_DIVISION_BASE = """
    SELECT d.id, d.house, d.division_date, d.number, d.title, d.motion_text, d.ayes, d.noes, d.result,
           d.is_remote, d.is_deferred, d.tally_reconciled, d.ayes_listed, d.noes_listed, d.debate_id,
           d.government_won, deb.window_id, deb.heading, deb.section
      FROM division d
      LEFT JOIN debate deb ON deb.id = d.debate_id
     WHERE d.title IS NOT NULL AND length(d.title) > 8
       AND d.ayes IS NOT NULL AND d.noes IS NOT NULL
"""


async def division_result(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """Was it agreed to? The one division fact a person is most likely to ask for."""
    rows = _stratify(await _rows(conn, _DIVISION_BASE), budget, rng)
    phrasings = [
        "Did the {house} agree to {title} in the division on {date}?",
        "What was the outcome of the {house} division on {title} on {date}?",
        "On {date}, was the motion on {title} carried in the {house}?",
        "Was {title} passed or rejected when the {house} divided on {date}?",
    ]
    items = []
    for row in rows:
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(house=row["house"], title=row["title"].strip(), date=_pretty_date(row["division_date"]))
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="division_result",
                criticality="critical",  # a wrong division outcome is the definition of a harmful answer
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                section=row["section"] or "",
                date=str(row["division_date"]),
                subcategory="remote" if row["is_remote"] else ("deferred" if row["is_deferred"] else "ordinary"),
                expected_structured_facts={
                    "division_id": row["id"],
                    "house": row["house"],
                    "division_number": row["number"],
                    "date": str(row["division_date"]),
                    "result": row["result"],
                    "ayes": row["ayes"],
                    "noes": row["noes"],
                },
                nuggets=(Nugget(f"The division was {row['result']} ({row['ayes']} to {row['noes']}).", True),),
                forbidden_claims=(
                    f"That the division was {'negatived' if row['result'] == 'agreed' else 'agreed'}.",
                ),
                notes="Official tellers' figures. Never count the names in division_vote to answer this.",
            )
        )
    return items


async def division_tally(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """How many voted. Deliberately drawn towards the divisions that do NOT reconcile."""
    rows = await _rows(conn, _DIVISION_BASE + " AND d.ayes_listed IS NOT NULL")
    unreconciled = [r for r in rows if r["tally_reconciled"] is False]
    reconciled = [r for r in rows if r["tally_reconciled"] is not False]
    chosen = _stratify(unreconciled, budget // 2, rng) + _stratify(reconciled, budget - budget // 2, rng)
    phrasings = [
        "How many members voted in favour of {title} in the {house} on {date}?",
        "What were the Ayes and Noes when the {house} divided on {title} on {date}?",
        "How many voted against {title} on {date}?",
        "Give the voting figures for the {house} division on {title} on {date}.",
    ]
    items = []
    for row in chosen:
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(house=row["house"], title=row["title"].strip(), date=_pretty_date(row["division_date"]))
        mismatch = row["tally_reconciled"] is False
        nuggets = [Nugget(f"The Ayes were {row['ayes']} and the Noes {row['noes']}.", True)]
        if mismatch:
            nuggets.append(
                Nugget("The per-member lists are recorded separately from the tellers' count and differ here.", False)
            )
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="division_tally",
                criticality="critical",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                section=row["section"] or "",
                date=str(row["division_date"]),
                subcategory="unreconciled" if mismatch else "reconciled",
                expected_structured_facts={
                    "division_id": row["id"],
                    "ayes": row["ayes"],
                    "noes": row["noes"],
                    "ayes_listed": row["ayes_listed"],
                    "noes_listed": row["noes_listed"],
                    "tally_reconciled": row["tally_reconciled"],
                    "must_cite": "official tellers' figures (division.ayes / division.noes)",
                },
                nuggets=tuple(nuggets),
                forbidden_claims=(
                    f"That {row['ayes_listed']} members voted Aye (that is the length of the name list, "
                    f"not the tellers' figure)."
                    if mismatch
                    else "Any figure other than the official Ayes and Noes.",
                ),
                notes="I-8: cite the official tally; the name list may differ." if mismatch else "",
            )
        )
    return items


async def division_vote(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """How one named member voted. Over-samples proxy and remote votes: 2020's remote divisions are the only place in
    the corpus where "who voted" and "who was present" genuinely come apart.
    """
    rows = await _rows(
        conn,
        """
        SELECT v.division_id, v.mnis_id, v.lobby, v.member_name, v.party, v.is_teller, v.proxy_name,
               d.house, d.division_date, d.title, d.number, d.is_remote, deb.window_id
          FROM division_vote v
          JOIN division d ON d.id = v.division_id
          LEFT JOIN debate deb ON deb.id = d.debate_id
         WHERE v.lobby IN ('aye', 'no')
           AND v.member_name IS NOT NULL
           AND d.title IS NOT NULL AND length(d.title) > 8
        """,
    )
    proxies = [r for r in rows if r["proxy_name"]]
    ordinary = [r for r in rows if not r["proxy_name"]]
    chosen = _stratify(proxies, min(budget // 3, len(proxies)), rng)
    chosen += _stratify(ordinary, budget - len(chosen), rng)
    phrasings = [
        "How did {member} vote on {title} on {date}?",
        "Which lobby did {member} go through in the division on {title} on {date}?",
        "Did {member} support {title} when it was divided on, on {date}?",
        "Was {member} an Aye or a No on {title} on {date}?",
    ]
    items = []
    for row in chosen:
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(
            member=row["member_name"], title=row["title"].strip(), date=_pretty_date(row["division_date"])
        )
        facts = {
            "division_id": row["division_id"],
            "mnis_id": row["mnis_id"],
            "member": row["member_name"],
            "lobby": row["lobby"],
            "party_at_vote": row["party"],
            "is_teller": row["is_teller"],
        }
        nuggets = [Nugget(f"{row['member_name']} voted {row['lobby'].capitalize()}.", True)]
        if row["proxy_name"]:
            facts["proxy_name"] = row["proxy_name"]
            nuggets.append(Nugget(f"The vote was cast by proxy by {row['proxy_name']}.", True))
        if row["is_remote"]:
            facts["is_remote"] = True
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="division_vote",
                criticality="critical",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                date=str(row["division_date"]),
                subcategory="proxy" if row["proxy_name"] else ("remote" if row["is_remote"] else "ordinary"),
                expected_structured_facts=facts,
                nuggets=tuple(nuggets),
                forbidden_claims=(
                    f"That {row['member_name']} voted {'No' if row['lobby'] == 'aye' else 'Aye'}.",
                ),
            )
        )
    return items


async def party_at_date(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """Which party someone sat for on a date, taken from how they are recorded in a division they voted in."""
    rows = await _rows(
        conn,
        """
        SELECT DISTINCT ON (v.mnis_id, d.division_date)
               v.mnis_id, v.member_name, v.party, d.division_date, d.house, deb.window_id
          FROM division_vote v
          JOIN division d ON d.id = v.division_id
          LEFT JOIN debate deb ON deb.id = d.debate_id
         WHERE v.party IS NOT NULL AND v.party <> '' AND v.member_name IS NOT NULL
         ORDER BY v.mnis_id, d.division_date
        """,
    )
    phrasings = [
        "Which party did {member} sit for on {date}?",
        "What was {member}'s party on {date}?",
        "On {date}, which party was {member} a member of?",
    ]
    items = []
    for row in _stratify(rows, budget, rng):
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(member=row["member_name"], date=_pretty_date(row["division_date"]))
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="party_at_date",
                criticality="major",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                date=str(row["division_date"]),
                expected_structured_facts={
                    "mnis_id": row["mnis_id"],
                    "member": row["member_name"],
                    "date": str(row["division_date"]),
                    "party": row["party"],
                    "basis": "the party recorded against this member in a division list on that date",
                },
                nuggets=(Nugget(f"{row['member_name']} sat for {row['party']} on that date.", True),),
                forbidden_claims=("The member's present-day party, if it differs from the one held on that date.",),
            )
        )
    return items


async def office_held_at_date(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """Who held an office on a date -- restricted to offices with exactly one holder on that date."""
    rows = await _rows(
        conn,
        """
        WITH windowed AS (
            SELECT w.window_id, w.start_date, w.end_date FROM corpus_window w
        ),
        holders AS (
            SELECT o.office_key, o.office, o.mnis_id, m.name, w.window_id,
                   GREATEST(o.start_date, w.start_date) AS probe
              FROM member_office o
              JOIN member_id_map m ON m.mnis_id = o.mnis_id
              JOIN windowed w
                ON o.start_date <= w.end_date
               AND (o.end_date IS NULL OR o.end_date >= w.start_date)
             WHERE o.source = 'mnis_biography' AND o.start_date IS NOT NULL
        )
        SELECT office_key, office, mnis_id, name, window_id, probe
          FROM holders h
         WHERE NOT EXISTS (
               SELECT 1 FROM member_office o2
                WHERE o2.office_key = h.office_key
                  AND o2.mnis_id <> h.mnis_id
                  AND o2.start_date <= h.probe
                  AND (o2.end_date IS NULL OR o2.end_date >= h.probe))
        """,
    )
    phrasings = [
        "Who was the {office} on {date}?",
        "Who held the post of {office} on {date}?",
        "On {date}, who was serving as {office}?",
    ]
    items = []
    for row in _stratify(rows, budget, rng):
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(office=row["office"], date=_pretty_date(row["probe"]))
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="office_held_at_date",
                criticality="major",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                date=str(row["probe"]),
                expected_structured_facts={
                    "office": row["office"],
                    "office_key": row["office_key"],
                    "date": str(row["probe"]),
                    "mnis_id": row["mnis_id"],
                    "holder": row["name"],
                },
                nuggets=(Nugget(f"{row['name']} held that office on {_pretty_date(row['probe'])}.", True),),
                forbidden_claims=("The holder of the office at any other date, stated as the holder on this one.",),
            )
        )
    return items


# How a stage-date question is worded when the stage belongs to a House.
_STAGE_PHRASINGS = (
    "When did the {title} have its {stage} in the {house}?",
    "On what date did the {house} take the {stage} of the {title}?",
    "What was the date of the {stage} of the {title} in the {house}?",
)

# ...and when it belongs to neither.
_STAGE_PHRASINGS_HOUSELESS = (
    "When did the {title} have its {stage}?",
    "On what date was the {stage} of the {title}?",
    "What was the date of the {stage} of the {title}?",
)


def _stage_question(title: str, stage: str, house: str | None, rng: random.Random) -> str:
    """One stage-date question. Never names a House the stage does not have."""
    choices = _STAGE_PHRASINGS if house else _STAGE_PHRASINGS_HOUSELESS
    return choices[rng.randrange(len(choices))].format(title=title, stage=stage, house=house or "")


async def bill_stage_date(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """When a bill reached a stage. Taken from bill_stage_sitting, which is the sitting actually held."""
    rows = await _rows(
        conn,
        """
        -- Grouped by (bill, stage name, house) and not by bill_stage.id, because that is the granularity
        -- the QUESTION has: "when did the Bill have its Second Reading in the Lords" does not name a
        -- bill_stage row. HAVING count(DISTINCT s.id) = 1 then throws away the bills where that phrase
        -- picks out more than one stage (a bill carried over between sessions has two), which would make
        -- the question ambiguous rather than exact.
        SELECT b.id AS bill_id, b.title, s.stage_name, s.house, MIN(t.date) AS first_sitting,
               MAX(t.date) AS last_sitting, count(*) AS sittings
          FROM bill_stage_sitting t
          JOIN bill_stage s ON s.id = t.bill_stage_id
          JOIN bill b ON b.id = t.bill_id
         WHERE b.title IS NOT NULL AND s.stage_name IS NOT NULL
         GROUP BY b.id, b.title, s.stage_name, s.house
        HAVING count(DISTINCT s.id) = 1
        """,
    )
    items = []
    for row in _stratify(rows, budget, rng, key="house"):
        question = _stage_question(row["title"], row["stage_name"], row["house"], rng)
        facts = {
            "bill_id": row["bill_id"],
            "bill": row["title"],
            "stage": row["stage_name"],
            "house": row["house"],
            "first_sitting": str(row["first_sitting"]),
            "sittings": row["sittings"],
        }
        if row["sittings"] > 1:
            facts["last_sitting"] = str(row["last_sitting"])
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="bill_stage_date",
                criticality="major",
                expected_behaviour="answer",
                provenance="db_template",
                house=row["house"] or "",
                date=str(row["first_sitting"]),
                subcategory="multi_sitting" if row["sittings"] > 1 else "single_sitting",
                expected_structured_facts=facts,
                nuggets=(
                    Nugget(f"It was on {_pretty_date(row['first_sitting'])}.", True),
                )
                if row["sittings"] == 1
                else (
                    Nugget(f"The stage began on {_pretty_date(row['first_sitting'])}.", True),
                    Nugget(f"It ran over {row['sittings']} sittings, ending {_pretty_date(row['last_sitting'])}.", False),
                ),
                notes="This stage sat more than once; a single date is an incomplete answer."
                if row["sittings"] > 1
                else "",
            )
        )
    return items


async def counts(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """Counting questions, which are where a retrieval-only system most reliably guesses."""
    per_month = await _rows(
        conn,
        """
        -- The count is over ALL divisions of that house and month. Grouping by window as well would split
        -- one month in two whenever a division has no linked debate (and so no window), and the gold
        -- answer would then be a fraction of the true count. The window here only labels the item.
        WITH months AS (
            SELECT d.house, date_trunc('month', d.division_date)::date AS month, count(*) AS n
              FROM division d WHERE d.division_date IS NOT NULL GROUP BY 1, 2
        )
        SELECT m.house, m.month, m.n,
               (SELECT w.window_id FROM corpus_window w
                 WHERE m.month <= w.end_date AND m.month >= date_trunc('month', w.start_date)::date
                 ORDER BY w.start_date LIMIT 1) AS window_id
          FROM months m
        """,
    )
    per_day = await _rows(
        conn,
        """
        SELECT window_id, house, date, count(*) AS n
          FROM debate WHERE date IS NOT NULL GROUP BY 1, 2, 3 HAVING count(*) >= 5
        """,
    )
    per_speaker = await _rows(
        conn,
        """
        SELECT s.window_id, m.name AS speaker_name, s.mnis_id, count(*) AS n,
               w.start_date, w.end_date
          FROM speech s
          JOIN member_id_map m ON m.mnis_id = s.mnis_id
          JOIN corpus_window w ON w.window_id = s.window_id
         WHERE s.mnis_id IS NOT NULL AND NOT s.is_boilerplate AND m.name IS NOT NULL
         GROUP BY 1, 2, 3, 5, 6 HAVING count(*) BETWEEN 5 AND 400
        """,
    )
    items: list[GoldItem] = []
    third = max(1, budget // 3)

    for row in _stratify(per_month, third, rng):
        month = f"{row['month']:%B %Y}"
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=f"How many divisions did the {row['house']} hold in {month}?",
                category="count",
                criticality="major",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                date=str(row["month"]),
                subcategory="divisions_per_month",
                expected_structured_facts={"house": row["house"], "month": str(row["month"]), "divisions": row["n"]},
                nuggets=(Nugget(f"There were {row['n']}.", True),),
                forbidden_claims=("A count inferred from however many divisions happen to be retrieved.",),
            )
        )
    for row in _stratify(per_day, third, rng):
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=f"How many separate debates were there in the {row['house']} on {_pretty_date(row['date'])}?",
                category="count",
                criticality="minor",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                date=str(row["date"]),
                subcategory="debates_per_day",
                expected_structured_facts={"house": row["house"], "date": str(row["date"]), "debates": row["n"]},
                nuggets=(Nugget(f"There were {row['n']}.", True),),
            )
        )
    for row in _stratify(per_speaker, budget - len(items), rng):
        span = f"between {_pretty_date(row['start_date'])} and {_pretty_date(row['end_date'])}"
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=f"How many times did {row['speaker_name']} speak {span}?",
                category="count",
                criticality="minor",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                date=str(row["start_date"]),
                subcategory="speeches_per_member",
                expected_structured_facts={
                    "mnis_id": row["mnis_id"],
                    "member": row["speaker_name"],
                    "window": row["window_id"],
                    "from": str(row["start_date"]),
                    "to": str(row["end_date"]),
                    "contributions": row["n"],
                    "excludes": "boilerplate",
                },
                nuggets=(Nugget(f"{row['n']} contributions, excluding procedural boilerplate.", True),),
                notes="The corpus covers only these dates; a count without that caveat overstates the record.",
            )
        )
    return items


async def who_spoke(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """Who opened a debate. Retrieval-testable: the opening speech is the labelled relevant speech."""
    rows = await _rows(
        conn,
        """
        -- `first` picks the genuinely first non-boilerplate speech of the debate, with NO other filter on
        -- it. Filtering inside that DISTINCT ON -- for a member row, or for a long enough first sentence --
        -- silently redefines "opened the debate" as "opened it, ignoring the speeches I could not use",
        -- and produces confidently wrong gold answers. The usability conditions are applied afterwards, as
        -- a reason to DROP the debate rather than to move the answer.
        WITH first AS (
            SELECT DISTINCT ON (d.id)
                   d.id AS debate_id, d.heading, d.date, d.house, d.section, d.window_id,
                   s.id AS speech_id, s.mnis_id, s.speaker_party
              FROM debate d
              JOIN speech s ON s.debate_id = d.id AND NOT s.is_boilerplate
             WHERE d.heading IS NOT NULL AND length(d.heading) > 6
             ORDER BY d.id, s.seq
        )
        SELECT f.*, m.name AS speaker_name, c.body
          FROM first f
          JOIN member_id_map m ON m.mnis_id = f.mnis_id
          JOIN chunk c ON c.speech_id = f.speech_id AND c.sent_idx = 0
         WHERE m.name IS NOT NULL AND length(c.body) >= %(min_chars)s
        """,
        {"min_chars": MIN_EVIDENCE_CHARS},
    )
    phrasings = [
        "Who opened the debate on {heading} on {date}?",
        "Which member spoke first in the {house} debate on {heading} on {date}?",
        "Who led off on {heading} in the {house} on {date}?",
    ]
    items = []
    for row in _stratify(rows, budget, rng):
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(heading=row["heading"], date=_pretty_date(row["date"]), house=row["house"])
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="who_spoke",
                criticality="major",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"] or "",
                section=row["section"] or "",
                date=str(row["date"]),
                evidence=(Evidence(row["speech_id"], row["body"].strip()),),
                expected_structured_facts={
                    "debate_id": row["debate_id"],
                    "heading": row["heading"],
                    "opened_by": row["speaker_name"],
                    "mnis_id": row["mnis_id"],
                    "party": row["speaker_party"],
                },
                nuggets=(Nugget(f"{row['speaker_name']} opened it.", True),),
                forbidden_claims=("Any member who did not in fact speak first in that debate.",),
            )
        )
    return items


async def debate_of_division(conn: AsyncConnection, budget: int, rng: random.Random, ids: _Ids) -> list[GoldItem]:
    """"Find where division N was debated" -- the join between the votes APIs and Hansard, asked as a question."""
    rows = await _rows(
        conn,
        """
        SELECT d.id AS division_id, d.house, d.division_date, d.number, d.title, d.result, d.ayes, d.noes,
               deb.id AS debate_id, deb.heading, deb.window_id, deb.section,
               (SELECT count(*) FROM speech s WHERE s.debate_id = deb.id AND NOT s.is_boilerplate) AS speeches
          FROM division d
          JOIN debate deb ON deb.id = d.debate_id
         WHERE d.title IS NOT NULL AND deb.heading IS NOT NULL
        """,
    )
    rows = [r for r in rows if r["speeches"] >= 3]
    phrasings = [
        "Which debate was division {number} in the {house} on {date} taken during?",
        "What was being debated when the {house} divided on {title} on {date}?",
        "Find the debate that led to the {house} division on {title} on {date}.",
    ]
    items = []
    for row in _stratify(rows, budget, rng):
        evidence = await _rows(
            conn,
            """
            -- The longest contributions, not the first: the opening lines of a debate are often the chair
            -- grouping amendments ("With this it will be convenient to discuss the following:"), which is
            -- true of the debate but answers nothing about it.
            SELECT s.id AS speech_id, c.body
              FROM speech s
              JOIN chunk c ON c.speech_id = s.id AND c.sent_idx = 0
             WHERE s.debate_id = %(debate_id)s AND NOT s.is_boilerplate
               AND length(c.body) >= %(min_chars)s AND length(s.text) >= 600
             ORDER BY length(s.text) DESC LIMIT 3
            """,
            {"debate_id": row["debate_id"], "min_chars": MIN_EVIDENCE_CHARS},
        )
        if not evidence:
            continue
        template = phrasings[rng.randrange(len(phrasings))]
        question = template.format(
            number=row["number"], house=row["house"], date=_pretty_date(row["division_date"]), title=row["title"].strip()
        )
        items.append(
            GoldItem(
                id=ids.next("db"),
                question=question,
                category="multi_speech",
                criticality="major",
                expected_behaviour="answer",
                provenance="db_template",
                window=row["window_id"] or "",
                house=row["house"],
                section=row["section"] or "",
                date=str(row["division_date"]),
                subcategory="debate_of_division",
                evidence=tuple(Evidence(e["speech_id"], e["body"].strip()) for e in evidence),
                expected_structured_facts={
                    "division_id": row["division_id"],
                    "debate_id": row["debate_id"],
                    "heading": row["heading"],
                    "result": row["result"],
                },
                nuggets=(Nugget(f"The debate was {row['heading']}.", True),),
                notes="Linked by the DebateSectionExtId Hansard states for the division.",
            )
        )
    return items


TEMPLATES = {
    "division_result": division_result,
    "division_tally": division_tally,
    "division_vote": division_vote,
    "party_at_date": party_at_date,
    "office_held_at_date": office_held_at_date,
    "bill_stage_date": bill_stage_date,
    "count": counts,
    "who_spoke": who_spoke,
    "debate_of_division": debate_of_division,
}


async def generate(conn: AsyncConnection, budget: Budget | None = None, seed: int = 20260924) -> list[GoldItem]:
    """Every database-derived item, deterministic given the seed and the corpus."""
    budget = budget or Budget()
    ids = _Ids()
    items: list[GoldItem] = []
    for name, template in TEMPLATES.items():
        rng = random.Random(f"{seed}:{name}")  # per template, so changing one budget cannot reshuffle the others
        produced = await template(conn, getattr(budget, name), rng, ids)
        for item in produced:
            item.split = _split_for(item.window)
        items += produced
    return items
