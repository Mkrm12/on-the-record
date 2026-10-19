"""The automatic validators. Nothing reaches the maintainer's review queue until every one of these has run."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from ingest.text import clean_text
from procedure import opener as OP

from goldset.schema import (
    GoldItem,
    content_tokens,
    jaccard,
    longest_common_run,
    normalise_question,
    to_dict,
    validate as validate_schema,
)

# `speech.speaker_role` values that preside rather than speak to the business.
PRESIDING_ROLES = OP.PRESIDING_ROLES  # one definition (S05): migration 029 and the gate make stored roles exact
# The Chair's printed forms, for presiding Chairmen tagged `office_holder`.
CHAIR_PRINTED = r"^The (?:(?:First |Second )?Deputy )?Chairman(?: of Ways and Means)?(?: \(|$)|^The (?:Temporary )?Chair(?: \(|$)"

MAX_CONSECUTIVE_RUN = 4

# And it may not share more than this share of content vocabulary with the quote.
DEFAULT_JACCARD_MAX = 0.35

# Two questions closer than this on token-Jaccard are treated as near-duplicates.
NEAR_DUPLICATE_JACCARD = 0.8

SEVERITY = ("error", "warning")


@dataclass(frozen=True)
class Finding:
    item_id: str
    check: str
    severity: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


def _normalise_text(text: str) -> str:
    """Fold the differences that are not differences: markup, smart quotes, dashes, NBSP, whitespace."""
    text = unicodedata.normalize("NFKC", clean_text(text))
    for fancy, plain in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                         ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")):  # fmt: skip
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip().lower()


# --- 1. quotes exist, verbatim, in the speech they are attributed to ---------------------------------


async def check_quotes(conn: AsyncConnection, items: list[GoldItem]) -> list[Finding]:
    """Every evidence quote must appear in its cited speech, and every cited speech must exist."""
    wanted = sorted({e.speech_id for i in items for e in i.evidence})
    if not wanted:
        return []
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT id, text, window_id, is_boilerplate FROM speech WHERE id = ANY(%(ids)s)", {"ids": wanted}
        )
        speeches = {r["id"]: r for r in await cur.fetchall()}

    findings = []
    for item in items:
        for e in item.evidence:
            speech = speeches.get(e.speech_id)
            if speech is None:
                findings.append(Finding(item.id, "quote", "error", f"speech {e.speech_id} is not in the corpus"))
                continue
            if _normalise_text(e.quote) not in _normalise_text(speech["text"] or ""):
                findings.append(
                    Finding(item.id, "quote", "error", f"the quote is not verbatim in speech {e.speech_id}")
                )
            if item.window and speech["window_id"] and speech["window_id"] != item.window:
                findings.append(
                    Finding(
                        item.id,
                        "quote",
                        "error",
                        f"item is tagged {item.window} but speech {e.speech_id} is in {speech['window_id']}",
                    )
                )
            if speech["is_boilerplate"] and item.category != "boilerplate_trap":
                findings.append(
                    Finding(item.id, "quote", "warning", f"speech {e.speech_id} is flagged boilerplate")
                )
    return findings


# --- 2. every database-derived fact still matches the database ---------------------------------------


async def _fact_check_one(conn: AsyncConnection, item: GoldItem) -> list[str]:  # noqa: C901 - one branch per category
    """Re-ask the database the question, and compare. Returns a list of disagreements."""
    facts = item.expected_structured_facts
    problems: list[str] = []

    async def one(sql: str, params: dict) -> dict | None:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()

    def same(name: str, expected: Any, found: Any) -> None:
        if str(expected) != str(found):
            problems.append(f"{name}: gold says {expected!r}, the database says {found!r}")

    if item.category in ("division_result", "division_tally"):
        row = await one(
            "SELECT result, ayes, noes, ayes_listed, noes_listed, tally_reconciled FROM division WHERE id = %(id)s",
            {"id": facts.get("division_id")},
        )
        if row is None:
            return [f"division {facts.get('division_id')} no longer exists"]
        for key in ("result", "ayes", "noes", "ayes_listed", "noes_listed", "tally_reconciled"):
            if key in facts:
                same(key, facts[key], row[key])

    elif item.category == "division_vote":
        row = await one(
            "SELECT lobby, party, is_teller, proxy_name FROM division_vote "
            "WHERE division_id = %(d)s AND mnis_id = %(m)s",
            {"d": facts.get("division_id"), "m": facts.get("mnis_id")},
        )
        if row is None:
            return [f"no vote by member {facts.get('mnis_id')} in division {facts.get('division_id')}"]
        same("lobby", facts.get("lobby"), row["lobby"])
        if facts.get("proxy_name"):
            same("proxy_name", facts["proxy_name"], row["proxy_name"])

    elif item.category == "party_at_date":
        row = await one(
            "SELECT v.party FROM division_vote v JOIN division d ON d.id = v.division_id "
            "WHERE v.mnis_id = %(m)s AND d.division_date = %(date)s LIMIT 1",
            {"m": facts.get("mnis_id"), "date": facts.get("date")},
        )
        if row is None:
            return [f"member {facts.get('mnis_id')} has no recorded vote on {facts.get('date')}"]
        same("party", facts.get("party"), row["party"])

    elif item.category == "office_held_at_date":
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                "SELECT DISTINCT mnis_id FROM member_office WHERE office_key = %(k)s "
                "AND start_date <= %(d)s AND (end_date IS NULL OR end_date >= %(d)s)",
                {"k": facts.get("office_key"), "d": facts.get("date")},
            )
            holders = [r["mnis_id"] for r in await cur.fetchall()]
        if holders != [facts.get("mnis_id")]:
            problems.append(f"holders of {facts.get('office_key')!r} on {facts.get('date')} are now {holders}")

    elif item.category == "bill_stage_date":
        row = await one(
            "SELECT MIN(t.date) AS first_sitting, count(*) AS sittings FROM bill_stage_sitting t "
            "JOIN bill_stage s ON s.id = t.bill_stage_id "
            "WHERE t.bill_id = %(b)s AND s.stage_name = %(stage)s AND s.house IS NOT DISTINCT FROM %(house)s",
            {"b": facts.get("bill_id"), "stage": facts.get("stage"), "house": facts.get("house")},
        )
        if row is None or row["sittings"] == 0:
            return [f"bill {facts.get('bill_id')} has no {facts.get('stage')!r} sitting"]
        same("first_sitting", facts.get("first_sitting"), row["first_sitting"])

    elif item.category == "count":
        if item.subcategory == "divisions_per_month":
            row = await one(
                "SELECT count(*) AS n FROM division WHERE house = %(h)s "
                "AND date_trunc('month', division_date)::date = %(m)s",
                {"h": facts.get("house"), "m": facts.get("month")},
            )
            same("divisions", facts.get("divisions"), row["n"] if row else None)
        elif item.subcategory == "debates_per_day":
            row = await one(
                "SELECT count(*) AS n FROM debate WHERE house = %(h)s AND date = %(d)s",
                {"h": facts.get("house"), "d": facts.get("date")},
            )
            same("debates", facts.get("debates"), row["n"] if row else None)
        elif item.subcategory == "speeches_per_member":
            row = await one(
                "SELECT count(*) AS n FROM speech WHERE mnis_id = %(m)s AND window_id = %(w)s AND NOT is_boilerplate",
                {"m": facts.get("mnis_id"), "w": facts.get("window")},
            )
            same("contributions", facts.get("contributions"), row["n"] if row else None)

    elif item.category == "who_spoke":
        # The opener is the first speech that is neither boilerplate nor the Chair's.
        row = await one(
            "SELECT s.id, s.mnis_id, s.speaker_person, m.name FROM speech s JOIN member_id_map m ON m.mnis_id = "
            "s.mnis_id WHERE s.debate_id = %(d)s AND NOT s.is_boilerplate AND coalesce(s.speaker_role, '') <> ALL "
            "(%(chair)s) AND coalesce(s.speaker_name, '') !~ %(printed)s ORDER BY s.seq LIMIT 1",
            {"d": facts.get("debate_id"), "chair": list(PRESIDING_ROLES), "printed": CHAIR_PRINTED},
        )
        if row is None:
            return [f"debate {facts.get('debate_id')} has no non-boilerplate speech"]
        if "mnis_id" in facts:
            same("mnis_id", facts["mnis_id"], row["mnis_id"])
        # The generator names the opener as the record printed them (speaker_person); their current name (member_id_map)
        # is the same member, so either is right, and nothing else is.
        if str(facts.get("opened_by")) not in {str(row["speaker_person"] or ""), str(row["name"])}:
            problems.append(f"opened_by: gold says {facts.get('opened_by')!r}, the record printed "
                            f"{row['speaker_person']!r} (now {row['name']!r})")

    return problems


async def check_facts(conn: AsyncConnection, items: list[GoldItem]) -> list[Finding]:
    """Every expected_structured_fact, re-derived from the database it came from."""
    findings = []
    for item in items:
        if not item.expected_structured_facts:
            continue
        try:
            for problem in await _fact_check_one(conn, item):
                findings.append(Finding(item.id, "fact", "error", problem))
        except Exception as exc:  # noqa: BLE001 - a broken check must name its item, not kill the run
            findings.append(Finding(item.id, "fact", "error", f"fact check failed: {type(exc).__name__}: {exc}"))
    return findings


# --- 3. the anti-overlap filter, measured -------------------------------------------------------------


def overlap_report(items: list[GoldItem]) -> dict[str, Any]:
    """The distribution of question-to-quote overlap, so the threshold is chosen from data."""
    runs, jaccards = [], []
    for item in items:
        for e in item.evidence:
            runs.append(longest_common_run(item.question, e.quote))
            jaccards.append(jaccard(item.question, e.quote))
    if not runs:
        return {"n": 0}

    def percentile(values: list[float], p: float) -> float:
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 4)

    return {
        "n": len(runs),
        "run": {
            "max": max(runs),
            "p50": percentile([float(r) for r in runs], 0.5),
            "p90": percentile([float(r) for r in runs], 0.9),
            "histogram": dict(sorted(Counter(runs).items())),
        },
        "jaccard": {
            "max": round(max(jaccards), 4),
            "p50": percentile(jaccards, 0.5),
            "p90": percentile(jaccards, 0.9),
            "p95": percentile(jaccards, 0.95),
        },
    }


def choose_jaccard_threshold(items: list[GoldItem], quantile: float = 0.9) -> float:
    """A threshold taken from the observed distribution of NON-paraphrase items."""
    lexical = [
        jaccard(i.question, e.quote)
        for i in items
        if i.category in ("exact_term", "numeric", "quote_verification")
        for e in i.evidence
    ]
    if len(lexical) < 10:
        return DEFAULT_JACCARD_MAX
    ordered = sorted(lexical)
    measured = ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]
    # Never looser than the default: a corpus where everything overlaps must not raise the bar for everyone.
    return round(min(DEFAULT_JACCARD_MAX, max(0.15, measured)), 4)


def check_overlap(items: list[GoldItem], jaccard_max: float | None = None) -> list[Finding]:
    """Paraphrase items must not copy their answer. Others are only warned about."""
    threshold = DEFAULT_JACCARD_MAX if jaccard_max is None else jaccard_max
    findings = []
    for item in items:
        for e in item.evidence:
            run, overlap = longest_common_run(item.question, e.quote), jaccard(item.question, e.quote)
            if item.category == "paraphrase":
                if run > MAX_CONSECUTIVE_RUN:
                    findings.append(
                        Finding(item.id, "overlap", "error",
                                f"copies {run} consecutive words from {e.speech_id} (limit {MAX_CONSECUTIVE_RUN})")  # fmt: skip
                    )
                if overlap > threshold:
                    findings.append(
                        Finding(item.id, "overlap", "error",
                                f"token-Jaccard {overlap:.2f} with {e.speech_id} exceeds {threshold:.2f}")  # fmt: skip
                    )
            elif run >= 8:
                findings.append(
                    Finding(item.id, "overlap", "warning", f"copies {run} consecutive words from {e.speech_id}")
                )
    return findings


# --- 4. duplicates and near-duplicates ----------------------------------------------------------------


def _answer_key(item: GoldItem) -> str:
    """What an item is actually asking for, as opposed to how it asks."""
    facts = item.expected_structured_facts
    return repr(sorted((k, str(v)) for k, v in facts.items())) + repr(sorted(item.relevant_speech_ids))


def check_duplicates(items: list[GoldItem]) -> list[Finding]:
    """Exact duplicates are caught by the schema; this finds the near ones, which are worse."""
    findings: list[Finding] = []
    by_token: dict[str, list[GoldItem]] = defaultdict(list)
    for item in items:
        for token in content_tokens(item.question):
            by_token[token].append(item)

    compared: set[tuple[str, str]] = set()
    for bucket in by_token.values():
        if len(bucket) > 60:  # a stopword-like token: comparing everything with everything helps nobody
            continue
        for i, a in enumerate(bucket):
            for b in bucket[i + 1 :]:
                pair = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                if pair in compared:
                    continue
                compared.add(pair)
                if normalise_question(a.question) == normalise_question(b.question):
                    continue  # the schema already reports this one as an outright duplicate
                if jaccard(a.question, b.question) < NEAR_DUPLICATE_JACCARD:
                    continue
                if _answer_key(a) != _answer_key(b):
                    findings.append(
                        Finding(pair[1], "discriminating_pair", "warning",
                                f"reads almost like {pair[0]} but has a different answer: a near-miss pair, kept")  # fmt: skip
                    )
                else:
                    findings.append(
                        Finding(pair[1], "duplicate", "error", f"same wording AND same answer as {pair[0]}")
                    )
    return findings


# --- 5. abstain items really are absent ---------------------------------------------------------------


# An anchor term must be rarer than this share of chunks.
MAX_ANCHOR_DF_RATIO = 0.005

# And absence needs at least this many independent anchors to be testable at all.
MIN_ANCHORS = 2

_YEAR = re.compile(r"\b(19|20)\d{2}\b")


async def _anchors(conn: AsyncConnection, question: str, total: int) -> list[str]:
    """The question's rarest usable terms, as LEXEMES."""
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            """
            SELECT d.lexeme, d.df
              FROM unnest(tsvector_to_array(to_tsvector('english', %(q)s))) AS l(lexeme)
              JOIN lexeme_df d ON d.lexeme = l.lexeme
             WHERE d.df <= %(max)s
             ORDER BY d.df ASC LIMIT 4
            """,
            {"q": question, "max": max(1, int(total * MAX_ANCHOR_DF_RATIO))},
        )
        return [r["lexeme"] for r in await cur.fetchall()]


async def check_absence(
    conn: AsyncConnection, items: list[GoldItem]
) -> tuple[list[Finding], dict[str, list[dict]]]:
    """Establish, as far as a machine honestly can, that an abstain item has no answer in the record."""
    findings: list[Finding] = []
    candidates: dict[str, list[dict]] = {}

    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT window_id, start_date, end_date FROM corpus_window")
        windows = [(r["window_id"], r["start_date"], r["end_date"]) for r in await cur.fetchall()]
        await cur.execute("SELECT count(*) AS n FROM chunk WHERE NOT is_boilerplate")
        total = (await cur.fetchone())["n"] or 1
    covered_years = {y for _, start, end in windows for y in range(start.year, end.year + 1)}

    for item in items:
        if item.expected_behaviour == "abstain_out_of_window":
            years = {int(m.group(0)) for m in _YEAR.finditer(item.question)}
            inside = sorted(years & covered_years)
            if inside:
                findings.append(
                    Finding(item.id, "absence", "error",
                            f"the question names {inside}, which the corpus covers: not out of window")  # fmt: skip
                )
            elif not years:
                findings.append(
                    Finding(item.id, "absence", "warning",
                            "no year in the question, so being out of window cannot be checked; owner to confirm")  # fmt: skip
                )
            continue

        if item.expected_behaviour != "abstain_not_in_record":
            continue

        anchors = await _anchors(conn, item.question, total)
        if len(anchors) < MIN_ANCHORS:
            findings.append(
                Finding(item.id, "absence", "warning",
                        f"only {len(anchors)} rare term(s) to test with ({anchors}); owner to confirm absence")  # fmt: skip
            )
            continue

        async with conn.cursor(row_factory=dict_row) as cur:
            # The anchors are already lexemes, so they go in as a tsquery directly. plainto_tsquery would stem them a
            # second time, and a doubly stemmed lexeme matches nothing.
            await cur.execute(
                "SELECT c.speech_id, c.body, c.debate_date, c.house FROM chunk c "
                "WHERE NOT c.is_boilerplate AND c.ts @@ to_tsquery('simple', %(q)s) LIMIT 5",
                {"q": " & ".join(f"'{a}'" for a in anchors)},
            )
            hits = list(await cur.fetchall())
        candidates[item.id] = [
            {
                "speech_id": h["speech_id"],
                "body": h["body"],
                "date": str(h["debate_date"] or ""),
                "house": h["house"] or "",
            }
            for h in hits
        ]
        findings.append(
            Finding(item.id, "absence", "warning",
                    f"{len(hits)} sentence(s) carry all of {anchors}; absence needs confirming in review")  # fmt: skip
        )
    return findings, candidates


# --- the whole battery --------------------------------------------------------------------------------


async def run_all(conn: AsyncConnection, items: list[GoldItem], jaccard_max: float | None = None) -> dict[str, Any]:
    """Every check, plus the measured overlap distribution and a verdict per item."""
    threshold = choose_jaccard_threshold(items) if jaccard_max is None else jaccard_max
    findings: list[Finding] = [
        Finding("-", "schema", "error", message)
        for message in validate_schema({"schema_version": 2, "items": [to_dict(i) for i in items]})
    ]
    findings += await check_quotes(conn, items)
    findings += await check_facts(conn, items)
    findings += check_overlap(items, threshold)
    findings += check_duplicates(items)
    absence_findings, absence_candidates = await check_absence(conn, items)
    findings += absence_findings

    failed = {f.item_id for f in findings if f.is_error}
    return {
        "items": len(items),
        "jaccard_threshold": threshold,
        "overlap": overlap_report(items),
        "absence_candidates": absence_candidates,
        "errors": sum(1 for f in findings if f.is_error),
        "warnings": sum(1 for f in findings if not f.is_error),
        "failed_items": sorted(failed),
        "by_check": dict(Counter(f.check for f in findings)),
        "findings": [
            {"item": f.item_id, "check": f.check, "severity": f.severity, "message": f.message} for f in findings
        ],
    }
