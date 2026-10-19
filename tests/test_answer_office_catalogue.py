"""The record's offices, read once per connection and matched in Python."""

from __future__ import annotations

import pytest

from answer import resolve as RS
from app import safety
from ingest.runtime import run

# The scan as it stood before the catalogue (resolve.py, S06), kept as the reference the catalogue must agree with.
SCAN = """
    SELECT DISTINCT office_key, office, length(office) AS office_length
    FROM (SELECT office_key, regexp_replace(btrim(office), '\\s+', ' ', 'g') AS office
            FROM member_office WHERE office IS NOT NULL) o
    WHERE %s ILIKE '%%' || regexp_replace(office, '^the\\s+', '', 'i') || '%%'
    ORDER BY office_length DESC, office, office_key LIMIT 20
"""

OFFICES = (
    "Chancellor of the Exchequer ",  # 32 biography posts end in a space
    "Minister of State  (Home Office)",  # and 3 hold a double one
    "Minister of State (Home Office)",
    "The Lord Chancellor",  # a leading "the" is not part of what a question must contain
    "Lord Chancellor",
    "Prime Minister",
    "Deputy Prime Minister",
    "Deputy Chief Whip (House of Lords)",
    "Chief Whip (House of Lords)",
    "Minister for Women",  # two names of one length: the order between them is the database's
    "Minister for Wales",
    *(f"Parliamentary Under-Secretary {n}" for n in range(30)),  # more matches than the cap
)


@pytest.fixture
def offices(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE member_office CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (6101, 'A Holder', 'Commons')")
    for office in OFFICES:
        scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, source, evidence) "
                        "VALUES (6101, %s, lower(btrim(%s)), '2020-01-01', 'mnis_biography', 0)", (office, office))
    yield scratch_dsn
    scratch.execute("TRUNCATE member_office CASCADE")


def _on_one_connection(dsn, work):
    """Run `work(conn, statements)` on one fresh connection, counting the statements it sends."""
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        statements: list[str] = []
        send = conn.execute

        async def counted(query, *args, **kwargs):
            statements.append(" ".join(str(query).split()))
            return await send(query, *args, **kwargs)

        conn.execute = counted
        try:
            return await work(conn, statements)
        finally:
            await conn.close()
    return run(go())


QUESTIONS = [
    "Who was Chancellor of the Exchequer on 4 March 2020?",
    "Who was the Minister of State (Home Office) on 4 March 2020?",
    "Who was the Lord Chancellor on 4 March 2020?",
    "Who was the Deputy Prime Minister on 4 March 2020?",
    "Who was the Deputy Chief Whip (House of Lords) on 4 March 2020?",
    "Who was Minister for Women and Minister for Wales on 4 March 2020?",
    "Who was the Parliamentary Under-Secretary 1 on 4 March 2020?",
    "Who was the  Prime   Minister on 4 March 2020?",
    "Who was the Green Party Chancellor of the Exchequer on 4 March 2020?",
    "Who chaired the Health Committee?",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_the_catalogue_names_the_offices_the_scan_named_in_its_order(offices, question):
    async def work(conn, _statements):
        scanned = await RS._all(conn, SCAN, (" ".join(question.split()),))
        return scanned, await RS.office_candidates(conn, question)

    scanned, matched = _on_one_connection(offices, work)
    assert [(r["office_key"], r["office"]) for r in matched] == [(r["office_key"], r["office"]) for r in scanned]


def test_more_than_twenty_offices_named_come_back_as_the_first_twenty(offices):
    question = "Who held " + ", ".join(f"Parliamentary Under-Secretary {n}" for n in range(25)) + "?"

    async def work(conn, _statements):
        return await RS._all(conn, SCAN, (question,)), await RS.office_candidates(conn, question)

    scanned, matched = _on_one_connection(offices, work)
    assert len(matched) == 20 and [r["office"] for r in matched] == [r["office"] for r in scanned]


def test_an_office_name_is_matched_literally_never_as_a_pattern(scratch, scratch_dsn):
    safety.require_test(scratch_dsn)
    scratch.execute("TRUNCATE member_office CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (6102, 'B Holder', 'Commons')")
    for office in ("Minister_for Sport", "Minister for 100% Renewables"):
        scratch.execute("INSERT INTO member_office (mnis_id, office, office_key, start_date, source, evidence) "
                        "VALUES (6102, %s, lower(%s), '2020-01-01', 'mnis_biography', 0)", (office, office))

    async def work(conn, _statements):
        return [[r["office"] for r in await RS.office_candidates(conn, q)] for q in (
            "Who was the Minister for Sport in 2020?",  # "_" read as any character matched this
            "Who was the Minister for 100 per cent Renewables?",  # and "%" as any run of characters this
            "Who was the Minister_for Sport in 2020?")]

    wildcard_underscore, wildcard_percent, literal = _on_one_connection(scratch_dsn, work)
    scratch.execute("TRUNCATE member_office CASCADE")
    assert wildcard_underscore == [] and wildcard_percent == []
    assert literal == ["Minister_for Sport"]


def test_the_record_s_offices_are_read_once_per_connection(offices):
    async def work(conn, statements):
        for question in QUESTIONS:
            await RS.office_candidates(conn, question)
            await RS.office_key(conn, question)
        return [s for s in statements if "member_office" in s]

    reads = _on_one_connection(offices, work)
    assert len(reads) == 1, reads


def test_the_deputy_relation_reads_the_same_catalogue(offices):
    from answer import contract as K
    from answer import office_relations as OR

    prov = K.Provenance(route="structured", corpus_version="v2", retrieval_config_hash="e57c6cdfd8793d76", recipe="r",
                        prompt_version="p", model_id="m")  # fmt: skip

    async def work(conn, statements):
        from datetime import date

        answers = [await OR.answer(conn, q, date(2020, 3, 4), prov) for q in (
            "Who was the deputy to the Prime Minister on 4 March 2020?",
            "Who was the deputy to the Chief Whip (House of Lords) on 4 March 2020?")]
        return answers, [s for s in statements if "member_office" in s and "start_date" not in s]

    answers, reads = _on_one_connection(offices, work)
    offered = [[c["office"] for c in (a.facets or {}).get("clarify", [])] for a, _why in answers]
    assert offered == [["Deputy Prime Minister"], ["Deputy Chief Whip (House of Lords)"]], answers
    assert len(reads) == 1, reads
