"""The automatic validators, against a scratch corpus built row by row."""

from __future__ import annotations

import psycopg
import pytest
from goldset.schema import Evidence, GoldItem
from goldset.validate import (
    MAX_CONSECUTIVE_RUN,
    check_absence,
    check_duplicates,
    check_facts,
    check_overlap,
    check_quotes,
    choose_jaccard_threshold,
    overlap_report,
    run_all,
)
from ingest.runtime import run

QUOTE = "The Government have committed sixty three million pounds for transition costs."
FANCY = "He said “the evidence base is not there” — and pressed the point."


@pytest.fixture
def seeded(scratch, scratch_dsn):
    """A scratch corpus with two speeches, one of them carrying Hansard's typographic markup."""
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
        "VALUES ('W0', 'v2', '2026-09-01', '2026-09-17', 'dev'), "
        "       ('W3', 'v2', '2020-03-01', '2020-05-31', 'heldout')"
    )
    scratch.execute(
        "INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
        "VALUES ('d1', '2026-09-02', 'Commons', 'Local Government', 'Commons Chamber', 'other', 'W0', 'test')"
    )
    for sid, text in (("s1", f"Opening words. {QUOTE} More words."), ("s2", FANCY)):
        scratch.execute(
            "INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id) VALUES (%s, 'd1', 0, 'A Member', %s, 'W0')",
            (sid, text),
        )
    scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body) "
                    "VALUES ('s1', 'd1', 0, 0, %s, %s), ('s1', 'd1', 1, 1, %s, %s)",
                    (QUOTE, QUOTE, "Transition funding for councils was discussed at length.", QUOTE))  # fmt: skip
    # `check_absence` reads its anchors out of lexeme_df, which holds Postgres's `english` STEMS, not raw words.
    scratch.execute("INSERT INTO lexeme_df (lexeme, df) VALUES ('transit', 1), ('fund', 1), ('unspent', 1)")
    return scratch_dsn


def with_conn(dsn, fn):
    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            return await fn(conn)

    return run(go())


def item(**kwargs) -> GoldItem:
    base = dict(
        id="hum0001",
        question="How much was committed for transition costs?",
        category="exact_term",
        criticality="major",
        expected_behaviour="answer",
        provenance="passage_llm",
        window="W0",
        evidence=(Evidence("s1", QUOTE),),
    )
    return GoldItem(**(base | kwargs))


# --- quotes --------------------------------------------------------------------------------------------


def test_a_verbatim_quote_passes(seeded):
    assert with_conn(seeded, lambda c: check_quotes(c, [item()])) == []


def test_a_quote_that_is_not_in_the_speech_is_an_error(seeded):
    bad = item(evidence=(Evidence("s1", "The Government committed nothing whatsoever to anyone."),))
    findings = with_conn(seeded, lambda c: check_quotes(c, [bad]))
    assert [f.check for f in findings] == ["quote"]
    assert findings[0].is_error


def test_a_speech_that_does_not_exist_is_an_error(seeded):
    bad = item(evidence=(Evidence("nope", QUOTE),))
    findings = with_conn(seeded, lambda c: check_quotes(c, [bad]))
    assert "not in the corpus" in findings[0].message


def test_typographic_quotes_dashes_and_whitespace_are_not_differences(seeded):
    """Hansard prints curly quotes and em dashes; anything re-typing a quote uses ASCII. If that counted as a mismatch
    the validator would reject correct citations and teach the generator to avoid quoting.
    """
    ascii_version = item(evidence=(Evidence("s2", 'He said "the evidence base is not there" - and pressed the point.'),))
    assert with_conn(seeded, lambda c: check_quotes(c, [ascii_version])) == []


def test_inline_markup_in_the_speech_is_stripped_before_comparing(scratch, scratch_dsn):
    """`speech.text` keeps Hansard's markup and `chunk.body` does not, so a quote taken from the index is genuinely not
    a substring of its own speech until both are cleaned.
    """
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('W0', 'test', '2026-09-01', '2026-09-17', 'dev') ON CONFLICT DO NOTHING")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
                    "VALUES ('d1', '2026-09-02', 'Commons', 'H', 'Commons Chamber', 'other', 'W0', 'test')")  # fmt: skip
    scratch.execute("INSERT INTO speech (id, debate_id, seq, text, window_id) VALUES ('s1', 'd1', 0, %s, 'W0')",
                    ("We considered the <em>School Funding</em> survey published on 19 April.",))  # fmt: skip
    clean = item(window="", evidence=(Evidence("s1", "We considered the School Funding survey published on 19 April."),))
    assert with_conn(scratch_dsn, lambda c: check_quotes(c, [clean])) == []


def test_an_item_tagged_with_the_wrong_window_is_an_error(seeded):
    findings = with_conn(seeded, lambda c: check_quotes(c, [item(window="W3")]))
    assert any("is in W0" in f.message for f in findings)


# --- facts ---------------------------------------------------------------------------------------------


def test_a_fact_that_no_longer_matches_the_database_is_an_error(scratch, scratch_dsn):
    scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, ayes, noes, result) "
                    "VALUES (10, 10, 'Commons', '2026-09-08', 'A Bill', 300, 200, 'agreed')")  # fmt: skip
    wrong = item(
        id="db0001", category="division_result", evidence=(), window="",
        expected_structured_facts={"division_id": 10, "result": "negatived", "ayes": 300, "noes": 200},
    )
    findings = with_conn(scratch_dsn, lambda c: check_facts(c, [wrong]))
    assert any("gold says 'negatived'" in f.message for f in findings)

    right = item(id="db0002", category="division_result", evidence=(), window="",
                 expected_structured_facts={"division_id": 10, "result": "agreed", "ayes": 300, "noes": 200})  # fmt: skip
    assert with_conn(scratch_dsn, lambda c: check_facts(c, [right])) == []


def test_a_fact_check_that_raises_names_its_item_instead_of_killing_the_run(scratch_dsn):
    broken = item(id="db0003", category="division_result", evidence=(), window="",
                  expected_structured_facts={"division_id": "not-an-integer"})  # fmt: skip
    findings = with_conn(scratch_dsn, lambda c: check_facts(c, [broken, item()]))
    assert findings[0].item_id == "db0003"
    assert "fact check failed" in findings[0].message


def test_an_item_with_no_structured_facts_is_skipped(scratch_dsn):
    assert with_conn(scratch_dsn, lambda c: check_facts(c, [item()])) == []


# --- the overlap filter ----------------------------------------------------------------------------------


def test_a_paraphrase_that_copies_a_long_run_is_rejected():
    copied = item(
        category="paraphrase",
        question="The Government have committed sixty three million pounds for transition costs, did they?",
    )
    findings = check_overlap([copied], 1.0)
    assert any("consecutive words" in f.message and f.is_error for f in findings)


def test_a_genuine_paraphrase_passes():
    genuine = item(category="paraphrase", question="Was any money set aside to help councils reorganise?")
    assert check_overlap([genuine], 0.35) == []


def test_a_paraphrase_over_the_jaccard_threshold_is_rejected():
    close = item(category="paraphrase", question="Government committed transition costs, how much?")
    assert any("token-Jaccard" in f.message for f in check_overlap([close], 0.1))


def test_a_non_paraphrase_that_copies_heavily_is_only_warned_about():
    """An exact-term question is SUPPOSED to share wording with its answer; that is the category."""
    copied = item(question="The Government have committed sixty three million pounds for transition costs?")
    findings = check_overlap([copied], 0.35)
    assert findings and not any(f.is_error for f in findings)


def test_the_jaccard_threshold_is_chosen_from_the_lexical_categories():
    lexical = [
        item(id=f"hum{i:04d}", question=f"What about transition costs number {i}?",
             evidence=(Evidence("s1", QUOTE),))  # fmt: skip
        for i in range(12)
    ]
    threshold = choose_jaccard_threshold(lexical)
    assert 0.15 <= threshold <= 0.35


def test_the_threshold_falls_back_to_the_default_when_there_is_too_little_data():
    from goldset.validate import DEFAULT_JACCARD_MAX

    assert choose_jaccard_threshold([item()]) == DEFAULT_JACCARD_MAX


def test_the_overlap_report_measures_the_distribution():
    report = overlap_report([item(), item(id="hum0002", question="Different words entirely about housing?")])
    assert report["n"] == 2
    assert report["run"]["max"] >= 1
    assert set(report["jaccard"]) >= {"max", "p50", "p90", "p95"}
    assert MAX_CONSECUTIVE_RUN == 4


def test_the_overlap_report_handles_a_set_with_no_evidence():
    assert overlap_report([]) == {"n": 0}


# --- duplicates ------------------------------------------------------------------------------------------


def test_two_questions_with_the_same_wording_and_the_same_answer_are_a_duplicate():
    a = item(id="db0001", category="division_result", evidence=(),
             question="Did the Commons agree to the Health Bill on 8 September 2026?",
             expected_structured_facts={"division_id": 1})  # fmt: skip
    b = item(id="db0002", category="division_result", evidence=(),
             question="Did the Commons agree to the Health Bill measure on 8 September 2026?",
             expected_structured_facts={"division_id": 1})  # fmt: skip
    findings = check_duplicates([a, b])
    assert [f.check for f in findings] == ["duplicate"]
    assert findings[0].is_error


def test_two_questions_with_the_same_wording_but_different_answers_are_kept():
    """New Clause 6 against New Clause 140 is the pair a system most needs to tell apart."""
    a = item(id="db0001", category="division_result", evidence=(),
             question="Did the Commons agree to the Health Bill New Clause 6 on 8 September 2026?",
             expected_structured_facts={"division_id": 1})  # fmt: skip
    b = item(id="db0002", category="division_result", evidence=(),
             question="Did the Commons agree to the Health Bill New Clause 9 on 8 September 2026?",
             expected_structured_facts={"division_id": 2})  # fmt: skip
    findings = check_duplicates([a, b])
    assert [f.check for f in findings] == ["discriminating_pair"]
    assert not findings[0].is_error


def test_unrelated_questions_produce_nothing():
    a = item(id="hum0001", question="What was said about chalk streams in the water debate?")
    b = item(id="hum0002", question="How many flights were cancelled in the air traffic failure?")
    assert check_duplicates([a, b]) == []


# --- absence ---------------------------------------------------------------------------------------------


def test_an_out_of_window_item_naming_a_covered_year_is_an_error(seeded):
    bad = item(id="adv0001", category="out_of_window", expected_behaviour="abstain_out_of_window",
               evidence=(), window="", question="What did MPs say about the Budget in 2026?")  # fmt: skip
    findings, _ = with_conn(seeded, lambda c: check_absence(c, [bad]))
    assert findings[0].is_error
    assert "which the corpus covers" in findings[0].message


def test_an_out_of_window_item_naming_an_uncovered_year_passes(seeded):
    good = item(id="adv0002", category="out_of_window", expected_behaviour="abstain_out_of_window",
                evidence=(), window="", question="What did MPs say about the Olympics in 2012?")  # fmt: skip
    findings, _ = with_conn(seeded, lambda c: check_absence(c, [good]))
    assert findings == []


def test_an_out_of_window_item_with_no_year_is_flagged_for_the_owner(seeded):
    vague = item(id="adv0003", category="out_of_window", expected_behaviour="abstain_out_of_window",
                 evidence=(), window="", question="What did MPs say when it all happened?")  # fmt: skip
    findings, _ = with_conn(seeded, lambda c: check_absence(c, [vague]))
    assert findings[0].severity == "warning"
    assert "cannot be checked" in findings[0].message


def test_a_near_miss_item_is_warned_about_and_carries_its_counter_evidence(seeded):
    """A near miss is on-topic by construction, so co-occurrence cannot settle it. The check has to hand the closest
    passages to a human rather than pretend to a verdict.
    """
    near = item(id="adv0004", category="near_miss_abstention", expected_behaviour="abstain_not_in_record",
                evidence=(), window="", question="How much transition funding was left unspent?")  # fmt: skip
    findings, candidates = with_conn(seeded, lambda c: check_absence(c, [near]))
    assert findings and not any(f.is_error for f in findings)
    assert "adv0004" in candidates


def test_an_answering_item_is_not_checked_for_absence(seeded):
    findings, candidates = with_conn(seeded, lambda c: check_absence(c, [item()]))
    assert findings == [] and candidates == {}


# --- the whole battery -------------------------------------------------------------------------------------


def test_run_all_collects_every_check_and_names_the_failures(seeded):
    good = item()
    bad = item(id="hum0002", question="A question whose quote is simply not there at all?",
               evidence=(Evidence("s1", "Words that appear nowhere in this speech whatsoever."),))  # fmt: skip
    report = with_conn(seeded, lambda c: run_all(c, [good, bad]))
    assert report["items"] == 2
    assert report["failed_items"] == ["hum0002"]
    assert report["errors"] >= 1
    assert "overlap" in report and "jaccard_threshold" in report


def test_run_all_reports_schema_problems_too(seeded):
    report = with_conn(seeded, lambda c: run_all(c, [item(window="W3", split="dev")]))
    assert any(f["check"] == "schema" for f in report["findings"])
