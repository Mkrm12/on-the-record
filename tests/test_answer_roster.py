"""The roster and debate identification."""

from __future__ import annotations

from datetime import date

import pytest

from answer import roster as R
from ingest.runtime import run as run_async


def candidate(debate_id, score, **kw):
    return R.DebateCandidate(debate_id=debate_id, score=score, **kw)


# --- confidence: a margin, not a raw score --------------------------------------------------------------


def test_a_single_candidate_is_certain():
    """There is nothing it could be confused with, so the margin is 1.0 however low the raw score."""
    assert R.confidence([candidate("d1", 0.01)]) == 1.0


def test_confidence_is_the_share_of_the_top_two():
    """A retrieval score is not calibrated and means nothing on its own; "twice the runner-up" transfers across
    queries, which is what a threshold needs.
    """
    assert R.confidence([candidate("a", 0.9), candidate("b", 0.1)]) == pytest.approx(0.9)
    assert R.confidence([candidate("a", 0.5), candidate("b", 0.5)]) == pytest.approx(0.5)
    assert R.confidence([candidate("a", 3.0), candidate("b", 1.0)]) == pytest.approx(0.75)


def test_confidence_ignores_everything_below_the_runner_up():
    """A long tail of weak candidates must not dilute a clear winner into a clarify."""
    strong = [candidate("a", 0.9), candidate("b", 0.1)]
    assert R.confidence(strong) == R.confidence(strong + [candidate(f"x{i}", 0.01) for i in range(20)])


def test_no_candidates_is_zero_confidence():
    assert R.confidence([]) == 0.0


def test_all_zero_scores_are_not_certainty():
    """Dividing by a zero total must not read as 1.0. Nothing matched, so nothing is confident."""
    assert R.confidence([candidate("a", 0.0), candidate("b", 0.0)]) == 0.0


# --- pick_debate: answer, or ask; never guess -----------------------------------------------------------


def test_a_clear_winner_is_answered():
    choice = R.pick_debate([candidate("a", 0.9), candidate("b", 0.1)])
    assert choice.ok and choice.debate_id == "a"
    assert not choice.needs_clarification


def test_a_near_tie_asks_instead_of_guessing():
    """The real case: "Which member spoke first in the Lords debate on Arrangement of Business" — a heading that recurs
    on most Lords sitting days. Measured on the live corpus, ten candidates at margin 0.500.
    """
    choice = R.pick_debate([candidate("a", 0.50), candidate("b", 0.49), candidate("c", 0.48)])
    assert not choice.ok
    assert choice.needs_clarification
    assert [c.debate_id for c in choice.clarify] == ["a", "b", "c"]
    assert "leads the runner-up by too little" in choice.reason


def test_a_clarify_offers_exactly_three_candidates():
    """says three. More is a list nobody reads; fewer hides a real alternative."""
    choice = R.pick_debate([candidate(f"d{i}", 0.5 - i * 0.001) for i in range(9)])
    assert len(choice.clarify) == R.CLARIFY_CANDIDATES == 3


def test_no_candidates_is_neither_an_answer_nor_a_clarify():
    """Nothing to choose between is a different outcome from a hard choice, and a clarify listing nothing would be an
    empty question put to the user.
    """
    choice = R.pick_debate([])
    assert not choice.ok and not choice.needs_clarification
    assert "no debate" in choice.reason


def test_ties_are_broken_deterministically():
    """Two identical scores must not make the answer depend on row order, or the same question answers differently on
    two runs.
    """
    first = R.pick_debate([candidate("b", 0.9), candidate("a", 0.9)], threshold=0.0)
    second = R.pick_debate([candidate("a", 0.9), candidate("b", 0.9)], threshold=0.0)
    assert first.debate_id == second.debate_id == "a"


def test_the_threshold_is_the_swept_one():
    """Swept on dev: below 0.52 one debate is answered wrongly; from 0.52 to 0.90 that becomes a clarify and the curve
    is flat, because dev holds exactly one item in the discriminating band. The value is chosen on what the measure
    means, not on the sweep, and the module says so.
    """
    assert R.DEBATE_CONFIDENCE_THRESHOLD == 0.55


@pytest.mark.parametrize(("margin", "answered"), [(0.99, True), (0.60, True), (0.56, True),
                                                  (0.55, True), (0.54, False), (0.50, False)])  # fmt: skip
def test_the_threshold_is_inclusive_at_its_own_value(margin, answered):
    """A margin exactly at the threshold answers. An off-by-one here silently changes the clarify rate."""
    scores = [candidate("a", margin), candidate("b", 1.0 - margin)]
    assert R.pick_debate(scores).ok is answered


# --- the SQL half ---------------------------------------------------------------------------------------

DEBATE = "r-debate"
OTHER = "r-other"


@pytest.fixture
def roster_db(scratch, scratch_dsn):
    """A debate opened by the Chair with a procedural utterance, then the substantive speakers."""
    c = scratch
    c.execute("TRUNCATE corpus_window, debate, speech, chunk, member_id_map CASCADE")
    c.execute("""INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role)
                 VALUES ('WR', 'v2', '2026-05-01', '2026-05-31', 'dev')""")
    c.execute("""INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version)
                 VALUES (%s, '2026-05-12', 'Commons', 'Housing Supply', 'Commons Chamber', 'WR', 'v2'),
                        (%s, '2026-05-19', 'Commons', 'Housing Supply', 'Commons Chamber', 'WR', 'v2')""",
              (DEBATE, OTHER))
    c.execute("""INSERT INTO member_id_map (mnis_id, name, party) VALUES
                 (4048, 'Caroline Nokes', 'Con'), (1171, 'Liam Byrne', 'Lab'), (1002, 'B Member', 'LD')""")
    c.execute("""INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, is_boilerplate,
                                     boilerplate_reason, speaker_party, window_id)
                 VALUES ('rs1', %s, 1, 'Madam Deputy Speaker (Caroline Nokes)', 4048,
                         'We now come to the debate on housing supply.', true, 'chair_procedural', 'Con', 'WR'),
                        ('rs2', %s, 2, 'Liam Byrne (Birmingham Hodge Hill) (Lab)', 1171,
                         'I beg to move that this House considers housing supply.', false, NULL, 'Lab', 'WR'),
                        ('rs3', %s, 3, 'B Member', 1002, 'I support the motion on supply.', false, NULL, 'LD', 'WR'),
                        ('rs4', %s, 4, 'Liam Byrne (Birmingham Hodge Hill) (Lab)', 1171,
                         'I thank the honourable member.', false, NULL, 'Lab', 'WR'),
                        ('rs5', %s, 5, 'An Unmapped Member', NULL, 'A word on supply.', false, NULL, NULL, 'WR'),
                        ('rs6', %s, 1, 'B Member', 1002, 'A different sitting entirely.', false, NULL, 'LD', 'WR')""",
              (DEBATE, DEBATE, DEBATE, DEBATE, DEBATE, OTHER))
    # `ts` is a GENERATED column — Postgres derives the tsvector from the body, so the fixture must not supply it and
    # the full-text ranking below is the production one rather than a hand-built approximation.
    c.execute("""INSERT INTO chunk (id, speech_id, debate_id, seq, sent_idx, body, parent_body, is_boilerplate,
                                    mnis_id, debate_date, house, section, window_id)
                 VALUES (1, 'rs2', %s, 2, 0, 'I beg to move that this House considers housing supply.',
                         'I beg to move that this House considers housing supply.',
                         false, 1171, '2026-05-12', 'Commons', 'Commons Chamber', 'WR'),
                        (2, 'rs3', %s, 3, 0, 'I support the motion on supply.',
                         'I support the motion on supply.', false, 1002,
                         '2026-05-12', 'Commons', 'Commons Chamber', 'WR'),
                        (3, 'rs4', %s, 4, 0, 'I thank the honourable member.',
                         'I thank the honourable member.', false, 1171,
                         '2026-05-12', 'Commons', 'Commons Chamber', 'WR')""",
              (DEBATE, DEBATE, DEBATE))
    yield scratch_dsn
    c.execute("TRUNCATE corpus_window, debate, speech, chunk, member_id_map CASCADE")


def call(fn, dsn, *args, **kwargs):
    import psycopg

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await fn(conn, *args, **kwargs)
        finally:
            await conn.close()

    return run_async(go())


def test_including_boilerplate_never_names_the_chair(roster_db):
    """The default. The Chair's "We now come to the debate" is the record's first speech and is not the opening one.
    """
    found = call(R.first_speaker, roster_db, DEBATE, True)
    assert found["mnis_id"] == 1171
    assert found["is_boilerplate"] is False


def test_excluding_boilerplate_names_the_member_who_moved_the_motion(roster_db):
    found = call(R.first_speaker, roster_db, DEBATE, False)
    assert found["mnis_id"] == 1171
    assert found["is_boilerplate"] is False


def test_the_default_is_recorded_as_a_constant(roster_db):
    """So the decision is visible in one place rather than spread through call sites. It decides whether a member's
    formal motion counts, never whether the Chair does.
    """
    assert R.FIRST_SPEAKER_INCLUDES_BOILERPLATE is True
    assert call(R.first_speaker, roster_db, DEBATE)["mnis_id"] == 1171


def test_an_unknown_debate_says_so_rather_than_returning_nobody(roster_db):
    assert "missing" in call(R.first_speaker, roster_db, "no-such-debate")


def test_the_opener_carries_party_and_office_at_the_date(roster_db):
    found = call(R.first_speaker, roster_db, DEBATE, False)
    assert found["party"] == "Lab", "what Hansard printed on the day"
    assert found["date"] == date(2026, 5, 12) and found["house"] == "Commons"


# --- the roster -----------------------------------------------------------------------------------------


def test_the_roster_is_in_speaking_order_with_contribution_counts(roster_db):
    built = call(R.roster, roster_db, [DEBATE])
    assert [e.speaker_name for e in built.entries] == ["Liam Byrne", "B Member", "An Unmapped Member"]
    assert [e.contributions for e in built.entries] == [2, 1, 1]


def test_the_chair_is_neither_a_speaker_in_the_roster_nor_the_debates_opener(roster_db):
    """The distinction the record itself draws: the Chair calling an item is not a contribution to it, so the Chair is
    not a participant, and it does not open the debate either.
    """
    built = call(R.roster, roster_db, [DEBATE])
    assert 4048 not in {e.mnis_id for e in built.entries}
    assert call(R.first_speaker, roster_db, DEBATE)["mnis_id"] == 1171


def test_names_are_the_mapped_ones_not_the_printed_string(roster_db):
    """Hansard prints "Liam Byrne (Birmingham Hodge Hill) (Lab)" in one field. A roster showing that has put three
    facts in the field reserved for one, with the party then appearing twice and able to disagree with
    `party_at_date`.
    """
    built = call(R.roster, roster_db, [DEBATE])
    assert built.entries[0].speaker_name == "Liam Byrne"
    assert "(" not in built.entries[0].speaker_name


def test_an_unmapped_speaker_is_listed_rather_than_dropped(roster_db):
    """A speaker the mapper could not resolve is still a speaker. Dropping them presents an incomplete list as a
    complete one, and the reader cannot tell.
    """
    built = call(R.roster, roster_db, [DEBATE])
    unmapped = [e for e in built.entries if e.mnis_id is None]
    assert len(unmapped) == 1 and unmapped[0].speaker_name == "An Unmapped Member"


def test_every_speaker_gets_a_sentence_even_when_the_query_matches_only_one(roster_db):
    """The bug this pins: the query-ranked pass filled only the speakers it matched, and the positional fallback ran
    only when NOBODY matched — so a query hitting one speaker left 63 of 64 entries with no sentence at all, a
    roster that names people and shows nothing they said.
    """
    built = call(R.roster, roster_db, [DEBATE], 1.0, "honourable member")
    mapped = [e for e in built.entries if e.mnis_id is not None]
    assert all(e.best_chunk_id is not None for e in mapped)
    # And the query still wins where it matched: "I thank the honourable member" is Byrne's second chunk.
    assert next(e for e in built.entries if e.mnis_id == 1171).best_chunk_id == 3


def test_without_a_query_the_first_sentence_is_used(roster_db):
    built = call(R.roster, roster_db, [DEBATE])
    assert next(e for e in built.entries if e.mnis_id == 1171).best_chunk_id == 1


def test_a_query_that_matches_nothing_still_yields_sentences(roster_db):
    """A roster must not become empty because the question used words nobody said."""
    built = call(R.roster, roster_db, [DEBATE], 1.0, "zeppelin actuarial")
    assert all(e.best_chunk_id is not None for e in built.entries if e.mnis_id is not None)


def test_several_debates_can_be_rostered_at_once(roster_db):
    built = call(R.roster, roster_db, [DEBATE, OTHER])
    assert built.debate_ids == (DEBATE, OTHER)
    assert next(e for e in built.entries if e.mnis_id == 1002).contributions == 2, "one in each debate"


def test_a_roster_needs_a_debate(roster_db):
    with pytest.raises(ValueError, match="at least one debate"):
        call(R.roster, roster_db, [])


def test_the_confidence_travels_with_the_roster(roster_db):
    """So a downstream answer can disclose how sure the debate identification was."""
    assert call(R.roster, roster_db, [DEBATE], 0.62).confidence == pytest.approx(0.62)


# --- gaps -----------------------------------------------------------------------------------------------


def test_an_incomplete_roster_says_so(roster_db):
    out = call(R.roster_with_gaps, roster_db, [DEBATE])
    assert out["complete"] is False
    assert out["unattributed"] == 1
    assert "may be incomplete" in out["caveat"]


def test_a_complete_roster_carries_no_caveat(roster_db):
    out = call(R.roster_with_gaps, roster_db, [OTHER])
    assert out["complete"] is True and out["caveat"] == ""


# --- identifying the debate from a heading --------------------------------------------------------------


def test_a_recurring_heading_produces_several_candidates(roster_db):
    """Two sittings with the same heading, which is the situation `pick_debate` exists for."""
    found = call(R.candidates_by_heading, roster_db, "Housing Supply", "Commons")
    assert {c.debate_id for c in found} == {DEBATE, OTHER}
    assert not R.pick_debate(found).ok, "a tie on heading alone must ask, not guess"


def test_a_date_narrows_a_recurring_heading_to_one_debate(roster_db):
    """Which is why the resolver passes the date through: the question usually says which sitting it means."""
    found = call(R.candidates_by_heading, roster_db, "Housing Supply", "Commons", date(2026, 5, 12))
    assert [c.debate_id for c in found] == [DEBATE]
    assert R.pick_debate(found).debate_id == DEBATE


def test_a_heading_that_matches_nothing_returns_no_candidates(roster_db):
    assert call(R.candidates_by_heading, roster_db, "zeppelin actuarial", "Commons") == []


def test_prose_with_punctuation_does_not_raise(roster_db):
    """`websearch_to_tsquery`, not `to_tsquery`: a roster that threw on an apostrophe would be unusable, and the
    questions are prose — "Who opened the debate on Carer's Allowance?".
    """
    assert call(R.candidates_by_heading, roster_db, "Who opened the debate on Carer's Allowance?") == []
