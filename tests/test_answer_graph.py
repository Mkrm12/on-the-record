"""The relationship tools, against a scratch Postgres with rows this test controls."""

from __future__ import annotations

from datetime import date

import pytest

from answer import graph as G
from ingest.runtime import run as run_async

BILL = 9001
OTHER_BILL = 9002


@pytest.fixture
def graph_db(scratch, scratch_dsn):
    """One bill, two stages in two Houses, three sittings, three debates, speeches and two divisions."""
    c = scratch
    c.execute("TRUNCATE bill, bill_stage, bill_stage_sitting, debate, speech, division, division_vote, "
              "member_id_map CASCADE")
    c.execute("""INSERT INTO bill (id, title, originating_house, current_house, is_act, is_defeated, is_withdrawn)
                 VALUES (%s, 'Test Governance Bill', 'Commons', 'Lords', false, false, false),
                        (%s, 'Unrelated Bill', 'Lords', 'Lords', true, false, false)""", (BILL, OTHER_BILL))
    c.execute("""INSERT INTO bill_stage (id, bill_id, stage_name, house, date, sort_order)
                 VALUES (10, %s, 'Second reading', 'Commons', '2026-05-12', 1),
                        (11, %s, 'Committee stage', 'Commons', '2026-05-19', 2),
                        (12, %s, 'Second reading', 'Lords',   '2026-06-02', 3),
                        (13, %s, 'Second reading', 'Lords',   '2026-06-09', 1)""",
              (BILL, BILL, BILL, OTHER_BILL))
    c.execute("""INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role)
                 VALUES ('TG', 'test', '2026-01-01', '2026-12-31', 'dev') ON CONFLICT DO NOTHING""")
    c.execute("""INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version)
                 VALUES ('d-sr',  '2026-05-12', 'Commons', 'Test Governance Bill', 'Debates', 'TG', 'test'),
                        ('d-cttee','2026-05-19', 'Commons', 'Committee', 'Debates', 'TG', 'test'),
                        ('d-lords','2026-06-02', 'Lords',   'Test Governance Bill', 'Debates', 'TG', 'test'),
                        ('d-other','2026-06-09', 'Lords',   'Unrelated Bill', 'Debates', 'TG', 'test')""")
    # Two sittings of stage 11 pointing at one debate: the join that a cycle guard has to survive.
    c.execute("""INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, house, debate_id)
                 VALUES (20, 10, %s, '2026-05-12', 'Commons', 'd-sr'),
                        (21, 11, %s, '2026-05-19', 'Commons', 'd-cttee'),
                        (22, 11, %s, '2026-05-20', 'Commons', 'd-cttee'),
                        (23, 12, %s, '2026-06-02', 'Lords',   'd-lords'),
                        (24, 13, %s, '2026-06-09', 'Lords',   'd-other')""",
              (BILL, BILL, BILL, BILL, OTHER_BILL))
    c.execute("""INSERT INTO member_id_map (mnis_id, name, party) VALUES
                 (1001, 'A Member', 'Labour'), (1002, 'B Member', 'Conservative')""")
    c.execute("""INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, is_boilerplate, window_id)
                 VALUES ('s1', 'd-sr', 1, 'A Member', 1001, 'I beg to move that the Bill be read a second time.', false, 'TG'),
                        ('s2', 'd-sr', 2, 'B Member', 1002, 'The House will rise at six.', false, 'TG'),
                        ('s3', 'd-sr', 3, 'A Member', 1001, 'I thank my honourable friend.', false, 'TG'),
                        ('s4', 'd-sr', 4, 'The Speaker', NULL, 'Order.', true, 'TG'),
                        ('s5', 'd-lords', 1, 'B Member', 1002, 'My Lords, I support the Bill.', false, 'TG'),
                        ('s6', 'd-other', 1, 'A Member', 1001, 'A word on the Unrelated Bill.', false, 'TG')""")
    # 30: a Commons division whose name lists do not reconcile, reachable through its debate. 31: reachable ONLY through
    # bill_stage_id, with no debate_id at all. source_id is NOT NULL and `division_id_namespace_ck` fixes its relation
    # to id: a Commons division has id = source_id, a Lords one id = source_id + 1000000, because the two Votes APIs
    # number independently and one table has to hold both without collision.
    c.execute("""INSERT INTO division (id, source_id, debate_id, bill_stage_id, house, number, division_date,
                                      title, ayes, noes, result, ayes_listed, noes_listed, ayes_named, noes_named,
                                      tally_reconciled)
                 VALUES (30, 30, 'd-sr', 10, 'Commons', 1, '2026-05-12', 'Second reading',
                         316, 295, 'agreed', 317, 297, 315, 295, false),
                        (31, 31, NULL, 11, 'Commons', 2, '2026-05-19', 'Amendment 4',
                         100, 120, 'negatived', 100, 120, 100, 120, true)""")
    # `division_vote_lobby_ck` fixes the vocabulary: aye | no | no_vote | conflict, lower case.
    c.execute("""INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, is_teller)
                 VALUES (30, 1001, 'aye', 'A Member', false),
                        (30, 1002, 'no',  'B Member', true),
                        (31, 1001, 'no',  'A Member', false)""")
    # The DSN, not the connection: `conn.info.dsn` omits the password, so a pool built from it cannot authenticate and
    # every call waited out a 30-second PoolTimeout instead of failing usefully.
    yield scratch_dsn
    c.execute("TRUNCATE bill, bill_stage, bill_stage_sitting, debate, speech, division, division_vote, "
              "member_id_map CASCADE")


def call(coro_fn, dsn, *args, **kwargs):
    """Run one async tool against the scratch database."""
    import psycopg

    async def go():
        aconn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await coro_fn(aconn, *args, **kwargs)
        finally:
            await aconn.close()

    return run_async(go())


# --- bill_walk ------------------------------------------------------------------------------------------


def test_a_bill_walk_returns_its_own_stages_in_order_and_no_others(graph_db):
    walk = call(G.bill_walk, graph_db, BILL)
    assert walk.ok
    assert walk.title == "Test Governance Bill"
    assert [s.id for s in walk.stages] == [10, 11, 12], "sort_order, and only this bill's stages"
    assert [s.house for s in walk.stages] == ["Commons", "Commons", "Lords"]


def test_a_stage_with_two_sittings_counts_them_both(graph_db):
    walk = call(G.bill_walk, graph_db, BILL)
    committee = next(s for s in walk.stages if s.id == 11)
    assert committee.sitting_count == 2
    assert committee.debate_ids == ("d-cttee",), "two sittings, one debate, listed once"


def test_a_division_recorded_against_a_stage_with_no_debate_is_still_found(graph_db):
    """Division 31 has bill_stage_id and no debate_id. Reaching a bill's divisions only through its debates would lose
    it silently — the answer would be "one division" when the record holds two.
    """
    walk = call(G.bill_walk, graph_db, BILL)
    found = {d.id for s in walk.stages for d in s.divisions}
    assert found == {30, 31}


def test_a_nonreconciling_tally_always_carries_the_i8_caveat(graph_db):
    walk = call(G.bill_walk, graph_db, BILL)
    thirty = next(d for s in walk.stages for d in s.divisions if d.id == 30)
    assert "do not reconcile" in thirty.caveat and "I-8" in thirty.caveat
    assert "315 Ayes" in thirty.caveat and "316" in thirty.caveat, "the names and the figure must both appear"
    assert "tellers' figures are the result" in thirty.caveat


def test_a_reconciling_tally_carries_no_caveat(graph_db):
    walk = call(G.bill_walk, graph_db, BILL)
    thirty_one = next(d for s in walk.stages for d in s.divisions if d.id == 31)
    assert thirty_one.caveat == ""


def test_boilerplate_is_excluded_from_the_speech_count(graph_db):
    """'Order.' is not a contribution, and counting it inflates every per-bill total."""
    walk = call(G.bill_walk, graph_db, BILL)
    assert walk.speech_count == 4, "s1, s2, s3 in d-sr and s5 in d-lords; s4 is boilerplate"
    assert walk.speaker_count == 2


def test_the_date_range_comes_from_the_debates_not_the_stages(graph_db):
    walk = call(G.bill_walk, graph_db, BILL)
    assert walk.first_date == date(2026, 5, 12)
    assert walk.last_date == date(2026, 6, 2)


def test_an_unknown_bill_says_so_rather_than_returning_an_empty_bill(graph_db):
    """An empty walk and a missing bill look identical to a caller, and one of them means "ask differently"."""
    walk = call(G.bill_walk, graph_db, 999999)
    assert not walk.ok
    assert "999999" in walk.missing
    assert walk.stages == ()


# --- the recursive spine --------------------------------------------------------------------------------


def test_the_spine_reaches_every_kind_at_full_depth(graph_db):
    nodes = call(G.spine, graph_db, BILL, depth=5)
    kinds = {n.kind for n in nodes}
    assert kinds == {"bill", "bill_stage", "bill_stage_sitting", "debate", "division"}


def test_depth_limits_what_comes_back(graph_db):
    """The point of the recursion: one query, a caller-chosen depth."""
    assert {n.kind for n in call(G.spine, graph_db, BILL, depth=0)} == {"bill"}
    assert {n.kind for n in call(G.spine, graph_db, BILL, depth=1)} == {"bill", "bill_stage"}
    assert {n.kind for n in call(G.spine, graph_db, BILL, depth=2)} == {
        "bill", "bill_stage", "bill_stage_sitting"}


def test_two_sittings_on_one_debate_yield_that_debate_once_per_path_and_never_loop(graph_db):
    """Sittings 21 and 22 both point at d-cttee. The walk must terminate and must not emit a node twice on the same
    path — the cycle guard, not the depth cap, is what makes that true.
    """
    nodes = call(G.spine, graph_db, BILL, depth=5)
    debates = [n for n in nodes if n.kind == "debate"]
    assert sorted(n.id for n in debates) == ["d-cttee", "d-cttee", "d-lords", "d-sr"], \
        "once per distinct path, never more"
    assert all(n.depth <= 5 for n in nodes)


def test_every_node_records_the_parent_it_was_reached_through(graph_db):
    nodes = call(G.spine, graph_db, BILL, depth=5)
    root = [n for n in nodes if n.depth == 0]
    assert len(root) == 1 and root[0].parent_id is None
    assert all(n.parent_kind and n.parent_id for n in nodes if n.depth > 0)


def test_the_spine_does_not_wander_into_another_bill(graph_db):
    nodes = call(G.spine, graph_db, BILL, depth=5)
    assert "d-other" not in {n.id for n in nodes if n.kind == "debate"}
    assert "13" not in {n.id for n in nodes if n.kind == "bill_stage"}


@pytest.mark.parametrize("depth", [-1, 6, 99])
def test_a_depth_beyond_the_cap_is_refused_not_clamped(graph_db, depth):
    """Silently giving a caller 5 when they asked for 9 hides that they misunderstood the graph."""
    with pytest.raises(G.GraphError, match="depth"):
        call(G.spine, graph_db, BILL, depth=depth)


def test_an_unknown_bill_walks_to_nothing(graph_db):
    assert call(G.spine, graph_db, 999999, depth=5) == []


# --- debate_walk ----------------------------------------------------------------------------------------


def test_participants_come_from_sql_in_speaking_order_with_counts(graph_db):
    walk = call(G.debate_walk, graph_db, "d-sr")
    assert walk.ok
    assert [p.speaker_name for p in walk.participants] == ["A Member", "B Member"]
    assert [p.contributions for p in walk.participants] == [2, 1]
    assert walk.speech_count == 3, "boilerplate excluded"


def test_a_debate_walk_names_the_bill_it_concerned(graph_db):
    assert call(G.debate_walk, graph_db, "d-sr").bill_ids == (BILL,)


def test_a_debates_divisions_carry_their_caveats(graph_db):
    walk = call(G.debate_walk, graph_db, "d-sr")
    assert [d.id for d in walk.divisions] == [30]
    assert "do not reconcile" in walk.divisions[0].caveat


def test_an_unknown_debate_says_so(graph_db):
    walk = call(G.debate_walk, graph_db, "no-such-debate")
    assert not walk.ok and "no-such-debate" in walk.missing


# --- member_walk ----------------------------------------------------------------------------------------


def test_a_member_walk_returns_their_speeches_and_votes(graph_db):
    walk = call(G.member_walk, graph_db, 1001)
    assert walk.ok and walk.name == "A Member"
    assert [s.speech_id for s in walk.speeches] == ["s1", "s3", "s6"]
    assert {v.division_id for v in walk.votes} == {30, 31}
    assert walk.debate_count == 2


def test_a_date_range_narrows_both_speeches_and_votes(graph_db):
    walk = call(G.member_walk, graph_db, 1001, date(2026, 6, 1), date(2026, 6, 30))
    assert [s.speech_id for s in walk.speeches] == ["s6"]
    assert [v.division_id for v in walk.votes] == []


def test_tellers_are_counted_separately(graph_db):
    """A walk that cannot say who tellered cannot explain a tally that does not add up."""
    walk = call(G.member_walk, graph_db, 1002)
    assert walk.teller_count == 1
    assert [v.is_teller for v in walk.votes] == [True]


def test_a_truncated_speech_list_does_not_shrink_the_reported_total(graph_db):
    """`limit` bounds the list a caller gets; the counts are computed unlimited. A limit that also lowered the total
    would make "how often did they speak" answer "as often as we felt like listing".
    """
    walk = call(G.member_walk, graph_db, 1001, limit=1)
    assert len(walk.speeches) == 1
    assert walk.debate_count == 2, "the total is not the length of the list"


def test_an_unknown_member_says_so(graph_db):
    walk = call(G.member_walk, graph_db, 424242)
    assert not walk.ok and "424242" in walk.missing


# --- speech_to_vote -------------------------------------------------------------------------------------


def test_a_speech_links_to_how_its_speaker_voted_in_that_debate(graph_db):
    """The join that makes "argued for it, then voted against it" a SQL question rather than a reading."""
    result = call(G.speech_to_vote, graph_db, "s2")
    assert result["speaker_name"] == "B Member"
    assert [v["lobby"] for v in result["votes"]] == ["no"]
    assert result["votes"][0]["is_teller"] is True


def test_a_speech_in_a_debate_with_no_division_returns_no_votes(graph_db):
    assert call(G.speech_to_vote, graph_db, "s5")["votes"] == []


def test_an_unknown_speech_says_so(graph_db):
    assert "missing" in call(G.speech_to_vote, graph_db, "no-such-speech")


# --- the gate itself ------------------------------------------------------------------------------------


def test_the_latency_gate_is_the_one_the_brief_states():
    assert G.LATENCY_GATE_MS == 100.0
    assert G.LATENCY_TARGET_MS == 25.0


def test_every_query_runs_under_a_statement_timeout():
    """A graph tool without a timeout is a graph tool that will one day be the reason nothing responds."""
    assert G.STATEMENT_TIMEOUT_MS == 2_000


def test_a_bill_timeline_that_spans_the_sealed_window_says_stages_may_be_withheld():
    """QA pass 3 (finding 11): a stage whose sittings touch the window is not served, and the timeline showed the bill
    without it and without a word. When the served stages lie on both sides of the window, the walk says so.
    """
    from datetime import date

    from answer import graph as G

    def stage(day):
        return G.StageSummary(id=1, stage_name="x", date=day, first_sitting=day, last_sitting=day)

    assert "not yet available" in G._window_note((stage(date(2026, 3, 2)), stage(date(2026, 9, 8))))
    assert G._window_note((stage(date(2026, 2, 2)), stage(date(2026, 3, 8)))) == ""
    assert G._window_note((stage(None),)) == ""
