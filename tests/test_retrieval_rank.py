"""The pure decisions inside the search engine: what the reranker reads, how its scores are used, and how a hit is
expanded to its surrounding sentences. Decided by experiment.
"""

import pytest

from retrieval.hybrid import Candidate, assemble_context, rank_candidates, rerank_text


def cand(**kw):
    base = dict(chunk_id=1, speech_id="s1", debate_id="d1", seq=10, sent_idx=3, body="The sentence.", parent_body="The window. The sentence. More.",
                speaker=None, party=None, office=None, heading=None, rrf=0.03, dense_rank=1, sparse_rank=None, speaker_rank=None)
    base.update(kw)
    return Candidate(**base)


# --- what the reranker reads ------------------------------------------------------------------------------------


def test_the_rerank_text_is_the_sentence_headed_by_speaker_party_office_and_debate():
    c = cand(speaker="Alison McGovern", party="Labour", office="The Minister for Housing", heading="Housing Bill", body="Rents are too high.")
    assert rerank_text(c) == "Alison McGovern (Labour), The Minister for Housing, in the debate 'Housing Bill': Rents are too high."


@pytest.mark.parametrize(
    "fields,expected",
    [
        (dict(speaker="Ann Lee", heading="Housing"), "Ann Lee, in the debate 'Housing': S."),
        (dict(speaker="Ann Lee", party="Labour"), "Ann Lee (Labour): S."),
        (dict(speaker="Ann Lee"), "Ann Lee: S."),
        (dict(heading="Housing"), "In the debate 'Housing': S."),
        (dict(party="Labour", office="The Minister"), "S."),  # a party or office without a speaker names nobody
        (dict(), "S."),
    ],
)
def test_missing_metadata_just_shortens_the_header(fields, expected):
    assert rerank_text(cand(body="S.", **fields)) == expected


def test_the_rerank_text_never_uses_the_window():
    assert "window" not in rerank_text(cand(speaker="Ann Lee", body="Only this."))


# --- how the reranker's scores are used -----------------------------------------------------------------------------


def order(ranked):
    return [r.index for r in ranked]


def test_off_keeps_the_fused_order_and_ignores_any_scores():
    assert order(rank_candidates(4, [0.1, 0.9, 0.5, 0.2], "off")) == [0, 1, 2, 3]
    assert order(rank_candidates(3, None, "blend")) == [0, 1, 2]  # no scores at all: nothing to blend


def test_replace_sorts_by_score_and_breaks_ties_by_fused_order():
    got = rank_candidates(4, [1.0, 5.0, 5.0, -2.0], "replace")
    assert order(got) == [1, 2, 0, 3] and [r.score for r in got] == [5.0, 5.0, 1.0, -2.0]


def test_blend_is_reciprocal_rank_fusion_of_the_fused_order_and_the_rerank_order():
    # fused ranks 1,2,3; rerank ranks 3,1,2 -> scores 1/61+1/63, 1/62+1/61, 1/63+1/62
    got = rank_candidates(3, [0.1, 0.9, 0.5], "blend", k=60)
    assert order(got) == [1, 0, 2]  # 0.03252 > 0.03227 > 0.03200
    expected = {0: 1 / 61 + 1 / 63, 1: 1 / 62 + 1 / 61, 2: 1 / 63 + 1 / 62}
    assert {r.index: r.score for r in got} == pytest.approx(expected)
    assert order(got) == sorted(expected, key=lambda i: (-expected[i], i))


def test_blend_lets_a_strong_reranker_disagreement_move_a_candidate_up_but_not_unboundedly():
    scores = [0.0] * 20 + [9.0]  # the reranker loves the candidate the fusion ranked 21st
    got = order(rank_candidates(21, scores, "blend"))
    assert got[0] == 0  # the top fused candidate is not thrown out on one voter's say-so
    assert 3 < got.index(20) < 15  # but the reranker's favourite climbs from 21st to about 10th


def test_a_candidate_the_reranker_could_not_score_keeps_its_fused_contribution_only_in_blend_and_goes_last_in_replace():
    scores = [0.5, None, 0.9]
    blend = rank_candidates(3, scores, "blend", k=60)
    assert {r.index: r.score for r in blend}[1] == pytest.approx(1 / 62)
    replace = rank_candidates(3, scores, "replace")
    assert order(replace) == [2, 0, 1] and replace[-1].score is None


def test_every_candidate_appears_exactly_once_in_every_mode():
    for mode in ("off", "replace", "blend"):
        got = order(rank_candidates(6, [3.0, None, 1.0, 1.0, None, 2.0], mode))
        assert sorted(got) == list(range(6)), mode


def test_no_candidates_no_ranking_and_a_bad_mode_is_refused():
    assert rank_candidates(0, [], "blend") == []
    with pytest.raises(ValueError):
        rank_candidates(2, [1.0, 2.0], "shuffle")


def test_scores_must_line_up_with_the_candidates():
    with pytest.raises(ValueError):
        rank_candidates(3, [1.0, 2.0], "blend")


def test_non_finite_scores_are_treated_as_unscored_never_ranked():
    got = rank_candidates(3, [float("nan"), 1.0, float("inf")], "replace")
    assert order(got) == [1, 0, 2]


# --- expanding a hit to its surroundings ---------------------------------------------------------------------------


def rows(*seqs):
    return [(s, s, f"sentence {s}", None) for s in seqs]  # (seq, sent_idx, body, column_ref)


def test_the_context_is_the_contiguous_run_around_the_anchor_inside_the_range():
    got = assemble_context(anchor_seq=10, lo=8, hi=12, rows=rows(*range(5, 20)))
    assert [r[0] for r in got] == [8, 9, 10, 11, 12]


def test_the_context_stops_at_a_gap_such_as_a_boilerplate_sentence_or_the_start_of_the_speech():
    got = assemble_context(anchor_seq=10, lo=6, hi=14, rows=rows(6, 7, 9, 10, 11, 14))  # 8 missing, 12-13 missing
    assert [r[0] for r in got] == [9, 10, 11]


def test_an_anchor_with_no_neighbours_is_its_own_context():
    assert [r[0] for r in assemble_context(10, 8, 12, rows(10))] == [10]


def test_rows_out_of_order_are_sorted_and_a_missing_anchor_gives_nothing():
    assert [r[0] for r in assemble_context(10, 8, 12, rows(11, 9, 10))] == [9, 10, 11]
    assert assemble_context(10, 8, 12, rows(8, 9, 11, 12)) == []
    assert assemble_context(10, 8, 12, []) == []


def test_rows_outside_the_asked_range_are_ignored():
    assert [r[0] for r in assemble_context(10, 10, 10, rows(9, 10, 11))] == [10]
