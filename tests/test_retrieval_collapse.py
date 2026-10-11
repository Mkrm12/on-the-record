"""Collapsing sibling hits before they reach the reranker."""

from dataclasses import dataclass


from ingest.chunk import DEFAULT_WINDOW, window_bounds
from retrieval.collapse import OVERLAP_SPAN, WINDOW_RADIUS, Cluster, collapse


@dataclass(frozen=True)
class H:
    chunk_id: int
    speech_id: str
    sent_idx: int
    parent_body: str = ""

    def __post_init__(self):
        if not self.parent_body:
            object.__setattr__(self, "parent_body", f"window of {self.speech_id} around {self.sent_idx}")


def ids(clusters):
    return [[m.chunk_id for m in c.members] for c in clusters]


def test_the_constants_match_the_chunkers_window():
    assert WINDOW_RADIUS == DEFAULT_WINDOW // 2 == 2 and OVERLAP_SPAN == 2 * WINDOW_RADIUS == 4


def test_the_span_is_exactly_where_two_sentences_windows_stop_sharing_a_sentence():
    total = 40
    for i in range(10, 20):
        a = set(range(*window_bounds(i, total, DEFAULT_WINDOW)))
        for gap in range(1, 8):
            b = set(range(*window_bounds(i + gap, total, DEFAULT_WINDOW)))
            assert bool(a & b) == (gap <= OVERLAP_SPAN), (i, gap)


def test_nothing_in_nothing_out():
    assert collapse([]) == []


def test_a_single_hit_is_its_own_cluster():
    (c,) = collapse([H(1, "s1", 5)])
    assert c.anchor.chunk_id == 1 and c.siblings == [] and ids([c]) == [[1]]


def test_neighbouring_sentences_of_one_speech_become_one_candidate_anchored_on_the_best_ranked():
    hits = [H(1, "s1", 5), H(2, "s1", 6), H(3, "s1", 4), H(4, "s1", 9)]  # best first
    (c,) = collapse(hits)
    assert c.anchor.chunk_id == 1 and ids([c]) == [[1, 2, 3, 4]]  # 9 is exactly 4 away from 5: its window still overlaps


def test_a_hit_five_or_more_sentences_away_is_a_separate_candidate():
    assert ids(collapse([H(1, "s1", 5), H(2, "s1", 10)])) == [[1], [2]]


def test_clusters_are_anchored_not_chained_so_one_long_run_cannot_grow_without_bound():
    hits = [H(1, "s1", 5), H(2, "s1", 9), H(3, "s1", 13), H(4, "s1", 17)]  # each 4 from the next, but only 1 anchors 2
    assert ids(collapse(hits)) == [[1, 2], [3, 4]]


def test_the_better_ranked_hit_is_the_anchor_whatever_its_position_in_the_speech():
    assert collapse([H(1, "s1", 8), H(2, "s1", 6)])[0].anchor.chunk_id == 1
    assert collapse([H(2, "s1", 6), H(1, "s1", 8)])[0].anchor.chunk_id == 2


def test_the_same_position_in_different_speeches_is_never_merged():
    assert ids(collapse([H(1, "s1", 5), H(2, "s2", 5), H(3, "s3", 6)])) == [[1], [2], [3]]


def test_a_third_speech_hit_lands_in_its_own_speechs_cluster_not_the_first_one_seen():
    hits = [H(1, "s1", 5), H(2, "s2", 5), H(3, "s1", 6), H(4, "s2", 7)]
    assert ids(collapse(hits)) == [[1, 3], [2, 4]]


def test_identical_parent_text_in_different_speeches_collapses_to_the_better_ranked():
    hits = [H(1, "s1", 0, "Yes."), H(2, "s2", 0, "Yes."), H(3, "s3", 0, "Something else.")]
    clusters = collapse(hits)
    assert ids(clusters) == [[1, 2], [3]]
    assert {m.speech_id for m in clusters[0].members} == {"s1", "s2"}  # both speeches stay visible on the cluster


def test_clusters_come_out_in_the_order_their_anchors_ranked():
    hits = [H(1, "s2", 1), H(2, "s1", 1), H(3, "s2", 2), H(4, "s3", 1)]
    assert [c.anchor.chunk_id for c in collapse(hits)] == [1, 2, 4]


def test_the_limit_counts_clusters_not_hits():
    hits = [H(1, "s1", 1), H(2, "s1", 2), H(3, "s1", 3), H(4, "s2", 1), H(5, "s3", 1), H(6, "s4", 1)]
    assert ids(collapse(hits, limit=2)) == [[1, 2, 3], [4]]
    assert collapse(hits, limit=0) == []
    assert len(collapse(hits, limit=None)) == 4


def test_a_chunk_that_appears_twice_is_counted_once():
    assert ids(collapse([H(1, "s1", 5), H(1, "s1", 5), H(2, "s9", 5)])) == [[1], [2]]


def test_the_span_can_be_changed():
    hits = [H(1, "s1", 5), H(2, "s1", 6)]
    assert ids(collapse(hits, span=0)) == [[1], [2]]
    assert ids(collapse(hits, span=1)) == [[1, 2]]


def test_a_cluster_is_a_plain_container():
    a, b = H(1, "s", 1), H(2, "s", 2)
    c = Cluster(a, [b])
    assert c.members == [a, b] and Cluster(a).siblings == []


def test_every_input_chunk_ends_up_in_exactly_one_cluster():
    hits = [H(i, f"s{i % 4}", i % 11, f"p{i % 17}") for i in range(80)]
    clusters = collapse(hits)
    members = [m.chunk_id for c in clusters for m in c.members]
    assert sorted(members) == sorted({h.chunk_id for h in hits}) and len(members) == len(set(members))
