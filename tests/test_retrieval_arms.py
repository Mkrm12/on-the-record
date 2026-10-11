"""The E1 bake-off arms, against hand-worked values."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from retrieval import arms
from retrieval.collapse import OVERLAP_SPAN, collapse
from retrieval.hybrid import RERANK_TEXT_MODES, Candidate, SearchConfig, rerank_text


@dataclass
class Cand:
    """The fields the arm hooks read off a Candidate."""

    chunk_id: int
    speech_id: str
    debate_id: str
    rrf: float
    body: str = "text"
    sent_idx: int = 0
    parent_body: str = ""


def cand(chunk_id, speech, debate, rrf, **kw):
    kw.setdefault("parent_body", f"window of chunk {chunk_id}")
    return Cand(chunk_id, speech, debate, rrf, **kw)


# --- the property every arm comparison rests on ---------------------------------------------------------


def test_every_arm_default_is_production_behaviour():
    """`SearchConfig` must be untouched by the arms existing. Anything else and the baseline is not the baseline."""
    base = SearchConfig()
    assert base.collapse_span == OVERLAP_SPAN
    assert base.set_pool is None
    assert base.debate_boost == 0
    assert base.stock_penalty == 1.0
    assert base.rerank_text_mode == "sentence"
    assert base.pool == 20


@pytest.mark.parametrize("arm", arms.ARMS, ids=lambda a: a.key)
def test_every_arm_changes_something_and_says_why(arm):
    """An arm that changes nothing would be screened, found to do nothing, and logged as a real negative result -- a
    wrong measurement rather than a missing one.
    """
    base = SearchConfig()
    applied = arm.apply(base)
    assert applied != base, f"{arm.key} is a no-op"
    assert arm.why and len(arm.why) > 20, f"{arm.key} has no stated reason"
    for name in arm.changes:
        assert hasattr(base, name), f"{arm.key} sets {name!r}, which is not a SearchConfig field"


def test_an_arm_that_needs_something_built_says_so():
    assert arms.BY_KEY["B1"].requires, "B1 needs the stock-sentence table and must declare it"
    assert not arms.BY_KEY["S1"].requires


# --- the set-question gate ------------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "who spoke about chalk streams",
    "Who took part in the debate on the Finance Bill?",
    "which MPs mentioned the Chagos islands",
    "which peers opposed the tobacco bill",
    "what did MPs say about digital ID",
    "list the members who voted against",
    "views on public ownership of water",
    "contributions to the Budget debate",
    "the debate on assisted dying",
    "everyone who mentioned SaxaVord",
    "all the speakers in Westminster Hall",
])
def test_the_set_gate_catches_a_set_question(query):
    assert arms.looks_like_a_set_question(query) is True


@pytest.mark.parametrize("query", [
    "when did the Finance Act 2026 receive Royal Assent",
    "how much are we paying for the chagos deal",
    "what did the Chancellor say about growth",
    "who is the Prime Minister",  # a single fact, not a roster
    "what was said about licensing the first spaceport launch",
    "",
])
def test_the_set_gate_leaves_a_single_fact_question_alone(query):
    assert arms.looks_like_a_set_question(query) is False


def test_the_set_gate_is_a_rule_over_text_and_never_a_gold_category():
    """Gating on the gold label would leak the answer into the retrieval and make every set metric meaningless. The
    gate must be computable from the query alone.
    """
    import inspect

    source = inspect.getsource(arms.looks_like_a_set_question)
    assert "category" not in source and "gold" not in source.lower()


# --- S1: one slot per speech ----------------------------------------------------------------------------


def test_s1_makes_a_speech_occupy_exactly_one_slot():
    """Two sentences of one speech, far apart, so the default span leaves them as two results."""
    far = [cand(1, "s1", "d1", 0.9, sent_idx=0), cand(2, "s1", "d1", 0.8, sent_idx=50)]
    assert len(collapse(far, span=OVERLAP_SPAN)) == 2
    merged = collapse(far, span=arms.MERGE_WHOLE_SPEECH)
    assert len(merged) == 1 and merged[0].anchor.chunk_id == 1 and len(merged[0].siblings) == 1


def test_s1_does_not_merge_different_speeches():
    two = [cand(1, "s1", "d1", 0.9), cand(2, "s2", "d1", 0.8, sent_idx=40)]
    assert len(collapse(two, span=arms.MERGE_WHOLE_SPEECH)) == 2


def test_s1_keeps_the_best_chunk_as_the_representative():
    """Its other matched sentences travel with it as evidence, so the anchor must be the best one."""
    merged = collapse([cand(7, "s1", "d1", 0.9, sent_idx=0), cand(8, "s1", "d1", 0.5, sent_idx=30)],
                      span=arms.MERGE_WHOLE_SPEECH)  # fmt: skip
    assert merged[0].anchor.chunk_id == 7
    assert [c.chunk_id for c in merged[0].siblings] == [8]


# --- B1: stock sentences --------------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [
    ("Question put and agreed to.", "Question put, and agreed to!"),
    ("I thank my hon. Friend.", "i thank my hon friend"),
    ("  Order.  Order!  ", "order order"),
])
def test_normalising_makes_two_printings_of_one_stock_sentence_match(a, b):
    assert arms.normalise_sentence(a) == arms.normalise_sentence(b)


def test_normalising_keeps_different_sentences_different():
    assert arms.normalise_sentence("The Bill is read.") != arms.normalise_sentence("The Bill is rejected.")
    assert arms.normalise_sentence(None) == "" and arms.normalise_sentence("") == ""


def test_b1_demotes_a_stock_sentence_below_a_real_one():
    stock = frozenset({arms.normalise_sentence("Question put and agreed to.")})
    got, penalised = arms.penalise_stock(
        [cand(1, "s1", "d1", 0.10, body="Question put and agreed to."),
         cand(2, "s2", "d1", 0.09, body="The Chancellor said growth was 2%.")], stock, 0.5)  # fmt: skip
    assert penalised == 1
    assert [c.chunk_id for c in got] == [2, 1], "0.10 x 0.5 = 0.05 falls below 0.09"


def test_b1_leaves_the_order_alone_when_nothing_is_stock():
    original = [cand(1, "s1", "d1", 0.9), cand(2, "s2", "d1", 0.8)]
    got, penalised = arms.penalise_stock(original, frozenset(), 0.5)
    assert penalised == 0 and [c.chunk_id for c in got] == [1, 2]


def test_b1_is_stable_for_equal_scores():
    """Ties keep the fused order. Without this, the arm's result would depend on sort implementation."""
    got, _ = arms.penalise_stock([cand(1, "s1", "d1", 0.5), cand(2, "s2", "d1", 0.5)], frozenset(), 0.5)
    assert [c.chunk_id for c in got] == [1, 2]


def test_b1_refuses_a_multiplier_outside_the_unit_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        arms.penalise_stock([], frozenset(), 1.5)


# --- D1: the debate leg ---------------------------------------------------------------------------------


def test_debates_are_ranked_by_their_best_chunk_not_by_volume():
    """MaxP. With a sum, a long debate with many mediocre chunks would outrank the one short debate that actually
    answers the question.
    """
    candidates = [
        cand(1, "s1", "long", 0.30), cand(2, "s2", "long", 0.30), cand(3, "s3", "long", 0.30),
        cand(4, "s4", "short", 0.80),
    ]  # fmt: skip
    assert [d for d, _ in arms.rank_debates(candidates)] == ["short", "long"]


def test_debate_ranking_is_stable_for_equal_best_scores():
    candidates = [cand(1, "s1", "d1", 0.5), cand(2, "s2", "d2", 0.5)]
    assert [d for d, _ in arms.rank_debates(candidates)] == ["d1", "d2"]


def test_d1_brings_several_distinct_speeches_of_the_leading_debate_forward():
    """The failure it fixes: the pool spends every slot on one speech's sentences, and the reranker never sees the
    debate's other contributors.
    """
    candidates = [
        cand(1, "s1", "d1", 0.90), cand(2, "s1", "d1", 0.89), cand(3, "s1", "d1", 0.88),  # one speech
        cand(4, "s2", "d1", 0.40), cand(5, "s3", "d1", 0.39),  # its colleagues
    ]  # fmt: skip
    got, promoted = arms.boost_debates(candidates, top_debates=1, per_debate=3)
    assert promoted == 3
    assert [c.speech_id for c in got[:3]] == ["s1", "s2", "s3"], "one slot per speech, not per sentence"


def test_d1_adds_nothing_that_was_not_already_retrieved():
    """It reorders the pool. That matters for the judging gate: it cannot introduce an unjudged document."""
    candidates = [cand(i, f"s{i}", f"d{i % 2}", 1 / i) for i in range(1, 8)]
    got, _ = arms.boost_debates(candidates, top_debates=2, per_debate=2)
    assert sorted(c.chunk_id for c in got) == sorted(c.chunk_id for c in candidates)
    assert len(got) == len(candidates)


def test_d1_is_off_at_zero_and_returns_the_list_untouched():
    candidates = [cand(1, "s1", "d1", 0.9), cand(2, "s2", "d2", 0.8)]
    got, promoted = arms.boost_debates(candidates, top_debates=0, per_debate=3)
    assert promoted == 0 and [c.chunk_id for c in got] == [1, 2]


def test_d1_refuses_a_zero_per_debate():
    with pytest.raises(ValueError, match="at least 1"):
        arms.boost_debates([cand(1, "s1", "d1", 0.9)], top_debates=1, per_debate=0)


def test_d1_round_robins_so_no_single_debate_takes_every_slot():
    """Best of each leading debate first, then the seconds. A debate-major order would spend the front of the pool on
    one debate and defeat the point.
    """
    candidates = [
        cand(1, "a1", "A", 0.9), cand(2, "a2", "A", 0.8),
        cand(3, "b1", "B", 0.7), cand(4, "b2", "B", 0.6),
    ]  # fmt: skip
    got, _ = arms.boost_debates(candidates, top_debates=2, per_debate=2)
    assert [c.speech_id for c in got] == ["a1", "b1", "a2", "b2"]


# --- R1: what the reranker reads ------------------------------------------------------------------------


def _candidate(body: str, parent: str) -> Candidate:
    return Candidate(chunk_id=1, speech_id="s1", debate_id="d1", seq=1, sent_idx=0, body=body,
                     parent_body=parent, speaker="Ms Smith", party="Lab", office=None,
                     heading="Space Industry", rrf=0.5, dense_rank=1, sparse_rank=None,
                     speaker_rank=None)  # fmt: skip


def test_r1_reads_the_window_while_production_reads_the_sentence():
    c = _candidate("The matched sentence.", "Before. The matched sentence. After.")
    assert rerank_text(c) == "Ms Smith (Lab), in the debate 'Space Industry': The matched sentence."
    assert rerank_text(c, "window") == (
        "Ms Smith (Lab), in the debate 'Space Industry': Before. The matched sentence. After.")


def test_r1_falls_back_to_the_sentence_when_there_is_no_window():
    c = _candidate("Only this.", "")
    assert rerank_text(c, "window").endswith("Only this.")


def test_the_header_is_present_in_both_modes():
    """The header is why a cross-encoder can tell that a speaker name in the query matches anything."""
    c = _candidate("Body.", "Window.")
    for mode in RERANK_TEXT_MODES:
        assert "Ms Smith (Lab)" in rerank_text(c, mode) and "Space Industry" in rerank_text(c, mode)


def test_an_unknown_rerank_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be one of"):
        rerank_text(_candidate("a", "b"), "paragraph")


# --- the registry and the bundle ------------------------------------------------------------------------


def test_resolving_nothing_gives_every_implemented_arm():
    assert arms.resolve(None) == list(arms.ARMS)
    assert [a.key for a in arms.resolve(["S1", "P50"])] == ["S1", "P50"]


def test_an_arm_that_is_only_on_paper_is_named_with_what_it_needs():
    with pytest.raises(arms.UnknownArm, match="not implemented"):
        arms.resolve(["C1"])
    with pytest.raises(arms.UnknownArm, match="unknown arm"):
        arms.resolve(["nonsense"])


def test_the_paper_arms_are_declared_so_coverage_cannot_be_overstated():
    assert set(arms.NOT_IMPLEMENTED) == {"D2", "C1", "Q1", "N1", "A1", "O1", "R2"}
    assert all(reason for reason in arms.NOT_IMPLEMENTED.values())
    assert not set(arms.NOT_IMPLEMENTED) & set(arms.BY_KEY), "an arm cannot be both"


def test_a_bundle_applies_every_arms_flags():
    bundled = arms.bundle(arms.resolve(["S1", "P50", "R1"]), SearchConfig())
    assert bundled.collapse_span == arms.MERGE_WHOLE_SPEECH
    assert bundled.pool == 50
    assert bundled.rerank_text_mode == "window"


def test_a_bundle_refuses_two_mutually_exclusive_arms():
    """Letting the last one silently win would make the bundle's identity depend on iteration order."""
    with pytest.raises(arms.UnknownArm, match="mutually exclusive"):
        arms.bundle(arms.resolve(["P30", "P50"]), SearchConfig())


def test_the_exclusive_groups_are_reported_for_stage_b():
    assert sorted(arms.groups(arms.ARMS)["pool"]) == ["P30", "P50", "P50g"]
    assert arms.groups([arms.BY_KEY["S1"]]) == {}


# --- SearchConfig validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(("field", "value"), [
    ("collapse_span", -1),
    ("debate_boost", -1),
    ("debate_speeches", 0),
    ("set_pool", 0),
    ("stock_penalty", 1.5),
    ("stock_penalty", -0.1),
    ("rerank_text_mode", "paragraph"),
])
def test_an_out_of_range_arm_flag_is_refused_at_construction(field, value):
    with pytest.raises(ValueError):
        SearchConfig(**{field: value})


def test_the_arm_flags_accept_their_intended_values():
    SearchConfig(collapse_span=arms.MERGE_WHOLE_SPEECH, set_pool=50, debate_boost=5, debate_speeches=3,
                 stock_penalty=0.5, rerank_text_mode="window")  # fmt: skip
