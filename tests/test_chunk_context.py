"""Chunks after Phase 1 completion: parent windows never contain boilerplate, and every chunk knows its Hansard column
(read, carried across speeches, or inferred before a debate's first marker).
"""

import pytest

from ingest.chunk import ChunkDraft, build_chunks, chunk_debate_texts, speech_to_chunks

CHAIR = "Mr Speaker"
MEMBER = "Jim Shannon (Strangford) (DUP)"


def M(ref):
    return f'<span id="{ref}" class="column-number" data-column-number="{ref}"></span>'


def sents(n):
    return [f"S{i}." for i in range(n)]


# --------------------------------------------------------------------------- build_chunks with a boilerplate mask
# ---------------------------------------------------------------------------


def test_parent_window_skips_boilerplate_sentences_and_stays_full_size():
    flags = [False, False, False, True, False, False, False]
    chunks = build_chunks(sents(7), window=5, boilerplate=flags)
    assert chunks[2].parent_body == "S0. S1. S2. S4. S5."  # S3 is skipped; the window is refilled from the far side
    assert chunks[4].parent_body == "S1. S2. S4. S5. S6."
    assert chunks[0].parent_body == "S0. S1. S2."
    assert chunks[6].parent_body == "S4. S5. S6."


def test_a_boilerplate_child_is_flagged_and_its_parent_is_itself():
    chunks = build_chunks(sents(5), window=5, boilerplate=[False, False, True, False, False])
    assert chunks[2].is_boilerplate is True and chunks[2].parent_body == "S2."
    assert all(not chunks[i].is_boilerplate for i in (0, 1, 3, 4))


def test_sent_idx_still_counts_every_sentence_including_boilerplate():
    chunks = build_chunks(sents(4), window=3, boilerplate=[True, False, True, False])
    assert [c.sent_idx for c in chunks] == [0, 1, 2, 3]
    assert [c.body for c in chunks] == sents(4)


def test_all_boilerplate_speech():
    chunks = build_chunks(sents(3), window=5, boilerplate=[True] * 3)
    assert [c.parent_body for c in chunks] == ["S0.", "S1.", "S2."]


def test_no_boilerplate_means_behaviour_is_unchanged():
    assert build_chunks(sents(6), window=5) == build_chunks(sents(6), window=5, boilerplate=[False] * 6)


@pytest.mark.parametrize("window", [1, 3, 5])
@pytest.mark.parametrize(
    "flags",
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0],
    ],
)
def test_parent_matches_an_independent_oracle(window, flags):
    """Oracle: filter the boilerplate out first, then take a clipped window over what is left."""
    n = len(flags)
    kept = [i for i in range(n) if not flags[i]]
    radius = window // 2
    chunks = build_chunks(sents(n), window, boilerplate=[bool(f) for f in flags])
    for i, chunk in enumerate(chunks):
        if flags[i]:
            assert chunk.parent_body == f"S{i}."
            continue
        pos = kept.index(i)
        expected = " ".join(f"S{k}." for k in kept[max(0, pos - radius) : pos + radius + 1])
        assert chunk.parent_body == expected
        assert not any(chunks[k].body in chunk.parent_body.split(" ") for k in range(n) if flags[k])


def test_boilerplate_mask_must_match_the_sentences():
    with pytest.raises(ValueError):
        build_chunks(sents(3), window=3, boilerplate=[False, False])


# --------------------------------------------------------------------------- speech_to_chunks: flags are computed by
# the rules engine at chunk time ---------------------------------------------------------------------------


def test_chair_formulas_are_flagged_and_excluded_from_windows():
    text = "Order. I call the Minister. The Government are wrong. More text here."
    chunks = speech_to_chunks(text, window=3, speaker_name=CHAIR)
    assert [c.is_boilerplate for c in chunks] == [True, True, False, False]
    assert chunks[2].parent_body == "The Government are wrong. More text here."
    assert chunks[3].parent_body == "The Government are wrong. More text here."
    assert chunks[0].parent_body == "Order."


def test_the_same_words_from_a_member_are_not_boilerplate():
    chunks = speech_to_chunks("Order. I call the Minister. The Government are wrong.", window=3, speaker_name=MEMBER)
    assert not any(c.is_boilerplate for c in chunks)
    assert chunks[1].parent_body == "Order. I call the Minister. The Government are wrong."  # the middle child sees both neighbours


def test_stage_directions_and_give_way_requests_are_flagged_for_any_speaker():
    chunks = speech_to_chunks("The scheme is costly. Will the hon. Member give way? It is not. [Interruption.] It never was.", window=5, speaker_name=MEMBER)
    assert [c.is_boilerplate for c in chunks] == [False, True, False, True, False]
    assert chunks[0].parent_body == "The scheme is costly. It is not. It never was."


def test_default_speaker_is_unknown_so_chair_only_formulas_do_not_fire():
    assert not any(c.is_boilerplate for c in speech_to_chunks("Order. Something else."))


def test_chunkdraft_defaults_keep_old_call_sites_working():
    d = ChunkDraft(0, "Body.", "Body.")
    assert d.is_boilerplate is False and d.column_ref is None and d.column_basis is None


# --------------------------------------------------------------------------- Columns across a debate
# ---------------------------------------------------------------------------


def cols(drafts):
    return [(c.column_ref, c.column_end_ref, c.column_basis) for c in drafts]


def test_columns_before_the_first_marker_are_inferred_then_read_then_carried():
    items = [
        (MEMBER, f"Alpha beta. Gamma {M('101')}delta. Epsilon."),
        ("Some Member (Seat) (Lab)", "Zeta."),
        ("Another Member (Seat) (Con)", f"{M('102')}Eta. Theta."),
    ]
    first, second, third = chunk_debate_texts(items, window=3)
    assert cols(first) == [("100", None, "inferred"), ("100", "101", "inferred"), ("101", None, "marker")]
    assert cols(second) == [("101", None, "carried")]
    assert cols(third) == [("102", None, "marker"), ("102", None, "marker")]
    assert [c.column_number for c in first] == [100, 100, 101]


def test_a_marker_in_a_later_speech_overrides_the_carry():
    items = [(MEMBER, f"One. {M('50')}Two."), (MEMBER, f"Three {M('51')}four. Five.")]
    _, second = chunk_debate_texts(items, window=3)
    assert cols(second) == [("50", "51", "carried"), ("51", None, "marker")]


def test_speeches_before_the_first_marker_of_the_debate_are_inferred():
    items = [(MEMBER, "Early speech."), (MEMBER, f"Later {M('30')}speech.")]
    first, second = chunk_debate_texts(items, window=3)
    assert cols(first) == [("29", None, "inferred")]
    assert cols(second) == [("29", "30", "inferred")]


def test_debate_without_any_marker_has_no_columns():
    (a, b) = chunk_debate_texts([(MEMBER, "One."), (MEMBER, "Two.")], window=3)
    assert cols(a) == cols(b) == [(None, None, None)]


def test_grand_committee_refs_infer_with_their_suffix():
    (a,) = chunk_debate_texts([(MEMBER, f"Before. {M('278GC')}After.")], window=3)
    assert cols(a) == [("277GC", None, "inferred"), ("278GC", None, "marker")]
    assert a[1].column_number == 278


def test_an_empty_speech_neither_breaks_nor_resets_the_carry():
    items = [(MEMBER, f"{M('7')}One."), (MEMBER, ""), (MEMBER, "Two.")]
    a, b, c = chunk_debate_texts(items, window=3)
    assert b == [] and cols(c) == [("7", None, "carried")]


def test_debate_texts_returns_one_list_per_speech_in_order():
    out = chunk_debate_texts([(MEMBER, "Alpha."), (MEMBER, ""), (MEMBER, "Bravo one. Charlie two.")], window=3)
    assert [len(x) for x in out] == [1, 0, 2]


def test_speech_to_chunks_without_a_carry_leaves_unmarked_text_unknown():
    assert all(c.column_ref is None for c in speech_to_chunks("One. Two."))
    chunks = speech_to_chunks(f"One. {M('5')}Two.")
    assert cols(chunks) == [(None, None, None), ("5", None, "marker")]
