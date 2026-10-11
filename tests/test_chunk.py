"""Cleaning, sentence splitting, and parent-window boundaries."""

import pytest

from ingest.chunk import (
    DEFAULT_WINDOW,
    ChunkDraft,
    build_chunks,
    clean_text,
    speech_to_chunks,
    split_sentences,
    window_bounds,
)


def sents(n: int) -> list[str]:
    return [f"S{i}." for i in range(n)]


def parent_of(chunks: list[ChunkDraft], i: int) -> list[str]:
    return chunks[i].parent_body.split(" ")


# --------------------------------------------------------------------------- window_bounds: half-open [lo, hi),
# symmetric radius = window // 2, clipped ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index,total,window,expected",
    [
        (0, 10, 5, (0, 3)),  # first sentence: nothing before it
        (1, 10, 5, (0, 4)),
        (2, 10, 5, (0, 5)),  # first full window
        (5, 10, 5, (3, 8)),  # interior: two either side
        (7, 10, 5, (5, 10)),  # last full window
        (8, 10, 5, (6, 10)),
        (9, 10, 5, (7, 10)),  # last sentence: nothing after it
        (0, 1, 5, (0, 1)),  # single-sentence speech
        (1, 3, 5, (0, 3)),  # window wider than the speech
        (4, 10, 1, (4, 5)),  # window=1 is the child alone
        (4, 10, 3, (3, 6)),
    ],
)
def test_window_bounds(index, total, window, expected):
    assert window_bounds(index, total, window) == expected


@pytest.mark.parametrize("window", [0, -1, 2, 4, 6])
def test_window_must_be_positive_and_odd(window):
    # An even window has no centre, so "surrounding" would be ambiguous.
    with pytest.raises(ValueError):
        window_bounds(0, 10, window)
    with pytest.raises(ValueError):
        build_chunks(sents(3), window)


# --------------------------------------------------------------------------- build_chunks
# ---------------------------------------------------------------------------


def test_empty_speech_yields_no_chunks():
    assert build_chunks([]) == []


def test_single_sentence_parent_is_itself():
    (only,) = build_chunks(["Order."])
    assert only == ChunkDraft(sent_idx=0, body="Order.", parent_body="Order.")


def test_default_window_is_five():
    assert DEFAULT_WINDOW == 5


def test_body_is_the_sentence_and_sent_idx_is_contiguous():
    chunks = build_chunks(sents(7))
    assert [c.sent_idx for c in chunks] == list(range(7))
    assert [c.body for c in chunks] == sents(7)


def test_ten_sentences_window_five_exact_parents():
    chunks = build_chunks(sents(10), window=5)
    assert parent_of(chunks, 0) == ["S0.", "S1.", "S2."]  # first: clipped left
    assert parent_of(chunks, 1) == ["S0.", "S1.", "S2.", "S3."]
    assert parent_of(chunks, 2) == ["S0.", "S1.", "S2.", "S3.", "S4."]
    assert parent_of(chunks, 5) == ["S3.", "S4.", "S5.", "S6.", "S7."]
    assert parent_of(chunks, 7) == ["S5.", "S6.", "S7.", "S8.", "S9."]
    assert parent_of(chunks, 8) == ["S6.", "S7.", "S8.", "S9."]
    assert parent_of(chunks, 9) == ["S7.", "S8.", "S9."]  # last: clipped right


def test_speech_shorter_than_window_every_parent_is_whole_speech():
    chunks = build_chunks(sents(3), window=5)
    assert all(c.parent_body == "S0. S1. S2." for c in chunks)


def test_two_sentence_speech():
    a, b = build_chunks(["One.", "Two."])
    assert a.parent_body == b.parent_body == "One. Two."


@pytest.mark.parametrize("window", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("n", range(1, 15))
def test_parent_matches_brute_force_slice(n, window):
    """Independent oracle: recompute each window from first principles."""
    radius = window // 2
    chunks = build_chunks(sents(n), window)
    for i, chunk in enumerate(chunks):
        expected = [f"S{j}." for j in range(n) if abs(j - i) <= radius]
        assert chunk.parent_body.split(" ") == expected
        assert chunk.body in chunk.parent_body
        # The child appears exactly once, at its own position in the window.
        assert chunk.parent_body.split(" ").count(chunk.body) == 1


# --------------------------------------------------------------------------- clean_text: Hansard text is ~1/3 HTML;
# markup must not reach the embeddings ---------------------------------------------------------------------------


def test_clean_removes_column_number_markers():
    raw = 'Mia’s family.<span id="148" class="column-number" data-column-number="148"></span>'
    assert clean_text(raw) == "Mia’s family."


def test_clean_unwraps_inline_tags_without_adding_spaces():
    assert clean_text("He <em>rose</em>—and <strong>sat</strong>.") == "He rose—and sat."


def test_clean_unescapes_entities():
    assert clean_text("Tom &amp; Jerry &#163;5") == "Tom & Jerry £5"


def test_clean_written_question_wrappers_become_paragraph_breaks():
    raw = "<Question><QNum>149</QNum><QuestionText>To ask about water.</QuestionText></Question>"
    lines = clean_text(raw).split("\n")
    assert [line for line in lines if line] == ["149", "To ask about water."]


def test_clean_committee_transcript_question_label_stays_with_its_question():
    # Evidence sessions: <strong>Q</strong><QNum/> then the question, all one paragraph.
    raw = (
        '<strong>Q</strong><strong><QNum HRSContentId="{5D5F}"> </QNum></strong> '
        "In your experience, how often does article 8 matter?"
    )
    assert clean_text(raw) == "Q In your experience, how often does article 8 matter?"


def test_clean_leaves_bare_less_than_alone():
    assert clean_text("Where a < b and 3 <4 holds.") == "Where a < b and 3 <4 holds."


def test_clean_normalises_nbsp_and_runs_of_spaces():
    assert clean_text("a  b   c") == "a b c"


def test_clean_handles_none_and_blank():
    assert clean_text(None) == ""
    assert clean_text("  \n \t ") == ""


# --------------------------------------------------------------------------- split_sentences: Hansard abbreviations
# must not create fake sentence breaks ---------------------------------------------------------------------------


def test_hon_friend_is_not_a_sentence_boundary():
    text = "I thank my hon. Friend for her intervention. I will give way."
    assert split_sentences(text) == [
        "I thank my hon. Friend for her intervention.",
        "I will give way.",
    ]


@pytest.mark.parametrize(
    "title",
    ["Mr.", "Mrs.", "Ms.", "Dr.", "Rt. Hon.", "Prof.", "Rev.", "St."],
)
def test_titles_are_not_boundaries(title):
    text = f"I agree with {title} Smith on that. He is right."
    assert split_sentences(text) == [f"I agree with {title} Smith on that.", "He is right."]


def test_hon_and_gallant_member_variants():
    text = "The right hon. Member for X and the hon. and learned Lady agree. So do I."
    assert split_sentences(text) == [
        "The right hon. Member for X and the hon. and learned Lady agree.",
        "So do I.",
    ]


def test_decimals_and_currency_do_not_split():
    text = "The scheme costs £1.5 billion. That is 2.3 per cent of spend."
    assert split_sentences(text) == [
        "The scheme costs £1.5 billion.",
        "That is 2.3 per cent of spend.",
    ]


def test_question_and_exclamation_marks_split():
    assert split_sentences("Is that right? Yes! It is.") == ["Is that right?", "Yes!", "It is."]


def test_paragraph_break_is_a_hard_boundary_even_without_punctuation():
    # A heading with no full stop must not be glued onto the next paragraph.
    text = "Clause 5 Amendment\nI beg to move the amendment."
    assert split_sentences(text) == ["Clause 5 Amendment", "I beg to move the amendment."]


def test_fragments_with_no_letters_are_dropped():
    # Column references like "65.]" left behind by interruptions carry no meaning.
    assert split_sentences("He spoke. 65.]\nShe replied.") == ["He spoke.", "She replied."]


def test_interruption_marker_is_kept():
    assert "[Interruption.]" in split_sentences("Order.\n[Interruption.]\nThe hon. Member may continue.")


def test_paragraph_break_inside_a_quoted_sentence_is_rejoined():
    # Real Hansard: a block quote is cut mid-sentence by a paragraph break.
    text = "He called it “one of the best things the last Government”\nhad done. It was a shame."
    assert split_sentences(text) == [
        "He called it “one of the best things the last Government” had done.",
        "It was a shame.",
    ]


def test_paragraph_after_terminal_punctuation_stays_separate_even_if_lowercase():
    # Only an unfinished sentence may be continued by the next paragraph.
    assert split_sentences("He left.\nand then nothing.") == ["He left.", "and then nothing."]


def test_lowercase_after_question_mark_inside_quote_is_rejoined():
    # Punkt always breaks at "?", even inside "...sepsis?’” campaign in 2016".
    text = "It adopted “Just ask ‘Could it be sepsis?’” campaign in 2016 when he was Secretary."
    assert split_sentences(text) == [text]


def test_sentence_starting_with_quote_or_digit_is_not_treated_as_continuation():
    assert split_sentences("He said this.\n“Quote begins.”\n2024 was hard.") == [
        "He said this.",
        "“Quote begins.”",
        "2024 was hard.",
    ]


def test_blank_input_yields_nothing():
    assert split_sentences("") == []
    assert split_sentences("  \n\n ") == []


def test_no_sentence_has_leading_or_trailing_whitespace():
    for s in split_sentences("  First one.  \n\n  Second one.   Third one.  "):
        assert s == s.strip() and s


# --------------------------------------------------------------------------- speech_to_chunks: the full raw-text ->
# chunk pipeline ---------------------------------------------------------------------------


def test_speech_to_chunks_end_to_end_with_markup():
    raw = (
        "My hon. Friend is right.<span class=\"column-number\"></span> "
        "Costs are up &amp; rising. Where next? Home. Away. Back again."
    )
    chunks = speech_to_chunks(raw, window=3)
    assert [c.body for c in chunks] == [
        "My hon. Friend is right.",
        "Costs are up & rising.",
        "Where next?",
        "Home.",
        "Away.",
        "Back again.",
    ]
    assert chunks[0].parent_body == "My hon. Friend is right. Costs are up & rising."
    assert chunks[3].parent_body == "Where next? Home. Away."
    assert chunks[5].parent_body == "Away. Back again."


def test_speech_to_chunks_markup_only_speech_yields_nothing():
    assert speech_to_chunks('<span class="column-number"></span>') == []
    assert speech_to_chunks(None) == []
    assert speech_to_chunks("") == []


def test_each_debate_is_committed_as_it_is_chunked(scratch, scratch_dsn):
    """Durability, not tidiness. The SELECT that lists the pending debates opens an implicit transaction, and inside
    one `conn.transaction` is only a savepoint (gotcha 37) -- so without an explicit commit first, 6,836 debates and
    1.19M chunks all sit in ONE transaction for the better part of an hour: invisible from outside, lost entirely if
    the run is killed, and holding vacuum off the table throughout.
    """
    import psycopg

    from app.db import make_pool
    from ingest import chunk as chunk_mod
    from ingest.runtime import run

    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('TC', 'test', '2026-09-01', '2026-09-30', 'dev') ON CONFLICT DO NOTHING")
    for i in range(3):
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, window_id, corpus_version) "
                        "VALUES (%s, '2026-09-01', 'Commons', 'A debate', 'Commons Chamber', 'TC', 'test')", (f"d{i}",))
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, text, window_id) "
                        "VALUES (%s, %s, 0, 'Ms Test', %s, 'TC')",
                        (f"s{i}", f"d{i}", "One sentence here. And a second one."))

    seen: list[int] = []

    async def go():
        pool = make_pool(scratch_dsn, min_size=1, max_size=2)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                cur = await conn.execute(chunk_mod.SQL_PENDING_DEBATES, {"rebuild": False, "limit": None})
                ids = [r[0] for r in await cur.fetchall()]
                await conn.commit()
                await conn.set_autocommit(True)
                for debate_id in ids:
                    await chunk_mod.chunk_debate(conn, debate_id, chunk_mod.DEFAULT_WINDOW, chunk_mod.Stats())
                    with psycopg.connect(scratch_dsn) as watcher:
                        seen.append(watcher.execute("SELECT count(*) FROM chunk").fetchone()[0])
        finally:
            await pool.close()

    run(go())
    assert len(seen) == 3
    assert seen == sorted(seen) and seen[0] > 0, f"nothing was visible until the end: {seen}"
    assert seen[-1] == scratch.execute("SELECT count(*) FROM chunk").fetchone()[0]
