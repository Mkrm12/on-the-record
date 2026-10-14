"""Text primitives: column-marker extraction, sentence spans with offsets, and column assignment."""

import pytest

from ingest.text import (
    ColumnInfo,
    Marker,
    Span,
    assign_columns,
    clean_text,
    clean_text_with_markers,
    column_number,
    previous_ref,
    split_sentence_spans,
    split_sentences,
)


def M(ref, extra=""):
    return f'<span id="{ref}" class="column-number" data-column-number="{ref}"{extra}></span>'


# --------------------------------------------------------------------------- clean_text_with_markers
# ---------------------------------------------------------------------------


def test_marker_mid_text_records_the_offset_of_the_next_word():
    cleaned, markers = clean_text_with_markers(f"male {M('1341')}overcapacity. Next.")
    assert cleaned == "male overcapacity. Next."
    assert markers == [Marker(5, "1341")]
    assert cleaned[5:].startswith("overcapacity")


def test_marker_at_start_and_at_end():
    assert clean_text_with_markers(f"{M('12')}Hello there.") == ("Hello there.", [Marker(0, "12")])
    assert clean_text_with_markers(f"Hello there.{M('12')}") == ("Hello there.", [Marker(12, "12")])


def test_two_markers_in_one_speech():
    cleaned, markers = clean_text_with_markers(f"a {M('1')}b {M('2')}c")
    assert cleaned == "a b c" and markers == [Marker(2, "1"), Marker(4, "2")]


def test_marker_between_spaces_does_not_leave_a_double_space():
    cleaned, markers = clean_text_with_markers(f"a {M('1')} b")
    assert cleaned == "a b" and markers == [Marker(2, "1")]


def test_marker_on_its_own_line_does_not_leave_a_blank_line():
    cleaned, markers = clean_text_with_markers(f"First.\n{M('7')}\nSecond.")
    assert cleaned == "First.\nSecond." and markers == [Marker(7, "7")]


def test_marker_between_a_full_stop_and_a_paragraph_break():
    cleaned, markers = clean_text_with_markers(f"material. {M('36')}\r\n\r\nThere is also")
    assert cleaned == "material.\nThere is also"
    assert markers == [Marker(9, "36")]
    assert cleaned[markers[0].offset + 1 :].startswith("There")  # the new column starts with the next paragraph


def test_marker_followed_by_leading_space_at_the_start_is_stripped_and_offset_clamped():
    cleaned, markers = clean_text_with_markers(f"{M('3')} Hello.")
    assert cleaned == "Hello." and markers == [Marker(0, "3")]


@pytest.mark.parametrize("ref", ["278GC", "45WH", "1", "9999"])
def test_refs_are_kept_as_printed(ref):
    assert clean_text_with_markers(f"x {M(ref)}y")[1] == [Marker(2, ref)]


def test_attribute_order_and_extra_attributes_do_not_matter():
    tag = '<span class="column-number" data-column-number="9" id="9" data-house-id="1" data-volume-number="5"></span>'
    assert clean_text_with_markers(f"a {tag}b")[1] == [Marker(2, "9")]
    tag = '<span data-external-id="X" data-column-number="10" class="column-number" id="10"></span>'
    assert clean_text_with_markers(f"a {tag}b")[1] == [Marker(2, "10")]


def test_class_list_containing_column_number():
    tag = '<span class="foo column-number" data-column-number="3"></span>'
    assert clean_text_with_markers(f"a {tag}b")[1] == [Marker(2, "3")]


def test_marker_without_a_value_is_dropped_not_guessed():
    cleaned, markers = clean_text_with_markers('a <span class="column-number"></span>b')
    assert cleaned == "a b" and markers == []


def test_other_spans_are_just_tags():
    cleaned, markers = clean_text_with_markers('a <span class="other">b</span> c')
    assert cleaned == "a b c" and markers == []


def test_a_cross_reference_span_is_a_citation_not_a_column_marker():
    # Real Hansard: quoted Official Report references carry data-column-number too, but cite ANOTHER column.
    cite = '<span id="vol_col26" class="cross-reference" data-house-id="1" data-column-number="26" data-volume-number="" data-external-id="A23D">col. 26</span>'
    cleaned, markers = clean_text_with_markers(f"as said {cite} earlier")
    assert markers == [] and cleaned == "as said col. 26 earlier"
    assert clean_text_with_markers('a <span class="cross-reference" data-column-number="26"></span>b')[1] == []


def test_marker_inside_emphasis():
    cleaned, markers = clean_text_with_markers(f"<em>word{M('5')}next</em>")
    assert cleaned == "wordnext" and markers == [Marker(4, "5")]


def test_empty_input():
    assert clean_text_with_markers(None) == ("", [])
    assert clean_text_with_markers("") == ("", [])
    assert clean_text_with_markers("  ") == ("", [])


@pytest.mark.parametrize(
    "raw",
    [
        f"male {M('1341')}overcapacity. Next.",
        f"{M('12')}Hello there.",
        f"Hello there.{M('12')}",
        f"a {M('1')} b",
        f"First.\n{M('7')}\nSecond.",
        f"{M('3')} Hello.",
        f"<p>One.</p>{M('4')}<p>Two &amp; three.</p>",
        f"He <em>rose</em>—and {M('8')}sat.",
        f"a  {M('9')} b",
        f"line one\r\n\r\n{M('10')}line two",
        f'<Question>4.  <QuestionText>Whether {M("11")}he has.</QuestionText></Question>',
        f"{M('1')}{M('2')}{M('3')}",
        "no markers at all",
        # Real shapes that used to leave a space before the newline (9 of 2,484 marked speeches).
        f"material. {M('36')}\r\n\r\nThere is also",
        f"the bottle”.\r {M('173')}\r\n\r\nNor are the",
        f"a\n {M('1')} b",
        f"a {M('1')}\n b",
        f"a \n{M('1')}\n b",
        f"a \n {M('1')} \n b",
        f"x {M('1')} {M('2')} \n y",
    ],
)
def test_cleaned_text_is_identical_to_clean_text(raw):
    """Offsets are only meaningful if the marker-aware text is exactly the text everything else sees."""
    assert clean_text_with_markers(raw)[0] == clean_text(raw)


def test_offsets_always_fall_inside_the_text():
    for raw in (f"{M('1')}{M('2')} x {M('3')}", f"a{M('1')}", f"{M('1')}"):
        cleaned, markers = clean_text_with_markers(raw)
        assert all(0 <= m.offset <= len(cleaned) for m in markers)
        assert [m.offset for m in markers] == sorted(m.offset for m in markers)


# --------------------------------------------------------------------------- split_sentence_spans
# ---------------------------------------------------------------------------


def test_spans_have_correct_offsets():
    assert split_sentence_spans("One. Two.") == [Span("One.", 0, 4), Span("Two.", 5, 9)]


def test_paragraph_boundaries_and_offsets():
    assert split_sentence_spans("Head\nBody text.") == [Span("Head", 0, 4), Span("Body text.", 5, 15)]


def test_continuation_across_a_paragraph_break_keeps_the_start_of_the_first_piece():
    spans = split_sentence_spans("quote”\nhad done. Next.")
    assert spans[0] == Span("quote” had done.", 0, 16)
    assert spans[1].text == "Next." and spans[1].start == 17


def test_indented_paragraphs():
    assert split_sentence_spans("  Indented. Two.") == [Span("Indented.", 2, 11), Span("Two.", 12, 16)]


def test_fragments_with_no_letters_are_dropped_but_offsets_stay_true():
    text = "He spoke. 65.]\nShe replied."
    spans = split_sentence_spans(text)
    assert [s.text for s in spans] == ["He spoke.", "She replied."]
    assert text[spans[1].start : spans[1].end] == "She replied."


@pytest.mark.parametrize(
    "text",
    [
        "I thank my hon. Friend for her intervention. I will give way.",
        "Clause 5 Amendment\nI beg to move the amendment.",
        "It adopted “Just ask ‘Could it be sepsis?’” campaign in 2016 when he was Secretary.",
        "The scheme costs £1.5 billion. That is 2.3 per cent of spend.",
        "Order.\n[Interruption.]\nThe hon. Member may continue.",
        "",
        "   \n  ",
    ],
)
def test_spans_and_split_sentences_agree_and_locate_themselves(text):
    spans = split_sentence_spans(text)
    assert [s.text for s in spans] == split_sentences(text)
    for s in spans:
        assert text[s.start : s.end].split()[0] == s.text.split()[0]  # the span really is where it claims
        assert s.start < s.end <= len(text)


# --------------------------------------------------------------------------- column_number / previous_ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ref,number", [("1341", 1341), ("278GC", 278), ("45WH", 45), ("GC12", None), ("", None), (None, None)])
def test_column_number(ref, number):
    assert column_number(ref) == number


@pytest.mark.parametrize(
    "ref,previous",
    [("1341", "1340"), ("278GC", "277GC"), ("100", "99"), ("2", "1"), ("1", None), ("0", None), ("GC12", None), (None, None), ("", None)],
)
def test_previous_ref(ref, previous):
    assert previous_ref(ref) == previous


# --------------------------------------------------------------------------- assign_columns
# ---------------------------------------------------------------------------


def columns_for(raw, carry=None):
    cleaned, markers = clean_text_with_markers(raw)
    return assign_columns(split_sentence_spans(cleaned), markers, carry)


def test_no_markers_no_carry_means_unknown():
    assert columns_for("One. Two.") == [ColumnInfo(None, None, None, None)] * 2


def test_no_markers_carry_passes_through_with_its_basis():
    assert columns_for("One. Two.", ("100", "carried")) == [ColumnInfo("100", 100, None, "carried")] * 2
    assert columns_for("One.", ("100", "inferred")) == [ColumnInfo("100", 100, None, "inferred")]


def test_sentence_before_at_and_after_a_marker():
    raw = f"First sentence here. Second one starts {M('101')}and ends. Third one."
    a, b, c = columns_for(raw, ("100", "carried"))
    assert a == ColumnInfo("100", 100, None, "carried")
    assert b == ColumnInfo("100", 100, "101", "carried")  # starts in 100, ends in 101
    assert c == ColumnInfo("101", 101, None, "marker")


def test_crossing_sentence_without_a_carry_has_an_unknown_start_and_a_known_end():
    a, b, c = columns_for(f"First sentence here. Second one starts {M('101')}and ends. Third one.")
    assert a == ColumnInfo(None, None, None, None)
    assert b == ColumnInfo(None, None, "101", None)
    assert c == ColumnInfo("101", 101, None, "marker")


def test_marker_exactly_at_a_sentence_start_belongs_to_that_sentence():
    a, b = columns_for(f"One. {M('5')}Two three.", ("4", "carried"))
    assert a == ColumnInfo("4", 4, None, "carried")
    assert b == ColumnInfo("5", 5, None, "marker")


def test_two_markers_in_one_sentence_end_in_the_last():
    (s,) = columns_for(f"It starts {M('7')}here and runs {M('8')}on.", ("6", "carried"))
    assert (s.ref, s.end_ref, s.basis) == ("6", "8", "carried")


def test_suffixed_refs_keep_their_suffix_and_numeric_part():
    (s,) = columns_for(f"{M('278GC')}Grand committee text.")
    assert s == ColumnInfo("278GC", 278, None, "marker")


def test_marker_after_the_last_sentence_changes_nothing_for_it():
    (s,) = columns_for(f"Only sentence.{M('9')}", ("8", "carried"))
    assert s == ColumnInfo("8", 8, None, "carried")
