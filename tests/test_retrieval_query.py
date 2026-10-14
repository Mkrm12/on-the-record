"""What a user typed becomes something safe to search with (pure functions, no database)."""

import pytest

from retrieval.query import MAX_QUERY_CHARS, clean_query


@pytest.mark.parametrize("text", [None, "", " ", "\n\t \r", "...", "?!", "—", "[]", "\x00", "​​", "\ud800", "😀😀", 42, b"bytes", ["a"]])
def test_nothing_searchable_gives_an_empty_query(text):
    assert clean_query(text) == ("", False)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What did the minister say?", "What did the minister say?"),
        ("  spaced   out \n query\t here ", "spaced out query here"),
        ("Line one\r\nline two", "Line one line two"),
        ("NHS\x00 waiting\x00lists", "NHS waiting lists"),  # a NUL is dropped or becomes a gap, never kept
        ("abc \ud800 def", "abc def"),  # a lone surrogate (JSON "\ud800") would crash the tokenizer downstream
        ("wa​iting li﻿sts", "waiting lists"),  # zero-width space and BOM are invisible but split words
        ("abc ‮ def", "abc def"),  # a right-to-left override
        ("bell\x07 and\x1b[31m escape", "bell and [31m escape"),  # control characters become gaps
        ("£35 billion for 2.4% of GDP", "£35 billion for 2.4% of GDP"),
        ("naïve café 😀 query", "naïve café 😀 query"),
    ],
)
def test_dirty_text_is_cleaned_but_real_text_is_not_rewritten(text, expected):
    assert clean_query(text) == (expected, False)


@pytest.mark.parametrize("text", ["مرحبا بالعالم", "你好世界，议会辩论", "Привет мир", "ελληνικά κείμενο", "日本語のクエリ"])
def test_other_scripts_are_searchable(text):
    assert clean_query(text) == (text, False)


@pytest.mark.parametrize(
    "text",
    ["'; DROP TABLE chunk; --", 'a" OR 1=1 --', "a & b | !c", "x:* <-> y", "name\\'; --", "%s %(name)s {0} ${x}", "<script>alert(1)</script>", "a'b\"c`d"],
)
def test_hostile_strings_pass_through_untouched_because_they_are_only_ever_used_as_query_parameters(text):
    cleaned, truncated = clean_query(text)
    assert cleaned == " ".join(text.split()) and not truncated


def test_a_query_at_the_limit_is_kept_and_one_beyond_it_is_cut_and_flagged():
    exact = "a" * MAX_QUERY_CHARS
    assert clean_query(exact) == (exact, False)
    cleaned, truncated = clean_query("a" * (MAX_QUERY_CHARS + 1))
    assert truncated and len(cleaned) == MAX_QUERY_CHARS


def test_a_long_paste_is_cut_at_a_word_boundary():
    text = "houses " * 1000  # 7 characters a word, so a cut at 2000 would land inside a word (2000 / 7 is not whole)
    cleaned, truncated = clean_query(text)
    assert truncated and len(cleaned) <= MAX_QUERY_CHARS and cleaned.endswith("houses") and not cleaned.endswith(" ")
    assert set(cleaned.split()) == {"houses"}  # no half word at the end


def test_a_huge_single_word_is_cut_hard_because_there_is_no_boundary_to_cut_at():
    cleaned, truncated = clean_query("x" * 100_000)
    assert truncated and len(cleaned) == MAX_QUERY_CHARS


def test_whitespace_that_pads_a_query_over_the_limit_is_collapsed_first_so_it_is_not_truncated():
    cleaned, truncated = clean_query("hello" + " " * 50_000 + "world")
    assert cleaned == "hello world" and not truncated


def test_the_result_is_stable_cleaning_twice_changes_nothing():
    dirty = "  a\x00b \ud800 c​d \n e  "
    once = clean_query(dirty)[0]
    assert clean_query(once) == (once, False)


def test_a_truncated_query_that_is_cut_in_the_middle_of_only_punctuation_is_still_searchable_text():
    cleaned, truncated = clean_query("word " + "." * (MAX_QUERY_CHARS + 50))
    assert cleaned.startswith("word") and truncated
