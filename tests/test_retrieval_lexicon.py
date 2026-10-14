"""The keyword leg's statistics (pure part). The database loader is tested in test_retrieval_hybrid.py."""

import math

import pytest

from retrieval.lexicon import MAX_TERMS, Lexicon, min_mass


def lex(**df):
    return Lexicon(n_docs=1000, avg_len=11.0, df=df)


def test_idf_is_the_lucene_form_and_falls_as_a_term_gets_commoner():
    lx = lex(rare=2, mid=100, common=900)
    assert lx.idf("rare") == pytest.approx(math.log(1 + (1000 - 2 + 0.5) / (2 + 0.5)))
    assert lx.idf("rare") > lx.idf("mid") > lx.idf("common") > 0


def test_idf_stays_positive_even_for_a_term_in_every_document():
    assert lex(everywhere=1000).idf("everywhere") > 0


def test_an_unknown_term_has_the_highest_idf_but_is_never_selected_because_it_cannot_match():
    lx = lex(seen=5)
    assert lx.idf("never") > lx.idf("seen")
    assert lx.select_terms(["never", "seen"]) == [("seen", lx.idf("seen"))]


def test_terms_are_ordered_by_idf_then_alphabetically_so_the_choice_is_deterministic():
    lx = lex(aa=10, bb=10, cc=500, dd=1)
    assert [t for t, _ in lx.select_terms(["cc", "bb", "aa", "dd"])] == ["dd", "aa", "bb", "cc"]


def test_duplicates_count_once():
    lx = lex(aa=10)
    assert lx.select_terms(["aa", "aa", "aa"]) == [("aa", lx.idf("aa"))]


def test_only_the_most_informative_terms_are_kept_when_a_paste_has_too_many():
    lx = Lexicon(n_docs=100_000, avg_len=11.0, df={f"t{i:03d}": 10 + i for i in range(100)})
    picked = [t for t, _ in lx.select_terms([f"t{i:03d}" for i in range(100)])]
    assert len(picked) == MAX_TERMS == 24 and picked == [f"t{i:03d}" for i in range(24)]  # the rarest 24
    assert len(lx.select_terms([f"t{i:03d}" for i in range(100)], max_terms=5)) == 5


def test_no_terms_in_no_terms_out():
    assert lex(a=1).select_terms([]) == [] and lex().select_terms(["x"]) == []


def test_min_mass_is_a_share_of_the_total_idf_weight():
    terms = [("a", 4.0), ("b", 6.0)]
    assert min_mass(terms, 0.4) == pytest.approx(4.0) and min_mass(terms, 0.0) == 0.0 and min_mass(terms, 1.0) == pytest.approx(10.0)
    assert min_mass([], 0.4) == 0.0


def test_an_empty_corpus_gives_a_usable_lexicon():
    empty = Lexicon.empty()
    assert empty.n_docs == 0 and empty.avg_len > 0 and empty.df == {} and empty.select_terms(["x"]) == []


def test_the_average_length_is_never_zero_because_sql_divides_by_it():
    assert Lexicon(n_docs=5, avg_len=0.0, df={}).avg_len > 0
