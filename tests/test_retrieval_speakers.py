"""Resolving MPs and peers named in a question, so retrieval can use who spoke."""

import time

import pytest

from retrieval.speakers import MAX_MEMBERS, Resolution, SpeakerIndex, name_keys, strip_mentions

ROWS = [
    (1482, "Graham Stuart", "Stuart, Graham", "Rt Hon Graham Stuart MP"),
    (4083, "Alison McGovern", "McGovern, Alison", "Alison McGovern MP"),
    (5208, "Perran Moon", "Moon, Perran", "Perran Moon MP"),
    (4460, "Ms Nusrat Ghani", "Ghani, Ms Nusrat", "Ms Nusrat Ghani MP"),
    (100, "Sir Geoffrey Clifton-Brown", "Clifton-Brown, Sir Geoffrey", "Sir Geoffrey Clifton-Brown MP"),
    (200, "Baroness Taylor of Stevenage", "Taylor of Stevenage, B.", "The Rt Hon. the Baroness Taylor of Stevenage"),
    (300, "John Smith", "Smith, John", "John Smith MP"),
    (301, "Jane Smith", "Smith, Jane", "Jane Smith MP"),
    (400, "Ian Green", "Green, Ian", "Ian Green MP"),
    (500, "Keir Starmer", "Starmer, Keir", "Rt Hon Sir Keir Starmer KC MP"),
    (600, "Neil O'Brien", "O'Brien, Neil", "Neil O'Brien MP"),
    (700, "Will Quince", "Quince, Will", "Will Quince MP"),
    (800, "Theresa May", "May, Theresa", "Rt Hon Theresa May MP"),
    (960, "Sir Jim Green-Smith", "Green-Smith, Sir Jim", "Sir Jim Green-Smith MP"),  # one common word, one distinctive
]
COMMON = {"green"}  # what the searcher's corpus statistics would call a common word
INDEX = SpeakerIndex.from_rows(ROWS, is_common=lambda word: word in COMMON)


def ids(query):
    return set(INDEX.resolve(query).member_ids)


# --- the keys a member is known by ------------------------------------------------------------------------------


def test_keys_come_from_the_name_the_full_title_and_the_list_form():
    full, surname = name_keys("Ms Nusrat Ghani", "Ghani, Ms Nusrat", "Ms Nusrat Ghani MP")
    assert full == {("ms", "nusrat", "ghani"), ("nusrat", "ghani")} and surname == ("ghani",)


def test_titles_and_post_nominals_are_stripped_from_the_full_title_form():
    full, _ = name_keys("Keir Starmer", "Starmer, Keir", "Rt Hon Sir Keir Starmer KC MP")
    assert ("keir", "starmer") in full


def test_a_peers_territorial_title_is_one_name():
    full, surname = name_keys("Baroness Taylor of Stevenage", "Taylor of Stevenage, B.", "The Rt Hon. the Baroness Taylor of Stevenage")
    assert ("baroness", "taylor", "of", "stevenage") in full and ("taylor", "of", "stevenage") in full
    assert surname == ("taylor", "of", "stevenage")


def test_hyphens_split_and_apostrophes_vanish_so_both_spellings_meet():
    full, surname = name_keys("Sir Geoffrey Clifton-Brown", "Clifton-Brown, Sir Geoffrey", None)
    assert ("geoffrey", "clifton", "brown") in full and surname == ("clifton", "brown")
    assert name_keys("Neil O'Brien", "O'Brien, Neil", None)[1] == ("obrien",)


def test_a_member_with_no_usable_name_has_no_keys():
    assert name_keys(None, None, None) == (set(), None) and name_keys("", "", "") == (set(), None)


# --- resolving full names ----------------------------------------------------------------------------------------


def test_a_full_name_resolves_and_reports_what_matched():
    r = INDEX.resolve("What did Perran Moon raise at Business questions?")
    assert r.member_ids == (5208,) and r.mentions == ("Perran Moon",) and not r.ambiguous and bool(r)


@pytest.mark.parametrize("query", ["what did perran moon raise", "WHAT DID PERRAN MOON RAISE", "Perran   Moon?", "perran-moon"])
def test_a_full_name_needs_no_particular_case_or_spacing(query):
    assert ids(query) == {5208}


@pytest.mark.parametrize("query", ["Rt Hon Graham Stuart MP said", "Graham Stuart's speech", "Graham Stuart’s speech", "Did Graham Stuart agree?"])
def test_titles_and_possessives_around_a_name_do_not_matter(query):
    assert ids(query) == {1482}


def test_two_people_in_one_question_are_both_found_in_order_of_mention():
    r = INDEX.resolve("Did Graham Stuart agree with Alison McGovern?")
    assert r.member_ids == (1482, 4083) and r.mentions == ("Graham Stuart", "Alison McGovern")


@pytest.mark.parametrize("query", ["Ms Nusrat Ghani", "Nusrat Ghani", "ms ghani", "Ms Ghani", "so Ghani said that", "what did Ghani say"])
def test_a_name_with_a_title_inside_it_resolves_however_it_is_written(query):
    assert ids(query) == {4460}


@pytest.mark.parametrize("query", ["Neil O'Brien", "Neil O’Brien", "NEIL OBRIEN", "what did O'Brien say"])
def test_apostrophes_are_ignored(query):
    assert ids(query) == {600}


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Sir Geoffrey Clifton-Brown", {100}),
        ("Geoffrey Clifton Brown", {100}),
        ("what did Clifton-Brown say", {100}),
        ("Sir Geoffrey", set()),  # a given name alone is never enough
    ],
)
def test_hyphenated_surnames_resolve_as_a_unit(query, expected):
    assert ids(query) == expected


@pytest.mark.parametrize("query", ["Baroness Taylor of Stevenage", "Taylor of Stevenage", "what did the Baroness Taylor of Stevenage say"])
def test_a_peer_resolves_by_the_full_territorial_title(query):
    assert ids(query) == {200}


def test_half_a_peers_title_is_not_a_match():
    assert ids("Baroness Taylor said") == set()


# --- bare surnames: the risky case ------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["What did Starmer say about housing?", "Did the Minister agree with Starmer?", "Mr Starmer said", "Sir Starmer"])
def test_a_capitalised_surname_in_the_middle_of_a_question_resolves(query):
    assert ids(query) == {500}


@pytest.mark.parametrize("query", ["what did starmer say", "Starmer said what?", "starmer", "STARMER"])
def test_a_lowercase_or_sentence_initial_bare_surname_does_not(query):
    assert ids(query) == set()


def test_an_all_caps_surname_after_the_first_word_counts_as_capitalised():
    assert ids("what did STARMER say") == {500}


def test_a_title_makes_a_lowercase_or_initial_surname_safe():
    assert ids("mr starmer said") == {500} and ids("Dr Starmer said") == {500}


def test_a_surname_that_is_a_common_word_needs_a_title():
    assert ids("What did Green say about energy?") == set()  # could be the colour, the party or the MP
    assert ids("What did Mr Green say?") == {400} and ids("What did Ian Green say?") == {400}


def test_the_word_the_is_not_a_title_so_the_green_party_is_not_mr_green():
    assert ids("What did the Green party say?") == set() and ids("Did the Starmer government agree?") == {500}  # "the" adds nothing either way


def test_the_same_person_named_twice_is_listed_once():
    r = INDEX.resolve("Did Graham Stuart agree with Mr Stuart?")
    assert r.member_ids == (1482,) and len(r.mentions) == 2


def test_accents_and_case_are_ignored_in_both_the_names_and_the_query():
    idx = SpeakerIndex.from_rows([(950, "José Müller", "Müller, José", None)])
    assert idx.resolve("What did Jose Muller say?").member_ids == (950,) and idx.resolve("what did JOSÉ MÜLLER say").member_ids == (950,)


def test_a_surname_is_common_only_when_every_part_of_it_is():
    assert ids("What did Green-Smith say?") == {960}  # "green" is common but "smith" is not, so the pair identifies a person
    assert ids("What did Green say?") == set()


def test_a_surname_that_is_an_auxiliary_verb_is_never_accepted_bare():
    assert ids("What did May say about it?") == set()  # "may" is a stopword, so corpus statistics cannot flag it
    assert ids("What did Theresa May say?") == {800} and ids("Mrs May said") == {800}


def test_a_shared_surname_returns_every_holder_and_says_it_is_ambiguous():
    r = INDEX.resolve("What did Smith say about tax?")
    assert set(r.member_ids) == {300, 301} and r.ambiguous


def test_a_full_name_settles_the_ambiguity():
    r = INDEX.resolve("What did Jane Smith say about tax?")
    assert r.member_ids == (301,) and not r.ambiguous


def test_a_surname_shared_by_too_many_people_is_not_resolved():
    many = [(1000 + i, f"Person{i} Jones", f"Jones, Person{i}", None) for i in range(6)]
    assert SpeakerIndex.from_rows(many).resolve("What did Jones say?").member_ids == ()


def test_a_first_name_alone_is_never_a_match():
    assert ids("Will the Minister confirm?") == set() and ids("Keir said it") == set() and ids("Perran did") == set()


def test_ordinary_words_that_appear_in_names_do_not_resolve():
    assert ids("What is the Minister going to do about housing?") == set()
    assert ids("Who is the Prime Minister?") == set()


# --- the result and its use ------------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", None, "no names here at all", "....", "😀"])
def test_no_names_means_an_empty_resolution(query):
    r = INDEX.resolve(query)
    assert r == Resolution() and not r and r.member_ids == ()


def test_at_most_max_members_are_returned_in_order_of_mention():
    rows = [(i, f"First{i} Surname{i}", f"Surname{i}, First{i}", None) for i in range(20)]
    query = " and ".join(f"First{i} Surname{i}" for i in range(20))
    got = SpeakerIndex.from_rows(rows).resolve(query).member_ids
    assert len(got) == MAX_MEMBERS == 8 and got == tuple(range(8))


def test_stripping_removes_exactly_the_matched_words_with_their_title_and_post_nominal():
    q = "What did Rt Hon Graham Stuart MP criticise about the Act?"
    r = INDEX.resolve(q)
    assert strip_mentions(q, r) == "What did criticise about the Act?"
    assert r.mentions == ("Rt Hon Graham Stuart MP",)
    q2 = "What did Perran Moon raise at Business questions?"
    assert strip_mentions(q2, INDEX.resolve(q2)) == "What did raise at Business questions?"
    assert strip_mentions("no names", Resolution()) == "no names"


def test_stripping_handles_two_mentions_and_keeps_the_rest_intact():
    q = "Did Graham Stuart agree with Alison McGovern about the Bill?"
    assert strip_mentions(q, INDEX.resolve(q)) == "Did agree with about the Bill?"


def test_a_long_query_is_resolved_quickly():
    query = "housing planning tax " * 300 + "What did Graham Stuart say"
    started = time.perf_counter()
    assert ids(query) == {1482}
    assert time.perf_counter() - started < 0.5


def test_rows_without_names_or_with_repeats_are_tolerated():
    idx = SpeakerIndex.from_rows([(1, None, None, None), (2, "Ann Lee", "Lee, Ann", None), (2, "Ann Lee", "Lee, Ann", None)])
    assert idx.resolve("Ann Lee").member_ids == (2,) and len(idx) == 1


def test_an_empty_index_resolves_nothing_and_never_fails():
    assert SpeakerIndex.from_rows([]).resolve("Graham Stuart") == Resolution()
