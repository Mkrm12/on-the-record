"""src/eval/perturb.py: one fact changed, always a fact the entry itself states."""

from __future__ import annotations

import re
from datetime import date

import pytest

from eval.perturb import JUDGE_SAFE, KINDS, SEALED, NotPerturbable, perturb, perturbable_kinds

RESULT = ("The Commons divided on 28 January 2026 (division 421) and the motion was agreed to, 294 votes to 108.",
          ["The division was agreed (294 to 108)."], ["That the division was negatived."])
TALLY = ("77 voted Aye and 317 voted No, as counted by the tellers.", ["The Ayes were 77 and the Noes 317."], [])
VOTE = ("Mick Whitley (Labour) voted Aye. Cast by proxy by Grahame Morris.",
        ["Mick Whitley voted Aye.", "The vote was cast by proxy by Grahame Morris."], ["That Mick Whitley voted No."])
STAGE = ("Education (Values of British Citizenship) Bill [HL] reached Order of Commitment discharged in the Lords on "
         "4 November 2024.", ["It was on 4 November 2024."], [])
BIG = ("The corpus holds 1,437,856 chunks.", ["1,437,856 chunks are held."], [])


def _p(case, kind, seed=1):
    answer, refs, must_not = case
    return perturb(answer, refs, kind, seed=seed, must_not_claim=must_not)


@pytest.mark.parametrize("case,kind", [(RESULT, "number"), (TALLY, "number"), (BIG, "number"), (STAGE, "date"),
                                       (RESULT, "result"), (VOTE, "result")])
def test_every_perturbation_changes_one_fact_the_entry_states(case, kind):
    answer, refs, _ = case
    out = _p(case, kind)
    assert out.text != answer and out.kind == kind
    assert out.original in answer and out.replacement in out.text
    assert out.contradicts in refs + list(case[2]), "it must name the reference point the change now contradicts"
    assert all(out.replacement not in r for r in refs), "the new value must appear in no reference point"


def test_a_number_inside_a_date_or_an_id_the_entry_does_not_state_is_never_changed():
    for seed in range(40):
        out = _p(RESULT, "number", seed)
        assert out.original in {"294", "108"}, out.original  # never 28, 2026 (the date) or 421 (the division number)
        assert "28 January 2026" in out.text and "division 421" in out.text


def test_thousands_separators_survive():
    assert re.fullmatch(r"\d{1,3}(,\d{3})+", _p(BIG, "number").replacement)


def test_a_result_flips_to_its_opposite_and_keeps_the_sentence():
    assert _p(RESULT, "result").text.endswith("the motion was negatived, 294 votes to 108.")
    assert _p(VOTE, "result").text.startswith("Mick Whitley (Labour) voted No.")


def test_a_shifted_date_keeps_its_format_and_never_lands_in_the_sealed_window():
    for seed in range(60):
        out = _p(STAGE, "date", seed)
        m = re.search(r"(\d{1,2}) (\w+) (\d{4})\.$", out.text)
        assert m, out.text
    window = (date(2026, 3, 25), "Debate on 28 March 2026.", ["It was on 28 March 2026."])
    for seed in range(60):
        out = perturb(window[1], window[2], "date", seed=seed)
        d = re.search(r"(\d{1,2}) (\w+) (\d{4})", out.replacement)
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                  "October", "November", "December"]
        new = date(int(d.group(3)), months.index(d.group(2)) + 1, int(d.group(1)))
        assert not SEALED[0] <= new <= SEALED[1], out.replacement


def test_the_same_seed_gives_the_same_perturbation_and_seeds_vary():
    assert _p(RESULT, "number", 7) == _p(RESULT, "number", 7)
    assert len({_p(TALLY, "number", s).text for s in range(20)}) > 1


def test_never_a_no_op():
    for case in (RESULT, TALLY, VOTE, STAGE, BIG):
        for kind in perturbable_kinds(case[0], case[1], case[2]):
            for seed in range(25):
                assert _p(case, kind, seed).text != case[0]


@pytest.mark.parametrize("answer,refs,kind", [
    ("Rebecca Paul said it was a holiday tax.", ["Rebecca Paul used the words."], "number"),  # no figure at all
    ("The motion was agreed to, 294 votes to 108.", ["It passed easily."], "number"),  # figures the entry never states
    ("Debate on 4 November 2024.", ["It was in the autumn."], "date"),
    ("(the system declined: no passage in the record answers this question)", ["It was agreed."], "result"),
    ("", ["The Ayes were 77."], "number"),
])
def test_an_answer_without_a_stated_fact_of_that_kind_is_refused(answer, refs, kind):
    with pytest.raises(NotPerturbable):
        perturb(answer, refs, kind, seed=1)


def test_only_figures_dates_and_results_are_judge_safe():
    assert JUDGE_SAFE == {"number", "date", "result"} and set(KINDS) - JUDGE_SAFE == {"negation"}
    assert _p(RESULT, "number").judge_safe and not perturb("She said she did not support it.",
                                                           ["She did not support it."], "negation", seed=1).judge_safe


def test_an_unknown_kind_is_an_error_not_a_silent_skip():
    with pytest.raises(ValueError, match="unknown perturbation kind"):
        perturb("x", ["x"], "speaker", seed=1)
