"""What a division answer says, and what it must not claim."""

from __future__ import annotations

from answer import facts as F


def vote(**kw):
    base = {"member": "Alan Mak", "party_at_vote": "Conservative", "lobby": "aye", "is_teller": False,
            "proxy_name": None, "division_number": 12, "title": "Agriculture Bill: Third Reading",
            "house": "Commons", "date": "2020-05-13"}  # fmt: skip
    return base | kw


def test_a_vote_names_the_division_it_was_cast_in():
    sentence = F._s_division_vote(vote())
    assert sentence == "Alan Mak (Conservative) voted Aye in division 12 (Agriculture Bill: Third Reading) on 13 May 2020."


def test_no_vote_recorded_is_not_stated_as_a_choice():
    sentence = F._s_division_vote(vote(lobby="no_vote"))
    assert "cast no vote" not in sentence
    assert "no vote recorded" in sentence and "division 12" in sentence


def test_a_short_list_says_a_vote_may_have_gone_unrecorded():
    v = vote(lobby="no_vote", ayes=215, noes=19, ayes_named=214, noes_named=19)
    sentence = F._s_division_vote(v)
    assert "1 fewer member than the tellers counted" in sentence and "may have gone unrecorded" in sentence
    whole = vote(lobby="no_vote", ayes=215, noes=19, ayes_named=215, noes_named=19)
    assert "unrecorded" not in F._s_division_vote(whole)


def test_a_cast_vote_carries_no_shortness_caveat():
    assert "unrecorded" not in F._s_division_vote(vote(ayes=215, noes=19, ayes_named=214, noes_named=19))


def test_a_result_names_the_division_by_number_and_title():
    v = {"house": "Lords", "division_number": 1, "title": "Post Office (Horizon System) Offences Bill",
         "date": "2024-05-23", "result": "negatived", "ayes": 76, "noes": 111}  # fmt: skip
    sentence = F._s_division_result(v)
    assert "(division 1: Post Office (Horizon System) Offences Bill)" in sentence
    assert sentence.endswith("76 votes to 111.")


def test_a_tally_names_the_division_by_number_and_title():
    v = {"house": "Commons", "division_number": 159, "title": "Finance (No. 2) Bill: Third Reading",
         "date": "2024-05-23", "ayes": 215, "noes": 19}  # fmt: skip
    sentence = F._s_division_tally(v)
    assert sentence.startswith("In the Commons division on 23 May 2024 (division 159: Finance (No. 2) Bill: Third "
                               "Reading), 215 voted Aye and 19 voted No")  # fmt: skip


def test_a_sentence_without_the_division_s_name_still_reads():
    """Values from before the title was carried (a cached answer) render as they did."""
    assert F._s_division_vote({"member": "Alan Mak", "lobby": "no"}) == "Alan Mak voted No."
    assert F._s_division_tally({"ayes": 3, "noes": 4}) == "3 voted Aye and 4 voted No, as counted by the tellers."


def test_several_divisions_sharing_a_result_are_each_named():
    facts = [F.Fact("division_result", {"division_id": i}, value={
        "house": "Lords", "division_number": n, "title": t, "date": "2026-09-09", "result": "agreed",
        "ayes": a, "noes": b}) for i, n, t, a, b in ((1, 1, "FSM Bill [HL]", 248, 163), (2, 2, "FSM Bill [HL]", 188, 125))]
    sentence = F.several_results_sentence(facts)
    assert "division 1, 248 to 163" in sentence and "division 2, 188 to 125" in sentence
    assert sentence.count("FSM Bill [HL]") == 1, "a title every division shares is said once"
