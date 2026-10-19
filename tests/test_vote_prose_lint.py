"""No vote sentence says "absent", "abstained" or "did not vote"."""

from __future__ import annotations

import itertools

import pytest

from answer import facts as F

FORBIDDEN = ["He was absent.", "she abstained", "Abstaining in the lobby", "did not vote", "didn't vote",
             "cast no vote", "did not take part in the vote", "ABSENT"]
ALLOWED = ["is not recorded in the published list", "is listed in division 4 with no vote recorded", "voted Aye",
           "a vote may have gone unrecorded", "listed, no vote recorded"]


@pytest.mark.parametrize("text", FORBIDDEN)
def test_the_words_the_lists_cannot_prove_are_recognised(text):
    assert F.UNPROVABLE_VOTE_WORDS.search(text)


@pytest.mark.parametrize("text", ALLOWED)
def test_what_the_list_shows_is_not_flagged(text):
    assert not F.UNPROVABLE_VOTE_WORDS.search(text)


SHAPES = [
    {"lobby": lobby, "is_teller": teller, "proxy_name": proxy, "ayes": ayes, "ayes_named": named}
    for lobby, teller, proxy, (ayes, named) in itertools.product(
        ("aye", "no", "no_vote", "conflict"), (False, True), (None, "A Whip"), ((300, 300), (300, 298)))
]


@pytest.mark.parametrize("shape", SHAPES)
def test_no_vote_sentence_says_what_the_lists_cannot_prove(shape):
    value = {"division_id": 7, "division_number": 7, "title": "Health Bill: Report Stage", "date": "2025-11-19",
             "house": "Commons", "member": "A Member", "party_at_vote": "Labour", "noes": 200, "noes_named": 200,
             **shape}
    for named in (True, False):
        sentence = F._s_division_vote(value, named=named)
        assert not F.UNPROVABLE_VOTE_WORDS.search(sentence), sentence


@pytest.mark.parametrize("house", ["Commons", "Lords"])
def test_no_vote_table_says_what_the_lists_cannot_prove(house):
    facts = [F.Fact("division_vote", {}, value={"division_id": i, "division_number": i, "house": house,
                                                "member": "A Member", **shape})
             for i, shape in enumerate(SHAPES, start=1)]  # fmt: skip
    meta = {i: (i, f"Division {i}") for i in range(1, len(SHAPES) + 1)}
    table = F.summary_table("division_vote", facts, meta, house=house, date="2025-11-19",
                            unlisted=("A Member is not recorded in the published list for division 99.",))
    assert not F.UNPROVABLE_VOTE_WORDS.search(table), table


def test_a_vote_probe_answer_that_says_so_is_a_harm():
    """The probes read every vote answer on the production path, so the full run is this lint over real answers."""
    from answer.contract import Abstention, Answer, Provenance
    from eval.probes import core
    from eval.probes import divisions as PD

    prov = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t",
                      prompt_version="t", model_id="none")  # fmt: skip
    trial = core.Trial(trial_id="t", cls="ST-division_vote", category="division_vote", question="q?", form="canonical",
                       subject={"division_id": 1, "mnis_id": 2, "role": "absent"})
    verdict = PD.verdict_for([])
    said = Answer(question="q?", provenance=prov, abstention=Abstention.NOT_IN_RECORD,
                  abstention_detail="A Member was absent from division 1.")
    assert verdict(trial, said)[0] == "harm"
