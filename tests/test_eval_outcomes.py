"""A correct decline is scored as one."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_answers_e04", ROOT / "scripts" / "eval_answers.py")
EA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EA)


def item(behaviour, forbidden=()):
    return SimpleNamespace(expected_behaviour=behaviour, forbidden_claims=tuple(forbidden), nuggets=())


def asked(abstention=None, clarify=None, rendered="An answer."):
    from answer import contract as K

    reason = K.Abstention(abstention) if abstention else None
    answer = SimpleNamespace(abstention=reason, facets={"clarify": clarify} if clarify else {}, rendered=rendered,
                             claims=(), structured_facts=(), coverage_note="")
    return SimpleNamespace(answer=answer)


@pytest.mark.parametrize("behaviour", ["abstain_not_in_record", "abstain_out_of_window", "clarify_ambiguous"])
@pytest.mark.parametrize("declined", [dict(abstention="not_in_record"), dict(abstention="outside_window"),
                                      dict(clarify=["which member?"])])
def test_a_decline_where_one_is_expected_is_a_correct_abstention(behaviour, declined):
    outcome, _ = EA.deterministic_outcome(item(behaviour), asked(**declined))
    assert outcome == "correct_abstention"


@pytest.mark.parametrize("behaviour", ["answer", "correct_false_premise", "partial", "neutralise_loaded_framing"])
def test_a_decline_where_an_answer_is_expected_is_an_over_abstention(behaviour):
    """A false premise is CORRECTED, a loaded question answered neutrally: declining either is a loss."""
    outcome, _ = EA.deterministic_outcome(item(behaviour), asked(abstention="not_in_record"))
    assert outcome == "over_abstention"


@pytest.mark.parametrize("behaviour", ["answer", "abstain_not_in_record"])
def test_an_answer_is_left_to_the_judge(behaviour):
    assert EA.deterministic_outcome(item(behaviour), asked())[0] is None


def test_every_behaviour_the_gold_files_use_is_known():
    """The structural guard: a new vocabulary value must be classified, never silently read as 'answer'."""
    seen = set()
    # gold_w5.json is the sealed exam: present in the private checkout, never in a public one.
    names = ["gold_v2.json"] + (["gold_w5.json"] if (ROOT / "tests" / "data" / "gold_w5.json").exists() else [])
    for name in names:
        data = json.loads((ROOT / "tests" / "data" / name).read_text(encoding="utf-8"))
        seen |= {i.get("expected_behaviour") for i in (data["items"] if isinstance(data, dict) else data)}
    assert seen <= EA.EXPECTS_DECLINE | EA.EXPECTS_ANSWER, seen - (EA.EXPECTS_DECLINE | EA.EXPECTS_ANSWER)


def test_an_unknown_behaviour_is_refused_not_guessed():
    with pytest.raises(ValueError, match="unknown expected_behaviour"):
        EA.deterministic_outcome(item("abstain_because_reasons"), asked(abstention="not_in_record"))


# The exit status of a run, and the rows a judge has not read -------------------------------


def run_report(**over):
    base = {"judge": "gemini-deep-research", "stopped_early": None, "ungraded": 0,
            "scored": {"gate": {"passed": True, "reason": ""}}}
    base.update(over)
    return base


def test_only_a_complete_judged_passing_run_exits_zero():
    assert EA.exit_code(run_report()) == 0


@pytest.mark.parametrize("over", [
    {"judge": EA.NO_JUDGE},  # nothing judged: "we could not tell"
    {"scored": {"gate": {"passed": False, "reason": "1 harmful"}}},
    {"stopped_early": "the judge's quota is exhausted"},  # the gate passed on the part that ran
    {"ungraded": 3},  # a judge failure: an unread answer could be harmful
    {"scored": {}},
])
def test_a_partial_unjudged_or_failing_run_never_exits_zero(over):
    assert EA.exit_code(run_report(**over)) == 1


def test_a_row_no_judge_read_is_labelled_for_the_offline_judge():
    by, why = EA.ungraded_label(provider=None, by="deterministic", why=None)
    assert by == EA.AWAITING_JUDGE == "awaiting-judge"
    assert "gemini_judge.py" in why
    assert EA.ungraded_label(provider=object(), by="judge-failed", why="the judge failed (x)") == \
        ("judge-failed", "the judge failed (x)")


# --- F6: a decline the reader sees in the MODEL's words goes to the judge ------------------------------


@pytest.mark.parametrize("behaviour", ["answer", "correct_false_premise", "clarify_ambiguous", "abstain_not_in_record"])
@pytest.mark.parametrize("reason", ["false_premise", "ambiguous"])
def test_a_decline_in_the_models_own_words_is_left_to_the_judge(behaviour, reason):
    """A false premise's correction and an ambiguity's question back reach the reader as the model wrote them, and no
    check reads them. Graded deterministically, a wrong correction could never be harmful and a right one was scored
    an over-abstention (an answer in all but name).
    """
    worded = asked(abstention=reason, rendered="No: that was said in the Lords, by a different member.")
    assert EA.deterministic_outcome(item(behaviour), worded) == (None, "")


def test_a_forbidden_claim_in_a_models_decline_is_still_harmful_without_a_judge():
    bad = item("correct_false_premise", forbidden=["the minister resigned"])
    worded = asked(abstention="false_premise", rendered="No: the minister resigned in 2024.")
    assert EA.deterministic_outcome(bad, worded)[0] == "harmful"


def test_the_systems_own_decline_sentences_stay_deterministic():
    """F5: not-in-record, insufficient and outside-window are the system's own words, so the outcome is decidable.
    """
    for reason in ("not_in_record", "insufficient_evidence", "outside_window"):
        assert EA.deterministic_outcome(item("answer"), asked(abstention=reason))[0] == "over_abstention"


# --- the abstention breakdown: every cowardly decline is in exactly one bucket -----------------------------


def _decline(i, *, evidence=True, related=False, generated=0, verified=0, rank=3):
    return {"id": f"x{i:03d}", "abstention": "not_in_record", "gold_evidence_retrieved": evidence,
            "related_evidence_retrieved": related, "gold_evidence_rank": rank if evidence else None,
            "claims_generated": generated, "claims_verified": verified, "cards": 10}


def test_the_breakdown_puts_every_cowardly_decline_in_one_bucket():
    """Three causes, three opposite fixes: the generator wrote nothing (the prompt), the verifier dropped everything
    (the threshold), or claims verified and the relevance gate declined them as off the question.
    """
    rows = ([_decline(i, generated=0) for i in range(3)]
            + [_decline(3 + i, generated=4, verified=0) for i in range(2)]
            + [_decline(5, generated=5, verified=2)]
            + [_decline(6, evidence=False, related=True), _decline(7, evidence=False)]
            + [{"id": "a1", "abstention": None, "gold_evidence_retrieved": True, "gold_evidence_rank": 1,
                "claims_generated": 3, "claims_verified": 3}])
    got = EA.abstention_breakdown(rows)
    assert got["cowardly_abstentions"] == 6
    assert (got["of_which_generator_refused"], got["of_which_verifier_dropped_all"],
            got["of_which_relevance_gate_declined"]) == (3, 2, 1)
    assert sum(got[k] for k in ("of_which_generator_refused", "of_which_verifier_dropped_all",
                                "of_which_relevance_gate_declined")) == got["cowardly_abstentions"]
    assert got["honest_abstentions_evidence_absent"] == 2
    assert (got["of_those_only_related_material"], got["of_those_nothing_relevant_at_all"]) == (1, 1)
    assert got["evidence_in_deck"] == 7 and got["items_with_gold_evidence"] == 9
    assert got["cowardice_rate"] == round(6 / 7, 4)
