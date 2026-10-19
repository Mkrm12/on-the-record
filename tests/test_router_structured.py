"""The router decides a fact question from its TEXT — production has no gold category."""

from __future__ import annotations

import pytest

from answer import router as R


@pytest.mark.parametrize("question, category", [
    # division_vote: how a named member voted in one division
    ("How did Diane Abbott vote on the Rwanda Bill on 12 December 2023?", "division_vote"),
    ("Did Lord Frost vote for the amendment on 3 March 2025?", "division_vote"),
    ("Was Baroness Hoey a Content or a Not Content on the motion on 4 June 2025?", "division_vote"),
    ("Which lobby did Mel Stride go through on the Finance Bill third reading on 23 May 2024?", "division_vote"),
    ("Did Jeremy Hunt back the new clause when the Commons divided on 7 September 2026?", "division_vote"),
    ("How did Sir Iain Duncan Smith vote in the division on 1 May 2024?", "division_vote"),
    # division_tally: the numbers
    ("What was the vote count on the Assisted Dying Bill second reading on 29 November 2024?", "division_tally"),
    ("How many MPs voted against the Finance Bill on 23 May 2024?", "division_tally"),
    ("By how many votes was the Sentencing Bill amendment defeated on 20 January 2026?", "division_tally"),
    ("What were the numbers on each side when the Lords divided on 16 March 2026?", "division_tally"),
    ("Give me the Ayes and the Noes for the division on 9 December 2025.", "division_tally"),
    # division_result: carried or not
    ("Was the Planning Bill amendment carried in the Lords on 9 September 2026?", "division_result"),
    ("Did the Commons approve the draft regulations on 28 January 2026?", "division_result"),
    ("Did the House of Lords reject the motion on 21 May 2024?", "division_result"),
    ("What was the result of the vote on the Pension Schemes Bill on 16 March 2026?", "division_result"),
    ("Did the Government win the division on the Employment Rights Bill on 17 November 2025?", "division_result"),
    # party_at_date
    ("Which party did Chuka Umunna belong to on 1 March 2019?", "party_at_date"),
    ("What party was Lord Timpson in on 2 March 2026?", "party_at_date"),
    ("What was Scott Arthur's party affiliation in February 2026?", "party_at_date"),
    # office_held_at_date
    ("Who was the Home Secretary on 1 September 2026?", "office_held_at_date"),
    ("Who held the office of Chief Secretary to the Treasury on 1 October 2025?", "office_held_at_date"),
    ("Who was serving as Leader of the House of Lords on 1 May 2024?", "office_held_at_date"),
    # bill_stage_date
    ("When did the Renters' Rights Bill get its second reading in the Commons?", "bill_stage_date"),
    ("On what date did the Employment Rights Act 2025 receive Royal Assent?", "bill_stage_date"),
    ("When was the report stage of the Crime and Policing Bill in the Lords?", "bill_stage_date"),
    ("What date was the third reading of the Planning and Infrastructure Bill?", "bill_stage_date"),
    # count
    ("How many divisions were held in the Commons in March 2026?", "count"),
    ("How many debates took place in Westminster Hall on 8 September 2026?", "count"),
    ("How many times did Lord Meston speak in June 2024?", "count"),
])  # fmt: skip
def test_a_fact_question_is_recognised_from_its_words(question, category):
    decision = R.route(question)
    assert (decision.route, decision.category) == ("structured", category), decision.why


@pytest.mark.parametrize("question", [
    "Why did Diane Abbott vote against the Rwanda Bill?",  # asks for reasons: text
    "What did Lord Frost say before the vote on the amendment?",  # asks what was said
    "What arguments were made against the Assisted Dying Bill?",  # the shape of a debate
    "How did the debate on the Planning Bill go in the Lords?",  # not a vote
    "Who spoke in the debate on the Pension Schemes Bill on 16 March 2026?",  # the set route
    "What did MPs say about the result of the Rwanda vote?",  # talk about a result, not the result
    "Explain what the Employment Rights Bill does.",
    "What did the Home Secretary say about small boats on 1 September 2026?",  # an office holder's words, not who held it
])  # fmt: skip
def test_a_question_about_what_was_said_is_never_a_fact_lookup(question):
    assert R.route(question).route != "structured"


# A day's divisions asked about together (QA pass 6, S05): the division noun in the plural is the same question, and was
# routed to text search ("What was the result of the divisions on 9 March 2026?").
@pytest.mark.parametrize(("question", "category"), [
    ("What was the result of the divisions on 9 March 2026?", "division_result"),
    ("What were the results of the divisions on 9 March 2026?", "division_result"),
    ("What was the outcome of the Lords divisions on 9 March 2026?", "division_result"),
    ("What were the outcomes of the votes in the Commons on 9 March 2026?", "division_result"),
    ("What were the results of the Commons votes on 9 March 2026?", "division_result"),
    ("What was the outcome of the Lords\u2019 votes on 9 March 2026?", "division_result"),  # a House owns its votes
    ("What was the outcome of the House's votes on 9 March 2026?", "division_result"),
    ("How many divisions were there on 9 March 2026?", "count"),
])  # fmt: skip
def test_the_division_noun_in_the_plural_is_the_same_question(question, category):
    from answer import resolve as RS

    decision = R.route(question)
    assert (decision.route, decision.category) == ("structured", category), decision.why
    if category == "division_result":
        assert not RS.names_business(question), "'divisions' and 'outcomes' ask about divisions; they name no business"


@pytest.mark.parametrize("question", [
    # a member's votes are theirs, not the House's: never answered with the House's result (QA pass 7, Med-5)
    "What was the outcome of Diane Abbott's votes on the Welfare Reform Bill on 12 March 2025?",
    "What was the outcome of Diane Abbott\u2019s vote on the Welfare Reform Bill on 12 March 2025?",
    "What was the result of Lord Young's votes on the Planning Bill on 9 September 2026?",
    "What was the outcome of the Conservatives' votes on the Planning Bill on 9 September 2026?",
    # "divisions" in a party's sense is not a vote (QA pass 7, L9): the plural reads only after a result or outcome
    "Which debate exposed divisions within the Labour Party on welfare on 1 July 2025?",
    "Did the rebels support the government in the divisions on welfare on 1 July 2025?",
])  # fmt: skip
def test_a_members_votes_or_a_partys_divisions_are_never_the_houses_result(question):
    decision = R.route(question)
    assert decision.category not in ("division_result", "division_debate", "division_vote"), (question, decision.why)


def test_a_debate_behind_a_division_keeps_its_own_category():
    decision = R.route("Which debate was division 149 in the Commons on 8 May 2024 taken during?")
    assert (decision.route, decision.category) == ("structured", "division_debate")


@pytest.mark.parametrize("question", [
    "Which debate was division 149 in the Commons on 8 May 2024 taken during?",
    "What was being debated when the Lords divided on the Border Security Bill on 5 November 2025?",
    "Find the debate that led to the Commons division on the Planning Bill on 11 September 2026.",
    "In which debate did the Commons vote on new clause 7 on 13 May 2020?",
    "What debate preceded the Lords division on amendment 4 on 9 September 2026?",
    "During which debate did the Commons divide on the Railways Bill?",
    "What was under discussion when the House voted on the Finance Bill on 8 May 2024?",
])  # fmt: skip
def test_every_way_of_asking_for_the_debate_behind_a_vote_reaches_it(question):
    assert R.route(question).category == "division_debate"


def test_a_lords_reason_named_in_a_title_does_not_make_a_fact_question_a_why_question():
    """Found by the ST probes: "Lords Reason 120B" is a message between the Houses, named in the division's title. The
    guard against "why" questions read its "Reason" and sent the result, the tally and every member's vote to text
    search.
    """
    q = ("What was the result of the Commons division on Employment Rights Bill: motion relating to Lords Reason 120B "
         "on 5 November 2025?")
    decision = R.route(q)
    assert (decision.route, decision.category, decision.filters.house) == ("structured", "division_result", "Commons")
    assert R.route("Why did the Commons reject the Lords Reasons on the Employment Rights Bill?").route != "structured"
    assert R.route("What were the reasons given for the vote on the Planning Bill?").route != "structured"


@pytest.mark.parametrize(("question", "house"), [
    ("How did Edward Leigh vote in the division on Social Housing Bill [Lords]: Reasoned Amendment to Second Reading "
     "on 10 September 2026?", None),  # the Bill began in the Lords; the vote is not said to be there
    ("How did Lord Ahmad vote on the Financial Services and Markets Bill [HL] on 9 September 2026?", None),
    # (a bare "Lord" never names the House: Lord Chancellor and Lord President can be MPs; the resolver tries both)
    ("Did the Lords pass the Financial Services and Markets Bill [HL] on 9 September 2026?", "Lords"),
    ("Did the Commons agree to the Social Housing Bill [Lords] on 10 September 2026?", "Commons"),
])  # fmt: skip
def test_a_bill_s_house_of_origin_marker_is_part_of_its_name_not_the_house_voting(question, house):
    """Found by the ST probes: "[Lords]" and "[HL]" mark a Bill that began in the Lords, whichever House is now voting
    on it, so they are part of the Bill's name. A Commons division on one was looked for in the Lords.
    """
    assert R.route(question).filters.house == house


@pytest.mark.parametrize(("question", "category"), [
    ("What was the result of the Commons division on Opposition day debate on flooding: Mr Corbyn's motion on "
     "4 March 2020?", "division_result"),
    ("What were the Ayes and Noes in the Commons division on Opposition day debate on flooding: Mr Corbyn's motion "
     "on 4 March 2020?", "division_tally"),
    ("How did Angela Eagle vote in the division on Opposition day debate on flooding: Mr Corbyn's motion on "
     "4 March 2020?", "division_vote"),
    ("Was the Government amendment to the Opposition day debate on seasonal work agreed to in the division?",
     "division_result"),
])  # fmt: skip
def test_a_debate_named_inside_a_division_s_title_does_not_make_it_a_question_about_the_debate(question, category):
    """Found by the first ST probes: any question mentioning a vote and the word "debate" was sent to
    `division_debate`, so a division TITLED "Opposition day debate on..." could not be asked about at all.
    """
    assert R.route(question).category == category


def test_a_speech_and_a_vote_together_is_multi_hop_not_a_fact():
    assert R.route("What did Stuart Andrew say about new clause 65, and how did he vote?").route == "multi_hop"


@pytest.mark.parametrize("question", ["How did my MP vote on the assisted dying bill?",
                                      "Did our local MP support the Rwanda Bill?",
                                      "How did my representative vote on the Finance Bill?"])
def test_a_member_the_question_does_not_name_is_asked_about_not_guessed(question):
    decision = R.route(question)
    assert decision.route == "clarify" and decision.clarify, "the record cannot know who 'my MP' is"


# --- the gold category decides nothing ------------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["division_tally", "division_vote", "party_at_date", "count"])
def test_a_gold_category_never_decides_the_route(category):
    """An evaluation that routes by gold category measures the router against its own answer key."""
    decision = R.route("Tell me about the vote.", category)
    assert decision.route != "structured" or decision.category is None


def test_the_evaluation_reports_router_accuracy_from_the_words():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "eval_answers_e03", Path(__file__).resolve().parents[1] / "scripts" / "eval_answers.py")
    EA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EA)
    rows = [{"id": "a", "category": "division_vote", "route": "structured", "fact": "division_vote"},
            {"id": "b", "category": "division_tally", "route": "structured", "fact": "division_result"},
            {"id": "c", "category": "party_at_date", "route": "retrieval", "fact": None},
            {"id": "d", "category": "multi_speech", "route": "structured", "fact": "division_debate"},
            {"id": "e", "category": "paraphrase", "route": "structured", "fact": "count"},
            {"id": "f", "category": "paraphrase", "route": "retrieval", "fact": None}]
    got = EA.router_accuracy(rows)
    assert (got["fact_questions"], got["right_fact"], got["accuracy"]) == (3, 1, 0.3333)
    assert got["misrouted"] == ["b", "c"] and got["sent_to_a_fact_ids"] == ["e"]


def test_no_evaluation_script_hands_the_router_a_gold_category():
    """The leak, closed at the callers too: a gold category passed to `ask` or a route call is ignored today, and must
    not be relied on tomorrow.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    offenders = []
    for path in root.glob("*.py"):
        for args in re.findall(r"(?:\.ask|route_with_model|\.route)\(([^)]*)\)", path.read_text(encoding="utf-8")):
            # `gold_category=item.category` is the legitimate path: the trace and the router's accuracy.
            if re.search(r"\bitem\.category\b", re.sub(r"\bgold_category\s*=\s*item\.category\b", "", args)):
                offenders.append(path.name)
    assert offenders == []


def test_the_route_is_the_same_with_or_without_a_category():
    q = "Was the Planning Bill amendment carried in the Lords on 9 September 2026?"
    with_gold, without = R.route(q, "division_result"), R.route(q)
    assert (with_gold.route, with_gold.category) == (without.route, without.category)
