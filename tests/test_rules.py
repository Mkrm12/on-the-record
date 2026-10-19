"""The procedure rules engine."""

import pytest

from ingest.rules import (
    boilerplate_reason,
    classify_debate,
    classify_motion,
    find_events,
    parse_speaker,
    sentence_boilerplate_reason,
)


def types(text, speaker=None):
    return [e.event_type for e in find_events(text, speaker)]


def event(text, event_type, speaker=None):
    matches = [e for e in find_events(text, speaker) if e.event_type == event_type]
    assert matches, f"no {event_type!r} in {[e.event_type for e in find_events(text, speaker)]}"
    return matches[0]


CHAIR = "Mr Speaker"
LORDS_CHAIR = "The Deputy Chairman of Committees (Lord Faulkner of Worcester) (Lab)"
MEMBER = "Jim Shannon (Strangford) (DUP)"


# =========================================================================== 1. parse_speaker
# ===========================================================================


def test_speaker_commons_member_with_seat_and_party():
    s = parse_speaker("Jim Shannon (Strangford) (DUP)")
    assert (s.person, s.constituency, s.party, s.role) == ("Jim Shannon", "Strangford", "DUP", "member")
    assert s.office is None and not s.is_chair and not s.remote


def test_speaker_lord_with_party_only():
    s = parse_speaker("Baroness Twycross (Lab)")
    assert (s.person, s.party, s.constituency, s.role) == ("Baroness Twycross", "Lab", None, "member")


def test_speaker_plain_name():
    s = parse_speaker("Angela Rayner")
    assert (s.person, s.party, s.constituency, s.office, s.role) == ("Angela Rayner", None, None, None, "member")


@pytest.mark.parametrize("party", ["Lab/Co-op", "Lab Co-op", "Non-Afl", "Your Party", "Reform", "Green", "SNP", "PC", "CB", "Ind", "UUP"])
def test_speaker_party_tokens_seen_in_the_corpus(party):
    assert parse_speaker(f"Some Member (Some Seat) ({party})").party == party


@pytest.mark.parametrize("party", ["Lab", "Con", "LD", "CB", "Non-Afl", "Ind", "GP", "Green", "Reform", "DUP", "UUP", "Bp", "Your Party"])
def test_speaker_single_bracket_party_is_recognised_not_mistaken_for_a_seat(party):
    # With only one bracket there is no second group to disambiguate: the party list must decide.
    s = parse_speaker(f"Lord Someone ({party})")
    assert (s.party, s.constituency) == (party, None)


def test_speaker_prime_minister_is_an_office_without_a_person():
    s = parse_speaker("The Prime Minister")
    assert (s.office, s.person, s.role) == ("The Prime Minister", None, "minister")


def test_speaker_minister_office_with_person():
    s = parse_speaker("The Minister for Housing and Planning (Matthew Pennycook)")
    assert (s.office, s.person, s.role) == ("The Minister for Housing and Planning", "Matthew Pennycook", "minister")
    assert s.constituency is None


def test_speaker_office_containing_commas_with_lord_and_party():
    s = parse_speaker(
        "The Parliamentary Under-Secretary of State, Foreign, Commonwealth and Development Office (Lord Wood of Anfield) (Lab)"
    )
    assert s.office == "The Parliamentary Under-Secretary of State, Foreign, Commonwealth and Development Office"
    assert (s.person, s.party, s.role) == ("Lord Wood of Anfield", "Lab", "minister")


def test_speaker_office_not_starting_with_the_is_found_via_the_peer_in_brackets():
    s = parse_speaker(
        "Captain of the King’s Bodyguard of the Yeomen of the Guard and Deputy Chief Whip (Baroness Anderson of Stoke-on-Trent) (Lab)"
    )
    assert s.office.startswith("Captain of the King")
    assert (s.person, s.party, s.role) == ("Baroness Anderson of Stoke-on-Trent", "Lab", "minister")


def test_speaker_lord_in_waiting_whip():
    s = parse_speaker("Lord in Waiting/Government Whip (Lord Katz) (Lab)")
    assert (s.office, s.person, s.party) == ("Lord in Waiting/Government Whip", "Lord Katz", "Lab")


def test_speaker_mr_speaker():
    s = parse_speaker("Mr Speaker")
    assert (s.office, s.role, s.is_chair) == ("Mr Speaker", "speaker", True)


def test_speaker_deputy_speaker_with_and_without_person():
    s = parse_speaker("Madam Deputy Speaker (Ms Nusrat Ghani)")
    assert (s.office, s.person, s.role, s.is_chair) == ("Madam Deputy Speaker", "Ms Nusrat Ghani", "deputy_speaker", True)
    assert parse_speaker("Madam Deputy Speaker").role == "deputy_speaker"


def test_speaker_in_the_chair_is_a_role_not_a_seat():
    s = parse_speaker("Dame Siobhain McDonagh (in the Chair)")
    assert (s.person, s.role, s.is_chair, s.constituency) == ("Dame Siobhain McDonagh", "chair", True, None)


def test_speaker_lord_speaker():
    s = parse_speaker("The Lord Speaker (Lord Forsyth of Drumlean)")
    assert (s.office, s.person, s.role, s.is_chair) == ("The Lord Speaker", "Lord Forsyth of Drumlean", "lord_speaker", True)


@pytest.mark.parametrize(
    "name",
    [
        "The Deputy Chairman of Committees (Lord Faulkner of Worcester) (Lab)",
        "The Deputy Speaker (Baroness McIntosh of Hudnall) (Lab)",
        "The Senior Deputy Speaker",
        # the pre-2016 title, inside the ten-year span
        "The Principal Deputy Chairman of Committees (Lord Brabazon of Tara) (Con)",
    ],
)
def test_speaker_lords_deputies_are_chairs(name):
    s = parse_speaker(name)
    assert s.role == "lords_deputy" and s.is_chair


def test_the_temporary_chair_of_ways_and_means_presides():
    assert parse_speaker("The Temporary Chair of Ways and Means").is_chair


def test_speaker_peer_title_starting_with_the_is_a_person_not_an_office():
    for name in ("The Earl of Courtown (Con)", "The Lord Bishop of Norwich", "The Viscount Younger of Leckie (Con)"):
        s = parse_speaker(name)
        assert s.office is None and s.person.startswith("The ") and s.role == "member", name
    assert parse_speaker("The Earl of Courtown (Con)").party == "Con"


def test_speaker_remote_flag_is_captured_and_stripped():
    s = parse_speaker("Helen Morgan (North Shropshire) (LD) [R]")
    assert (s.person, s.constituency, s.party, s.remote) == ("Helen Morgan", "North Shropshire", "LD", True)


def test_speaker_unicode_spaces_are_normalised():
    # Real attribution: en-spaces (U+2002) between the tokens and a trailing space.
    s = parse_speaker("Ben Obese-Jecty (Huntingdon) (Con) ")
    assert (s.person, s.constituency, s.party) == ("Ben Obese-Jecty", "Huntingdon", "Con")


def test_speaker_maiden_speech_flag_is_not_a_party():
    s = parse_speaker("The Minister of State, Department for Business, Innovation, Science and Trade (Lord Sarwar) (Lab) (Maiden Speech)")
    assert (s.person, s.party, s.role) == ("Lord Sarwar", "Lab", "minister")
    assert "maiden_speech" in s.flags


def test_speaker_valedictory_speech_flag():
    assert "valedictory_speech" in parse_speaker("Lord Someone (Con) (Valedictory Speech)").flags


def test_speaker_remote_and_virtual_flags():
    assert parse_speaker("Helen Morgan (North Shropshire) (LD) [R]").flags == ("remote",)
    s = parse_speaker("Lord Someone (Lab) [V]")
    assert s.remote and s.flags == ("virtual",) and s.party == "Lab"
    assert parse_speaker("Angela Rayner").flags == ()


@pytest.mark.parametrize(
    "name,role",
    [
        ("The Second Deputy Chairman of Ways and Means (Caroline Nokes)", "deputy_speaker"),
        ("The Second Deputy Chairman", "deputy_speaker"),
        ("The Economic Secretary to the Treasury (Lucy Rigby)", "minister"),
        ("The Exchequer Secretary to the Treasury (Dan Tomlinson)", "minister"),
        ("The Chief Secretary to the Treasury (Emma Reynolds)", "minister"),
        ("The Second Church Estates Commissioner (Marsha De Cordova)", "office_holder"),
        ("The Shadow Secretary of State for Health (Someone)", "shadow_minister"),
    ],
)
def test_speaker_office_roles_seen_in_the_corpus(name, role):
    s = parse_speaker(name)
    assert s.role == role
    assert s.is_chair == (role == "deputy_speaker")


def test_speaker_commons_member_with_seat_only():
    s = parse_speaker("Alex Sobel (Leeds Central and Headingley)")
    assert (s.person, s.constituency, s.party) == ("Alex Sobel", "Leeds Central and Headingley", None)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_speaker_missing_is_unknown(value):
    s = parse_speaker(value)
    assert s.role == "unknown" and s.person is None and not s.is_chair


# =========================================================================== 2. find_events
# ===========================================================================

# ---- motions --------------------------------------------------------------


def test_beg_to_move_with_that_captures_the_motion_text():
    e = event("I beg to move, That the Bill be now read a Second time.\n\nGood housing is the foundation.", "motion_moved")
    assert e.description == "That the Bill be now read a Second time."
    assert e.motion_type == "substantive"
    assert e.raw_phrase.startswith("I beg to move")


def test_beg_to_move_motion_on_the_next_paragraph_is_still_captured():
    # Real shape: "I beg to move,\r\n\r\nThat this House has considered...
    e = event("I beg to move,\r\n\r\nThat this House has considered Israel and Palestine.\r\n\r\nAs this is my first debate...", "motion_moved")
    assert e.description == "That this House has considered Israel and Palestine."
    assert e.motion_type == "neutral"


def test_motion_moved_fires_once_when_motion_is_on_the_next_paragraph():
    assert types("I beg to move,\n\nThat this House has considered X.\n\nText.").count("motion_moved") == 1
    assert "motion_stated" not in types("I beg to move,\n\nThat this House has considered X.\n\nText.")


def test_lords_beg_to_move_without_comma_and_lowercase_that():
    e = event("My Lords, I beg to move that this Bill be now read a second time.", "motion_moved")
    assert e.description == "that this Bill be now read a second time."


def test_bare_beg_to_move_has_no_description_and_unknown_type():
    e = event("I beg to move.\r\n\r\nAmendment to the Motion", "motion_moved")
    assert e.description is None and e.motion_type is None


def test_beg_to_move_mid_speech_is_not_a_motion():
    assert "motion_moved" not in types("The Minister said he would beg to move nothing. I beg to move on to housing.")
    assert "motion_moved" not in types("Earlier I heard, I beg to move, That the Bill be read, and I disagreed.")


def test_leave_to_bring_in_a_bill_is_a_substantive_motion_and_ten_minute_rule_event():
    text = "I beg to move, That leave be given to bring in a Bill to make provision about algorithms.\n\nThank you."
    assert event(text, "motion_moved").motion_type == "substantive"
    assert event(text, "leave_to_bring_in_bill").description.startswith("That leave be given to bring in a Bill")


def test_that_this_house_has_considered_alone_is_a_neutral_stated_motion():
    e = event("That this House has considered the future of rail.", "motion_stated")
    assert e.motion_type == "neutral"


def test_substantive_stated_motion():
    e = event("That this House doth agree with the Lords in their Amendment 1.", "motion_stated")
    assert e.motion_type == "substantive"


def test_take_note_motion_is_neutral():
    assert event("My Lords, I beg to move that this House takes note of the Report.", "motion_moved").motion_type == "neutral"


def test_amendment_moved_commons_style():
    e = event("I beg to move amendment 5, page 3, line 4, leave out ‘x’ and insert ‘y’.", "amendment_proposed")
    assert e.description.startswith("amendment 5, page 3, line 4")


def test_new_clause_moved():
    assert "amendment_proposed" in types("I beg to move new clause 2—Duty to report.")


def test_amendment_moved_lords_style():
    e = event("My Lords, in moving Amendment 59, I will also speak to Amendment 60 in my name.", "amendment_proposed")
    assert "59" in e.description


def test_amendment_stage_direction():
    assert "amendment_proposed" in types("Amendment proposed: 1, page 1, line 3, leave out ‘x’.")


# ---- question stages ------------------------------------------------------


def test_question_proposed_variants():
    assert "question_proposed" in types("Motion made, and Question proposed, That the Bill be now read a Second time.")
    assert "question_proposed" in types("Question proposed.")
    assert "question_proposed" in types("The Question is, That the amendment be made.", CHAIR)


def test_the_question_is_as_on_the_order_paper_is_not_an_event():
    assert types("The Question is as on the Order Paper.", CHAIR) == []


def test_question_put_is_distinct_from_put_and_agreed():
    assert types("Question put.") == ["question_put"]
    assert types("Question put, That the amendment be made.") == ["question_put"]
    assert types("Question put and agreed to.") == ["agreed_without_division"]
    assert types("Question put and negatived.") == ["negatived_without_division"]


def test_question_accordingly_agreed_or_negatived():
    assert types("Question accordingly agreed to.") == ["question_agreed"]
    assert types("Question accordingly negatived.") == ["question_negatived"]


def test_lords_motion_agreed():
    assert "question_agreed" in types("On Question, Motion agreed.")


def test_question_put_inside_a_ministers_speech_is_not_an_event():
    # 72 real speeches contain "Question put"; they are prose, e.g.
    assert types("I would like to answer the Question put to me earlier by my hon. Friend.", "The Minister for X (Y Z)") == []


# ---- divisions ------------------------------------------------------------


def test_house_divided_and_committee_divided():
    assert "division_called" in types("The House divided: Ayes 262, Noes 159.")
    assert "division_called" in types("The Committee divided: Ayes 8, Noes 5.")
    assert "division_called" in types("The Committee proceeded to a Division.")


def test_test_the_opinion_of_the_house_calls_a_division():
    assert "division_called" in types("I wish to test the opinion of the House.")


def test_division_result_captures_tallies():
    e = event("The House divided: Ayes 262, Noes 159.", "division_result")
    assert e.description == "Ayes 262, Noes 159"


def test_lords_content_tallies():
    e = event("The Committee divided: Contents 258, Not-Contents 109.", "division_result")
    assert e.description == "Contents 258, Not-Contents 109"


def test_division_number_is_captured():
    e = event("Division No. 75]\n\nAYES", "division_number")
    assert e.description == "75"


def test_prose_about_divisions_is_not_an_event():
    assert types("If there is a Division in the Chamber while we are sitting the Committee will adjourn.") == []
    assert types("The Ayes have it, in my opinion, but I doubt that the House divided on this.") == []


# ---- resolutions / orders -------------------------------------------------


def test_resolved_captures_the_resolution():
    e = event("Resolved, That this House has considered the matter.", "resolved")
    assert e.description == "That this House has considered the matter."


def test_ordered_captures_the_order():
    e = event("Ordered, That the debate be now adjourned.", "ordered")
    assert e.description == "That the debate be now adjourned."


def test_resolved_and_ordered_with_the_text_on_the_next_paragraph():
    # Official Hansard puts the paragraph break after the comma.
    assert event("Resolved,\nThat this House has considered the matter.", "resolved").description == "That this House has considered the matter."
    assert event("Ordered,\nThat the debate be now adjourned.", "ordered").description == "That the debate be now adjourned."


def test_resolved_with_nothing_after_it_is_not_an_event():
    assert types("Resolved,") == []


def test_commons_numbered_oral_questions():
    e = event("T1. If she will make a statement on her departmental responsibilities.", "question_asked")
    assert e.description.startswith("T1. If she will make a statement")
    assert "question_asked" in types("10. What steps he is taking to improve UK-EU foreign and security co-operation.")
    assert "question_asked" in types("1. If he will take steps with Cabinet colleagues to review arms export licensing to Israel.")


def test_commons_oral_question_as_rendered_from_question_markup():
    # Real: <Question...>4. <QuestionText...>Whether he has set a target...</QuestionText></Question>
    raw = (
        '<Question HRSContentId="{A98C5EE5}">4.  <QuestionText HRSContentId="{9F262E7C}">'
        "Whether he has set a target for the deportation of foreign national offenders.</QuestionText></Question>"
    )
    e = event(raw, "question_asked")
    assert e.description == "4. Whether he has set a target for the deportation of foreign national offenders."


def test_urgent_question_opener():
    text = "(Urgent Question): To ask the Secretary of State for the Home Department if she will make a statement on prisons."
    e = event(text, "urgent_question", MEMBER)
    assert "Home Department" in e.description


def test_petition_presented():
    e = event("I rise to present a petition on behalf of 563 constituents from East Grinstead.", "petition_presented")
    assert e.description.startswith("I rise to present a petition")
    assert "petition_presented" in types("I have the honour to present a petition from the residents of Truro.")


def test_talking_about_a_petition_is_not_presenting_one():
    assert types("The petition was presented last week and I signed it.") == []
    assert types("I wish the hon. Member well in presenting her petition.") == []


def test_statement_worded_as_an_update_to_the_house():
    text = "With permission, I would like to provide the House with a brief update on the additional pressures on health this summer."
    e = event(text, "statement_begin")
    assert e.description.startswith("the additional pressures on health")


def test_numbered_paragraph_that_is_not_a_question_is_ignored():
    assert types("3. The Bill will come into force in April.") == []
    assert types("Clause 10. What follows is a list.") == []


def test_resolved_in_ordinary_prose_is_not_an_event():
    assert types("We have resolved the issue with the hon. Member. Resolved, then, to move on.") == []
    assert types("Many have tried to resolve this.") == []


# ---- bill stages ----------------------------------------------------------


def test_second_reading_motion_and_direction():
    assert "second_reading" in types("I beg to move, That the Bill be now read a Second time.")
    assert "second_reading" in types("That the Bill be now read a second time.", CHAIR)
    assert "second_reading" in types("Bill read a Second time.")
    assert "second_reading" in types("My Lords, I beg to move that this Bill be now read a second time.")


def test_clause_read_a_second_time_is_not_a_bill_second_reading():
    assert "second_reading" not in types("I beg to move, That the clause be read a Second time.")


def test_mentioning_second_reading_in_a_speech_is_not_an_event():
    assert types("At Second Reading the Minister promised more. The Bill was read a second time last month, I recall.") == []


def test_third_reading():
    assert "third_reading" in types("That the Bill do now pass.")
    assert "third_reading" in types("Bill read the Third time and passed.")
    assert "third_reading" in types("I beg to move, That the Bill be now read the Third time.")


def test_bill_committed():
    assert "bill_committed" in types("Bill committed to a Committee of the whole House.")


# ---- amendments / clauses -------------------------------------------------


@pytest.mark.parametrize(
    "text,outcome",
    [
        ("Amendment agreed to.", "agreed to"),
        ("Amendment negatived.", "negatived"),
        ("Amendment made.", "made"),
        ("Amendment 1 agreed.", "agreed"),
        ("Amendment withdrawn.", "withdrawn"),
    ],
)
def test_amendment_result(text, outcome):
    assert event(text, "amendment_result").description == outcome


def test_amendment_withdrawn_request():
    assert "amendment_withdrawn" in types("I beg leave to withdraw the amendment.")
    assert "amendment_withdrawn" in types("I withdraw the amendment.")
    assert "amendment_withdrawn" in types("In that case, I am grateful to the Minister and I therefore beg leave to withdraw the amendment.")


def test_withdrawing_funding_is_not_an_amendment_withdrawal():
    assert types("The Government withdrew the funding and I want them to withdraw the cut.") == []
    assert types("UK Research and Innovation withdrawing funding for the Jodrell Bank Observatory.") == []


def test_clause_result():
    assert "clause_result" in types("Clause 5 agreed to.")
    assert "clause_result" in types("New Clause 2 negatived.")


def test_closure():
    assert "closure" in types("I beg to move, That the Question be now put.")
    assert "closure" in types("That the Question be now put.", CHAIR)


def test_closure_phrase_followed_by_more_words_is_not_a_closure():
    # The closure motion is exactly "That the Question be now put." and nothing after it.
    assert types("The question be now put to the Minister, but he will not answer.") == []
    assert types("I doubt that the Question be now put is a fair way to end this debate.") == []


# ---- chair / sitting ------------------------------------------------------


def test_royal_assent_announcement_by_the_chair():
    text = "I have to notify the House, in accordance with the Royal Assent Act 1967, that the King has signified his Royal Assent to the following Acts."
    assert "royal_assent" in types(text, CHAIR)


def test_royal_assent_mentioned_by_a_member_is_not_an_event():
    # Real: "if this Bill received Royal Assent, the four-year implementation period..."
    text = "I point out that if this Bill received Royal Assent, the four-year implementation period would apply."
    assert types(text, "Lauren Edwards") == []
    assert types(text, CHAIR) == []


def test_royal_assent_wording_from_a_member_is_not_an_announcement():
    # The wording matches the announcement pattern, so only the chair guard stops it.
    text = "I am told the King has signified his Royal Assent to the following Acts."
    assert types(text, MEMBER) == []
    assert "royal_assent" in types(text, CHAIR)


def test_ordered_mid_line_is_prose_not_an_order():
    assert types("The Speaker said the House Ordered, That the debate stand adjourned, and so it did.") == []


def test_point_of_order_wording_mid_sentence_is_prose():
    assert types("He asked, on a point of order, whether the ruling stands.") == []
    assert types("Several colleagues rose on a point of order last week.") == []


def test_point_of_order():
    assert "point_of_order" in types("On a point of order, Mr Speaker.")
    assert "point_of_order" in types("Further to that point of order, Madam Deputy Speaker,")
    assert "point_of_order" in types("I rise on a point of order.")


def test_point_of_order_mentioned_in_prose_is_not_one():
    assert types("The hon. Member made a point of order last week that I did not follow.") == []


def test_adjournment_motion_and_sitting_adjourned():
    assert "adjournment_motion" in types("I beg to move, That this House do now adjourn.")
    assert "adjournment_motion" in types("That this House do now adjourn.", CHAIR)
    assert "sitting_adjourned" in types("The House adjourned at 10.30 pm.")
    assert "sitting_adjourned" in types("Adjourned at 3.31 pm.")
    assert "sitting_adjourned" in types("Sitting suspended.")


def test_adjournment_prose_is_not_an_event():
    assert types("We must not adjourn this debate without an answer.") == []


def test_urgent_question_is_only_recognised_from_the_chair():
    assert "urgent_question" in types("I call Bob Blackman to ask an urgent question.", CHAIR)
    assert types("I call Bob Blackman to ask an urgent question.", MEMBER) == []
    assert types("We have two further urgent questions, a statement and lots of business.", CHAIR) == []


def test_prorogation():
    assert "prorogation" in types("Prorogation\n\nThe Lord Speaker read the Commission.")


# ---- statements / questions / interests -----------------------------------


def test_ministerial_statement_commons():
    e = event("With permission, Mr Speaker, I will make a statement on local government reorganisation. This Government are committed...", "statement_begin")
    assert e.description == "local government reorganisation"


def test_ministerial_statement_without_the_speaker_address_and_wish_to():
    assert "statement_begin" in types("With permission, I will make a statement on public order following the protest.")
    assert "statement_begin" in types("With permission, Mr Speaker, I wish to make a statement on the direction of the country.")
    assert "statement_begin" in types("With permission, Madam Deputy Speaker, I would like to make a statement on the next steps.")


def test_lords_statement_repeat():
    assert "statement_begin" in types("My Lords, with the leave of the House, I shall now repeat a Statement made in another place.")


def test_lords_oral_question_asked():
    e = event("To ask His Majesty’s Government what assessment they have made of the implications of funding cuts.", "question_asked")
    assert e.description.startswith("To ask His Majesty")


def test_declaration_of_interest():
    e = event("My Lords, I declare an interest as chair of the Forestry Commission and my other environmental interests in the register.", "declaration_of_interest")
    assert "Forestry Commission" in e.description
    assert "declaration_of_interest" in types("I refer the House to my entry in the Register of Members’ Financial Interests.")
    assert "declaration_of_interest" in types("I draw attention to my registered interests as a councillor.")


def test_talk_about_interests_is_not_a_declaration():
    assert types("The Government should consider the interest rate and the public interest.") == []
    assert types("I refer to the interest rate rise announced last week.") == []
    assert types("I draw attention to the interests of the public.") == []


def test_maiden_speech_is_first_person_only():
    assert "maiden_speech" in types("It is a great privilege to make my maiden speech in this House.")
    assert types("I congratulate the hon. Member on her maiden speech.") == []


# ---- ordering / offsets ---------------------------------------------------


def test_char_start_indexes_the_cleaned_text():
    text = "Order.\n\nOn a point of order, Mr Speaker."
    e = event(text, "point_of_order")
    cleaned = "Order.\nOn a point of order, Mr Speaker."
    assert cleaned[e.char_start :].startswith("On a point of order")


def test_events_are_sorted_by_position_and_unique():
    evs = find_events("Question put.\nQuestion put and agreed to.\nResolved, That the Bill be read.")
    assert [e.char_start for e in evs] == sorted(e.char_start for e in evs)
    assert len({(e.event_type, e.char_start) for e in evs}) == len(evs)


def test_html_is_stripped_before_matching():
    assert "point_of_order" in types("<em>On a point of order, Mr Speaker.</em>")
    assert "motion_moved" in types('<strong>I beg to move, </strong>That the Bill be read.<span class="column-number"></span>')


def test_no_text_no_events():
    assert find_events(None) == []
    assert find_events("") == []
    assert find_events("   \n ") == []


# =========================================================================== 3. boilerplate
# ===========================================================================


def test_empty_speech_is_boilerplate_with_reason_empty():
    assert boilerplate_reason("") == "empty"
    assert boilerplate_reason(None) == "empty"
    assert boilerplate_reason('<span class="column-number"></span>') == "empty"
    assert boilerplate_reason("65.]") == "empty"


@pytest.mark.parametrize(
    "text,speaker,reason",
    [
        ("Order.", CHAIR, "order_call"),
        ("Order, order.", CHAIR, "order_call"),
        ("Order. Order.", "Madam Deputy Speaker (Caroline Nokes)", "order_call"),
        ("I call the Liberal Democrat spokesperson.", CHAIR, "chair_call"),
        ("I call Alex Sobel, who will speak for up to 15 minutes.", "Madam Deputy Speaker", "chair_call"),
        ("Order. I call the Minister.", CHAIR, "order_call"),
        ("The Question is as on the Order Paper.", CHAIR, "question_formula"),
        ("The Question is, That the amendment be made.", CHAIR, "question_formula"),
        ("Question put and agreed to.", None, "question_formula"),
        ("Question accordingly agreed to.", None, "question_formula"),
        ("The House divided: Ayes 262, Noes 159.", None, "question_formula"),
        ("Division No. 75]", None, "question_formula"),
        ("Resolved, That this House has considered the matter.", None, "resolution"),
        ("Ordered, That the debate be now adjourned.", None, "resolution"),
        ("Will the hon. Gentleman give way?", MEMBER, "give_way"),
        ("Will my right hon. Friend give way on that point?", MEMBER, "give_way"),
        ("On that point, will the hon. Lady give way?", MEMBER, "give_way"),
        ("<em>rose—</em>", MEMBER, "stage_direction"),
        ("<em>indicated assent.</em>", MEMBER, "stage_direction"),
        ("<em>[Interruption.]</em>", MEMBER, "stage_direction"),
        ("My Lords—", "Lord Pitt-Watson", "stage_direction"),
        ("That the Bill be now read a second time.", CHAIR, "formal_motion"),
        ("That the Bill do now pass.", "Lord Pitt-Watson", "formal_motion"),
        ("I beg to move, That the clause be read a Second time.", "The Minister for Secondary Care (Karin Smyth)", "formal_motion"),
        ("I beg to move.", MEMBER, "formal_motion"),
        ("I wish to test the opinion of the House.", MEMBER, "formal_motion"),
        ("I beg leave to withdraw the amendment.", "Lord Moylan (Con)", "formal_motion"),
        ("I withdraw the amendment.", "Baroness Kramer (LD)", "formal_motion"),
        ("Content.", LORDS_CHAIR, "division_formula"),
        ("Not content.", LORDS_CHAIR, "division_formula"),
    ],
)
def test_whole_speech_boilerplate(text, speaker, reason):
    assert boilerplate_reason(text, speaker) == reason


def test_mixed_formula_speech_reports_the_first_reason():
    assert boilerplate_reason("Order. I call the Minister.", CHAIR) == "order_call"


@pytest.mark.parametrize(
    "text,speaker",
    [
        # substantive content
        ("I beg to move, That this House has considered Israel and Palestine.\n\nAs this is my first debate as Minister...", MEMBER),
        ("I beg to move, That the Bill be now read a Second time.\n\nGood housing is the foundation of a flourishing life.", MEMBER),
        ("Will the Minister give way? I ask because the figures are wrong.", MEMBER),
        # a give-way formula needs a real addressee (the / my / he / she / I), not any subject after "would"
        ("Would it not be wise to give way on that point?", MEMBER),
        ("Thank you.", MEMBER),
        ("Yes!", MEMBER),
        ("The Government are wrong on housing.", CHAIR),
        # chair-only formulas must not fire for members
        ("I call the shadow Minister.", MEMBER),
        ("Order.", MEMBER),
    ],
)
def test_not_boilerplate(text, speaker):
    assert boilerplate_reason(text, speaker) is None


def test_i_call_is_only_boilerplate_from_the_chair():
    assert boilerplate_reason("I call the shadow Minister.", CHAIR) == "chair_call"
    assert boilerplate_reason("I call the shadow Minister.", "Jim Shannon (Strangford) (DUP)") is None
    assert boilerplate_reason("I call the shadow Minister.", None) is None


def test_chair_housekeeping_is_boilerplate_only_from_the_chair():
    text = "There will be a formal two-minute time limit on speeches."
    assert boilerplate_reason(text, "Sir Desmond Swayne (in the Chair)") == "chair_housekeeping"
    assert boilerplate_reason(text, MEMBER) is None
    assert boilerplate_reason("Order. The time limit is eight minutes.", "Lord in Waiting/Government Whip (Lord Katz) (Lab)") is None


def test_chair_housekeeping_alone_and_after_order():
    assert boilerplate_reason("We have two further urgent questions and lots of business.", CHAIR) == "chair_housekeeping"
    # Real: "Order.
    text = "Order. We have about 10 minutes for questions and answers, so they must be brief."
    assert boilerplate_reason(text, "Madam Deputy Speaker (Ms Nusrat Ghani)") == "order_call"
    assert boilerplate_reason(text, MEMBER) is None


@pytest.mark.parametrize(
    "text,reason",
    [
        ("That brings us to the Front-Bench contributions. I call the shadow Minister.", "chair_housekeeping"),
        ("For the final and 125th question, I call Gideon Amos.", "chair_call"),
        ("Order. People should not leave when a Member is on their feet; it is discourteous.", "order_call"),
        ("Colleagues must resume their seats.", "chair_housekeeping"),
    ],
)
def test_chair_housekeeping_seen_in_the_corpus(text, reason):
    assert boilerplate_reason(text, "Madam Deputy Speaker (Ms Nusrat Ghani)") == reason
    assert boilerplate_reason(text, MEMBER) is None


def test_bare_resolved_or_ordered_sentence_is_a_formula():
    assert sentence_boilerplate_reason("Resolved,", None) == "resolution"
    assert sentence_boilerplate_reason("Ordered,", None) == "resolution"
    assert sentence_boilerplate_reason("Resolved, then, to move on.", None) is None


def test_long_chair_speech_is_not_housekeeping():
    text = "Order. " + "The Chair must rule on a substantive matter of privilege raised earlier, and here is the reasoning in full. " * 4
    assert boilerplate_reason(text, CHAIR) is None


def test_sentence_level_reason_matches_speech_level():
    assert sentence_boilerplate_reason("Order.", CHAIR) == "order_call"
    assert sentence_boilerplate_reason("[Interruption.]", MEMBER) == "stage_direction"
    assert sentence_boilerplate_reason("The Government are wrong.", MEMBER) is None
    assert sentence_boilerplate_reason("I call the Minister.", MEMBER) is None


def test_speech_with_one_substantive_sentence_is_not_boilerplate_but_sentences_are_flagged_individually():
    text = "Order. The Government are wrong on housing."
    assert boilerplate_reason(text, CHAIR) is None
    assert sentence_boilerplate_reason("Order.", CHAIR) == "order_call"
    assert sentence_boilerplate_reason("The Government are wrong on housing.", CHAIR) is None


# =========================================================================== classify_motion
# ===========================================================================


@pytest.mark.parametrize(
    "text,expected",
    [
        ("That this House has considered the future of rail.", "neutral"),
        ("that this House has considered Israel and Palestine.", "neutral"),
        ("That this House takes note of the Report of the Economic Affairs Committee.", "neutral"),
        ("That this House has considered the matter of local government reorganisation.", "neutral"),
        ("That this House doth agree with the Lords in their Amendment 1.", "substantive"),
        ("That this House regrets that the Government have failed.", "substantive"),
        ("That the Bill be now read a Second time.", "substantive"),
        ("That leave be given to bring in a Bill to make provision.", "substantive"),
        ("That the draft Regulations be approved.", "substantive"),
    ],
)
def test_classify_motion(text, expected):
    assert classify_motion(text) == expected


def test_classify_motion_unknown_for_blank():
    assert classify_motion(None) is None
    assert classify_motion("") is None


def test_this_house_has_considered_only_counts_at_the_start():
    assert classify_motion("That this House regrets that it has considered nothing.") == "substantive"


# =========================================================================== classify_debate
# ===========================================================================


@pytest.mark.parametrize(
    "heading,section,house,events,expected",
    [
        ("Engagements", "Commons Chamber", "Commons", set(), "prime_ministers_questions"),
        ("Topical Questions", "Commons Chamber", "Commons", set(), "topical_questions"),
        ("Business of the House", "Commons Chamber", "Commons", set(), "business_statement"),
        ("Arrangement of Business", "Grand Committee", "Lords", set(), "business_statement"),
        ("Speaker’s Statement", "Commons Chamber", "Commons", set(), "speaker_statement"),
        ("Point of Order", "Commons Chamber", "Commons", set(), "point_of_order"),
        ("Points of Order", "Commons Chamber", "Commons", set(), "point_of_order"),
        ("Delegated Legislation", "Commons Chamber", "Commons", set(), "delegated_legislation"),
        ("Draft Armed Forces (Service Complaints Miscellaneous Provisions) (Amendment) Regulations 2026", "General Committees", "Commons", set(), "delegated_legislation"),
        ("Building Societies Act 1986 (Assimilation to Company Law) Order 2026", "Grand Committee", "Lords", set(), "delegated_legislation"),
        ("Windsor Framework (Retail Movement Scheme: Plant Health) (Amendment) Regulations 2026", "Lords Chamber", "Lords", set(), "delegated_legislation"),
        ("Health Bill", "Commons Chamber", "Commons", set(), "bill"),
        ("Social Housing Bill [HL]", "Lords Chamber", "Lords", set(), "bill"),
        ("Social Housing Bill [Lords]", "Commons Chamber", "Commons", set(), "bill"),
        ("Deaths of Members", "Lords Chamber", "Lords", set(), "tributes"),
        ("Retirement of a Member: Lord Jopling", "Lords Chamber", "Lords", set(), "tributes"),
        ("Humble Address", "Lords Chamber", "Lords", set(), "humble_address"),
        ("Preparing for an Ageing Society (Economic Affairs Committee Report)", "Lords Chamber", "Lords", set(), "committee_report"),
        # events beat headings and sections
        ("Anything", "Commons Chamber", "Commons", {"adjournment_motion"}, "adjournment"),
        ("Anything", "Commons Chamber", "Commons", {"urgent_question"}, "urgent_question"),
        ("Local Government Reorganisation", "Commons Chamber", "Commons", {"statement_begin"}, "statement"),
        ("Rewiring the State", "Commons Chamber", "Commons", {"leave_to_bring_in_bill", "motion_moved"}, "ten_minute_rule"),
        ("Health Bill", "Commons Chamber", "Commons", {"second_reading"}, "second_reading"),
        ("Health Bill", "Commons Chamber", "Commons", {"third_reading"}, "third_reading"),
        ("Copthorne Hotel", "Commons Chamber", "Commons", {"petition_presented"}, "petition"),
        ("Nepal: Flooding", "Lords Chamber", "Lords", {"question_asked"}, "oral_questions"),
        ("Foreign, Commonwealth and Development Office", "Commons Chamber", "Commons", {"question_asked"}, "oral_questions"),
        ("Housing", "Commons Chamber", "Commons", {"question_asked", "statement_begin"}, "statement"),
        # section fallbacks
        ("Local Authorities: Aluminium Recycling", "Westminster Hall", "Commons", set(), "westminster_hall_debate"),
        ("Cyber Security", "Grand Committee", "Lords", set(), "grand_committee"),
        ("Israel and Palestine", "Commons Chamber", "Commons", {"motion_moved"}, "general_debate"),
        ("Strait of Hormuz", "Commons Chamber", "Commons", set(), "other"),
    ],
)
def test_classify_debate(heading, section, house, events, expected):
    assert classify_debate(heading, section, house, events) == expected


def test_classify_debate_tolerates_missing_fields():
    assert classify_debate(None, None, None, set()) == "other"
    assert classify_debate("", "", "", set()) == "other"


def test_special_headings_beat_events():
    # A Business Statement contains "With permission, I will make a statement" but is still business.
    assert classify_debate("Business of the House", "Commons Chamber", "Commons", {"statement_begin"}) == "business_statement"
    assert classify_debate("Engagements", "Commons Chamber", "Commons", {"motion_moved"}) == "prime_ministers_questions"
    assert classify_debate("Points of Order", "Commons Chamber", "Commons", {"question_put"}) == "point_of_order"


def test_bill_word_in_the_middle_of_a_heading_is_not_a_bill():
    assert classify_debate("Billing Practices in Water Companies", "Commons Chamber", "Commons", set()) == "other"


# --------------------------------------------------------------------------- A new Parliament opening: events no 13-day
# window could contain --------------------------------------------------------------------------- W1 holds a
# dissolution and a State Opening.


def test_members_being_sworn_is_an_event():
    text = ("The following Members took and subscribed the Oath, or made and subscribed the "
            "Affirmation required by law: Right honourable Sir Edward Julian Leigh")
    events = {e.event_type for e in find_events(text)}
    assert "oath_or_affirmation" in events


def test_the_speaker_swearing_alone_is_also_an_event():
    text = ("The Speaker first, alone, standing on the upper step of the Chair, took and "
            "subscribed the Oath required by law.")
    assert "oath_or_affirmation" in {e.event_type for e in find_events(text)}


def test_the_words_of_the_oath_and_of_the_affirmation_both_count():
    for text in ("I swear by Almighty God that I will be faithful and bear true allegiance",
                 "I do solemnly, sincerely and truly declare and affirm that I will be faithful"):
        assert "oath_or_affirmation" in {e.event_type for e in find_events(text)}, text


def test_a_speaker_election_is_an_event():
    text = ("Before I take the Chair as Speaker-Elect, I wish to thank the House for the "
            "honour it has again bestowed upon me.")
    assert "speaker_elected" in {e.event_type for e in find_events(text)}


def test_a_humble_address_motion_is_an_event_with_its_motion_text():
    text = "That an Humble Address be presented to His Majesty, as follows: Most Gracious Sovereign"
    events = [e for e in find_events(text) if e.event_type == "humble_address"]
    assert events and events[0].description.startswith("That an Humble Address")


def test_ordinary_prose_about_an_oath_or_a_speaker_is_not_an_event():
    """The guard rail. These rules fire on formulaic minuting, not on anyone who happens to mention the Speaker or an
    affirmation in a speech.
    """
    for text in ("I want to affirm my commitment to the Speaker's ruling on this matter.",
                 "The Minister should take the Chair of the inquiry seriously.",
                 "We swear that this Government has failed the country."):
        types = {e.event_type for e in find_events(text)}
        assert not types & {"oath_or_affirmation", "speaker_elected", "humble_address"}, text


# Every heading below is a REAL one, taken from W1 or W3.
@pytest.mark.parametrize("heading, expected", [
    ("Election of Speaker", "speaker_election"),
    ("Members Sworn", "members_sworn"),
    ("Oaths and Affirmations", "members_sworn"),
    ("Debate on the Address", "address_debate"),
    ("King’s Speech", "address_debate"),
    ("King’s Speech (4th Day)", "address_debate"),
    ("His Majesty’s Most Gracious Speech", "address_debate"),
    ("Prorogation", "opening_or_closing"),
    ("Remote Voting", "proceedings_arrangement"),
    ("Hybrid Substantive Proceedings", "proceedings_arrangement"),
    ("Hybrid Proceedings in Parliament", "proceedings_arrangement"),
    ("Hybrid Proceedings (Extension of Temporary Orders)", "proceedings_arrangement"),
    ("Proceedings during the Pandemic and Hybrid Scrutiny Proceedings", "proceedings_arrangement"),
    ("Remote Division Result: Amendment 39", "remote_division"),
])
def test_the_headings_of_a_new_parliament_are_classified_rather_than_falling_through(heading, expected):
    """Without these, W1's and W3's most distinctive days are all `other`, and the load report's rules coverage says
    nothing about them.
    """
    assert classify_debate(heading, "Commons Chamber", "Commons", set()) == expected


# These are also real headings from W1 and W3, and every one of them is a trap: an unanchored version of the patterns
# above swallows all seven.
@pytest.mark.parametrize("heading, expected", [
    ("Asylum Seekers:  Missing from Registered Address", "other"),
    ("Councillors: Publication of Addresses", "other"),
    ("Protection of Free Speech", "other"),
    ("Courts (Remote Hearings) Bill", "bill"),
    ("Covid-19: Remote Working", "other"),
    ("Remote Access to Church Services", "other"),
    ("Divorce, Dissolution and Separation Bill [HL]", "bill"),
    ("Senior Deputy Speaker", "other"),
    ("Deputy Speakers", "other"),
])
def test_the_headings_those_patterns_must_not_swallow(heading, expected):
    assert classify_debate(heading, "Commons Chamber", "Commons", set()) == expected


def test_a_speakers_statement_is_still_a_speakers_statement():
    """`Speaker’s Statement` is checked before `Election of Speaker`, and must stay that way: both exist in W1, 18
    times and 34 times.
    """
    assert classify_debate("Speaker’s Statement", "Commons Chamber", "Commons", set()) == "speaker_statement"
    assert classify_debate("Lord Speaker’s Statement", "Lords Chamber", "Lords", set()) == "speaker_statement"


def test_a_remote_division_result_is_an_event():
    assert "remote_division_result" in {e.event_type for e in find_events("Remote Division result: Ayes 320, Noes 253")}


# --- the ten-year span: labels from 2015-2021 that W0-W5 never contained -----


@pytest.mark.parametrize("label, person, seat, party", [
    ("Douglas Carswell (Clacton) (UKIP)", "Douglas Carswell", "Clacton", "UKIP"),
    ("Mark Durkan (Foyle) (SDLP)", "Mark Durkan", "Foyle", "SDLP"),
    ("Chuka Umunna (Streatham) (Change UK)", "Chuka Umunna", "Streatham", "Change UK"),
    ("Anna Soubry (Broxtowe) (TIG)", "Anna Soubry", "Broxtowe", "TIG"),
    ("Heidi Allen (South Cambridgeshire) (Independent Group for Change)",
     "Heidi Allen", "South Cambridgeshire", "Independent Group for Change"),
    ("Kenny MacAskill (East Lothian) (Alba)", "Kenny MacAskill", "East Lothian", "Alba"),
    ("Lady Hermon (North Down) (Ind)", "Lady Hermon", "North Down", "Ind"),
    ("Lord Pearson of Rannoch (UKIP)", "Lord Pearson of Rannoch", None, "UKIP"),
    ("Lord Willoughby de Broke (UKIP)", "Lord Willoughby de Broke", None, "UKIP"),
    ("Baroness Hoey (Non-Afl)", "Baroness Hoey", None, "Non-Afl"),
    ("Lord Fox (LD) [V]", "Lord Fox", None, "LD"),
    ("Viscount Ridley (Con)", "Viscount Ridley", None, "Con"),
    ("Lord Hendy (Lab) (Maiden Speech)", "Lord Hendy", None, "Lab"),
])  # fmt: skip
def test_member_labels_from_the_whole_ten_year_span(label, person, seat, party):
    """Party is read from the SHAPE of the label, not from a list of the parties 2020 and 2024-26 happened to hold.
    """
    who = parse_speaker(label)
    assert (who.person, who.constituency, who.party) == (person, seat, party)


@pytest.mark.parametrize("label, person, party", [
    ("Lord Stewartby (Con)", "Lord Stewartby", "Con"),  # a real peer, not "Steward"
    ("Lord Highbury (Lab)", "Lord Highbury", "Lab"),  # "High" guards a word, not a prefix
    ("Lord Greaves (LD)", "Lord Greaves", "LD"),
    ("Lord Justice-Smith (Crossbench)", "Lord Justice-Smith", "Crossbench"),  # hypothetical: a hyphen is not a word end
])  # fmt: skip
def test_a_peer_whose_title_merely_begins_like_an_office_keeps_their_party(label, person, party):
    who = parse_speaker(label)
    assert (who.person, who.party, who.office) == (person, party, None)


def test_a_single_bracket_after_an_mp_is_still_a_seat():
    """Only a PEER's single bracket is a party: "Lady" is not a peer title here (Lady Hermon sat for North Down)."""
    who = parse_speaker("Lady Hermon (North Down)")
    assert (who.constituency, who.party) == ("North Down", None)


@pytest.mark.parametrize("label, office, person", [
    ("The Prime Minister (Boris Johnson)", "The Prime Minister", "Boris Johnson"),
    ("The Secretary of State for Health and Social Care (Matt Hancock)",
     "The Secretary of State for Health and Social Care", "Matt Hancock"),
    ("The Lord Speaker (Lord Fowler)", "The Lord Speaker", "Lord Fowler"),
    # "The Lord..." is an office; its holder's name carries no honorific (27 real speeches, caught before writing)
    ("The Lord Chancellor and Secretary of State for Justice (Alex Chalk)",
     "The Lord Chancellor and Secretary of State for Justice", "Alex Chalk"),
    ("The Lord Commissioner of His Majesty’s Treasury (Taiwo Owatemi)",
     "The Lord Commissioner of His Majesty’s Treasury", "Taiwo Owatemi"),
    # The same offices WITHOUT "The" read the holder as a peer's party and the office as a person.
    ("Lord Chancellor (Alex Chalk)", "Lord Chancellor", "Alex Chalk"),
    ("Lord President of the Council (Penny Mordaunt)", "Lord President of the Council", "Penny Mordaunt"),
    ("Lord Commissioner of His Majesty’s Treasury (Taiwo Owatemi)",
     "Lord Commissioner of His Majesty’s Treasury", "Taiwo Owatemi"),
    ("Lord Advocate (Dorothy Bain)", "Lord Advocate", "Dorothy Bain"),
    ("Lord Privy Seal (Lord True)", "Lord Privy Seal", "Lord True"),
])  # fmt: skip
def test_an_office_with_its_holder_is_never_read_as_a_party(label, office, person):
    who = parse_speaker(label)
    assert (who.office, who.person, who.party) == (office, person, None)


def test_covid_remote_contributions_are_flagged_in_both_houses():
    assert "virtual" in parse_speaker("Theresa May (Maidenhead) (Con) [V]").flags
    assert "virtual" in parse_speaker("Lord Fox (LD) [V]").flags
