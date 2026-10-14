"""Every one of these shipped because nothing tested it, and every one was found by reading the `answer_trace` rows of
a failed run rather than by reasoning about the code. They are gathered in one file because they share a cause worth
naming: each was a check that looked right and passed for the wrong reason.
"""

from __future__ import annotations

import psycopg
import pytest

from answer import facts as F
from answer import resolve as RS
from answer import router as RO

# --- 1. a clause number that CONTRADICTS must lose ------------------------------------------------------ The original
# comparison was `title_marks & asked_marks`, a plain set intersection over strings like {"newclause1", "no2"}.

FINANCE_Q = "Was Mick Whitley an Aye or a No on Finance (No. 2) Bill Committee: New Clause 3 on 8 May 2024?"
FINANCE_TITLES = [f"Finance (No. 2) Bill Committee: New Clause {n}" for n in (1, 2, 3, 4)]


def test_the_named_clause_beats_the_others_of_the_same_bill():
    """The exact defect: four divisions, one Bill, one day, and the question names one of them."""
    scores = {title: RS.title_score(FINANCE_Q, title) for title in FINANCE_TITLES}
    best = max(scores, key=scores.get)
    assert best.endswith("New Clause 3")
    runner_up = sorted(scores.values(), reverse=True)[1]
    assert scores[best] - runner_up >= RS.MIN_MARGIN, "must beat the runner-up by the resolver's own margin"
    assert scores[best] >= RS.MIN_SCORE


def test_a_matching_mark_elsewhere_does_not_rescue_a_contradicting_one():
    """"No. 2" matching is not permission to ignore that New Clause 1 is not New Clause 3."""
    assert RS.title_score(FINANCE_Q, "Finance (No. 2) Bill Committee: New Clause 1") < \
           RS.title_score(FINANCE_Q, "Finance (No. 2) Bill Committee: New Clause 3")  # fmt: skip


def test_a_contradiction_is_penalised_harder_than_a_gap():
    """Naming a DIFFERENT clause is positive evidence of the wrong division; naming none is merely silence."""
    contradicts = RS.title_score("was New Clause 3 agreed?", "Health Bill: New Clause 9")
    silent = RS.title_score("was the Health Bill motion agreed?", "Health Bill: New Clause 9")
    assert contradicts < silent


def test_the_question_s_own_date_is_not_title_evidence():
    """The date already chose the day; it is not a title."""
    q = "What was the result of the Commons division on 20 May 2020?"
    assert RS.title_score(q, "Business of the House (20 May)") < RS.MIN_SCORE


def test_a_title_that_names_a_date_is_still_found_when_the_question_names_the_title():
    q = "What was the result of the vote on Business of the House (20 May) on 20 May 2020?"
    assert RS.title_score(q, "Business of the House (20 May)") >= RS.MIN_SCORE
    assert RS.title_score(q, "Business of the House (20 May)") - RS.title_score(q, "Trade Bill: Second Reading") \
        >= RS.MIN_MARGIN  # fmt: skip


def test_a_month_named_in_a_title_still_counts_when_the_question_quotes_it():
    """Review pass 2: only full dates are the question's filter. "June 2025" inside a quoted title is part of the
    title, and stripping it halved the score.
    """
    q = "What was the result of the Commons division on Spending Review (June 2025) on 18 June 2025?"
    assert RS.title_score(q, "Spending Review (June 2025)") >= 1.0


def test_a_year_that_is_part_of_a_title_still_counts():
    q = "Did the Commons agree to the Draft Medical Devices (Fees Amendment) Regulations 2026 on 28 January 2026?"
    assert RS.title_score(q, "Draft Medical Devices (Fees Amendment) Regulations 2026") >= RS.MIN_SCORE


def test_marks_are_grouped_by_kind():
    """The data structure the fix turns on: a flat set cannot tell a different clause from no clause."""
    found = RS._marks("Health Bill: New Clause 6, Amendment 12")
    assert found["new clause"] == {"6"} and found["amendment"] == {"12"}


# --- 2. a House named inside an amendment reference is not the House that was sitting ------------------- Division 2239
# is a COMMONS division called "Sentencing Bill: motion to disagree with Lords Amendment 7", and the bare word "Lords"
# forced house=Lords onto any question quoting it, after which no division matched.

GITTINS_Q = ("Did Becky Gittins support Sentencing Bill: motion to disagree with Lords Amendment 7 "
             "when it was divided on, on 20 January 2026?")


def test_a_quoted_amendment_title_does_not_decide_the_house():
    """The original defect. Not "it means Commons" — it means nothing, and nothing is the safe answer."""
    assert RO.extract_filters(GITTINS_Q).house is None


@pytest.mark.parametrize(("question", "expected"), [
    # Drawn from outside W0-W5 on purpose: these are the probes that caught the overfitted rule.
    ("Did the Lords agree to the Commons Amendment 4 on 3 March 2027?", "Lords"),
    ("How did the Lords vote on the Commons reasons on 4 April 2027?", "Lords"),
    ("Was Lords Message considered in the Lords on 9 May 2028?", "Lords"),
    ("What did the Commons do with Lords Amendment 7?", "Commons"),
    # And the forms this corpus itself uses, which the first regex missed.
    ("Government motion to disagree with the Lords in their Amendment 5", None),
    ("was the motion to disagree with Lords Amendments 3 and 4 agreed on 1 May 2026?", None),
    ("was the motion to disagree with Lords Reasons agreed on 1 May 2026?", None),
])
def test_an_explicit_house_outstanks_a_house_inside_a_title(question, expected):
    """Blanked rather than skipped: an explicit "in the Lords" elsewhere in the same question still counts."""
    assert RO.extract_filters(question).house == expected


@pytest.mark.parametrize("question", [
    "How many divisions did the Lords hold in September 2026?",
    "What did the Lords say about housing?",
    "Was the motion agreed in the House of Lords?",
    "How did the peers vote?",
])
def test_a_genuine_lords_question_still_reads_as_lords(question):
    """The fix must not cost the ordinary case. "in the Lords" means the Lords."""
    assert RO.extract_filters(question).house == "Lords"


# --- 2b. the mark kinds are Parliament's, not this corpus's ---------------------------------------------- W0-W5 titles
# carry only `amendment`, `new clause`, `no.` and `clause`.


@pytest.mark.parametrize(("text", "kind", "number"), [
    ("Health Bill: Schedule 4", "schedule", "4"),
    ("Health Bill: New Schedule 2", "new schedule", "2"),
    ("Business of the House: Motion 2", "motion", "2"),
    ("Standing Order No. 24 application", "standing order", "24"),
    ("Ways and Means Motion 3", "ways and means motion", "3"),
    ("Instruction 1 to the Committee", "instruction", "1"),
])
def test_a_mark_kind_parliament_uses_is_visible(text, kind, number):
    assert RS._marks(text).get(kind) == {number}


def test_new_schedule_is_not_read_as_schedule():
    """Same reason "new clause" is not "clause": they identify different things, and a scorer that conflates them
    reports agreement between a question and the wrong division.
    """
    assert "schedule" not in RS._marks("New Schedule 2")


def test_the_abbreviated_lords_amendment_this_corpus_uses_is_seen():
    """`LA62` is how the Employment Rights Bill ping-pong titles write Lords Amendment 62, and it was invisible. It
    must land in the same bucket as the spelled-out form or the comparison reads a contradiction.
    """
    assert RS._marks("motion to insist on disagreement to LA62") == RS._marks("Lords Amendment 62")
    assert RS._marks("to LA62").get("amendment") == {"62"}


# --- 3. the asker's spelling of a name, not the corpus's -------------------------------------------------
# `member_id_map` stores "Dr Al Pinkerton" and the match requires the stored name to appear in the question, so "Was Al
# Pinkerton an Aye or a No" matched nothing.

MEMBERS = [
    (5288, "Dr Al Pinkerton", "Commons"),
    (4542, "Lord Smith of Hindhead", "Lords"),
    (1606, "John Howell", "Commons"),
    (4032, "Sir Stuart Andrew", "Commons"),
    (3308, "Viscount Chandos", "Lords"),
]


@pytest.fixture
def members(scratch, scratch_dsn):
    scratch.execute("TRUNCATE member_id_map CASCADE")
    for mnis, name, house in MEMBERS:
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (%s, %s, %s)", (mnis, name, house))
    yield scratch_dsn
    scratch.execute("TRUNCATE member_id_map CASCADE")


def resolved(dsn: str, question: str):
    from ingest.runtime import run as run_async

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await RS.mnis_id(conn, question)
        finally:
            await conn.close()

    return run_async(go())


@pytest.mark.parametrize(("question", "expected"), [
    ("Was Al Pinkerton an Aye or a No on 27 October 2025?", 5288),
    ("Was Dr Al Pinkerton an Aye or a No?", 5288),
    ("Did Stuart Andrew vote for it?", 4032),
    ("Did John Howell vote for it?", 1606),
])
def test_a_member_is_found_without_the_stored_honorific(members, question, expected):
    assert resolved(members, question)[0] == expected


def test_a_peerage_title_is_not_stripped(members):
    """The reason the honorific list excludes Lord and Baroness: stripping them would let a bare surname match a peer,
    and "Smith" is not "Lord Smith of Hindhead".
    """
    found, why = resolved(members, "Did Smith vote for it?")
    assert found is None, why


def test_a_peer_is_still_found_by_their_full_form(members):
    assert resolved(members, "Did Lord Smith of Hindhead support it?")[0] == 4542
    assert resolved(members, "Was Viscount Chandos an Aye or a No?")[0] == 3308


def test_the_honorific_pattern_leaves_peerage_forms_alone():
    import re

    strip = re.compile(RS._HONORIFIC, re.I)
    assert strip.sub("", "Dr Al Pinkerton") == "Al Pinkerton"
    assert strip.sub("", "Sir Stuart Andrew") == "Stuart Andrew"
    assert strip.sub("", "Lord Smith of Hindhead") == "Lord Smith of Hindhead"
    assert strip.sub("", "Baroness Featherstone") == "Baroness Featherstone"


# --- 3b.

LIST_TITLE = "Agriculture Bill: Report Stage: New Clause 2 (Neil Parish)"
LIST_MEMBERS = [(4484, "Alan Mak"), (4072, "Neil Parish"), (8, "Baroness May of Maidenhead"), (4018, "Jesse Norman"),
                (9018, "Jesse Norman"), (4597, "Kemi Badenoch")]  # 9018 and 4597 cast no vote in 784
LIST_VOTES = [(4484, "Alan Mak", "no"), (4072, "Neil Parish", "aye"), (8, "Theresa May", "no"),
              (4018, "Jesse Norman", "no")]  # 9018 is not in this division


@pytest.fixture
def division_list(scratch, scratch_dsn):
    scratch.execute("TRUNCATE member_id_map, division, division_vote CASCADE")
    for mnis, name in LIST_MEMBERS:
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (%s, %s, 'Commons')", (mnis, name))
    scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, ayes, noes) "
                    "VALUES (784, 784, 'Commons', '2020-05-13', %s, 1, 3)", (LIST_TITLE,))  # fmt: skip
    for mnis, printed, lobby in LIST_VOTES:
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby) VALUES (784, %s, %s, %s)",
                        (mnis, printed, lobby))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE member_id_map, division, division_vote CASCADE")


def resolved_in(dsn: str, question: str, division_ids=(784,)):
    from ingest.runtime import run as run_async

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await RS.mnis_id(conn, question, division_ids=division_ids)
        finally:
            await conn.close()

    return run_async(go())


@pytest.mark.parametrize("question", [
    f"How did Alan Mak vote in the division on {LIST_TITLE} on 13 May 2020?",
    f"Which way did Alan Mak vote on {LIST_TITLE} on 13 May 2020?",
    "How did Alan Mak vote on Neil Parish's new clause on 13 May 2020?",  # the mover outside the title
    f"Was Alan Mak an Aye or a No on {LIST_TITLE.upper()}?",  # the title in another case
])
def test_a_name_inside_the_title_or_beside_the_verb_s_subject_is_not_the_voter(division_list, question):
    assert resolved_in(division_list, question)[0] == 4484


@pytest.mark.parametrize("question", [
    "How did Boris Johnson vote on Neil Parish's new clause 2 to the Agriculture Bill on 13 May 2020?",
    "How did the Prime Minister vote on Neil Parish's new clause on 13 May 2020?",
    "How did Boris Johnson vote on the Agriculture Bill new clause 2 (Neil Parish) on 13 May 2020?",
    "How did Labour MPs vote on Neil Parish's new clause on 13 May 2020?",
])  # fmt: skip
def test_a_named_subject_who_is_not_in_the_list_never_lets_another_name_answer(division_list, question):
    """When the member asked about is absent from the list, the fallback matched the whole question and the mover named
    elsewhere answered. The subject of the verb is who is asked about; if it names nobody the record can place, the
    answer is a refusal, never someone else.
    """
    found, why = resolved_in(division_list, question)
    assert found != 4072, why


def test_the_mover_is_still_found_when_the_question_asks_about_the_mover(division_list):
    assert resolved_in(division_list, f"How did Neil Parish vote in the division on {LIST_TITLE}?")[0] == 4072


def test_a_member_is_found_under_the_name_the_division_list_printed(division_list):
    found, why = resolved_in(division_list, f"How did Theresa May vote in the division on {LIST_TITLE}?")
    assert found == 8 and "list" in why


def test_a_name_two_members_share_is_settled_by_who_is_in_the_division(division_list):
    assert resolved_in(division_list, f"How did Jesse Norman vote in the division on {LIST_TITLE}?")[0] == 4018
    found, why = resolved_in(division_list, "Did Jesse Norman speak about farming?", division_ids=())
    assert found is None and "2 members" in why, "with no division to settle it, a shared name is still refused"


def test_a_member_outside_the_list_is_still_resolved_so_the_record_can_say_they_did_not_vote(division_list):
    """4597 cast no vote in 784. The fact engine then says so, which is the true answer; refusing the name would hide
    it.
    """
    found, why = resolved_in(division_list, f"How did Kemi Badenoch vote in the division on {LIST_TITLE}?")
    assert found == 4597 and "list" not in why


# --- 4. what two candidate divisions actually disagree about --------------------------------------------
# `pipeline.ask` answers an ambiguous division question when every candidate gives the same answer.


def result_fact(division_id: int, **over):
    value = {"division_id": division_id, "house": "Lords", "division_number": division_id,
             "date": "2026-09-09", "result": "agreed", "ayes": 248, "noes": 163}  # fmt: skip
    value.update(over)
    return F.Fact("division_result", {}, value=value)


def test_candidates_differing_only_by_identity_agree():
    facts = [result_fact(i) for i in (11, 12, 13)]
    assert len({f.answer_fields() for f in facts}) == 1


def test_candidates_differing_in_the_answer_do_not_agree():
    facts = [result_fact(11), result_fact(12, result="negatived"), result_fact(13)]
    assert len({f.answer_fields() for f in facts}) > 1


def test_a_differing_tally_is_a_disagreement_for_a_result_question():
    facts = [result_fact(11), result_fact(12, ayes=188, noes=125)]
    assert len({f.answer_fields() for f in facts}) > 1


def test_a_vote_question_compares_only_the_lobby():
    """Which division a member voted in is not the answer; how they voted is."""
    facts = [F.Fact("division_vote", {}, value={"division_id": i, "mnis_id": 3308, "member": "Viscount Chandos",
                                                "lobby": "no", "party_at_vote": "Labour", "is_teller": False})
             for i in (1003712, 1003713, 1003714)]  # fmt: skip
    assert len({f.answer_fields() for f in facts}) == 1


@pytest.mark.parametrize("differing", [{"is_teller": True}, {"proxy_name": "Lord Whip"}])
def test_candidates_that_differ_in_anything_the_sentence_states_do_not_agree(differing):
    """Found by the ST probes: Baroness Williams of Trafford told in one of the Public Authorities Bill divisions on 21
    October 2025 and only voted in another. Comparing the lobby alone called them the same answer, so "voted Content
    and acted as a teller" was said of a division in which she did not tell.
    """
    base = {"division_id": 1003404, "mnis_id": 4311, "member": "Baroness Williams of Trafford", "lobby": "aye",
            "party_at_vote": "Conservative", "is_teller": False, "proxy_name": None}
    one = F.Fact("division_vote", {}, value=base)
    other = F.Fact("division_vote", {}, value={**base, "division_id": 1003405, **differing})
    assert one.answer_fields() != other.answer_fields()
    assert one.sentence() != other.sentence() or differing.get("proxy_name"), "the sentence says what differs"


def test_tallies_that_differ_in_reconciliation_do_not_agree():
    base = {"division_id": 1, "ayes": 100, "noes": 90, "tally_reconciled": True}
    assert (F.Fact("division_tally", {}, value=base).answer_fields()
            != F.Fact("division_tally", {}, value={**base, "division_id": 2, "tally_reconciled": False}).answer_fields())


# --- 4b. agreement is claimed over every candidate, never the first eight --------------------------------- The Earl of
# Courtown told for the Not-Contents in the ninth of eleven Crime and Policing Bill divisions on 18 March 2026 and voted
# Content in the first eight.


@pytest.fixture
def eleven(scratch, scratch_dsn):
    scratch.execute("TRUNCATE member_id_map, division, division_vote CASCADE")
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (3359, 'The Earl of Courtown', 'Lords')")
    for n in range(1, 12):
        did = 1_000_000 + n
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, number, title, ayes, noes, "
                        "result) VALUES (%s, %s, 'Lords', '2026-03-18', %s, 'Crime and Policing Bill', 200, 100, "
                        "'agreed')", (did, n, n))  # fmt: skip
        lobby = "no" if n == 9 else "aye"
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby, is_teller) "
                        "VALUES (%s, 3359, 'The Earl of Courtown', %s, %s)", (did, lobby, n == 9))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE member_id_map, division, division_vote CASCADE")


def test_agreement_is_claimed_only_when_every_candidate_agrees(eleven):
    from answer import pipeline as P
    from answer.contract import Provenance
    from ingest.runtime import run as run_async

    question = "How did The Earl of Courtown vote in the division on Crime and Policing Bill on 18 March 2026?"
    prov = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                      model_id="none")  # fmt: skip

    async def go():
        conn = await psycopg.AsyncConnection.connect(eleven)
        try:
            return await P.structured_answer(conn, question, RO.route(question), prov)
        finally:
            await conn.close()

    answer, notes = run_async(go())
    assert answer is not None and "give the same answer" not in (answer.rendered or ""), notes
    assert answer.facets.get("clarify"), "the divisions differ for this member, so the reader is asked which"
    assert len(answer.facets["clarify"]) == 11, "every candidate is offered: the 9th is the one he voted against in"


# --- 5. a structured answer is not a debug dump ---------------------------------------------------------- `render`
# joined every key of `value`, and the answer path used it, so a reader was shown "division_id 1822,... tally_reconciled
# True, must_cite official tellers' figures".

INTERNALS = ("division_id", "must_cite", "tally_reconciled", "ayes_listed", "noes_listed", "ayes_named", "noes_named",
             "mnis_id", "office_key")


@pytest.mark.parametrize(("template", "value"), [
    ("division_result", {"division_id": 1822, "house": "Commons", "division_number": 241, "date": "2026-05-14",
                         "result": "negatived", "ayes": 270, "noes": 286}),
    ("division_tally", {"division_id": 1822, "ayes": 11, "noes": 182, "ayes_listed": 13, "noes_listed": 184,
                        "tally_reconciled": True, "must_cite": "official tellers' figures"}),
    ("division_vote", {"division_id": 5, "mnis_id": 4212, "member": "Sarah Hall", "lobby": "aye",
                       "party_at_vote": "Labour", "is_teller": False, "proxy_name": None}),
    ("office_held_at_date", {"office": "Chancellor of the Exchequer", "office_key": "chancellor",
                             "date": "2026-01-15", "mnis_id": 7, "holder": "Rachel Reeves"}),
    ("count", {"house": "Lords", "month": "2026-09-01", "divisions": 17}),
])
def test_no_internal_field_reaches_a_reader(template, value):
    said = F.Fact(template, {}, value=value).sentence()
    for key in INTERNALS:
        assert key not in said, f"{template} leaked {key}: {said}"


def test_the_caveat_is_never_dropped_from_a_sentence():
    fact = F.Fact("division_tally", {}, value={"division_id": 1, "ayes": 77, "noes": 317, "ayes_listed": 80,
                                               "noes_listed": 317, "tally_reconciled": False},
                  caveat="The published name lists do not reconcile with the tellers' count (I-8).")  # fmt: skip
    assert "do not reconcile" in fact.sentence()


def test_the_records_own_verdict_word_survives_translation():
    """"Negatived" is demystified for the reader AND kept, because the gold nuggets are written in it and because it is
    what a reader would find if they looked the division up.
    """
    said = F.Fact("division_result", {}, value={"house": "Commons", "division_number": 3, "date": "2026-05-14",
                                                "result": "negatived", "ayes": 270, "noes": 286}).sentence()
    assert "negatived" in said and "rejected" in said
    assert "270" in said and "286" in said


def test_a_redundant_gloss_is_not_added():
    """"agreed to — "agreed", in the record's own word" is noise; the plain form already contains the term."""
    said = F.Fact("division_result", {}, value={"house": "Lords", "division_number": 1, "date": "2026-09-09",
                                                "result": "agreed", "ayes": 194, "noes": 138}).sentence()
    assert "in the record's own word" not in said
    assert "agreed to" in said


def test_a_vote_recorded_on_both_sides_is_not_resolved_into_one():
    """62 votes corpus-wide put a member in both lobbies. Saying which way they voted would invent a resolution the
    record does not contain.
    """
    said = F.Fact("division_vote", {}, value={"division_id": 5, "mnis_id": 1, "member": "A Member",
                                              "lobby": "conflict", "party_at_vote": "Labour",
                                              "is_teller": False}).sentence()  # fmt: skip
    assert "both sides" in said and "cannot be stated" in said


def test_render_is_unchanged_for_diagnostics():
    """`render` keeps its field-joined form: the scorer and the logs compare fields, and only the reader-facing path
    moved to `sentence`.
    """
    fact = F.Fact("division_tally", {}, value={"ayes": 77, "noes": 317}, caveat="do not reconcile (I-8)")
    assert fact.render() == "ayes 77, noes 317. do not reconcile (I-8)"


# --- 6. the progress line must never hide the harmful count ---------------------------------------------


def test_the_progress_line_always_shows_harmful():
    """It printed `Counter.most_common(3)`, and `harmful` is by design never among the three commonest outcomes. A dev
    run with ten harmful answers read as clean for twenty minutes, and I reported it as clean.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_answers as EA

    rows = [{"outcome": "correct"}] * 50 + [{"outcome": "over_abstention"}] * 30 + \
           [{"outcome": "missing_vital_nugget"}] * 10 + [{"outcome": "harmful"}] * 2  # fmt: skip
    assert "HARMFUL 2" in EA._progress(rows)


def test_the_progress_line_shows_a_zero_rather_than_omitting_it():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_answers as EA

    assert "HARMFUL 0" in EA._progress([{"outcome": "correct"}])


# --- 7. the serving timeout must not mutate the frozen config -------------------------------------------


def test_the_search_config_is_frozen_and_the_override_uses_replace():
    """`SearchConfig` is a frozen dataclass and `config.statement_timeout_ms = n` raises. A whole 223-item run failed
    in 3.1 seconds on this, every question reporting FrozenInstanceError as a retrieval failure.
    """
    import dataclasses

    import pytest as _pytest

    from answer.pipeline import Frozen

    frozen = Frozen.load()
    config = frozen.search_config()
    assert config.__dataclass_params__.frozen, "if this ever becomes mutable, the replace() below is still correct"
    with _pytest.raises(dataclasses.FrozenInstanceError):
        config.statement_timeout_ms = 20_000
    replaced = dataclasses.replace(config, statement_timeout_ms=20_000)
    assert replaced.statement_timeout_ms == 20_000
    assert config.statement_timeout_ms == 5000, "the built object is untouched"
    assert frozen.config["statement_timeout_ms"] == 5000, "the frozen dict is untouched"
    assert frozen.verify_hash(), "and the hash still describes its own contents"


# --- 8. the cross-encoder score warns the reader; it never refuses -------------------------------------- Measured on
# 108 dev items: the fused RRF score has AUC 0.562 as a discriminator of "does this deck hold the answer" — no
# information — because RRF sums 1/(k+rank) across arms and measures arm agreement, not relevance.


def a_card(score: float | None):
    from answer import cards as C

    return C.Card(speech_id="s1", debate_id="d1", score=0.032, rerank_score=score,
                  sentences=[C.Sentence(chunk_id=1, text="The House rose at six o'clock.", matched=True)])  # fmt: skip


def test_a_weak_deck_is_flagged_and_still_answerable():
    """`enough` stays True — the reader gets the answer AND is told the match was weak. It cost seven partial answers
    to remove two near misses, which the claim-level relevance check removes without that cost.
    """
    from answer import generate as G

    found = G.sufficiency(G.C.Deck(cards=[a_card(-4.0)]))
    assert found.enough is True
    assert found.weak_match is True
    assert found.thin is True, "which is what puts the caveat on the answer"


def test_the_weak_match_decline_is_one_switch_and_off():
    from answer import generate as G

    assert G.WEAK_MATCH_DECLINES is False
    G.WEAK_MATCH_DECLINES = True
    try:
        found = G.sufficiency(G.C.Deck(cards=[a_card(-4.0)]))
        assert found.enough is False and found.reason == G.Abstention.INSUFFICIENT_EVIDENCE
        assert found.top_rerank == -4.0, "still recorded, for re-calibration on real traffic"
    finally:
        G.WEAK_MATCH_DECLINES = False


def test_a_strong_deck_is_not_flagged_as_weak():
    from answer import generate as G

    found = G.sufficiency(G.C.Deck(cards=[a_card(6.0)]))
    assert found.weak_match is False


def test_the_threshold_is_the_measured_one_and_is_negative():
    """Negative on purpose: it fires when the cross-encoder scores the deck's BEST passage as non-relevant, not merely
    as mediocre. At -2.0 it flags 9 of 33 weak decks and 11 of 75 good ones.
    """
    from answer import generate as G

    assert G.WEAK_RERANK_SCORE == -2.0


def test_the_score_is_recorded_even_when_it_changes_nothing():
    """So the threshold can be re-calibrated from real traffic rather than from the 108 dev items that set it."""
    from answer import generate as G

    assert G.sufficiency(G.C.Deck(cards=[a_card(1.25)])).top_rerank == 1.25


def test_a_deck_with_no_rerank_score_is_not_flagged():
    """Reranking can be off, and "unknown" must not read as "weak" — the same three-valued discipline as
    `gold_evidence_retrieved`.
    """
    from answer import generate as G

    found = G.sufficiency(G.C.Deck(cards=[a_card(None)]))
    assert found.weak_match is False and found.top_rerank is None


def test_tallies_that_differ_in_the_names_their_lists_hold_do_not_agree():
    base = {"division_id": 1, "ayes": 100, "noes": 90, "tally_reconciled": False, "ayes_named": 99, "noes_named": 90}
    assert (F.Fact("division_tally", {}, value=base).answer_fields()
            != F.Fact("division_tally", {}, value={**base, "division_id": 2, "ayes_named": 98}).answer_fields())


# --- 3c. a peer by the short form people use ------------------------------------------------------------ Review pass
# 2: "How did Lord Ahmad vote...?" matched nobody, because the list prints "Lord Ahmad of Wimbledon".


@pytest.fixture
def peers(scratch, scratch_dsn):
    scratch.execute("TRUNCATE member_id_map, division, division_vote CASCADE")
    for mnis, name in ((3001, "Lord Ahmad of Wimbledon"), (3002, "Lord Smith of Hindhead"),
                       (3003, "Lord Smith of Finsbury"), (3004, "Baroness Hayter of Kentish Town")):
        scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (%s, %s, 'Lords')", (mnis, name))
    scratch.execute("INSERT INTO division (id, source_id, house, division_date, title, ayes, noes) "
                    "VALUES (1000900, 900, 'Lords', '2026-09-09', 'Health Bill', 2, 1)")  # fmt: skip
    for mnis, printed in ((3001, "Lord Ahmad of Wimbledon"), (3002, "Lord Smith of Hindhead"),
                          (3004, "Baroness Hayter of Kentish Town")):
        scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby) "
                        "VALUES (1000900, %s, %s, 'aye')", (mnis, printed))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE member_id_map, division, division_vote CASCADE")


@pytest.mark.parametrize(("question", "expected"), [
    ("How did Lord Ahmad vote in the division on Health Bill on 9 September 2026?", 3001),
    ("How did Baroness Hayter vote in the division on Health Bill on 9 September 2026?", 3004),
    ("How did Lord Ahmad of Wimbledon vote in the division on Health Bill on 9 September 2026?", 3001),
])  # fmt: skip
def test_a_peer_is_found_by_the_short_form_when_only_one_member_carries_it(peers, question, expected):
    assert resolved_in(peers, question, division_ids=(1000900,))[0] == expected


def test_a_short_form_two_peers_share_is_refused_even_when_only_one_of_them_voted(peers):
    found, why = resolved_in(peers, "How did Lord Smith vote in the division on Health Bill on 9 September 2026?",
                             division_ids=(1000900,))  # fmt: skip
    assert found is None and "2" in why


# --- 3d.


def _structured(dsn, question):
    from answer import pipeline as P
    from answer.contract import Provenance
    from ingest.runtime import run as run_async

    prov = Provenance(route="structured", corpus_version="t", retrieval_config_hash="t", recipe="t", prompt_version="t",
                      model_id="none")  # fmt: skip

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await P.structured_answer(conn, question, RO.route(question), prov)
        finally:
            await conn.close()

    return run_async(go())


def test_a_member_not_in_the_division_s_list_is_told_so_from_the_record(division_list):
    """The vote fact found no row and the route fell through to searching speeches, which cannot hold a division list.
    The list is the record: say what it shows.
    """
    answer, notes = _structured(division_list, f"How did Kemi Badenoch vote in the division on {LIST_TITLE} on 13 May 2020?")
    assert answer is not None, notes
    assert answer.abstention is not None and answer.abstention.value == "not_in_record"
    assert "Kemi Badenoch" in answer.abstention_detail and "not recorded" in answer.abstention_detail
    assert "may have gone unrecorded" not in answer.abstention_detail  # nothing says the lists are short here


def test_an_absence_from_a_list_that_runs_short_of_the_tellers_says_a_vote_may_be_missing(division_list, scratch):
    """No absence is claimed beyond what the record proves. Lists two names short of the figures cannot show that a
    member did not vote, only that no vote is recorded.
    """
    scratch.execute("UPDATE division SET noes = 5, ayes_named = 1, noes_named = 3, tally_reconciled = false "
                    "WHERE id = 784")  # fmt: skip
    answer, _ = _structured(division_list, f"How did Kemi Badenoch vote in the division on {LIST_TITLE} on 13 May 2020?")
    assert "may have gone unrecorded" in answer.abstention_detail and "2" in answer.abstention_detail


# --- 3e. review pass 2's harder probes: a peer's designation, and a clause no division carries ----------


def test_a_peer_named_with_a_designation_is_never_matched_to_a_shorter_peer_s_name(peers, scratch):
    """Found by the absent-member probes: "Lord Moynihan of Chelsea" was answered for "Lord Moynihan", a different peer
    whose name is a whole-word prefix of the one asked about.
    """
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, house) VALUES (924, 'Lord Moynihan', 'Lords'), "
                    "(5007, 'Lord Moynihan of Chelsea', 'Lords')")  # fmt: skip
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, member_name, lobby) "
                    "VALUES (1000900, 924, 'Lord Moynihan', 'no')")  # fmt: skip
    found, why = resolved_in(peers, "How did Lord Moynihan of Chelsea vote in the division on Health Bill on "
                                    "9 September 2026?", division_ids=(1000900,))  # fmt: skip
    assert found == 5007, why  # resolved from the member record, so the record can say he cast no vote here
    assert resolved_in(peers, "How did Lord Moynihan vote in the division on Health Bill on 9 September 2026?",
                       division_ids=(1000900,))[0] == 924  # fmt: skip


def test_a_clause_number_that_contradicts_a_title_makes_it_no_candidate_at_all():
    """Found by MR-02: "New Clause 900" scored every new clause that day above the threshold, all equal, so all four
    were answered as "matching this description". A contradicting number is positive evidence of the wrong division.
    """
    q = "What was the result of the Commons division on Finance (No. 2) Bill Committee: New Clause 900 on 8 May 2024?"
    assert all(RS.title_score(q, t) < RS.MIN_SCORE for t in FINANCE_TITLES)
    assert max(FINANCE_TITLES, key=lambda t: RS.title_score(FINANCE_Q, t)).endswith("New Clause 3")


@pytest.fixture
def finance_day(scratch, scratch_dsn):
    scratch.execute("TRUNCATE division, division_vote CASCADE")
    for n in (1, 2, 3, 4):
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, number, title, ayes, noes, result) "
                        "VALUES (%s, %s, 'Commons', '2024-05-08', %s, %s, 200, 300, 'negatived')",
                        (1809 + n, 1809 + n, 140 + n, f"Finance (No. 2) Bill Committee: New Clause {n}"))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE division, division_vote CASCADE")


def test_a_division_the_record_does_not_hold_is_said_not_to_be_there_with_that_day_s_divisions(finance_day):
    answer, notes = _structured(finance_day, "What was the result of the Commons division on Finance (No. 2) Bill "
                                             "Committee: New Clause 900 on 8 May 2024?")  # fmt: skip
    assert answer is not None, notes
    assert not answer.structured_facts, "no division's answer is given for one the record does not hold"
    assert answer.abstention is not None and answer.abstention.value == "not_in_record"
    assert "New Clause 900" in answer.abstention_detail and "New Clause 3" in answer.abstention_detail


@pytest.fixture
def mixed_day(scratch, scratch_dsn):
    scratch.execute("TRUNCATE division, division_vote CASCADE")
    rows = [(785, 785, "Commons", "2020-05-13", 43, "Agriculture Bill: Report Stage: New Clause 2 (Neil Parish)", None),
            (787, 787, "Commons", "2020-05-13", 45, "Agriculture Bill: Third Reading", None),
            (1000801, 801, "Lords", "2026-09-09", 1, "Financial Services and Markets Bill [HL]",
             "Baroness Vere of Norbiton moved amendment 10."),
            (1000802, 802, "Lords", "2026-09-09", 2, "Financial Services and Markets Bill [HL]",
             "Lord Sharkey moved amendment 12."),
            (950, 950, "Commons", "2026-03-03", 7, "Health Bill: Third Reading", None)]
    for did, src, house, day, number, title, motion in rows:
        scratch.execute("INSERT INTO division (id, source_id, house, division_date, number, title, motion_text, ayes, "
                        "noes, result) VALUES (%s, %s, %s, %s, %s, %s, %s, 200, 100, 'agreed')",
                        (did, src, house, day, number, title, motion))  # fmt: skip
    yield scratch_dsn
    scratch.execute("TRUNCATE division, division_vote CASCADE")


def candidates(dsn, question, house, when):
    from datetime import date as D_

    from ingest.runtime import run as run_async

    async def go():
        conn = await psycopg.AsyncConnection.connect(dsn)
        try:
            return await RS.division_candidates(conn, question, house=house, when=D_.fromisoformat(when),
                                                number=RS.division_number(question))  # fmt: skip
        finally:
            await conn.close()

    return run_async(go())


def test_a_numbered_clause_no_division_carries_never_falls_back_to_a_stage_without_one(mixed_day):
    """MR-02, the 51 that remained: "New Clause 900 (Neil Parish)" matched "Agriculture Bill: Third Reading" on its
    shared words, because a title with no clause number had nothing to contradict.
    """
    tied, why = candidates(mixed_day, "What was the result of the Commons division on Agriculture Bill: Report Stage: "
                                      "New Clause 900 (Neil Parish) on 13 May 2020?", "Commons", "2020-05-13")
    assert tied == [], why


def test_the_only_division_that_day_is_not_the_answer_to_a_clause_it_does_not_carry(mixed_day):
    tied, why = candidates(mixed_day, "What was the result of the Commons vote on New Clause 5 to the Health Bill on "
                                      "3 March 2026?", "Commons", "2026-03-03")
    assert tied == [], why


def test_a_lords_amendment_number_in_the_motion_tells_two_same_titled_divisions_apart(mixed_day):
    tied, why = candidates(mixed_day, "Did the Lords agree to amendment 12 to the Financial Services and Markets Bill "
                                      "[HL] on 9 September 2026?", "Lords", "2026-09-09")
    assert tied == [1000802], why


def test_a_party_question_finds_a_member_by_the_name_that_day_s_lists_printed(division_list):
    """Found by the party probes: "Which party was Theresa May in on 13 May 2020?" matched nobody, since the member
    record holds her peerage. The question's date names the lists whose printed names apply; 9 of 1,822 party trials
    failed this way (Redwood, Grayling, Sharma, two Archbishops of Canterbury).
    """
    from answer import router as R2
    from ingest.runtime import run as run_async

    q = "Which party was Theresa May in on 13 May 2020?"

    async def go():
        conn = await psycopg.AsyncConnection.connect(division_list)
        try:
            return await RS.parameters(conn, "party_at_date", q, R2.route(q).filters)
        finally:
            await conn.close()

    params, notes = run_async(go())
    assert params.get("mnis_id") == 8, notes
