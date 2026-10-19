"""The guards the mutation suite says were untested: every one of these was a live mutant that survived."""

from __future__ import annotations

import httpx
import psycopg
import pytest
from goldset import audit25, authored, dbgen, judge as judging, llm, review, split, validate
from goldset.schema import (
    CATEGORIES,
    OUTCOME_CREDIT,
    Evidence,
    GoldItem,
    jaccard,
    to_dict,
    validate as validate_schema,
)
from goldset.scoring import Graded, score, wilson_upper
from ingest.runtime import run


def with_conn(dsn, fn):
    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            return await fn(conn)

    return run(go())


def item(**kwargs) -> GoldItem:
    base = dict(
        id="hum0001", question="What did the Minister say about the levy?", category="exact_term",
        criticality="major", expected_behaviour="answer", provenance="passage_llm",
        evidence=(Evidence("s1", "The Government published their response to the consultation."),),
    )  # fmt: skip
    return GoldItem(**(base | kwargs))


# --- schema ---------------------------------------------------------------------------------------------


def test_an_ambiguous_item_may_and_should_carry_evidence():
    """`clarify_ambiguous` is deliberately not in NON_ANSWERING: an ambiguous question usually has several genuinely
    relevant speeches, one per reading, and stripping them would leave nothing to check against.
    """
    ambiguous = item(id="adv0001", category="ambiguous", expected_behaviour="clarify_ambiguous",
                     forbidden_claims=("Answering for one reading as though it were the only one.",))  # fmt: skip
    assert validate_schema({"schema_version": 2, "items": [to_dict(ambiguous)]}) == []


def test_outcome_credit_gives_nothing_for_over_abstention_or_a_missing_vital_nugget():
    assert OUTCOME_CREDIT["over_abstention"] == 0.0
    assert OUTCOME_CREDIT["missing_vital_nugget"] == 0.0
    assert OUTCOME_CREDIT["correct_abstention"] == 1.0


def test_over_abstention_and_a_missing_vital_nugget_earn_no_points():
    assert score([Graded("a", "critical", "over_abstention")])["quality"]["points_earned"] == 0.0
    assert score([Graded("a", "critical", "missing_vital_nugget")])["quality"]["points_earned"] == 0.0


def test_jaccard_divides_by_the_union_not_by_the_larger_side():
    """The two agree whenever one side is a subset of the other, so the difference only shows on sets that genuinely
    diverge -- which is every real question-and-quote pair. Dividing by max(len) inflates the score, and an inflated
    overlap score rejects honest paraphrases while passing copied ones.
    """
    question, quote = "housing planning councils", "councils funding reform levy"
    union = jaccard(question, quote)
    by_larger = 1 / max(len({"housing", "planning", "councils"}), len({"councils", "funding", "reform", "levy"}))
    assert union == pytest.approx(1 / 6)
    assert union < by_larger


def test_the_wilson_bound_tightens_as_the_sample_grows():
    """A bound that ignored n would let 1 error in 10 support the same claim as 30 in 300."""
    assert wilson_upper(1, 10) > wilson_upper(10, 100) > wilson_upper(100, 1000)
    assert wilson_upper(0, 50) > wilson_upper(0, 500)


# --- the overlap threshold -------------------------------------------------------------------------------


def test_the_threshold_comes_from_the_lexical_categories_not_from_the_paraphrases():
    """Measuring it on the paraphrases themselves makes the filter agree with whatever it was handed: a set full of
    copied paraphrases would raise the bar until the copies passed.
    """
    quote = "The Government published their response to the consultation."
    # Exact-term items that happen to share almost nothing: "lexically close" is LOW on this corpus.
    lexical = [
        item(id=f"hum{i:04d}", question=f"Distinct wording number {i} regarding housing?",
             evidence=(Evidence("s1", quote),))  # fmt: skip
        for i in range(12)
    ]
    # Paraphrases that copy heavily.
    copies = [
        item(id=f"adv{i:04d}", category="paraphrase",
             question="The Government published their response to the consultation, yes?",
             evidence=(Evidence("s1", quote),))  # fmt: skip
        for i in range(12)
    ]
    assert validate.choose_jaccard_threshold(lexical + copies) == validate.choose_jaccard_threshold(lexical)
    assert validate.choose_jaccard_threshold(lexical + copies) < validate.DEFAULT_JACCARD_MAX


def test_the_measured_threshold_is_never_looser_than_the_default():
    """A corpus where everything overlaps must not raise the bar for everyone."""
    saturated = [
        item(id=f"hum{i:04d}", question="The Government published their response to the consultation?",
             evidence=(Evidence("s1", "The Government published their response to the consultation."),))  # fmt: skip
        for i in range(20)
    ]
    assert validate.choose_jaccard_threshold(saturated) <= validate.DEFAULT_JACCARD_MAX


# --- absence anchors --------------------------------------------------------------------------------------


@pytest.fixture
def absence_corpus(scratch, scratch_dsn):
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
        "VALUES ('W0', 'v2', '2026-09-01', '2026-09-17', 'dev')"
    )
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
                    "VALUES ('d1','2026-09-02','Commons','H','Commons Chamber','other','W0','test')")  # fmt: skip
    scratch.execute("INSERT INTO speech (id, debate_id, seq, text, window_id) VALUES ('s1','d1',0,'x','W0')")
    # 200 chunks so that 0.5% of them is 1: a lexeme with df 1 is rare, df 50 is structural.
    for i in range(200):
        scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body) "
                        "VALUES ('s1','d1',%s,%s,%s,'p')", (i, i, f"The minister discussed unicorns number {i}."))  # fmt: skip
    scratch.execute("INSERT INTO lexeme_df (lexeme, df) VALUES ('minist', 50), ('unicorn', 1), ('discuss', 50)")
    return scratch_dsn


def test_a_structural_word_is_not_accepted_as_an_absence_anchor(absence_corpus):
    """With the ratio wide open, 'minister' anchors the search and every abstain item looks answerable."""
    anchors = with_conn(absence_corpus, lambda c: validate._anchors(c, "What did the minister discuss?", 200))
    assert anchors == []


def test_a_rare_word_is_accepted_as_an_anchor(absence_corpus):
    anchors = with_conn(absence_corpus, lambda c: validate._anchors(c, "What about unicorns?", 200))
    assert anchors == ["unicorn"]


def test_one_anchor_is_not_enough_to_claim_absence(absence_corpus):
    """'Swanwick' appears, but "which OTHER company runs Swanwick" is still unanswerable."""
    near = item(id="adv0001", category="near_miss_abstention", expected_behaviour="abstain_not_in_record",
                evidence=(), window="", question="What about unicorns?")  # fmt: skip
    findings, candidates = with_conn(absence_corpus, lambda c: validate.check_absence(c, [near]))
    assert "only 1 rare term" in findings[0].message
    assert candidates == {}


def test_a_near_miss_with_enough_anchors_records_its_counter_evidence(absence_corpus):
    with psycopg.connect(absence_corpus, autocommit=True) as conn:
        conn.execute("INSERT INTO lexeme_df (lexeme, df) VALUES ('number', 1)")
    near = item(id="adv0002", category="near_miss_abstention", expected_behaviour="abstain_not_in_record",
                evidence=(), window="", question="What about unicorns number seven?")  # fmt: skip
    _, candidates = with_conn(absence_corpus, lambda c: validate.check_absence(c, [near]))
    assert candidates["adv0002"], "the closest passages must reach the owner, not be thrown away"


# --- the authored seeds -------------------------------------------------------------------------------------


@pytest.fixture
def corpus(scratch, scratch_dsn):
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) VALUES "
        "('W0','v2','2026-09-01','2026-09-17','dev'), ('W3','v2','2020-03-01','2020-05-31','heldout'), "
        "('W4','v2','2026-01-01','2026-03-31','dev_test')"
    )
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, party) VALUES (7,'A Member','Labour'), "
                    "(8,'B Member','Conservative')")  # fmt: skip
    for did, date, window in (("d1", "2026-09-02", "W0"), ("d3", "2020-03-02", "W3"), ("d4", "2026-01-05", "W4")):
        scratch.execute("INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
                        "VALUES (%s,%s,'Commons','Local Government Reorganisation','Commons Chamber','other',%s,'test')",
                        (did, date, window))  # fmt: skip
        sentence = "This debate matters to my constituents and I will explain exactly why that is so."
        scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, window_id) "
                        "VALUES (%s,%s,0,'A Member',7,%s,%s)", (f"s{did}", did, sentence * 9, window))  # fmt: skip
        scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body) "
                        "VALUES (%s,%s,0,0,%s,%s)", (f"s{did}", did, sentence, sentence))  # fmt: skip
    return scratch_dsn


ANCHOR = "matters to my constituents"


def test_an_authored_item_from_the_heldout_window_is_born_heldout(corpus):
    seed = [dict(q="What did the member say about constituents in 2020?", cat="exact_term", crit="major",
                 ev=[("sd3", ANCHOR)], nug=[("They explained.", True)])]  # fmt: skip
    items, _ = with_conn(corpus, lambda c: authored.build(c, seed))
    assert items[0].window == "W3" and items[0].split == "heldout"


def test_a_cross_window_item_is_tagged_with_no_window_rather_than_the_first_speech(corpus):
    """Tagging it W0 would make a two-window question look like a W0 question, and the per-window breakdown would
    quietly attribute it to the wrong place.
    """
    seed = [dict(q="Has reorganisation come up in more than one period of the record?", cat="cross_window",
                 crit="major", ev=[("sd1", ANCHOR), ("sd4", ANCHOR)], nug=[("Yes.", True)])]  # fmt: skip
    items, _ = with_conn(corpus, lambda c: authored.build(c, seed))
    assert items[0].window == "" and items[0].house == ""
    assert items[0].subcategory == "spans_W0_W4"


def test_an_authored_items_id_prefix_distinguishes_the_two_halves(corpus):
    seed = [
        dict(q="Did the member mention constituents at all?", cat="exact_term", crit="major",
             ev=[("sd1", ANCHOR)], nug=[("Yes.", True)]),  # fmt: skip
        dict(q="Why did the member say constituents were irrelevant here?", cat="false_premise", crit="critical",
             ev=[("sd1", ANCHOR)], nug=[("They did not.", True)], forbid=["That they said it."]),  # fmt: skip
    ]
    items, _ = with_conn(corpus, lambda c: authored.build(c, seed))
    assert [i.id[:3] for i in items] == ["hum", "adv"]


# --- generation ---------------------------------------------------------------------------------------------


@pytest.fixture
def gen_corpus(corpus):
    with psycopg.connect(corpus, autocommit=True) as conn:
        # Two divisions in one month, one of them with no linked debate: the count must cover both.
        conn.execute(
            "INSERT INTO division (id, source_id, house, division_date, number, title, ayes, noes, result, "
            "debate_id, ayes_listed, noes_listed, tally_reconciled) VALUES "
            "(1,1,'Commons','2026-09-02',1,'Local Government Bill',300,200,'agreed','d1',301,200,false),"
            "(2,2,'Commons','2026-09-09',2,'Another Bill',150,250,'negatived',NULL,150,250,true)"
        )
        # One office with two holders on the same day: it must never be asked as though it had one answer.
        conn.execute(
            "INSERT INTO member_office (mnis_id, office, office_key, start_date, end_date, source, evidence) VALUES "
            "(7,'Shared Office','shared office','2026-09-01','2026-09-30','mnis_biography',1),"
            "(8,'Shared Office','shared office','2026-09-01','2026-09-30','mnis_biography',1),"
            "(7,'Sole Office','sole office','2026-09-01','2026-09-30','mnis_biography',1)"
        )
        # One bill whose "Second reading" exists as two stage rows: the question would be ambiguous.
        conn.execute("INSERT INTO bill (id, title) VALUES (1,'Ambiguous Bill'), (2,'Clear Bill')")
        conn.execute(
            "INSERT INTO bill_stage (id, bill_id, stage_name, house) VALUES "
            "(1,1,'Second reading','Commons'),(2,1,'Second reading','Commons'),(3,2,'Second reading','Commons')"
        )
        conn.execute(
            "INSERT INTO bill_stage_sitting (id, bill_stage_id, bill_id, date, house) VALUES "
            "(1,1,1,'2026-09-02','Commons'),(2,2,1,'2026-09-09','Commons'),(3,3,2,'2026-09-03','Commons')"
        )
    return corpus


def test_a_tally_question_forbids_the_name_list_count(gen_corpus):
    """12 of 15 Commons divisions do not reconcile. A system that counts names instead of quoting the tellers gives a
    subtly wrong number, and nothing but this forbidden claim catches it.
    """
    items = with_conn(gen_corpus, lambda c: dbgen.generate(c, seed=3))
    tallies = [i for i in items if i.category == "division_tally"]
    assert tallies and all(i.forbidden_claims for i in tallies)
    unreconciled = [i for i in tallies if i.subcategory == "unreconciled"]
    assert any("301" in f for i in unreconciled for f in i.forbidden_claims)


def test_the_monthly_division_count_covers_divisions_with_no_linked_debate(gen_corpus):
    items = with_conn(gen_corpus, lambda c: dbgen.generate(c, seed=3))
    monthly = [i for i in items if i.subcategory == "divisions_per_month"]
    assert monthly and all(i.expected_structured_facts["divisions"] == 2 for i in monthly)


def test_an_office_with_two_holders_is_never_asked_as_though_it_had_one_answer(gen_corpus):
    items = with_conn(gen_corpus, lambda c: dbgen.generate(c, seed=3))
    offices = {i.expected_structured_facts["office_key"] for i in items if i.category == "office_held_at_date"}
    assert "shared office" not in offices
    assert "sole office" in offices


def test_a_bill_stage_that_sat_under_two_stage_rows_is_not_asked_about(gen_corpus):
    """"When did the Bill have its Second Reading in the Lords" does not name a bill_stage row, so a bill carried over
    between sessions has two answers and the question is not exact. The precondition is asserted first: without it,
    a fixture that failed to load would make this test pass vacuously.
    """
    with psycopg.connect(gen_corpus) as conn:
        assert conn.execute("SELECT count(*) FROM bill_stage_sitting").fetchone()[0] == 3, "fixture not loaded"

    items = with_conn(gen_corpus, lambda c: dbgen.generate(c, seed=3))
    stages = [i for i in items if i.category == "bill_stage_date"]
    bills = {i.expected_structured_facts["bill"] for i in stages}
    assert "Clear Bill" in bills, "the unambiguous bill should be asked about"
    assert "Ambiguous Bill" not in bills

    # And the contract directly: every generated item names exactly one stage row.
    with psycopg.connect(gen_corpus) as conn:
        for one in stages:
            facts = one.expected_structured_facts
            rows = conn.execute(
                "SELECT count(DISTINCT s.id) FROM bill_stage_sitting t JOIN bill_stage s ON s.id = t.bill_stage_id "
                "WHERE t.bill_id = %s AND s.stage_name = %s AND s.house IS NOT DISTINCT FROM %s",
                (facts["bill_id"], facts["stage"], facts["house"]),
            ).fetchone()[0]
            assert rows == 1, f"{one.id} names {rows} stage rows, so its date is not the only answer"


def test_who_opened_the_debate_means_the_first_speech_not_the_first_usable_one(gen_corpus):
    """Filtering inside the DISTINCT ON would redefine "opened" as "opened, ignoring what I could not use", and produce
    a confidently wrong gold answer.
    """
    with psycopg.connect(gen_corpus, autocommit=True) as conn:
        text = "A short opener."
        conn.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, window_id) "
                     "VALUES ('sfirst','d1',-1,'B Member',8,%s,'W0')", (text,))  # fmt: skip
        conn.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body) "
                     "VALUES ('sfirst','d1',99,0,%s,%s)", (text, text))  # fmt: skip
    items = with_conn(gen_corpus, lambda c: dbgen.generate(c, seed=3))
    # 'sfirst' is now the first speech but its only sentence is below MIN_EVIDENCE_CHARS, so the debate must be DROPPED
    # rather than attributed to the second speaker.
    assert not [i for i in items if i.category == "who_spoke" and i.expected_structured_facts["debate_id"] == "d1"]


# --- the split -------------------------------------------------------------------------------------------------


def test_a_stratum_of_one_reaches_both_sides_across_seeds():
    """The case that separates alternating from slicing. Slicing sends a lone item to the same side every time (index 0
    is always below the cut), so a category with one item per stratum -- which is most of the adversarial categories
    once window and house are crossed in -- ends up entirely on one side, and becomes unmeasurable on the other.
    """
    landed = set()
    for seed in range(20):
        one = [item(category="loaded", expected_behaviour="neutralise_loaded_framing",
                    window="W1", house="Commons")]  # fmt: skip
        split.assign(one, seed=seed)
        landed.add(one[0].split)
    assert landed == {"dev", "test"}, f"a stratum of one only ever landed in {landed}"


def test_a_stratum_of_three_is_never_all_on_one_side():
    for seed in range(12):
        items = [item(id=f"hum{i:04d}", question=f"Loaded question {i}?", category="loaded",
                      expected_behaviour="neutralise_loaded_framing", window="W1", house="Commons")  # fmt: skip
                 for i in range(3)]
        split.assign(items, seed=seed)
        assert len({i.split for i in items}) == 2


def test_a_category_entirely_on_one_side_is_reported():
    items = [item(id=f"hum{i:04d}", question=f"Question {i} here?") for i in range(4)]
    split.assign(items)
    for one in items:
        one.split = "dev"
    assert split.report(items)["single_sided_categories"] == ["exact_term"]


# --- judging --------------------------------------------------------------------------------------------------


def test_an_unparsable_grade_is_left_unjudged_rather_than_scored_zero():
    """A default of 0 is not neutral: it is the claim that a retrieved passage does not help, which is exactly what the
    recall numbers are built on.
    """

    class Mute(llm.Client):
        async def batch(self, prompts, *, system="", max_tokens=64):
            return [None] * len(prompts)

    pool = judging.Pool(pairs=[judging.Pair("i1", "s1", "Q?", "P")])
    grades = run(judging.judge(pool, Mute(key="k", base="https://example.invalid", model="m")))
    assert grades == {}


def test_a_parsed_grade_is_kept():
    class Three(llm.Client):
        async def batch(self, prompts, *, system="", max_tokens=64):
            return ["3"] * len(prompts)

    pool = judging.Pool(pairs=[judging.Pair("i1", "s1", "Q?", "P")])
    assert run(judging.judge(pool, Three(key="k", base="https://example.invalid", model="m"))) == {"i1": {"s1": 3}}


def test_a_grade_is_not_parsed_out_of_an_unrelated_digit():
    assert llm.parse_grade("I award this a 7 out of 10") is None


def test_cohen_kappa_corrects_for_chance_rather_than_reporting_raw_agreement():
    """Two graders who both say 3 nine times in ten agree 90% of the time by accident alone."""
    a = {("q", f"s{i}"): (3 if i < 90 else 0) for i in range(100)}
    b = {("q", f"s{i}"): (3 if i < 90 else 3) for i in range(100)}
    result = judging.cohen_kappa(a, b)
    assert result["agreement"] == 0.9
    assert result["kappa"] < 0.5, "raw agreement would have reported 0.9 and passed the gate"


def test_the_kappa_gate_is_the_briefs_threshold():
    middling = {("q", f"s{i}"): i % 4 for i in range(100)}
    shifted = {k: (v if i % 3 else (v + 1) % 4) for i, (k, v) in enumerate(middling.items())}
    result = judging.cohen_kappa(middling, shifted)
    assert 0.2 < result["kappa"] < 0.8
    assert result["passes_gate"] == (result["kappa"] >= 0.6)


class _Stub:
    """A searcher that returns fixed hits, or raises, without touching a database or a GPU."""

    def __init__(self, hits=(), fail=False):
        self._hits, self._fail = hits, fail

    async def search(self, question, top_k=30, config=None):
        if self._fail:
            raise RuntimeError("the GPU service went away")
        return type("R", (), {"hits": [type("H", (), {"speech_id": s})() for s in self._hits]})()


def test_the_known_relevant_speeches_are_pooled_even_if_nothing_retrieves_them(corpus):
    """The twelve nothing found are the ones that matter most, and they are precisely what a retrieval-only pool would
    lose.
    """
    one = item(evidence=(Evidence("sd1", "a quote long enough to pass validation here"),))
    variant = type("V", (), {"name": "dense", "config": None})()
    pool = with_conn(corpus, lambda c: judging.build_pool(c, _Stub(hits=["sd4"]), [one], [variant]))
    assert {p.speech_id for p in pool.pairs} == {"sd1", "sd4"}


def test_a_failed_search_is_recorded_rather_than_silently_becoming_an_empty_ranking(corpus):
    one = item(evidence=(Evidence("sd1", "a quote long enough to pass validation here"),))
    variant = type("V", (), {"name": "dense", "config": None})()
    pool = with_conn(corpus, lambda c: judging.build_pool(c, _Stub(fail=True), [one], [variant]))
    assert pool.retrieved[one.id]["_errors"]


# --- the LLM client ----------------------------------------------------------------------------------------------


def test_the_error_never_carries_the_response_body(monkeypatch):
    """A provider's error body can echo the request, and these prompts carry corpus text and the model name. Patching
    the transport rather than the method means the real `complete` is what is tested.
    """
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": "bad prompt: SECRET PASSAGE TEXT"})
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))

    client = llm.Client(key="k", base="https://example.invalid", model="m")
    with pytest.raises(llm.LlmError) as caught:
        run(client.complete("a prompt containing a speech"))
    assert "400" in str(caught.value)
    assert "SECRET PASSAGE TEXT" not in str(caught.value)


def test_a_successful_completion_returns_the_message_content(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    client = llm.Client(key="k", base="https://example.invalid", model="m")
    assert run(client.complete("x")) == "2"


def test_an_unexpected_response_shape_is_an_llm_error_not_a_key_error(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"unexpected": True}))
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    client = llm.Client(key="k", base="https://example.invalid", model="m")
    with pytest.raises(llm.LlmError, match="unexpected response shape"):
        run(client.complete("x"))


def test_one_failed_prompt_does_not_lose_the_rest_of_the_batch():
    class Flaky(llm.Client):
        async def complete(self, prompt, *, system="", max_tokens=64, temperature=0.0):
            if prompt == "bad":
                raise RuntimeError("rate limited")
            return "2"

    replies = run(Flaky(key="k", base="https://example.invalid", model="m").batch(["a", "bad", "c"]))
    assert replies == ["2", None, "2"]


# --- the review pack ----------------------------------------------------------------------------------------------


def test_the_queue_is_ordered_for_the_owners_time_not_by_id(corpus):
    """Dev-split critical items come before held-out ones: stopping half way still leaves the half that every adoption
    decision is made on.
    """
    items = [
        item(id="hum9001", criticality="critical", split="heldout", window="W3",
             question="A held-out critical question?"),  # fmt: skip
        item(id="hum9002", criticality="critical", split="dev", question="A dev critical question?"),
    ]
    pack = with_conn(corpus, lambda c: review.build_pack(c, items, spot_check=0))
    assert [e["id"] for e in pack["queue"]] == ["hum9002", "hum9001"]


def test_the_estimate_includes_the_calibration_sample(corpus):
    pool = judging.Pool(pairs=[judging.Pair(f"i{i}", f"s{i}", "Q?", "P") for i in range(40)])
    with_pool = with_conn(corpus, lambda c: review.build_pack(c, [item(criticality="critical")], pool=pool,
                                                             kappa_sample=40, spot_check=0))  # fmt: skip
    without = with_conn(corpus, lambda c: review.build_pack(c, [item(criticality="critical")], spot_check=0))
    assert with_pool["plan"]["kappa"] == 40
    assert with_pool["plan"]["estimated_minutes"] > without["plan"]["estimated_minutes"]


def test_the_review_pack_shows_the_passage_around_the_quote(corpus):
    """A quote cannot be checked without its context: "He will know..." reads as a Minister speaking until the
    surrounding paragraph shows it is addressed TO one.
    """
    one = item(evidence=(Evidence("sd1", "matters to my constituents and I will explain"),))
    pack = with_conn(corpus, lambda c: review.build_pack(c, [one], spot_check=0))
    evidence = pack["items"][one.id]["evidence"][0]
    assert len(evidence["context"]) > len(evidence["quote"])
    assert evidence["quote"] in evidence["context"]


# --- the audit of the original 25 ------------------------------------------------------------------------------------


@pytest.fixture
def v1_gold(tmp_path, corpus):
    import json

    path = tmp_path / "gold_eval.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "queries": [
            {"id": "g001", "question": "What did the member say about constituents?", "category": "exact_term",
             "status": "provisional", "expected_abstain": False,
             "relevant": [{"speech_id": "sd1", "evidence_quote": "matters to my constituents"}], "notes": ""},
            {"id": "g002", "question": "What did the member claim about the money?", "category": "exact_term",
             "status": "provisional", "expected_abstain": False,
             "relevant": [{"speech_id": "sd1", "evidence_quote": "a phrase that is not in the speech"}], "notes": ""},
            {"id": "g023", "question": "What was said about an Antarctic research station?", "category": "abstain",
             "status": "provisional", "expected_abstain": True, "relevant": [], "notes": ""},
        ],
    }), encoding="utf-8")  # fmt: skip
    return path, corpus


def test_the_carried_over_items_are_pinned_to_dev_permanently(v1_gold):
    path, dsn = v1_gold
    items, _ = with_conn(dsn, lambda c: audit25.migrate(c, path))
    assert {i.split for i in items} == {"dev"}
    assert {i.subcategory for i in items} == {"carried_from_v1"}
    # And the splitter must not move them afterwards.
    split.assign(items)
    assert {i.split for i in items} == {"dev"}


def test_a_v1_quote_that_is_not_verbatim_is_reported(v1_gold):
    path, dsn = v1_gold
    _, findings = with_conn(dsn, lambda c: audit25.audit(c, path))
    assert any(f.item_id == "g002" and f.kind == "quote" for f in findings)


def test_the_briefs_doubts_are_carried_into_the_items_notes(v1_gold):
    path, dsn = v1_gold
    items, _ = with_conn(dsn, lambda c: audit25.migrate(c, path))
    by_id = {i.id: i for i in items}
    assert "save children" in by_id["old0001"].notes
    assert "[brief]" in by_id["old0002"].notes


def test_a_carried_over_abstain_item_has_no_nuggets_and_forbids_answering(v1_gold):
    path, dsn = v1_gold
    by_id = {i.id: i for i in with_conn(dsn, lambda c: audit25.migrate(c, path))[0]}
    abstaining = by_id["old0023"]
    assert abstaining.nuggets == ()
    assert abstaining.forbidden_claims and abstaining.expected_behaviour == "abstain_not_in_record"
    assert validate_schema({"schema_version": 2, "items": [to_dict(abstaining)]}) == []


def test_every_category_the_authored_seeds_use_is_in_the_vocabulary():
    from goldset.adversarial_items import ADVERSARIAL
    from goldset.human_items import HUMAN

    assert {s["cat"] for s in HUMAN + ADVERSARIAL} <= set(CATEGORIES)
