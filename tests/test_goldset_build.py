"""Generation, the authored seeds, the split, judging and the review pack."""

from __future__ import annotations

import json

import psycopg
import pytest
from goldset import authored, dbgen, judge as judging, llm, review, split
from goldset.schema import Evidence, GoldItem
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


def many(n: int, **kwargs) -> list[GoldItem]:
    return [item(id=f"hum{i:04d}", question=f"Question number {i} about housing policy?", **kwargs) for i in range(n)]


# --- the split ---------------------------------------------------------------------------------------


def test_w3_items_are_always_heldout_whatever_they_arrived_as():
    items = many(10, window="W3") + many(10)
    for i in items[:10]:
        i.split = "dev"  # as if something had assigned it wrongly upstream
    split.assign(items)
    assert all(i.split == "heldout" for i in items if i.window == "W3")


def test_the_original_25_stay_dev_for_good():
    """Every tuned parameter in the system was chosen on them, so a number computed on them is not a test result
    however the split file reads.
    """
    items = many(20, subcategory="carried_from_v1") + many(20)
    split.assign(items)
    assert all(i.split == "dev" for i in items if i.subcategory == "carried_from_v1")


def test_everything_else_splits_close_to_evenly():
    items = [item(id=f"hum{i:04d}", question=f"Question {i} about transport?", window="W1") for i in range(200)]
    split.assign(items)
    dev = sum(1 for i in items if i.split == "dev")
    assert 90 <= dev <= 110


def test_a_small_category_does_not_land_entirely_on_one_side():
    """Six loaded questions all in test would make that category unmeasurable on dev, which is where every adoption
    decision is made.
    """
    items = []
    for category in ("loaded", "misattribution", "ambiguous"):
        for i in range(6):
            items.append(item(id=f"adv{len(items):04d}", question=f"{category} question {i} here?",
                              category=category, expected_behaviour="answer"))  # fmt: skip
    split.assign(items)
    assert split.report(items)["single_sided_categories"] == []


def test_the_split_is_deterministic_for_a_seed_and_changes_with_it():
    a, b, c = many(60), many(60), many(60)
    split.assign(a, seed=1)
    split.assign(b, seed=1)
    split.assign(c, seed=2)
    assert [i.split for i in a] == [i.split for i in b]
    assert [i.split for i in a] != [i.split for i in c]


def test_the_split_hash_identifies_the_assignment_not_the_wording():
    items = many(40)
    split.assign(items, seed=1)
    first = split.split_hash(items)
    items[0].question = "Completely different wording for this one?"
    assert split.split_hash(items) == first
    items[0].split = "test" if items[0].split == "dev" else "dev"
    assert split.split_hash(items) != first


def test_the_split_report_breaks_down_by_window_house_and_category():
    items = many(20, window="W1", house="Commons") + many(20, window="W2", house="Lords")
    for i, one in enumerate(items):
        one.id = f"hum{i:04d}"
        one.question = f"Question {i} about something?"
    split.assign(items)
    report = split.report(items)
    assert set(report["by_window"]) == {"W1", "W2"}
    assert set(report["by_house"]) == {"Commons", "Lords"}
    assert report["dev"] + report["test"] == 40


# --- generation against a scratch corpus ---------------------------------------------------------------


@pytest.fixture
def corpus(scratch, scratch_dsn):
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
        "VALUES ('W0', 'v2', '2026-09-01', '2026-09-17', 'dev'), ('W3', 'v2', '2020-03-01', '2020-05-31', 'heldout')"
    )
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, party) VALUES (7, 'A Member', 'Labour')")
    for did, date, window in (("d1", "2026-09-02", "W0"), ("d3", "2020-03-02", "W3")):
        scratch.execute(
            "INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
            "VALUES (%s, %s, 'Commons', 'Local Government Reorganisation', 'Commons Chamber', 'other', %s, 'test')",
            (did, date, window),
        )
        text = "This debate matters to my constituents and I will explain exactly why that is so."
        scratch.execute(
            "INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, window_id) "
            "VALUES (%s, %s, 0, 'A Member', 7, %s, %s)",
            (f"s{did}", did, text * 9, window),
        )
        scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body) "
                        "VALUES (%s, %s, 0, 0, %s, %s)", (f"s{did}", did, text, text))  # fmt: skip
        scratch.execute(
            "INSERT INTO division (id, source_id, house, division_date, number, title, ayes, noes, result, "
            "debate_id, ayes_listed, noes_listed, tally_reconciled) "
            "VALUES (%s, %s, 'Commons', %s, 1, 'Local Government Bill', 300, 200, 'agreed', %s, 300, 200, true)",
            (1 if did == "d1" else 2, 1 if did == "d1" else 2, date, did),
        )
    scratch.execute("INSERT INTO division_vote (division_id, mnis_id, lobby, member_name, party) "
                    "VALUES (1, 7, 'aye', 'A Member', 'Labour')")  # fmt: skip
    return scratch_dsn


def test_generation_is_deterministic_for_a_seed(corpus):
    first = with_conn(corpus, lambda c: dbgen.generate(c, seed=5))
    second = with_conn(corpus, lambda c: dbgen.generate(c, seed=5))
    assert [(i.id, i.question) for i in first] == [(i.id, i.question) for i in second]


def test_generated_items_from_the_heldout_window_are_born_heldout(corpus):
    items = with_conn(corpus, lambda c: dbgen.generate(c, seed=5))
    assert items, "the fixture should produce at least one item"
    assert all(i.split == "heldout" for i in items if i.window == "W3")


def test_generation_never_writes_to_the_database(corpus):
    """Read-only is a property worth testing, not assuming: a generator that writes would corrupt the corpus it is
    measuring, and the corruption would look like data.
    """
    before = {}
    with psycopg.connect(corpus) as conn:
        for table in ("speech", "debate", "division", "chunk"):
            before[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    with_conn(corpus, lambda c: dbgen.generate(c, seed=5))
    with psycopg.connect(corpus) as conn:
        for table, n in before.items():
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == n


def test_a_division_result_item_carries_the_official_tally_and_forbids_the_opposite(corpus):
    items = with_conn(corpus, lambda c: dbgen.generate(c, seed=5))
    results = [i for i in items if i.category == "division_result"]
    assert results
    one = results[0]
    assert one.criticality == "critical"
    assert one.expected_structured_facts["ayes"] == 300
    assert any("negatived" in f for f in one.forbidden_claims)


def test_stratify_spreads_across_groups_rather_than_taking_the_biggest(corpus):
    import random

    rows = [{"window_id": "W0"}] * 100 + [{"window_id": "W3"}] * 3
    taken = dbgen._stratify(rows, 6, random.Random(1))
    assert sum(1 for r in taken if r["window_id"] == "W3") == 3


# --- the authored seeds --------------------------------------------------------------------------------


def test_an_anchor_is_resolved_to_the_verbatim_sentence(corpus):
    seed = [dict(q="What did the member say about constituents?", cat="exact_term", crit="major",
                 ev=[("sd1", "matters to my constituents")], nug=[("They explained why.", True)])]  # fmt: skip
    items, unresolved = with_conn(corpus, lambda c: authored.build(c, seed))
    assert unresolved == []
    assert items[0].evidence[0].quote.startswith("This debate matters")
    assert items[0].window == "W0"  # read off the speech, never asserted by the seed


def test_an_anchor_that_is_not_there_is_reported_and_costs_only_its_own_evidence(corpus):
    seed = [dict(q="A question about something absent from the record?", cat="exact_term", crit="major",
                 ev=[("sd1", "a phrase that does not occur")], nug=[("x", True)])]  # fmt: skip
    items, unresolved = with_conn(corpus, lambda c: authored.build(c, seed))
    assert len(items) == 1 and items[0].evidence == ()
    assert unresolved[0].reason == "anchor not found in the speech"


def test_an_anchor_matching_across_a_sentence_boundary_still_resolves(scratch, scratch_dsn):
    scratch.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
                    "VALUES ('W0', 'test', '2026-09-01', '2026-09-17', 'dev') ON CONFLICT DO NOTHING")
    scratch.execute("INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
                    "VALUES ('d1', '2026-09-02', 'Commons', 'H', 'Commons Chamber', 'other', 'W0', 'test')")  # fmt: skip
    scratch.execute("INSERT INTO speech (id, debate_id, seq, text, window_id) "
                    "VALUES ('s1', 'd1', 0, 'First part. Second part.', 'W0')")
    scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body) "
                    "VALUES ('s1','d1',0,0,'First part.','x'), ('s1','d1',1,1,'Second part.','x')")  # fmt: skip
    seed = [dict(q="Does this anchor straddle two sentences?", cat="exact_term", crit="major",
                 ev=[("s1", "First part. Second")], nug=[("yes", True)])]  # fmt: skip
    items, unresolved = with_conn(scratch_dsn, lambda c: authored.build(c, seed))
    assert unresolved == [] and items[0].evidence[0].quote == "First part. Second part."


def test_evidence_mixing_the_heldout_window_with_another_is_refused(corpus):
    seed = [dict(q="A question drawing on two different windows at once?", cat="cross_window", crit="major",
                 ev=[("sd1", "matters to my constituents"), ("sd3", "matters to my constituents")],
                 nug=[("x", True)])]  # fmt: skip
    with pytest.raises(authored.SeedError, match="held-out window"):
        with_conn(corpus, lambda c: authored.build(c, seed))


def test_a_seed_in_a_category_with_several_behaviours_must_say_which(corpus):
    seed = [dict(q="An ambiguous multi-part question with no behaviour given?", cat="multi_part",
                 crit="major", ev=[], nug=[("x", True)])]  # fmt: skip
    with pytest.raises(authored.SeedError, match="needs an explicit behaviour"):
        with_conn(corpus, lambda c: authored.build(c, seed))


def test_the_id_prefix_follows_the_behaviour_not_the_category(corpus):
    """`quote_verification` is a human category that turns adversarial when the quote is flipped."""
    seed = [
        dict(q="Did the member say their constituents mattered?", cat="quote_verification", crit="major",
             behaviour="answer", ev=[("sd1", "matters to my constituents")], nug=[("yes", True)]),  # fmt: skip
        dict(q="Did the member say constituents were irrelevant to this debate?", cat="quote_verification",
             crit="critical", behaviour="correct_false_premise", ev=[("sd1", "matters to my constituents")],
             nug=[("no", True)], forbid=["That they said it was irrelevant."]),  # fmt: skip
    ]
    items, _ = with_conn(corpus, lambda c: authored.build(c, seed))
    assert items[0].id.startswith("hum") and items[1].id.startswith("adv")


def test_the_shipped_seeds_all_carry_what_their_category_requires():
    """A structural check that needs no database: every seed names a known category, and every seed whose category
    allows more than one behaviour says which one it means.
    """
    from goldset.adversarial_items import ADVERSARIAL
    from goldset.human_items import HUMAN
    from goldset.schema import CATEGORIES, CRITICALITY_POINTS

    for seed in HUMAN + ADVERSARIAL:
        assert seed["cat"] in CATEGORIES, seed["q"]
        assert seed["crit"] in CRITICALITY_POINTS, seed["q"]
        assert seed.get("behaviour") or CATEGORIES[seed["cat"]], seed["q"]
        assert seed["nug"], f"an item with no nuggets cannot be scored: {seed['q']}"


def test_no_two_shipped_seeds_ask_the_same_question():
    from goldset.adversarial_items import ADVERSARIAL
    from goldset.human_items import HUMAN
    from goldset.schema import normalise_question

    keys = [normalise_question(s["q"]) for s in HUMAN + ADVERSARIAL]
    assert len(keys) == len(set(keys))


def test_abstaining_seeds_carry_no_evidence():
    from goldset.adversarial_items import ADVERSARIAL
    from goldset.schema import CATEGORIES

    for seed in ADVERSARIAL:
        if (seed.get("behaviour") or CATEGORIES[seed["cat"]]).startswith("abstain"):
            assert not seed.get("ev"), seed["q"]


# --- judging -------------------------------------------------------------------------------------------


def test_a_grade_is_parsed_out_of_a_chatty_reply():
    assert llm.parse_grade("2") == 2
    assert llm.parse_grade("I would say 3 because it answers directly.") == 3
    assert llm.parse_grade("not a number") is None
    assert llm.parse_grade(None) is None


def test_json_is_parsed_out_of_a_fenced_reply():
    assert llm.parse_json_object('```json\n{"grade": 2}\n```') == {"grade": 2}
    assert llm.parse_json_object("no object here") is None
    assert llm.parse_json_object('prose {"a": {"b": 1}} more') == {"a": {"b": 1}}


def test_the_cost_estimate_scales_with_the_prompts_and_names_its_basis():
    small = llm.estimate(["a" * 400], "deepseek-flash")
    big = llm.estimate(["a" * 400] * 100, "deepseek-flash")
    assert big.usd > small.usd
    assert big.calls == 100
    assert "chars/token" in small.basis


def test_an_unlisted_model_estimates_pessimistically_and_says_so():
    known = llm.estimate(["a" * 4000], "deepseek-flash")
    unknown = llm.estimate(["a" * 4000], "some-new-model")
    assert unknown.usd > known.usd
    assert "pessimistic fallback" in unknown.basis


def test_a_client_with_no_key_is_not_configured_and_refuses_to_call(monkeypatch):
    """The settings fall-back has to be neutralised, or this passes or fails depending on whether the machine running
    the tests happens to have a DEEPSEEK_KEY in its.env.
    """
    monkeypatch.setattr(llm.settings, "deepseek_key", "")
    client = llm.Client(key="", base="https://example.invalid", model="m")
    assert not client.configured
    with pytest.raises(llm.LlmError, match="no API key"):
        run(client.models())


def test_the_key_never_appears_in_the_clients_repr():
    client = llm.Client(key="sk-supersecret-value", base="https://example.invalid", model="m")
    assert "supersecret" not in repr(client)


def test_the_user_agent_carries_no_personal_data():
    client = llm.Client(key="k", base="https://example.invalid", model="m")
    agent = client._headers()["User-Agent"]
    assert "@" not in agent and agent == "on-the-record/goldset"


def test_apply_grades_keeps_a_floor_under_the_speeches_the_item_was_built_from():
    """A judge disagreeing with ground truth is a fact about the judge, not a reason to unlabel it."""
    one = item(graded_relevance={})
    judging.apply_grades([one], {"hum0001": {"s1": 0, "s2": 3}}, ["dense"])
    assert one.graded_relevance == {"s1": 2, "s2": 3}
    assert one.pooled_by == ("dense",)


def test_judged_at_k_reports_how_much_of_each_variants_top_ten_was_graded():
    pool = judging.Pool(variants=["dense", "full"])
    pool.retrieved = {"q1": {"dense": ["a", "b"], "full": ["a", "z"]}}
    scores = judging.judged_at_k(pool, {"q1": {"a": 3, "b": 1}}, k=10)
    assert scores == {"dense": 1.0, "full": 0.5}


def test_judged_at_k_is_zero_rather_than_an_error_when_nothing_was_retrieved():
    assert judging.judged_at_k(judging.Pool(variants=["dense"]), {}, k=10) == {"dense": 0.0}


def test_cohen_kappa_is_one_on_perfect_agreement_and_zero_on_chance():
    grades = {("q", f"s{i}"): i % 4 for i in range(40)}
    perfect = judging.cohen_kappa(grades, dict(grades))
    assert perfect["kappa"] == 1.0 and perfect["passes_gate"]
    shifted = judging.cohen_kappa(grades, {k: (v + 1) % 4 for k, v in grades.items()})
    assert shifted["kappa"] < 0.1 and not shifted["passes_gate"]


def test_cohen_kappa_reports_n_so_a_small_sample_cannot_masquerade_as_a_result():
    assert judging.cohen_kappa({}, {})["n"] == 0
    assert judging.cohen_kappa({("q", "s"): 3}, {("q", "s"): 3})["n"] == 1


def test_a_judging_prompt_carries_the_speaker_house_and_date():
    pair = judging.Pair("i1", "s1", "What was said?", "A passage.", "A Member", "Lab", "Commons", "2026-09-02", "H")
    prompt = pair.prompt()
    assert "A Member" in prompt and "Commons" in prompt and "2026-09-02" in prompt
    assert "3 = Perfectly relevant" in prompt


# --- the review pack ----------------------------------------------------------------------------------


def test_the_review_queue_puts_critical_authored_items_first(corpus):
    items = [
        item(id="db0001", criticality="critical", provenance="db_template", question="A template question?"),
        item(id="adv0001", criticality="critical", provenance="adversarial", question="An adversarial question?"),
        item(id="hum0002", criticality="minor", question="A minor authored question?"),
    ]
    pack = with_conn(corpus, lambda c: review.build_pack(c, items, spot_check=1))
    assert pack["queue"][0]["id"] == "adv0001"
    assert pack["queue"][0]["phase"].startswith("1 of 3")


def test_the_review_pack_estimates_the_owners_time(corpus):
    pack = with_conn(corpus, lambda c: review.build_pack(c, many(8, criticality="critical")))
    assert pack["plan"]["critical"] == 8
    assert pack["plan"]["estimated_minutes"] == pytest.approx(8 * review.SECONDS_PER_ITEM / 60, abs=1)


def test_the_rendered_page_is_self_contained_and_embeds_its_pack(corpus, tmp_path):
    pack = with_conn(corpus, lambda c: review.build_pack(c, many(3, criticality="critical")))
    page = review.render(pack, tmp_path)
    html = page.read_text(encoding="utf-8")
    assert "__PACK__" not in html
    assert "<script" in html and "src=" not in html.split("<script id=\"pack\"")[0].split("<style")[0]
    assert json.loads((tmp_path / "pack.json").read_text(encoding="utf-8"))["gold_hash"] == pack["gold_hash"]


def test_a_closing_script_tag_inside_the_data_cannot_break_out_of_the_page(corpus, tmp_path):
    """An item quoting '</script>' would otherwise end the host script tag early and blank the tool."""
    nasty = item(question="What about </script><h1>this</h1> in a question?")
    pack = with_conn(corpus, lambda c: review.build_pack(c, [nasty]))
    html = review.render(pack, tmp_path).read_text(encoding="utf-8")
    assert "</script><h1>" not in html
    assert "<\\/script>" in html
