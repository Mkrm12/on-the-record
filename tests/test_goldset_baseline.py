"""The authored review baseline, the explanatory review tool, and the judging failures they exist to catch."""

from __future__ import annotations

import json

import httpx
import psycopg
import pytest
from goldset import baseline, judge as judging, llm, review
from goldset.schema import Evidence, GoldItem
from ingest.runtime import run


def with_conn(dsn, fn):
    async def go():
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            return await fn(conn)

    return run(go())


@pytest.fixture
def corpus(scratch, scratch_dsn):
    """The same scratch corpus the other goldset suites build on: one Commons debate per window."""
    scratch.execute(
        "INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role) "
        "VALUES ('W0', 'v2', '2026-09-01', '2026-09-17', 'dev')"
    )
    scratch.execute("INSERT INTO member_id_map (mnis_id, name, party) VALUES (7, 'A Member', 'Labour')")
    scratch.execute(
        "INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version) "
        "VALUES ('d1', '2026-09-02', 'Commons', 'Local Government Reorganisation', 'Commons Chamber', 'other', 'W0', 'test')"
    )
    text = "This debate matters to my constituents and I will explain exactly why that is so."
    scratch.execute(
        "INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, window_id) "
        "VALUES ('s1', 'd1', 0, 'A Member', 7, %s, 'W0')",
        (text * 9,),
    )
    return scratch_dsn


def item(**kwargs) -> GoldItem:
    base = dict(
        id="hum0001", question="What did the Minister say about the levy?", category="exact_term",
        criticality="critical", expected_behaviour="answer", provenance="passage_llm",
        evidence=(Evidence("s1", "The Government published their response to the consultation."),),
    )  # fmt: skip
    return GoldItem(**(base | kwargs))


def pack_with(kappa_passage: str = "A passage that answers it plainly.", **kwargs) -> dict:
    """The smallest pack shaped like a real one: one calibration pair and one item."""
    return {
        "gold_hash": "h", "split_hash": "s", "built_at": "",
        "items": {"adv0001": {"id": "adv0001", "question": "Q?", "category": "false_premise",
                              "evidence": [{"speech_id": "s1", "quote": "a quote",
                                            "context": "before a quote after"}]}},  # fmt: skip
        "kappa": {"k:adv0001:s9": {"item_id": "adv0001", "speech_id": "s9", "question": "Q?",
                                   "passage": kappa_passage}},  # fmt: skip
        "queue": [
            {"id": "adv0001", "kind": "item", "phase": "1 of 3 - critical items"},
            {"id": "k:adv0001:s9", "kind": "kappa", "phase": "2 of 3 - judge calibration"},
        ],
        **kwargs,
    }


def annotation(**kwargs) -> baseline.Annotation:
    base = dict(n=1, kind="kappa", grade=2, summary="It says something.", why="Because it does.",
                marks=(), mark_kind="none", item="adv0001", sid="s9")  # fmt: skip
    return baseline.Annotation(**(base | kwargs))


# A judging run that grades nothing must not look like a run that worked -------------------


def test_an_empty_reply_is_an_error_rather_than_a_grade_of_nothing(monkeypatch):
    """A reasoning model given a small max_tokens spends it thinking and returns content "" with finish_reason
    "length"; HTTP says 200 and every other check passes, so the grade is simply lost. Patching the transport means
    the real `complete` is what is tested.
    """
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "reasoning_content": "thinking"},
                               "finish_reason": "length"}]},  # fmt: skip
        )
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    client = llm.Client(key="k", base="https://example.invalid", model="m")
    with pytest.raises(llm.LlmError, match="empty reply"):
        run(client.complete("grade this"))


def test_a_reply_of_only_whitespace_counts_as_empty_too(monkeypatch):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "  \n  "}}]})
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    client = llm.Client(key="k", base="https://example.invalid", model="m")
    with pytest.raises(llm.LlmError, match="empty reply"):
        run(client.complete("x"))


def test_reasoning_effort_reaches_the_request_only_when_set(monkeypatch):
    seen: list[dict] = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "2"}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    run(llm.Client(key="k", base="https://e.invalid", model="m").complete("x"))
    run(llm.Client(key="k", base="https://e.invalid", model="m", reasoning_effort="none").complete("x"))
    assert "reasoning_effort" not in seen[0]
    assert seen[1]["reasoning_effort"] == "none"


def test_a_judging_run_counts_what_it_lost_so_zero_grades_cannot_pass_as_success():
    class Silent(llm.Client):
        async def batch(self, prompts, *, system="", max_tokens=64):
            return [None] * len(prompts)

    pool = judging.Pool(pairs=[judging.Pair(f"i{i}", f"s{i}", "Q?", "P") for i in range(5)])
    result = run(judging.judge_run(pool, Silent(key="k", base="https://example.invalid", model="m")))
    assert result.graded == 0 and result.offered == 5 and result.no_reply == 5
    assert result.yield_rate == 0.0
    assert "0 of 5" in result.line()


def test_an_unparsable_reply_is_counted_separately_from_no_reply_at_all():
    class Chatty(llm.Client):
        async def batch(self, prompts, *, system="", max_tokens=64):
            return ["I would rather not say", None, "2"]

    pool = judging.Pool(pairs=[judging.Pair(f"i{i}", f"s{i}", "Q?", "P") for i in range(3)])
    result = run(judging.judge_run(pool, Chatty(key="k", base="https://example.invalid", model="m")))
    assert (result.graded, result.unparsable, result.no_reply) == (1, 1, 1)


def test_the_smoke_test_fails_when_the_judge_returns_something_ungradeable():
    class Waffle(llm.Client):
        async def complete(self, prompt, *, system="", max_tokens=64, temperature=0.0):
            return "I am unable to assist with that request."

    ok, detail = run(judging.smoke_test(Waffle(key="k", base="https://example.invalid", model="m")))
    assert not ok and "no grade" in detail


def test_the_smoke_test_passes_on_a_judge_that_answers_with_a_digit():
    class Fine(llm.Client):
        async def complete(self, prompt, *, system="", max_tokens=64, temperature=0.0):
            return "3"

    ok, detail = run(judging.smoke_test(Fine(key="k", base="https://example.invalid", model="m")))
    assert ok and "3" in detail


def test_the_grade_budget_leaves_room_for_a_model_that_reasons_before_answering():
    """4 tokens was exactly enough for a model that answers directly and silently impossible for one that thinks first.
    """
    assert judging.GRADE_MAX_TOKENS >= 32


# --- the judgment cache -----------------------------------------------------------------------------


def test_the_pool_and_raw_grades_round_trip_so_a_re_render_never_re_spends(tmp_path):
    pool = judging.Pool(
        pairs=[judging.Pair("i1", "s1", "Q?", "P", speaker="A Member")],
        variants=["dense", "full"],
        retrieved={"i1": {"dense": ["s1"], "full": ["s1"]}},
    )
    path = judging.save_judgments(pool, {"i1": {"s1": 3}}, model="m", path=tmp_path / "j.json")
    back, grades, meta = judging.load_judgments(path)
    assert grades == {"i1": {"s1": 3}}
    assert back.variants == ["dense", "full"] and back.retrieved == pool.retrieved
    assert back.pairs[0].question == "Q?" and back.pairs[0].speaker == "A Member"
    assert meta["model"] == "m"


def test_a_cache_from_another_version_is_refused_rather_than_read_as_empty(tmp_path):
    path = tmp_path / "j.json"
    path.write_text(json.dumps({"version": 99, "pairs": [], "grades": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        judging.load_judgments(path)


def test_the_cache_holds_raw_grades_not_the_floored_ones():
    """`apply_grades` floors a known-relevant speech at 2. Kappa against a floored grade would credit the judge with
    agreement the scoring scheme invented.
    """
    one = item(id="hum0001")
    raw = {"hum0001": {"s1": 0}}
    judging.apply_grades([one], raw, ["dense"])
    assert one.graded_relevance["s1"] == 2
    assert raw["hum0001"]["s1"] == 0, "apply_grades must not mutate the grades it was given"


def test_flatten_and_confusion_describe_how_two_graders_differ():
    a = {("i", "s1"): 3, ("i", "s2"): 1, ("i", "s3"): 0}
    b = {("i", "s1"): 3, ("i", "s2"): 2, ("i", "s3"): 3}
    table = judging.confusion(a, b)
    assert table["exact"] == 1 and table["adjacent_or_better"] == 2 and table["severe"] == 1
    assert table["table"]["0"]["3"] == 1
    assert judging.flatten({"i": {"s1": 3}}) == {("i", "s1"): 3}


# --- highlight resolution ---------------------------------------------------------------------------


def test_a_highlight_matches_through_curly_quotes_and_em_dashes():
    """A reviewer types straight quotes and hyphens; Hansard stores typographic ones. A span that fails to match
    renders as an unmarked passage, which reads exactly like 'nothing here answers it'.
    """
    text = "He said “the levy” — and nothing more."
    found = baseline.locate(text, 'said "the levy" - and nothing')
    assert found is not None
    start, end = found
    assert text[start:end] == "said “the levy” — and nothing"


def test_a_highlight_matches_across_collapsed_whitespace_and_returns_real_offsets():
    text = "One   two\n\nthree four"
    start, end = baseline.locate(text, "two three")
    assert text[start:end] == "two\n\nthree"


def test_a_highlight_that_is_not_in_the_passage_is_reported_not_silently_dropped():
    resolved = baseline.resolve(
        pack_with(), [annotation(marks=("words that are absent",), mark_kind="answer")], []
    )
    assert not resolved.ok
    assert any("highlight not found" in p for p in resolved.problems)


def test_an_item_highlight_is_found_in_whichever_evidence_passage_holds_it():
    """A multi_speech or multi_hop item's judgment often turns on its SECOND speech, and looking only in the first
    would report a span that is plainly there as missing.
    """
    pack = {
        "gold_hash": "h", "split_hash": "s",
        "items": {"adv0001": {"id": "adv0001", "question": "Q?", "evidence": [
            {"speech_id": "s1", "quote": "q1", "context": "the first speech says one thing"},
            {"speech_id": "s2", "quote": "q2", "context": "the second speech says quite another"},
        ]}},  # fmt: skip
        "kappa": {},
        "queue": [{"id": "adv0001", "kind": "item", "phase": "1 of 3 - critical items"}],
    }
    resolved = baseline.resolve(
        pack, [],
        [baseline.Annotation(n=1, kind="item", decision="verified", summary="s", why="w",
                             marks=("says quite another",), mark_kind="answer", item="adv0001")],  # fmt: skip
    )
    assert resolved.ok, resolved.problems
    spans = resolved.items["adv0001"]["evidence_spans"]
    assert spans[0] == [] and spans[1] == [[18, 36]]
    assert pack["items"]["adv0001"]["evidence"][1]["context"][18:36] == "says quite another"


def test_overlapping_highlights_are_merged_so_the_markup_cannot_nest():
    assert baseline.merge_spans([(0, 5), (3, 9), (20, 22)]) == [(0, 9), (20, 22)]


def test_a_span_is_resolved_to_offsets_rather_than_shipped_as_a_string():
    resolved = baseline.resolve(
        pack_with("The Minister ruled it out entirely."),
        [annotation(marks=("ruled it out",), mark_kind="answer")],
        [],
    )
    assert resolved.ok
    spans = resolved.kappa["k:adv0001:s9"]["spans"]
    assert spans == [[13, 25]]


# --- the checksum guard -----------------------------------------------------------------------------


def test_an_annotation_pointing_at_the_wrong_pair_refuses_the_whole_baseline():
    """A hundred hand-written keys invite a transposition that nothing would ever notice."""
    resolved = baseline.resolve(pack_with(), [annotation(sid="WRONGSID")], [])
    assert not resolved.ok
    assert any("expected speech" in p for p in resolved.problems)


def test_an_annotation_for_the_wrong_item_is_refused_too():
    resolved = baseline.resolve(pack_with(), [annotation(item="adv9999")], [])
    assert not resolved.ok
    assert any("expected item" in p for p in resolved.problems)


def test_an_annotation_past_the_end_of_the_queue_is_refused():
    resolved = baseline.resolve(pack_with(), [annotation(n=7)], [])
    assert not resolved.ok
    assert any("no calibration pair at that position" in p for p in resolved.problems)


@pytest.mark.parametrize(
    "bad, expected",
    [
        ({"grade": 9}, "grade must be"),
        ({"summary": ""}, "plain-English summary"),
        ({"why": ""}, "one-sentence reason"),
        ({"mark_kind": "sideways"}, "mark_kind must be"),
        ({"mark_kind": "answer"}, "no span was given"),
        ({"marks": ("x",), "mark_kind": "none"}, "spans were given"),
    ],
)
def test_a_malformed_annotation_says_exactly_what_is_wrong(bad, expected):
    problems = annotation(**bad).validate()
    assert any(expected in p for p in problems), problems


def test_an_item_decision_outside_the_vocabulary_is_refused():
    problems = baseline.Annotation(n=1, kind="item", decision="looks fine to me",
                                   summary="s", why="w").validate()  # fmt: skip
    assert any("decision must be one of" in p for p in problems)


# --- the mechanical spot-check ----------------------------------------------------------------------


def test_a_null_party_is_not_a_defect_because_the_speaker_sits_for_none():
    """A check that cries wolf on correct rows is worse than no check: the reviewer learns to click through it. The
    Speaker and some peers legitimately have no party.
    """
    clean = {"question": "Who opened the debate on Committees on 2 March 2020?",
             "expected_structured_facts": {"opened_by": "Sir Lindsay Hoyle", "party": None},
             "nuggets": [{"text": "Sir Lindsay Hoyle opened it.", "vital": True}]}  # fmt: skip
    assert baseline.check_structured_facts(clean) == []


def test_an_empty_answer_field_is_still_a_defect():
    problems = baseline.check_structured_facts(
        {"question": "q", "expected_structured_facts": {"opened_by": ""},
         "nuggets": [{"text": "t", "vital": True}]}  # fmt: skip
    )
    assert any("empty value for opened_by" in p for p in problems)


def test_a_question_that_scopes_a_house_the_answer_leaves_null_is_flagged():
    """Found by eye on two Royal Assent items: Royal Assent is taken in neither House, but the template still wrote 'in
    the Commons' into the question.
    """
    problems = baseline.check_structured_facts(
        {"question": "When did the Finance Act 2026 have its Royal Assent in the Commons?",
         "expected_structured_facts": {"stage": "Royal Assent", "house": None, "first_sitting": "2026-03-18"},
         "nuggets": [{"text": "It was on 18 March 2026.", "vital": True}]}  # fmt: skip
    )
    assert any("scopes an answer to a House" in p for p in problems)


def test_a_placeholder_left_in_a_sql_answer_is_caught():
    problems = baseline.check_structured_facts(
        {"question": "q", "expected_structured_facts": {"holder": "TO_VERIFY"},
         "nuggets": [{"text": "t", "vital": True}]}  # fmt: skip
    )
    assert any("placeholder" in p for p in problems)


def test_an_item_with_no_vital_nugget_cannot_distinguish_wrong_from_incomplete():
    problems = baseline.check_structured_facts(
        {"question": "q", "expected_structured_facts": {"a": 1}, "nuggets": [{"text": "t", "vital": False}]}
    )
    assert any("no vital nugget" in p for p in problems)


def test_the_automatic_spot_check_never_verifies_an_item_a_check_flagged():
    pack = {
        "items": {"db0001": {"id": "db0001", "category": "count", "question": "How many?",
                             "expected_structured_facts": {}, "nuggets": []}},  # fmt: skip
        "queue": [{"id": "db0001", "kind": "item", "phase": "3 of 3 - database spot-check"}],
    }
    [entry] = baseline.auto_item_annotations(pack)
    assert entry["decision"] == "edit"
    assert "no expected_structured_facts" in entry["why"]


# --- the export -------------------------------------------------------------------------------------


def test_the_exported_file_uses_the_same_field_names_the_browser_writes():
    """The maintainer must be able to accept the baseline, change six items in the tool and save over it."""
    resolved = baseline.resolve(pack_with(), [annotation()], [])
    payload = baseline.export(resolved, pack_with())
    assert set(payload) >= {"gold_hash", "split_hash", "decisions", "owner_grades", "edits", "counts"}
    assert payload["owner_grades"] == {"k:adv0001:s9": 2}
    assert payload["rationales"]["k:adv0001:s9"]["why"]


def test_baseline_grades_come_back_keyed_the_way_cohen_kappa_compares_them():
    resolved = baseline.resolve(pack_with(), [annotation(grade=3)], [])
    assert baseline.baseline_grades(resolved) == {("adv0001", "s9"): 3}
    agreement = judging.cohen_kappa(baseline.baseline_grades(resolved), {("adv0001", "s9"): 3})
    assert agreement["n"] == 1 and agreement["agreement"] == 1.0


# --- the rendered page ------------------------------------------------------------------------------


def test_the_page_marks_the_baseline_span_and_offers_approve_all(corpus, tmp_path):
    pack = with_conn(corpus, lambda c: review.build_pack(c, [item()], spot_check=0))
    pack["baseline"] = {
        "kappa": {}, "problems": [], "reviewer": "claude",
        "items": {"hum0001": {"decision": "verified", "summary": "A plain summary.",
                              "why": "Because the evidence says so.", "mark_kind": "answer",
                              "checks": [], "evidence_spans": [[[4, 14]]]}},  # fmt: skip
    }
    html = review.render(pack, tmp_path).read_text(encoding="utf-8")
    assert "Approve all" in html
    assert 'mark class="${cls}"' in html or "mark class=" in html
    assert "What this passage actually says" in html
    # Three columns now: the blind baseline, the judge, and the maintainer -- whose grade is the one the gate is
    # computed from, so it has to be present and labelled as the one that counts.
    assert "The readings" in html
    assert "blind baseline" in html and "the one that counts" in html
    assert "Weighting" in html, "the owner must be able to re-weight an item"
    assert "relevance gate at" in html, "the live gate panel must be in the page"


def test_the_pack_carries_the_judges_raw_grade_for_each_calibration_pair(corpus):
    one = item(id="hum0001")
    pool = judging.Pool(pairs=[judging.Pair("hum0001", "s1", "Q?", "A passage.")], variants=["dense"])
    pack = with_conn(
        corpus,
        lambda c: review.build_pack(c, [one], pool=pool, spot_check=0,
                                    judge_grades={"hum0001": {"s1": 1}}, judge_model="deepseek-flash",
                                    with_baseline=False),  # fmt: skip
    )
    assert pack["judge"]["model"] == "deepseek-flash"
    assert pack["judge"]["grades"] == {"k:hum0001:s1": 1}


def test_a_stale_baseline_is_reported_and_withheld_rather_than_shown_beside_the_wrong_passage(corpus):
    """A rationale written for one passage, rendered beside another, is worse than no rationale."""
    resolved = baseline.resolve(pack_with(), [annotation(sid="NOPE")], [])
    assert not resolved.ok and resolved.kappa == {}

# --- the spend floor ---------------------------------------------------------------------------------


def _client_with_balance(amount, monkeypatch, floor=4.0, ceiling=100.0):
    """A client whose provider reports `amount` USD, or None to report nothing at all."""
    monkeypatch.setattr(llm.settings, "min_balance_usd", floor)
    monkeypatch.setattr(llm.settings, "spend_confirm_gbp", ceiling)
    client = llm.Client(key="k", base="https://example.invalid", model="m")

    async def balance():
        if amount is None:
            return {"balance_infos": []}
        return {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": str(amount)}]}

    monkeypatch.setattr(client, "balance", balance)
    return client


def test_a_run_that_would_break_the_floor_is_refused(monkeypatch):
    """The maintainer set a floor. A batch cannot be stopped half way through, so the check is before it."""
    ok, why = run(_client_with_balance("4.20", monkeypatch).affordable(0.49))
    assert not ok
    assert "below the $4.00 floor" in why


def test_a_run_that_leaves_the_floor_intact_is_allowed(monkeypatch):
    ok, why = run(_client_with_balance("9.00", monkeypatch).affordable(0.49))
    assert ok and "left afterwards" in why


def test_a_run_at_or_above_the_approval_bar_refuses_however_healthy_the_balance(monkeypatch):
    """The maintainer's rule: pennies freely, 50p+ needs a word first. A fat balance is not permission."""
    ok, why = run(_client_with_balance("500.00", monkeypatch, floor=0.0, ceiling=0.50).affordable(0.69))
    assert not ok
    assert "needs approval first" in why and "GBP 0.54" in why


def test_a_penny_run_goes_through_without_asking(monkeypatch):
    ok, why = run(_client_with_balance("3.51", monkeypatch, floor=0.0, ceiling=0.50).affordable(0.02))
    assert ok and "under the GBP 0.50 bar" in why


def test_the_approval_bar_can_be_raised_once_the_owner_has_approved(monkeypatch):
    client = _client_with_balance("3.51", monkeypatch, floor=0.0, ceiling=0.50)
    assert not run(client.affordable(0.69))[0]
    assert run(client.affordable(0.69, None, 2.00))[0], "an explicit bar is how an approved run proceeds"


def test_the_bar_is_checked_before_the_balance_so_the_reason_is_the_real_one(monkeypatch):
    """A run that is both too big and unaffordable should say it is too big: that is the fixable half."""
    ok, why = run(_client_with_balance("0.10", monkeypatch, floor=0.05, ceiling=0.50).affordable(5.00))
    assert not ok and "needs approval first" in why


def test_a_provider_that_will_not_say_what_is_left_is_refused_not_assumed_fine(monkeypatch):
    """Unknown is a reason to stop and ask, not a reason to spend."""
    ok, why = run(_client_with_balance(None, monkeypatch).affordable(0.01))
    assert not ok and "did not report a balance" in why


def test_a_floor_of_zero_means_no_floor(monkeypatch):
    """The default now. A provider that will not report a balance no longer blocks a run, because there is no floor to
    check it against -- but the approval bar still applies.
    """
    ok, _why = run(_client_with_balance(None, monkeypatch, floor=0.0, ceiling=0.0).affordable(999.0))
    assert ok, "no floor and no bar means nothing to refuse on"
    assert llm.settings.min_balance_usd == 0.0 or True  # the live default is asserted below


def test_the_shipped_defaults_are_no_floor_and_a_fifty_pence_bar():
    from app.config import Settings

    fresh = Settings(_env_file=None)
    assert fresh.min_balance_usd == 0.0, "the floor was lifted deliberately"
    assert fresh.spend_confirm_gbp == 0.50, "the owner's rule: tell me before 50p+"


def test_the_balance_is_read_as_a_number_and_a_junk_value_does_not_crash(monkeypatch):
    assert run(_client_with_balance("4.34", monkeypatch).available_usd()) == 4.34
    assert run(_client_with_balance("not a number", monkeypatch).available_usd()) is None


# --- binarised relevance -----------------------------------------------------------------------------


def test_binarised_agreement_ignores_the_zero_versus_one_argument():
    """Recall and nDCG read one bit: relevant at grade >= 2, or not. Two graders can disagree constantly about on-
    topic-but-useless and still make the identical relevance call every time.
    """
    mine = {("i", f"s{n}"): g for n, g in enumerate([0, 1, 1, 0, 3, 2])}
    theirs = {("i", f"s{n}"): g for n, g in enumerate([1, 0, 0, 1, 3, 2])}
    four = judging.cohen_kappa(mine, theirs)
    binary = judging.binary_agreement(mine, theirs)
    assert four["kappa"] < 0.6, "the four-level number should be poor here"
    assert binary["kappa"] == 1.0 and binary["passes_gate"]
    assert binary["both_relevant"] == 2 and binary["neither_relevant"] == 4


def test_binarised_agreement_still_catches_a_judge_that_misses_relevant_passages():
    mine = {("i", f"s{n}"): g for n, g in enumerate([3, 3, 2, 2, 0, 0])}
    theirs = {("i", f"s{n}"): g for n, g in enumerate([1, 0, 1, 0, 0, 0])}
    binary = judging.binary_agreement(mine, theirs)
    assert binary["only_first_relevant"] == 4 and binary["both_relevant"] == 0
    assert not binary["passes_gate"]


def test_binarised_agreement_reports_n_zero_rather_than_a_number_when_nothing_overlaps():
    assert judging.binary_agreement({}, {})["n"] == 0


def test_the_relevance_threshold_is_two_because_one_is_on_topic_not_answering():
    assert judging.RELEVANT_AT == 2


def test_the_judge_token_budget_is_settable_for_a_reasoning_model():
    """64 fits a judge that answers directly; a reasoning judge needs several hundred, and the difference is exactly
    what has to be measurable when the agreement gate fails.
    """
    seen = {}

    class Recording(llm.Client):
        async def batch(self, prompts, *, system="", max_tokens=64):
            seen["max_tokens"] = max_tokens
            return ["2"] * len(prompts)

    pool = judging.Pool(pairs=[judging.Pair("i1", "s1", "Q?", "P")])
    run(judging.judge_run(pool, Recording(key="k", base="https://e.invalid", model="m"), max_tokens=512))
    assert seen["max_tokens"] == 512

# --- the gate --------------------------------------------------------------------------------------


def test_a_judge_that_agrees_about_relevance_is_usable_even_if_it_argues_about_zero_versus_one():
    """Recall, MRR and nDCG's gain read only that call, so gating on the four-level number would have thrown away a
    judge that is right about the question being asked.
    """
    owner = {("i", f"s{n}"): g for n, g in enumerate([0, 1, 1, 0, 1, 0, 3, 2, 2, 3])}
    judge = {("i", f"s{n}"): g for n, g in enumerate([1, 0, 0, 1, 0, 1, 3, 2, 2, 3])}
    verdict = judging.gate_verdict(owner, judge)
    assert verdict["usable_for_ranking"] is True
    assert verdict["usable_for_graded_ndcg"] is False
    assert "NOT for graded nDCG" in verdict["verdict"]


def test_a_judge_that_misses_real_answers_is_not_usable_at_all():
    owner = {("i", f"s{n}"): g for n, g in enumerate([3, 3, 2, 2, 0, 0, 0, 0])}
    judge = {("i", f"s{n}"): g for n, g in enumerate([1, 0, 1, 1, 0, 0, 0, 0])}
    verdict = judging.gate_verdict(owner, judge)
    assert verdict["usable_for_ranking"] is False
    assert "not usable" in verdict["verdict"]


def test_a_judge_that_agrees_on_every_level_is_usable_for_everything():
    owner = {("i", f"s{n}"): n % 4 for n in range(40)}
    verdict = judging.gate_verdict(owner, dict(owner))
    assert verdict["usable_for_ranking"] and verdict["usable_for_graded_ndcg"]
    assert verdict["verdict"] == "usable, including for graded nDCG"


def test_the_gate_says_so_when_nothing_was_graded_by_both():
    verdict = judging.gate_verdict({("i", "s")}, {}) if False else judging.gate_verdict({}, {})
    assert verdict["usable_for_ranking"] is False
    assert verdict["verdict"] == "no pairs graded by both"


def test_the_relevance_threshold_is_configurable_because_it_is_a_weighting_choice():
    """The maintainer can move what counts as relevant in the review tool, so the measure has to follow."""
    owner = {("i", f"s{n}"): g for n, g in enumerate([1, 1, 0, 0])}
    judge = {("i", f"s{n}"): g for n, g in enumerate([1, 1, 0, 0])}
    at_one = judging.gate_verdict(owner, judge, threshold=1)
    at_two = judging.gate_verdict(owner, judge, threshold=2)
    assert at_one["binarised"]["both_relevant"] == 2
    assert at_two["binarised"]["both_relevant"] == 0
    assert at_one["threshold"] == 1 and at_two["threshold"] == 2


def test_the_gate_value_is_the_briefs_and_lives_in_one_place():
    assert judging.GATE == 0.6


# --- the cost estimate -------------------------------------------------------------------------------


def test_the_estimate_predicts_output_tokens_rather_than_the_cap():
    """max_tokens is a ceiling, not a prediction. Costing every call at the ceiling made a 100-call check look like a
    $0.15 run and tripped the balance floor on runs that cost pennies.
    """
    assert judging.expected_output_tokens("none") < judging.expected_output_tokens("")
    pool = judging.Pool(pairs=[judging.Pair("i", "s", "Q?", "P" * 400)] * 10)
    off = judging.estimate_judging(pool, "deepseek-flash", "none")
    on = judging.estimate_judging(pool, "deepseek-flash", "")
    assert on.usd > off.usd
    assert off.output_tokens == 10 * judging.expected_output_tokens("none")


def test_reasoning_on_is_estimated_dearer_on_the_dearer_model():
    pool = judging.Pool(pairs=[judging.Pair("i", "s", "Q?", "P" * 400)] * 10)
    assert (judging.estimate_judging(pool, "deepseek-v4-pro", "").usd
            > judging.estimate_judging(pool, "deepseek-flash", "").usd)

# --- a capped run must not persist anything ---------------------------------------------------------


def test_a_capped_run_is_a_dry_run_and_a_full_run_is_not():
    """This invariant took three attempts because each write site had its own copy of the test: the judgments cache was
    protected, then the gold set was still stripped, then the review pack. Naming it once is what stops the next
    write site missing it.
    """
    assert judging.persists(None) is True
    assert judging.persists(20) is False


def test_a_limit_of_zero_is_a_limit_and_not_the_absence_of_one():
    """`if limit else pool.pairs` treated 0 as "no limit", so the one value meaning "send nothing" would have sent all
    3,960 pairs and spent the whole budget.
    """
    assert judging.persists(0) is False

    class Counting(llm.Client):
        sent = -1

        async def batch(self, prompts, *, system="", max_tokens=64):
            Counting.sent = len(prompts)
            return ["2"] * len(prompts)

    pool = judging.Pool(pairs=[judging.Pair(f"i{i}", f"s{i}", "Q?", "P") for i in range(30)])
    client = Counting(key="k", base="https://e.invalid", model="m")
    assert run(judging.judge_run(pool, client, limit=0)).offered == 0
    assert Counting.sent == 0, "a limit of 0 must send nothing"
    assert run(judging.judge_run(pool, client, limit=None)).offered == 30


def test_a_capped_runs_estimate_prices_only_what_it_will_send():
    """The estimate priced the whole pool regardless of the cap, so the balance floor refused a run worth a third of a
    penny on the grounds that it could not afford 69 cents.
    """
    pool = judging.Pool(pairs=[judging.Pair(f"i{i}", f"s{i}", "Q?", "P" * 400) for i in range(200)])
    full = judging.estimate_judging(pool, "deepseek-flash", "none")
    capped = judging.estimate_judging(pool, "deepseek-flash", "none", limit=20)
    assert capped.calls == 20 and full.calls == 200
    assert capped.usd < full.usd / 5
