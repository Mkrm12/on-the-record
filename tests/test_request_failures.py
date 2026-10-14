"""A request that FAILED is not an answer, not a decline, and never a measurement."""

from __future__ import annotations

import dataclasses
import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from answer import contract as K
from answer import generate as G
from answer import pipeline as P
from answer import routes as RT
from goldset import judges as J
from goldset.llm import LlmError
from ingest.runtime import run

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_failure_tests", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EA = _script("eval_answers")
GR = _script("e3_gate_report")
PROV = K.Provenance(route="retrieval", resolved_filters={}, corpus_version="v2", retrieval_config_hash="c", recipe="r",
                    prompt_version="p", model_id="m")  # fmt: skip


class _Refusing:

    model = "deepseek-flash"

    async def chat(self, *a, **k):
        raise LlmError("chat/completions returned 401")


class _Hit:
    def __init__(self, chunk_id, debate_id):
        self.chunk_id, self.debate_id, self.rank, self.score = chunk_id, debate_id, 1, 1.0
        self.speech_id, self.sentence, self.passage = f"s{chunk_id}", "Anchor sentence.", "Anchor sentence."
        self.seq, self.sent_idx = 1, 0
        self.date, self.house, self.party, self.office, self.heading = date(2026, 9, 15), "Commons", "Lab", None, "Buses"
        self.speaker, self.url, self.column_ref, self.section, self.rerank_score = "A Member", None, None, "Chamber", 4.0
        self.sibling_chunk_ids = ()
        self.context_sentences = ((1, 0, "Anchor sentence.", None, chunk_id),)


def _one_deck():
    from answer import cards as C

    return C.build_deck([_Hit(101, "dA")], budget_tokens=3_000)


# --- the generator records its own failure ----------------------------------------------------------------------------


def test_a_refused_provider_is_recorded_as_a_failure_not_only_as_a_decline():
    generated = run(G.answer_question("What did members say about buses?", _one_deck(), _Refusing(), PROV))
    assert generated.answer.abstention is not None, "a reader is told it could not answer; nothing is invented"
    assert "401" in generated.generator_failed
    assert generated.degraded and "could not be reached" in generated.degraded_reason


def test_a_model_that_declines_is_not_a_failure():
    class _Declining:
        model = "deepseek-flash"

        async def chat(self, *a, **k):
            class R:
                content = '{"claims": [], "abstention": "not_in_record", "why": "nothing on buses"}'
                model, fingerprint, finish_reason = "deepseek-flash", "fp", "stop"
                prompt_tokens, completion_tokens, cached_tokens, total_tokens = 100, 10, 0, 110

                def cost_usd(self, model=None):
                    return 0.0001

            return R()

    generated = run(G.answer_question("What did members say about buses?", _one_deck(), _Declining(), PROV))
    assert generated.answer.abstention is not None and generated.generator_failed == ""


# --- an overview fails when every debate's generator failed ----------------------------------------------------------


def _overview_with(monkeypatch, results):
    queue = iter(results)

    async def fake_answer_question(question, deck, *a, **k):
        return next(queue)

    async def fake_facets(conn, chunk_ids):
        return {"label": "passages found by search", "total": len(chunk_ids)}

    monkeypatch.setattr(RT.G, "answer_question", fake_answer_question)
    monkeypatch.setattr(RT, "facets", fake_facets)

    class _Client:
        async def chat(self, *a, **k):
            raise RuntimeError("no reduction")

    return run(RT.overview_route(None, "What did members say about buses?", [_Hit(101, "dA"), _Hit(202, "dB")],
                                 _Client(), K.Provenance(**{**PROV.model_dump(), "route": "overview"})))


def _failed_generated(detail="LlmError: chat/completions returned 401"):
    answer = K.Answer(question="q", provenance=PROV, abstention=K.Abstention.INSUFFICIENT_EVIDENCE,
                      abstention_detail="x")  # fmt: skip
    return SimpleNamespace(answer=answer, claims_generated=0, claims_dropped=0, generator_failed=detail)


def test_an_overview_whose_every_debate_failed_is_a_failure(monkeypatch):
    overview = _overview_with(monkeypatch, [_failed_generated(), _failed_generated()])
    assert overview.answer.abstention is not None and "401" in overview.generator_failed


def test_an_overview_that_verified_nothing_while_a_debate_failed_is_a_failure_not_a_decline(monkeypatch):
    """One debate's generator failed and the other found nothing: "none produced a claim that could be verified" is not
    established — the failed debate was never read — so the request failed.
    """
    declined = SimpleNamespace(answer=SimpleNamespace(claims=()), claims_generated=0, claims_dropped=0,
                               generator_failed="")  # fmt: skip
    overview = _overview_with(monkeypatch, [_failed_generated(), declined])
    assert overview.answer.abstention is not None and "401" in overview.generator_failed


def test_an_overview_with_one_debate_answered_is_not_a_failure(monkeypatch):
    verified = K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True, window_ok=True,
                              checked_by="test")  # fmt: skip
    good = K.Claim(id="c1", kind="said", text="A Member said buses were late.", speaker_name="A Member",
                   citations=(K.Citation(speech_id="s202", chunk_ids=(202,), quote="Anchor sentence."),),
                   verification=verified)  # fmt: skip
    answered = SimpleNamespace(answer=SimpleNamespace(claims=(good,)), claims_generated=1, claims_dropped=0,
                               generator_failed="")  # fmt: skip
    overview = _overview_with(monkeypatch, [_failed_generated(), answered])
    assert overview.generator_failed == "" and overview.answer.rendered
    # the reader is told the overview is drawn from fewer debates than were found, and why
    assert overview.failed_debates == 1
    assert "1 of the 2 debates could not be read" in overview.answer.rendered


def test_an_overview_debate_answered_without_the_verifier_marks_the_overview_degraded(monkeypatch):
    verified = K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True, window_ok=True,
                              checked_by="test")  # fmt: skip
    good = K.Claim(id="c1", kind="said", text="A Member said buses were late.", speaker_name="A Member",
                   citations=(K.Citation(speech_id="s101", chunk_ids=(101,), quote="Anchor sentence."),),
                   verification=verified)  # fmt: skip
    plain = SimpleNamespace(answer=SimpleNamespace(claims=(good,)), claims_generated=1, claims_dropped=0,
                            generator_failed="", degraded=False)  # fmt: skip
    degraded = SimpleNamespace(answer=SimpleNamespace(claims=()), claims_generated=2, claims_dropped=2,
                               generator_failed="", degraded=True, degraded_reason="no entailment model available")
    overview = _overview_with(monkeypatch, [plain, degraded])
    assert overview.degraded_reason == "no entailment model available" and overview.generator_failed == ""
    asked = P.Asked(answer=overview.answer, route="overview", overview=overview)
    assert asked.degraded_reason == "no entailment model available"


# --- the request is failed, and a failed request is never cached ----------------------------------------------------


def test_the_request_failure_comes_from_the_request_the_generator_or_the_overview():
    answer = K.Answer(question="q", provenance=PROV, abstention=K.Abstention.INSUFFICIENT_EVIDENCE, abstention_detail="x")
    assert P.request_failure(P.Asked(answer=answer, route="retrieval")) == ""
    assert "401" in P.request_failure(P.Asked(answer=answer, route="retrieval", generated=_failed_generated()))
    overview = SimpleNamespace(generator_failed="LlmError: 401")
    assert "401" in P.request_failure(P.Asked(answer=answer, route="overview", overview=overview))
    assert P.request_failure(P.Asked(answer=answer, route="retrieval", failed_reason="OperationalError: gone")) == \
        "OperationalError: gone"


def test_ask_never_caches_a_failed_request_and_traces_it_as_an_error():
    """Structural: the store is skipped for a failed request, and `finish` records the failure on the trace."""
    import inspect

    source = inspect.getsource(P.ask)
    store_at = source.index("# 7. store")
    assert "request_failure(result)" in source[store_at:source.index("await K.store(", store_at)]
    finish = source[source.index("async def finish("):]
    assert "request_failure(result)" in finish[: finish.index("trace = TR.Trace(")]


# --- the evaluation records a failed row, never an outcome ----------------------------------------------------------


def test_a_failed_request_becomes_a_failed_row_not_an_outcome():
    item = SimpleNamespace(id="hum0001", category="topic_speech", criticality="major")
    answer = K.Answer(question="q", provenance=PROV, abstention=K.Abstention.INSUFFICIENT_EVIDENCE, abstention_detail="x")
    asked = P.Asked(answer=answer, route="retrieval", generated=_failed_generated())
    row = EA.failed_row(item, asked, "LlmError: chat/completions returned 401")
    assert row["outcome"] is None and row["by"] == EA.FAILED and "401" in row["why"]


def test_the_report_counts_failed_rows_apart_from_ungraded_ones():
    base = {"criticality": "minor", "category": "topic_speech", "route": "retrieval"}
    rows = [{"id": "a", "outcome": "correct", "by": "deterministic-structured", **base},
            {"id": "b", "outcome": None, "by": EA.FAILED, **base},
            {"id": "c", "outcome": None, "by": EA.AWAITING_JUDGE, **base}]
    fields = EA.outcome_fields(rows)
    assert fields["failed"] == 1 and fields["ungraded"] == 2


def test_a_run_with_a_failed_request_never_exits_0():
    judged_clean = {"judge": "gemini/x", "scored": {"gate": {"passed": True}}, "stopped_early": "", "ungraded": 0,
                    "failed": 1}
    assert EA.exit_code(judged_clean) == 1


def test_the_gate_report_counts_a_failed_request_as_unknown_never_clean():
    rows = ([{"id": f"x{i:03d}", "outcome": "correct", "by": "gemini", "route": "retrieval", "vital_present": 1,
              "vital_total": 1} for i in range(20)]
            + [{"id": "x020", "outcome": None, "by": EA.FAILED, "route": "retrieval"}])
    gold = {r["id"]: SimpleNamespace(expected_behaviour="answer") for r in rows}
    report = {"rows": rows, "items": 21, "gate_evaluated": True, "trace": {"overall": {"p95_ms": 4000}},
              "router": {"fact_questions": 1, "right_fact": 1, "accuracy": 1.0, "sent_to_a_fact": 0}}
    verdicts = {name: (verdict, value) for name, _, value, verdict in GR.bars(report, gold)}
    verdict, value = verdicts["Harmful answers"]
    assert verdict == "INCOMPLETE" and "1 request FAILED" in value


# --- the run loop itself: failures are counted, reported, and stop the run --------------------------------------------


class _Pool:
    def connection(self):
        pool = self

        class _Conn:
            async def __aenter__(self):
                return pool

            async def __aexit__(self, *exc):
                return False

        return _Conn()

    async def __aexit__(self, *exc):
        return False


def _run_loop(monkeypatch, tmp_path, outcomes, *, problem=None, limit=12, checkpoint=None):
    """`eval_answers.run` with the provider, the database and the reports directory stubbed out: `outcomes` says, per
    question asked, whether the request failed ("fail") or produced an answer ("ok"). `checkpoint` (rows) resumes.
    """
    queue = list(outcomes)
    asked_ids: list[str] = []

    class _Verifier:
        async def verify(self, pairs):
            return [1.0 for _ in pairs]

    runtime = SimpleNamespace(model_id="deepseek-flash", spend_usd=0.0, verifier=_Verifier())

    async def fake_build_runtime(with_verifier=True):
        return runtime, _Pool()

    async def fake_ask(rt, conn, question, **kw):
        asked_ids.append(kw["gold_item_id"])
        kind = queue.pop(0) if queue else "ok"
        if kind == "fail":
            answer = K.Answer(question=question, provenance=PROV, abstention=K.Abstention.INSUFFICIENT_EVIDENCE,
                              abstention_detail="the generator could not be reached (401)")  # fmt: skip
            return P.Asked(answer=answer, route="retrieval", generated=_failed_generated())
        answer = K.Answer(question=question, provenance=PROV, facets={"x": 1}, rendered="An answer.")
        if kind == "degraded":
            generated = SimpleNamespace(answer=answer, degraded=True, degraded_reason="no entailment model available",
                                        generator_failed="", claims_generated=2, claims_verified=1,
                                        claims_dropped=1, claims_rejected=0)  # fmt: skip
            return P.Asked(answer=answer, route="retrieval", generated=generated)
        return P.Asked(answer=answer, route="retrieval")

    async def nothing(conn, run_id):
        return {}

    monkeypatch.setattr(EA.PL, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(EA.PL, "ask", fake_ask)
    monkeypatch.setattr(EA.TR, "summary", nothing)
    monkeypatch.setattr(EA.TR, "liveness", nothing)
    monkeypatch.setattr(EA, "generator_problem", lambda rt: problem)
    monkeypatch.setattr(EA, "checkpoint_path", lambda split, label="": tmp_path / "checkpoint.jsonl")
    monkeypatch.setattr(EA, "append_checkpoint", lambda *a, **k: None)
    if checkpoint is not None:
        monkeypatch.setattr(EA, "load_checkpoint", lambda split, force, label="": (list(checkpoint), [], "unit-run"))
    report = run(EA.run("dev", limit, "hansard_v2", False, 0.10, "unit-run", label="unit",
                        resume=checkpoint is not None))  # fmt: skip
    return report, asked_ids


def test_resuming_a_no_judge_run_asks_again_only_what_failed(monkeypatch, tmp_path):
    """--resume is how a failed request is re-asked. In a no-judge run an answer awaiting the judge is complete — the
    run can do nothing more for it — so asking it again would pay twice for the same answer (and replace the one the
    rest of the run was measured with).
    """
    first = [i.id for i in EA.load_gold() if i.split == "dev"][:6]
    base = {"category": "c", "criticality": "minor", "route": "retrieval", "rendered": "An answer."}
    stored = ([{"id": i, "outcome": None, "by": EA.AWAITING_JUDGE, **base} for i in first[:3]]
              + [{"id": first[3], "outcome": None, "by": EA.FAILED, **base}])
    report, asked = _run_loop(monkeypatch, tmp_path, ["ok"] * 6, limit=6, checkpoint=stored)
    assert asked == first[3:]
    assert report["failed"] == 0 and sum(1 for r in report["rows"] if r["by"] == EA.AWAITING_JUDGE) == 6


def test_a_run_of_failed_requests_stops_and_is_reported_as_failed(monkeypatch, tmp_path):
    report, asked = _run_loop(monkeypatch, tmp_path, ["fail"] * 12)
    assert len(asked) == EA.MAX_GENERATION_FAILURES, "a refused key does not recover: stop, do not grind on"
    assert report["failed"] == EA.MAX_GENERATION_FAILURES and "failed in a row" in report["stopped_early"]
    assert all(r["by"] == EA.FAILED and r["outcome"] is None for r in report["rows"])
    assert report["by_outcome"] == {} or sum(report["by_outcome"].values()) == 0, "nothing was graded"
    assert report["abstention_breakdown"]["abstentions"] == 0, "a failure is never counted as a decline"
    assert EA.exit_code(report) == 1


def test_an_answer_between_failures_resets_the_count(monkeypatch, tmp_path):
    report, asked = _run_loop(monkeypatch, tmp_path, ["fail"] * 4 + ["ok"] + ["fail"] * 4 + ["ok"] * 3, limit=12)
    assert len(asked) == 12 and not report["stopped_early"]
    assert report["failed"] == 8 and sum(1 for r in report["rows"] if r["by"] == EA.AWAITING_JUDGE) == 4


def test_an_answer_produced_without_the_verifier_is_marked_in_its_row_and_counted(monkeypatch, tmp_path):
    """The warm-up proves the verifier is there at the START; a GPU service that dies mid-run degrades every later
    answer — a different system, which the rows did not record, so the run's numbers described a mix as if it were
    the verified system.
    """
    report, _ = _run_loop(monkeypatch, tmp_path, ["ok", "degraded", "ok"], limit=3)
    assert report["degraded"] == 1
    assert [r.get("degraded", False) for r in report["rows"]] == [False, True, False]
    assert "no entailment model" in report["rows"][1]["degraded_reason"]
    judged = {**report, "judge": "gemini/x", "scored": {"gate": {"passed": True}}, "ungraded": 0, "failed": 0}
    assert EA.exit_code(judged) == 1


def test_the_gate_report_does_not_pass_a_run_with_unverified_answers():
    rows = [{"id": f"x{i:03d}", "outcome": "correct", "by": "gemini", "route": "retrieval", "vital_present": 1,
             "vital_total": 1, "degraded": i == 0} for i in range(20)]
    gold = {r["id"]: SimpleNamespace(expected_behaviour="answer") for r in rows}
    report = {"rows": rows, "items": 20, "gate_evaluated": True, "trace": {"overall": {"p95_ms": 4000}},
              "router": {"fact_questions": 1, "right_fact": 1, "accuracy": 1.0, "sent_to_a_fact": 0}}
    verdicts = {name: (verdict, value) for name, _, value, verdict in GR.bars(report, gold)}
    verdict, value = verdicts["Every answer verified (no degraded answers)"]
    assert verdict == "INCOMPLETE" and "1 of 20" in value


def test_a_refused_generator_stops_the_run_before_any_question_is_asked(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="refused the key"):
        _run_loop(monkeypatch, tmp_path, ["ok"], problem="deepseek refused the key (HTTP 401)")


# --- the generator is checked before anything is spent ---------------------------------------------------------------


def test_the_generator_is_checked_against_the_providers_live_list_before_any_spend(monkeypatch):
    seen = {}

    def fake_verify(provider):
        seen["model"] = provider.model
        return "the key was refused (HTTP 401)"

    monkeypatch.setattr(EA.J, "verify_model", fake_verify)
    runtime = SimpleNamespace(model_id="deepseek-flash")
    assert EA.generator_problem(runtime) == "the key was refused (HTTP 401)" and seen["model"] == "deepseek-flash"


def test_a_refused_key_is_named_as_one_not_as_an_exception_class(monkeypatch):
    import httpx

    request = httpx.Request("GET", "https://api.deepseek.com/v1/models")

    def refuse(provider, timeout=10.0):
        raise httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))

    monkeypatch.setattr(J, "listed_models", refuse)
    problem = J.verify_model(dataclasses.replace(J.BY_NAME["deepseek"]))
    assert "401" in problem and "refused" in problem
