"""An overview does not answer a question about a period the record does not cover."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from answer import contract as K
from answer import generate as G
from answer import pipeline as P
from answer import routes as RT
from ingest.runtime import run

PROV = K.Provenance(route="overview", resolved_filters={}, corpus_version="v2", retrieval_config_hash="c", recipe="r",
                    prompt_version="p", model_id="m")  # fmt: skip
VERIFIED = K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True, window_ok=True,
                          checked_by="test")  # fmt: skip


class _Hit:
    def __init__(self, chunk_id, debate_id):
        self.chunk_id, self.debate_id, self.rank, self.score = chunk_id, debate_id, 1, 1.0
        self.speech_id, self.sentence, self.passage = f"s{chunk_id}", "Anchor sentence.", "Anchor sentence."
        self.seq, self.sent_idx = 1, 0
        self.date, self.house, self.party, self.office, self.heading = date(2026, 9, 15), "Commons", "Lab", None, "Grenfell"
        self.speaker, self.url, self.column_ref, self.section, self.rerank_score = "A Member", None, None, "Chamber", 4.0
        self.sibling_chunk_ids = ()
        self.context_sentences = ((1, 0, "Anchor sentence.", None, chunk_id),)


def answered(chunk_id):
    claim = K.Claim(id=f"c{chunk_id}", kind="said", text="A Member said the fire was a disgrace.", speaker_name="A Member",
                    citations=(K.Citation(speech_id=f"s{chunk_id}", chunk_ids=(chunk_id,), quote="Anchor sentence."),),
                    verification=VERIFIED)  # fmt: skip
    return SimpleNamespace(answer=SimpleNamespace(claims=(claim,), abstention=None), claims_generated=1,
                           claims_dropped=0, generator_failed="", degraded=False)  # fmt: skip


def declined(reason=K.Abstention.OUTSIDE_WINDOW):
    return SimpleNamespace(answer=SimpleNamespace(claims=(), abstention=reason), claims_generated=0, claims_dropped=0,
                           generator_failed="", degraded=False)  # fmt: skip


def overview_of(monkeypatch, results):
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

    hits = [_Hit(100 + n, f"d{n}") for n in range(len(results))]
    return run(RT.overview_route(None, "What did MPs say about the Grenfell Tower fire in the week it happened?", hits,
                                 _Client(), PROV))


def test_most_debates_declining_as_out_of_window_declines_the_overview(monkeypatch):
    """5 debates out of window, 2 answering with later material."""
    overview = overview_of(monkeypatch, [declined()] * 5 + [answered(105), answered(106)])
    assert overview.answer.abstention == K.Abstention.OUTSIDE_WINDOW
    assert overview.answer.abstention_detail == G.READER_REASONS[K.Abstention.OUTSIDE_WINDOW]
    assert not overview.answer.claims and not overview.answer.rendered
    assert overview.debate_declines == {"outside_window": 5}


def test_one_debate_out_of_window_among_answers_does_not_decline(monkeypatch):
    overview = overview_of(monkeypatch, [declined()] + [answered(101 + n) for n in range(4)])
    assert overview.answer.abstention is None and overview.answer.rendered
    assert overview.debate_declines == {"outside_window": 1}


def test_other_decline_reasons_do_not_make_a_question_out_of_window(monkeypatch):
    """Debates that hold nothing on the subject decline `not_in_record` — the ordinary case for a broad overview."""
    overview = overview_of(monkeypatch, [declined(K.Abstention.NOT_IN_RECORD)] * 5 + [answered(105), answered(106)])
    assert overview.answer.abstention is None and overview.answer.rendered
    assert overview.debate_declines == {"not_in_record": 5}


def test_a_failed_debate_is_not_counted_as_read(monkeypatch):
    """Two out-of-window declines among three debates READ is a majority, however many others failed."""
    failed = SimpleNamespace(answer=SimpleNamespace(claims=(), abstention=K.Abstention.INSUFFICIENT_EVIDENCE),
                             claims_generated=0, claims_dropped=0, generator_failed="LlmError: 500", degraded=False)
    overview = overview_of(monkeypatch, [failed, failed, declined(), declined(), answered(104)])
    assert overview.answer.abstention == K.Abstention.OUTSIDE_WINDOW


def test_the_router_clarifications_first_letter_is_a_capital_even_inside_a_quote():
    answer = P.clarify_answer("What did the Prime Minister say?", PROV,
                              why="'the prime minister' names different people across the corpus and the question "
                                  "gives no date", clarify=("which date or period do you mean?",))  # fmt: skip
    assert answer.rendered.startswith("'The prime minister' names different people")
