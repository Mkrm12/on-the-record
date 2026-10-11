"""What a reader is shown is never empty, and a speaker is named once."""

from __future__ import annotations

from datetime import date

import pytest

from answer import contract as K
from answer import pipeline as P
from answer import routes as RT
from answer import verify as V
from ingest.runtime import run

PROV = dict(route="retrieval", resolved_filters={}, corpus_version="v2", retrieval_config_hash="c", recipe="r",
            prompt_version="p", model_id="m")  # fmt: skip
VERIFIED = K.Verification(quote_ok=True, numbers_ok=True, attribution_ok=True, citation_ok=True, window_ok=True,
                          checked_by="test")  # fmt: skip


def claim(text: str, speaker: str | None, cid: str = "c1", chunk: int = 1) -> K.Claim:
    return K.Claim(id=cid, kind="summary", text=text, speaker_name=speaker, date=date(2026, 9, 10),
                   citations=(K.Citation(speech_id="s1", chunk_ids=(chunk,), quote="the system failed"),),
                   verification=VERIFIED)  # fmt: skip


def rendered(text: str, speaker: str | None) -> str:
    return K.Answer(question="q", provenance=K.Provenance(**PROV), claims=(claim(text, speaker),)).render()


# --- one attribution, whichever name for the speaker the claim leads with --------------------------------------------


@pytest.mark.parametrize("text, speaker, expected", [
    # The surname
    ("Mather said that the system failed.", "Keir Mather", "Keir Mather said that the system failed."),
    ("Nandy argued that the system failed.", "Lisa Nandy", "Lisa Nandy argued that the system failed."),
    # an honorific and the surname
    ("Mr Mather said the system failed.", "Keir Mather", "Keir Mather said the system failed."),
    ("Coffey said the system failed.", "Dr Thérèse Coffey (now Baroness Coffey)",
     "Dr Thérèse Coffey (now Baroness Coffey) said the system failed."),
    # A member shown "(now …)", named by the printed surname
    ("Gove told the House that the system failed.", "Michael Gove (now Lord Gove)",
     "Michael Gove (now Lord Gove) told the House that the system failed."),
    # a peer's title without the territorial designation
    ("Baroness Smith said that the system failed.", "Baroness Smith of Malvern",
     "Baroness Smith of Malvern said that the system failed."),
    # a surname of three letters is safe once a reporting verb must follow ("May I begin" has none)
    ("May said that the system failed.", "Theresa May", "Theresa May said that the system failed."),
])  # fmt: skip
def test_a_claim_naming_its_speaker_by_a_shorter_name_is_attributed_once_by_the_printed_name(text, speaker, expected):
    assert rendered(text, speaker) == f"{expected} [2026-09-10]"


@pytest.mark.parametrize("text, speaker", [
    ("Keir Mather said that the system failed.", "Keir Mather"),
    ("Michael Gove said that the system failed.", "Michael Gove (now Lord Gove)"),
    ("Lord Gove said that the system failed.", "Michael Gove (now Lord Gove)"),
    ("Keir Mather's report said the system failed.", "Keir Mather"),  # a full name needs no reporting verb
])  # fmt: skip
def test_a_claim_naming_its_speaker_in_full_is_unchanged(text, speaker):
    assert rendered(text, speaker) == f"{text} [2026-09-10]"


@pytest.mark.parametrize("text", [
    "She said that the system failed.",
    "He said that the system failed.",
    "They said that the system failed.",
])  # fmt: skip
def test_a_pronoun_is_never_replaced_by_the_speakers_name(text):
    """"She said" may be the speaker or someone the speaker quoted ("The Minister told us … She said …"). Replacing it
    with the speaker's name could turn reported speech into a misattribution — harmful — so the claim keeps the
    explicit attribution in front: inelegant, never false.
    """
    assert rendered(text, "Helen Morgan") == f"Helen Morgan said: {text} [2026-09-10]"


@pytest.mark.parametrize("text", [
    "Reeves said that the system failed.",  # someone else's surname: reported speech, kept as written
    "Mather's report said the system failed.",  # a possessive is not an attribution
    "Matheson said that the system failed.",  # a longer word beginning with the surname
    "May I say that the system failed.",  # the surname as an ordinary word, with no reporting verb after it
    "Smith said that the system failed.",  # a peer is not named by the surname alone
])  # fmt: skip
def test_a_lead_that_is_not_the_speakers_own_short_name_keeps_the_explicit_attribution(text):
    speaker = "Baroness Smith of Malvern" if text.startswith("Smith") else (
        "Theresa May" if text.startswith("May") else "Keir Mather")
    assert rendered(text, speaker) == f"{speaker} said: {text} [2026-09-10]"


@pytest.mark.parametrize("text, speaker, lead", [
    ("Mather said that X.", "Keir Mather", "Mather"),
    ("mather said that X.", "Keir Mather", "mather"),
    ("Keir Mather said that X.", "Keir Mather", "Keir Mather"),
    ("Baroness Smith said X.", "Baroness Smith of Malvern", "Baroness Smith"),
    ("Baroness Smith of Malvern said X.", "Baroness Smith of Malvern", "Baroness Smith of Malvern"),
    ("Dame Eleanor Laing said X.", "Dame Eleanor Laing (now Baroness Laing of Elderslie)", "Dame Eleanor Laing"),
    ("Laing said X.", "Dame Eleanor Laing (now Baroness Laing of Elderslie)", "Laing"),
    ("Baroness Laing said X.", "Dame Eleanor Laing (now Baroness Laing of Elderslie)", "Baroness Laing"),
    ("Lincoln Jopp said X.", "Lincoln Jopp (Spelthorne) (Con)", "Lincoln Jopp"),
    ("Jopp said X.", "Lincoln Jopp (Spelthorne) (Con)", "Jopp"),
])  # fmt: skip
def test_speaker_lead_returns_exactly_the_words_that_name_the_speaker(text, speaker, lead):
    assert K.speaker_lead(text, speaker) == lead


@pytest.mark.parametrize("text, speaker", [
    ("She said X.", "Helen Morgan"),
    ("Reeves said X.", "Keir Mather"),
    ("Mather's view is X.", "Keir Mather"),
    ("Smith said X.", "Baroness Smith of Malvern"),
    ("Malvern said X.", "Baroness Smith of Malvern"),  # a territorial designation is not a name
    ("Speaker said X.", "Mr Speaker"),  # an office is not a surname
    ("Chancellor said X.", "The Lord Chancellor"),
    ("Housing said X.", "The Minister for Housing"),  # an office's last word is not a surname
    ("Keir Matherson said X.", "Keir Mather"),  # a full name inside a longer one is another person
    ("X happened.", "Keir Mather"),
    ("Mather said that X.", None),
    ("Mather said that X.", ""),
])  # fmt: skip
def test_speaker_lead_is_none_for_anything_else(text, speaker):
    assert K.speaker_lead(text, speaker) is None


# --- the verifier asks entailment about the substance, by the same definition ----------------------------------------


@pytest.mark.parametrize("text, speaker", [
    ("Baroness Smith said that the levy will rise.", "Baroness Smith of Malvern"),
    ("Mr Mather said that the levy will rise.", "Keir Mather"),
    ("Baroness Laing said that the levy will rise.", "Dame Eleanor Laing (now Baroness Laing of Elderslie)"),
])  # fmt: skip
def test_entailment_is_asked_about_the_substance_for_every_short_name(text, speaker):
    assert V.strip_attribution(text, speaker) == "The levy will rise."


def test_a_pronoun_lead_is_not_stripped_for_entailment_either():
    """One definition for both: what the renderer will not treat as the speaker, the verifier does not strip."""
    text = "She said that the levy will rise."
    assert V.strip_attribution(text, "Helen Morgan") == text


# --- the overview is shown --------------------------------------------------------------------------------------------


class _Hit:
    def __init__(self, chunk_id, debate_id):
        self.chunk_id, self.debate_id, self.rank, self.score = chunk_id, debate_id, 1, 1.0
        self.speech_id, self.sentence, self.passage = "s1", "Anchor sentence.", "Anchor sentence."
        self.seq, self.sent_idx = 1, 0
        self.date, self.house, self.party, self.office, self.heading = date(2026, 9, 15), "Commons", "Lab", None, "Fracking"
        self.speaker, self.url, self.column_ref, self.section, self.rerank_score = "A Member", None, None, "Chamber", 4.0
        self.sibling_chunk_ids = ()
        self.context_sentences = ((1, 0, "Anchor sentence.", None, chunk_id),)


class _Generated:
    def __init__(self, claims, generated=0, dropped=0):
        self.answer = type("A", (), {"claims": tuple(claims)})()
        self.claims_generated, self.claims_dropped = generated, dropped


def _overview(monkeypatch, per_debate, reply=None, score_batch=None, hits=None):
    """Run the overview route with one canned per-debate result per debate and an optional reduction reply."""
    results = iter(per_debate)

    async def fake_answer_question(question, deck, *a, **k):
        return next(results)

    async def fake_facets(conn, chunk_ids):
        return {"label": "passages found by search", "total": len(chunk_ids)}

    monkeypatch.setattr(RT.G, "answer_question", fake_answer_question)
    monkeypatch.setattr(RT, "facets", fake_facets)

    class _Reply:
        content = reply or '{"summary": []}'
        total_tokens = 0

        def cost_usd(self):
            return 0.0

    class _Client:
        async def chat(self, *a, **k):
            if reply is None:
                raise RuntimeError("no reduction in this test")
            return _Reply()

    hits = hits or [_Hit(101, "dA")]
    return run(RT.overview_route(None, "What did members say about fracking?", hits, _Client(),
                                 K.Provenance(**{**PROV, "route": "overview"}), score_batch=score_batch))


def test_an_overview_with_verified_claims_is_shown_to_its_reader(monkeypatch):
    first = claim("A Member said that fracking was paused.", "A Member", "c1", 101)
    overview = _overview(monkeypatch, [_Generated([first], generated=2, dropped=1)])
    text = overview.answer.rendered
    assert text, "an overview that answered must show its answer"
    assert "A Member said that fracking was paused." in text
    assert "This is not everything said on the subject." in text, "the coverage note is never dropped"
    assert overview.answer.abstention is None


def test_a_checked_summary_sentence_leads_the_overview(monkeypatch):
    first = claim("A Member said that fracking was paused.", "A Member", "c1", 101)
    reply = '{"summary": [{"text": "Members discussed the pause on fracking.", "from": ["c1"]}]}'

    def passes(premises, hypotheses):
        return [0.99 for _ in hypotheses]

    overview = _overview(monkeypatch, [_Generated([first])], reply=reply, score_batch=passes)
    assert overview.answer.rendered.startswith("Members discussed the pause on fracking.")
    assert overview.summary_claims == 1


def test_a_summary_sentence_that_failed_its_check_is_never_shown(monkeypatch):
    first = claim("A Member said that fracking was paused.", "A Member", "c1", 101)
    reply = '{"summary": [{"text": "The House agreed to ban fracking for ever.", "from": ["c1"]}]}'

    def fails(premises, hypotheses):
        return [0.01 for _ in hypotheses]

    overview = _overview(monkeypatch, [_Generated([first])], reply=reply, score_batch=fails)
    assert "A Member said that fracking was paused." in overview.answer.rendered, "the verified claims still show"
    assert "ban fracking" not in overview.answer.rendered
    assert overview.summary_dropped == 1


def test_with_no_verifier_no_summary_is_written_so_none_is_shown_unchecked(monkeypatch):
    """'s degradation drops paraphrases when no verifier is reachable; a summary sentence IS a paraphrase. It used to
    be written and kept unchecked when `score_batch` was None — invisible only because nothing rendered the
    overview.
    """
    first = claim("A Member said that fracking was paused.", "A Member", "c1", 101)
    calls: list[str] = []
    results = iter([_Generated([first])])

    async def fake_answer_question(question, deck, *a, **k):
        return next(results)

    async def fake_facets(conn, chunk_ids):
        return {"label": "passages found by search", "total": len(chunk_ids)}

    monkeypatch.setattr(RT.G, "answer_question", fake_answer_question)
    monkeypatch.setattr(RT, "facets", fake_facets)

    class _Client:
        async def chat(self, *a, **k):
            calls.append("reduce")
            raise RuntimeError("must not be called")

    overview = run(RT.overview_route(None, "What did members say about fracking?", [_Hit(101, "dA")], _Client(),
                                     K.Provenance(**{**PROV, "route": "overview"}), score_batch=None))  # fmt: skip
    assert calls == [], "no verifier, no reduction call"
    assert overview.summary_claims == 0
    assert overview.answer.rendered.startswith("A Member said that fracking was paused.")


def test_an_overview_counts_the_claims_its_debates_generated_and_dropped(monkeypatch):
    """The trace's claims columns were empty for every overview: they read the retrieval route's `Generated` only.
    """
    a = claim("A Member said that fracking was paused.", "A Member", "c1", 101)
    b = claim("A Member said that drilling stopped.", "A Member", "c2", 202)
    overview = _overview(monkeypatch, [_Generated([a], generated=3, dropped=2), _Generated([b], generated=1)],
                         hits=[_Hit(101, "dA"), _Hit(202, "dB")])  # fmt: skip
    assert (overview.claims_generated, overview.per_debate_claims, overview.claims_dropped) == (4, 2, 2)


# --- clarifications are shown, and the one exit every answer passes -------------------------------------------------


def test_a_clarify_route_answer_states_its_question_back():
    """"What did my MP say?" — the router clarifies; the reader used to be shown nothing."""
    answer = P.clarify_answer("What did my MP say about buses?", K.Provenance(**{**PROV, "route": "clarify"}),
                              why="the question does not say which member ('my MP')",
                              clarify=("which MP or peer do you mean — a name or a constituency?",))  # fmt: skip
    assert answer.rendered == ("The question does not say which member ('my MP'). Which MP or peer do you mean — a "
                               "name or a constituency?")
    assert answer.facets["clarify"] and answer.abstention is None


def test_a_common_phrase_clarification_states_its_question_back(monkeypatch):
    from answer import resolve as RS

    async def many(conn, phrase):
        return 467

    monkeypatch.setattr(RS, "phrase_speakers", many)
    answer = run(P.common_phrase_clarification(None, "Who said 'it is a pleasure to serve under your chairmanship'?",
                                               K.Provenance(**PROV)))  # fmt: skip
    assert answer.rendered.startswith("Those words were used by 467 different members in the record")
    assert answer.rendered.endswith("Say which debate, date or member you mean.")


def test_reader_ready_renders_an_answer_that_was_built_without_its_text():
    answer = K.Answer(question="q", provenance=K.Provenance(**PROV), claims=(claim("Mather said X.", "Keir Mather"),))
    assert answer.rendered == ""
    ready = P.reader_ready(answer)
    assert ready.rendered == "Keir Mather said X. [2026-09-10]"


def test_reader_ready_leaves_a_rendered_answer_and_a_decline_alone():
    shown = K.Answer(question="q", provenance=K.Provenance(**PROV), facets={"x": 1}, rendered="Already shown.")
    assert P.reader_ready(shown) is shown
    declined = K.Answer(question="q", provenance=K.Provenance(**PROV), abstention=K.Abstention.NOT_IN_RECORD,
                        abstention_detail="nothing found")  # fmt: skip
    assert P.reader_ready(declined) is declined


def test_an_answer_with_nothing_a_reader_could_be_shown_becomes_a_decline_in_the_systems_words():
    """Facets alone (counts, ids) are not an answer. Shown as nothing, a reader cannot tell "no answer" from "a blank
    page"; it is declined instead, and says why — never an empty success.
    """
    bare = K.Answer(question="q", provenance=K.Provenance(**PROV), facets={"total": 12})
    ready = P.reader_ready(bare)
    assert ready.abstention == K.Abstention.INSUFFICIENT_EVIDENCE
    assert ready.abstention_detail and not ready.facets and not ready.rendered


def test_the_claim_counts_come_from_the_overview_when_there_is_no_generator_reply():
    answer = K.Answer(question="q", provenance=K.Provenance(**PROV), facets={"x": 1}, rendered="shown")
    overview = RT.Overview(answer=answer, per_debate_claims=2, claims_generated=4, claims_dropped=2)
    asked = P.Asked(answer=answer, route="overview", overview=overview)
    assert asked.claim_counts == {"claims_generated": 4, "claims_verified": 2, "claims_dropped": 2,
                                  "claims_rejected": None}

    class _G:
        claims_generated, claims_verified, claims_dropped, claims_rejected = 3, 1, 2, 0

    assert P.Asked(answer=answer, route="retrieval", generated=_G()).claim_counts == {
        "claims_generated": 3, "claims_verified": 1, "claims_dropped": 2, "claims_rejected": 0}
    assert set(P.Asked(answer=answer, route="set").claim_counts.values()) == {None}


def test_every_exit_of_ask_passes_reader_ready():
    """Structural: `finish` is the one exit of `ask`, and it must hand every answer to `reader_ready` first."""
    import inspect

    source = inspect.getsource(P.ask)
    finish = source[source.index("async def finish("):]
    assert "reader_ready(" in finish[: finish.index("trace = TR.Trace(")]
