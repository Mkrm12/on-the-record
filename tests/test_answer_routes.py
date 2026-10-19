"""The set and overview routes and the corrective loop."""

from __future__ import annotations

import json
from datetime import date

import pytest

from answer import generate as G
from answer import roster as R
from answer import routes as RT
from answer.contract import Abstention, Answer, Provenance
from ingest.runtime import run as run_async

PROV = Provenance(route="set", corpus_version="v2-w0w5", retrieval_config_hash="e57c6cdfd8793d76",
                  recipe="r", prompt_version="p3", model_id="deepseek-flash")  # fmt: skip


def run(coro):
    return run_async(coro)


# --- the deterministic query rewrite --------------------------------------------------------------------


@pytest.mark.parametrize(("asked", "expected"), [
    ("did the tories U-turn on fracking", "Conservative"),
    ("what did the tory party say", "Conservative"),
    ("the lib dems on housing", "Liberal Democrat"),
    ("who slammed the minister", "criticised"),
    ("was the scheme axed", "abolished"),
    ("nhs funding", "National Health Service"),
])
def test_colloquial_words_are_rewritten_into_the_records_register(asked, expected):
    """The failure is lexical, not semantic: Hansard says "Conservative" and never says "U-turn", so the question
    retrieves nothing until its vocabulary matches the record's.
    """
    rewritten, changed = RT.rewrite_query(asked)
    assert expected in rewritten and changed


def test_a_question_already_in_the_records_register_is_not_rewritten():
    """An empty change means the caller must not spend a second retrieval on an identical query."""
    rewritten, changed = RT.rewrite_query("What did the Minister say about housing?")
    assert changed == "" and rewritten == "What did the Minister say about housing?"


def test_the_rewrite_is_a_search_query_not_prose():
    """"did the Conservative reversed its position on fracking" is ungrammatical and that is fine — it is fed to BM25
    and an embedder, neither of which parses syntax. A model-driven rewrite would fix the grammar and change the
    meaning, which is a different question answered without telling anyone.
    """
    rewritten, _ = RT.rewrite_query("did the tories U-turn on fracking")
    assert "Conservative" in rewritten and "reversed its position" in rewritten


def test_rewriting_is_case_insensitive_and_respects_word_boundaries():
    assert "Conservative" in RT.rewrite_query("The TORIES voted against")[0]
    # "category" contains "tory" and must not be touched.
    assert RT.rewrite_query("which category of spending")[1] == ""


# --- the set route --------------------------------------------------------------------------------------


def candidate(debate_id, score, **kw):
    return R.DebateCandidate(debate_id=debate_id, score=score, **kw)


class FakeConn:
    """Stands in for the roster's connection. `set_route` is given its candidates, so only the roster is faked."""

    def __init__(self, entries=(), unattributed=0):
        self.entries = entries
        self.unattributed = unattributed


async def fake_roster_with_gaps(conn, debate_ids, **kwargs):
    from answer.contract import Roster, RosterEntry

    entries = tuple(RosterEntry(mnis_id=i + 1, speaker_name=name, contributions=1)
                    for i, name in enumerate(conn.entries))  # fmt: skip
    return {
        "roster": Roster(debate_ids=tuple(debate_ids), confidence=kwargs.get("confidence_score", 1.0),
                         entries=entries),  # fmt: skip
        "unattributed": conn.unattributed, "unattributed_names": conn.unattributed,
        "complete": conn.unattributed == 0,
        "caveat": "" if conn.unattributed == 0 else "1 speaker(s) could not be matched; the list may be incomplete.",
    }


@pytest.fixture(autouse=True)
def _roster(monkeypatch):
    """The route's SQL is faked: the roster, the threads (each candidate its own record) and the openers (none)."""
    async def no_threads(conn, debate_ids):
        return {}

    async def no_opener(conn, debate_ids, **kwargs):
        return {"missing": "faked"}

    async def not_only_the_chair(conn, thread):
        return False

    monkeypatch.setattr(R, "roster_with_gaps", fake_roster_with_gaps)
    monkeypatch.setattr(RT.T, "threads_of", no_threads)
    monkeypatch.setattr(R, "thread_opener", no_opener)
    monkeypatch.setattr(RT, "_only_the_chair", not_only_the_chair)


def test_a_confident_debate_yields_the_sql_roster_with_no_model_call():
    """The point is not economy. A roster is a closed set of facts, and a generative step between the reader and a
    correct list can only make it worse.
    """
    conn = FakeConn(entries=("A Member", "B Member"))
    result = run(RT.set_route(conn, "who spoke on housing?", PROV,
                              candidates=[candidate("d1", 0.95), candidate("d2", 0.02)]))  # fmt: skip
    assert result.answer.roster is not None
    assert result.speakers == 2
    assert result.answer.abstention is None


def test_an_uncertain_debate_returns_clarify_with_three_candidates():
    """The worst failure available to this route is a correct roster for the wrong sitting: every name real, every
    count right, the answer about a different day.
    """
    conn = FakeConn(entries=("A Member",))
    result = run(RT.set_route(conn, "who spoke on arrangement of business?", PROV,
                              candidates=[candidate("d1", 0.5, title="Arrangement of Business", date=date(2026, 5, 1)),
                                          candidate("d2", 0.49, title="Arrangement of Business"),
                                          candidate("d3", 0.48, title="Arrangement of Business")]))  # fmt: skip
    assert result.answer.roster is None
    assert len(result.clarify) == 3
    assert result.answer.facets["clarify"]
    # The question back lists its options in the words the reader sees, each with its date or House
    assert result.answer.rendered.startswith("Several debates match this description: 'Arrangement of Business'")
    assert result.answer.rendered.count("Arrangement of Business") == 3 and "Say which one you mean" in result.answer.rendered


def test_no_candidate_abstains_rather_than_clarifying():
    """Nothing to choose between is a different outcome from a hard choice, and a clarify listing nothing would be an
    empty question put to the user.
    """
    result = run(RT.set_route(FakeConn(), "who spoke about zeppelins?", PROV, candidates=[]))
    assert result.answer.abstention is Abstention.NOT_IN_RECORD


def test_a_debate_with_no_contributions_abstains():
    result = run(RT.set_route(FakeConn(entries=()), "who spoke?", PROV, candidates=[candidate("d1", 0.99)]))
    assert result.answer.abstention is Abstention.NOT_IN_RECORD
    assert "records no contributions" in result.answer.abstention_detail


def test_an_incomplete_roster_carries_its_caveat():
    """A speaker the mapper could not resolve is still a speaker; presenting an incomplete list as complete is the
    failure the reader cannot see.
    """
    conn = FakeConn(entries=("A Member",), unattributed=1)
    result = run(RT.set_route(conn, "who spoke?", PROV, candidates=[candidate("d1", 0.99)]))
    assert "may be incomplete" in result.answer.coverage_note


def test_the_debate_confidence_travels_with_the_answer():
    conn = FakeConn(entries=("A Member",))
    result = run(RT.set_route(conn, "who spoke?", PROV, candidates=[candidate("d1", 0.8), candidate("d2", 0.2)]))
    assert result.answer.roster.confidence == pytest.approx(0.8)


# --- clustering -----------------------------------------------------------------------------------------


class Hit:
    def __init__(self, chunk_id, debate_id, rank, when=None, house="Commons", party="Lab"):
        self.chunk_id, self.debate_id, self.rank = chunk_id, debate_id, rank
        self.date, self.house, self.party = when, house, party
        self.sibling_chunk_ids = ()
        self.speech_id = f"s{chunk_id}"
        self.sentence = self.passage = "text"
        self.seq = self.sent_idx = 0
        self.speaker = "A Member"
        self.office = self.heading = self.url = None
        self.column_ref = None
        self.context_sentences = None
        self.score = 1.0 / rank
        self.section = "Commons Chamber"


def test_clustering_groups_by_debate_and_leads_with_the_best():
    hits = [Hit(1, "dA", 1), Hit(2, "dA", 4), Hit(3, "dB", 2), Hit(4, "dC", 3)]
    clustered = RT.cluster_by_debate(hits)
    assert clustered[0][0] == "dA"
    assert {d for d, _ in clustered} == {"dA", "dB", "dC"}
    assert len(clustered[0][1]) == 2, "both of dA's hits travel together"


def test_clustering_prefers_a_debate_that_adds_spread():
    """asks for relevance *and* spread. Eight passages from one afternoon is a summary of one afternoon presented as a
    summary of a subject.
    """
    hits = [
        Hit(1, "dA", 1, when=date(2026, 1, 1), house="Commons", party="Lab"),
        Hit(2, "dB", 2, when=date(2026, 1, 1), house="Commons", party="Lab"),  # adds nothing new
        Hit(3, "dC", 3, when=date(2026, 6, 1), house="Lords", party="Con"),  # adds date, house and party
    ]
    clustered = RT.cluster_by_debate(hits, limit=2)
    assert [d for d, _ in clustered] == ["dA", "dC"], "the more novel debate is taken over the closer-ranked one"


def test_clustering_respects_the_limit():
    hits = [Hit(i, f"d{i}", i) for i in range(1, 15)]
    assert len(RT.cluster_by_debate(hits, limit=8)) == 8


def test_clustering_nothing_yields_nothing():
    assert RT.cluster_by_debate([]) == []


# --- the reduction --------------------------------------------------------------------------------------


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    async def chat(self, prompt, *, system="", max_tokens=0, temperature=0.0, json_object=False):
        self.calls += 1
        body = self.replies.pop(0) if self.replies else {}
        if isinstance(body, Exception):
            raise body

        class Reply:
            content = body if isinstance(body, str) else json.dumps(body)
            model = "deepseek-flash"
            fingerprint = "fp"
            finish_reason = "stop"
            prompt_tokens, completion_tokens, cached_tokens, total_tokens = 500, 80, 0, 580

            def cost_usd(self, model=None):
                return 0.0002

        return Reply()


def test_the_reduction_prompt_is_given_verified_claims_not_the_corpus():
    """'s rule, and the reason an overview can be trusted: a summary sentence is checked against sentences that already
    passed verification, so it cannot introduce anything the record did not support.
    """
    assert "already been checked against the parliamentary record" in RT.REDUCE_PROMPT
    assert "Do not add anything they do not say" in RT.REDUCE_PROMPT
    assert "never to a party" in RT.REDUCE_PROMPT


def test_the_reduction_prompt_forbids_new_names_numbers_and_dates():
    for forbidden in ("name", "number", "date"):
        assert forbidden in RT.REDUCE_PROMPT


# --- facets ---------------------------------------------------------------------------------------------


def test_facets_never_claim_to_be_everything_said():
    """A facet count over a retrieved pool is a fact about the search. Presenting it as a fact about Parliament is the
    most plausible-looking lie this system could tell, because the number is real and the noun is wrong.
    """
    empty = run(RT.facets(None, []))
    assert "found by search" in empty["label"]
    assert empty["total"] == 0


# --- the corrective loop --------------------------------------------------------------------------------


class Deck:
    def __init__(self, n):
        self.cards = [object()] * n


def generated(abstention=None):
    answer = Answer(question="q", provenance=PROV, abstention=abstention,
                    abstention_detail="x" if abstention else "",
                    facets={} if abstention else {"ok": True})  # fmt: skip
    return G.Generated(answer=answer)


def loop(question, outcomes, max_rewrites=RT.MAX_REWRITES, budget=None):
    """`outcomes` is one Generated per attempt, in order."""
    seen: list[str] = []

    async def search(query):
        seen.append(query)
        return Deck(3)

    async def answer_fn(query, deck):
        return outcomes[min(len(seen) - 1, len(outcomes) - 1)]

    result = run(RT.with_corrections(search, answer_fn, question,
                                     budget=budget or G.Budget(), max_rewrites=max_rewrites))  # fmt: skip
    return result, seen


def test_a_successful_first_attempt_is_never_rewritten():
    result, seen = loop("did the tories U-turn", [generated()])
    assert len(seen) == 1 and result.rewrites == 0


def test_a_missing_evidence_abstention_triggers_one_rewrite():
    result, seen = loop("did the tories U-turn on fracking",
                        [generated(Abstention.NOT_IN_RECORD), generated()])  # fmt: skip
    assert len(seen) == 2
    assert "Conservative" in seen[1]
    assert result.rewrites == 1
    assert result.generated.answer.abstention is None


def test_an_ambiguous_question_is_not_rewritten():
    """Rewriting cannot fix ambiguity and would only obscure it."""
    result, seen = loop("what did the Prime Minister say", [generated(Abstention.AMBIGUOUS)])
    assert len(seen) == 1 and result.rewrites == 0


def test_a_false_premise_is_not_rewritten():
    result, seen = loop("when did the tories abolish the Lords", [generated(Abstention.FALSE_PREMISE)])
    assert len(seen) == 1 and result.rewrites == 0


def test_a_question_with_nothing_to_rewrite_is_not_retried():
    """No change means an identical query, and a second identical retrieval is a spend with a known answer."""
    result, seen = loop("what did the Minister say about housing",
                        [generated(Abstention.NOT_IN_RECORD)])  # fmt: skip
    assert len(seen) == 1 and result.rewrites == 0


def test_a_rewrite_that_also_fails_leaves_the_original_answer_standing():
    """The reader is told why the question *as asked* could not be answered, not why a reworded one could not — which
    would be an answer to a question nobody asked.
    """
    result, seen = loop("did the tories U-turn on fracking",
                        [generated(Abstention.NOT_IN_RECORD), generated(Abstention.NOT_IN_RECORD)])  # fmt: skip
    assert result.generated.answer.abstention is Abstention.NOT_IN_RECORD
    assert len(result.attempts) == 2, "the attempt is recorded even though its answer was discarded"


def test_the_rewrite_cap_is_honoured():
    assert RT.MAX_REWRITES == 2
    result, seen = loop("did the tories U-turn on fracking",
                        [generated(Abstention.NOT_IN_RECORD)] * 4, max_rewrites=1)  # fmt: skip
    assert result.rewrites <= 1 and len(seen) <= 2


def test_an_exhausted_budget_stops_the_loop():
    budget = G.Budget(seconds=0.0)
    result, seen = loop("did the tories U-turn on fracking",
                        [generated(Abstention.NOT_IN_RECORD), generated()], budget=budget)  # fmt: skip
    assert len(seen) == 1, "no rewrite is attempted once the wall clock is gone"


def test_every_attempt_is_recorded_for_the_trace():
    """A guard is only a guard if its use is visible afterwards."""
    result, _ = loop("did the tories U-turn on fracking",
                     [generated(Abstention.NOT_IN_RECORD), generated()])  # fmt: skip
    assert [a.outcome for a in result.attempts] == ["not_in_record", "answer"]
    assert result.attempts[1].what_changed
    assert result.attempts[1].rewritten_from == "did the tories U-turn on fracking"
