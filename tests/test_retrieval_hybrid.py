"""The hybrid search engine against a scratch database whose every row and vector is placed by hand."""

import asyncio

import pytest
from corpus import Corpus, vec
from pgvector import Vector

from app.db import make_pool
from ingest.runtime import run
from retrieval.gpu_client import GpuBusy, GpuUnavailable
from retrieval.hybrid import MAX_TOP_K, HybridSearcher, RetrievalError, SearchConfig

QUERY_VEC = vec((0, 1))


class FakeGpu:
    """Stands in for GpuClient: returns the vectors and scores the test chooses and records every call."""

    def __init__(self, vectors=None, scores=None, embed_error=None, rerank_error=None):
        self.vectors, self.scores = vectors or {}, scores
        self.embed_error, self.rerank_error = embed_error, rerank_error
        self.embed_calls, self.rerank_calls = [], []

    async def embed_query(self, text):
        self.embed_calls.append(text)
        if self.embed_error:
            raise self.embed_error
        return Vector(self.vectors.get(text, QUERY_VEC))

    async def rerank(self, query, candidates):
        self.rerank_calls.append((query, list(candidates)))
        if self.rerank_error:
            raise self.rerank_error
        if callable(self.scores):
            return self.scores(query, candidates)
        return list(self.scores) if self.scores is not None else [0.0] * len(candidates)


def cfg(**kw):
    """Exact-arithmetic defaults for the tests: no keyword threshold, no reranker, no expansion."""
    base = dict(sparse_coverage=0.0, rerank="off", expand_radius=0)
    base.update(kw)
    return SearchConfig(**base)


def go(dsn, query, *, gpu=None, config=None, top_k=10, pool_size=1, after=None, prepare=None):
    """Run one search on a fresh pool. `after(pool)` runs with the same pool afterwards, for checks on the connection.
    """
    gpu = gpu or FakeGpu()

    async def main():
        pool = make_pool(dsn, min_size=1, max_size=pool_size, served=False)
        await pool.open(wait=True, timeout=10)
        try:
            searcher = HybridSearcher(pool, gpu, config or cfg())
            if prepare:
                await prepare(searcher)
            result = await searcher.search(query, top_k=top_k)
            return result, (await after(pool) if after else None)
        finally:
            await pool.close()

    return run(main())


def housing(scratch):
    """Five one-sentence speeches with known dense ranks (c1..c4 along axis 0 mixed with axis 1) and known keyword
    scores.
    """
    c = Corpus(scratch)
    c.debate("d1", "Housing")
    ids = {}
    for name, text, v in [
        ("c1", "Housing policy reform is needed.", vec((0, 1))),
        ("c2", "The policy on rents was clear.", vec((0, 1), (1, 1))),
        ("c3", "Housing benefit rules changed.", vec((0, 1), (1, 2))),
        ("c4", "An unrelated weather report today.", vec((0, 1), (1, 4))),
        ("c5", "Housing policy for housing.", None),  # no vector of its own: never a dense hit
    ]:
        ids[name] = c.speech("d1", f"s_{name}", [text], vectors=[v])[0]
    return ids


# --------------------------------------------------------------------------- RRF arithmetic and the join
# ---------------------------------------------------------------------------


def test_rrf_scores_are_exactly_the_reciprocal_rank_sums_with_k_60(scratch, scratch_dsn):
    ids = housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", config=cfg(dense_limit=4, sparse_limit=10, speaker=False))
    by_id = {h.chunk_id: h for h in result.hits}
    # dense: c1, c2, c3, c4 (by vector distance). keyword: c5 (shortest, both terms), c1, c2, c3 (BM25 length norm)
    assert {n: by_id[i].legs["dense"] for n, i in ids.items() if by_id[i].legs["dense"]} == {"c1": 1, "c2": 2, "c3": 3, "c4": 4}
    assert {n: by_id[i].legs["sparse"] for n, i in ids.items() if by_id[i].legs["sparse"]} == {"c5": 1, "c1": 2, "c2": 3, "c3": 4}
    expected = {"c1": 1 / 61 + 1 / 62, "c2": 1 / 62 + 1 / 63, "c3": 1 / 63 + 1 / 64, "c5": 1 / 61, "c4": 1 / 64}
    assert {n: by_id[i].rrf for n, i in ids.items()} == pytest.approx(expected)
    assert [h.chunk_id for h in result.hits] == [ids[n] for n in ("c1", "c2", "c3", "c5", "c4")]  # best fused first


def test_weights_scale_each_legs_contribution(scratch, scratch_dsn):
    ids = housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", config=cfg(dense_limit=4, sparse_limit=10, speaker=False, weights=(2.0, 0.5, 1.0)))
    by_id = {h.chunk_id: h for h in result.hits}
    assert by_id[ids["c1"]].rrf == pytest.approx(2 / 61 + 0.5 / 62)
    assert by_id[ids["c5"]].rrf == pytest.approx(0.5 / 61)


def test_k_is_configurable(scratch, scratch_dsn):
    ids = housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", config=cfg(dense_limit=4, sparse_limit=10, speaker=False, rrf_k=10))
    assert {h.chunk_id: h.rrf for h in result.hits}[ids["c1"]] == pytest.approx(1 / 11 + 1 / 12)


def test_the_join_is_on_the_chunk_id_so_many_chunks_in_one_debate_are_not_multiplied(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "One big debate")
    for i in range(8):  # eight chunks, all in the SAME debate, all hits in both legs
        c.speech("d1", f"s{i}", [f"Housing policy statement number {i}."], vectors=[vec((0, 1), (1, 0.1 * (i + 1)))])
    result, _ = go(scratch_dsn, "housing policy", config=cfg(speaker=False), top_k=50)
    ids = [h.chunk_id for h in result.hits]
    assert len(ids) == len(set(ids)) == 8  # not 8 x 8
    assert all(h.rrf <= 2 / 61 + 1e-12 for h in result.hits)  # no rank inflation


def test_each_leg_is_limited_before_the_join_so_only_the_top_of_each_leg_is_fused(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Housing")
    for i in range(10):
        c.speech("d1", f"s{i}", [f"Housing policy item {i}."], vectors=[vec((0, 1), (1, 0.05 * (i + 1)))])
    result, _ = go(scratch_dsn, "housing policy", config=cfg(dense_limit=3, sparse_limit=2, speaker=False), top_k=50)
    assert sum(h.legs["dense"] is not None for h in result.hits) == 3
    assert sum(h.legs["sparse"] is not None for h in result.hits) == 2
    assert 3 <= len(result.hits) <= 5  # the union of the two limited lists, nothing more
    assert result.diagnostics["legs"] == {"dense": 3, "sparse": 2, "speaker": 0}


def test_the_fused_limit_caps_the_candidate_list(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Housing")
    ids = [c.speech("d1", f"s{i}", [f"Housing policy item {i}."], vectors=[vec((0, 1), (1, 0.05 * (i + 1)))])[0] for i in range(12)]
    result, _ = go(scratch_dsn, "housing policy", config=cfg(fused_limit=4, speaker=False), top_k=50)
    assert len(result.hits) == 4 and result.diagnostics["fused"] == 4
    assert [h.chunk_id for h in result.hits] == ids[:4]  # the four BEST of twelve (closest, and first by id), best first


def test_ties_are_broken_by_chunk_id_so_results_are_deterministic(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Housing")
    first = c.speech("d1", "s1", ["Alpha bravo."], vectors=[vec((0, 1))])[0]
    second = c.speech("d1", "s2", ["Alpha charlie."], vectors=[vec((0, 1))])[0]  # a different passage at exactly the same distance
    a, _ = go(scratch_dsn, "zulu", config=cfg(speaker=False, sparse=False))
    b, _ = go(scratch_dsn, "zulu", config=cfg(speaker=False, sparse=False))
    assert [h.chunk_id for h in a.hits] == [h.chunk_id for h in b.hits] == [first, second]  # equal distance: the lower id first


# --------------------------------------------------------------------------- both legs exclude boilerplate; a leg with
# nothing to say changes nothing ---------------------------------------------------------------------------


def test_boilerplate_is_excluded_from_both_legs_even_if_it_somehow_holds_a_vector(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Housing")
    good = c.speech("d1", "s1", ["Housing policy is important."], vectors=[vec((0, 1), (1, 1))])[0]
    (bp,) = c.speech("d1", "s2", ["Housing policy question put and agreed to."], boilerplate=[True])
    scratch.execute("UPDATE chunk SET embedding = %s WHERE id = %s", (Vector(QUERY_VEC), bp))  # a stray vector: the worst case
    result, _ = go(scratch_dsn, "housing policy", config=cfg(speaker=False))
    assert [h.chunk_id for h in result.hits] == [good]


def test_no_keyword_matches_but_a_strong_dense_match_still_returns_the_dense_hit(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Weather")
    (hit,) = c.speech("d1", "s1", ["Rainfall totals were high."], vectors=[vec((0, 1))])
    result, _ = go(scratch_dsn, "zebra quokka", config=cfg(speaker=False))  # words that occur nowhere
    assert [h.chunk_id for h in result.hits] == [hit]
    assert result.hits[0].legs == {"dense": 1, "sparse": None, "speaker": None}
    assert result.diagnostics["terms"] == [] and result.diagnostics["legs"]["sparse"] == 0


def test_a_query_of_only_stopwords_has_no_keyword_terms_and_still_searches_densely(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Housing")
    (hit,) = c.speech("d1", "s1", ["Something about houses."], vectors=[vec((0, 1))])
    result, _ = go(scratch_dsn, "what is the", config=cfg(speaker=False))
    assert [h.chunk_id for h in result.hits] == [hit] and result.diagnostics["terms"] == []


def test_a_strong_keyword_match_with_no_dense_neighbour_is_still_found(scratch, scratch_dsn):
    ids = housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", config=cfg(dense=False, speaker=False))
    assert result.hits[0].chunk_id == ids["c5"] and all(h.legs["dense"] is None for h in result.hits)


def test_the_keyword_threshold_removes_weak_matches_but_keeps_covering_ones(scratch, scratch_dsn):
    ids = housing(scratch)  # "housing policy": c5 and c1 cover both terms, c2 and c3 only one (half the IDF mass)
    loose, _ = go(scratch_dsn, "housing policy", config=cfg(dense=False, speaker=False, sparse_coverage=0.0))
    strict, _ = go(scratch_dsn, "housing policy", config=cfg(dense=False, speaker=False, sparse_coverage=0.6))
    assert {h.chunk_id for h in loose.hits} == {ids["c1"], ids["c2"], ids["c3"], ids["c5"]}
    assert {h.chunk_id for h in strict.hits} == {ids["c1"], ids["c5"]}  # only sentences covering >= 60% of the query's weight


def test_a_rare_term_outweighs_common_ones(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Mixed")
    for i in range(30):
        c.speech("d1", f"common{i}", ["The government said the minister spoke."])
    rare = c.speech("d1", "rare", ["Seaton Carew beach."])[0]
    both = c.speech("d1", "few", ["The government said."])[0]
    result, _ = go(scratch_dsn, "government minister Seaton", config=cfg(dense=False, speaker=False))
    assert result.hits[0].chunk_id == rare  # one match on "seaton" beats matches on two common words
    assert both in [h.chunk_id for h in result.hits]


# --------------------------------------------------------------------------- GUCs: SET LOCAL inside the transaction,
# nothing leaks to the pool ---------------------------------------------------------------------------


def test_the_hnsw_settings_apply_during_the_search_and_are_gone_afterwards(scratch, scratch_dsn):
    housing(scratch)

    async def after(pool):
        async with pool.connection() as conn:  # the pool has one connection, so this is the very session the search used
            ef = (await (await conn.execute("SHOW hnsw.ef_search")).fetchone())[0]
            scan = (await (await conn.execute("SHOW hnsw.iterative_scan")).fetchone())[0]
            timeout = (await (await conn.execute("SHOW statement_timeout")).fetchone())[0]
            await conn.rollback()
            return ef, scan, timeout

    result, leaked = go(scratch_dsn, "housing policy", config=cfg(ef_search=77, statement_timeout_ms=4321), after=after)
    assert result.diagnostics["gucs"] == {"ef_search": "77", "iterative_scan": "strict_order", "statement_timeout": "4321ms"}
    assert leaked == ("40", "off", "0")


def test_the_default_settings_are_the_ones_the_brief_asks_for():
    config = SearchConfig()
    assert config.ef_search == 100 and config.rrf_k == 60 and config.dense_limit == config.sparse_limit == config.speaker_limit == 50


@pytest.mark.parametrize("bad", [0, -5, 1001, "100; DROP TABLE chunk", 1.5, None, True])
def test_an_out_of_range_or_non_integer_ef_search_is_refused_before_any_sql(bad):
    with pytest.raises((ValueError, TypeError)):
        SearchConfig(ef_search=bad)  # ef_search is the one value formatted into SET LOCAL, so it must be a plain integer


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rerank="shuffle"), dict(weights=(1.0, 1.0)), dict(weights=(1.0, -1.0, 1.0)), dict(weights=(1.0, float("nan"), 1.0)),
        dict(sparse_coverage=1.5), dict(sparse_coverage=-0.1), dict(rrf_k=0), dict(pool=0), dict(fused_limit=0), dict(dense_limit=-1),
        dict(statement_timeout_ms=-1), dict(expand_radius=-1), dict(max_terms=0),
    ],
)
def test_nonsense_configuration_is_refused_when_it_is_built_not_when_a_search_runs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SearchConfig(**kwargs)


def test_the_boundary_values_are_accepted():
    SearchConfig(ef_search=1, sparse_coverage=0.0, expand_radius=0, dense_limit=0, weights=(0, 0, 0))
    SearchConfig(ef_search=1000, sparse_coverage=1.0)


# --------------------------------------------------------------------------- speaker injection
# ---------------------------------------------------------------------------


def speaker_corpus(scratch):
    c = Corpus(scratch)
    c.member(10, "Alison McGovern", list_as="McGovern, Alison", party="Labour")
    c.debate("d1", "Housing")
    for i in range(12):  # distractors that beat her sentence on the dense leg
        c.speech("d1", f"other{i}", [f"Weather report number {i}."], vectors=[vec((0, 1), (2, 0.01 * (i + 1)))])
    hers = c.speech("d1", "hers", ["We must fund the sector properly."], mnis_id=10, speaker_name="Alison McGovern", party="Labour", vectors=[vec((5, 1))])[0]
    return hers


SPEAKER_QUERY_VEC = vec((0, 1), (5, 0.6))


def test_a_named_speakers_speech_is_found_through_the_speaker_leg_though_dense_and_keyword_miss_it(scratch, scratch_dsn):
    hers = speaker_corpus(scratch)
    q = "What did Alison McGovern say about elderly support?"
    result, _ = go(scratch_dsn, q, gpu=FakeGpu(vectors={q: SPEAKER_QUERY_VEC}), config=cfg(dense_limit=5), top_k=50)
    hit = next(h for h in result.hits if h.chunk_id == hers)
    assert hit.legs == {"dense": None, "sparse": None, "speaker": 1}
    assert hit.rrf == pytest.approx(1 / 61)  # a speaker-only hit scores exactly one leg's reciprocal rank
    assert result.diagnostics["speakers"]["member_ids"] == [10] and result.diagnostics["speakers"]["mentions"] == ["Alison McGovern"]


def test_a_member_who_exists_but_has_never_spoken_is_not_resolved(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.member(10, "Alison McGovern", list_as="McGovern, Alison")  # a voter only: no speeches
    c.debate("d1", "Housing")
    c.speech("d1", "s1", ["Something else entirely."], vectors=[vec((0, 1))])
    result, _ = go(scratch_dsn, "What did Alison McGovern say about housing?", config=cfg())
    assert result.diagnostics["speakers"]["member_ids"] == []  # nothing of hers to find, so no speaker leg and no false claim


def test_without_her_name_the_same_sentence_is_not_found(scratch, scratch_dsn):
    hers = speaker_corpus(scratch)
    q = "What did the member say about elderly support?"
    result, _ = go(scratch_dsn, q, gpu=FakeGpu(vectors={q: SPEAKER_QUERY_VEC}), config=cfg(dense_limit=5), top_k=50)
    assert hers not in [h.chunk_id for h in result.hits] and result.diagnostics["speakers"]["member_ids"] == []


def test_naming_a_speaker_boosts_but_does_not_exclude_other_speakers(scratch, scratch_dsn):
    speaker_corpus(scratch)
    q = "What did Alison McGovern say about elderly support?"
    result, _ = go(scratch_dsn, q, gpu=FakeGpu(vectors={q: SPEAKER_QUERY_VEC}), config=cfg(dense_limit=5), top_k=50)
    assert len(result.hits) > 1 and any(h.speaker != "Alison McGovern" for h in result.hits)  # a name used as a subject still works


def test_the_speaker_leg_takes_each_speeches_best_sentence_so_a_far_off_speech_still_enters(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.member(10, "Alison McGovern", list_as="McGovern, Alison")
    c.debate("d1", "Business of the House")
    for i in range(6):  # six of her speeches; only the last has the sentence far from the query
        c.speech("d1", f"m{i}", [f"Local matter number {i}."], mnis_id=10, speaker_name="Alison McGovern", vectors=[vec((0, 1), (3, 0.1 * (i + 1)))])
    far = c.speech("d1", "far", ["Dentists in Cornwall are scarce."], mnis_id=10, speaker_name="Alison McGovern", vectors=[vec((9, 1))])[0]
    q = "What did Alison McGovern raise?"
    result, _ = go(scratch_dsn, q, config=cfg(dense=False, sparse=False, speaker_limit=50), top_k=50)
    assert far in [h.chunk_id for h in result.hits] and len(result.hits) == 7  # one candidate per speech of hers


def test_the_speaker_is_not_used_as_a_keyword(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.member(20, "Rowan Moon", list_as="Moon, Rowan")
    c.debate("d1", "Space")
    c.speech("d1", "s1", ["The moon landing was historic."])
    c.speech("d1", "s2", ["Tax rises hit families."])
    c.speech("d1", "s3", ["Nothing to see here."], mnis_id=20, speaker_name="Rowan Moon")  # only members who have spoken are resolvable
    result, _ = go(scratch_dsn, "What did Rowan Moon say about tax?", config=cfg())
    assert result.diagnostics["terms"] == ["tax"]  # "moon" belongs to the name, not the topic
    assert result.diagnostics["keyword_text"] == "What did say about tax?"


def test_the_rerank_text_carries_the_speaker_party_and_debate(scratch, scratch_dsn):
    speaker_corpus(scratch)
    q = "What did Alison McGovern say about elderly support?"
    gpu = FakeGpu(vectors={q: SPEAKER_QUERY_VEC})
    go(scratch_dsn, q, gpu=gpu, config=cfg(dense_limit=5, rerank="blend"), top_k=50)
    (_, texts), = gpu.rerank_calls
    assert "Alison McGovern (Labour), in the debate 'Housing': We must fund the sector properly." in texts


def test_a_name_that_is_not_in_the_corpus_resolves_to_nobody(scratch, scratch_dsn):
    speaker_corpus(scratch)
    result, _ = go(scratch_dsn, "What did Zaphod Beeblebrox say?", config=cfg())
    assert result.diagnostics["speakers"]["member_ids"] == []


def test_the_speaker_leg_can_be_switched_off(scratch, scratch_dsn):
    hers = speaker_corpus(scratch)
    q = "What did Alison McGovern say about elderly support?"
    result, _ = go(scratch_dsn, q, gpu=FakeGpu(vectors={q: SPEAKER_QUERY_VEC}), config=cfg(dense_limit=5, speaker=False), top_k=50)
    assert hers not in [h.chunk_id for h in result.hits] and result.diagnostics["speakers"]["member_ids"] == []


def test_the_speaker_leg_counts_speeches_not_sentences_and_honours_its_limit(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.member(10, "Alison McGovern", list_as="McGovern, Alison")
    c.debate("d1", "Business of the House")
    for i in range(6):  # six speeches of three sentences each
        c.speech("d1", f"m{i}", [f"Point {i} one.", f"Point {i} two.", f"Point {i} three."], mnis_id=10, speaker_name="Alison McGovern",
                 vectors=[vec((0, 1), (3, 0.1 * (i + 1)))] * 3)
    q = "What did Alison McGovern raise?"
    everything, _ = go(scratch_dsn, q, config=cfg(dense=False, sparse=False, speaker_limit=50), top_k=50)
    limited, _ = go(scratch_dsn, q, config=cfg(dense=False, sparse=False, speaker_limit=3), top_k=50)
    assert everything.diagnostics["legs"]["speaker"] == 6  # one sentence per speech, not eighteen
    assert limited.diagnostics["legs"]["speaker"] == 3 and len(limited.hits) == 3


def test_the_speaker_leg_never_returns_boilerplate(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.member(10, "Alison McGovern", list_as="McGovern, Alison")
    c.debate("d1", "Business of the House")
    good = c.speech("d1", "good", ["A real remark."], mnis_id=10, speaker_name="Alison McGovern", vectors=[vec((0, 1), (1, 1))])[0]
    (bp,) = c.speech("d1", "bp", ["Question put and agreed to."], mnis_id=10, speaker_name="Alison McGovern", boilerplate=[True])
    scratch.execute("UPDATE chunk SET embedding = %s WHERE id = %s", (Vector(QUERY_VEC), bp))  # a stray vector that would win
    result, _ = go(scratch_dsn, "What did Alison McGovern raise?", config=cfg(dense=False, sparse=False), top_k=50)
    assert [h.chunk_id for h in result.hits] == [good]


def test_the_keyword_statistics_ignore_boilerplate(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    c.speech("d1", "s1", ["Ordinary remark."])
    c.speech("d1", "s2", ["Zebra procedural words."], boilerplate=[True])
    result, _ = go(scratch_dsn, "zebra", config=cfg(dense=False, speaker=False))
    assert result.diagnostics["terms"] == [] and result.hits == []  # "zebra" occurs only in boilerplate, which is never searched


def test_a_single_candidate_is_not_sent_to_the_reranker(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    c.speech("d1", "s1", ["The quokka act."], vectors=[vec((0, 1))])
    gpu = FakeGpu()
    result, _ = go(scratch_dsn, "quokka", gpu=gpu, config=cfg(rerank="blend", speaker=False, dense_limit=1))
    assert len(result.hits) == 1 and gpu.rerank_calls == [] and result.hits[0].rerank_score is None


def test_an_identical_passage_in_another_speech_is_one_candidate_that_remembers_both_speeches(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    first = c.speech("d1", "s1", ["The quokka act."], vectors=[vec((0, 1))])[0]
    c.speech("d1", "s2", ["The quokka act."], vectors=[vec((0, 1))])
    result, _ = go(scratch_dsn, "quokka", config=cfg(speaker=False), top_k=10)
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.chunk_id == first and hit.speech_id == "s1" and hit.also_in_speeches == ("s2",) and len(hit.sibling_chunk_ids) == 1


# --------------------------------------------------------------------------- sibling collapse before the reranker
# ---------------------------------------------------------------------------


def sibling_corpus(scratch):
    c = Corpus(scratch)
    c.debate("d1", "Budget")
    sentences = [f"Filler sentence {i}." for i in range(8)]
    sentences[2], sentences[3], sentences[4] = "The budget deficit is large.", "The budget deficit worries us.", "The budget deficit must fall."
    vectors = [None, None, vec((0, 1)), vec((0, 1), (1, 0.1)), vec((0, 1), (1, 0.2)), None, None, None]
    sib = c.speech("d1", "sib", sentences, vectors=vectors)
    other = c.speech("d1", "other", ["The budget deficit was discussed elsewhere."], vectors=[vec((0, 1), (1, 0.5))])[0]
    return sib, other


def test_adjacent_hits_of_one_speech_become_one_candidate_and_one_reranker_slot(scratch, scratch_dsn):
    sib, other = sibling_corpus(scratch)
    gpu = FakeGpu()
    result, _ = go(scratch_dsn, "budget deficit", gpu=gpu, config=cfg(rerank="blend", speaker=False, dense_limit=4))
    assert len(result.hits) == 2  # the three neighbouring sentences are one result, the other speech another
    first = result.hits[0]
    assert first.chunk_id == sib[2] and set(first.sibling_chunk_ids) == {sib[3], sib[4]}
    (_, texts), = gpu.rerank_calls
    assert len(texts) == 2 and result.diagnostics["clusters"] == 2 and result.diagnostics["fused"] == 4  # 4 chunks in, 2 slots used


def test_siblings_far_apart_in_one_speech_stay_separate(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Budget")
    sentences = [f"Filler sentence {i}." for i in range(14)]
    sentences[1], sentences[9] = "The budget deficit is large.", "The budget deficit must fall."
    vectors = [None] * 14
    vectors[1], vectors[9] = vec((0, 1)), vec((0, 1), (1, 0.1))
    c.speech("d1", "far", sentences, vectors=vectors)
    result, _ = go(scratch_dsn, "budget deficit", config=cfg(speaker=False, dense_limit=2))
    assert len(result.hits) == 2 and all(not h.sibling_chunk_ids for h in result.hits)


def test_the_pool_counts_clusters_so_collapsing_frees_slots_for_other_speeches(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Budget")
    sentences = [f"Filler {i}." for i in range(6)]
    sentences[1:5] = [f"The budget deficit point {i}." for i in range(4)]
    c.speech("d1", "big", sentences, vectors=[None] + [vec((0, 1), (1, 0.01 * (i + 1))) for i in range(4)] + [None])
    for i in range(3):
        c.speech("d1", f"o{i}", [f"The budget deficit remark {i}."], vectors=[vec((0, 1), (1, 0.5 + 0.1 * i))])
    gpu = FakeGpu()
    result, _ = go(scratch_dsn, "budget deficit", gpu=gpu, config=cfg(rerank="blend", pool=4, speaker=False))
    (_, texts), = gpu.rerank_calls
    assert len(texts) == 4  # four candidates: the collapsed speech and all three others (not four sentences of one speech)


# --------------------------------------------------------------------------- reranking: usage, blending and failure
# ---------------------------------------------------------------------------


def test_blend_combines_the_fused_order_with_the_reranker_order(scratch, scratch_dsn):
    ids = housing(scratch)
    scores = lambda q, texts: [1.0 if "rents" in t else 0.0 for t in texts]  # the reranker loves c2 alone
    gpu = FakeGpu(scores=scores)
    result, _ = go(scratch_dsn, "housing policy", gpu=gpu, config=cfg(rerank="blend", dense_limit=4, sparse_limit=10, speaker=False))
    order = [h.chunk_id for h in result.hits]
    assert order[0] == ids["c1"] and order.index(ids["c2"]) < 3  # c2 was 2nd fused and 1st for the reranker: it stays high
    top = next(h for h in result.hits if h.chunk_id == ids["c2"])
    assert top.rerank_score == 1.0 and top.score == pytest.approx(1 / 62 + 1 / 61)


def test_replace_ranks_purely_by_the_reranker(scratch, scratch_dsn):
    ids = housing(scratch)
    scores = lambda q, texts: [5.0 if "weather" in t else 0.0 for t in texts]
    result, _ = go(scratch_dsn, "housing policy", gpu=FakeGpu(scores=scores), config=cfg(rerank="replace", dense_limit=4, sparse_limit=10, speaker=False))
    assert result.hits[0].chunk_id == ids["c4"] and result.hits[0].score == 5.0


def test_off_never_calls_the_reranker(scratch, scratch_dsn):
    housing(scratch)
    gpu = FakeGpu()
    result, _ = go(scratch_dsn, "housing policy", gpu=gpu, config=cfg(rerank="off", speaker=False))
    assert gpu.rerank_calls == [] and all(h.rerank_score is None for h in result.hits)


def test_the_reranker_reads_sentence_with_header_never_the_window(scratch, scratch_dsn):
    sibling_corpus(scratch)
    gpu = FakeGpu()
    go(scratch_dsn, "budget deficit", gpu=gpu, config=cfg(rerank="blend", speaker=False, dense_limit=4))
    (_, texts), = gpu.rerank_calls
    assert texts[0] == "In the debate 'Budget': The budget deficit is large."
    assert all("Filler sentence" not in t for t in texts)  # the window around it would have included these


def test_the_reranker_is_asked_about_the_cleaned_query(scratch, scratch_dsn):
    housing(scratch)
    gpu = FakeGpu()
    go(scratch_dsn, "  housing \x00 policy \n", gpu=gpu, config=cfg(rerank="blend", speaker=False))
    assert gpu.rerank_calls[0][0] == "housing policy" and gpu.embed_calls == ["housing policy"]


@pytest.mark.parametrize("error", [GpuUnavailable("down"), GpuBusy("busy")])
def test_a_failing_reranker_degrades_to_the_fused_order_with_a_warning(scratch, scratch_dsn, error):
    ids = housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", gpu=FakeGpu(rerank_error=error), config=cfg(rerank="blend", dense_limit=4, sparse_limit=10, speaker=False))
    assert result.hits[0].chunk_id == ids["c1"] and any("rerank" in w.lower() for w in result.warnings)
    assert all(h.rerank_score is None for h in result.hits)


def test_partial_scores_are_tolerated(scratch, scratch_dsn):
    housing(scratch)
    gpu = FakeGpu(scores=lambda q, texts: [None if i % 2 else 1.0 for i in range(len(texts))])
    result, _ = go(scratch_dsn, "housing policy", gpu=gpu, config=cfg(rerank="blend", dense_limit=4, sparse_limit=10, speaker=False))
    assert len(result.hits) == 5 and not result.warnings


# --------------------------------------------------------------------------- the query vector: is_query embedding, and
# degrading without the GPU ---------------------------------------------------------------------------


def test_when_the_gpu_is_down_the_search_falls_back_to_keywords_and_says_so(scratch, scratch_dsn):
    ids = housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", gpu=FakeGpu(embed_error=GpuUnavailable("refused")), config=cfg(sparse_limit=10))
    assert result.hits and result.hits[0].chunk_id == ids["c5"]
    assert any("dense" in w.lower() or "embed" in w.lower() for w in result.warnings)
    assert all(h.legs["dense"] is None and h.legs["speaker"] is None for h in result.hits)


def test_with_no_gpu_and_no_keyword_terms_the_result_is_empty_and_explained_not_an_error(scratch, scratch_dsn):
    housing(scratch)
    result, _ = go(scratch_dsn, "zebra quokka", gpu=FakeGpu(embed_error=GpuUnavailable("refused")), config=cfg())
    assert result.hits == [] and result.warnings


# --------------------------------------------------------------------------- expansion to the surrounding sentences,
# batched ---------------------------------------------------------------------------


def expansion_corpus(scratch):
    c = Corpus(scratch)
    c.debate("d1", "Long speech")
    c.speech("d1", "before", ["Earlier speech one.", "Earlier speech two."])
    sentences = [f"Sentence number {i}." for i in range(12)]
    sentences[6] = "The rare quokka appears here."
    vectors = [None] * 12
    vectors[6] = vec((0, 1))
    boiler = [False] * 12
    boiler[3] = True
    mid = c.speech("d1", "mid", sentences, vectors=vectors, boilerplate=boiler)
    c.speech("d1", "after", ["Later speech one."])
    return mid


def test_a_hit_is_expanded_to_its_neighbours_within_the_speech_and_stops_at_boilerplate(scratch, scratch_dsn):
    mid = expansion_corpus(scratch)
    result, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=4, speaker=False, dense_limit=1), top_k=1)
    (hit,) = result.hits
    assert hit.chunk_id == mid[6]
    # sentences 2..10 would be the radius-4 range; sentence 3 is boilerplate, so the contiguous run is 4..10
    assert hit.context == " ".join(f"Sentence number {i}." if i != 6 else "The rare quokka appears here." for i in range(4, 11))
    assert [s[1] for s in hit.context_sentences][0] == 4


def test_expansion_never_crosses_into_another_speech(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Edges")
    c.speech("d1", "before", ["Earlier speech one.", "Earlier speech two."])
    (hit_id,) = c.speech("d1", "mine", ["The rare quokka opens the speech."], vectors=[vec((0, 1))])
    c.speech("d1", "after", ["Later speech one."])
    result, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=5, speaker=False), top_k=1)
    assert result.hits[0].context == "The rare quokka opens the speech."


def test_a_radius_of_zero_gives_no_context_and_saves_the_query(scratch, scratch_dsn):
    expansion_corpus(scratch)
    result, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=0, speaker=False), top_k=1)
    assert result.hits[0].context is None and result.hits[0].context_sentences is None


def test_expansion_is_one_batched_query_however_many_hits(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Many")
    for i in range(6):
        c.speech("d1", f"s{i}", [f"Intro {i}.", f"The quokka number {i}.", f"Outro {i}."], vectors=[None, vec((0, 1), (1, 0.1 * (i + 1))), None])
    one, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=1, speaker=False), top_k=1)
    six, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=1, speaker=False), top_k=6)
    assert one.diagnostics["queries"] == six.diagnostics["queries"] and len(six.hits) == 6 and all(h.context for h in six.hits)


def test_only_the_returned_hits_are_expanded(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Many")
    for i in range(6):
        c.speech("d1", f"s{i}", [f"The quokka number {i}."], vectors=[vec((0, 1), (1, 0.1 * (i + 1)))])
    result, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=2, speaker=False), top_k=2)
    assert len(result.hits) == 2 and all(h.context for h in result.hits)


# --------------------------------------------------------------------------- hit contents
# ---------------------------------------------------------------------------


def test_a_hit_carries_everything_a_citation_needs(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.member(30, "Ann Lee", list_as="Lee, Ann", party="Labour")
    c.debate("d1", "Housing Bill", date="2026-09-03")
    (cid,) = c.speech("d1", "s1", ["The quokka act."], mnis_id=30, speaker_name="Ann Lee (Bath) (Lab)", party="Labour", office="The Minister", vectors=[vec((0, 1))])
    result, _ = go(scratch_dsn, "quokka", config=cfg(speaker=False))
    hit = result.hits[0]
    assert hit.speaker == "Ann Lee" and hit.score == hit.rrf and hit.rerank_score is None  # the member's own name, not the parsed label
    assert (hit.rank, hit.chunk_id, hit.speech_id, hit.debate_id) == (1, cid, "s1", "d1")
    assert (hit.speaker, hit.party, hit.office, hit.heading, hit.house) == ("Ann Lee", "Labour", "The Minister", "Housing Bill", "Commons")
    assert str(hit.date) == "2026-09-03" and hit.sentence == "The quokka act." and hit.passage == "The quokka act."
    assert hit.url is None or isinstance(hit.url, str)  # hansard_url is unverified; it only has to be text or absent


def test_ranks_are_consecutive_from_one_and_top_k_is_respected(scratch, scratch_dsn):
    housing(scratch)
    result, _ = go(scratch_dsn, "housing policy", config=cfg(speaker=False), top_k=3)
    assert [h.rank for h in result.hits] == [1, 2, 3]


def test_the_same_search_twice_gives_identical_results(scratch, scratch_dsn):
    housing(scratch)
    a, _ = go(scratch_dsn, "housing policy", config=cfg(rerank="blend"))
    b, _ = go(scratch_dsn, "housing policy", config=cfg(rerank="blend"))
    assert [(h.chunk_id, h.score) for h in a.hits] == [(h.chunk_id, h.score) for h in b.hits]


# --------------------------------------------------------------------------- awkward queries: the search never fails on
# what a user types ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "...", "😀", None, "\x00", "\ud800", 42])
def test_an_empty_or_letterless_query_returns_nothing_and_touches_neither_the_database_nor_the_gpu(query):
    gpu = FakeGpu()

    async def main():
        pool = make_pool("postgresql://nobody@127.0.0.1:1/none", min_size=1, max_size=1, served=False)  # never opened: any use would raise
        return await HybridSearcher(pool, gpu, cfg()).search(query)

    result = run(main())
    assert result.hits == [] and result.warnings and gpu.embed_calls == [] and gpu.rerank_calls == []


@pytest.mark.parametrize(
    "query",
    ["'; DROP TABLE chunk; --", 'a" OR "1"="1', "a & b | !c :* <-> (x)", "\\'; --", "%s %(x)s {0}", "<script>alert(1)</script>", "a\x00b\ud800c", "Ünïcödé 你好 مرحبا"],
)
def test_hostile_and_odd_text_is_just_text(scratch, scratch_dsn, query):
    housing(scratch)
    result, _ = go(scratch_dsn, query, config=cfg())
    assert isinstance(result.hits, list)
    assert scratch.execute("SELECT count(*) FROM chunk").fetchone()[0] == 5  # nothing was dropped or altered


def test_a_huge_paste_is_truncated_with_a_warning_and_still_searched(scratch, scratch_dsn):
    housing(scratch)
    paste = "housing policy " * 2000
    gpu = FakeGpu()
    result, _ = go(scratch_dsn, paste, gpu=gpu, config=cfg())
    assert any("shortened" in w or "truncated" in w for w in result.warnings) and len(gpu.embed_calls[0]) <= 2000
    assert result.hits


def test_a_long_query_uses_only_the_most_informative_terms(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    words = [f"word{i:03d}" for i in range(30)]
    for i, w in enumerate(words):
        for n in range(i + 1):  # word000 is rare, word029 is common
            c.speech("d1", f"{w}-{n}", [f"Sentence with {w}."])
    result, _ = go(scratch_dsn, " ".join(words), config=cfg(dense=False, speaker=False))
    assert len(result.diagnostics["terms"]) == 24 and result.diagnostics["terms"][0] == "word000"


@pytest.mark.parametrize("top_k", [0, -1, MAX_TOP_K + 1, 10**6])
def test_a_silly_top_k_is_a_programming_error(scratch, scratch_dsn, top_k):
    housing(scratch)
    with pytest.raises(ValueError):
        go(scratch_dsn, "housing", top_k=top_k)


# --------------------------------------------------------------------------- infrastructure failures are reported as
# RetrievalError, not raw driver errors ---------------------------------------------------------------------------


def test_an_unopened_pool_is_a_retrieval_error(scratch_dsn):
    async def main():
        pool = make_pool(scratch_dsn, min_size=1, max_size=1, served=False)
        await HybridSearcher(pool, FakeGpu(), cfg()).search("housing")

    with pytest.raises(RetrievalError):
        run(main())


def test_a_pool_that_fails_after_the_statistics_loaded_is_still_a_retrieval_error(scratch, scratch_dsn):
    housing(scratch)

    async def main():
        pool = make_pool(scratch_dsn, min_size=1, max_size=1, served=False)
        await pool.open(wait=True, timeout=10)
        searcher = HybridSearcher(pool, FakeGpu(), cfg())
        await searcher.refresh()  # the statistics are cached, so the failure comes from the search's own queries
        await pool.close()
        await searcher.search("housing policy")

    with pytest.raises(RetrievalError):
        run(main())


def test_the_reranker_pool_bounds_how_many_candidates_are_scored_and_returned(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    for i in range(6):
        c.speech("d1", f"s{i}", [f"The quokka number {i}."], vectors=[vec((0, 1), (1, 0.1 * (i + 1)))])
    gpu = FakeGpu()
    result, _ = go(scratch_dsn, "quokka", gpu=gpu, config=cfg(rerank="blend", pool=3, speaker=False), top_k=50)
    (_, texts), = gpu.rerank_calls
    assert len(texts) == 3 and len(result.hits) == 3 and result.diagnostics["clusters"] == 3 and result.diagnostics["fused"] == 6


def test_a_database_that_cannot_be_reached_is_a_retrieval_error():
    async def main():
        pool = make_pool("postgresql://postgres:postgres@127.0.0.1:1/none", min_size=1, max_size=1, timeout=0.5, served=False)
        await pool.open(wait=False)
        try:
            await HybridSearcher(pool, FakeGpu(), cfg()).search("housing")
        finally:
            await pool.close()

    with pytest.raises(RetrievalError):
        run(main())


# --------------------------------------------------------------------------- statistics refresh, concurrency
# ---------------------------------------------------------------------------


def test_a_refresh_picks_up_new_terms(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    c.speech("d1", "s1", ["Alpha bravo."])

    async def main():
        pool = make_pool(scratch_dsn, min_size=1, max_size=2, served=False)
        await pool.open(wait=True, timeout=10)
        try:
            searcher = HybridSearcher(pool, FakeGpu(), cfg(dense=False, speaker=True))
            first = await searcher.search("zulu")  # loads the statistics: "zulu" is unknown
            scratch.execute("INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, text, window_id)"
                            " VALUES ('s9', 'd1', 9, 'Zed Zulu', 77, 'Zulu charlie.', 'T0')")
            scratch.execute("INSERT INTO member_id_map (mnis_id, name, name_list_as) VALUES (77, 'Zed Zulu', 'Zulu, Zed')")
            scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body, debate_date, window_id) VALUES ('s9', 'd1', 99, 0, 'Zulu charlie.', 'Zulu charlie.', '2026-09-01', 'T0')")
            stale = await searcher.search("zulu")
            await searcher.refresh()
            fresh = await searcher.search("zulu")
            return first, stale, fresh
        finally:
            await pool.close()

    first, stale, fresh = run(main())
    assert first.diagnostics["terms"] == [] and stale.diagnostics["terms"] == []  # the cached statistics do not know it yet
    assert fresh.diagnostics["terms"] == ["zulu"] and fresh.hits


def test_statistics_expire_on_a_timer(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    c.speech("d1", "s1", ["Alpha bravo."])

    async def main():
        pool = make_pool(scratch_dsn, min_size=1, max_size=2, served=False)
        await pool.open(wait=True, timeout=10)
        try:
            searcher = HybridSearcher(pool, FakeGpu(), cfg(dense=False, speaker=False, lexicon_ttl_s=0.0))
            await searcher.search("alpha")
            scratch.execute("INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body, debate_date, window_id) VALUES ('s1', 'd1', 99, 5, 'Zulu charlie.', 'Zulu charlie.', '2026-09-01', 'T0')")
            return await searcher.search("zulu")
        finally:
            await pool.close()

    assert run(main()).diagnostics["terms"] == ["zulu"]  # ttl 0: reloaded on every search


def test_many_concurrent_searches_each_get_their_own_answer(scratch, scratch_dsn):
    c = Corpus(scratch)
    c.debate("d1", "Words")
    for i in range(15):
        c.speech("d1", f"s{i}", [f"The uniqueword{i:02d} statement."], vectors=[vec((0, 1), (1, 0.1 * (i + 1)))])

    async def main():
        pool = make_pool(scratch_dsn, min_size=2, max_size=4, served=False)
        await pool.open(wait=True, timeout=10)
        try:
            searcher = HybridSearcher(pool, FakeGpu(), cfg(speaker=False, dense=False))
            results = await asyncio.gather(*[searcher.search(f"uniqueword{i:02d}") for i in range(15)])
            return results
        finally:
            await pool.close()

    results = run(main())
    for i, r in enumerate(results):
        assert r.hits and f"uniqueword{i:02d}" in r.hits[0].sentence.lower()


def test_every_expanded_sentence_carries_its_own_chunk_id(scratch, scratch_dsn):
    """The expansion returns each sentence's id, so the card can make every sentence the model reads citable. Each id
    must be THAT sentence's — an id attached to its neighbour would let a claim cite words it does not hold.
    """
    mid = expansion_corpus(scratch)
    result, _ = go(scratch_dsn, "quokka", config=cfg(expand_radius=4, speaker=False, dense_limit=1), top_k=1)
    (hit,) = result.hits
    assert all(len(s) == 5 for s in hit.context_sentences)
    stored = dict(scratch.execute("SELECT id, body FROM chunk WHERE id = ANY(%s)",
                                  ([s[4] for s in hit.context_sentences],)).fetchall())
    assert all(stored[s[4]] == s[2] for s in hit.context_sentences), "an id points at another sentence's text"
    assert mid[6] in {s[4] for s in hit.context_sentences}


# The dense leg is written for the configured index --------------------------------------------------------


def test_the_search_runs_the_dense_leg_written_for_its_configured_index(scratch, scratch_dsn, monkeypatch):
    """The searcher must hand its `dense_index` to the SQL. If it did not, a frozen "vector" configuration would run
    the halfvec ORDER BY (a sequential scan against the full-precision index) or the reverse — and both still return
    results, so nothing but this would notice.
    """
    import retrieval.hybrid as hybrid_module

    housing(scratch)
    real = hybrid_module.fuse_sql
    seen: list[str] = []

    def spy(filters, exact, dense_index="halfvec"):
        seen.append(dense_index)
        return real(filters, exact, dense_index)

    monkeypatch.setattr(hybrid_module, "fuse_sql", spy)
    hits = {}
    for form in ("vector", "halfvec"):
        result, _ = go(scratch_dsn, "housing policy", config=cfg(dense_index=form))
        hits[form] = [h.chunk_id for h in result.hits]
    assert seen == ["vector", "halfvec"]
    assert hits["vector"] == hits["halfvec"] and hits["vector"]
