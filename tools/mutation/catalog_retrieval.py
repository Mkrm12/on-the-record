"""Two suites: retrieval the database pool and vector adaptation (src/app/db.py), query hygiene, speaker resolution,
collapse, corpus statistics, metrics, the evaluation harness, the GPU client, and the pure ranking decisions in
hybrid.py. Fast: no scratch database is needed except by the db tests. search the SQL and the orchestration in
hybrid.py, run against the scratch-database tests (each mutant creates and drops its own database, so mutants run in
parallel safely).
"""

from mutation_engine import Mutant, Project

_D = "src/app/db.py"
_Q = "src/retrieval/query.py"
_C = "src/retrieval/collapse.py"
_L = "src/retrieval/lexicon.py"
_M = "src/retrieval/metrics.py"
_S = "src/retrieval/speakers.py"
_G = "src/retrieval/gpu_client.py"
_E = "src/retrieval/evaluation.py"
_H = "src/retrieval/hybrid.py"

TESTS = [
    "tests/test_db_vectors.py", "tests/test_retrieval_query.py", "tests/test_retrieval_collapse.py", "tests/test_retrieval_lexicon.py",
    "tests/test_retrieval_metrics.py", "tests/test_retrieval_speakers.py", "tests/test_retrieval_evaluation.py",
    "tests/test_retrieval_gpu_client.py", "tests/test_retrieval_rank.py",
]
SEARCH_TESTS = ["tests/test_retrieval_hybrid.py", "tests/test_retrieval_rank.py"]
SEARCH_PROJECT = Project(dirs=("src", "tests", "db"))  # the scratch-database fixture builds from db/schema.sql and the migrations

_RAW = [
    # --- src/app/db.py: vector adaptation and the pool
    (_D, "a bool is accepted as a coordinate", "if isinstance(x, bool) or not isinstance(x, numbers.Real):", "if not isinstance(x, numbers.Real):"),
    (_D, "any object is accepted as a coordinate", "if isinstance(x, bool) or not isinstance(x, numbers.Real):", "if isinstance(x, bool):"),
    (_D, "the wrong size is accepted", "if len(items) != dim:", "if False:"),
    (_D, "NaN and inf are accepted", "if not all(math.isfinite(x) for x in stored):", "if False:"),
    (_D, "a float32 overflow goes unnoticed", "stored = vector.to_list()  # as float32, which is how Postgres will hold it", "stored = [float(x) for x in items]"),
    (_D, "the zero vector is accepted", "if not allow_zero and not any(stored):", "if False:"),
    (_D, "allow_zero is ignored", "if not allow_zero and not any(stored):", "if not any(stored):"),
    (_D, "a multi-dimensional array is not named", "if value.ndim != 1:", "if False:"),
    (_D, "numpy arrays are not recognised", 'elif hasattr(value, "ndim") and hasattr(value, "tolist"):', "elif False:"),
    (_D, "an existing Vector is not accepted", "if isinstance(value, Vector):", "if False:"),
    (_D, "the configure hook leaves its transaction open", "await conn.commit()", "pass"),
    (_D, "the configure hook does not register the vector type", "await register_vector_async(conn)", "pass"),
    (_D, "the pool is built without the configure hook", "configure=_configure,", "configure=None,"),
    (_D, "the pool does not check a connection when it is borrowed", "check=_CHECK,", "check=None,"),
    (_D, "the pool opens at import", "open=False,", "open=True,"),
    (_D, "a connection attempt can hang", '"connect_timeout": CONNECT_TIMEOUT_S', '"connect_timeout": 0'),
    (_D, "a checkout can wait forever", "timeout: float = CHECKOUT_TIMEOUT_S,", "timeout: float = 0,"),
    (_D, "the configured DSN is ignored", "dsn or settings.pg_dsn", 'dsn or "postgresql://nowhere/none"'),
    (_D, "caller connection arguments replace the defaults", 'kwargs={"connect_timeout": CONNECT_TIMEOUT_S, **(kwargs or {})},', 'kwargs=kwargs or {"connect_timeout": CONNECT_TIMEOUT_S},'),
    (_D, "the vector size is wrong", "EMBEDDING_DIM = 384", "EMBEDDING_DIM = 385"),
    # --- query hygiene
    (_Q, "control characters are kept", 'text = _INVISIBLE.sub("", _CONTROL.sub(" ", text))', 'text = _INVISIBLE.sub("", text)'),
    (_Q, "invisible characters are kept", 'text = _INVISIBLE.sub("", _CONTROL.sub(" ", text))', 'text = _CONTROL.sub(" ", text)'),
    (_Q, "lone surrogates are kept", 'text = text.encode("utf-8", "ignore").decode("utf-8")', "text = text"),
    (_Q, "whitespace is not collapsed", 'text = " ".join(text.split())', "text = text.strip()"),
    (_Q, "a query at the limit is flagged as truncated", "truncated = len(text) > MAX_QUERY_CHARS", "truncated = len(text) >= MAX_QUERY_CHARS"),
    (_Q, "a long query is cut inside a word", "text = (cut[:space] if space > MAX_QUERY_CHARS - 200 else cut).rstrip()", "text = cut.rstrip()"),
    (_Q, "a query with no text is not rejected", 'return (text, truncated) if _has_text(text) else ("", False)', "return text, truncated"),
    (_Q, "a non-string is not rejected", "if not isinstance(text, str):", "if False:"),
    # --- collapse
    (_C, "the overlap span is one sentence too wide", "abs(c.anchor.sent_idx - hit.sent_idx) <= span", "abs(c.anchor.sent_idx - hit.sent_idx) <= span + 1"),
    (_C, "hits exactly at the span are not merged", "abs(c.anchor.sent_idx - hit.sent_idx) <= span", "abs(c.anchor.sent_idx - hit.sent_idx) < span"),
    (_C, "identical passages in other speeches are not merged", "home = by_text.get(hit.parent_body)", "home = None"),
    (_C, "hits in other speeches are merged by position", "for c in by_speech.get(hit.speech_id, ())", "for cl in by_speech.values() for c in cl"),
    (_C, "a repeated chunk is counted twice", "if hit.chunk_id in seen:", "if False:"),
    (_C, "the limit keeps one cluster too many", "clusters[:limit]", "clusters[: limit + 1]"),
    (_C, "siblings are listed worst first", "home.siblings.append(hit)", "home.siblings.insert(0, hit)"),
    (_C, "the span is the radius, not twice it", "OVERLAP_SPAN = 2 * WINDOW_RADIUS", "OVERLAP_SPAN = WINDOW_RADIUS"),
    (_C, "the window radius is wrong", "WINDOW_RADIUS = 2", "WINDOW_RADIUS = 3"),
    (_C, "a cluster is not registered under its speech", "by_speech.setdefault(hit.speech_id, []).append(cluster)", "pass"),
    (_C, "a cluster is not registered under its text", "by_text[hit.parent_body] = cluster", "pass"),
    # --- corpus statistics
    (_L, "IDF can go negative", "math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))", "math.log((self.n_docs - df + 0.5) / (df + 0.5))"),
    (_L, "IDF uses the wrong smoothing", "(df + 0.5))", "(df + 1.5))"),
    (_L, "terms that occur nowhere are kept", "if self.df.get(lx, 0) > 0", "if self.df.get(lx, 0) >= 0"),
    (_L, "the rarest terms are dropped first", "terms.sort(key=lambda t: (-t[1], t[0]))", "terms.sort(key=lambda t: (t[1], t[0]))"),
    (_L, "equally rare terms have no fixed order", "terms.sort(key=lambda t: (-t[1], t[0]))", "terms.sort(key=lambda t: -t[1])"),
    (_L, "a repeated term counts twice", "dict.fromkeys(lexemes)", "lexemes"),
    (_L, "the term cap is ignored", "return terms[:max_terms]", "return terms"),
    (_L, "the default term cap changes", "MAX_TERMS = 24", "MAX_TERMS = 25"),
    (_L, "coverage adds instead of scaling", "return coverage * sum(idf for _, idf in terms)", "return coverage + sum(idf for _, idf in terms)"),
    (_L, "a zero average length is allowed", "if not self.avg_len > 0:", "if False:"),
    # --- metrics
    (_M, "speeches are not kept in rank order", "return list(dict.fromkeys(ranked))", "return list(reversed(list(dict.fromkeys(ranked))))"),
    (_M, "found looks one place too far", "len(set(relevant) & set(ranked[:k]))", "len(set(relevant) & set(ranked[: k + 1]))"),
    (_M, "recall of nothing is zero, not undefined", "if relevant else None", "if relevant else 0.0"),
    (_M, "the reciprocal rank uses the wrong position", "return 1 / position", "return 1 / (position + 1)"),
    (_M, "the reciprocal rank ignores the cut", "for position, speech in enumerate(ranked[:k], start=1):\n        if speech in wanted:", "for position, speech in enumerate(ranked, start=1):\n        if speech in wanted:"),
    (_M, "the nDCG discount is wrong", "dcg += 1 / math.log2(position + 1)", "dcg += 1 / math.log2(position + 2)"),
    (_M, "nDCG counts a repeated speech twice", "if speech in wanted and speech not in seen:", "if speech in wanted:"),
    (_M, "the ideal ranking is longer than the cut", "min(len(wanted), k)", "len(wanted)"),
    (_M, "the bootstrap counts ties as wins", "wins=sum(d > 0 for d in diffs)", "wins=sum(d >= 0 for d in diffs)"),
    (_M, "the bootstrap accepts no data", "if not diffs:", "if False:"),
    # --- speaker resolution
    (_S, "'the' counts as a title before a surname", '_TITLES_BEFORE_SURNAME = _TITLE_WORDS - {"the"}', "_TITLES_BEFORE_SURNAME = _TITLE_WORDS"),
    (_S, "a bare surname needs no capital", "capitalised = first_unit > 0 and units[first_unit][2][:1].isupper()", "capitalised = True"),
    (_S, "a sentence-initial surname is accepted", "capitalised = first_unit > 0 and", "capitalised = True and"),
    (_S, "a common surname is accepted bare", "if titled or (capitalised and not common):", "if titled or capitalised:"),
    (_S, "one common part makes a surname common", "common = all(t in _ALWAYS_COMMON or self._is_common(t) for t in span)", "common = any(t in _ALWAYS_COMMON or self._is_common(t) for t in span)"),
    (_S, "the ambiguity cap is too loose", "MAX_AMBIGUOUS = 4", "MAX_AMBIGUOUS = 8"),
    (_S, "the member cap is too loose", "MAX_MEMBERS = 8", "MAX_MEMBERS = 9"),
    (_S, "possessives break names", '_tokens(_POSSESSIVE.sub("", text))', "_tokens(text)"),
    (_S, "apostrophes split names", "re.sub(r\"['’`´.]\", \"\", text)", 're.sub(r"[`´.]", "", text)'),
    (_S, "accents are kept", "if not unicodedata.combining(ch)", "if True"),
    (_S, "case is kept", ").casefold()", ")"),
    (_S, "titles are not stripped from names", "while start < end and tokens[start] in _TITLE_WORDS:", "while False:"),
    (_S, "the title stays in the stripped text", "while first > 0 and set(_tokens(units[first - 1][2])) <= _TITLE_WORDS and _tokens(units[first - 1][2]):", "while False:"),
    (_S, "the post-nominal stays in the stripped text", "while last + 1 < len(units) and _tokens(units[last + 1][2]) and set(_tokens(units[last + 1][2])) <= _POST_NOMINALS:", "while False:"),
    (_S, "ambiguity is never reported", "ambiguous = ambiguous or len(ids) > 1", "ambiguous = False"),
    (_S, "a member named twice is listed twice", "member_ids.extend(m for m in sorted(ids) if m not in member_ids)", "member_ids.extend(sorted(ids))"),
    (_S, "mentions are not in order of appearance", "found.sort(key=lambda f: f[0])", "found.sort(key=lambda f: -f[0])"),
    (_S, "stripping leaves the gaps", 'return " ".join(query.split())', "return query"),
    # --- the GPU client
    (_G, "queries are embedded as passages", '{"texts": [text], "is_query": True}', '{"texts": [text], "is_query": False}'),
    (_G, "the expected vector size is wrong", "EMBEDDING_DIM = 384", "EMBEDDING_DIM = 385"),
    (_G, "the unit-length tolerance is far too loose", "_UNIT_TOLERANCE = 1e-3", "_UNIT_TOLERANCE = 5.0"),
    (_G, "non-finite coordinates are accepted", "if not all(math.isfinite(x) for x in floats):", "if False:"),
    (_G, "the unit-length check is skipped", "if abs(math.sqrt(sum(x * x for x in floats)) - 1.0) > _UNIT_TOLERANCE:", "if False:"),
    (_G, "several vectors are accepted for one query", "if not isinstance(vectors, list) or len(vectors) != 1:", "if not isinstance(vectors, list) or len(vectors) < 1:"),
    (_G, "a wrong-size vector is accepted", "if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:", "if not isinstance(vector, list):"),
    (_G, "a blank query is sent to the service", "if not isinstance(text, str) or not text.strip():", "if False:"),
    (_G, "a rerank of nothing calls the service", "if not candidates:\n            return []", "if False:\n            return []"),
    (_G, "a bad rerank index is accepted", " or not 0 <= index < len(candidates) or index in seen:", ":"),
    (_G, "a non-finite rerank score is accepted", " or isinstance(score, bool) or not math.isfinite(score):", " or isinstance(score, bool):"),
    (_G, "a dropped connection does not count against the breaker", 'self.breaker.failure()\n                raise GpuUnavailable(f"the GPU service is unreachable ({type(exc).__name__})") from None', 'raise GpuUnavailable(f"the GPU service is unreachable ({type(exc).__name__})") from None'),
    (_G, "a read timeout is reported as unavailable", 'raise GpuTimeout("the GPU service did not answer in time") from None', 'raise GpuUnavailable("the GPU service did not answer in time") from None'),
    (_G, "a 503 is not retried", "if attempt < self.retries:", "if False:"),
    (_G, "a 503 is retried once too often", "if attempt < self.retries:", "if attempt <= self.retries:"),
    (_G, "the retry count is off by one", "for attempt in range(self.retries + 1):", "for attempt in range(self.retries):"),
    (_G, "Retry-After is not capped", "return min(wanted, self.max_backoff)", "return wanted"),
    (_G, "the back-off does not grow", "0.25 * 2**attempt", "0.25"),
    (_G, "a 500 is taken for a client error", "if response.status_code >= 500:", "if response.status_code > 500:"),
    (_G, "a 4xx is taken for success", "if response.status_code >= 400:", "if response.status_code > 499:"),
    (_G, "a busy service trips the breaker", '                raise GpuBusy("the GPU service is busy (503) and stayed busy through the retries")', '                self.breaker.failure()\n                raise GpuBusy("the GPU service is busy (503) and stayed busy through the retries")'),
    (_G, "the breaker opens one failure late", "if self._failures >= self.threshold:", "if self._failures > self.threshold:"),
    (_G, "the breaker stays shut at exactly the cooldown", "self._clock() - self._opened_at >= self.cooldown", "self._clock() - self._opened_at > self.cooldown"),
    (_G, "a success does not reset the breaker", "self._failures, self._opened_at = 0, None", "pass"),
    (_G, "a successful call does not close the breaker", "            self.breaker.success()\n", ""),
    (_G, "the default timeouts are unbounded", "DEFAULT_TIMEOUT = httpx.Timeout(connect=2.0, read=30.0, write=5.0, pool=5.0)", "DEFAULT_TIMEOUT = httpx.Timeout(None)"),
    # --- the evaluation harness
    (_E, "identical passages in other speeches do not count", "for h in hits for s in (h.speech_id, *h.also_in_speeches))", "for h in hits for s in (h.speech_id,))"),
    (_E, "abstain questions are scored", "if not o.abstain:\n            groups[o.category].append(score_outcome(o, k))", "if True:\n            groups[o.category].append(score_outcome(o, k))"),
    (_E, "runs over different queries are compared", "if {o.qid for o in base} != {o.qid for o in other}:", "if False:"),
    (_E, "paired differences include abstain questions", "        if o.abstain:\n            continue\n        a, b", "        a, b"),
    (_E, "AUC ignores ties", "(p > n) + 0.5 * (p == n)", "(p > n)"),
    (_E, "AUC of nothing is a number", "if not positives or not negatives:", "if False:"),
    (_E, "the dense baseline keeps the keyword leg", 'Variant("dense", replace(b, sparse=False, speaker=False, rerank="off")', 'Variant("dense", replace(b, speaker=False, rerank="off")'),
    (_E, "the naive hybrid keeps the threshold", 'replace(b, speaker=False, sparse_coverage=0.0, rerank="off")', 'replace(b, speaker=False, rerank="off")'),
    (_E, "variant B has no reranker", 'Variant("hybrid+rerank", replace(b, speaker=False),', 'Variant("hybrid+rerank", replace(b, speaker=False, rerank="off"),'),
    (_E, "full-replace blends", 'replace(b, rerank="replace")', 'replace(b, rerank="blend")'),
    (_E, "the top rerank score is the lowest", "max(scored) if scored else None", "min(scored) if scored else None"),
    (_E, "a failed search stops the run", "except Exception as exc:  # noqa: BLE001", "except ZeroDivisionError as exc:  # noqa: BLE001"),
    (_E, "the percentile is off by one", "math.ceil(p / 100 * len(ordered)) - 1", "math.ceil(p / 100 * len(ordered))"),
    # --- pure decisions in hybrid.py
    (_H, "the header names a party without a speaker", "if who and c.party:", "if c.party:"),
    (_H, "the header names an office without a speaker", "if who and c.office:", "if c.office:"),
    (_H, "the header leaves out the speaker", "prefix = f\"{who}, in the debate '{c.heading}': \" if who else f\"In the debate '{c.heading}': \"", "prefix = f\"In the debate '{c.heading}': \""),
    (_H, "the header leaves out the speaker when there is no debate", 'prefix = f"{who}: " if who else ""', 'prefix = ""'),
    (_H, "off mode still uses the scores", 'if mode == "off" or scores is None:', "if scores is None:"),
    (_H, "a non-finite score is ranked", "if scores[i] is not None and math.isfinite(scores[i])]", "if scores[i] is not None]"),
    (_H, "the lowest score wins", "by_score = sorted(usable, key=lambda i: (-scores[i], i))", "by_score = sorted(usable, key=lambda i: (scores[i], i))"),
    (_H, "the fused rank is off by one in the blend", "1 / (k + i + 1) +", "1 / (k + i) +"),
    (_H, "the reranker rank is off by one in the blend", "1 / (k + rerank_rank[i])", "1 / (k + rerank_rank[i] + 1)"),
    (_H, "the blend puts the worst first", "return [Ranked(i, blended[i]) for i in sorted(range(n), key=lambda i: (-blended[i], i))]", "return [Ranked(i, blended[i]) for i in sorted(range(n), key=lambda i: (blended[i], i))]"),
    (_H, "mismatched scores are accepted", "if scores is not None and len(scores) != n:", "if False:"),
    (_H, "an unknown rerank mode is accepted", "if mode not in RERANK_MODES:\n        raise ValueError(f\"mode must", "if False:\n        raise ValueError(f\"mode must"),
    (_H, "the context range excludes its first row", "if lo <= r[0] <= hi}", "if lo < r[0] <= hi}"),
    (_H, "the context range excludes its last row", "if lo <= r[0] <= hi}", "if lo <= r[0] < hi}"),
    (_H, "the context does not grow backwards", "while start - 1 in by_seq:", "while False:"),
    (_H, "the context does not grow forwards", "while end + 1 in by_seq:", "while False:"),
    (_H, "a missing anchor is not noticed", "if anchor_seq not in by_seq:", "if False:"),
]  # fmt: skip

_SEARCH = [
    # --- configuration validation (its tests live beside the search tests)
    (_H, "the default ef_search changes", "ef_search: int = 100", "ef_search: int = 40"),
    (_H, "the default RRF k changes", "rrf_k: int = 60", "rrf_k: int = 30"),
    (_H, "the default leg limit changes", "dense_limit: int = 50", "dense_limit: int = 51"),
    (_H, "ef_search may be zero", 'self.ef_search, "ef_search", 1, 1000)', 'self.ef_search, "ef_search", 0, 1000)'),
    (_H, "ef_search may exceed pgvector's range", 'self.ef_search, "ef_search", 1, 1000)', 'self.ef_search, "ef_search", 1, 1001)'),
    (_H, "a bool is accepted as an integer setting", "if isinstance(value, bool) or not isinstance(value, int):", "if not isinstance(value, int):"),
    (_H, "an unknown rerank mode is accepted in a config", "if self.rerank not in RERANK_MODES:", "if False:"),
    (_H, "bad weights are accepted", "if len(self.weights) != 3 or not all(", "if False and not all("),
    (_H, "a coverage above one is accepted", "if not 0.0 <= self.sparse_coverage <= 1.0:", "if False:"),
    # --- the fused SQL
    (_H, "the dense leg returns boilerplate", "WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate{filter}\n        ORDER BY", "WHERE c.embedding IS NOT NULL{filter}\n        ORDER BY"),
    (_H, "the keyword leg returns boilerplate", "WHERE NOT c.is_boilerplate{filter}\n        GROUP BY c.id", "WHERE true{filter}\n        GROUP BY c.id"),
    (_H, "the speaker leg returns boilerplate", "AND NOT c.is_boilerplate AND c.embedding IS NOT NULL{filter}\n),", "AND c.embedding IS NOT NULL{filter}\n),"),
    (_H, "the dense leg has no limit before the join", "        LIMIT %(dense_limit)s\n", ""),
    (_H, "the keyword leg has no limit before the join", "        LIMIT %(sparse_limit)s\n", ""),
    (_H, "the speaker leg has no limit before the join", "        LIMIT %(speaker_limit)s\n", ""),
    (_H, "the keyword coverage threshold is gone", "        HAVING sum(t.idf) >= %(min_mass)s\n", ""),
    (_H, "keyword scores ignore sentence length", "%(b)s * length(c.ts) / %(avg_len)s", "%(b)s"),
    (_H, "the dense RRF term is off by one", "%(w_dense)s::float8 / (%(k)s + d.rnk)", "%(w_dense)s::float8 / (%(k)s + d.rnk + 1)"),
    (_H, "the keyword RRF term is off by one", "%(w_sparse)s::float8 / (%(k)s + s.rnk)", "%(w_sparse)s::float8 / (%(k)s + s.rnk + 1)"),
    (_H, "the speaker RRF term is off by one", "%(w_speaker)s::float8 / (%(k)s + p.rnk)", "%(w_speaker)s::float8 / (%(k)s + p.rnk + 1)"),
    (_H, "keyword-only hits are dropped by the join", "FULL OUTER JOIN sparse s USING (id)", "LEFT JOIN sparse s USING (id)"),
    (_H, "speaker-only hits are dropped by the join", "FULL OUTER JOIN speaker p USING (id)", "LEFT JOIN speaker p USING (id)"),
    (_H, "the fused list is cut worst first", "ORDER BY rrf DESC, id LIMIT %(fused_limit)s", "ORDER BY rrf ASC, id LIMIT %(fused_limit)s"),
    (_H, "the fused list is returned worst first", "ORDER BY f.rrf DESC, f.id", "ORDER BY f.rrf ASC, f.id"),
    (_H, "the speaker leg takes every sentence, not each speech's best", "SELECT DISTINCT ON (speech_id) id, speech_id, dist FROM member_chunks", "SELECT id, speech_id, dist FROM member_chunks"),
    (_H, "the speaker leg is not restricted to the named members", "WHERE sp.mnis_id = ANY(%(members)s::int[])", "WHERE sp.mnis_id IS NOT NULL"),
    (_H, "the party is not reported", "coalesce(cc.party_on_date_base, party_base(cc.party_as_attributed)) AS party", "cc.party_on_date_base AS party"),
    (_H, "the raw speaker label wins over the member's name", "coalesce(cc.member_name, cc.speaker_name) AS speaker", "coalesce(cc.speaker_name, cc.member_name) AS speaker"),
    # --- the transaction and its settings
    (_H, "the iterative scan setting is wrong", "SET LOCAL hnsw.iterative_scan = 'strict_order'", "SET LOCAL hnsw.iterative_scan = 'off'"),
    (_H, "ef_search leaks into the session", 'f"SET LOCAL hnsw.ef_search = {cfg.ef_search}"', 'f"SET hnsw.ef_search = {cfg.ef_search}"'),
    (_H, "the statement timeout leaks into the session", 'f"SET LOCAL statement_timeout = {cfg.statement_timeout_ms}"', 'f"SET statement_timeout = {cfg.statement_timeout_ms}"'),
    (_H, "ef_search is not applied", 'f"SET LOCAL hnsw.ef_search = {cfg.ef_search}"', 'f"SET LOCAL hnsw.ef_search = 40"'),
    (_H, "the transaction guard is removed", "if conn.info.transaction_status != pq.TransactionStatus.IDLE:", "if False:", "the pool hands out idle connections (test_every_connection_the_pool_hands_out_understands_vector_and_is_left_idle), so the guard is defence in depth and cannot change behaviour here"),
    # --- orchestration
    (_H, "the pool counts hits, not clusters", "collapse(candidates, limit=cfg.pool)", "collapse(candidates, limit=None)"),
    (_H, "a single candidate is sent to the reranker", 'if cfg.rerank == "off" or len(clusters) < 2:', 'if cfg.rerank == "off":'),
    (_H, "the reranker is called although it is off", 'if cfg.rerank == "off" or len(clusters) < 2:', "if len(clusters) < 2:"),
    (_H, "a reranker failure is not survived", "except GpuError as exc:\n            result.warnings.append(f\"the reranker", "except ZeroDivisionError as exc:\n            result.warnings.append(f\"the reranker"),
    (_H, "an embedding failure is not survived", "except (GpuError, VectorError) as exc:", "except ZeroDivisionError as exc:"),
    (_H, "the speaker's name stays in the keyword query", "keyword_text = strip_mentions(cleaned, resolution) if resolution else cleaned", "keyword_text = cleaned"),
    (_H, "the query is embedded only for the dense leg", "if (cfg.dense or (cfg.speaker and resolution)) else None", "if cfg.dense else None"),
    (_H, "the dense leg runs without a vector", "cfg.dense_limit if (cfg.dense and vector is not None) else 0", "cfg.dense_limit"),
    (_H, "expansion runs with a radius of zero", "if radius <= 0 or not clusters:", "if not clusters:"),
    (_H, "the expansion range starts too late", "los.append(min(seqs) - radius)", "los.append(min(seqs) + radius)"),
    (_H, "the expansion range ends too early", "his.append(max(seqs) + radius)", "his.append(max(seqs) - radius)"),
    (_H, "top_k above the maximum is accepted", "not 1 <= top_k <= MAX_TOP_K", "not 1 <= top_k"),
    (_H, "top_k below one is accepted", "not 1 <= top_k <= MAX_TOP_K", "not top_k <= MAX_TOP_K"),
    (_H, "the hit's score ignores the fused score when nothing reranked", "score=ranked.score if ranked.score is not None else a.rrf", "score=ranked.score or 0.0"),
    (_H, "duplicate speeches are not reported", "also_in_speeches=tuple(dict.fromkeys(m.speech_id for m in cluster.siblings if m.speech_id != a.speech_id))", "also_in_speeches=()"),
    (_H, "a database failure is not wrapped", "except psycopg.Error as exc:\n            raise RetrievalError(f\"the database search failed", "except ZeroDivisionError as exc:\n            raise RetrievalError(f\"the database search failed"),
    # --- statistics
    (_L if False else "src/retrieval/lexicon.py", "the statistics count boilerplate", "SELECT word, ndoc FROM ts_stat('SELECT ts FROM chunk WHERE NOT is_boilerplate')", "SELECT word, ndoc FROM ts_stat('SELECT ts FROM chunk')"),
    (_H, "only members who have spoken are not required", " WHERE EXISTS (SELECT 1 FROM speech s WHERE s.mnis_id = m.mnis_id)", ""),
]  # fmt: skip

# The vector index and the query that uses it, the switch, migration 023, gate I01, amending frozen --- retrieval.
_V = "src/retrieval/vector_index.py"
_W = "scripts/switch_vector_index.py"
_A = "scripts/audit_invariants.py"
_P = "src/answer/pipeline.py"
_M23 = "db/migrations/023_halfvec_index.sql"

INDEX_TESTS = ["tests/test_vector_index.py", "tests/test_switch_vector_index.py", "tests/test_frozen_amend.py",
               "tests/test_audit_invariants.py", "tests/test_retrieval_scale.py", "tests/test_retrieval_hybrid.py"]
INDEX_PROJECT = Project(dirs=("src", "tests", "db", "config", "scripts"))

_INDEX = [
    # the shared definition
    (_V, "the halfvec index uses the L2 operator class", '"halfvec": "halfvec_cosine_ops"', '"halfvec": "halfvec_l2_ops"'),
    (_V, "the full-precision index uses the L2 operator class", '"vector": "vector_cosine_ops"', '"vector": "vector_l2_ops"'),
    (_V, "the query vector is not cast to the index's type", "return f\"{column}{CAST['halfvec']} <=> {param}{CAST['halfvec']}\"",
     "return f\"{column}{CAST['halfvec']} <=> {param}::vector\""),
    (_V, "the column is not cast in the ORDER BY", "return f\"{column}{CAST['halfvec']} <=> {param}{CAST['halfvec']}\"",
     "return f\"{column} <=> {param}{CAST['halfvec']}\""),
    (_V, "the index expression loses its cast", "else f\"(embedding{CAST['halfvec']})\"", "else \"(embedding)\""),
    (_V, "an unknown form is accepted", "if form not in FORMS:", "if False:"),
    (_V, "an index name that is not an identifier is accepted",
     "if not isinstance(name, str) or not _IDENTIFIER.match(name):", "if not isinstance(name, str):"),
    (_V, "a CONCURRENTLY build is not concurrent", 'how = "CONCURRENTLY " if concurrently else ""', 'how = ""'),
    (_V, "an L2 index passes for the full-precision form", r'if re.search(r"\(embedding vector_cosine_ops\)", text):',
     r'if re.search(r"\(embedding vector_", text):'),
    (_V, "an IVFFlat index passes for HNSW", 'if "USING hnsw" not in text:', "if False:"),
    (_V, "a frozen file that names no form is read as halfvec", 'config.get("dense_index", "vector")',
     'config.get("dense_index", "halfvec")'),
    (_V, "the halfvec build is sized from the finished index, not the graph", '"halfvec": 1_600', '"halfvec": 1_280'),
    (_V, "the gate's spelling of the halfvec index is wrong", "halfvec_cosine_ops)%\"}", "halfvec_l2_ops)%\"}"),
    # the dense leg
    (_H, "an unknown dense_index is accepted", "if self.dense_index not in vector_index.FORMS:", "if False:"),
    (_H, "the searcher ignores its configured index form", "fuse_sql(cfg.filters, exact, cfg.dense_index)",
     "fuse_sql(cfg.filters, exact)"),
    (_H, "the dense leg ranks by the half-precision distance", "SELECT c.id, c.embedding <=> %(qvec)s::vector AS dist\n        FROM chunk c\n        WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate{filter}\n        ORDER BY {order}",
     "SELECT c.id, c.embedding::halfvec(384) <=> %(qvec)s::halfvec(384) AS dist\n        FROM chunk c\n        WHERE c.embedding IS NOT NULL AND NOT c.is_boilerplate{filter}\n        ORDER BY {order}"),
    # the switch
    (_W, "swap accepts a side index of the other form", 'if nxt is None or not nxt["valid"] or nxt["form"] != form:',
     'if nxt is None or not nxt["valid"]:'),
    (_W, "swap accepts an invalid side index", 'if nxt is None or not nxt["valid"] or nxt["form"] != form:',
     'if nxt is None or nxt["form"] != form:'),
    (_W, "a leftover side index of the wrong form is kept", 'if old["valid"] and old["form"] == form:', 'if True:'),
    (_W, "run rebuilds an index that is already right", 'if live is not None and live["valid"] and live["form"] == form:',
     "if False:"),
    (_W, "the build is not concurrent", "conn.execute(V.index_sql(form, name=NEXT, concurrently=True))",
     "conn.execute(V.index_sql(form, name=NEXT, concurrently=False))"),
    (_W, "the rollback database is not refused", "dsn = safety.refuse_protected(make_conninfo(settings.pg_dsn, dbname=database))",
     "dsn = make_conninfo(settings.pg_dsn, dbname=database)"),
    # migration 023
    (_M23, "the migration rebuilds over existing vectors", "IF EXISTS (SELECT 1 FROM chunk WHERE embedding IS NOT NULL) THEN",
     "IF false THEN"),
    (_M23, "the migration rebuilds an index that is already halfvec",
     "IF current_def LIKE '%::halfvec(384)) halfvec_cosine_ops)%' THEN", "IF false THEN"),
    # gate I01
    (_A, "I01 accepts an invalid index", "OR NOT i.indisvalid ", ""),
    (_A, "I01 accepts an index of any form", "OR pg_get_indexdef(i.indexrelid) NOT LIKE", "OR false AND pg_get_indexdef(i.indexrelid) NOT LIKE"),
    (_A, "I01 accepts a missing index", "AND (i.indexrelid IS NULL OR", "AND (false OR"),
    # frozen retrieval
    (_P, "a frozen file that omits a knob is accepted", 'missing = sorted(known - set(self.config) - {"filters"})', "missing = []"),
    (_P, "an amendment over a stale hash is accepted", 'if frozen_hash(config) != data["config_hash"]:', "if False:"),
    (_P, "an amendment without its evidence is accepted",
     'if not str(reason or "").strip() or not str(evidence or "").strip():', 'if not str(reason or "").strip():'),
    (_P, "a no-op amendment is recorded", "if not changed:", "if False:"),
    (_P, "an amendment is not validated as a SearchConfig",
     'Frozen(config=config, config_hash=new_hash, recipe=data.get("embedding_recipe", "")).search_config()  # validates',
     "pass  # validates"),
    (_P, "the start-up check never warns", "if live != wanted:", "if False:"),
]  # fmt: skip

MUTANTS = [Mutant(*row) for row in _RAW]
SEARCH_MUTANTS = [Mutant(*row) for row in _SEARCH]
INDEX_MUTANTS = [Mutant(*row) for row in _INDEX]
