-- The answer cache becomes a provenance record, not a key/value store.

BEGIN;

-- The original four columns stay, so existing readers keep working.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS id bigserial;

-- Provenance: what produced this answer.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS corpus_version         text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS retrieval_config_hash  text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS recipe                 text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS prompt_version         text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS model_id               text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS system_fingerprint     text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS code_version           text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS chunk_count            bigint;

-- The question as asked and as understood.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS question          text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS question_norm     text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS route             text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS resolved_filters  jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS asked_house       text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS asked_section     text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS asked_from        date;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS asked_to          date;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS window_label      text;

-- The outcome.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS outcome            text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS abstention_reason  text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS abstention_detail  text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS coverage_note      text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS answer_json        jsonb;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS structured_facts   jsonb NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS facets             jsonb NOT NULL DEFAULT '{}'::jsonb;

-- Verification.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS verified_at            timestamptz;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS verifier               text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS verifier_precision     text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS claim_count            int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS claims_verified        int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS claims_dropped         int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS quote_failures         int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS number_failures        int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS attribution_failures   int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS citation_failures      int;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS entailment_min_score   real;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS degraded              boolean NOT NULL DEFAULT false;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS degraded_reason       text;

-- Cost and latency, per stage, so a p95 is a query rather than a re-run.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS retrieval_ms   integer;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS generation_ms  integer;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS verify_ms      integer;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS total_ms       integer;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS prompt_tokens      integer;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS completion_tokens  integer;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS cost_usd           numeric(12, 6);

-- Retrieval, kept so a wrong answer can be diagnosed without re-running the search it came from.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS retrieved_chunk_ids  bigint[];
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS cited_chunk_ids      bigint[];
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS cited_speech_ids     text[];
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS cited_debate_ids     text[];
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS speaker_mnis_ids     int[];
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS division_ids         int[];
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS retrieved_count      int;

-- Provenance of the evaluation itself: which gold item, which split, which session produced the row.
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS gold_item_id   text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS gold_split     text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS gold_category  text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS session_label  text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS run_id         text;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS is_evaluation  boolean NOT NULL DEFAULT false;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS updated_at     timestamptz NOT NULL DEFAULT now();
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS hit_count      integer NOT NULL DEFAULT 0;
ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS last_hit_at    timestamptz;

-- Backfill.
UPDATE answer_cache SET
    corpus_version        = COALESCE(corpus_version, 'unknown-pre-e2'),
    retrieval_config_hash = COALESCE(retrieval_config_hash, 'unknown-pre-e2'),
    recipe                = COALESCE(recipe, 'unknown-pre-e2'),
    prompt_version        = COALESCE(prompt_version, 'unknown-pre-e2'),
    model_id              = COALESCE(model_id, 'unknown-pre-e2'),
    verified_at           = COALESCE(verified_at, created_at, now()),
    verifier              = COALESCE(verifier, 'none-pre-e2'),
    outcome               = COALESCE(outcome, 'answer')
WHERE corpus_version IS NULL OR retrieval_config_hash IS NULL OR recipe IS NULL
   OR prompt_version IS NULL OR model_id IS NULL OR verified_at IS NULL OR verifier IS NULL
   OR outcome IS NULL;

-- Now the constraints.
ALTER TABLE answer_cache ALTER COLUMN corpus_version        SET NOT NULL;
ALTER TABLE answer_cache ALTER COLUMN retrieval_config_hash SET NOT NULL;
ALTER TABLE answer_cache ALTER COLUMN recipe                SET NOT NULL;
ALTER TABLE answer_cache ALTER COLUMN prompt_version        SET NOT NULL;
ALTER TABLE answer_cache ALTER COLUMN model_id              SET NOT NULL;
ALTER TABLE answer_cache ALTER COLUMN verified_at           SET NOT NULL;
ALTER TABLE answer_cache ALTER COLUMN verifier              SET NOT NULL;
-- NOT NULL rather than left to the CHECK below.
ALTER TABLE answer_cache ALTER COLUMN outcome               SET NOT NULL;

-- An answer or a reasoned abstention, never both and never neither — the same invariant the Pydantic contract enforces
-- in `Answer`, restated here because a second writer could reach this table without going through it.
ALTER TABLE answer_cache DROP CONSTRAINT IF EXISTS answer_cache_outcome_ck;
ALTER TABLE answer_cache ADD CONSTRAINT answer_cache_outcome_ck
    CHECK (outcome IN ('answer', 'clarify', 'abstain'));

-- `abstention_reason IS NOT NULL` is not redundant with the IN list, for the same three-valued reason: with a NULL
-- reason the first branch evaluates to NULL and the second to false, NULL OR false is NULL, and the row is accepted.
ALTER TABLE answer_cache DROP CONSTRAINT IF EXISTS answer_cache_abstention_ck;
ALTER TABLE answer_cache ADD CONSTRAINT answer_cache_abstention_ck CHECK (
    (outcome = 'abstain' AND abstention_reason IS NOT NULL AND abstention_reason IN
        ('not_in_record', 'outside_window', 'ambiguous', 'false_premise', 'insufficient_evidence'))
    OR (outcome <> 'abstain' AND abstention_reason IS NULL)
);

-- A cached answer that dropped every claim is an abstention wearing an answer's clothes.
ALTER TABLE answer_cache DROP CONSTRAINT IF EXISTS answer_cache_verified_ck;
ALTER TABLE answer_cache ADD CONSTRAINT answer_cache_verified_ck CHECK (
    claims_verified IS NULL OR claim_count IS NULL
    OR (claims_verified >= 0 AND claims_verified <= claim_count)
);

-- The real cache key.
ALTER TABLE answer_cache DROP CONSTRAINT IF EXISTS answer_cache_pkey;
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conrelid = 'answer_cache'::regclass AND contype = 'p') THEN
        ALTER TABLE answer_cache ADD PRIMARY KEY (id);
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS answer_cache_provenance_key ON answer_cache
    (query_hash, corpus_version, retrieval_config_hash, recipe, prompt_version, model_id);
-- The lookup a cache read actually performs, covering the key without touching the heap.
CREATE INDEX IF NOT EXISTS answer_cache_query_idx ON answer_cache (query_hash);

-- Indexes chosen for the questions that get asked when something is wrong.
CREATE INDEX IF NOT EXISTS answer_cache_route_idx        ON answer_cache (route, outcome);
CREATE INDEX IF NOT EXISTS answer_cache_outcome_idx      ON answer_cache (outcome, abstention_reason);
CREATE INDEX IF NOT EXISTS answer_cache_created_idx      ON answer_cache (created_at DESC);
CREATE INDEX IF NOT EXISTS answer_cache_window_idx       ON answer_cache (window_label, asked_house);
CREATE INDEX IF NOT EXISTS answer_cache_gold_idx         ON answer_cache (gold_split, gold_category)
    WHERE gold_item_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS answer_cache_run_idx          ON answer_cache (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS answer_cache_degraded_idx     ON answer_cache (degraded) WHERE degraded;
CREATE INDEX IF NOT EXISTS answer_cache_config_idx       ON answer_cache (retrieval_config_hash, corpus_version);
CREATE INDEX IF NOT EXISTS answer_cache_model_idx        ON answer_cache (model_id, prompt_version);
CREATE INDEX IF NOT EXISTS answer_cache_latency_idx      ON answer_cache (route, total_ms);
-- GIN over the cited ids: "which cached answers cite this speech" is the first question asked when a chunk turns out to
-- be wrong, and without this it is a sequential scan over every answer ever given.
CREATE INDEX IF NOT EXISTS answer_cache_cited_chunks_idx  ON answer_cache USING gin (cited_chunk_ids);
CREATE INDEX IF NOT EXISTS answer_cache_cited_speeches_idx ON answer_cache USING gin (cited_speech_ids);
CREATE INDEX IF NOT EXISTS answer_cache_speakers_idx      ON answer_cache USING gin (speaker_mnis_ids);
CREATE INDEX IF NOT EXISTS answer_cache_filters_idx       ON answer_cache USING gin (resolved_filters);
CREATE INDEX IF NOT EXISTS answer_cache_facts_idx         ON answer_cache USING gin (structured_facts);

COMMENT ON TABLE answer_cache IS
    'One row per answer given, with the provenance needed to decide whether it may be reused. The cache key is '
    '(query_hash, corpus_version, retrieval_config_hash, recipe, prompt_version, model_id): the same question '
    'answered under a different corpus or a different frozen retrieval config is a different question.';
COMMENT ON COLUMN answer_cache.retrieval_config_hash IS
    'The frozen retrieval configuration this answer was produced under. E1 froze it at e57c6cdfd8793d76 with no '
    'arm adopted. A cache hit across two hashes would serve an answer from a system that no longer exists.';
COMMENT ON COLUMN answer_cache.corpus_version IS
    'Which corpus the answer was produced against. W5 added 351,989 chunks, so the same question over 1,085,867 '
    'chunks and over 1,437,856 is not the same question.';
COMMENT ON COLUMN answer_cache.verifier IS
    'Who checked this answer. NOT NULL: a cache is the one place an unverified answer can outlive the process '
    'that failed to check it, and truth rule 3 says unverified text is never shown.';
COMMENT ON COLUMN answer_cache.verifier_precision IS
    'fp32 or bf16, from the §5.5 agreement benchmark. Flan-T5 is precision-sensitive, so the precision an '
    'answer was checked at is part of what checked it.';
COMMENT ON COLUMN answer_cache.outcome IS
    'answer | clarify | abstain. Three values, not two: clarify is a first-class outcome and collapsing it into '
    'abstain would hide the case the system handles best.';
COMMENT ON COLUMN answer_cache.degraded IS
    'True when a fallback path produced this answer (no entailment model, GPU service down, deterministic '
    'checks only). Indexed, because "how often did we degrade" must be answerable after the fact.';
COMMENT ON COLUMN answer_cache.chunk_count IS
    'Corpus size in chunks at answer time. Recorded alongside corpus_version because a label can be reused and '
    'a count cannot.';

COMMIT;
