-- The trace table: one row per request, whether or not it produced an answer.

BEGIN;

CREATE TABLE IF NOT EXISTS answer_trace (
    id              bigserial PRIMARY KEY,
    created_at      timestamptz NOT NULL DEFAULT now(),

    -- what was asked
    query_hash      text NOT NULL,
    question        text,
    route           text NOT NULL,
    route_by        text,
    route_why       text,
    resolved_filters jsonb NOT NULL DEFAULT '{}'::jsonb,

    -- which system answered
    corpus_version  text NOT NULL,
    retrieval_config_hash text NOT NULL,
    recipe          text,
    prompt_version  text,
    model_id        text,
    served_model    text,
    fingerprint     text,
    verifier        text,
    verifier_precision text,

    -- the outcome
    outcome         text NOT NULL,
    abstention_reason text,
    cache_hit       boolean NOT NULL DEFAULT false,
    degraded        boolean NOT NULL DEFAULT false,
    degraded_reason text,
    failed_reason   text,

    -- what happened on the way
    retrieval_hits  integer,
    cards           integer,
    cards_dropped   integer,
    claims_generated integer,
    claims_rejected integer,
    claims_verified integer,
    claims_dropped  integer,
    repairs         integer NOT NULL DEFAULT 0,
    rewrites        integer NOT NULL DEFAULT 0,

    -- what it cost
    retrieval_ms    integer,
    generation_ms   integer,
    verify_ms       integer,
    total_ms        integer,
    prompt_tokens   integer,
    completion_tokens integer,
    cached_tokens   integer,
    cost_usd        numeric(12, 6),
    llm_calls       integer NOT NULL DEFAULT 0,

    -- provenance of the run itself
    run_id          text,
    session_label   text,
    gold_item_id    text,
    gold_split      text,
    gold_category   text,
    is_evaluation   boolean NOT NULL DEFAULT false
);

ALTER TABLE answer_trace DROP CONSTRAINT IF EXISTS answer_trace_outcome_ck;
ALTER TABLE answer_trace ADD CONSTRAINT answer_trace_outcome_ck
    CHECK (outcome IN ('answer', 'clarify', 'abstain', 'error'));

-- 'error' exists here and not in `answer_cache` for a reason: a request that fell over is a fact about the system worth
-- keeping, and is exactly what must never be cached as though it were an answer.
ALTER TABLE answer_trace DROP CONSTRAINT IF EXISTS answer_trace_abstention_ck;
ALTER TABLE answer_trace ADD CONSTRAINT answer_trace_abstention_ck CHECK (
    abstention_reason IS NULL OR abstention_reason IN
        ('not_in_record', 'outside_window', 'ambiguous', 'false_premise', 'insufficient_evidence')
);

CREATE INDEX IF NOT EXISTS answer_trace_created_idx  ON answer_trace (created_at DESC);
CREATE INDEX IF NOT EXISTS answer_trace_route_idx    ON answer_trace (route, outcome);
CREATE INDEX IF NOT EXISTS answer_trace_run_idx      ON answer_trace (run_id) WHERE run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS answer_trace_gold_idx     ON answer_trace (gold_split, gold_category)
    WHERE gold_item_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS answer_trace_query_idx    ON answer_trace (query_hash);
CREATE INDEX IF NOT EXISTS answer_trace_latency_idx  ON answer_trace (route, total_ms);
CREATE INDEX IF NOT EXISTS answer_trace_degraded_idx ON answer_trace (degraded) WHERE degraded;
CREATE INDEX IF NOT EXISTS answer_trace_failed_idx   ON answer_trace (failed_reason)
    WHERE failed_reason IS NOT NULL;

COMMENT ON TABLE answer_trace IS
    'One row per request, including the ones that produced no cacheable answer. `answer_cache` is one row per '
    'distinct answer; this is the population that tells over-abstention from a quiet failure (endgame §6.4).';
COMMENT ON COLUMN answer_trace.question IS
    'Stored only when the caller passes it. §6.4: the question text is kept only when a config flag allows it. An '
    'evaluation run sets it because the gold text is already in the repository; a production request does not.';
COMMENT ON COLUMN answer_trace.failed_reason IS
    'Why a request produced nothing. Exists because a call that returns empty content is still billed, and an '
    'earlier budget guard recorded $0.00000 for exactly those requests.';
COMMENT ON COLUMN answer_trace.rewrites IS
    'Corrective query rewrites used, capped at 2 by §6.4. A guard is only a guard if its use is visible afterwards.';

COMMIT;
