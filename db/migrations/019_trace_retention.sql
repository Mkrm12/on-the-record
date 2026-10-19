-- One row per request, forever, is millions a year in production.

BEGIN;

CREATE TABLE IF NOT EXISTS answer_trace_daily (
    day               date    NOT NULL,
    route             text    NOT NULL,
    outcome           text    NOT NULL,
    requests          bigint  NOT NULL DEFAULT 0 CHECK (requests >= 0),
    degraded          bigint  NOT NULL DEFAULT 0 CHECK (degraded >= 0),
    cache_hits        bigint  NOT NULL DEFAULT 0 CHECK (cache_hits >= 0),
    failed            bigint  NOT NULL DEFAULT 0 CHECK (failed >= 0),
    llm_calls         bigint  NOT NULL DEFAULT 0 CHECK (llm_calls >= 0),
    prompt_tokens     bigint  NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
    completion_tokens bigint  NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
    cost_usd          numeric NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    total_ms_sum      bigint  NOT NULL DEFAULT 0 CHECK (total_ms_sum >= 0),
    total_ms_max      integer,
    PRIMARY KEY (day, route, outcome)
);

COMMENT ON TABLE answer_trace_daily IS
    'Ordinary (non-evaluation) answer_trace rows older than the retention window, folded per day, route and outcome '
    'by scripts/trace_retention.py. Evaluation rows are never folded or deleted.';

COMMIT;
