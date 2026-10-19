-- Seven `answer_cache` columns that nothing has ever written or read.

BEGIN;

ALTER TABLE answer_cache
    DROP COLUMN IF EXISTS attribution_failures,
    DROP COLUMN IF EXISTS citation_failures,
    DROP COLUMN IF EXISTS cited_debate_ids,
    DROP COLUMN IF EXISTS division_ids,
    DROP COLUMN IF EXISTS entailment_min_score,
    DROP COLUMN IF EXISTS number_failures,
    DROP COLUMN IF EXISTS quote_failures;

COMMIT;
