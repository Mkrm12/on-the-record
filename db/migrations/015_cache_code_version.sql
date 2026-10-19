-- The answer cache keys on the code that produced the answer.

BEGIN;

ALTER TABLE answer_cache ADD COLUMN IF NOT EXISTS code_version text;

DROP INDEX IF EXISTS answer_cache_provenance_key;
-- NULLS NOT DISTINCT (PostgreSQL 15+; this server is 16): in an ordinary unique index two NULLs are distinct, so two
-- rows with no code_version would BOTH be admitted and the cache would stop deduplicating — caught by
-- test_the_identical_answer_under_identical_provenance_is_one_row the moment the column joined the key.
CREATE UNIQUE INDEX IF NOT EXISTS answer_cache_provenance_code_key ON answer_cache
    (query_hash, corpus_version, retrieval_config_hash, recipe, prompt_version, model_id, code_version)
    NULLS NOT DISTINCT;

COMMENT ON COLUMN answer_cache.code_version IS
    'Fingerprint of the code that produced the answer (answer.keys.code_version). Part of the cache key since '
    'Session 18; NULL on rows stored before, which therefore never match a lookup.';

COMMIT;
