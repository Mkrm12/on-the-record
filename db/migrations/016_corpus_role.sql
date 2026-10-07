-- The `corpus` window role — loaded and searchable, never evaluated, carrying no gold.

BEGIN;

ALTER TABLE corpus_window DROP CONSTRAINT IF EXISTS corpus_window_role_check;
ALTER TABLE corpus_window ADD CONSTRAINT corpus_window_role_check
    CHECK (role IN ('dev', 'dev_test', 'heldout', 'candidate', 'corpus'));

COMMIT;
