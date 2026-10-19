-- The second pass of the foundation audit: keys that could not hold, and what the columns mean.

BEGIN;

CREATE OR REPLACE FUNCTION pg_temp.add_constraint(p_table text, p_name text, p_definition text) RETURNS void AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = p_name) THEN
        EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I %s', p_table, p_name, p_definition);
    END IF;
END $$ LANGUAGE plpgsql;

-- ---- procedure_event: a sound idempotency key -----------------------------------------------------------------------

ALTER TABLE procedure_event ALTER COLUMN event_type SET NOT NULL;
ALTER TABLE procedure_event ALTER COLUMN debate_id SET NOT NULL;
SELECT pg_temp.add_constraint('procedure_event', 'procedure_event_rule_key_ck',
    'CHECK (rule IS NULL OR (speech_id IS NOT NULL AND char_start IS NOT NULL))');

-- ---- columns the loader always writes
-- ---------------------------------------------------------------------------------

ALTER TABLE debate ALTER COLUMN corpus_version SET NOT NULL;
ALTER TABLE hansard_correction ALTER COLUMN window_id SET NOT NULL;
ALTER TABLE bill ALTER COLUMN title SET NOT NULL;
SELECT pg_temp.add_constraint('bill', 'bill_title_clean_ck', 'CHECK (title = btrim(title) AND title <> '''')');

-- ---- what the columns mean
-- ----------------------------------------------------------------------------------------------

COMMENT ON COLUMN division.bill_link IS
    'How `bill_stage_id` was found by `ingest.bills.link_division_stage`. Linked: title_stage, title_date, '
    'title_amendment (each optionally +numbered: "(No. N)" ignored, or +fuzzy: same words in any order), debate, '
    'debate_weak. Unlinked: none (names a bill, no stage found), ambiguous (refused), not_a_bill (names no bill, or a '
    'deferred division on an instrument). R15 in the integrity gate holds the vocabulary both ways.';
COMMENT ON COLUMN bill_stage.debate_id IS
    'DERIVED: the debate of the stage''s earliest linked sitting, set and cleared by `ingest.bills.link_all` (D19).';
COMMENT ON COLUMN member_id_map.missing_endpoints IS
    'Member API endpoints whose fetch FAILED (the fetch manifest records `<endpoint>:failed`). A 404 is absence '
    '(no LatestElectionResult for a peer or a former MP) and is never listed here.';
COMMENT ON COLUMN member_id_map.is_active IS
    'False for a membership that has ended, whatever the API''s status says: it keeps "Current Member" on MPs who '
    'left at a dissolution. NULL only when the API gives neither a status nor an end.';
COMMENT ON COLUMN chunk.window_id IS
    'Copied from the debate by the chunker (C-10) so a filtered search never misses a new chunk; D05 audits it.';

COMMIT;
