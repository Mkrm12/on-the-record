-- The foundation audit's schema hardening: the database refuses bad data instead of trusting the loader.

BEGIN;

CREATE OR REPLACE FUNCTION pg_temp.add_constraint(p_table text, p_name text, p_definition text) RETURNS void AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = p_name) THEN
        EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I %s', p_table, p_name, p_definition);
    END IF;
END $$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION pg_temp.validate(p_table text, p_name text) RETURNS void AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = p_name AND NOT convalidated) THEN
        EXECUTE format('ALTER TABLE %I VALIDATE CONSTRAINT %I', p_table, p_name);
    END IF;
END $$ LANGUAGE plpgsql;

-- ---- foreign keys: structural references, loaded in order within a window ----------------------------------------

SELECT pg_temp.add_constraint('chunk', 'chunk_debate_fk',
    'FOREIGN KEY (debate_id) REFERENCES debate(id) NOT VALID');
SELECT pg_temp.add_constraint('chunk', 'chunk_window_fk',
    'FOREIGN KEY (window_id) REFERENCES corpus_window(window_id) NOT VALID');
SELECT pg_temp.add_constraint('speech', 'speech_window_fk',
    'FOREIGN KEY (window_id) REFERENCES corpus_window(window_id) NOT VALID');
SELECT pg_temp.add_constraint('division', 'division_debate_fk',
    'FOREIGN KEY (debate_id) REFERENCES debate(id) ON DELETE SET NULL NOT VALID');
SELECT pg_temp.add_constraint('procedure_event', 'procedure_event_debate_fk',
    'FOREIGN KEY (debate_id) REFERENCES debate(id) NOT VALID');
SELECT pg_temp.add_constraint('bill_stage', 'bill_stage_debate_fk',
    'FOREIGN KEY (debate_id) REFERENCES debate(id) ON DELETE SET NULL NOT VALID');
SELECT pg_temp.add_constraint('bill_stage_sitting', 'bill_stage_sitting_debate_fk',
    'FOREIGN KEY (debate_id) REFERENCES debate(id) ON DELETE SET NULL NOT VALID');
SELECT pg_temp.add_constraint('hansard_correction', 'hansard_correction_speech_fk',
    'FOREIGN KEY (corrected_speech_id) REFERENCES speech(id) ON DELETE SET NULL NOT VALID');
SELECT pg_temp.add_constraint('hansard_correction', 'hansard_correction_window_fk',
    'FOREIGN KEY (window_id) REFERENCES corpus_window(window_id) NOT VALID');
SELECT pg_temp.add_constraint('coverage', 'coverage_window_fk',
    'FOREIGN KEY (window_id) REFERENCES corpus_window(window_id) NOT VALID');
SELECT pg_temp.add_constraint('load_report', 'load_report_window_fk',
    'FOREIGN KEY (window_id) REFERENCES corpus_window(window_id) NOT VALID');

SELECT pg_temp.validate('chunk', 'chunk_debate_fk');
SELECT pg_temp.validate('chunk', 'chunk_window_fk');
SELECT pg_temp.validate('speech', 'speech_window_fk');
SELECT pg_temp.validate('division', 'division_debate_fk');
SELECT pg_temp.validate('procedure_event', 'procedure_event_debate_fk');
SELECT pg_temp.validate('bill_stage', 'bill_stage_debate_fk');
SELECT pg_temp.validate('bill_stage_sitting', 'bill_stage_sitting_debate_fk');
SELECT pg_temp.validate('hansard_correction', 'hansard_correction_speech_fk');
SELECT pg_temp.validate('hansard_correction', 'hansard_correction_window_fk');
SELECT pg_temp.validate('coverage', 'coverage_window_fk');
SELECT pg_temp.validate('load_report', 'load_report_window_fk');

-- ---- closed vocabularies ------------------------------------------------------------------------------------------

SELECT pg_temp.add_constraint('debate', 'debate_house_ck', $c$CHECK (house IN ('Commons', 'Lords'))$c$);
SELECT pg_temp.add_constraint('division', 'division_house_ck', $c$CHECK (house IN ('Commons', 'Lords'))$c$);
SELECT pg_temp.add_constraint('chunk', 'chunk_house_ck', $c$CHECK (house IS NULL OR house IN ('Commons', 'Lords'))$c$);
SELECT pg_temp.add_constraint('bill_stage', 'bill_stage_house_ck',
    $c$CHECK (house IS NULL OR house IN ('Commons', 'Lords'))$c$);
SELECT pg_temp.add_constraint('bill_stage_sitting', 'bill_stage_sitting_house_ck',
    $c$CHECK (house IS NULL OR house IN ('Commons', 'Lords'))$c$);
SELECT pg_temp.add_constraint('division', 'division_result_ck',
    $c$CHECK (result IS NULL OR result IN ('agreed', 'negatived', 'tied'))$c$);

-- ---- ranges and consistency ---------------------------------------------------------------------------------------

SELECT pg_temp.add_constraint('division', 'division_tally_nonnegative_ck',
    'CHECK ((ayes IS NULL OR ayes >= 0) AND (noes IS NULL OR noes >= 0))');
-- `>=` on purpose: a casting vote is not in the tellers' figures (see the header).
SELECT pg_temp.add_constraint('division', 'division_result_consistent_ck', $c$CHECK (
    result IS NULL OR ayes IS NULL OR noes IS NULL
    OR (result = 'agreed' AND ayes >= noes)
    OR (result = 'negatived' AND noes >= ayes)
    OR (result = 'tied' AND ayes = noes))$c$);
SELECT pg_temp.add_constraint('member_office', 'member_office_interval_ck',
    'CHECK (end_date IS NULL OR start_date IS NULL OR end_date >= start_date)');
SELECT pg_temp.add_constraint('member_id_map', 'member_membership_interval_ck',
    'CHECK (membership_end_date IS NULL OR membership_start_date IS NULL OR membership_end_date >= membership_start_date)');
SELECT pg_temp.add_constraint('chunk', 'chunk_position_ck', 'CHECK (sent_idx >= 0 AND seq >= 0)');

-- ---- NOT NULL where the loader now guarantees the value ----------------------------------------------------------
-- `shape_day` quarantines an item with no sitting date or House (it used to crash on the first and store NULL for the
-- second) and stores an absent text as '' — so these are backstops that should never fire.

ALTER TABLE debate ALTER COLUMN date SET NOT NULL;
ALTER TABLE debate ALTER COLUMN house SET NOT NULL;
ALTER TABLE debate ALTER COLUMN window_id SET NOT NULL;
ALTER TABLE speech ALTER COLUMN debate_id SET NOT NULL;
ALTER TABLE speech ALTER COLUMN window_id SET NOT NULL;
ALTER TABLE speech ALTER COLUMN text SET NOT NULL;
ALTER TABLE division ALTER COLUMN house SET NOT NULL;
ALTER TABLE division ALTER COLUMN division_date SET NOT NULL;
ALTER TABLE division_vote ALTER COLUMN lobby SET NOT NULL;
ALTER TABLE bill_stage ALTER COLUMN bill_id SET NOT NULL;

-- ---- documentation that travels with the schema -------------------------------------------------------------------

COMMENT ON COLUMN hansard_correction.debate_id IS
    'The id of the Written/Ministerial Corrections SECTION the correction is published in — NOT the debate being '
    'corrected, and it resolves to no debate row. The corrected statement is `corrected_speech_id`, linked by '
    '`ingest.corrections` from the correction''s own markup (0 of 304 were linked before Session 18).';
COMMENT ON COLUMN hansard_correction.link_basis IS
    'How `corrected_speech_id` was found: shingle:<overlap>, or why it was not — weak:<best>, ambiguous:<best>, '
    'no_candidate, empty_body.';
COMMENT ON COLUMN bill_stage_sitting.debate_link IS
    'How `debate_id` was found (title_exact, title_prefix, +multi), or formal_no_debate for a stage never debated '
    '(a first reading, Royal Assent), or none. `none` and `formal_no_debate` are deliberately distinct.';
COMMENT ON COLUMN division.bill_link IS
    'How `bill_stage_id` was found by `ingest.bills.link_division_stage`: title_stage, title_date, debate, '
    'ambiguous (refused), or none. The Corpus v2 loader never ran this linker before Session 18 (0 of 586 linked).';

COMMIT;
