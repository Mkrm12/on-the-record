-- debate.section_order orders debates within a SECTION of a sitting (Commons Chamber, Westminster Hall, Grand
-- Committee,...), each with its own sequence: unique per (date, section) in all 396 debates (checked), but colliding
-- across sections of the same date and House.

BEGIN;

DROP INDEX IF EXISTS debate_sitting_order_idx;
CREATE INDEX IF NOT EXISTS debate_section_order_idx ON debate (date, section, section_order);

COMMENT ON COLUMN debate.section_order IS 'Hansard DebateSectionOrder: position of the debate within its section of the sitting; unique per (date, section), NOT per (date, house).';
COMMENT ON COLUMN debate.section_id    IS 'Hansard DebateSectionId (numeric); debate.id is the DebateSectionExtId GUID.';
COMMENT ON COLUMN speech.hrs_tag       IS 'Hansard HRSTag as served. Case varies for the same tag (hs_Para / hs_para): compare with lower(). Question and hs_TabledBy mark written-question items.';
COMMENT ON COLUMN chunk.column_basis   IS 'marker = read from a column marker earlier in the same speech; carried = last marker of an earlier speech in the debate; inferred = before the debate''s first marker (that marker''s column minus one).';
COMMENT ON COLUMN member_id_map.date_of_birth IS 'Only when Wikidata gives a full day. See dob_precision / dob_earliest / dob_latest / member_age for anything less precise.';

COMMIT;
