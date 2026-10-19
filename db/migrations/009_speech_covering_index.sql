-- A covering index for "what did this member say", the one tool that missed the p95 gate.

BEGIN;

CREATE INDEX IF NOT EXISTS speech_mnis_covering_idx
    ON speech (mnis_id, debate_id, seq)
    INCLUDE (id, speaker_name)
    WHERE NOT is_boilerplate AND mnis_id IS NOT NULL;

COMMENT ON INDEX speech_mnis_covering_idx IS
    'Covering index for member_walk. Before it, returning 200 of a member''s speeches read 2,597 heap blocks '
    'because speech.text came along for the ride; member_walk''s p95 was 183.78 ms against a 100 ms gate.';

COMMIT;

ANALYZE speech;
