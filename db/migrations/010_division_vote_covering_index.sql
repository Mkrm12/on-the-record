-- A covering index for "how did this member vote", the last of the tools short of the 25 ms target.

BEGIN;

CREATE INDEX IF NOT EXISTS division_vote_member_covering_idx
    ON division_vote (mnis_id, division_id)
    INCLUDE (lobby, is_teller, proxy_name)
    WHERE mnis_id IS NOT NULL;

COMMENT ON INDEX division_vote_member_covering_idx IS
    'Covering index for member_walk''s vote half. Before it, (mnis_id) alone located the rows but carried none '
    'of the columns read, so each one cost a heap visit into a 292,480-row table.';

COMMIT;

ANALYZE division_vote;
