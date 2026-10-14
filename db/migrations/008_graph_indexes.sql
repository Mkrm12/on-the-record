-- Indexes on the foreign keys the relationship tools walk.

BEGIN;

-- The big one.
CREATE INDEX IF NOT EXISTS speech_debate_idx     ON speech (debate_id);
-- Ordered by seq, because a debate's speeches are always read in order and the roster needs min(seq).
CREATE INDEX IF NOT EXISTS speech_debate_seq_idx ON speech (debate_id, seq) WHERE NOT is_boilerplate;
-- `member → speeches`, the other direction of the same edge.
CREATE INDEX IF NOT EXISTS speech_mnis_idx       ON speech (mnis_id) WHERE mnis_id IS NOT NULL;

-- The bill spine.
CREATE INDEX IF NOT EXISTS bill_stage_sitting_stage_idx ON bill_stage_sitting (bill_stage_id);
CREATE INDEX IF NOT EXISTS bill_stage_sitting_bill_idx  ON bill_stage_sitting (bill_id);

-- `debate_bill` gets nothing, because it is a VIEW, not a table: `SELECT...

-- A division reaches a bill through its stage as well as through its debate, and only the debate side was indexed, so
-- the stage side was a scan of every division.
CREATE INDEX IF NOT EXISTS division_bill_stage_idx ON division (bill_stage_id) WHERE bill_stage_id IS NOT NULL;

-- `division_vote` is indexed on (division_id, mnis_id) and on mnis_id, which covers both directions of the vote edge.
CREATE INDEX IF NOT EXISTS division_vote_teller_idx ON division_vote (division_id) WHERE is_teller;

CREATE INDEX IF NOT EXISTS member_office_mnis_idx ON member_office (mnis_id, start_date, end_date);

-- Member attribute lookups at a date: party_at_date and office_held_at_date are asked once per claim, so they are on the
-- hot path of every answer that names a person.

-- Bill lookup by title is how a question names a bill, and there was nothing for it.
CREATE INDEX IF NOT EXISTS bill_title_lower_idx ON bill (lower(title));

COMMENT ON INDEX speech_debate_idx IS
    'The most-walked edge in the system. Absent until 008, which made every per-debate aggregate a sequential '
    'scan of 154,881 rows and put bill_walk''s p95 within 1.8 ms of the 100 ms gate.';

COMMIT;

-- Outside the transaction: the planner cannot use what it does not know the size of, and a fresh index on a 154,881-row
-- table with stale statistics is an index the planner may decline.
ANALYZE speech;
ANALYZE bill_stage_sitting;
ANALYZE division;
ANALYZE division_vote;
ANALYZE member_id_map;
