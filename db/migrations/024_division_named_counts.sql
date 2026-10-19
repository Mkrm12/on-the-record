-- 024: the names each lobby's list records, and one definition of a reconciled tally.

BEGIN;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM division_vote v JOIN division d ON d.id = v.division_id
                WHERE d.house = 'Commons' AND v.lobby = 'conflict' AND v.is_teller) THEN
        RAISE EXCEPTION 'A Commons member both told and was named in the other lobby, so the named counts cannot be '
            'derived from division_vote. Reload those divisions (scripts/ingest_divisions.py), then migrate again.';
    END IF;
END $$;

ALTER TABLE division ADD COLUMN IF NOT EXISTS ayes_named integer;
ALTER TABLE division ADD COLUMN IF NOT EXISTS noes_named integer;
ALTER TABLE division DROP CONSTRAINT IF EXISTS division_named_counts_ck;

WITH named AS (
    SELECT d.id,
           count(*) FILTER (WHERE (v.lobby = 'aye' AND (d.house = 'Lords' OR NOT v.is_teller))
                               OR (v.lobby = 'conflict' AND 'listed_aye' = ANY (v.flags)))::integer AS ayes_named,
           count(*) FILTER (WHERE (v.lobby = 'no' AND (d.house = 'Lords' OR NOT v.is_teller))
                               OR (v.lobby = 'conflict' AND 'listed_no' = ANY (v.flags)))::integer AS noes_named
      FROM division d
      LEFT JOIN division_vote v ON v.division_id = d.id
     GROUP BY d.id
), derived AS (
    SELECT d.id, n.ayes_named, n.noes_named,
           CASE WHEN d.ayes IS NULL OR d.noes IS NULL THEN NULL
                ELSE n.ayes_named = d.ayes AND n.noes_named = d.noes END AS tally_reconciled,
           -- the loader's order: each side's mismatch first, then every other anomaly as it was recorded
           ARRAY(SELECT m FROM (VALUES
                    (1, CASE WHEN d.ayes IS NOT NULL AND n.ayes_named <> d.ayes
                             THEN format('aye_count_mismatch:official=%s,named=%s', d.ayes, n.ayes_named) END),
                    (2, CASE WHEN d.noes IS NOT NULL AND n.noes_named <> d.noes
                             THEN format('no_count_mismatch:official=%s,named=%s', d.noes, n.noes_named) END)
                 ) AS sides(k, m) WHERE m IS NOT NULL ORDER BY k)
           || ARRAY(SELECT a FROM unnest(d.anomalies) WITH ORDINALITY AS kept(a, i)
                     WHERE a !~ '^(aye|no)_count_mismatch:' ORDER BY i) AS anomalies
      FROM division d JOIN named n ON n.id = d.id
)
UPDATE division d SET
    ayes_named = x.ayes_named,
    noes_named = x.noes_named,
    tally_reconciled = x.tally_reconciled,
    anomalies = x.anomalies
  FROM derived x
 WHERE x.id = d.id
   AND (d.ayes_named, d.noes_named, d.tally_reconciled, d.anomalies)
       IS DISTINCT FROM (x.ayes_named, x.noes_named, x.tally_reconciled, x.anomalies);

ALTER TABLE division ADD CONSTRAINT division_named_counts_ck CHECK (
    (ayes_named IS NULL OR ayes_named >= 0) AND (noes_named IS NULL OR noes_named >= 0)
    AND (ayes_named IS NULL OR noes_named IS NULL OR ayes IS NULL OR noes IS NULL
         OR tally_reconciled IS NOT DISTINCT FROM (ayes_named = ayes AND noes_named = noes)));

COMMENT ON COLUMN division.ayes_named IS 'Members the Ayes (Contents) list names, as the figure counts them: Commons tellers apart, Lords tellers in; each once; a member in both lobbies counts in each (D-037).';
COMMENT ON COLUMN division.noes_named IS 'Members the Noes (Not-Contents) list names, by the same definition as ayes_named (D-037).';
COMMENT ON COLUMN division.tally_reconciled IS 'ayes_named = ayes AND noes_named = noes; NULL when a figure is not recorded (D-037). Nothing is allowed for: a list two short has lost two names.';

COMMIT;
