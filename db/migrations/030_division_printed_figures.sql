-- 030: Hansard's printed division figures beside the Votes API's.
ALTER TABLE division ADD COLUMN IF NOT EXISTS printed_ayes integer;
ALTER TABLE division ADD COLUMN IF NOT EXISTS printed_noes integer;
ALTER TABLE division DROP CONSTRAINT IF EXISTS division_printed_pair_ck;
ALTER TABLE division ADD CONSTRAINT division_printed_pair_ck
    CHECK ((printed_ayes IS NULL) = (printed_noes IS NULL) AND coalesce(printed_ayes, 0) >= 0
           AND coalesce(printed_noes, 0) >= 0);
COMMENT ON COLUMN division.printed_ayes IS 'Hansard''s printed Ayes (Contents), where its division record gives them (D-069)';
COMMENT ON COLUMN division.printed_noes IS 'Hansard''s printed Noes (Not-Contents), where its division record gives them (D-069)';

-- The served view was created as SELECT *, which Postgres expands once: re-created, it carries the new columns.
CREATE OR REPLACE VIEW served.division AS
    SELECT * FROM public.division
     WHERE division_date < served.sealed_from() OR division_date > served.sealed_to();
