-- 022: English Votes for English Laws, as recorded.

BEGIN;

CREATE OR REPLACE FUNCTION pg_temp.add_constraint(p_table text, p_name text, p_definition text) RETURNS void AS $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = p_name) THEN
        EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I %s', p_table, p_name, p_definition);
    END IF;
END $$ LANGUAGE plpgsql;

ALTER TABLE division ADD COLUMN IF NOT EXISTS evel_type text;
ALTER TABLE division ADD COLUMN IF NOT EXISTS evel_country text;
ALTER TABLE division ADD COLUMN IF NOT EXISTS double_majority_ayes integer;
ALTER TABLE division ADD COLUMN IF NOT EXISTS double_majority_noes integer;

SELECT pg_temp.add_constraint('division', 'division_double_majority_ck',
    'CHECK ((double_majority_ayes IS NULL OR double_majority_ayes >= 0)
        AND (double_majority_noes IS NULL OR double_majority_noes >= 0)
        AND ((double_majority_ayes IS NULL AND double_majority_noes IS NULL) OR house = ''Commons''))');

UPDATE division SET
    evel_type            = nullif(btrim(raw->>'EVELType'), ''),
    evel_country         = nullif(btrim(raw->>'EVELCountry'), ''),
    double_majority_ayes = (raw->>'DoubleMajorityAyeCount')::integer,
    double_majority_noes = (raw->>'DoubleMajorityNoCount')::integer
WHERE house = 'Commons' AND raw IS NOT NULL
  AND (evel_type, evel_country, double_majority_ayes, double_majority_noes)
      IS DISTINCT FROM (nullif(btrim(raw->>'EVELType'), ''), nullif(btrim(raw->>'EVELCountry'), ''),
                        (raw->>'DoubleMajorityAyeCount')::integer, (raw->>'DoubleMajorityNoCount')::integer);

COMMENT ON COLUMN division.evel_type IS 'Commons Votes API EVELType (English Votes for English Laws, 2015-2021); NULL when blank.';
COMMENT ON COLUMN division.evel_country IS 'Commons Votes API EVELCountry: whose MPs formed the second majority; NULL when blank.';
COMMENT ON COLUMN division.double_majority_ayes IS 'EVEL second count, Ayes, of the MPs in evel_country. result is NULL where the two majorities disagree.';
COMMENT ON COLUMN division.double_majority_noes IS 'EVEL second count, Noes, of the MPs in evel_country.';

COMMIT;
