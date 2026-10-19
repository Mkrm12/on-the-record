-- Phase 1 completion: member dates of birth (Wikidata), Bills, column citations, Hansard API metadata we had dropped,
-- and the views/functions Phase 2 retrieval will build on.

BEGIN;

-- --------------------------------------------------------------------------- member_id_map: Wikidata-derived facts.
-- date_of_birth is set ONLY when the source gives a full day.
ALTER TABLE member_id_map
    ADD COLUMN IF NOT EXISTS date_of_birth       date,
    ADD COLUMN IF NOT EXISTS dob_precision       text,
    ADD COLUMN IF NOT EXISTS dob_earliest        date,
    ADD COLUMN IF NOT EXISTS dob_latest          date,
    ADD COLUMN IF NOT EXISTS birth_year          int,
    ADD COLUMN IF NOT EXISTS place_of_birth      text,
    ADD COLUMN IF NOT EXISTS date_of_death       date,
    ADD COLUMN IF NOT EXISTS wikidata_id         text,
    ADD COLUMN IF NOT EXISTS wikipedia_url       text,
    ADD COLUMN IF NOT EXISTS twfy_id             text,   -- TheyWorkForYou person number (Wikidata P2171)
    ADD COLUMN IF NOT EXISTS democracy_club_id   text,   -- Democracy Club candidate id (P6465)
    ADD COLUMN IF NOT EXISTS wikidata            jsonb NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS wikidata_fetched_at timestamptz;

ALTER TABLE member_id_map DROP CONSTRAINT IF EXISTS member_dob_precision_ck;
ALTER TABLE member_id_map ADD CONSTRAINT member_dob_precision_ck CHECK (dob_precision IN ('day', 'month', 'year'));
ALTER TABLE member_id_map DROP CONSTRAINT IF EXISTS member_dob_day_ck;
ALTER TABLE member_id_map ADD CONSTRAINT member_dob_day_ck CHECK (date_of_birth IS NULL OR dob_precision = 'day');

CREATE OR REPLACE VIEW member_age AS
SELECT mnis_id, name, date_of_birth, dob_precision, birth_year, dob_earliest, dob_latest, date_of_death,
       CASE WHEN date_of_birth IS NOT NULL THEN date_part('year', age(current_date, date_of_birth))::int END AS age_exact,
       CASE WHEN dob_latest    IS NOT NULL THEN date_part('year', age(current_date, dob_latest))::int    END AS age_min,
       CASE WHEN dob_earliest  IS NOT NULL THEN date_part('year', age(current_date, dob_earliest))::int  END AS age_max
FROM member_id_map;

-- Exact age on a date; NULL unless the full birth date is known (use member_age for ranges).
CREATE OR REPLACE FUNCTION member_age_on(p_mnis_id int, p_date date) RETURNS int
LANGUAGE sql STABLE AS $$
    SELECT date_part('year', age(p_date, date_of_birth))::int FROM member_id_map
    WHERE mnis_id = p_mnis_id AND date_of_birth IS NOT NULL
$$;

-- "Labour (Co-op)" is Labour with a sub-party, not a different party.
CREATE OR REPLACE FUNCTION party_base(p_party text) RETURNS text
LANGUAGE sql IMMUTABLE AS $$
    SELECT regexp_replace(p_party, '\s*\((?:Co-op|Co-operative)\)\s*$', '')
$$;

-- --------------------------------------------------------------------------- Hansard column citations.
ALTER TABLE chunk
    ADD COLUMN IF NOT EXISTS column_number    int,
    ADD COLUMN IF NOT EXISTS column_ref       text,
    ADD COLUMN IF NOT EXISTS column_end_ref   text,   -- set only when the sentence crosses into another column
    ADD COLUMN IF NOT EXISTS column_basis     text,
    ADD COLUMN IF NOT EXISTS embedding_model  text,   -- provenance: re-embed when the model changes
    ADD COLUMN IF NOT EXISTS embedded_at      timestamptz;
ALTER TABLE chunk DROP CONSTRAINT IF EXISTS chunk_column_basis_ck;
ALTER TABLE chunk ADD CONSTRAINT chunk_column_basis_ck CHECK (column_basis IN ('marker', 'carried', 'inferred'));

-- The Phase 2 embedding work queue: only chunks that are worth embedding and are not embedded yet.
CREATE INDEX IF NOT EXISTS chunk_pending_embedding_idx ON chunk (id) WHERE embedding IS NULL AND NOT is_boilerplate;

-- --------------------------------------------------------------------------- Hansard API fields the original ingest
-- dropped. debate.section_order is the running order of debates within a sitting, which the Timeline block cannot do
-- without. ---------------------------------------------------------------------------
ALTER TABLE speech
    ADD COLUMN IF NOT EXISTS item_id          int,
    ADD COLUMN IF NOT EXISTS hrs_tag          text,
    ADD COLUMN IF NOT EXISTS hansard_section  text,
    ADD COLUMN IF NOT EXISTS api_rank         int,
    ADD COLUMN IF NOT EXISTS api_member_name  text,
    ADD COLUMN IF NOT EXISTS meta             jsonb NOT NULL DEFAULT '{}';
ALTER TABLE debate
    ADD COLUMN IF NOT EXISTS section_id       int,
    ADD COLUMN IF NOT EXISTS section_order    int;
CREATE INDEX IF NOT EXISTS debate_sitting_order_idx ON debate (date, house, section_order);

-- --------------------------------------------------------------------------- Bills.
ALTER TABLE bill
    ADD COLUMN IF NOT EXISTS long_title            text,
    ADD COLUMN IF NOT EXISTS former_short_title    text,
    ADD COLUMN IF NOT EXISTS bill_type_id          int,
    ADD COLUMN IF NOT EXISTS originating_house     text,
    ADD COLUMN IF NOT EXISTS current_house         text,
    ADD COLUMN IF NOT EXISTS introduced_session_id int,
    ADD COLUMN IF NOT EXISTS is_act                boolean,
    ADD COLUMN IF NOT EXISTS is_defeated           boolean,
    ADD COLUMN IF NOT EXISTS is_withdrawn          boolean,
    ADD COLUMN IF NOT EXISTS last_update           timestamptz,
    ADD COLUMN IF NOT EXISTS raw                   jsonb       NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS fetched_at            timestamptz NOT NULL DEFAULT now();

ALTER TABLE bill_stage
    ADD COLUMN IF NOT EXISTS stage_type_id  int,
    ADD COLUMN IF NOT EXISTS abbreviation   text,     -- 1R 2R CS RS 3R COLA PM MR WAMR ...
    ADD COLUMN IF NOT EXISTS session_id     int,
    ADD COLUMN IF NOT EXISTS sort_order     int,
    ADD COLUMN IF NOT EXISTS first_sitting  date,
    ADD COLUMN IF NOT EXISTS last_sitting   date;
CREATE INDEX IF NOT EXISTS bill_stage_bill_idx ON bill_stage (bill_id);

CREATE TABLE IF NOT EXISTS bill_stage_sitting (
    id            int  PRIMARY KEY,
    bill_stage_id int  NOT NULL REFERENCES bill_stage(id) ON DELETE CASCADE,
    bill_id       int  NOT NULL REFERENCES bill(id)       ON DELETE CASCADE,
    date          date NOT NULL,
    house         text,
    debate_id     text,
    debate_link   text
);
CREATE INDEX IF NOT EXISTS bill_stage_sitting_date_idx   ON bill_stage_sitting (date);
CREATE INDEX IF NOT EXISTS bill_stage_sitting_debate_idx ON bill_stage_sitting (debate_id);

ALTER TABLE division ADD COLUMN IF NOT EXISTS bill_link text;   -- how bill_stage_id was found
ALTER TABLE division DROP CONSTRAINT IF EXISTS division_bill_stage_fk;
ALTER TABLE division ADD CONSTRAINT division_bill_stage_fk
    FOREIGN KEY (bill_stage_id) REFERENCES bill_stage(id) ON DELETE SET NULL;

CREATE OR REPLACE VIEW bill_sponsor AS
SELECT b.id AS bill_id, b.title AS bill_title,
       (s->'member'->>'memberId')::int AS mnis_id,
       s->'member'->>'name'            AS name,
       s->'member'->>'party'           AS party,
       s->'organisation'->>'name'      AS organisation,
       (s->>'sortOrder')::int          AS sort_order
FROM bill b
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(b.raw->'detail'->'sponsors') = 'array' THEN b.raw->'detail'->'sponsors' ELSE '[]'::jsonb END) s;

CREATE OR REPLACE VIEW debate_bill AS
SELECT t.debate_id, t.bill_id, b.title AS bill_title, st.id AS bill_stage_id, st.stage_name, st.abbreviation, t.date, t.debate_link
FROM bill_stage_sitting t
JOIN bill b        ON b.id  = t.bill_id
JOIN bill_stage st ON st.id = t.bill_stage_id
WHERE t.debate_id IS NOT NULL;

-- --------------------------------------------------------------------------- Citations and the retrieval-ready chunk
-- view. hansard_url builds the public page for a debate (optionally anchored to a contribution).
CREATE OR REPLACE FUNCTION hansard_url(p_house text, p_date date, p_debate_id text, p_heading text, p_speech_id text DEFAULT NULL)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT 'https://hansard.parliament.uk/' || p_house || '/' || p_date::text || '/debates/' || p_debate_id || '/'
           || coalesce(nullif(trim(both '-' from regexp_replace(coalesce(p_heading, ''), '[^A-Za-z0-9]+', '-', 'g')), ''), 'debate')
           || CASE WHEN p_speech_id IS NOT NULL THEN '#contribution-' || p_speech_id ELSE '' END
$$;

CREATE OR REPLACE VIEW chunk_context AS
SELECT c.id AS chunk_id, c.speech_id, c.debate_id, c.seq, c.sent_idx, c.body, c.parent_body,
       c.is_boilerplate, (c.embedding IS NOT NULL) AS is_embedded, c.embedding_model,
       c.column_ref, c.column_number, c.column_end_ref, c.column_basis,
       d.date, d.house, d.heading, d.section, d.debate_type, d.section_order,
       s.speaker_name, s.speaker_person, s.speaker_role, s.speaker_office,
       s.speaker_party AS party_as_attributed,
       s.mnis_id, m.name AS member_name, m.gender,
       member_party_on(s.mnis_id, d.date)                  AS party_on_date,
       party_base(member_party_on(s.mnis_id, d.date))      AS party_on_date_base,
       m.constituency, m.date_of_birth,
       hansard_url(d.house, d.date, d.id, d.heading, s.id) AS hansard_url
FROM chunk c
JOIN speech s ON s.id = c.speech_id
JOIN debate d ON d.id = s.debate_id
LEFT JOIN member_id_map m ON m.mnis_id = s.mnis_id;

COMMIT;
