-- Rich member metadata and parent/child chunks.

BEGIN;

-- --------------------------------------------------------------------------- member_id_map: mnis_id becomes the key.
-- mnis_id is the only identifier Hansard gives us (speech.mnis_id and division_vote.mnis_id join on it). person_id stays
-- as a nullable, unique slot for a future cross-source ID (e.g.
ALTER TABLE member_id_map DROP CONSTRAINT IF EXISTS member_id_map_pkey;
ALTER TABLE member_id_map DROP CONSTRAINT IF EXISTS member_id_map_mnis_id_key;
ALTER TABLE member_id_map DROP CONSTRAINT IF EXISTS member_id_map_person_id_key;
ALTER TABLE member_id_map ALTER COLUMN person_id DROP NOT NULL;
ALTER TABLE member_id_map ALTER COLUMN mnis_id SET NOT NULL;
ALTER TABLE member_id_map ADD PRIMARY KEY (mnis_id);
ALTER TABLE member_id_map ADD CONSTRAINT member_id_map_person_id_key UNIQUE (person_id);

-- Typed columns for the fields people filter on.
ALTER TABLE member_id_map
    ADD COLUMN IF NOT EXISTS name_list_as             text,
    ADD COLUMN IF NOT EXISTS name_full_title          text,
    ADD COLUMN IF NOT EXISTS name_address_as          text,
    ADD COLUMN IF NOT EXISTS gender                   text,
    ADD COLUMN IF NOT EXISTS party_id                 int,
    ADD COLUMN IF NOT EXISTS party_abbreviation       text,
    ADD COLUMN IF NOT EXISTS party_colour             text,
    ADD COLUMN IF NOT EXISTS house                    text,
    ADD COLUMN IF NOT EXISTS membership_from          text,  -- constituency (Commons) or 'Life peer' etc. (Lords)
    ADD COLUMN IF NOT EXISTS membership_from_id       int,
    ADD COLUMN IF NOT EXISTS member_since             date,  -- earliest house membership; no DOB in the API
    ADD COLUMN IF NOT EXISTS membership_start_date    date,
    ADD COLUMN IF NOT EXISTS membership_end_date      date,
    ADD COLUMN IF NOT EXISTS membership_end_reason    text,
    ADD COLUMN IF NOT EXISTS is_active                boolean,
    ADD COLUMN IF NOT EXISTS status                   text,
    ADD COLUMN IF NOT EXISTS thumbnail_url            text,
    ADD COLUMN IF NOT EXISTS synopsis                 text,
    ADD COLUMN IF NOT EXISTS last_election_result     text,
    ADD COLUMN IF NOT EXISTS last_election_date       date,
    ADD COLUMN IF NOT EXISTS last_election_majority   int,
    ADD COLUMN IF NOT EXISTS last_election_turnout    int,
    ADD COLUMN IF NOT EXISTS last_election_electorate int,
    ADD COLUMN IF NOT EXISTS raw                      jsonb       NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS missing_endpoints        text[]      NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS fetched_at               timestamptz NOT NULL DEFAULT now();

-- --------------------------------------------------------------------------- chunk: child sentence + its parent window.
-- sent_idx 0-based sentence position within its speech seq 0-based position of the chunk within its debate (speech
-- order, then sentence order) -- what (debate_id, seq) indexes parent_body the N-sentence window around `body`, never
-- crossing speeches The table is empty at this point, so tightening NULLability is safe.
-- ---------------------------------------------------------------------------
ALTER TABLE chunk
    ADD COLUMN IF NOT EXISTS sent_idx    int,
    ADD COLUMN IF NOT EXISTS parent_body text;

ALTER TABLE chunk ALTER COLUMN speech_id   SET NOT NULL;
ALTER TABLE chunk ALTER COLUMN debate_id   SET NOT NULL;
ALTER TABLE chunk ALTER COLUMN seq         SET NOT NULL;
ALTER TABLE chunk ALTER COLUMN body        SET NOT NULL;
ALTER TABLE chunk ALTER COLUMN sent_idx    SET NOT NULL;
ALTER TABLE chunk ALTER COLUMN parent_body SET NOT NULL;

-- Doubles as the upsert target and the index for "neighbours of this sentence".
ALTER TABLE chunk DROP CONSTRAINT IF EXISTS chunk_speech_sent_uq;
ALTER TABLE chunk ADD CONSTRAINT chunk_speech_sent_uq UNIQUE (speech_id, sent_idx);

COMMIT;
