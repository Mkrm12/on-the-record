CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE motion_type AS ENUM ('division', 'substantive', 'neutral');

CREATE TABLE debate (
    id text PRIMARY KEY,
    date date,
    house text,
    heading text,
    section text,
    debate_type text
);

CREATE TABLE speech (
    id text PRIMARY KEY,
    debate_id text REFERENCES debate(id),
    seq int,
    speaker_name text,
    mnis_id int,
    text text,
    timecode text,
    is_boilerplate boolean DEFAULT false
);

-- One row per sentence (the "child", embedded). parent_body is the N-sentence window around it (the "parent", handed to
-- the LLM); it never crosses speeches. sent_idx 0-based sentence position within its speech seq 0-based chunk position
-- within its debate (speech order, then sentence)
CREATE TABLE chunk (
    id bigserial PRIMARY KEY,
    speech_id text NOT NULL REFERENCES speech(id),
    debate_id text NOT NULL,
    seq int NOT NULL,
    sent_idx int NOT NULL,
    body text NOT NULL,
    parent_body text NOT NULL,
    embedding vector(384),
    ts tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED,
    CONSTRAINT chunk_speech_sent_uq UNIQUE (speech_id, sent_idx)
);

CREATE INDEX ON chunk USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunk USING gin (ts);
CREATE INDEX ON chunk (debate_id, seq);

-- Keyed on mnis_id, the only identifier Hansard gives us. person_id is a nullable slot for a future cross-source ID
-- (e.g.
CREATE TABLE member_id_map (
    mnis_id int PRIMARY KEY,
    person_id text UNIQUE,
    name text,
    name_list_as text,
    name_full_title text,
    name_address_as text,
    gender text,
    party text,               -- latest party; full history is in raw->'biography'->'partyAffiliations'
    party_id int,
    party_abbreviation text,
    party_colour text,
    house text,
    membership_from text,     -- constituency (Commons) or 'Life peer' etc. (Lords)
    membership_from_id int,
    constituency text,        -- Commons only
    member_since date,        -- earliest house membership; the API has no date of birth
    membership_start_date date,
    membership_end_date date,
    membership_end_reason text,
    is_active boolean,
    status text,
    thumbnail_url text,
    synopsis text,
    last_election_result text,
    last_election_date date,
    last_election_majority int,
    last_election_turnout int,
    last_election_electorate int,
    raw jsonb NOT NULL DEFAULT '{}',
    missing_endpoints text[] NOT NULL DEFAULT '{}',
    fetched_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE bill (
    id int PRIMARY KEY,
    title text
);

CREATE TABLE bill_stage (
    id int PRIMARY KEY,
    bill_id int REFERENCES bill(id),
    stage_name text,
    house text,
    date date,
    debate_id text
);

CREATE TABLE division (
    id int PRIMARY KEY,
    debate_id text,
    bill_stage_id int NULL,
    motion_text text,
    motion_type motion_type,
    ayes int,
    noes int,
    result text
);

CREATE TABLE division_vote (
    division_id int REFERENCES division(id),
    mnis_id int,
    lobby text,
    PRIMARY KEY (division_id, mnis_id)
);

CREATE TABLE procedure_event (
    id bigserial PRIMARY KEY,
    debate_id text,
    seq int,
    event_type text,
    raw_phrase text
);

CREATE TABLE answer_cache (
    query_hash text PRIMARY KEY,
    answer text,
    citations jsonb,
    created_at timestamptz DEFAULT now()
);
