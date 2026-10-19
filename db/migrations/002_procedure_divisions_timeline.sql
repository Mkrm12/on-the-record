-- Procedure events, boilerplate flags, divisions with per-vote party, and temporal views over the member metadata
-- already stored in member_id_map.raw.

BEGIN;

-- --------------------------------------------------------------------------- speech: parsed attribution + why a speech
-- was flagged boilerplate.
ALTER TABLE speech
    ADD COLUMN IF NOT EXISTS boilerplate_reason   text,
    ADD COLUMN IF NOT EXISTS speaker_office       text,
    ADD COLUMN IF NOT EXISTS speaker_person       text,
    ADD COLUMN IF NOT EXISTS speaker_constituency text,
    ADD COLUMN IF NOT EXISTS speaker_party        text,
    ADD COLUMN IF NOT EXISTS speaker_role         text,
    ADD COLUMN IF NOT EXISTS speaker_flags        text[] NOT NULL DEFAULT '{}';

-- Sentence-level flag: pure-procedure sentences inside otherwise substantive speeches.
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS is_boilerplate boolean NOT NULL DEFAULT false;

-- --------------------------------------------------------------------------- procedure_event: which speech, where in
-- it, what was said, and the motion text. ---------------------------------------------------------------------------
ALTER TABLE procedure_event
    ADD COLUMN IF NOT EXISTS speech_id   text REFERENCES speech(id),
    ADD COLUMN IF NOT EXISTS char_start  int,          -- offset into the cleaned speech text
    ADD COLUMN IF NOT EXISTS description text,         -- captured motion / resolution / topic text
    ADD COLUMN IF NOT EXISTS motion_type motion_type,  -- neutral | substantive, for motions
    ADD COLUMN IF NOT EXISTS rule        text;         -- which rule fired (audit)

ALTER TABLE procedure_event DROP CONSTRAINT IF EXISTS procedure_event_speech_type_pos_uq;
ALTER TABLE procedure_event ADD CONSTRAINT procedure_event_speech_type_pos_uq UNIQUE (speech_id, event_type, char_start);
CREATE INDEX IF NOT EXISTS procedure_event_debate_seq_idx ON procedure_event (debate_id, seq);
CREATE INDEX IF NOT EXISTS procedure_event_type_idx       ON procedure_event (event_type);

-- --------------------------------------------------------------------------- division: Commons AND Lords.
ALTER TABLE division
    ADD COLUMN IF NOT EXISTS house            text        NOT NULL DEFAULT 'Commons',
    ADD COLUMN IF NOT EXISTS source_id        int,
    ADD COLUMN IF NOT EXISTS number           int,           -- "Division No. 75"
    ADD COLUMN IF NOT EXISTS division_date    date,
    ADD COLUMN IF NOT EXISTS held_at          timestamp,     -- local time; midnight when the API gives none
    ADD COLUMN IF NOT EXISTS title            text,
    ADD COLUMN IF NOT EXISTS is_deferred      boolean     NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS ayes_listed      int,           -- voters actually stored (incl. tellers)
    ADD COLUMN IF NOT EXISTS noes_listed      int,
    ADD COLUMN IF NOT EXISTS no_vote_count    int,
    ADD COLUMN IF NOT EXISTS tally_reconciled boolean,       -- official counts == stored voters
    ADD COLUMN IF NOT EXISTS government_won   boolean,       -- Lords API only; Commons derived later
    ADD COLUMN IF NOT EXISTS debate_link      text,          -- how debate_id was found (or 'none')
    ADD COLUMN IF NOT EXISTS anomalies        text[]      NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS raw              jsonb       NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS fetched_at       timestamptz NOT NULL DEFAULT now();

ALTER TABLE division ALTER COLUMN source_id SET NOT NULL;
ALTER TABLE division DROP CONSTRAINT IF EXISTS division_house_source_uq;
ALTER TABLE division ADD CONSTRAINT division_house_source_uq UNIQUE (house, source_id);
ALTER TABLE division DROP CONSTRAINT IF EXISTS division_id_namespace_ck;
ALTER TABLE division ADD CONSTRAINT division_id_namespace_ck CHECK (
    (house = 'Commons' AND id = source_id) OR (house = 'Lords' AND id = source_id + 1000000)
);
CREATE INDEX IF NOT EXISTS division_date_idx   ON division (division_date);
CREATE INDEX IF NOT EXISTS division_debate_idx ON division (debate_id);

-- --------------------------------------------------------------------------- division_vote: party, seat and name AS AT
-- THE VOTE (they change over time), teller and proxy flags, and no-vote-recorded rows. lobby is normalised across
-- Houses: Commons Aye/No and Lords Content/Not Content become aye/no.
-- ---------------------------------------------------------------------------
ALTER TABLE division_vote
    ADD COLUMN IF NOT EXISTS member_name        text,
    ADD COLUMN IF NOT EXISTS party              text,
    ADD COLUMN IF NOT EXISTS party_abbreviation text,
    ADD COLUMN IF NOT EXISTS sub_party          text,
    ADD COLUMN IF NOT EXISTS member_from        text,   -- constituency (Commons) / peerage type (Lords)
    ADD COLUMN IF NOT EXISTS is_teller          boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS proxy_name         text,
    ADD COLUMN IF NOT EXISTS flags              text[]  NOT NULL DEFAULT '{}';
ALTER TABLE division_vote ALTER COLUMN lobby SET NOT NULL;
ALTER TABLE division_vote DROP CONSTRAINT IF EXISTS division_vote_lobby_ck;
ALTER TABLE division_vote ADD CONSTRAINT division_vote_lobby_ck CHECK (lobby IN ('aye', 'no', 'no_vote', 'conflict'));
CREATE INDEX IF NOT EXISTS division_vote_mnis_idx ON division_vote (mnis_id);

-- --------------------------------------------------------------------------- Temporal views over member_id_map.raw
-- (Members API biography).
CREATE OR REPLACE VIEW member_party_history AS
SELECT m.mnis_id,
       (a->>'id')::int         AS party_id,
       a->>'name'              AS party,
       (a->>'house')::int      AS house_id,
       (a->>'startDate')::date AS start_date,
       (a->>'endDate')::date   AS end_date
FROM member_id_map m
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(m.raw->'biography'->'partyAffiliations') = 'array'
         THEN m.raw->'biography'->'partyAffiliations' ELSE '[]'::jsonb END) a;

CREATE OR REPLACE VIEW member_post AS
SELECT m.mnis_id,
       k.kind,                                   -- government | opposition | other
       (p->>'id')::int         AS post_id,
       p->>'name'              AS post,
       p->>'additionalInfo'    AS department,
       (p->>'house')::int      AS house_id,
       (p->>'startDate')::date AS start_date,
       (p->>'endDate')::date   AS end_date
FROM member_id_map m
CROSS JOIN (VALUES ('government', 'governmentPosts'), ('opposition', 'oppositionPosts'), ('other', 'otherPosts')) AS k(kind, key)
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(m.raw->'biography'->k.key) = 'array'
         THEN m.raw->'biography'->k.key ELSE '[]'::jsonb END) p;

CREATE OR REPLACE VIEW member_committee AS
SELECT m.mnis_id,
       (c->>'id')::int         AS committee_id,
       c->>'name'              AS committee,
       c->>'additionalInfo'    AS role,
       (c->>'house')::int      AS house_id,
       (c->>'startDate')::date AS start_date,
       (c->>'endDate')::date   AS end_date
FROM member_id_map m
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(m.raw->'biography'->'committeeMemberships') = 'array'
         THEN m.raw->'biography'->'committeeMemberships' ELSE '[]'::jsonb END) c;

CREATE OR REPLACE VIEW member_seat AS
SELECT m.mnis_id,
       (r->>'id')::int         AS constituency_id,
       r->>'name'              AS constituency,
       (r->>'startDate')::date AS start_date,
       (r->>'endDate')::date   AS end_date,
       r->>'additionalInfo'    AS note
FROM member_id_map m
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(m.raw->'biography'->'representations') = 'array'
         THEN m.raw->'biography'->'representations' ELSE '[]'::jsonb END) r;

CREATE OR REPLACE VIEW member_house_term AS
SELECT m.mnis_id,
       (h->>'house')::int      AS house_id,
       h->>'name'              AS house,
       (h->>'startDate')::date AS start_date,
       (h->>'endDate')::date   AS end_date
FROM member_id_map m
CROSS JOIN LATERAL jsonb_array_elements(
    CASE WHEN jsonb_typeof(m.raw->'biography'->'houseMemberships') = 'array'
         THEN m.raw->'biography'->'houseMemberships' ELSE '[]'::jsonb END) h;

CREATE OR REPLACE FUNCTION member_party_on(p_mnis_id int, p_date date) RETURNS text
LANGUAGE sql STABLE AS $$
    SELECT party FROM member_party_history
    WHERE mnis_id = p_mnis_id AND start_date <= p_date AND (end_date IS NULL OR end_date >= p_date)
    ORDER BY start_date DESC LIMIT 1
$$;

-- Everything known about a speech in one row: the party is given three ways because they differ.
CREATE OR REPLACE VIEW speech_context AS
SELECT s.id AS speech_id, s.debate_id, s.seq, d.date, d.house, d.heading, d.section, d.debate_type,
       s.speaker_name, s.speaker_role, s.speaker_office, s.speaker_person, s.speaker_flags,
       s.speaker_party    AS party_as_attributed,          -- printed in Hansard at the time
       s.mnis_id, m.name  AS member_name, m.gender,
       member_party_on(s.mnis_id, d.date) AS party_on_date, -- from the member's affiliation history
       m.party            AS party_now,
       s.is_boilerplate, s.boilerplate_reason, s.text
FROM speech s
JOIN debate d ON d.id = s.debate_id
LEFT JOIN member_id_map m ON m.mnis_id = s.mnis_id;

COMMIT;
