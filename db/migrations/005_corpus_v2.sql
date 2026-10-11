-- Corpus v2.

BEGIN;

-- --------------------------------------------------------------------------- 1.

CREATE TABLE IF NOT EXISTS corpus_window (
    window_id       text PRIMARY KEY,
    corpus_version  text NOT NULL,
    start_date      date NOT NULL,
    end_date        date NOT NULL,
    role            text NOT NULL CHECK (role IN ('dev', 'dev_test', 'heldout', 'candidate')),
    loaded_at       timestamptz,
    speeches        integer,
    debates         integer,
    chunks          integer,
    note            text,
    CHECK (start_date <= end_date)
);

COMMENT ON TABLE  corpus_window       IS 'One row per corpus window. role = heldout means no retrieval or evaluation result may be looked at before the final generalisation report (W3, 2020-03-01..2020-05-31).';
COMMENT ON COLUMN corpus_window.role  IS 'dev = tuning only; dev_test = split at question level; heldout = scored once, at the end; candidate = defined but not loaded.';

-- --------------------------------------------------------------------------- 2.

ALTER TABLE debate ADD COLUMN IF NOT EXISTS window_id      text REFERENCES corpus_window(window_id);
ALTER TABLE debate ADD COLUMN IF NOT EXISTS corpus_version text;
ALTER TABLE speech ADD COLUMN IF NOT EXISTS window_id      text;

CREATE INDEX IF NOT EXISTS debate_window_idx ON debate (window_id, date);
CREATE INDEX IF NOT EXISTS speech_window_idx ON speech (window_id);

-- --------------------------------------------------------------------------- 3.

ALTER TABLE chunk ADD COLUMN IF NOT EXISTS window_id   text;
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS debate_date date;
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS house       text;
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS section     text;
ALTER TABLE chunk ADD COLUMN IF NOT EXISTS mnis_id     integer;

COMMENT ON COLUMN chunk.window_id   IS 'Denormalised from debate. A filtered search needs the predicate on the indexed table itself.';
COMMENT ON COLUMN chunk.mnis_id     IS 'Denormalised from speech, so the speaker leg is one table and not a join at 1.2M rows.';

CREATE INDEX IF NOT EXISTS chunk_window_idx  ON chunk (window_id)   WHERE NOT is_boilerplate;
CREATE INDEX IF NOT EXISTS chunk_house_idx   ON chunk (house)       WHERE NOT is_boilerplate;
CREATE INDEX IF NOT EXISTS chunk_section_idx ON chunk (section)     WHERE NOT is_boilerplate;
CREATE INDEX IF NOT EXISTS chunk_date_idx    ON chunk (debate_date) WHERE NOT is_boilerplate;
CREATE INDEX IF NOT EXISTS chunk_mnis_idx    ON chunk (mnis_id)     WHERE NOT is_boilerplate AND mnis_id IS NOT NULL;

-- --------------------------------------------------------------------------- 4.

ALTER TABLE chunk ADD COLUMN IF NOT EXISTS embedding_recipe text;

COMMENT ON COLUMN chunk.embedding_recipe IS 'model|text recipe|version, e.g. bge-small-en-v1.5|sentence|v1. Two vectors are comparable only if this matches.';

-- The worker's queue index must not hand back a row that is embedded under a DIFFERENT recipe as though it were
-- unembedded, and must find one that is.
CREATE INDEX IF NOT EXISTS chunk_recipe_idx ON chunk (embedding_recipe) WHERE NOT is_boilerplate;

-- --------------------------------------------------------------------------- 5.

CREATE TABLE IF NOT EXISTS hansard_correction (
    id            text PRIMARY KEY,            -- ContributionExtId of the correction itself
    window_id     text,
    debate_id     text,                        -- DebateSectionExtId as served; not a foreign key:
                                               -- a correction can point at a debate outside the window
    corrected_speech_id text,                  -- filled by Phase 3; NULL means "not linked yet"
    link_basis    text,                        -- how the link above was made, once it is
    date          date,
    house         text,
    section       text,
    speaker_name  text,
    mnis_id       integer,
    body          text,
    raw           jsonb NOT NULL DEFAULT '{}'::jsonb,
    fetched_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS hansard_correction_debate_idx ON hansard_correction (debate_id);
CREATE INDEX IF NOT EXISTS hansard_correction_window_idx ON hansard_correction (window_id, date);
CREATE INDEX IF NOT EXISTS hansard_correction_speech_idx ON hansard_correction (corrected_speech_id)
    WHERE corrected_speech_id IS NOT NULL;

COMMENT ON TABLE hansard_correction IS 'Written and Ministerial corrections as served by /search/contributions/Corrections.json. corrected_speech_id is Phase 3 work (verification layer); NULL until then.';

-- --------------------------------------------------------------------------- 6.

CREATE TABLE IF NOT EXISTS lexeme_df (
    lexeme     text PRIMARY KEY,
    df         bigint NOT NULL CHECK (df >= 0),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS corpus_stat (
    key        text PRIMARY KEY,
    value      double precision NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE lexeme_df   IS 'Document frequency per lexeme over non-boilerplate chunks, rebuilt by the loader. The keyword leg''s IDF comes from here, not from ts_stat.';
COMMENT ON TABLE corpus_stat IS 'Scalars the keyword leg needs: n_docs, avg_len, built_at. Kept beside lexeme_df so a half-written refresh cannot mix two corpora.';

-- --------------------------------------------------------------------------- 7.

CREATE TABLE IF NOT EXISTS member_office (
    id          bigserial PRIMARY KEY,
    mnis_id     integer NOT NULL REFERENCES member_id_map(mnis_id),
    office      text NOT NULL,          -- as Hansard attributes it, e.g. 'The Prime Minister'
    office_key  text NOT NULL,          -- normalised for lookup: lower case, no leading 'the'
    start_date  date,
    end_date    date,                   -- NULL = still held, as far as the source says
    source      text NOT NULL,          -- 'mnis_biography' | 'hansard_attribution'
    evidence    integer NOT NULL DEFAULT 0,   -- how many speeches support a hansard_attribution row
    UNIQUE (mnis_id, office_key, start_date, source)
);

CREATE INDEX IF NOT EXISTS member_office_key_idx ON member_office (office_key, start_date, end_date);

COMMENT ON TABLE member_office IS 'Who held a named office when. Rows sourced mnis_biography come from the Members API; hansard_attribution rows are inferred from AttributedTo in the speeches themselves and carry the number of speeches supporting them.';

-- --------------------------------------------------------------------------- 8.

CREATE TABLE IF NOT EXISTS load_report (
    id          bigserial PRIMARY KEY,
    window_id   text NOT NULL,
    step        text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    payload     jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS load_report_window_idx ON load_report (window_id, step, created_at DESC);

COMMIT;
