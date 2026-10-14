-- What the SOURCES do not hold, per window: facts about Parliament's data services, stored as data.

BEGIN;

CREATE TABLE IF NOT EXISTS source_gap (
    window_id   text        NOT NULL REFERENCES corpus_window(window_id),
    source      text        NOT NULL,
    house       text        NOT NULL CHECK (house IN ('Commons', 'Lords')),
    kind        text        NOT NULL,
    count       integer     NOT NULL CHECK (count >= 0),
    detail      text        NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (window_id, source, house, kind)
);

COMMENT ON TABLE source_gap IS
    'Coverage limits of the SOURCES (not of the load), per window: e.g. kind no_voter_lists = Hansard lists divisions '
    'the Votes API has no record of. Written by ingest.load from the fetch cache''s division reconciliation.';

COMMIT;
