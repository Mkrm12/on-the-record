-- The coverage table: what the corpus actually holds, per window x house x section — "this answer draws on the Commons
-- Chamber between 1 April and 31 July 2026" — which is the difference between an answer and an answer a reader can
-- place; * the `outside_window` abstention, which is how the system says "I cannot know that" instead of guessing; * the
-- timeline gaps, so a silence in the record can be reported as a silence rather than as an absence of evidence.

BEGIN;

CREATE TABLE IF NOT EXISTS coverage (
    window_id       text NOT NULL,
    house           text NOT NULL,
    section         text NOT NULL,

    corpus_version  text,
    -- Denormalised from corpus_window so that "never report a heldout window's numbers" is one predicate here rather
    -- than a join every caller has to remember.
    role            text,
    declared_start  date,
    declared_end    date,

    -- What the record actually holds, which is not the same as what was declared: a window declared to 31 July whose
    -- last sitting is 23 July is correct and worth being able to see.
    first_sitting   date,
    last_sitting    date,
    sitting_days    integer NOT NULL DEFAULT 0,

    debates         integer NOT NULL DEFAULT 0,
    speeches        integer NOT NULL DEFAULT 0,
    -- `contributions` excludes boilerplate; `speeches` does not.
    contributions   integer NOT NULL DEFAULT 0,
    boilerplate     integer NOT NULL DEFAULT 0,
    speakers        integer NOT NULL DEFAULT 0,
    -- Speeches with no MNIS id.
    unattributed    integer NOT NULL DEFAULT 0,

    chunks          integer NOT NULL DEFAULT 0,
    embedded_chunks integer NOT NULL DEFAULT 0,

    divisions       integer NOT NULL DEFAULT 0,
    division_votes  integer NOT NULL DEFAULT 0,
    -- Divisions whose published name lists do not reconcile with the tellers' count.
    unreconciled_divisions integer NOT NULL DEFAULT 0,

    -- The longest run of days inside the window with no sitting in this house and section, and where it falls.
    max_gap_days    integer,
    max_gap_start   date,
    max_gap_end     date,

    refreshed_at    timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (window_id, house, section)
);

CREATE INDEX IF NOT EXISTS coverage_house_idx   ON coverage (house, section);
CREATE INDEX IF NOT EXISTS coverage_dates_idx   ON coverage (first_sitting, last_sitting);
CREATE INDEX IF NOT EXISTS coverage_role_idx    ON coverage (role);
CREATE INDEX IF NOT EXISTS coverage_window_idx  ON coverage (window_id);

COMMENT ON TABLE coverage IS
    'One row per window x house x section: what the corpus holds, measured, not declared. Powers the '
    'coverage_note, the outside_window abstention and the timeline gaps (endgame §5.3). Membership is per '
    'window, never a date span: the six windows are not contiguous, so a date can sit between them and belong '
    'to none.';
COMMENT ON COLUMN coverage.role IS
    'dev | dev_test | heldout, from corpus_window. Denormalised so that excluding a held-out window is one '
    'predicate rather than a join every caller has to remember to write.';
COMMENT ON COLUMN coverage.max_gap_days IS
    'The longest run of days in this window with no sitting in this house and section. W3''s is the covid '
    'recess: the last March 2020 sitting is the 25th and the next is 21 April. That is a fact about 2020, not a '
    'fetch failure, and reporting it as a gap is how the system says so.';
COMMENT ON COLUMN coverage.unattributed IS
    'Speeches with no MNIS id. The reason a roster can be incomplete, so it is recorded where the coverage note '
    'can reach it rather than left to be rediscovered.';
COMMENT ON COLUMN coverage.last_sitting IS
    'The last sitting actually in the record, which may be earlier than declared_end and legitimately so — a '
    'window running to 31 July whose House rose on 23 July is correct.';

COMMIT;
