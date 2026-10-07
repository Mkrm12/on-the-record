-- Remote divisions as a typed fact rather than a field inside `raw`.

BEGIN;

ALTER TABLE division ADD COLUMN IF NOT EXISTS is_remote    boolean;
ALTER TABLE division ADD COLUMN IF NOT EXISTS remote_start timestamp;
ALTER TABLE division ADD COLUMN IF NOT EXISTS remote_end   timestamp;

COMMENT ON COLUMN division.is_remote IS
    'Commons: RemoteVotingStart is present. Lords: divisionWasExclusivelyRemote. NULL means the source said nothing either way, which is not the same as false.';

CREATE INDEX IF NOT EXISTS division_remote_idx ON division (is_remote) WHERE is_remote;

COMMIT;
