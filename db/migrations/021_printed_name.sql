-- 021: the name the record printed.

-- A printed name with a forename: two words or more, and not a courtesy title before a bare surname ("Mr Jones", "Dr
-- Whitehead", "Sir Edward" are not full).
CREATE OR REPLACE FUNCTION printed_full_name(p_name text) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT btrim(p_name) ~ '^\S+\s+\S'
       AND btrim(p_name) !~* '^(mr|mrs|ms|miss|mx|dr|sir|dame|rev|revd|professor|prof)\.?\s+\S+$'
$$;

COMMENT ON FUNCTION printed_full_name(text) IS
    'Whether a printed name carries a forename (N-01): "Mr Kevan Jones" yes, "Mr Jones" no.';

-- The fullest name the record printed for a member in ONE House: Hansard's own labels first, then the Votes API's name
-- in that House's division lists; the most frequent form, ties broken by the name so the answer never varies.
CREATE OR REPLACE FUNCTION member_printed_name(p_mnis_id integer, p_house text) RETURNS text
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT name FROM (
        SELECT btrim(s.speaker_person) AS name, count(*) AS n, 1 AS source_rank
          FROM speech s JOIN debate d ON d.id = s.debate_id
         WHERE s.mnis_id = p_mnis_id AND d.house = p_house AND printed_full_name(s.speaker_person)
         GROUP BY btrim(s.speaker_person)
        UNION ALL
        SELECT btrim(v.member_name), count(*), 2
          FROM division_vote v JOIN division dv ON dv.id = v.division_id
         WHERE v.mnis_id = p_mnis_id AND dv.house = p_house AND printed_full_name(v.member_name)
         GROUP BY btrim(v.member_name)
    ) named
    ORDER BY source_rank, n DESC, name
    LIMIT 1
$$;

COMMENT ON FUNCTION member_printed_name(integer, text) IS
    'The fullest name printed for a member in one House: Hansard labels, then that House''s division lists (N-01).';

-- How a reader is shown the speaker of one speech. no member (or no name) the printed label, as before still in the
-- House they spoke in the member's name, as before in another House now the name printed then, completed where the label
-- was terse, with the member's present name after it: "Michael Gove (now Lord Gove)"
CREATE OR REPLACE FUNCTION speaker_as_printed(p_speech_id text) RETURNS text
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT CASE
             WHEN m.name IS NULL THEN s.speaker_name
             WHEN m.house IS NULL OR m.house = d.house THEN m.name
             ELSE (SELECT CASE WHEN p.name IS NULL OR p.name = m.name THEN m.name
                               ELSE p.name || ' (now ' || m.name || ')' END
                     FROM (SELECT coalesce(
                                      CASE WHEN printed_full_name(s.speaker_person) THEN btrim(s.speaker_person) END,
                                      member_printed_name(s.mnis_id, d.house),
                                      nullif(btrim(s.speaker_person), ''),
                                      nullif(btrim(s.speaker_name), '')) AS name) p)
           END
      FROM speech s
      JOIN debate d ON d.id = s.debate_id
      LEFT JOIN member_id_map m ON m.mnis_id = s.mnis_id
     WHERE s.id = p_speech_id
$$;

COMMENT ON FUNCTION speaker_as_printed(text) IS
    'The speaker of a speech as a reader is shown them: the name printed at the time, never a later title (N-01).';
