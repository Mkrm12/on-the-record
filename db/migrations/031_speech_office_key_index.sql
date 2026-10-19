-- 031: an index on the office key `served.member_office` computes from each speech.
CREATE INDEX IF NOT EXISTS speech_office_key_idx
    ON public.speech ((regexp_replace(lower(btrim(speaker_office)), '^the\s+', '')))
    WHERE mnis_id IS NOT NULL AND speaker_office IS NOT NULL AND btrim(speaker_office) <> '';
