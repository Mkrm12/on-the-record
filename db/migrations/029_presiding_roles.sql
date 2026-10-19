-- 029: every stored printing of the Chair carries a presiding role.
WITH folded AS (
    SELECT id, regexp_replace(replace(lower(coalesce(speaker_name, '')), E'\u2019', ''''),
                              E'[[:space:].\u00a0\u2002\u2003\u2009]+', '', 'g') AS f
    FROM speech
    WHERE coalesce(speaker_role, '') NOT IN ('speaker', 'deputy_speaker', 'chair', 'lord_speaker', 'lords_deputy')
), chair AS (
    SELECT id, f FROM folded
    WHERE f ~ '^(?:mr|madam)(?:deputy)?speaker(?:-elect)?(?:\(|$)'
       OR f ~ '^the(?:senior)?(?:deputy)?speaker(?:\(|$)'
       OR f ~ '^thelordspeaker(?:\(|$)'
       OR f ~ '^the(?:principal)?(?:deputy)?chairmanofcommittees(?:\(|$)'
       OR f ~ '^(?:the)?(?:first|second|temporary)?(?:deputy)?chair(?:man)?ofwaysandmeans(?:\(|$)'
       OR f ~ '^the(?:temporary|(?:first|second)?deputy)?chair(?:man)?(?:\(|$)'
       OR f ~ '\(inthechair\)'
)
UPDATE speech s SET speaker_role = CASE
        WHEN c.f ~ '^(?:mr|madam)speaker(?:-elect)?(?:\(|$)|^thespeaker(?:\(|$)' THEN 'speaker'
        WHEN c.f ~ '^thelordspeaker(?:\(|$)' THEN 'lord_speaker'
        WHEN c.f ~ '^the(?:senior|second|third|principal)?deputy(?:speaker|chairmanofcommittees)(?:\(|$)'
          OR c.f ~ '^thechairmanofcommittees(?:\(|$)' THEN 'lords_deputy'
        WHEN c.f ~ '^(?:mr|madam)deputyspeaker(?:\(|$)|waysandmeans(?:\(|$)'
          OR c.f ~ '^the(?:first|second|third)deputychair(?:man)?(?:\(|$)' THEN 'deputy_speaker'
        ELSE 'chair' END
FROM chair c
WHERE s.id = c.id;
