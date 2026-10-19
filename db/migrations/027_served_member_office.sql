-- 027: `served.member_office`, the offices as the masked record shows them.

CREATE OR REPLACE VIEW served.member_office AS
    SELECT NULL::bigint AS id, s.mnis_id, s.speaker_office AS office,
           regexp_replace(lower(btrim(s.speaker_office)), '^the\s+', '') AS office_key,
           min(d.date) AS start_date, max(d.date) AS end_date, 'hansard_attribution'::text AS source,
           count(*)::integer AS evidence, false AS start_withheld, false AS end_withheld
      FROM served.speech s
      JOIN served.debate d ON d.id = s.debate_id
     WHERE s.mnis_id IS NOT NULL AND s.speaker_office IS NOT NULL AND btrim(s.speaker_office) <> ''
     GROUP BY s.mnis_id, s.speaker_office
    UNION ALL
    SELECT o.id, o.mnis_id, o.office, o.office_key,
           CASE WHEN o.start_date BETWEEN served.sealed_from() AND served.sealed_to()
                THEN served.sealed_to() + 1 ELSE o.start_date END,
           CASE WHEN o.end_date BETWEEN served.sealed_from() AND served.sealed_to()
                THEN served.sealed_from() - 1 ELSE o.end_date END,
           o.source, o.evidence,
           coalesce(o.start_date BETWEEN served.sealed_from() AND served.sealed_to(), false),
           coalesce(o.end_date BETWEEN served.sealed_from() AND served.sealed_to(), false)
      FROM public.member_office o
     WHERE o.source <> 'hansard_attribution'
       -- wholly inside the window; an open-ended post (end NULL) is not, and NULL must not drop it
       AND NOT coalesce(o.start_date >= served.sealed_from() AND o.end_date <= served.sealed_to(), false);

COMMENT ON VIEW served.member_office IS 'member_office as the masked record shows it: Hansard-attribution rows recomputed from served.speech; biography posts clipped at the sealed window, with the withheld side flagged (migration 027).';

GRANT SELECT ON served.member_office TO otr_reader;
