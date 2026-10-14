-- 028: bills are masked, and the reader role reads only what is served.

BEGIN;

CREATE OR REPLACE VIEW served.bill AS
    SELECT b.id, b.title, b.long_title, b.former_short_title, b.bill_type_id, b.originating_house,
           CASE WHEN w.touched THEN NULL ELSE b.current_house END AS current_house,
           b.introduced_session_id,
           CASE WHEN w.touched THEN NULL ELSE b.is_act END AS is_act,
           CASE WHEN w.touched THEN NULL ELSE b.is_defeated END AS is_defeated,
           CASE WHEN w.touched THEN NULL ELSE b.is_withdrawn END AS is_withdrawn,
           -- an update inside the window is clipped to the day before it: no in-window date leaks, and every served
           -- sitting after the window stays after this as-of, exactly as with the true date
           CASE WHEN b.last_update::date BETWEEN served.sealed_from() AND served.sealed_to()
                THEN (served.sealed_from() - 1)::timestamptz ELSE b.last_update END AS last_update,
           CASE WHEN w.touched THEN '{}'::jsonb ELSE b.raw END AS raw,
           b.fetched_at
      FROM public.bill b
      CROSS JOIN LATERAL (
          SELECT bool_or(d.day BETWEEN served.sealed_from() AND served.sealed_to()) AS touched,
                 bool_or(d.day < served.sealed_from() OR d.day > served.sealed_to()) AS seen
            FROM (SELECT t.date AS day FROM public.bill_stage_sitting t WHERE t.bill_id = b.id
                  UNION ALL
                  SELECT s.date FROM public.bill_stage s WHERE s.bill_id = b.id AND s.date IS NOT NULL) d
      ) w
     WHERE coalesce(w.seen, true);  -- seen outside the window, or never dated (bool_or over no rows is NULL)

GRANT SELECT ON served.bill TO otr_reader;

DO $$
DECLARE
    t text;
BEGIN
    FOR t IN SELECT v.table_name FROM information_schema.views v
               JOIN information_schema.tables p ON p.table_schema = 'public' AND p.table_name = v.table_name
                                               AND p.table_type = 'BASE TABLE'
              WHERE v.table_schema = 'served'
    LOOP
        EXECUTE format('REVOKE SELECT ON public.%I FROM otr_reader', t);
    END LOOP;
END
$$;

-- what the exit guard reads (`answer.sealing.record_dates`), and nothing else of those tables
GRANT SELECT (id, debate_id) ON public.speech TO otr_reader;
GRANT SELECT (id, date) ON public.debate TO otr_reader;
GRANT SELECT (id, debate_date) ON public.chunk TO otr_reader;

COMMIT;
