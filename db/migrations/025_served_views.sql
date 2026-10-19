-- 025: the sealed window in the database itself.

BEGIN;

CREATE SCHEMA IF NOT EXISTS served;

CREATE OR REPLACE FUNCTION served.sealed_from() RETURNS date LANGUAGE sql STABLE PARALLEL SAFE AS
$$ SELECT current_setting('otr.sealed_from')::date $$;
CREATE OR REPLACE FUNCTION served.sealed_to() RETURNS date LANGUAGE sql STABLE PARALLEL SAFE AS
$$ SELECT current_setting('otr.sealed_to')::date $$;

CREATE OR REPLACE VIEW served.debate AS
    SELECT * FROM public.debate
     WHERE date < served.sealed_from() OR date > served.sealed_to();

CREATE OR REPLACE VIEW served.speech AS
    SELECT s.* FROM public.speech s JOIN public.debate d ON d.id = s.debate_id
     WHERE d.date < served.sealed_from() OR d.date > served.sealed_to();

CREATE OR REPLACE VIEW served.chunk AS
    SELECT * FROM public.chunk
     WHERE debate_date < served.sealed_from() OR debate_date > served.sealed_to();

CREATE OR REPLACE VIEW served.chunk_context AS
    SELECT * FROM public.chunk_context
     WHERE date < served.sealed_from() OR date > served.sealed_to();

CREATE OR REPLACE VIEW served.division AS
    SELECT * FROM public.division
     WHERE division_date < served.sealed_from() OR division_date > served.sealed_to();

CREATE OR REPLACE VIEW served.division_vote AS
    SELECT v.* FROM public.division_vote v JOIN public.division d ON d.id = v.division_id
     WHERE d.division_date < served.sealed_from() OR d.division_date > served.sealed_to();

CREATE OR REPLACE VIEW served.bill_stage AS
    SELECT * FROM public.bill_stage
     WHERE NOT coalesce(date BETWEEN served.sealed_from() AND served.sealed_to(), false)
       AND NOT coalesce(first_sitting <= served.sealed_to() AND last_sitting >= served.sealed_from(), false)
       AND NOT coalesce(first_sitting BETWEEN served.sealed_from() AND served.sealed_to(), false)
       AND NOT coalesce(last_sitting BETWEEN served.sealed_from() AND served.sealed_to(), false);

CREATE OR REPLACE VIEW served.bill_stage_sitting AS
    SELECT * FROM public.bill_stage_sitting
     WHERE date < served.sealed_from() OR date > served.sealed_to();

CREATE OR REPLACE VIEW served.coverage AS
    SELECT * FROM public.coverage
     WHERE declared_end < served.sealed_from() OR declared_start > served.sealed_to();

CREATE OR REPLACE VIEW served.corpus_window AS
    SELECT * FROM public.corpus_window
     WHERE end_date < served.sealed_from() OR start_date > served.sealed_to();

ALTER TABLE answer_trace ADD COLUMN IF NOT EXISTS sealed_withheld text[];
COMMENT ON COLUMN answer_trace.sealed_withheld IS
    'What the exit guard withheld for this request (H-45). Empty on a run where the upstream layers held.';

COMMIT;
