-- Make the trace able to answer "was this abstention cowardly?" in SQL — the metric the maintainer asked for.

BEGIN;

ALTER TABLE answer_trace
    ADD COLUMN IF NOT EXISTS card_sentences integer,
    ADD COLUMN IF NOT EXISTS gold_evidence_retrieved boolean,
    ADD COLUMN IF NOT EXISTS gold_evidence_rank integer;

-- The index that the metric actually runs: abstentions of an evaluation run, by whether the evidence was there.
CREATE INDEX IF NOT EXISTS answer_trace_cowardice_idx
    ON answer_trace (run_id, outcome, gold_evidence_retrieved)
    WHERE is_evaluation;

COMMENT ON COLUMN answer_trace.cards IS
    'How many CARDS the generator was shown, measured from the deck. Was derived from answer.retrieved_chunk_ids, '
    'which `_abstain` does not set, so it read 0 for every abstention in a run and was mistaken for evidence that '
    'retrieval had returned nothing.';
COMMENT ON COLUMN answer_trace.card_sentences IS
    'How many citable sentences those cards held. A different question from the card count: a deck of 3 long cards '
    'and a deck of 10 short ones are not the same thing to a generator.';
COMMENT ON COLUMN answer_trace.gold_evidence_retrieved IS
    'Evaluation only. TRUE when a speech the gold item names as relevant was in the cards the generator was shown. '
    'An abstention with TRUE here is a **utility failure, not caution**: the evidence was in front of it. NULL means '
    'unknown — not an evaluation, or the item names no relevant speech — which is deliberately not FALSE, because a '
    'metric that counts unknown as failure invents failures.';
COMMENT ON COLUMN answer_trace.gold_evidence_rank IS
    'Where the best gold speech landed in the deck, 1-based, or NULL if it was absent. Separates "retrieval missed '
    'it" from "retrieval buried it" from "the generator ignored it".';

COMMIT;
