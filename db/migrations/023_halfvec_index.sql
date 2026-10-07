-- 023: the HNSW index in half precision.

DO $$
DECLARE
    current_def text;
BEGIN
    SELECT pg_get_indexdef(i.indexrelid) INTO current_def
    FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
    WHERE i.indrelid = 'chunk'::regclass AND c.relname = 'chunk_embedding_idx';

    IF current_def LIKE '%::halfvec(384)) halfvec_cosine_ops)%' THEN
        RETURN;
    END IF;

    IF EXISTS (SELECT 1 FROM chunk WHERE embedding IS NOT NULL) THEN
        RAISE EXCEPTION 'chunk_embedding_idx is not the halfvec index and chunk already holds vectors. Switch it '
            'without stopping searches: uv run python scripts/switch_vector_index.py run --form halfvec '
            '(about 30 minutes at 1.4M vectors), then migrate again.';
    END IF;

    DROP INDEX IF EXISTS chunk_embedding_idx;
    CREATE INDEX chunk_embedding_idx ON chunk
        USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
END $$;
