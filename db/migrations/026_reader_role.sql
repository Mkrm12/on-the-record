-- 026: a role that can read and nothing else, for the MCP server.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'otr_reader') THEN
        CREATE ROLE otr_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public, served TO otr_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO otr_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA served TO otr_reader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA served TO otr_reader;
-- tables and views a later migration adds are readable too
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO otr_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA served GRANT SELECT ON TABLES TO otr_reader;
