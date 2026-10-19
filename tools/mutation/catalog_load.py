"""Mutants for the Corpus v2 load path and the scale fixes."""

from mutation_engine import Mutant, Project

TESTS = ["tests/test_load.py", "tests/test_retrieval_scale.py"]

# The loader tests read the committed fixtures and build caches on disk; the scale tests need `tests/corpus.py` and the
# migrations, which DEFAULT_PROJECT already copies.
PROJECT = Project(dirs=("src", "tests", "config", "db"))

_RAW = [
    # --- the cache reader: the merge decides which of two answers the corpus keeps ------
    ("src/ingest/corpus_cache.py", "the LAST sort order wins the merge, not the first",
     "merged.setdefault(cid, item)", "merged[cid] = item"),
    ("src/ingest/corpus_cache.py", "a contribution served under two debate ids is not recorded",
     "conflicts = {cid: tuple(ids) for cid, ids in seen_debates.items() if len(ids) > 1}",
     "conflicts = {}"),
    ("src/ingest/corpus_cache.py", "the listed total is taken from the LAST page, not the first",
     "if listed is None and isinstance(payload.get(\"TotalResultCount\"), int):",
     "if isinstance(payload.get(\"TotalResultCount\"), int):"),
    ("src/ingest/corpus_cache.py", "unserved contributions are reported as a negative number",
     "return max((self.listed or 0) - len(self.items), 0)",
     "return (self.listed or 0) - len(self.items)"),
    ("src/ingest/corpus_cache.py", "a cache path may escape the cache root",
     "if not _inside_root(rel):", "if False:"),
    ("src/ingest/corpus_cache.py", "a .. segment is allowed, so the character class alone guards nothing",
     'return bool(_SAFE.match(rel)) and ".." not in rel.split("/") and not rel.startswith("/")',
     "return bool(_SAFE.match(rel))"),
    ("src/ingest/corpus_cache.py", "a missing optional member endpoint is not reported missing",
     "missing.append(key)", "pass"),
    ("src/ingest/corpus_cache.py", "days outside the window are loaded too",
     "if window.start <= day <= window.end:", "if True:"),

    # --- shaping: reproducing `hansard` depends on these exactly ------------------------
    ("src/ingest/load.py", "the stored text is cleaned, so every text hash changes",
     'return value.replace("\\x00", "") if value else value',
     "return value.strip() if value else value"),
    ("src/ingest/load.py", "the short text is preferred over the full one",
     'text=clean(item.get("ContributionTextFull") or item.get("ContributionText")),',
     'text=clean(item.get("ContributionText") or item.get("ContributionTextFull")),'),
    ("src/ingest/load.py", "an item that cannot be placed is dropped silently",
     'unplaceable.append(str(item.get("ContributionExtId") or item.get("ItemId") or "?"))',
     "pass"),
    ("src/ingest/load.py", "the long texts are duplicated into meta as well",
     "meta=json.dumps({k: v for k, v in item.items() if k not in _TEXT_KEYS}, ensure_ascii=False),",
     "meta=json.dumps(dict(item), ensure_ascii=False),"),
    ("src/ingest/load.py", "the window tag is never written on a debate",
     'section_order=debate["section_order"], window_id=window.id,',
     'section_order=debate["section_order"], window_id=None,'),
    ("src/ingest/load.py", "the window tag is never written on a speech",
     "meta=json.dumps({k: v for k, v in item.items() if k not in _TEXT_KEYS}, ensure_ascii=False),\n            window_id=window.id,",
     "meta=json.dumps({k: v for k, v in item.items() if k not in _TEXT_KEYS}, ensure_ascii=False),\n            window_id=None,"),
    ("src/ingest/load.py", "a correction is dated by the day it was filed under, not its own",
     'date.fromisoformat(sitting[:10]) if isinstance(sitting, str) else day,', "day,"),

    # --- idempotence: a resumable orchestrator re-runs finished steps -------------------
    ("src/ingest/load.py", "the upsert rewrites every row on a second load",
     'WHERE ({", ".join(f"{table}.{c}" for c in others)})\n              IS DISTINCT FROM ({", ".join(f"excluded.{c}" for c in others)})',
     ""),
    ("src/ingest/load.py", "a day read twice inserts the same debate twice in one statement",
     "unique_debates = list({d.id: d for d in debates}.values())", "unique_debates = debates"),
    ("src/ingest/load.py", "corrections are not deduplicated before the upsert",
     "if shaped and shaped[\"id\"] not in seen:", "if shaped:"),
    ("src/ingest/load.py", "the chunk filter sync rewrites rows that already agree",
     "AND (c.window_id, c.debate_date, c.house, c.section, c.mnis_id)\n          IS DISTINCT FROM (d.window_id, d.date, d.house, d.section, s.mnis_id)",
     ""),

    # --- division links: a stated link and a guessed one are not the same thing ---
    ("src/ingest/load.py", "Hansard's stated debate link is ignored in favour of the inference",
     "if debate_id and debate_id in known_debates:", "if False:"),
    ("src/ingest/load.py", "a link to a debate outside the window is stored anyway",
     'division["debate_id"], division["debate_link"] = None, "hansard_link_outside_window"',
     'division["debate_id"], division["debate_link"] = debate_id, "hansard_link_outside_window"'),
    ("src/ingest/load.py", "the link provenance is not recorded",
     'load.division_links[division["debate_link"]] += 1', "pass"),

    # --- which database may be written to ----------------------------------------------
    ("src/app/safety.py", "the rollback corpus may be written to",
     "if found in PROTECTED and expected not in PROTECTED:", "if False:"),
    ("src/app/safety.py", "any database is accepted as the target",
     "if found != expected:", "if False:"),
    ("src/app/safety.py", "a DSN with no database name is allowed through",
     "if not name:", "if False:"),
    ("src/app/safety.py", "a test may use a real database",
     "if not found.startswith(TEST_PREFIXES):", "if False:"),

    # The relative threshold ---------------------------------------------------
    ("src/retrieval/speakers.py", "the relative threshold ignores the corpus size",
     "return max(1, round(ratio * n_docs))", "return max(1, round(ratio))"),
    ("src/retrieval/speakers.py", "the threshold may fall to zero, making every word common",
     "return max(1, round(ratio * n_docs))", "return round(ratio * n_docs)"),
    ("src/retrieval/speakers.py", "the ratio is used even when none was given",
     "if ratio is None:\n        return absolute", "if False:\n        return absolute"),

    # Dates narrow, they do not erase -----------------------------------------
    ("src/retrieval/speakers.py", "a date that matches nobody removes every candidate",
     "if serving:\n                    ids = serving", "ids = serving"),
    ("src/retrieval/speakers.py", "a member with no recorded dates is filtered out",
     "if start is not None and on_date < start:", "if on_date < start:"),
    ("src/retrieval/speakers.py", "the end of a membership is ignored",
     "return not (end is not None and on_date > end)", "return True"),

    # --- offices: ambiguity is an answer ------------------------------------------------
    ("src/retrieval/offices.py", "an office with several holders is reported as resolved",
     "ambiguous=len(distinct) > 1,", "ambiguous=False,"),
    ("src/retrieval/offices.py", "an office held by many people at once resolves to one",
     "if key in PLURAL_OFFICES:", "if False:"),
    ("src/retrieval/offices.py", "the date is ignored, so the last holder always wins",
     "if on_date is None:\n            return rows", "return rows"),
    ("src/retrieval/offices.py", "a leading 'the' makes two spellings two offices",
     'key = _LEADING_THE.sub("", key).strip()', "key = key.strip()"),
    ("src/retrieval/offices.py", "an office ending before the date still counts",
     "and (r.end_date is None or on_date <= r.end_date)", ""),

    # --- filters: every leg, or none ----------------------------------------------------
    ("src/retrieval/hybrid.py", "the keyword leg ignores the filter",
     "WHERE NOT c.is_boilerplate{filter}\n        GROUP BY c.id",
     "WHERE NOT c.is_boilerplate\n        GROUP BY c.id"),
    ("src/retrieval/hybrid.py", "the speaker leg ignores the filter",
     "WHERE sp.mnis_id = ANY(%(members)s::int[]) AND NOT c.is_boilerplate AND c.embedding IS NOT NULL{filter}",
     "WHERE sp.mnis_id = ANY(%(members)s::int[]) AND NOT c.is_boilerplate AND c.embedding IS NOT NULL"),
    ("src/retrieval/hybrid.py", "an empty filter still emits a predicate",
     'return "".join(f" AND {p}" for p in parts)',
     'return " AND true" + "".join(f" AND {p}" for p in parts)'),
    ("src/retrieval/hybrid.py", "the exact-scan plan is used even with no filter",
     "if not cfg.exact_scan_max_rows or not cfg.filters:", "if False:"),
    ("src/retrieval/hybrid.py", "the exact-scan CTE is not materialised, so the index is used anyway",
     "scoped AS MATERIALIZED (", "scoped AS ("),
    ("src/retrieval/hybrid.py", "a dict is accepted where SearchFilters is required",
     "if not isinstance(self.filters, SearchFilters):", "if False:"),
]  # fmt: skip

MUTANTS = [Mutant(*row) for row in _RAW]
