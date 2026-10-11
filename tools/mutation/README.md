# Mutation testing

A mutant is one small deliberate bug (`>` becomes `>=`, a guard is deleted). A good test suite fails on every
mutant; a mutant that survives points at a behaviour no test pins down.

```powershell
uv run python tools/mutation/run.py                  # all suites, about twelve minutes: run them one at a time
uv run python tools/mutation/run.py --suite rules    # rules | divisions | core | gold | gpu | service | retrieval | search
uv run python tools/mutation/run.py --workers 1      # serial, easier to read while debugging
```

Every mutant runs in a temporary copy of `src/` and `tests/`, so the working tree is never modified, even if a
run is interrupted. The divisions suite runs database-backed tests, which use a transaction that is always
rolled back.

## Outcomes

| Outcome | Meaning | Fails the run? |
|---|---|---|
| KILLED | The tests failed (or hit the timeout, i.e. an infinite loop was noticed) | no |
| EQUIVALENT | Survived, and the catalog documents why the change cannot alter behaviour | no |
| SURVIVED | The tests still passed: a gap in the tests | yes |
| INVALID | The pattern was not found (the source moved on), the mutant did not import, or pytest errored | yes |

INVALID is what stops a stale catalog from passing silently. A run also refuses to start a suite whose
unmutated baseline is red, since every mutant would then look killed.

## Suites (status 2026-09-21: 418 of 418 killed or documented equivalent)

| Suite | Target | Tests run | Mutants |
|---|---|---|---|
| `rules` | `src/ingest/rules.py` | `tests/test_rules.py` | 18 |
| `divisions` | `src/ingest/divisions.py` | `tests/test_divisions.py` | 20 |
| `core` | `text`, `chunk`, `wikidata`, `bills`, `hansard_meta` | their test files | 36 |
| `gold` | `src/app/gold.py`, `src/ingest/window.py` | `tests/test_gold_eval.py`, `tests/test_window.py` | 19 |
| `gpu` | `gpu/embedder.py`, `gpu/check_cuda.py` | `test_embedder.py`, `test_check_cuda.py` without the tests that need a GPU | 36 |
| `service` | `gpu/api.py`, `gpu/reranker.py` | `test_api.py`, `test_reranker.py` without the tests that need a GPU | 105 |
| `retrieval` | `src/app/db.py`, `src/retrieval/` (query, collapse, lexicon, metrics, speakers, gpu_client, evaluation, and the pure ranking decisions in hybrid.py) | their test files, and `test_db_vectors.py` | 133 |
| `search` | the SQL and orchestration in `src/retrieval/hybrid.py` | `test_retrieval_hybrid.py` against a scratch database (about 15 s a mutant) | 51 |

## Adding a mutant

Add a row to the matching `catalog_*.py`: the file, a name, the exact text to find (the first occurrence is
replaced) and its replacement. Run that suite; the row must come out KILLED. If it survives, write the missing
test first, then re-run. If it survives because the change genuinely cannot alter behaviour, set
`equivalent="why"` on the `Mutant`. When source code is refactored and a row turns INVALID, update its `old`
text rather than deleting the row.

The `gpu` and `service` suites run with the `gpu/` project's own interpreter (`gpu/.venv`), so they need `uv sync` in
`gpu/`; without it `run.py` skips them with a message, and `--suite gpu` explains what is missing. What the real
models do is covered by the `test_gpu_*` tests and is not mutation tested. A suite's `Project` in
`mutation_engine.py` says which directories are copied and where pytest runs. Virtualenvs and caches are never copied
(`gpu/.venv` is several GB; copying it once made a run take 9 minutes instead of 30 seconds).
