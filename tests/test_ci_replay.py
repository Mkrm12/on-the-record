"""The local CI replay: CI's test job, run from the workflow file itself, on a clean tree."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ci"))

import replay as R  # noqa: E402

WORKFLOW = """
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - name: Lint
        run: uvx ruff check .

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:0.8.6-pg16
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: hansard_v2
    env:
      PG_DSN: postgresql://postgres:postgres@127.0.0.1:5432/hansard_v2
    steps:
      - name: Check out the repository
        uses: actions/checkout@abc # v7
        with:
          persist-credentials: false

      - name: Install dependencies from the lockfile
        run: uv sync --locked

      # a comment between steps
      - name: Apply the baseline schema
        run: |
          command -v psql >/dev/null || sudo apt-get install -y postgresql-client
          psql "$PG_DSN" -v ON_ERROR_STOP=1 -f db/schema.sql

      - name: Run the tests
        run: uv run pytest -q
"""


def test_the_test_jobs_service_and_steps_are_read_from_the_workflow():
    job = R.read_job(WORKFLOW, "test")
    assert job.image == "pgvector/pgvector:0.8.6-pg16"
    assert job.database == "hansard_v2" and job.password == "postgres"
    assert [s.name for s in job.steps] == ["Install dependencies from the lockfile", "Apply the baseline schema",
                                           "Run the tests"]  # `uses:` steps are the runner's, not the job's commands
    assert job.steps[1].run.splitlines() == [
        "command -v psql >/dev/null || sudo apt-get install -y postgresql-client",
        'psql "$PG_DSN" -v ON_ERROR_STOP=1 -f db/schema.sql']
    assert R.read_job(WORKFLOW, "lint").steps[0].run == "uvx ruff check ."


def test_the_real_workflows_test_job_is_replayed_step_for_step():
    """Every `run:` step of ci.yml's test job, in order, and the service it declares: nothing is restated here."""
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = R.read_job(text, "test")
    assert job.image.startswith("pgvector/pgvector:") and job.database == "hansard_v2"
    names = [s.name for s in job.steps]
    assert names[:4] == ["Install dependencies from the lockfile", "Apply the baseline schema", "Apply migrations",
                         "Migrations are complete and idempotent"]
    assert "Run the tests" in names and any("GPU" in n for n in names)
    assert all(s.run.strip() for s in job.steps)
    assert text.count("run:") >= len(job.steps)


@pytest.mark.parametrize(("argv", "stdin", "args"), [
    (['postgresql://x@127.0.0.1:5/hansard_v2', "-v", "ON_ERROR_STOP=1", "-f", "db/schema.sql"], "db/schema.sql",
     ["-v", "ON_ERROR_STOP=1"]),
    (['postgresql://x@127.0.0.1:5/hansard_v2', "-c", "SELECT 1"], None, ["-c", "SELECT 1"]),
])
def test_the_psql_shim_runs_the_steps_psql_inside_the_container(argv, stdin, args):
    """CI installs psql on the runner; the replay has Docker, so the step's own `psql "$PG_DSN"...` runs in the service
    container, its `-f` file fed on stdin.
    """
    assert R.psql_call(argv) == (stdin, args)


def test_a_psql_call_the_shim_cannot_translate_is_refused():
    with pytest.raises(ValueError):
        R.psql_call(["-h", "elsewhere", "-c", "SELECT 1"])  # no DSN first: not the workflow's form


def test_the_real_workflows_lint_job_is_replayed_too_with_no_service():
    """CI is both jobs: the lint job's ruff, secret scan and release-archive check run with no database."""
    job = R.read_job((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"), "lint")
    assert not job.image and [s.name for s in job.steps] == ["Lint", "No credentials in tracked files",
                                                             "A release archive holds no gold"]

