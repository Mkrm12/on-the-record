"""Code reaches a server only as a release archive, and the archive holds no gold, no W5 code and no working history.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_archive_t", ROOT / "scripts" / "release_archive.py")
RA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RA)

FORBIDDEN = ["tests/data/gold_v2.json", "tests/test_gold_w5.py", "logs/prompt.md", "reports/e3_eval_test_judged.json",
             "tools/review/review_template.html", "tools/mirror/build_mirror.py", "src/goldset/w5_items.py",
             "src/goldset/human_items.py", "src/goldset/adversarial_items.py", "src/goldset/authored.py",
             "scripts/build_gold.py", "scripts/build_gold_w5.py", "scripts/audit_w5_items.py",
             "scripts/w5_blind_worksheet.py", "data/gold_w5.json", ".env", "CLAUDE.md",
             # review pass 2: every script that holds, repairs or reads gold, and W5's tooling
             "scripts/migrate_gold_v3.py", "scripts/populate_gold_nuggets.py", "scripts/repair_gold.py",
             "scripts/review_gold.py", "scripts/audit_gold_objective.py", "scripts/w5_compare_graders.py"]
ALLOWED = ["src/answer/pipeline.py", "src/app/sealed.py", "config/sealed.toml", "db/schema.sql", "pyproject.toml",
           "scripts/pipeline.py", "src/goldset/llm.py"]


@pytest.mark.parametrize("path", FORBIDDEN)
def test_every_forbidden_path_is_refused(path):
    assert RA.path_problems([path]), path


def test_what_a_server_needs_is_allowed():
    assert RA.path_problems(ALLOWED) == []


@pytest.mark.parametrize("text", ["from goldset.w5_items import W5_AUTHORED", "import w5_blind_worksheet",
                                  "from goldset.human_items import HUMAN", "from goldset import adversarial_items"])
def test_a_file_that_holds_or_imports_gold_is_refused_whatever_its_name(text):
    assert RA.content_problems("src/app/innocent.py", text)


def test_ordinary_code_passes_the_content_check():
    assert RA.content_problems("src/answer/pipeline.py", "from goldset.llm import Client\n") == []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_real_gitattributes_leaves_every_forbidden_path_out_of_a_real_archive(tmp_path):
    """Built from the repository's own `.gitattributes`, so the rules that ship are the rules that are tested."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    shutil.copy2(ROOT / ".gitattributes", repo / ".gitattributes")
    for rel in [p for p in FORBIDDEN if p != ".env"] + ALLOWED:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {rel}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    archive = RA.build("HEAD", tmp_path / "release.tar.gz", repo=repo)
    names = RA.members(archive)
    assert set(ALLOWED) <= set(names)
    assert RA.problems(archive) == [], RA.problems(archive)
    assert not any(n.startswith(("tests/", "logs/", "reports/")) for n in names)


def test_an_archive_that_holds_gold_is_refused(tmp_path):
    import tarfile

    bad = tmp_path / "bad.tar.gz"
    payload = tmp_path / "gold_v2.json"
    payload.write_text("{}", encoding="utf-8")
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(payload, arcname=f"{RA.PREFIX}tests/data/gold_v2.json")
    assert any("tests/data/gold_v2.json" in p for p in RA.problems(bad))
