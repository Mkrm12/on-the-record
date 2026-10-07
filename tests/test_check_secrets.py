"""The credential scan. A scanner that finds nothing is indistinguishable from a scanner that is broken, so each test
plants a fake secret and demands it be found, and each pattern is checked against the near miss that made it worth
writing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_secrets  # noqa: E402

PLANTED = {
    "DeepSeek/OpenAI key": "sk-" + "a1b2c3d4e5f6g7h8i9j0k1l2m3n4",
    "Hugging Face token": "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345",
    "AWS access key id": "AKIA" + "IOSFODNN7EXAMPLE"[:16],
    "Google API key": "AIza" + "Sy" + "A" * 33,
    "password in a connection string": "postgresql://app:sup3rsecretvalue@db.example.com:5432/prod",
}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch tree the scanner treats as the repository."""
    monkeypatch.setattr(check_secrets, "ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize(("label", "value"), sorted(PLANTED.items()))
def test_every_pattern_finds_its_own_secret(repo, label, value):
    (repo / "leak.py").write_text(f'TOKEN = "{value}"\n', encoding="utf-8")
    findings = check_secrets.scan([repo / "leak.py"])
    assert len(findings) == 1 and label in findings[0]
    assert value not in findings[0], "a finding must not reprint the credential it found"


def test_a_finding_says_where_and_what_to_do(repo):
    (repo / "config" / "settings.py").parent.mkdir()
    (repo / "config" / "settings.py").write_text(f'K = "{PLANTED["Hugging Face token"]}"\n', encoding="utf-8")
    finding = check_secrets.scan([repo / "config" / "settings.py"])[0]
    assert finding.startswith("config/settings.py:1:") and "revoke it" in finding


@pytest.mark.parametrize("dsn", [
    "postgresql://postgres:postgres@127.0.0.1:5432/hansard_v2",  # the local container; not a secret
    "postgresql://u:p@127.0.0.1:5432/hansard_v2",  # a test stand-in
    "postgresql://postgres:postgres@db:5432/hansard",
])
def test_throwaway_connection_strings_are_not_flagged(repo, dsn):
    """The 15 false positives that made the length floor necessary. A check nobody reads catches nothing."""
    (repo / "test_thing.py").write_text(f'DSN = "{dsn}"\n', encoding="utf-8")
    assert check_secrets.scan([repo / "test_thing.py"]) == []


@pytest.mark.parametrize("text", [
    "hf_short",  # too short to be a token
    "sk-only-a-few",  # too short to be a key
    "AKIA_not_a_key",  # not the fixed length
    "see sk- prefixed keys in .env",  # prose about the shape
    "AIzaTooShort",
])
def test_near_misses_are_left_alone(repo, text):
    (repo / "notes.md").write_text(text + "\n", encoding="utf-8")
    assert check_secrets.scan([repo / "notes.md"]) == []


def test_binary_and_generated_files_are_skipped(repo):
    (repo / "weights.safetensors").write_text(PLANTED["Hugging Face token"], encoding="utf-8")
    assert check_secrets.scan([repo / "weights.safetensors"]) == []


def test_the_dotenv_file_is_where_secrets_belong(repo):
    """Finding a real token in `.env` is the system working, not a leak. What matters is whether git sees the file --
    and the check asks git, because an earlier version inferred it from a file listing and called a properly ignored
    `.env` "tracked".
    """
    (repo / ".env").write_text(f'HF_TOKEN={PLANTED["Hugging Face token"]}\n', encoding="utf-8")
    assert check_secrets.scan([repo / ".env"]) == []
    assert check_secrets.env_findings([repo / ".env"]) == [], "an untracked .env is correct"


def test_a_tracked_dotenv_is_a_finding(repo, monkeypatch):
    monkeypatch.setattr(check_secrets, "is_tracked", lambda name: name == ".env")
    findings = check_secrets.env_findings([])
    assert len(findings) == 1 and "git rm --cached .env" in findings[0]


def test_the_example_env_may_hold_placeholders_but_not_values(repo):
    (repo / ".env.example").write_text(
        "# comment\n"
        "PG_DSN=postgresql://postgres:postgres@127.0.0.1:5432/hansard\n"
        "DEEPSEEK_KEY=your-key-here\n"
        "HF_TOKEN=\n",
        encoding="utf-8",
    )
    assert check_secrets.env_findings([]) == []
    (repo / ".env.example").write_text(f"HF_TOKEN={PLANTED['Hugging Face token']}\n", encoding="utf-8")
    findings = check_secrets.env_findings([])
    assert len(findings) == 1 and "placeholder" in findings[0]


def test_the_repository_itself_is_clean():
    """The check that actually matters: run against this repository, as CI runs it."""
    assert check_secrets.main([]) == 0
