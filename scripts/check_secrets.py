"""Refuse to ship a credential. Runs in CI and locally; no network, no third-party action."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name -> (pattern, what to do about it).
PATTERNS: dict[str, tuple[re.Pattern[str], str]] = {
    "DeepSeek/OpenAI key": (
        re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
        "revoke it in the provider's console, then read it from .env instead",
    ),
    "Hugging Face token": (
        re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
        "revoke it at huggingface.co/settings/tokens, then read it from .env instead",
    ),
    "AWS access key id": (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "revoke it in IAM"),
    "Google API key": (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "revoke it in the Google console"),
    "password in a connection string": (
        # A postgres URL carrying a password that could plausibly be a real one: at least eight characters and not the
        # local container's throwaway pair.
        re.compile(r"postgres(?:ql)?://(?!postgres:postgres@)[^\s:/@]+:(?!postgres@)[^\s:/@]{8,}@"),
        "use $PG_DSN, or the local container's postgres:postgres, which is not a secret",
    ),
}

# Where a match is expected and means nothing: this file defines the patterns, and the tests exercise them.
ALLOWED: frozenset[str] = frozenset({
    "scripts/check_secrets.py",
    "tests/test_check_secrets.py",
})

# Binary and generated trees that are never worth reading.
SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", "data"})
SKIP_SUFFIXES = frozenset({".gz", ".zip", ".npy", ".bin", ".safetensors", ".png", ".jpg", ".pdf", ".pyc"})

# `.env` is the designated home for real credentials, so finding one in it is the system working.
SECRET_HOMES = frozenset({".env"})

PLACEHOLDERS = ("your-", "changeme", "xxx", "...", "<", "example", "replace")


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / name for name in out.stdout.split("\0") if name]


def all_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not SKIP_DIRS & set(path.relative_to(ROOT).parts)
    ]


def readable(path: Path) -> bool:
    return path.suffix.lower() not in SKIP_SUFFIXES and path.is_file()


def scan(paths: list[Path]) -> list[str]:
    """Every finding, as a line a reader can act on."""
    findings = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED or rel in SECRET_HOMES or not readable(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, (pattern, advice) in PATTERNS.items():
            for number, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    findings.append(f"{rel}:{number}: {name} -- {advice}")
    return findings


def is_tracked(name: str) -> bool:
    """Whether git has this path in the index. Asked of git, not inferred from a file listing: under --all the listing
    includes untracked files, and an earlier version of this check called an ignored.env "tracked" because it was in
    the list.
    """
    return subprocess.run(["git", "ls-files", "--error-unmatch", name], cwd=ROOT,
                          capture_output=True, text=True).returncode == 0  # fmt: skip


def env_findings(paths: list[Path]) -> list[str]:
    """`.env` must not be tracked, and `.env.example` must hold placeholders rather than real values."""
    findings = []
    for name in sorted(SECRET_HOMES):
        if is_tracked(name):
            findings.append(f"{name} is tracked by git: remove it with `git rm --cached {name}` and keep it ignored")
    example = ROOT / ".env.example"
    if example.is_file():
        for number, line in enumerate(example.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("\"'")
            if not value or any(hint in value.lower() for hint in PLACEHOLDERS) or value == "postgres:postgres":
                continue
            if _looks_local(value) or len(value) < 16:
                continue
            findings.append(f".env.example:{number}: {key.strip()} looks like a real value, not a placeholder")
    return findings


def _looks_local(value: str) -> bool:
    return any(hint in value for hint in ("127.0.0.1", "localhost", "postgres:postgres"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="scan every file, not only the tracked ones")
    args = p.parse_args(argv)

    paths = all_files() if args.all else tracked_files()
    findings = scan(paths) + env_findings(paths)
    for finding in findings:
        print(finding)
    print(f"\n{len(findings)} finding(s) in {len(paths):,} files"
          if findings else f"no credentials found in {len(paths):,} files")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
