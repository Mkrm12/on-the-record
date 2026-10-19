"""The release archive: the only way code reaches a server."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every path in an archive sits under this folder.
PREFIX = "otr/"

# Paths a release must never hold.
FORBIDDEN_PATHS = (
    (re.compile(r"^tests/"), "the tests and their data, the gold set among them"),
    (re.compile(r"^logs/"), "the working history"),
    (re.compile(r"^reports/"), "the reports, which hold judged answers and gold text"),
    (re.compile(r"^tools/(?:review|mirror)/"), "review pages and the mirror's own tooling"),
    (re.compile(r"^src/goldset/(?:human_items|adversarial_items|w5_items|authored)\.py$"), "the gold item modules"),
    (re.compile(r"^scripts/(?:build_gold|build_gold_w5|audit_w5_items|w5_blind_worksheet)\.py$"),
     "the scripts that build or read the gold"),
    # review pass 2: every script that holds, repairs or reads gold (the dev rewrites live in migrate_gold_v3.py, test
    # ids in repair_gold.py) and all of W5's tooling.
    (re.compile(r"^scripts/[^/]*(?:gold|w5)[^/]*\.py$", re.I), "a script that holds, repairs or reads the gold"),
    (re.compile(r"(?:^|/)(?:gold_w5[^/]*|[^/]*w5[^/]*)\.(?:json|md|jsonl)$", re.I), "W5 data"),
    (re.compile(r"(?:^|/)gold[^/]*\.json$", re.I), "gold data"),
    (re.compile(r"(?:^|/)\.env$"), "credentials"),
    (re.compile(r"^CLAUDE\.md$"), "the working instructions"),
)
# Content a release must never hold, whatever the file is called: the exam's items, or an import of any gold module.
FORBIDDEN_CONTENT = re.compile(
    r"\bW5_AUTHORED\b|goldset\.(?:w5_items|human_items|adversarial_items)\b"
    r"|from goldset import (?:w5_items|human_items|adversarial_items)\b|\bimport w5_blind_worksheet\b")


def path_problems(paths: list[str]) -> list[str]:
    """Each path that must not be in a release, with why."""
    out = []
    for path in paths:
        for pattern, why in FORBIDDEN_PATHS:
            if pattern.search(path):
                out.append(f"{path}: {why}")
                break
    return out


def content_problems(path: str, text: str) -> list[str]:
    m = FORBIDDEN_CONTENT.search(text)
    return [f"{path}: holds or imports gold ({m.group(0)!r})"] if m else []


def build(ref: str, out: Path, repo: Path = ROOT) -> Path:
    """`git archive` of `ref`, honouring that tree's own `.gitattributes`, as a.tar.gz under `PREFIX`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "archive", "--format=tar.gz", f"--prefix={PREFIX}", "-o", str(out), ref], cwd=repo,
                   check=True, capture_output=True)  # fmt: skip
    return out


def members(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        return [m.name[len(PREFIX):] if m.name.startswith(PREFIX) else m.name for m in tar.getmembers() if m.isfile()]


def problems(archive: Path) -> list[str]:
    """Everything in the archive that must not reach a server. Empty means it may be released."""
    out = path_problems(members(archive))
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            try:
                text = handle.read().decode("utf-8")
            except UnicodeDecodeError:
                continue
            name = member.name[len(PREFIX):] if member.name.startswith(PREFIX) else member.name
            out += content_problems(name, text)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--ref", default="HEAD", help="the tag (or, for --check-only, any ref) to archive")
    p.add_argument("--out", type=Path, default=ROOT / "dist", help="the folder the archive is written to")
    p.add_argument("--check-only", action="store_true", help="build into a temporary file, check it, and delete it")
    args = p.parse_args(argv)
    if args.check_only:
        with tempfile.TemporaryDirectory() as tmp:
            archive = build(args.ref, Path(tmp) / "release.tar.gz")
            found, count = problems(archive), len(members(archive))
    else:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.ref)
        archive = build(args.ref, args.out / f"otr-{name}.tar.gz")
        found, count = problems(archive), len(members(archive))
    if found:
        print(f"REFUSED: the release archive of {args.ref} holds {len(found)} thing(s) a server must never get:")
        for problem in found[:40]:
            print(f"  {problem}")
        return 1
    print(f"the release archive of {args.ref}: {count} files, no gold, no W5 code, no working history"
          + ("" if args.check_only else f" ({archive})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
