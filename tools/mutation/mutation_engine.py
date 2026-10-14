"""Mutation testing that never touches the working tree."""

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTEST_ARGS = ["-q", "--no-header", "-p", "no:cacheprovider", "-x"]


class MutationError(ValueError):
    pass


@dataclass(frozen=True)
class Mutant:
    path: str  # repo-relative file to mutate, e.g.
    name: str
    old: str  # text to find (first occurrence is replaced)
    new: str
    equivalent: str = ""  # why a surviving mutant is behaviourally identical; empty means it must be killed


@dataclass(frozen=True)
class Project:
    """Where a suite's code and tests live and how to run them (the gpu/ project has its own layout and interpreter).
    """

    dirs: tuple[str, ...] = ("src", "tests")  # repo-relative directories copied into the throw-away tree
    files: tuple[str, ...] = ()  # repo-relative single files a suite's tests read
    cwd: str = ""  # directory inside the copy to run pytest from
    python: str | None = None  # interpreter for pytest; default is the one running this tool
    pytest_args: tuple[str, ...] = ()  # e.g.


DEFAULT_PROJECT = Project()


@dataclass(frozen=True)
class Outcome:
    status: str  # killed | survived | equivalent | invalid
    detail: str = ""


def apply_mutation(source: str, old: str, new: str) -> str:
    if old == new:
        raise MutationError("old and new are identical: not a mutation")
    if old not in source:
        raise MutationError("pattern not found (the source changed; update the catalog)")
    return source.replace(old, new, 1)


def _copy_project(root: Path, work: Path, project: Project) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".pytest_cache", ".git")  # gpu/.venv is several GB
    for name in project.dirs:
        shutil.copytree(root / name, work / name, ignore=ignore)
    for name in project.files:
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, work / name)
    if (root / "pyproject.toml").exists():
        shutil.copy2(root / "pyproject.toml", work / "pyproject.toml")


def _pytest(work: Path, tests: list[str], timeout: float, project: Project) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")  # a stale.pyc must never mask a mutation
    return subprocess.run(
        [project.python or sys.executable, "-m", "pytest", *tests, *PYTEST_ARGS, *project.pytest_args],
        cwd=work / project.cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )  # fmt: skip


def baseline_passes(tests: list[str], root: Path = ROOT, timeout: float = 300, project: Project = DEFAULT_PROJECT) -> bool:
    """The unmutated copy must be green, otherwise every mutant would look 'killed'."""
    with tempfile.TemporaryDirectory(prefix="mutation-baseline-") as tmp:
        work = Path(tmp)
        _copy_project(root, work, project)
        try:
            return _pytest(work, tests, timeout, project).returncode == 0
        except (OSError, subprocess.TimeoutExpired):  # no such interpreter, or hung
            return False


def run_mutant(mutant: Mutant, tests: list[str], root: Path = ROOT, timeout: float = 180, project: Project = DEFAULT_PROJECT) -> Outcome:
    with tempfile.TemporaryDirectory(prefix="mutant-") as tmp:
        work = Path(tmp)
        _copy_project(root, work, project)
        target = work / mutant.path
        try:
            mutated = apply_mutation(target.read_text(encoding="utf-8"), mutant.old, mutant.new)
        except (MutationError, FileNotFoundError) as exc:
            return Outcome("invalid", str(exc))
        target.write_text(mutated, encoding="utf-8", newline="\n")
        try:
            proc = _pytest(work, tests, timeout, project)
        except subprocess.TimeoutExpired:
            return Outcome("killed", f"timeout after {timeout:g}s")
        except OSError as exc:  # the interpreter could not be started
            return Outcome("invalid", str(exc))
    last = (proc.stdout.strip().splitlines() or proc.stderr.strip().splitlines() or [""])[-1]
    if proc.returncode == 1:
        return Outcome("killed", last)
    if proc.returncode == 0:
        return Outcome("equivalent" if mutant.equivalent else "survived", mutant.equivalent)
    return Outcome("invalid", f"pytest exit {proc.returncode}: {last}")
