"""Replay CI locally: a clean tree, a throwaway Postgres for the job that has one, and the workflow's own steps."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Step:
    name: str
    run: str


@dataclass(frozen=True)
class Job:
    image: str
    database: str
    password: str
    env: dict[str, str] = field(default_factory=dict)
    steps: tuple[Step, ...] = ()


def _block(lines: list[str], start: int) -> tuple[list[str], int]:
    """The lines indented deeper than line `start`, and the index after them (blank lines included)."""
    depth = len(lines[start]) - len(lines[start].lstrip())
    out, i = [], start + 1
    while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > depth):
        out.append(lines[i])
        i += 1
    while out and not out[-1].strip():
        out.pop()
    return out, i


def _value(line: str) -> str:
    value = line.split(":", 1)[1].strip()
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"" else value


def read_job(text: str, job: str) -> Job:
    """The job's service image, database and password, its `env:`, and its `run:` steps in order (the subset of YAML
    GitHub workflows use here: block mappings, `- name:` steps, `run:` inline or as a `|` block).
    """
    lines = [line.rstrip("\n") for line in text.splitlines()]
    begin = next((i for i, line in enumerate(lines) if re.fullmatch(rf"\s+{re.escape(job)}:\s*", line)), None)
    if begin is None:
        raise ValueError(f"no job {job!r} in the workflow")
    body, _ = _block(lines, begin)
    image = database = password = ""
    env: dict[str, str] = {}
    steps: list[Step] = []
    i = 0
    while i < len(body):
        line, key = body[i], body[i].strip()
        if key.startswith("image:"):
            image = _value(key)
        elif key.startswith("POSTGRES_DB:"):
            database = _value(key)
        elif key.startswith("POSTGRES_PASSWORD:"):
            password = _value(key)
        elif key == "env:" and len(line) - len(line.lstrip()) == len(body[0]) - len(body[0].lstrip()):
            block, i = _block(body, i)
            env = {b.split(":", 1)[0].strip(): _value(b.strip()) for b in block if ":" in b}
            continue
        elif key.startswith("- name:"):
            name = _value(key[2:])
            block, after = _block(body, i)
            run = None
            for j, inner in enumerate(block):
                if inner.strip().startswith("run:"):
                    value = _value(inner.strip())
                    if value in ("|", "|-", ">", ">-"):
                        sub, _ = _block(block, j)
                        indent = min(len(s) - len(s.lstrip()) for s in sub if s.strip())
                        run = "\n".join(s[indent:] for s in sub)
                    else:
                        run = value
            if run is not None:
                steps.append(Step(name, run))
            i = after
            continue
        i += 1
    return Job(image, database, password, env, tuple(steps))


def psql_call(argv: list[str]) -> tuple[str | None, list[str]]:
    """The workflow's `psql "$PG_DSN" [options] [-f FILE]` as (the file to feed on stdin, the options to pass)."""
    if not argv or not argv[0].startswith("postgres"):
        raise ValueError(f"psql is called as the workflow calls it, with $PG_DSN first: {argv!r}")
    rest, stdin, i = [], None, 1
    while i < len(argv):
        if argv[i] == "-f" and i + 1 < len(argv):
            stdin, i = argv[i + 1], i + 2
            continue
        rest.append(argv[i])
        i += 1
    return stdin, rest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tree(repo: Path, source: str, into: Path) -> None:
    """What a checkout holds: `head` clones the commit; `worktree` copies tracked and unignored files (never `.env`).
    """
    if source == "head":
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(into)], check=True)
        return
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout.splitlines()  # fmt: skip
    for rel in listed:
        src = repo / rel
        if src.is_file():
            (into / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, into / rel)
    # A checkout is a repository: tests ask git which files are tracked (the secret scan, the mirror's allow-list), so
    # the copy is committed into a fresh one, with no hooks and an identity that is nobody's.
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=ci-replay", "-c", "user.email=ci-replay@example.invalid",
                                                "commit", "-q", "-m", "ci replay of the working tree"]):  # fmt: skip
        subprocess.run(["git", *args], cwd=into, check=True, capture_output=True)


def _shim(bin_dir: Path, container: str, database: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    python, me = Path(sys.executable).as_posix(), Path(__file__).resolve().as_posix()
    script = bin_dir / "psql"
    script.write_text(f'#!/usr/bin/env bash\nexec "{python}" "{me}" --psql {container} {database} "$@"\n',
                      encoding="utf-8", newline="\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def _run_psql(container: str, database: str, argv: list[str]) -> int:
    stdin, args = psql_call(argv)
    cmd = ["docker", "exec", "-i", container, "psql", "-U", "postgres", "-d", database, *args]
    if stdin is None:
        return subprocess.run(cmd).returncode
    with open(stdin, "rb") as fh:
        return subprocess.run(cmd, stdin=fh).returncode


def _ready(container: str, database: str, timeout: float = 90.0) -> None:
    """Ready over TCP: the image's init server listens on the socket only, so TCP means the final server is up."""
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        probe = subprocess.run(["docker", "exec", container, "pg_isready", "-h", "127.0.0.1", "-U", "postgres", "-d",
                                database], capture_output=True)  # fmt: skip
        if probe.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError(f"the service container {container} was not ready in {timeout:.0f}s")


def _run_job(name: str, job: Job, tree: Path, workdir: Path, bash: str) -> int:
    """One job's `run:` steps, in order, stopping at the first failure as GitHub does. A job with a Postgres service
    gets a throwaway container on a free loopback port, always removed.
    """
    logs = workdir / f"logs-{name}"  # beside the tree, never in it: a test that lists the tree's files must not see them
    logs.mkdir()
    env = {**os.environ, **job.env}
    env.pop("VIRTUAL_ENV", None)  # the replayed tree builds its own environment, as the runner does
    container = None
    if job.image and job.database:
        port, container = _free_port(), f"otr-ci-replay-{os.getpid()}-{name}"
        if "PG_DSN" in env:
            env["PG_DSN"] = re.sub(r"@([^/:]+):\d+/", f"@127.0.0.1:{port}/", env["PG_DSN"])
        _shim(workdir / f"bin-{name}", container, job.database)
        env["PATH"] = str(workdir / f"bin-{name}") + os.pathsep + env.get("PATH", "")
        print(f"job {name}: {len(job.steps)} steps on {job.image}, port {port}")
        subprocess.run(["docker", "run", "-d", "--rm", "--name", container, "-e", f"POSTGRES_PASSWORD={job.password}",
                        "-e", f"POSTGRES_DB={job.database}", "-p", f"127.0.0.1:{port}:5432", job.image],
                       check=True, capture_output=True)  # fmt: skip
    else:
        print(f"job {name}: {len(job.steps)} steps, no service")
    try:
        if container:
            _ready(container, job.database)
        for n, step in enumerate(job.steps, start=1):
            script = logs / f"{n:02d}.sh"
            script.write_text(step.run + "\n", encoding="utf-8", newline="\n")
            began = time.monotonic()
            # Output goes to a file, not a pipe: a step may write to /dev/stderr ("tee /dev/stderr"), which Git Bash
            # cannot open on a Windows pipe.
            log = logs / f"{n:02d}.log"
            with open(log, "wb") as out:
                code = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()], cwd=tree,
                                      env=env, stdout=out, stderr=subprocess.STDOUT).returncode  # fmt: skip
            print(f"  {'ok  ' if code == 0 else 'FAIL'} {step.name} ({time.monotonic() - began:.0f}s)")
            if code != 0:
                tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-25:]
                print("\n".join("      " + t for t in tail))
                print(f"      full log: {log}")
                return 1
        return 0
    finally:
        if container:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True)


def replay(repo: Path, source: str, keep: bool, jobs: tuple[str, ...] = ("lint", "test")) -> int:
    """Every named job of the replayed tree's own workflow, each on the same clean tree."""
    workdir = Path(tempfile.mkdtemp(prefix="otr-ci-replay-"))
    tree = workdir / "tree"
    tree.mkdir()
    _tree(repo, source, tree)
    bash = shutil.which("bash")
    if bash is None:
        raise SystemExit("bash is needed to run the workflow's steps as GitHub does")
    workflow = (tree / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    print(f"replaying {', '.join(jobs)} ({source} of {repo})")
    failed = 0
    try:
        for name in jobs:
            failed += _run_job(name, read_job(workflow, name), tree, workdir, bash)
    finally:
        if not (failed or keep):
            shutil.rmtree(workdir, ignore_errors=True)
    print("CI replay: " + (f"FAILED ({failed} job{'s' if failed != 1 else ''})" if failed else "every step passed"))
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--psql"]:
        return _run_psql(argv[1], argv[2], argv[3:])
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--repo", type=Path, default=ROOT, help="the repository to replay (default: this one)")
    p.add_argument("--source", choices=("worktree", "head"), default="worktree",
                   help="worktree: tracked and unignored files as they are now; head: the last commit")
    p.add_argument("--keep", action="store_true", help="keep the replayed tree and its logs even when every step passes")
    p.add_argument("--jobs", default="lint,test", help="the workflow's jobs to replay, in order (default: lint,test)")
    args = p.parse_args(argv)
    return replay(args.repo.resolve(), args.source, args.keep, tuple(j for j in args.jobs.split(",") if j))


if __name__ == "__main__":
    raise SystemExit(main())
