"""Every text file the repository's code writes pins its line ending."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Modules whose `open(file, mode)` takes the mode second, as the builtin does (and that accept `newline=`).
_MODULE_OPENS = {"io", "gzip", "bz2", "lzma", "builtins"}
# Temporary files, whose mode comes first and is binary unless it says otherwise.
_TEMPFILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}
# Opens that take no `newline=` at all: their modes are never text-mode writes in this sense.
_NOT_TEXT_OPENS = {"tarfile", "zipfile", "os", "codecs", "webbrowser", "Image", "shelve", "dbm"}


def _mode(node: ast.Call, position: int) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) else None
    if len(node.args) > position and isinstance(node.args[position], ast.Constant):
        value = node.args[position].value
        return value if isinstance(value, str) else None
    return None


def unpinned(source: str) -> list[tuple[int, str]]:
    """Each text-mode write in `source` that does not pin `newline=`: (line, what)."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or any(k.arg == "newline" for k in node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "write_text":
            found.append((node.lineno, "write_text"))
            continue
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if name == "TextIOWrapper":
            found.append((node.lineno, "TextIOWrapper"))
            continue
        if name in _TEMPFILES:
            mode = _mode(node, 0)
            if mode and "b" not in mode:
                found.append((node.lineno, f"{name}({mode!r})"))
            continue
        if name == "fdopen":
            mode = _mode(node, 1)
            if mode and any(c in mode for c in "wax") and "b" not in mode:
                found.append((node.lineno, f"fdopen({mode!r})"))
            continue
        if isinstance(func, ast.Name) and func.id == "open":
            mode = _mode(node, 1)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            if owner in _NOT_TEXT_OPENS:
                continue
            mode = _mode(node, 1 if owner in _MODULE_OPENS else 0)
        else:
            continue
        if mode and any(c in mode for c in "wax") and "b" not in mode:
            found.append((node.lineno, f"open({mode!r})"))
    return found


def _tracked_modules() -> list[Path]:
    # untracked modules too: a session's new files are not yet tracked when its checks run (QA pass 9, Low 10)
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src", "scripts",
                             "tools", "gpu"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    return [ROOT / name for name in listed if name.endswith(".py") and (ROOT / name).exists()]


def test_the_rule_sees_every_text_write_form():
    source = "\n".join([
        "Path('a').write_text('x')",
        "open('a', 'w')",
        "open('a', mode='a', encoding='utf-8')",
        "p.open('w')",
        "gzip.open('a.gz', 'wt')",
        "Path('a').write_text('x', newline='\\n')",  # pinned
        "open('a', 'wb')",  # bytes
        "open('a')",  # a read
        "tarfile.open('a.tgz', 'w:gz')",  # no newline= to pin
        "p.open('rb')",
        # QA pass 9, Low 10: the other ways a text file is opened for writing
        "tempfile.NamedTemporaryFile('w', suffix='.json')",
        "tempfile.NamedTemporaryFile(mode='w+')",
        "tempfile.NamedTemporaryFile()",  # binary by default
        "os.fdopen(fd, 'w')",
        "io.TextIOWrapper(buffer, encoding='utf-8')",
    ])
    assert [what for _, what in unpinned(source)] == ["write_text", "open('w')", "open('a')", "open('w')",
                                                      "open('wt')", "NamedTemporaryFile('w')",
                                                      "NamedTemporaryFile('w+')", "fdopen('w')", "TextIOWrapper"]


def test_every_tracked_module_pins_the_line_ending_of_what_it_writes():
    offenders = [f"{path.relative_to(ROOT).as_posix()}:{line} {what}"
                 for path in _tracked_modules() for line, what in unpinned(path.read_text(encoding="utf-8"))]
    assert not offenders, f"{len(offenders)} text writes without newline=:\n" + "\n".join(offenders[:60])
