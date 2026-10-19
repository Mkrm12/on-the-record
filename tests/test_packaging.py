"""A package missing from the wheel's list ships a wheel without that code."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _listed() -> set[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {Path(p).name for p in config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]}


def _imports(package: str) -> set[str]:
    found = set()
    for path in (SRC / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            found |= {n.split(".")[0] for n in names}
    return {n for n in found if (SRC / n / "__init__.py").is_file()}


def test_every_src_package_a_listed_package_imports_is_listed():
    listed = _listed()
    missing = {}
    seen, todo = set(), sorted(listed)
    while todo:
        package = todo.pop()
        if package in seen:
            continue
        seen.add(package)
        for needed in _imports(package):
            if needed not in listed:
                missing.setdefault(needed, set()).add(package)
            todo.append(needed)
    assert not missing, f"imported by a packaged module but not in the wheel: {missing}"
