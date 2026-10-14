"""The mutation engine itself: a checker that silently checks nothing is worse than no checker."""

import textwrap
from pathlib import Path

import pytest

from mutation_engine import Mutant, MutationError, Project, apply_mutation, baseline_passes, run_mutant


def test_apply_replaces_only_the_first_occurrence():
    assert apply_mutation("a+b; a+b", "a+b", "a-b") == "a-b; a+b"


def test_apply_rejects_a_pattern_that_is_not_there():
    with pytest.raises(MutationError, match="not found"):
        apply_mutation("return a + b", "a - b", "a * b")


def test_apply_rejects_a_no_op_mutation():
    with pytest.raises(MutationError, match="identical"):
        apply_mutation("x", "x", "x")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A tiny repo-shaped project: src/calc.py, tests/test_calc.py, pyproject with the same pytest layout."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\npythonpath = ["src"]\n')
    (tmp_path / "src" / "calc.py").write_text(
        textwrap.dedent(
            """\
            def add(a, b):
                return a + b


            def sign(n):
                if n > 0:
                    return 1
                return -1 if n < 0 else 0
            """
        )
    )
    (tmp_path / "tests" / "test_calc.py").write_text(
        textwrap.dedent(
            """\
            from calc import add, sign


            def test_add():
                assert add(2, 3) == 5


            def test_sign_positive_and_negative():
                assert sign(4) == 1 and sign(-4) == -1
            """
        )
    )
    return tmp_path


TESTS = ["tests/test_calc.py"]


def test_baseline_passes_on_the_unmutated_project(project):
    assert baseline_passes(TESTS, root=project)


def test_baseline_fails_when_the_tests_fail(project):
    (project / "tests" / "test_calc.py").write_text("def test_broken():\n    assert False\n")
    assert not baseline_passes(TESTS, root=project)


def test_a_mutant_the_tests_catch_is_killed(project):
    outcome = run_mutant(Mutant("src/calc.py", "add becomes subtract", "return a + b", "return a - b"), TESTS, root=project)
    assert outcome.status == "killed"


def test_a_mutant_the_tests_miss_survives(project):
    # sign(0) is never tested, so flipping the zero branch is not noticed.
    outcome = run_mutant(Mutant("src/calc.py", "zero branch", "else 0", "else 7"), TESTS, root=project)
    assert outcome.status == "survived"


def test_a_documented_equivalent_survivor_is_not_a_failure(project):
    mutant = Mutant("src/calc.py", "zero branch", "else 0", "else 7", equivalent="demo: pretend it cannot be observed")
    assert run_mutant(mutant, TESTS, root=project).status == "equivalent"


def test_a_mutant_that_breaks_the_import_is_invalid_not_killed(project):
    outcome = run_mutant(Mutant("src/calc.py", "syntax error", "return a + b", "return a +"), TESTS, root=project)
    assert outcome.status == "invalid"


def test_a_stale_pattern_is_invalid_not_skipped(project):
    outcome = run_mutant(Mutant("src/calc.py", "gone", "this text is not in the file", "x"), TESTS, root=project)
    assert outcome.status == "invalid" and "not found" in outcome.detail


def test_an_unknown_target_file_is_invalid(project):
    assert run_mutant(Mutant("src/missing.py", "x", "a", "b"), TESTS, root=project).status == "invalid"


def test_an_infinite_loop_is_killed_by_the_timeout(project):
    mutant = Mutant("src/calc.py", "hang", "return a + b", "while True:\n        pass")
    outcome = run_mutant(mutant, TESTS, root=project, timeout=5)
    assert outcome.status == "killed" and "timeout" in outcome.detail


def test_the_working_tree_is_never_modified(project):
    before = (project / "src" / "calc.py").read_text()
    run_mutant(Mutant("src/calc.py", "add becomes subtract", "return a + b", "return a - b"), TESTS, root=project)
    assert (project / "src" / "calc.py").read_text() == before


# --------------------------------------------------------------------------- a project that is not laid out as src/ +
# tests/ (the gpu/ project) ---------------------------------------------------------------------------


@pytest.fixture
def nested(tmp_path: Path) -> tuple[Path, Project]:
    """sub/ holds the code, its tests and its pytest config; extra/ is a sibling directory the tests read."""
    (tmp_path / "sub" / "tests").mkdir(parents=True)
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "data.txt").write_text("42")
    (tmp_path / "sub" / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n')
    (tmp_path / "sub" / "calc.py").write_text("def answer():\n    return 42\n\n\ndef other():\n    return 1\n")
    (tmp_path / "sub" / "tests" / "test_calc.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path

            from calc import answer, other


            def test_answer_matches_the_sibling_data_file():
                assert answer() == int((Path(__file__).resolve().parents[2] / "extra" / "data.txt").read_text())


            def test_other():
                assert other() == 1
            """
        )
    )
    return tmp_path, Project(dirs=("sub", "extra"), cwd="sub")


def test_a_custom_layout_is_copied_and_run_from_its_own_directory(nested):
    root, project = nested
    assert baseline_passes(["tests/test_calc.py"], root=root, project=project)
    outcome = run_mutant(Mutant("sub/calc.py", "answer changes", "return 42", "return 43"), ["tests/test_calc.py"], root=root, project=project)
    assert outcome.status == "killed"


def test_pytest_arguments_from_the_project_are_passed_through(nested):
    root, project = nested
    only_other = Project(dirs=project.dirs, cwd=project.cwd, pytest_args=("-k", "test_other"))
    # the mutant breaks answer, which the deselected test would have caught
    outcome = run_mutant(Mutant("sub/calc.py", "answer changes", "return 42", "return 43"), ["tests/test_calc.py"], root=root, project=only_other)
    assert outcome.status == "survived"


def test_virtualenvs_and_caches_are_never_copied(tmp_path):
    """gpu/.venv holds several GB of torch: copying it for every mutant made a run take minutes."""
    (tmp_path / "sub" / "tests").mkdir(parents=True)
    for junk in (".venv", ".pytest_cache", ".git", "__pycache__"):
        (tmp_path / "sub" / junk).mkdir()
        (tmp_path / "sub" / junk / "big.bin").write_text("x")
    (tmp_path / "sub" / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n')
    (tmp_path / "sub" / "tests" / "test_layout.py").write_text(
        textwrap.dedent(
            """\
            from pathlib import Path


            def test_only_the_code_was_copied():
                sub = Path(__file__).resolve().parents[1]
                assert not any((sub / junk).exists() for junk in (".venv", ".pytest_cache", ".git"))
                assert (sub / "pyproject.toml").exists()
            """
        )
    )
    assert baseline_passes(["tests/test_layout.py"], root=tmp_path, project=Project(dirs=("sub",), cwd="sub"))


def test_the_interpreter_can_be_chosen_per_project(nested):
    root, project = nested
    broken = Project(dirs=project.dirs, cwd=project.cwd, python="this-interpreter-does-not-exist")
    assert not baseline_passes(["tests/test_calc.py"], root=root, project=broken)
