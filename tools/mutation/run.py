"""Run the mutation suites. Nothing in src/ or tests/ is modified: every mutant runs in a temporary copy."""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import catalog_answer_text
import catalog_core
import catalog_corpus_fetch
import catalog_divisions
import catalog_gold
import catalog_goldset
import catalog_gpu
import catalog_load
import catalog_retrieval
import catalog_rules
import catalog_service
from mutation_engine import DEFAULT_PROJECT, baseline_passes, run_mutant

# name -> (mutants, tests to run, project layout)
SUITES = {
    "rules": (catalog_rules.MUTANTS, catalog_rules.TESTS, DEFAULT_PROJECT),
    "divisions": (catalog_divisions.MUTANTS, catalog_divisions.TESTS, catalog_divisions.PROJECT),
    "core": (catalog_core.MUTANTS, catalog_core.TESTS, catalog_core.PROJECT),
    "corpus_fetch": (catalog_corpus_fetch.MUTANTS, catalog_corpus_fetch.TESTS, catalog_corpus_fetch.PROJECT),
    "gold": (catalog_gold.MUTANTS, catalog_gold.TESTS, DEFAULT_PROJECT),
    "goldset": (catalog_goldset.MUTANTS, catalog_goldset.TESTS, catalog_goldset.PROJECT),
    "load": (catalog_load.MUTANTS, catalog_load.TESTS, catalog_load.PROJECT),
    "gpu": (catalog_gpu.MUTANTS, catalog_gpu.TESTS, catalog_gpu.PROJECT),
    "service": (catalog_service.MUTANTS, catalog_service.TESTS, catalog_service.PROJECT),
    "retrieval": (catalog_retrieval.MUTANTS, catalog_retrieval.TESTS, DEFAULT_PROJECT),
    "search": (catalog_retrieval.SEARCH_MUTANTS, catalog_retrieval.SEARCH_TESTS, catalog_retrieval.SEARCH_PROJECT),
    "index": (catalog_retrieval.INDEX_MUTANTS, catalog_retrieval.INDEX_TESTS, catalog_retrieval.INDEX_PROJECT),
    "reader": (catalog_answer_text.MUTANTS, catalog_answer_text.TESTS, catalog_answer_text.PROJECT),
}
GPU_ENV_SUITES = {"gpu", "service"}  # these run in gpu/.venv, so they need `uv sync` in gpu/


def run_suite(name: str, workers: int) -> tuple[int, int, list[str]]:
    """(killed or equivalent, total, problems)."""
    mutants, tests, project = SUITES[name]
    print(f"\n== {name}: {len(mutants)} mutants against {', '.join(tests)}", flush=True)
    if not baseline_passes(tests, project=project):
        return 0, len(mutants), [f"{name}: the unmutated baseline fails, so no result would mean anything"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda m: run_mutant(m, tests, project=project), mutants))
    good, problems = 0, []
    for mutant, outcome in zip(mutants, outcomes):
        label = {"killed": "KILLED    ", "equivalent": "EQUIVALENT", "survived": "SURVIVED  ", "invalid": "INVALID   "}[outcome.status]
        print(f"{label} {mutant.name}" + (f"  [{outcome.detail}]" if outcome.status in ("equivalent", "invalid") else ""))
        if outcome.status in ("killed", "equivalent"):
            good += 1
        else:
            problems.append(f"{name}: {mutant.name} ({outcome.status})")
    return good, len(mutants), problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    names = list(SUITES) if args.suite == "all" else [args.suite]
    for name in [n for n in names if n in GPU_ENV_SUITES and not catalog_gpu.available()]:
        if args.suite == name:
            print(f"The {name} suite needs the gpu/ environment: run `uv sync` in gpu/ first.")
            return 1
        print(f"Skipping the {name} suite: the gpu/ environment is not installed (run `uv sync` in gpu/).")
        names.remove(name)
    total_good = total = 0
    problems: list[str] = []
    for name in names:
        good, count, bad = run_suite(name, args.workers)
        total_good, total = total_good + good, total + count
        problems += bad
    print(f"\n{total_good}/{total} mutants killed or documented equivalent")
    for line in problems:
        print("PROBLEM:", line)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
