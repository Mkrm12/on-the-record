"""Mutants for the gold-set validator (src/app/gold.py) and the shared date window (src/ingest/window.py)."""

from mutation_engine import Mutant

TESTS = ["tests/test_gold_eval.py", "tests/test_window.py"]

_GOLD = "src/app/gold.py"
_WINDOW = "src/ingest/window.py"

_RAW = [
    (_GOLD, "paraphrase overlap limit removed", "> MAX_PARAPHRASE_OVERLAP", "> 1"),
    (_GOLD, "leak check applies to every category", 'if category == "paraphrase" and isinstance(question, str):', "if isinstance(question, str):"),
    (_GOLD, "duplicate ids allowed", "elif qid in seen_ids:", "elif False:"),
    (_GOLD, "duplicate questions allowed", "if key in seen_questions:", "if False:"),
    (_GOLD, "duplicate question check is case sensitive", 'key = " ".join(question.lower().split())', 'key = " ".join(question.split())'),
    (_GOLD, "abstain flag not cross-checked with the category", 'elif abstain != (category == "abstain"):', "elif False:"),
    (_GOLD, "abstain query may name relevant speeches", "if abstain is True and relevant:", "if False:"),
    (_GOLD, "answerable query may have no relevant speech", "if abstain is False and not relevant:", "if False:"),
    (_GOLD, "short evidence quotes allowed", "len(quote.strip()) < MIN_QUOTE_LENGTH", "len(quote.strip()) < 0"),
    (_GOLD, "a category may have a single query", "if counts[name] < MIN_PER_CATEGORY:", "if counts[name] < 1:"),
    (_GOLD, "minimum size off by one", "if len(queries) < MIN_QUERIES:", "if len(queries) <= MIN_QUERIES:"),
    (_GOLD, "'ed' not stemmed", '("ing", "ed", "es", "s")', '("ing", "es", "s")'),
    (_GOLD, "load ignores the status filter", 'if q["status"] in statuses', "if True"),
    (_GOLD, "load skips validation", "if errors := validate(data):", "if False:"),
    (_WINDOW, "reversed window accepted", "if args.start > args.end:", "if False:"),
    (_WINDOW, "single-day window rejected", "if args.start > args.end:", "if args.start >= args.end:"),
    (_WINDOW, "day list excludes the start day", "range((end - start).days + 1)", "range((end - start).days)"),
    (_WINDOW, "day list runs oldest first", "[end - timedelta(days=i)", "[start + timedelta(days=i)"),
    (_WINDOW, "days_newest_first accepts a reversed window", "if start > end:\n        raise ValueError", "if False:\n        raise ValueError"),
]  # fmt: skip

MUTANTS = [Mutant(*row) for row in _RAW]
