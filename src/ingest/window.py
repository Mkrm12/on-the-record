"""Shared --start / --end handling for scripts that walk a range of sitting days."""

import argparse
from datetime import date, timedelta


def parse_window(argv: list[str], default_start: date, default_end: date, description: str | None = None) -> tuple[date, date]:
    """(start, end) from the command line, both inclusive. Exits with status 2 on a bad or reversed window."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start", type=date.fromisoformat, default=default_start, help="first day, YYYY-MM-DD (default: %(default)s)")
    parser.add_argument("--end", type=date.fromisoformat, default=default_end, help="last day, YYYY-MM-DD (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error(f"--start {args.start} is after --end {args.end}")
    return args.start, args.end


def days_newest_first(start: date, end: date) -> list[date]:
    """Every calendar day from end back to start, inclusive."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    return [end - timedelta(days=i) for i in range((end - start).days + 1)]
