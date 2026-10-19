"""Date-window handling shared by the speech ingest scripts (fill_gaps, forensic_fill)."""

from datetime import date

import pytest

from ingest.window import days_newest_first, parse_window

START, END = date(2026, 8, 10), date(2026, 9, 19)


def test_defaults_when_no_arguments():
    assert parse_window([], START, END) == (START, END)


def test_start_and_end_override_the_defaults():
    assert parse_window(["--start", "2026-09-01", "--end", "2026-09-17"], START, END) == (date(2026, 9, 1), date(2026, 9, 17))


def test_one_bound_alone_keeps_the_other_default():
    assert parse_window(["--start", "2026-09-01"], START, END) == (date(2026, 9, 1), END)
    assert parse_window(["--end", "2026-09-01"], START, END) == (START, date(2026, 9, 1))


def test_a_single_day_window_is_allowed():
    assert parse_window(["--start", "2026-09-01", "--end", "2026-09-01"], START, END) == (date(2026, 9, 1), date(2026, 9, 1))


@pytest.mark.parametrize(
    "argv",
    [
        ["--start", "2026-09-18", "--end", "2026-09-17"],  # reversed
        ["--start", "2026-09-30"],  # start after the default end
        ["--end", "2026-08-01"],  # end before the default start
        ["--start", "17/09/2026"],  # not ISO
        ["--start", "2026-13-01"],  # not a date
        ["--start"],  # missing value
    ],
)
def test_bad_windows_are_rejected_before_anything_runs(argv):
    with pytest.raises(SystemExit) as exc:
        parse_window(argv, START, END)
    assert exc.value.code == 2


def test_days_are_newest_first_and_inclusive_at_both_ends():
    days = days_newest_first(date(2026, 9, 1), date(2026, 9, 4))
    assert days == [date(2026, 9, 4), date(2026, 9, 3), date(2026, 9, 2), date(2026, 9, 1)]


def test_one_day_gives_one_entry():
    assert days_newest_first(date(2026, 9, 1), date(2026, 9, 1)) == [date(2026, 9, 1)]


def test_day_count_matches_the_calendar_across_a_month_end():
    days = days_newest_first(START, END)
    assert len(days) == (END - START).days + 1 == 41
    assert days[0] == END and days[-1] == START
    assert len(set(days)) == len(days)


def test_reversed_window_raises_for_direct_callers():
    with pytest.raises(ValueError):
        days_newest_first(date(2026, 9, 2), date(2026, 9, 1))
