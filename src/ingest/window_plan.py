"""The ten-year window plan: the gaps around the existing windows, cut into calendar quarters."""

from __future__ import annotations

from datetime import date, timedelta

from ingest.corpus_fetch import WindowSpec

ROLE = "corpus"


def _quarter(day: date) -> tuple[int, int]:
    return day.year, (day.month - 1) // 3 + 1


def _quarter_end(day: date) -> date:
    year, quarter = _quarter(day)
    last_month = quarter * 3
    first_of_next = date(year + (last_month == 12), (last_month % 12) + 1, 1)
    return first_of_next - timedelta(days=1)


def gap_windows(existing: list[WindowSpec] | tuple[WindowSpec, ...], span_start: date, span_end: date,
                *, note: str = "") -> list[WindowSpec]:  # fmt: skip
    """The `corpus` windows that, with the existing ACTIVE windows, cover [span_start, span_end] exactly."""
    if span_start > span_end:
        raise ValueError(f"span start {span_start} is after span end {span_end}")
    covered = sorted((w.start, w.end) for w in existing if w.active)

    # The uncovered runs of days inside the span, in order.
    runs: list[tuple[date, date]] = []
    cursor = span_start
    for start, end in covered:
        if end < cursor or start > span_end:
            continue
        if start > cursor:
            runs.append((cursor, min(start - timedelta(days=1), span_end)))
        cursor = max(cursor, end + timedelta(days=1))
        if cursor > span_end:
            break
    if cursor <= span_end:
        runs.append((cursor, span_end))

    # Each run, cut at quarter boundaries.
    windows: list[WindowSpec] = []
    used: dict[str, int] = {}
    for run_start, run_end in runs:
        piece_start = run_start
        while piece_start <= run_end:
            piece_end = min(_quarter_end(piece_start), run_end)
            year, quarter = _quarter(piece_start)
            base = f"C{year}Q{quarter}"
            count = used.get(base, 0)
            used[base] = count + 1
            wid = base if count == 0 else f"{base}{chr(ord('a') + count)}"
            windows.append(WindowSpec(id=wid, start=piece_start, end=piece_end, role=ROLE, active=True,
                                      note=note or "Ten-year corpus extension: loaded and searchable, never evaluated."))
            piece_start = piece_end + timedelta(days=1)
    return windows


def to_toml(windows: list[WindowSpec], *, extends: str, header: str = "") -> str:
    """A plan file for `load_config`: `extends` the base config and adds these windows, nothing else."""
    lines = []
    if header:
        lines += [f"# {line}" if line else "#" for line in header.splitlines()] + [""]
    lines += [f'extends = "{extends}"', ""]
    for w in windows:
        lines += ["[[windows]]", f'id = "{w.id}"', f"start = {w.start.isoformat()}", f"end = {w.end.isoformat()}",
                  f'role = "{w.role}"', f"active = {'true' if w.active else 'false'}",
                  f'note = "{w.note}"', ""]
    return "\n".join(lines)
