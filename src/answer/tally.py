"""Whether a division's name lists match its tellers' figures, said once for every route."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reconciled(row: Mapping[str, Any]) -> bool | None:
    """The stored flag: True, False, or None when a figure or the lists are not recorded."""
    return row.get("tally_reconciled")


def caveat(row: Mapping[str, Any]) -> str:
    """The note a tally may not be quoted without, or "" when the lists match the figures."""
    ayes, noes = row.get("ayes"), row.get("noes")
    if ayes is None or noes is None:
        return "The tellers' figures are not recorded for this division."
    flag = reconciled(row)
    if flag is True:
        return ""
    if flag is None:
        # Unknown is not absent: say the lists are missing only when they are.
        if row.get("ayes_listed") is None or row.get("noes_listed") is None:
            return "No name lists are published for this division, so only the tellers' figures can be cited (I-8)."
        return ""
    named_ayes, named_noes = row.get("ayes_named"), row.get("noes_named")
    if named_ayes is None or named_noes is None:
        return (f"The published name lists do not reconcile with the tellers' figures ({ayes} / {noes}); the "
                f"tellers' figures are the result (I-8).")
    if row.get("house") == "Lords":
        names = f"{named_ayes} Contents and {named_noes} Not-Contents, tellers included"
    else:
        names = f"{named_ayes} Ayes and {named_noes} Noes, tellers apart"
    return (f"The published name lists do not reconcile with the tellers' figures: they name {names}, against the "
            f"tellers' {ayes} and {noes}. The tellers' figures are the result (I-8).")
