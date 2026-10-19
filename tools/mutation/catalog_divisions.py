"""Mutants for src/ingest/divisions.py, checked against tests/test_divisions.py."""

from mutation_engine import Mutant, Project

PATH = "src/ingest/divisions.py"
TESTS = ["tests/test_divisions.py"]

# The write-path tests build a scratch database from db/schema.sql plus the migrations, so the throw-away copy needs db/
# as well as src/ and tests/.
PROJECT = Project(dirs=("src", "tests", "db"))

_RAW = [
    ("conflict resolves to 'aye' (first-listed wins)", 'lobby = "conflict"', 'lobby = "aye"'),
    ("reconcile only excluding tellers", "official in (excluding, including)", "official == excluding"),
    ("reconcile only including tellers", "official in (excluding, including)", "official == including"),
    ("teller flag dropped", '"is_teller": any(t for _, _, t, _ in items)', '"is_teller": False'),
    ("Lords id not offset", "LORDS_ID_OFFSET if house == \"Lords\" else 0", "0"),
    ("tie counted as agreed", '"agreed" if ayes > noes else', '"agreed" if ayes >= noes else'),
    ("prefix match has no minimum length", "len(shorter) >= MIN_PREFIX_LENGTH", "len(shorter) >= 0"),
    ("tally ignores the date", "(t.house, t.date, t.ayes, t.noes) == (house, day,", "(t.house, t.ayes, t.noes) == (house,"),
    ("deferred window unbounded", "timedelta(0) < day - d.date <= DEFERRED_WINDOW", "timedelta(0) < day - d.date"),
    ("deferred divisions use the time window", "if not deferred and held_at and held_at.time()", "if held_at and held_at.time()"),
    ("distance uses later speeches", "before = [t for t in times if t <= at]", "before = [t for t in times if t >= at]"),
    ("link ignores house", "same_house = [d for d in debates if d.house == house]", "same_house = list(debates)"),
    ("also_no_vote not flagged", 'flags.append("also_no_vote")', "pass"),
    ("pagination stops one page early", "if len(batch) < page_size:", "if len(batch) <= page_size:"),
    ("missing member id kept", 'counts["vote_without_member_id"] += 1\n                continue', 'counts["vote_without_member_id"] += 1'),
    ("stale votes never removed", "await conn.execute(SQL_DELETE_STALE_VOTES", "pass  # await conn.execute(SQL_DELETE_STALE_VOTES"),
    ("stale-vote delete inverted (removes the current voters)", "mnis_id <> ALL(%s::int[])", "mnis_id = ANY(%s::int[])"),
    ("empty voter list wipes the stored votes", "        if parsed.votes:\n", "        if True:\n"),
    ("Lords motion text ignores notes", 'clean_text(detail.get("amendmentMotionNotes")) or clean_text(detail.get("notes")) or title', "title"),
    ("raw keeps member lists", "raw={k: v for k, v in detail.items() if k not in _COMMONS_LISTS}", "raw=dict(detail)"),

    # --- remote divisions ------
    ("a physical Commons division is marked remote",
     'division["is_remote"] = division["remote_start"] is not None',
     'division["is_remote"] = True'),
    ("no Commons division is ever marked remote",
     'division["is_remote"] = division["remote_start"] is not None',
     'division["is_remote"] = False'),
    ("the Lords timestamp is read as the Commons signal, marking every Lords division remote",
     'division["is_remote"] = bool(detail.get("divisionWasExclusivelyRemote"))',
     'division["is_remote"] = detail.get("remoteVotingStart") is not None'),
    ("an unparseable remote timestamp raises instead of being absence",
     "    try:\n        return datetime.fromisoformat(value) if value else None\n"
     "    except (TypeError, ValueError):\n        return None",
     "    return datetime.fromisoformat(value) if value else None"),
]  # fmt: skip

MUTANTS = [Mutant(PATH, name, old, new) for name, old, new in _RAW]
