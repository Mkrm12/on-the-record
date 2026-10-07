"""Mutants for src/ingest/rules.py, checked against tests/test_rules.py."""

from mutation_engine import Mutant

PATH = "src/ingest/rules.py"
TESTS = ["tests/test_rules.py"]

_RAW = [
    ("urgent_question loses chair_only", '_r("urgent_question", "urgent_question", r"\\bto ask (?:an )?urgent question\\b", chair_only=True', '_r("urgent_question", "urgent_question", r"\\bto ask (?:an )?urgent question\\b", chair_only=False'),
    ("royal_assent loses chair_only", 'Royal Assent$", chair_only=True', 'Royal Assent$", chair_only=False'),
    ("resolved loses line anchor", '_r("resolved", "resolved", r"^Resolved', '_r("resolved", "resolved", r"Resolved'),
    ("ordered loses line anchor", '_r("ordered", "ordered", r"^Ordered', '_r("ordered", "ordered", r"Ordered'),
    ("boilerplate all() -> any()", "return reasons[0] if all(reasons) else None", "return next((r for r in reasons if r), None)"),
    ("motion_stated not_within removed", 'not_within="beg_to_move"', "not_within=None"),
    ("closure loses end anchor", r"the Question be now put\.?[ \t]*$", r"the Question be now put"),
    ("beg_to_move loses line anchor", 'rf"^{_MOVE}(?:,?', 'rf"{_MOVE}(?:,?'),
    ("point_of_order loses line anchor", '"point_of_order",\n       r"^(?:On|Further', '"point_of_order",\n       r"(?:On|Further'),
    ("chair_only ignored in formulas", "if (is_chair or not formula.chair_only) and", "if True and"),
    ("event beats special heading (order swapped)", "    for pattern, kind in _SPECIAL_HEADINGS:\n        if pattern.search(heading):\n            return kind\n    for event_type, kind in _EVENT_DEBATE_TYPES:\n        if event_type in events:\n            return kind\n",
                                                   "    for event_type, kind in _EVENT_DEBATE_TYPES:\n        if event_type in events:\n            return kind\n    for pattern, kind in _SPECIAL_HEADINGS:\n        if pattern.search(heading):\n            return kind\n"),
    ("neutral motion regex too broad", r"(?:has\s+considered|takes?\s+note|notes?\s+the\s+matter)\b", r"(?:has\s+considered|takes?\s+note|notes?\s+the\s+matter|regrets)\b"),
    ("party set loses Your Party", '"Your Party", ', ""),
    ("remote flag never set", 'remote = bool({"remote", "virtual"} & set(flags))', "remote = False"),
    ("question_put allows put-and-agreed", r"(?![ \t]+and[ \t]+(?:agreed|negatived))", ""),
    ("second_reading accepts clause", "rf\"^{_STAGE_PREFIX}{_BILL}[ \\t]+(?:(?:be|is)", "rf\"^{_STAGE_PREFIX}(?:Bill|clause)[ \\t]+(?:(?:be|is)"),
    ("declaration drops 'my' requirement", r"[^.\n]{0,100}?\bmy\b[^.\n]{0,60}?\binterests?\b|", r"[^.\n]{0,100}?\binterests?\b|"),
    ("give_way any speaker text", r"(?:will|would|might|may|could|can)\s+(?:the|my|I|he|she)\b", r"(?:will|would|might|may|could|can)\s+"),

    # --- a new Parliament opening, and pandemic proceedings -----
    ("oaths are not recognised at all",
     '_r("oath_taken", "oath_or_affirmation",', '_r("oath_taken", "never_fires",'),
    ("the speaker election is not recognised",
     '_r("speaker_elected", "speaker_elected",', '_r("speaker_elected", "never_fires",'),
    # Each of the four below removes an anchor.
    ("the speaker heading is unanchored, so 'Senior Deputy Speaker' is a Speaker election",
     'r"^Election of (?:a |the )?Speaker|^Speaker[- ]Elect"', 'r"Speaker"'),
    ("the address heading is unanchored, so 'Registered Address' becomes an address debate",
     "|^Address in Reply", "|Address"),
    ("a Bill whose title mentions dissolution is classified as a prorogation",
     'r"^Prorogation$|^Dissolution$|^State Opening"', 'r"Prorogation|Dissolution|State Opening"'),
    ("a remote division heading is unanchored, so 'Remote Access' becomes a division",
     'r"^Remote Division"', 'r"Remote"'),
]  # fmt: skip

MUTANTS = [Mutant(PATH, name, old, new) for name, old, new in _RAW]
