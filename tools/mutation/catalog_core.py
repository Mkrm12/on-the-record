"""Mutants for the Phase 1 completion code (text, chunk, wikidata, bills, hansard_meta)."""

from mutation_engine import Mutant, Project

TESTS = [
    "tests/test_text.py", "tests/test_chunk.py", "tests/test_chunk_context.py",
    "tests/test_wikidata.py", "tests/test_bills.py", "tests/test_hansard_meta.py",
]  # fmt: skip

# test_chunk.py proves that each debate COMMITS as it is chunked, which needs a scratch database built from
# db/schema.sql plus the migrations, so the throw-away copy needs db/ as well as src/ and tests/.
PROJECT = Project(dirs=("src", "tests", "db"))

_RAW = [
    # --- text.py
    ("src/ingest/text.py", "seam: drop 'space before newline' rule", 'elif left == " " and right == "\\n":\n                out = out[:-1]', 'elif False:\n                out = out[:-1]'),
    ("src/ingest/text.py", "seam: drop 'doubled space' rule", 'if left == " " and right == " ":\n                cursor += 1', 'if False:\n                cursor += 1'),
    ("src/ingest/text.py", "marker at sentence start not counted", "i = bisect_right(offsets, span.start) - 1", "i = bisect_left(offsets, span.start) - 1"),
    ("src/ingest/text.py", "previous_ref allows column 0", "int(m.group(1)) <= 1", "int(m.group(1)) <= 0"),
    ("src/ingest/text.py", "indented paragraph offset ignored", "base = offset + len(paragraph) - len(paragraph.lstrip())", "base = offset"),
    ("src/ingest/text.py", "merged sentence loses first start", "spans[-1] = Span(spans[-1].text + \" \" + sentence, spans[-1].start, end)", "spans[-1] = Span(spans[-1].text + \" \" + sentence, start, end)"),
    ("src/ingest/text.py", "marker without value kept", 'return f"{_OPEN}{value.group(1)}{_CLOSE}" if value else ""', 'return f"{_OPEN}{value.group(1) if value else \'\'}{_CLOSE}"'),
    ("src/ingest/text.py", "other spans treated as markers", '\\bclass="[^"]*\\bcolumn-number\\b[^"]*"', '\\bclass="[^"]*"'),
    # --- chunk.py
    ("src/ingest/chunk.py", "boilerplate child not flagged", "chunks.append(ChunkDraft(i, sentence, sentence, True))", "chunks.append(ChunkDraft(i, sentence, sentence, False))"),
    ("src/ingest/chunk.py", "windows ignore the mask", "lo, hi = window_bounds(position[i], len(kept), window)\n        chunks.append(ChunkDraft(i, sentence, \" \".join(sentences[k] for k in kept[lo:hi])))", "lo, hi = window_bounds(i, total, window)\n        chunks.append(ChunkDraft(i, sentence, \" \".join(sentences[lo:hi])))"),
    ("src/ingest/chunk.py", "column carry never advances", 'return drafts, ((prepared.markers[-1].ref, "carried") if prepared.markers else carry)', "return drafts, carry"),
    ("src/ingest/chunk.py", "no inference before first marker", 'carry: Carry = (inferred, "inferred") if inferred else None', "carry: Carry = None"),
    ("src/ingest/chunk.py", "end column dropped", "column_end_ref=c.end_ref", "column_end_ref=None"),
    ("src/ingest/chunk.py", "chair-only formulas lose the speaker", "sentence_boilerplate_reason(s.text, speaker_name)", "sentence_boilerplate_reason(s.text, None)"),
    ("src/ingest/chunk.py", "mask length not validated", "elif len(boilerplate) != total:", "elif False:"),
    # --- wikidata.py
    ("src/ingest/wikidata.py", "preferred rank ignored", 'pool = [c for c in usable if c.rank == "PreferredRank"] or usable', "pool = usable"),
    ("src/ingest/wikidata.py", "deprecated statements used", 'if c.rank == "DeprecatedRank":\n            continue', 'if False:\n            continue'),
    ("src/ingest/wikidata.py", "year-only becomes a birthday", 'Dob(value if precision == "day" else None,', "Dob(value,"),
    ("src/ingest/wikidata.py", "conflicts are guessed", 'if len(top) > 1:\n        return _NONE._replace(note="conflicting_dob")', 'if False:\n        return _NONE._replace(note="conflicting_dob")'),
    ("src/ingest/wikidata.py", "no upper age bound", " or latest < _years_before(today, MAX_AGE)", ""),
    ("src/ingest/wikidata.py", "no lower age bound", " or earliest > _years_before(today, MIN_AGE)", ""),
    ("src/ingest/wikidata.py", "month 00 not repaired", "int(m.group(3)) or 1", "int(m.group(3))"),
    ("src/ingest/wikidata.py", "person_id keeps stale value", "person_id = u.person_id,", "person_id = coalesce(m.person_id, u.person_id),"),
    ("src/ingest/wikidata.py", "duplicate twfy not withheld", 'rows[mnis]["person_id"] = None', "pass"),
    ("src/ingest/wikidata.py", "precision 13 not day", 'precision = "day" if code >= 11 else', 'precision = "day" if code == 11 else'),
    # --- bills.py
    ("src/ingest/bills.py", "weak label removed", 'return via_debate[0].stage_id, "debate_weak" if weak else "debate"', 'return via_debate[0].stage_id, "debate"'),
    ("src/ingest/bills.py", "ambiguity guessed", "if len(hits) == 1:\n            return hits[0].stage_id, method", "if len(hits) >= 1:\n            return hits[0].stage_id, method"),
    ("src/ingest/bills.py", "deferred divisions linked", 'if division.get("is_deferred"):', "if False:"),
    ("src/ingest/bills.py", "house ignored", "on_the_day = [s for s in stages if s.house == house and day in s.dates]", "on_the_day = [s for s in stages if day in s.dates]"),
    ("src/ingest/bills.py", "stage keyword ignored", "hits = [s for s in hits if s.abbreviation in keywords]", "hits = hits"),
    ("src/ingest/bills.py", "date ignored", "if s.house == house and day in s.dates]", "if s.house == house]"),
    ("src/ingest/bills.py", "sitting link ignores day", "if d.date == day and (house is None or d.house == house)", "if (house is None or d.house == house)"),
    ("src/ingest/bills.py", "unassigned house kept", 'house = None if st.get("house") in (None, "Unassigned") else st["house"]', 'house = st.get("house")'),
    ("src/ingest/bills.py", "window edge exclusive", "start <= s[\"date\"] <= end", "start < s[\"date\"] < end"),
    # --- hansard_meta.py
    ("src/ingest/hansard_meta.py", "long texts kept in meta", "if k not in _TEXT_KEYS", "if True"),
    ("src/ingest/hansard_meta.py", "fetch_day ignores the reported total", "if not batch or skip >= ((payload or {}).get(\"TotalResultCount\") or 0):", "if not batch:"),
]  # fmt: skip

MUTANTS = [Mutant(*row) for row in _RAW]
