"""Data invariants of hansard_v2 — what must be true of the rows, whatever the schema declares."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psycopg  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from app import safety, sealed  # noqa: E402
from app.config import settings  # noqa: E402
from eval.oracles import openers as OO  # noqa: E402
from procedure import eras as ERAS  # noqa: E402
from procedure import opener as OP  # noqa: E402
from procedure import threads as T  # noqa: E402
from retrieval import vector_index as V  # noqa: E402

# I01: the live HNSW index must be the form frozen retrieval writes its ORDER BY for.
_FROZEN_FORM = V.frozen_form()
_INDEX_LIKE = V.INDEXDEF_LIKE[_FROZEN_FORM]

CHECKS: list[tuple[str, str, str]] = [
    # ---- R: referential ------------------------------------------------------------------------------
    ("R01", "speech.mnis_id resolves to member_id_map",
     "SELECT s.id, s.mnis_id, s.speaker_name FROM speech s WHERE s.mnis_id IS NOT NULL AND NOT EXISTS "
     "(SELECT 1 FROM member_id_map m WHERE m.mnis_id=s.mnis_id)"),
    ("R02", "division_vote.mnis_id resolves to member_id_map",
     "SELECT v.division_id, v.mnis_id, v.member_name FROM division_vote v WHERE NOT EXISTS "
     "(SELECT 1 FROM member_id_map m WHERE m.mnis_id=v.mnis_id)"),
    ("R03", "chunk.debate_id resolves to debate",
     "SELECT k.id, k.debate_id FROM chunk k WHERE NOT EXISTS (SELECT 1 FROM debate d WHERE d.id=k.debate_id)"),
    ("R04", "chunk.mnis_id resolves to member_id_map",
     "SELECT k.id, k.mnis_id FROM chunk k WHERE k.mnis_id IS NOT NULL AND NOT EXISTS "
     "(SELECT 1 FROM member_id_map m WHERE m.mnis_id=k.mnis_id)"),
    ("R05", "division.debate_id resolves to debate (when set)",
     "SELECT v.id, v.debate_id FROM division v WHERE v.debate_id IS NOT NULL AND NOT EXISTS "
     "(SELECT 1 FROM debate d WHERE d.id=v.debate_id)"),
    ("R06", "procedure_event.debate_id resolves to debate",
     "SELECT p.id, p.debate_id FROM procedure_event p WHERE NOT EXISTS (SELECT 1 FROM debate d WHERE d.id=p.debate_id)"),
    # R07 used to check `hansard_correction.debate_id` against `debate` and reported all 304 as violations.
    ("R07", "correction with no link decision at all (neither linked nor labelled why not)",
     "SELECT h.id FROM hansard_correction h WHERE h.corrected_speech_id IS NULL AND h.link_basis IS NULL"),
    ("R08", "hansard_correction.mnis_id resolves to member_id_map",
     "SELECT h.id, h.mnis_id FROM hansard_correction h WHERE h.mnis_id IS NOT NULL AND NOT EXISTS "
     "(SELECT 1 FROM member_id_map m WHERE m.mnis_id=h.mnis_id)"),
    ("R09", "bill_stage_sitting.debate_id resolves to debate (when set)",
     "SELECT b.id, b.debate_id FROM bill_stage_sitting b WHERE b.debate_id IS NOT NULL AND NOT EXISTS "
     "(SELECT 1 FROM debate d WHERE d.id=b.debate_id)"),
    ("R10", "speech.window_id resolves to corpus_window",
     "SELECT s.id, s.window_id FROM speech s WHERE NOT EXISTS (SELECT 1 FROM corpus_window w WHERE w.window_id=s.window_id)"),
    ("R11", "chunk.window_id resolves to corpus_window",
     "SELECT k.id, k.window_id FROM chunk k WHERE NOT EXISTS (SELECT 1 FROM corpus_window w WHERE w.window_id=k.window_id)"),
    ("R12", "member_office.mnis_id resolves (declared FK; sanity)",
     "SELECT o.id, o.mnis_id FROM member_office o WHERE NOT EXISTS (SELECT 1 FROM member_id_map m WHERE m.mnis_id=o.mnis_id)"),
    ("R13", "speech.mnis_id is NULL (speaker unresolved)",
     "SELECT s.id, s.speaker_name, s.speaker_role FROM speech s WHERE s.mnis_id IS NULL"),
    # A closed vocabulary, both ways.
    ("R15", "division.bill_link outside its vocabulary, or disagreeing with bill_stage_id",
     "SELECT id, bill_link, bill_stage_id FROM division WHERE bill_link IS NULL "
     "OR NOT (bill_link ~ '^(title_(stage|date|amendment)(\\+(numbered|fuzzy))?|debate(_weak)?)$' "
     "        OR bill_link IN ('none', 'ambiguous', 'not_a_bill')) "
     "OR (bill_link !~ '^(none|ambiguous|not_a_bill)$' AND bill_stage_id IS NULL) "
     "OR (bill_link IN ('none', 'ambiguous', 'not_a_bill') AND bill_stage_id IS NOT NULL)"),
    ("R14", "division has no debate_id (unplaced vote)",
     "SELECT v.id, v.house, v.division_date, left(v.title,60) FROM division v WHERE v.debate_id IS NULL"),
    # ---- D: denormalisation ---------------------------------------------------------------------------
    ("D01", "chunk.debate_id = its speech's debate_id",
     "SELECT k.id, k.debate_id, s.debate_id FROM chunk k JOIN speech s ON s.id=k.speech_id WHERE k.debate_id <> s.debate_id"),
    ("D02", "chunk.debate_date = its debate's date",
     "SELECT k.id, k.debate_date, d.date FROM chunk k JOIN debate d ON d.id=k.debate_id WHERE k.debate_date IS DISTINCT FROM d.date"),
    ("D03", "chunk.house = its debate's house",
     "SELECT k.id, k.house, d.house FROM chunk k JOIN debate d ON d.id=k.debate_id WHERE k.house IS DISTINCT FROM d.house"),
    ("D04", "chunk.section = its debate's section",
     "SELECT k.id, k.section, d.section FROM chunk k JOIN debate d ON d.id=k.debate_id WHERE k.section IS DISTINCT FROM d.section"),
    ("D05", "chunk.window_id = its debate's window_id",
     "SELECT k.id, k.window_id, d.window_id FROM chunk k JOIN debate d ON d.id=k.debate_id WHERE k.window_id IS DISTINCT FROM d.window_id"),
    ("D06", "chunk.mnis_id = its speech's mnis_id",
     "SELECT k.id, k.mnis_id, s.mnis_id FROM chunk k JOIN speech s ON s.id=k.speech_id WHERE k.mnis_id IS DISTINCT FROM s.mnis_id"),
    ("D07", "speech.window_id = its debate's window_id",
     "SELECT s.id, s.window_id, d.window_id FROM speech s JOIN debate d ON d.id=s.debate_id WHERE s.window_id IS DISTINCT FROM d.window_id"),
    # D08 first compared chunk.is_boilerplate with speech.is_boilerplate and reported 8,989 "violations" — but a chunk
    # is one sentence and the flag is per sentence ("Order." inside a substantive speech).
    ("D08", "a whole-boilerplate speech with a non-boilerplate sentence",
     "SELECT k.id FROM chunk k JOIN speech s ON s.id=k.speech_id WHERE s.is_boilerplate AND NOT k.is_boilerplate"),
    ("D09", "division.ayes_listed = count of its 'aye' votes",
     "SELECT d.id, d.ayes_listed, (SELECT count(*) FROM division_vote v WHERE v.division_id=d.id AND v.lobby='aye') "
     "FROM division d WHERE d.ayes_listed IS DISTINCT FROM (SELECT count(*) FROM division_vote v WHERE v.division_id=d.id AND v.lobby='aye')"),
    ("D10", "division.noes_listed = count of its 'no' votes",
     "SELECT d.id, d.noes_listed, (SELECT count(*) FROM division_vote v WHERE v.division_id=d.id AND v.lobby='no') "
     "FROM division d WHERE d.noes_listed IS DISTINCT FROM (SELECT count(*) FROM division_vote v WHERE v.division_id=d.id AND v.lobby='no')"),
    ("D11", "division.no_vote_count = count of its 'no_vote' rows",
     "SELECT d.id, d.no_vote_count, (SELECT count(*) FROM division_vote v WHERE v.division_id=d.id AND v.lobby='no_vote') "
     "FROM division d WHERE d.no_vote_count IS DISTINCT FROM (SELECT count(*) FROM division_vote v WHERE v.division_id=d.id AND v.lobby='no_vote')"),
    ("D12", "division.house = its debate's house",
     "SELECT v.id, v.house, d.house FROM division v JOIN debate d ON d.id=v.debate_id WHERE v.house IS DISTINCT FROM d.house"),
    # By Parliament's procedure, not by the sample: a division is voted in the debate it belongs to, on that day —
    # EXCEPT a deferred division, which is voted on a later Wednesday for an instrument debated earlier in committee.
    ("D13", "a non-deferred division on a different day from its debate",
     "SELECT v.id, v.division_date, d.date FROM division v JOIN debate d ON d.id=v.debate_id "
     "WHERE NOT v.is_deferred AND v.division_date IS DISTINCT FROM d.date"),
    ("D13d", "a deferred division whose debate is not 1-60 days before the vote",
     "SELECT v.id, v.division_date, d.date FROM division v JOIN debate d ON d.id=v.debate_id "
     "WHERE v.is_deferred AND NOT (d.date < v.division_date AND v.division_date - d.date <= 60)"),
    ("D14", "hansard_correction.date = its debate's date",
     "SELECT h.id, h.date, d.date FROM hansard_correction h JOIN debate d ON d.id=h.debate_id WHERE h.date IS DISTINCT FROM d.date"),
    ("D15", "chunk.embedding_recipe set iff embedding set",
     "SELECT k.id FROM chunk k WHERE (k.embedding IS NULL) <> (k.embedding_recipe IS NULL)"),
    ("D16", "corpus_window.speeches = actual speeches in window",
     "SELECT w.window_id, w.speeches, (SELECT count(*) FROM speech s WHERE s.window_id=w.window_id) FROM corpus_window w "
     "WHERE w.speeches IS DISTINCT FROM (SELECT count(*) FROM speech s WHERE s.window_id=w.window_id)"),
    ("D17", "corpus_window.debates = actual debates in window",
     "SELECT w.window_id, w.debates, (SELECT count(*) FROM debate d WHERE d.window_id=w.window_id) FROM corpus_window w "
     "WHERE w.debates IS DISTINCT FROM (SELECT count(*) FROM debate d WHERE d.window_id=w.window_id)"),
    ("D18", "corpus_window.chunks = actual chunks in window",
     "SELECT w.window_id, w.chunks, (SELECT count(*) FROM chunk k WHERE k.window_id=w.window_id) FROM corpus_window w "
     "WHERE w.chunks IS DISTINCT FROM (SELECT count(*) FROM chunk k WHERE k.window_id=w.window_id)"),
    # ---- X: domain ------------------------------------------------------------------------------------ A stage's
    # debate is derived from its earliest linked sitting, set AND cleared by bills.link_all.
    ("D20", "division with both counts but no result, and no recorded double-majority reason",
     "SELECT id, ayes, noes, anomalies FROM division WHERE result IS NULL AND ayes IS NOT NULL AND noes IS NOT NULL "
     "AND NOT EXISTS (SELECT 1 FROM unnest(anomalies) a WHERE a LIKE 'double\\_majority\\_%')"),
    ("D19", "bill_stage.debate_id disagrees with its earliest linked sitting's debate",
     "SELECT st.id, st.debate_id, u.debate_id FROM bill_stage st LEFT JOIN LATERAL (SELECT t.debate_id "
     "FROM bill_stage_sitting t WHERE t.bill_stage_id = st.id AND t.debate_id IS NOT NULL "
     "ORDER BY t.date, t.debate_id LIMIT 1) u ON true WHERE st.debate_id IS DISTINCT FROM u.debate_id"),
    # The named counts behind `tally_reconciled`, re-derived from the vote rows the way the migration's backfill does
    # (Commons tellers apart, Lords tellers in, a two-lobby member in each).
    ("D21", "division's named counts differ from the members its vote rows name (D-037)",
     "SELECT d.id, d.ayes_named, d.noes_named, n.ayes, n.noes FROM division d LEFT JOIN LATERAL (SELECT "
     "count(*) FILTER (WHERE (v.lobby='aye' AND (d.house='Lords' OR NOT v.is_teller)) "
     "OR (v.lobby='conflict' AND 'listed_aye' = ANY(v.flags))) AS ayes, "
     "count(*) FILTER (WHERE (v.lobby='no' AND (d.house='Lords' OR NOT v.is_teller)) "
     "OR (v.lobby='conflict' AND 'listed_no' = ANY(v.flags))) AS noes, "
     "count(*) FILTER (WHERE d.house='Commons' AND v.lobby='conflict' AND v.is_teller) AS unplaceable "
     "FROM division_vote v WHERE v.division_id = d.id) n ON true "
     "WHERE n.unplaceable > 0 OR (d.ayes_named, d.noes_named) IS DISTINCT FROM (n.ayes, n.noes)"),
    # The API's OrderInDebateSection is the speech's seq; a missing one sorts the speech last in its debate's chunks,
    # silently, and U01 cannot see a lone NULL.
    ("X23", "speech.seq is NULL (the API gave no order in its debate)",
     "SELECT id, debate_id FROM speech WHERE seq IS NULL"),
    ("X01", "debate.date lies inside its window's declared range",
     "SELECT d.id, d.date, w.start_date, w.end_date FROM debate d JOIN corpus_window w ON w.window_id=d.window_id "
     "WHERE d.date < w.start_date OR d.date > w.end_date"),
    ("X02", "division_date lies inside some window",
     "SELECT v.id, v.division_date FROM division v WHERE NOT EXISTS (SELECT 1 FROM corpus_window w "
     "WHERE v.division_date BETWEEN w.start_date AND w.end_date)"),
    ("X03", "division result consistent with tellers: agreed => ayes>noes",
     "SELECT id, house, result, ayes, noes FROM division WHERE result='agreed' AND ayes <= noes"),
    ("X04", "division result consistent with tellers: negatived => noes>ayes",
     "SELECT id, house, result, ayes, noes FROM division WHERE result='negatived' AND noes <= ayes"),
    ("X05", "division result consistent with tellers: tied => ayes=noes",
     "SELECT id, house, result, ayes, noes FROM division WHERE result='tied' AND ayes <> noes"),
    ("X06", "division tallies non-negative and non-trivial",
     "SELECT id, ayes, noes FROM division WHERE ayes < 0 OR noes < 0 OR ayes + noes = 0"),
    ("X07", "division tally within a House's size (<= 900)",
     "SELECT id, house, ayes, noes FROM division WHERE ayes + noes > 900"),
    ("X08", "member_office start <= end",
     "SELECT id, mnis_id, start_date, end_date FROM member_office WHERE end_date IS NOT NULL AND end_date < start_date"),
    ("X09", "member membership start <= end",
     "SELECT mnis_id, membership_start_date, membership_end_date FROM member_id_map "
     "WHERE membership_end_date IS NOT NULL AND membership_end_date < membership_start_date"),
    # X10-X12 are checked against the membership HISTORY (`member_house_term`, a view over `houseMemberships`).
    ("X10", "a vote cast when the member held no membership of the division's House (history)",
     "SELECT v.division_id, v.mnis_id, d.division_date FROM division_vote v JOIN division d ON d.id=v.division_id "
     "WHERE v.lobby IN ('aye','no') AND NOT EXISTS (SELECT 1 FROM member_house_term t WHERE t.mnis_id=v.mnis_id "
     "AND t.house=d.house AND d.division_date >= t.start_date AND (t.end_date IS NULL OR d.division_date <= t.end_date))"),
    ("X11", "party at the division (history) disagrees with the division list's party",
     "SELECT v.division_id, v.mnis_id, v.party, member_party_on(v.mnis_id, d.division_date) FROM division_vote v "
     "JOIN division d ON d.id=v.division_id WHERE v.lobby IN ('aye','no') AND "
     "party_base(member_party_on(v.mnis_id, d.division_date)) IS DISTINCT FROM party_base(v.party)"),
    ("X12", "a speech made when the speaker held no membership of that House (history)",
     "SELECT s.id, s.mnis_id, d.date, d.house FROM speech s JOIN debate d ON d.id=s.debate_id WHERE NOT s.is_boilerplate "
     "AND NOT EXISTS (SELECT 1 FROM member_house_term t WHERE t.mnis_id=s.mnis_id AND t.house=d.house "
     "AND d.date >= t.start_date AND (t.end_date IS NULL OR d.date <= t.end_date))"),
    ("X10s", "SNAPSHOT (expected >0): vote outside member_id_map's LATEST membership — why snapshot columns must not "
             "answer date questions",
     "SELECT v.division_id FROM division_vote v JOIN division d ON d.id=v.division_id JOIN member_id_map m "
     "ON m.mnis_id=v.mnis_id WHERE v.lobby IN ('aye','no') AND ((m.membership_start_date IS NOT NULL AND "
     "d.division_date < m.membership_start_date) OR (m.membership_end_date IS NOT NULL AND "
     "d.division_date > m.membership_end_date))"),
    ("X13", "chunk embedding dimension is 384",
     "SELECT id, vector_dims(embedding) FROM chunk WHERE embedding IS NOT NULL AND vector_dims(embedding) <> 384"),
    ("X14", "chunk body is empty or whitespace",
     "SELECT id, speech_id FROM chunk WHERE body IS NULL OR btrim(body) = ''"),
    ("X15", "speech text is empty or whitespace",
     "SELECT id, debate_id FROM speech WHERE text IS NULL OR btrim(text) = ''"),
    ("X16", "debate heading is empty",
     "SELECT id, date, house FROM debate WHERE heading IS NULL OR btrim(heading) = ''"),
    ("X17", "division title is empty",
     "SELECT id, division_date, house FROM division WHERE title IS NULL OR btrim(title) = ''"),
    ("X18", "bill title is empty",
     "SELECT id FROM bill WHERE title IS NULL OR btrim(title) = ''"),
    ("X19", "bill_stage_sitting date outside every window (not an error; how much the bill record reaches beyond the corpus)",
     "SELECT s.id, s.date FROM bill_stage_sitting s WHERE NOT EXISTS (SELECT 1 FROM corpus_window w "
     "WHERE s.date BETWEEN w.start_date AND w.end_date)"),
    # A check that could never fire used to sit here ("a NUL or a lone surrogate": a UTF-8 database refuses to store
    # either).
    ("X20", "stored text contains U+FFFD (text mangled by a decoding failure before it reached us)",
     "SELECT 'debate.heading' AS col, id FROM debate WHERE strpos(heading, chr(65533)) > 0 "
     "UNION ALL SELECT 'speech.text', id FROM speech WHERE strpos(text, chr(65533)) > 0 "
     "UNION ALL SELECT 'chunk.body', id::text FROM chunk WHERE strpos(body, chr(65533)) > 0 "
     "UNION ALL SELECT 'division.title', id::text FROM division WHERE strpos(title, chr(65533)) > 0 "
     "UNION ALL SELECT 'bill.title', id::text FROM bill WHERE strpos(title, chr(65533)) > 0 "
     "UNION ALL SELECT 'member.name', mnis_id::text FROM member_id_map WHERE strpos(name, chr(65533)) > 0 "
     "UNION ALL SELECT 'correction.body', id::text FROM hansard_correction WHERE strpos(body, chr(65533)) > 0"),
    # The Members API keeps "Current Member" on an MP who left at a dissolution; 222 were stored as sitting.
    ("X21", "member marked active (or given a current status) whose membership has ended",
     "SELECT mnis_id, name, membership_end_date FROM member_id_map "
     "WHERE (is_active OR status IS NOT NULL) AND membership_end_date <= current_date"),
    # A 404 is absence (a stub), never a failure; a FAILED profile is a member we do not know at all.
    ("X22", "member whose profile fetch failed",
     "SELECT mnis_id, name FROM member_id_map WHERE 'profile' = ANY(missing_endpoints)"),
    # ---- U: uniqueness --------------------------------------------------------------------------------
    ("U01", "duplicate speech seq within a debate",
     "SELECT debate_id, seq, count(*) FROM speech GROUP BY 1,2 HAVING count(*) > 1"),
    ("U02", "duplicate (speech, seq, sent_idx) chunk",
     "SELECT speech_id, seq, sent_idx, count(*) FROM chunk GROUP BY 1,2,3 HAVING count(*) > 1"),
    ("U03", "a member voting twice in one division",
     "SELECT division_id, mnis_id, count(*) FROM division_vote GROUP BY 1,2 HAVING count(*) > 1"),
    ("U04", "duplicate division (house, date, number)",
     "SELECT house, division_date, number, count(*) FROM division GROUP BY 1,2,3 HAVING count(*) > 1"),
    ("U05", "duplicate speech item_id",
     "SELECT item_id, count(*) FROM speech GROUP BY 1 HAVING count(*) > 1"),
    # By SOURCE: member_office holds the Members API and Hansard's own attributions side by side on purpose, and the
    # first version of this check omitted `source` and reported that design as a duplicate.
    ("U06", "duplicate member_office interval within one source",
     "SELECT mnis_id, office_key, start_date, source, count(*) FROM member_office GROUP BY 1,2,3,4 HAVING count(*) > 1"),
    ("U07", "identical speech text twice in one debate (a double-loaded contribution)",
     "SELECT debate_id, md5(text), count(*) FROM speech WHERE length(text) > 200 GROUP BY 1,2 HAVING count(*) > 1"),
    # ---- C: completeness ------------------------------------------------------------------------------
    ("C01", "debate with no speech",
     "SELECT d.id, d.date, d.heading FROM debate d WHERE NOT EXISTS (SELECT 1 FROM speech s WHERE s.debate_id=d.id)"),
    ("C02", "non-boilerplate speech with no chunk",
     "SELECT s.id, left(s.text,60) FROM speech s WHERE NOT s.is_boilerplate AND NOT EXISTS (SELECT 1 FROM chunk k WHERE k.speech_id=s.id)"),
    ("C03", "division with no vote rows",
     "SELECT d.id, d.house, d.division_date FROM division d WHERE NOT EXISTS (SELECT 1 FROM division_vote v WHERE v.division_id=d.id)"),
    ("C04", "bill with no stage",
     "SELECT b.id, b.title FROM bill b WHERE NOT EXISTS (SELECT 1 FROM bill_stage s WHERE s.bill_id=b.id)"),
    ("C05", "non-boilerplate chunk without an embedding",
     "SELECT id, speech_id FROM chunk WHERE embedding IS NULL AND NOT is_boilerplate"),
    ("C06", "member who spoke or voted but has no party",
     "SELECT m.mnis_id, m.name FROM member_id_map m WHERE (m.party IS NULL OR btrim(m.party)='') AND "
     "(EXISTS (SELECT 1 FROM speech s WHERE s.mnis_id=m.mnis_id) OR EXISTS (SELECT 1 FROM division_vote v WHERE v.mnis_id=m.mnis_id))"),
    ("C07", "a sitting day inside a window with debates in one House and none in the other (information)",
     "SELECT d.date, array_agg(DISTINCT d.house) FROM debate d GROUP BY d.date HAVING count(DISTINCT d.house) = 1"),
    # The linker's own verdict.
    ("C08", "division naming a bill that reached no stage (bill_link none or ambiguous: honest refusals)",
     "SELECT id, house, division_date, left(title, 70), bill_link FROM division WHERE bill_link IN ('none', 'ambiguous')"),
    ("C10", "a window whose divisions have no voter lists in the Votes API (source_gap: the source's limit, disclosed)",
     "SELECT window_id, house, count FROM source_gap WHERE kind = 'no_voter_lists'"),
    ("C09", "member with a failed optional fetch (the source's failures; a re-fetch may recover them)",
     "SELECT mnis_id, name, missing_endpoints FROM member_id_map WHERE cardinality(missing_endpoints) > 0"),
    # The two official sources of a division's figures part.
    ("D22", "a division whose Hansard printed figures differ from the Votes API's (D-069: both are stated)",
     "SELECT id, house, division_date, ayes, noes, printed_ayes, printed_noes FROM division "
     "WHERE printed_ayes IS NOT NULL AND (printed_ayes, printed_noes) IS DISTINCT FROM (ayes, noes)"),
    # ---- P: procedure (S05)
    # ------------------------------------------------------------------------------------------- Threads are built from
    # each record's place in its House's day, which must exist and be its own.
    ("P03", "a debate record with no place in its House's day, or sharing it (section_order NULL or duplicated): "
            "threads cannot be built",
     "SELECT d.id, d.house, d.date, d.section, d.section_order FROM debate d WHERE d.section_order IS NULL OR EXISTS "
     "(SELECT 1 FROM debate o WHERE o.id <> d.id AND o.house = d.house AND o.date = d.date "
     "AND coalesce(o.section, '') = coalesce(d.section, '') AND o.section_order = d.section_order)"),
    # ---- I: index ----------------------------------------------------------------------------------------------------
    ("I01", f"{V.INDEX_NAME} is a valid HNSW index of the form frozen retrieval queries ({_FROZEN_FORM}; "
            "switch with scripts/switch_vector_index.py)",
     f"SELECT '{V.INDEX_NAME}' AS index, coalesce(pg_get_indexdef(i.indexrelid), 'missing') AS definition, "
     "i.indisvalid AS valid FROM (SELECT 1) one "
     f"LEFT JOIN pg_class c ON c.relname = '{V.INDEX_NAME}' AND c.relkind = 'i' "
     "LEFT JOIN pg_index i ON i.indexrelid = c.oid "
     "WHERE EXISTS (SELECT 1 FROM chunk WHERE embedding IS NOT NULL) "
     f"AND (i.indexrelid IS NULL OR NOT i.indisvalid OR pg_get_indexdef(i.indexrelid) NOT LIKE '{_INDEX_LIKE}')"),
]


# Every debate's record, read once for the opener checks of one gate run (`shared_records`); outside one, each check
# reads the record afresh.
_SHARED: dict = {}


@contextmanager
def shared_records(conn):
    """P01, P05, P06 and P07 judge the same record: its speeches are read once for all four. A read that fails shares
    nothing, so each check reads again and reports its own error, as an erroring check always has.
    """
    try:
        _SHARED["records"] = OP.records_sync(conn)
    except Exception:  # noqa: BLE001 - each check then reports the error itself, and a hard one fails the gate
        conn.rollback()
    try:
        yield
    finally:
        _SHARED.clear()


def _records(conn) -> dict:
    return _SHARED["records"] if "records" in _SHARED else OP.records_sync(conn)


def _p01_opener_presides(conn) -> list[list]:
    """Each debate's opener, by `procedure.opener`, judged by the stored role and by the MR-08 oracle's own Chair.
    """
    out = []
    for did, rec in sorted(_records(conn).items()):
        speech = rec.found.speech
        if speech is not None and ((speech.speaker_role or "") in OP.PRESIDING_ROLES or OO.presiding(speech.speaker_name)):
            out.append([did, str(rec.date), speech.speech_id, speech.speaker_name])
    return out


def _p05_opener_says_nothing(conn) -> list[list]:
    """Both ways, by the MR-08 oracle's own reading of a contribution that says nothing (an empty one, or italic stage
    directions alone): an opener the rule chose that says nothing (QA pass 6), and a record the rule finds no opener
    in where a member's contribution, by the oracle's reading, says something (QA pass 7).
    """
    out = []
    for did, rec in sorted(_records(conn).items()):
        found = rec.found
        speech = found.speech
        if speech is not None and speech.whole is not None and OO.says_nothing(speech.whole):
            out.append([did, str(rec.date), speech.speech_id, speech.speaker_name])
        elif speech is None and not found.chair_business:
            spoke = next((s for s in rec.speeches if s.mnis_id is not None and s.whole is not None
                          and (s.speaker_role or "") not in OP.PRESIDING_ROLES and not OO.presiding(s.speaker_name)
                          and not OO.says_nothing(s.whole)), None)  # fmt: skip
            if spoke is not None:
                out.append([did, str(rec.date), spoke.speech_id, spoke.speaker_name])
    return out


def _p07_readers_part(conn) -> list[list]:
    """Every contribution read whole, wherever it falls in its record: the rule's reading
    (`procedure.opener.non_speech`) against the MR-08 oracle's (`says_nothing`). P05 judges openers only, while the
    roster's count of who spoke reads every contribution; and the two readers part on malformed markup the record
    does not hold today, so a load that brings it fails the gate here.
    """
    out = []
    for did, rec in sorted(_records(conn).items()):
        for speech in rec.speeches:
            if speech.whole is not None and OP.non_speech(speech.whole) != OO.says_nothing(speech.whole):
                out.append([did, str(rec.date), speech.speech_id, speech.speaker_name])
    return out


def _p06_chairs_statement_opened(conn) -> list[list]:
    """A member named as opening business the MR-08 oracle reads, by its heading, as the Chair's own statement."""
    out = []
    for did, rec in sorted(_records(conn).items()):
        speech = rec.found.speech
        if speech is not None and OO.chairs_statement(rec.heading):
            out.append([did, str(rec.date), speech.speech_id, speech.speaker_name])
    return out


def _p02_chair_without_role(conn) -> list[list]:
    """Attributions either definition reads as the Chair's, stored without a presiding role."""
    rows = conn.execute("SELECT speaker_name, coalesce(speaker_role, ''), count(*), min(id) FROM speech "
                        "WHERE coalesce(speaker_role, '') <> ALL(%s) AND speaker_name IS NOT NULL GROUP BY 1, 2 "
                        "ORDER BY 1", (list(OP.PRESIDING_ROLES),)).fetchall()  # fmt: skip
    return [[name, role, n, first] for name, role, n, first in rows if OP.presiding(name, None) or OO.presiding(name)]


def _p04_threads_unseparated(conn) -> list[list]:
    metas = T.all_metas_sync(conn)
    return [[a, b] for a, b in T.violations(T.build(metas), metas)]


# Checks computed in Python, because their rule lives there (`procedure.opener`, `procedure.threads`); reported and
# gated exactly like the SQL ones, examples withheld from the sealed window the same way.
COMPUTED = [
    ("P01", "a debate's opener (procedure.opener) is the Chair, by its stored role or the MR-08 oracle's printed forms "
            "(D-001)", _p01_opener_presides),
    ("P02", "a printing of the Chair stored without a presiding role (D-064: ingest's classifier and migration 029)",
     _p02_chair_without_role),
    ("P04", "two debate threads under one heading with no other business between them (H-10)", _p04_threads_unseparated),
    ("P05", "a debate's opener (procedure.opener) says nothing by the MR-08 oracle's reading: empty, or stage directions "
            "alone (QA pass 6)", _p05_opener_says_nothing),
    ("P06", "a member named as opening the Chair's own statement, by the MR-08 oracle's reading of the heading "
            "(QA pass 6)", _p06_chairs_statement_opened),
    ("P07", "a contribution the rule (procedure.opener.non_speech) and the MR-08 oracle read differently: says nothing "
            "or a speech (S05's real-data check)", _p07_readers_part),
]


# E01 to E04: every division's flags against `config/eras.toml`.
_ERA_ROWS_SQL = ("SELECT d.id, d.house, d.division_date, d.is_remote, d.is_deferred, d.evel_type, "
                 "d.double_majority_ayes, d.double_majority_noes, (SELECT count(*) FROM division_vote v "
                 "WHERE v.division_id = d.id AND v.proxy_name IS NOT NULL) AS proxy_votes FROM division d ORDER BY d.id")


def _era_check(check: str):
    def run(conn) -> list[list]:
        cur = conn.execute(_ERA_ROWS_SQL)
        columns = [c.name for c in cur.description or []]
        return ERAS.violations(dict(zip(columns, row)) for row in cur.fetchall())[check]

    return run


COMPUTED += [(check, ERAS.CHECKS[check], _era_check(check)) for check in sorted(ERAS.CHECKS)]


# Checks that are REPORTED but never fail the gate: facts about the record, or measurements of how far it reaches,
# rather than properties that must hold.
INFO = {
    "C10": "a coverage limit of Parliament's Votes API (the ten-year span reaches back to 2015); disclosed, not ours",
    "D22": "two official sources disagree (D-069); every answer states both, so the count is watched, not failed",
    "C08": "the linker's honest refusals (two stages one day, a title missing the word Bill); counted, not failures",
    "C09": "the Members API fails persistently for a few optional endpoints (HTTP 500); reported, not ours to fix",
    "R14": "deferred divisions whose committee debate is outside the corpus stay unplaced, labelled",
    "D08": "measured before being made hard",
    "X15": "Hansard's 'tabled by' markers are legitimately empty contributions",
    "X16": "the source may omit a heading; stored as NULL, not invented",
    "X19": "a bill's record reaches beyond the windows the corpus holds",
    "C07": "the Houses sit on different days",
    "C08": "a naive title check, superseded by the bill linker's own report",
}
# Checks whose violations are EXPECTED and whose point is the lesson (see the comment on X10s above).
EXPECTED = {"X10s"}

# Example rows shown per check, and how many are read to find them outside the sealed window.
EXAMPLES, PROBE = 3, 50

# Every id a check's row can name, of every table that dates the record: text ids (debates, speeches) and integer ids.
_SEALED_IDS = ("SELECT d.id FROM debate d WHERE d.id = ANY(%(ids)s) AND d.date BETWEEN %(lo)s AND %(hi)s "
               "UNION SELECT s.id FROM speech s JOIN debate d ON d.id = s.debate_id "
               "WHERE s.id = ANY(%(ids)s) AND d.date BETWEEN %(lo)s AND %(hi)s "
               "UNION SELECT v.id::text FROM division v WHERE v.id = ANY(%(nums)s) AND v.division_date "
               "BETWEEN %(lo)s AND %(hi)s "
               "UNION SELECT k.id::text FROM chunk k WHERE k.id = ANY(%(nums)s) AND k.debate_date BETWEEN %(lo)s AND %(hi)s "
               "UNION SELECT t.id::text FROM bill_stage_sitting t WHERE t.id = ANY(%(nums)s) AND t.date "
               "BETWEEN %(lo)s AND %(hi)s "
               "UNION SELECT g.id::text FROM bill_stage g WHERE g.id = ANY(%(nums)s) AND (g.date BETWEEN %(lo)s AND %(hi)s "
               "OR g.first_sitting BETWEEN %(lo)s AND %(hi)s OR g.last_sitting BETWEEN %(lo)s AND %(hi)s)")


def examples_outside_window(conn, sql: str, keep: int = EXAMPLES, probe: int = PROBE) -> tuple[list[list[str]], int]:
    """Up to `keep` example rows of a check's violations, none from the sealed window, and how many were withheld.
    """
    rows = [list(map(str, r)) for r in conn.execute(f"{sql} LIMIT {int(probe)}").fetchall()]
    return rows_outside_window(conn, rows, keep)


def rows_outside_window(conn, rows: list[list[str]], keep: int = EXAMPLES) -> tuple[list[list[str]], int]:
    """`examples_outside_window` for rows already read (the computed checks)."""
    rows = [list(map(str, r)) for r in rows]
    if not rows or not sealed.masked():
        return rows[:keep], 0
    w = sealed.window()
    cells = sorted({cell for row in rows for cell in row})
    nums = sorted({int(c) for c in cells if c.isdigit() and len(c) < 19})
    ids = {r[0] for r in conn.execute(_SEALED_IDS, {"ids": cells, "nums": nums, "lo": w.start,
                                                    "hi": w.end}).fetchall()}  # fmt: skip
    clean = [row for row in rows
             if not any(sealed.is_sealed(cell) or sealed.sealed_days_in(cell) or cell in ids for cell in row)]
    return clean[:keep], len(rows) - len(clean)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if any HARD check has a violation or errors; for the pipeline, after every load")
    parser.add_argument("--database", default=safety.TARGET)
    args = parser.parse_args(argv)

    results = []
    # Read-only in the server, not by convention: an audit that could write is not an audit.
    with psycopg.connect(make_conninfo(settings.pg_dsn, dbname=args.database),
                         options="-c default_transaction_read_only=on") as c:
        print(f"database: {c.execute('SELECT current_database()').fetchone()[0]}")
        c.execute("SET statement_timeout = '900s'")
        for code, what, sql in CHECKS:
            severity = "expected" if code in EXPECTED else ("info" if code in INFO else "hard")
            started = time.perf_counter()
            try:
                n = c.execute(f"SELECT count(*) FROM ({sql}) q").fetchone()[0]
                examples, withheld = examples_outside_window(c, sql) if n else ([], 0)
                ms = int((time.perf_counter() - started) * 1000)
                results.append({"code": code, "what": what, "severity": severity, "violations": n,
                                "examples": examples, "ms": ms,
                                **({"examples_withheld_sealed_window": withheld} if withheld else {})})
                mark = "ok " if n == 0 else ("!!!" if severity == "hard" else " . ")
                print(f"{code:5} {mark} {n:>9,}  [{severity:8}] {what}   ({ms} ms)")
                for e in (examples[:2] if severity == "hard" else []):
                    print(f"                 e.g. {e}")
            except Exception as exc:  # noqa: BLE001 - an erroring check is reported and, if HARD, fails the gate
                c.rollback()
                c.execute("SET statement_timeout = '900s'")
                results.append({"code": code, "what": what, "severity": severity,
                                "error": f"{type(exc).__name__}: {exc}"})
                print(f"{code:5} ERR  [{severity:8}] {what}: {type(exc).__name__}: {str(exc)[:120]}")
        with shared_records(c):
            for code, what, check in COMPUTED:
                severity = "expected" if code in EXPECTED else ("info" if code in INFO else "hard")
                started = time.perf_counter()
                try:
                    found = check(c)
                    examples, withheld = rows_outside_window(c, found[:PROBE]) if found else ([], 0)
                    ms = int((time.perf_counter() - started) * 1000)
                    results.append({"code": code, "what": what, "severity": severity, "violations": len(found),
                                    "examples": examples, "ms": ms,
                                    **({"examples_withheld_sealed_window": withheld} if withheld else {})})
                    mark = "ok " if not found else ("!!!" if severity == "hard" else " . ")
                    print(f"{code:5} {mark} {len(found):>9,}  [{severity:8}] {what}   ({ms} ms)")
                    for e in (examples[:2] if severity == "hard" else []):
                        print(f"                 e.g. {e}")
                except Exception as exc:  # noqa: BLE001 - as above: an erroring hard check fails the gate
                    c.rollback()
                    results.append({"code": code, "what": what, "severity": severity,
                                    "error": f"{type(exc).__name__}: {exc}"})
                    print(f"{code:5} ERR  [{severity:8}] {what}: {type(exc).__name__}: {str(exc)[:120]}")
    (ROOT / "reports" / "audit_invariants.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    hard = [r for r in results if r["severity"] == "hard"]
    failing = [r for r in hard if r.get("violations") or r.get("error")]
    print(f"\n{len(results)} checks: {len(hard)} hard ({len(failing)} failing), "
          f"{sum(r['severity'] == 'info' for r in results)} info, "
          f"{sum(r['severity'] == 'expected' for r in results)} expected")
    if failing:
        print("HARD FAILURES: " + ", ".join(r["code"] for r in failing))
    # An audit that ERRORS has not checked anything, so it can never be a pass: in gate mode an erroring hard check is a
    # failure exactly like a violated one.
    return 1 if (args.gate and failing) else 0


if __name__ == "__main__":
    sys.exit(main())
