"""Build small, fully controlled corpora in the scratch database for retrieval tests."""

import math
from collections import defaultdict

from pgvector import Vector

from ingest.chunk import build_chunks

DIM = 384
BACKGROUND = DIM - 1  # chunks with no stated vector sit on this axis: orthogonal to every test query, so never a dense hit


# The window every fixture debate and speech belongs to unless a test states another.
TEST_WINDOW = "T0"


def ensure_window(conn, window_id: str = TEST_WINDOW) -> str:
    conn.execute("INSERT INTO corpus_window (window_id, corpus_version, start_date, end_date, role)"
                 " VALUES (%s, 'test', '1900-01-01', '2100-12-31', 'dev') ON CONFLICT DO NOTHING", (window_id,))
    return window_id


def vec(*weights: tuple[int, float]) -> list[float]:
    """A unit vector with the given (axis, weight) coordinates, e.g. vec((0, 1), (1, 0.5))."""
    out = [0.0] * DIM
    for axis, weight in weights:
        out[axis] = weight
    norm = math.sqrt(sum(x * x for x in out))
    return [x / norm for x in out]


def distance(a: list[float], b: list[float]) -> float:
    """The cosine distance pgvector's <=> would give, for hand-checking expected ranks."""
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class Corpus:
    def __init__(self, conn):
        self.conn = conn
        self._next_seq: dict[str, int] = defaultdict(int)
        self._speech_count = 0

    def member(self, mnis_id, name, *, party="Labour", list_as=None, full_title=None, house="Commons"):
        self.conn.execute(
            "INSERT INTO member_id_map (mnis_id, name, party, name_list_as, name_full_title, house) VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (mnis_id) DO NOTHING",
            (mnis_id, name, party, list_as, full_title, house),
        )

    def debate(self, debate_id="d1", heading="A debate", date="2026-09-01", house="Commons"):
        ensure_window(self.conn)
        self.conn.execute(
            "INSERT INTO debate (id, date, house, heading, section, debate_type, window_id, corpus_version)"
            " VALUES (%s, %s, %s, %s, 'Chamber', 'other', %s, 'test') ON CONFLICT DO NOTHING",
            (debate_id, date, house, heading, TEST_WINDOW),
        )

    def speech(self, debate_id, speech_id, sentences, *, mnis_id=None, speaker_name=None, party=None, office=None, vectors=None, boilerplate=None):
        """One speech and its chunks; returns the chunk ids in sentence order. `vectors` is one per sentence (None
        entries use the background axis); `boilerplate` marks sentences the rules engine would flag.
        """
        self.conn.execute(
            "INSERT INTO speech (id, debate_id, seq, speaker_name, mnis_id, speaker_party, speaker_office, text,"
            " window_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (speech_id, debate_id, self._speech_count, speaker_name, mnis_id, party, office, " ".join(sentences),
             TEST_WINDOW),
        )
        self._speech_count += 1
        vectors = vectors or [None] * len(sentences)
        ids = []
        for draft, v in zip(build_chunks(sentences, boilerplate=boilerplate), vectors):
            seq = self._next_seq[debate_id]
            self._next_seq[debate_id] += 1
            # Born with its filter columns, as the chunker writes them and invariant D02 requires: retrieval's scoping
            # and the sealed-window mask both read `debate_date`, and an undated chunk is withheld.
            chunk_id = self.conn.execute(
                "INSERT INTO chunk (speech_id, debate_id, seq, sent_idx, body, parent_body, is_boilerplate, debate_date,"
                " window_id, house) SELECT %s, %s, %s, %s, %s, %s, %s, d.date, d.window_id, d.house FROM debate d"
                " WHERE d.id = %s RETURNING id",
                (speech_id, debate_id, seq, draft.sent_idx, draft.body, draft.parent_body, draft.is_boilerplate,
                 debate_id),
            ).fetchone()[0]
            if not draft.is_boilerplate:
                self.conn.execute("UPDATE chunk SET embedding = %s WHERE id = %s", (Vector(v or vec((BACKGROUND, 1.0))), chunk_id))
            ids.append(chunk_id)
        return ids
