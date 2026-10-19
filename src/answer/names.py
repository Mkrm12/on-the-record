"""The name the record printed."""

from __future__ import annotations

from collections.abc import Iterable

SQL_AS_PRINTED = "SELECT id, speaker_as_printed(id) FROM speech WHERE id = ANY(%s)"


async def as_printed(conn, speech_ids: Iterable[str]) -> dict[str, str]:
    """{speech id: the speaker as a reader is shown them} for every id the record can name."""
    ids = sorted({i for i in speech_ids if i})
    if not ids:
        return {}
    cur = await conn.execute(SQL_AS_PRINTED, (ids,))
    return {speech_id: name for speech_id, name in await cur.fetchall() if name}
