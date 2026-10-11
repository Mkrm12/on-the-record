"""What a user typed becomes something safe to search with. Pure functions, no database, no GPU."""

import re

MAX_QUERY_CHARS = 2000  # about 400 words; the embedder truncates at 512 tokens anyway, and a longer paste is not a question

# Control characters other than tab and newline (those are ordinary whitespace and collapse below) become a gap, so
# "a\x00b" does not glue two words together.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_INVISIBLE = re.compile("[­؀-؅؜۝܏᠎​-‏‪-‮⁠-⁤⁦-⁯﻿￹-￻\U000e0001\U000e0020-\U000e007f]")


def _has_text(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def clean_query(text: object) -> tuple[str, bool]:
    """(the query made safe, whether it had to be shortened). ("", False) when nothing searchable is left, and for a
    value that is not text at all.
    """
    if not isinstance(text, str):
        return "", False
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = _INVISIBLE.sub("", _CONTROL.sub(" ", text))
    text = " ".join(text.split())
    truncated = len(text) > MAX_QUERY_CHARS
    if truncated:
        cut = text[:MAX_QUERY_CHARS]
        space = cut.rfind(" ")
        text = (cut[:space] if space > MAX_QUERY_CHARS - 200 else cut).rstrip()
    return (text, truncated) if _has_text(text) else ("", False)
