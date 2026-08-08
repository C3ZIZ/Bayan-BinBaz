"""Shared text helpers.

``truncate_at_sentence`` exists because the original code cut fatwa text with
``text[:800] + "..."``. In a religious ruling that is not a cosmetic problem: a
cut placed before an exception clause ("...إلا إذا") can invert the ruling the
user reads. Cutting at a sentence boundary keeps each retained clause complete.
"""

from typing import Iterable

# Arabic full stop is ".", but Arabic text also ends clauses with ؛ ، ؟ !
_SENTENCE_ENDS: Iterable[str] = (".", "؟", "!", "؛", "\n")


def truncate_at_sentence(text: str, limit: int, min_ratio: float = 0.6) -> str:
    """Trim ``text`` to at most ``limit`` characters, ending on a sentence
    boundary when one exists past ``min_ratio`` of the limit.

    Falls back to a hard cut with an ellipsis so the caller always gets
    something bounded.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text

    window = text[:limit]
    best = -1
    for sep in _SENTENCE_ENDS:
        idx = window.rfind(sep)
        if idx > best:
            best = idx

    if best > limit * min_ratio:
        return window[: best + 1].strip()
    return window.strip() + "…"
