"""Split a multi-part question into its parts.

Three questions in one string average into a single embedding that matches none
of them. Measured on the real index:

    «حكم اكل لحم الخنزير»            → حكم طبخ المسلم لحم الخنزير لغير المسلمين  ✓
    «كم عدد ركعات صلاة المغرب»       → فتاوى عن عدد الركعات                      ✓
    both plus a third, in one string → الشفع والوتر / المسافر — neither topic     ✗

So each part is retrieved for separately and the candidate sets are merged
before the gate sees them.
"""

import re
from typing import List

# Arabic and Latin question marks both end a part.
_TERMINATORS = re.compile(r"[؟?]")

# Connectors that join a second question when no ؟ was typed. Each must be
# followed by an interrogative for the split to be safe: «وهل» starts a new
# question, while «وهو» and «وشراء» in the middle of one do not.
#
# The Arabic conjunction «و» is written attached to the next word, so «وهل» has
# no space in it — the attached and detached forms both have to be matched.
_INTERROGATIVES = r"(?:هل|ماذا|متى|أين|كيف|لماذا|أيهما|كم|ما|أي)"

_CONNECTOR = re.compile(
    r"[,،]?\s*(?:"
    # detached connector followed by a space, e.g. «كمان كم …»
    rf"(?:كمان|وكمان|أيضا|أيضًا|وأيضا|وأيضًا|ثم|و)\s+(?={_INTERROGATIVES}\b)"
    r"|"
    # attached «و» prefix, e.g. «وهل …»
    rf"(?<=\s)و(?={_INTERROGATIVES}\b)"
    r")"
)

MIN_PART_WORDS = 2
MAX_PARTS = 4


def split_questions(text: str) -> List[str]:
    """Return the question's parts, longest-first-order preserved.

    Always returns at least one item. A single question comes back unchanged, so
    callers can use this unconditionally.
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1. Split on explicit question marks, keeping each part's own mark.
    rough: List[str] = []
    start = 0
    for match in _TERMINATORS.finditer(text):
        rough.append(text[start : match.end()])
        start = match.end()
    tail = text[start:]
    if tail.strip():
        rough.append(tail)

    # 2. Within each, split on a connector that introduces a new interrogative.
    parts: List[str] = []
    for chunk in rough:
        parts.extend(p for p in _CONNECTOR.split(chunk))

    cleaned = []
    for p in parts:
        p = p.strip(" \t\n،,")
        if len(p.split()) >= MIN_PART_WORDS:
            cleaned.append(p)

    if len(cleaned) <= 1:
        return [text]
    return cleaned[:MAX_PARTS]


def is_multipart(text: str) -> bool:
    return len(split_questions(text)) > 1
