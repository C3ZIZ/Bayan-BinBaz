"""Citation marker handling.

The generation prompt requires every ruling to carry a ``[n]`` marker pointing at
one of the numbered fatwas it was given. These helpers enforce that contract on
the way out: a marker the model invented — pointing at a source that was never
supplied — is removed rather than shown, because a citation the reader cannot
verify is worse than no citation at all.

Markers are 1-based positions in the supplied source list, not fatwa ids; the
API maps them to fatwas and links.
"""

import re
from typing import List, Tuple

_MARKER = re.compile(r"\[(\d+)\]")


def extract_markers(text: str) -> List[int]:
    """Marker numbers in first-appearance order, deduplicated."""
    seen, out = set(), []
    for match in _MARKER.finditer(text or ""):
        n = int(match.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def strip_invalid_markers(text: str, n_sources: int) -> Tuple[str, List[int]]:
    """Remove markers outside ``1..n_sources``.

    Returns the cleaned text and the valid markers actually used, in order.
    """
    if n_sources < 0:
        raise ValueError("n_sources must not be negative")

    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= n_sources else ""

    cleaned = _MARKER.sub(replace, text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.،؛!؟])", r"\1", cleaned)
    return cleaned.strip(), extract_markers(cleaned)
