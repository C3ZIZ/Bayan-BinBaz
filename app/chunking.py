"""Split long fatwa answers into overlapping, token-bounded chunks.

Two reasons this exists:

1. The previous index encoded at ``max_length=512`` while BGE-M3 supports 8192.
   Measured against the real tokenizer, 17.7% of fatwas were truncated and 19.6%
   of all corpus tokens never reached a vector — that content was unretrievable.
2. A long fatwa averaged into one vector dilutes every individual point it makes.
   Chunking gives each point its own vector, which sharpens retrieval precision.

Splitting is on whitespace, so a word is never cut in half. Overlap keeps a
ruling and its exception clause from being separated at a chunk boundary.
"""

from typing import Callable, List


def chunk_text(
    text: str,
    token_len: Callable[[str], int],
    max_tokens: int = 450,
    overlap_tokens: int = 100,
) -> List[str]:
    """Split ``text`` into chunks of at most ``max_tokens``, overlapping by
    ``overlap_tokens``.

    ``token_len`` is injected so unit tests run without downloading BGE-M3;
    production passes the real tokenizer.

    A single word longer than ``max_tokens`` is emitted as its own oversized
    chunk rather than dropped — losing corpus content is worse than one long
    vector, and the model truncates it safely.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    words = text.split()
    if not words:
        return []

    if token_len(" ".join(words)) <= max_tokens:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        # Binary-search the largest window starting at `start` that still fits.
        lo, hi, best = start + 1, len(words), start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if token_len(" ".join(words[start:mid])) <= max_tokens:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1

        chunks.append(" ".join(words[start:best]))
        if best >= len(words):
            break
        # Always advance by at least one word so an oversized word terminates.
        start += max(1, (best - start) - overlap_tokens)

    return chunks
