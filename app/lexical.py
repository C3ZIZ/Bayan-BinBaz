"""BM25 lexical retrieval over the fatwa corpus.

Why lexical retrieval at all: dense BGE-M3 similarity is heavily compressed on
this corpus and misses exact terminology. Arabic religious text turns on
specific terms — أسماء, أحكام, hadith wording — where a lexical hit is decisive
and an embedding may blur it.

Why BM25 rather than BGE-M3's learned sparse vectors: the learned weights need
``return_sparse=True`` at encode time, i.e. a full re-encode of ~45k texts plus
a new committed artifact. BM25 builds from text in seconds at startup, adds zero
bytes to the repository, and fuses through RRF identically. If the eval later
shows learned sparse is worth the re-encode, the fusion layer does not change.

Index construction is pure NumPy over a sparse CSR-style layout; at 18.7k short
documents the whole thing is a few MB and builds in about a second.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .normalize import normalize_arabic

# Keep Arabic letters, Latin letters and digits; everything else is a separator.
_TOKEN = re.compile(r"[ء-يA-Za-z0-9]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> List[str]:
    """Normalize then split into terms. Normalization must match the corpus side
    so a query written without tashkeel still matches text that has it."""
    return _TOKEN.findall(normalize_arabic(text))


class BM25Index:
    """Okapi BM25. Built once at startup and held in memory."""

    def __init__(self, documents: Sequence[str], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)

        raw: Dict[str, List[Tuple[int, int]]] = {}
        doc_lengths = np.zeros(self.n_docs, dtype=np.float32)

        for doc_idx, doc in enumerate(documents):
            terms = tokenize(doc)
            doc_lengths[doc_idx] = len(terms)
            for term, freq in Counter(terms).items():
                raw.setdefault(term, []).append((doc_idx, freq))

        # Convert postings to NumPy arrays. As Python tuples this is ~1.9M
        # objects over the full corpus — a few hundred MB held permanently,
        # against a ~4GB box that also loads a 2.3GB embedder. As int32/float32
        # arrays it is ~15MB.
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            term: (
                np.fromiter((d for d, _ in post), dtype=np.int32, count=len(post)),
                np.fromiter((f for _, f in post), dtype=np.float32, count=len(post)),
            )
            for term, post in raw.items()
        }
        del raw

        self.doc_lengths = doc_lengths
        self.avg_doc_length = float(doc_lengths.mean()) if self.n_docs else 0.0

        # Standard BM25 idf with the +1 guard so a term present in every
        # document scores 0 rather than going negative.
        self._idf: Dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - len(idx) + 0.5) / (len(idx) + 0.5))
            for term, (idx, _) in self._postings.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        """BM25 score for every document. Shape (n_docs,)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avg_doc_length == 0.0:
            return scores

        for term in tokenize(query):
            posting = self._postings.get(term)
            if posting is None:
                continue
            idx, tf = posting
            idf = self._idf[term]
            norm = 1.0 - self.b + self.b * (self.doc_lengths[idx] / self.avg_doc_length)
            # np.add.at, not +=, so a term repeated in the query accumulates
            # correctly rather than being overwritten by fancy-index assignment.
            np.add.at(scores, idx, idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm))

        return scores

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Return ``(doc_index, score)`` for the best ``top_k`` documents."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.n_docs or not tokenize(query):
            return []

        scores = self.score_all(query)
        top_k = min(top_k, self.n_docs)
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in ordered]
