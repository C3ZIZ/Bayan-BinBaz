import ctypes
import gc
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fusion import mmr_select, reciprocal_rank_fusion
from .normalize import normalize_arabic

if TYPE_CHECKING:  # pragma: no cover - typing only
    from FlagEmbedding import BGEM3FlagModel

    from .lexical import BM25Index

# FlagEmbedding (and through it torch) is imported lazily inside _ensure_model.
# Importing it at module scope would pull ~1GB of framework into every process
# that merely wants to score vectors — including the web app, which loads the
# model lazily anyway, and the unit tests for score_fatwas.

INDEX_DIR = Path("data/index")
QUESTION_EMB_PATH = INDEX_DIR / "question_emb.npy"
ANSWER_EMB_PATH = INDEX_DIR / "answer_emb.npy"
ANSWER_PARENT_PATH = INDEX_DIR / "answer_parent.npy"
META_PATH = INDEX_DIR / "fatwas_meta.parquet"
MANIFEST_PATH = INDEX_DIR / "index_manifest.json"

# Embedding model config (overridable via env for CPU/GPU tuning).
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
# fp16 helps on GPU but is slow/unstable on CPU-only hosts — keep it off by default.
EMB_USE_FP16 = os.getenv("EMB_USE_FP16", "0") == "1"

# Weight on question-to-question similarity. The previous single-vector index was
# 81% answer text while users send short questions, which compressed a perfect
# self-match to a median cosine of 0.815. Tuned by eval/run_retrieval_eval.py.
ALPHA = float(os.getenv("RETRIEVAL_ALPHA", "0.65"))

# Hybrid retrieval: fuse dense and BM25 rankings via RRF, then diversify with
# MMR. All switchable so eval/run_retrieval_eval.py can A/B each piece.
USE_HYBRID = os.getenv("RETRIEVAL_HYBRID", "1") == "1"
USE_MMR = os.getenv("RETRIEVAL_MMR", "1") == "1"
CANDIDATE_DEPTH = int(os.getenv("RETRIEVAL_DEPTH", "50"))
MMR_POOL = int(os.getenv("RETRIEVAL_MMR_POOL", "25"))
MMR_LAMBDA = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.7"))
DENSE_WEIGHT = float(os.getenv("RETRIEVAL_DENSE_WEIGHT", "1.0"))
LEXICAL_WEIGHT = float(os.getenv("RETRIEVAL_LEXICAL_WEIGHT", "0.6"))


def _malloc_trim() -> None:
    """Best-effort: return freed heap arenas to the OS (glibc/Linux only)."""
    try:
        if platform.system() == "Linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def score_fatwas(
    query_vec: np.ndarray,
    question_emb: np.ndarray,
    answer_emb: np.ndarray,
    answer_parent: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """Blend question similarity with max-pooled answer-chunk similarity.

    A fatwa is scored by its BEST matching chunk, not its average, so a long
    fatwa's individual points are not diluted. Returns one score per fatwa.
    """
    q_scores = question_emb.astype(np.float32) @ query_vec
    n = len(q_scores)

    # Seed with -inf, not 0: a genuinely negative cosine must not be floored to
    # zero, which would inflate unrelated fatwas above related ones.
    a_scores = np.full(n, -np.inf, dtype=np.float32)
    if len(answer_emb):
        chunk_scores = answer_emb.astype(np.float32) @ query_vec
        np.maximum.at(a_scores, answer_parent, chunk_scores)

    # Fatwas with no chunks (defensive; the index guarantees at least one)
    # degenerate to their question score rather than dragging the blend down.
    missing = ~np.isfinite(a_scores)
    a_scores[missing] = q_scores[missing]

    return alpha * q_scores + (1.0 - alpha) * a_scores


class FatwaRetriever:
    """
    Holds the (cheap) prebuilt index in memory permanently, and the (expensive)
    BGE-M3 model only while it's needed. The model loads lazily on the first
    query and can be unloaded when the service is idle to free RAM.
    """

    def __init__(self):
        for p in (QUESTION_EMB_PATH, ANSWER_EMB_PATH, ANSWER_PARENT_PATH, META_PATH):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run build_index.py first.")

        # float16 on disk; cast to float32 per query inside score_fatwas.
        self.question_emb = np.load(QUESTION_EMB_PATH)
        self.answer_emb = np.load(ANSWER_EMB_PATH)
        self.answer_parent = np.load(ANSWER_PARENT_PATH)
        self.meta = pd.read_parquet(META_PATH)
        self.manifest = (
            json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if MANIFEST_PATH.exists()
            else {}
        )

        if len(self.question_emb) != len(self.meta):
            raise RuntimeError(
                f"question_emb rows ({len(self.question_emb)}) != "
                f"meta rows ({len(self.meta)})"
            )
        if len(self.answer_parent) != len(self.answer_emb):
            raise RuntimeError(
                f"answer_parent ({len(self.answer_parent)}) != "
                f"answer_emb rows ({len(self.answer_emb)})"
            )

        self._model: Optional["BGEM3FlagModel"] = None
        self._model_lock = threading.Lock()
        self._bm25: Optional["BM25Index"] = None
        self._bm25_lock = threading.Lock()

    # --- model lifecycle -----------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> "BGEM3FlagModel":
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from FlagEmbedding import BGEM3FlagModel

                    self._model = BGEM3FlagModel(
                        EMB_MODEL_NAME, use_fp16=EMB_USE_FP16
                    )
        return self._model

    def unload_model(self) -> bool:
        """Drop the BGE-M3 model and return freed memory to the OS.
        Returns True if it was loaded."""
        with self._model_lock:
            if self._model is None:
                return False
            self._model = None
        gc.collect()
        _malloc_trim()
        return True

    # --- search --------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a query. Normalization MUST match build_index.py — a mismatch
        degrades retrieval silently."""
        model = self._ensure_model()
        outputs = model.encode([normalize_arabic(text)], batch_size=1, max_length=512)
        vecs = outputs["dense_vecs"].astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs  # shape: (1, D)

    @property
    def bm25(self) -> "BM25Index":
        """Lexical index, built once on first use (~1s for 18.7k documents).

        Dense similarity on this corpus is compressed and blurs the exact
        terminology that Arabic religious text turns on; BM25 recovers it.
        """
        if self._bm25 is None:
            with self._bm25_lock:
                if self._bm25 is None:
                    from .lexical import BM25Index

                    docs = (
                        self.meta["question"].astype(str)
                        + " "
                        + self.meta["title"].astype(str)
                        + " "
                        + self.meta["answer"].astype(str)
                    ).tolist()
                    self._bm25 = BM25Index(docs)
        return self._bm25

    def _rank(self, question: str, depth: int) -> Tuple[List[int], np.ndarray]:
        """Return fused candidate row indices (best first) and the dense scores."""
        query_vec = self.embed(question)[0]
        dense = score_fatwas(
            query_vec, self.question_emb, self.answer_emb, self.answer_parent
        )

        depth = max(1, min(depth, len(dense)))
        part = np.argpartition(-dense, depth - 1)[:depth]
        dense_ranked = [int(i) for i in part[np.argsort(-dense[part])]]

        if not USE_HYBRID:
            return dense_ranked, dense

        lexical_ranked = [i for i, _ in self.bm25.search(question, top_k=depth)]
        fused = reciprocal_rank_fusion(
            [dense_ranked, lexical_ranked], weights=[DENSE_WEIGHT, LEXICAL_WEIGHT]
        )
        return fused, dense

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        mark_activity()

        candidates, dense = self._rank(question, depth=CANDIDATE_DEPTH)
        top_k = max(1, min(top_k, len(dense)))

        if USE_MMR and len(candidates) > top_k:
            pool = candidates[: min(len(candidates), MMR_POOL)]
            idx_sorted = mmr_select(
                pool,
                self.question_emb[pool].astype(np.float32),
                {i: float(dense[i]) for i in pool},
                top_n=top_k,
                lam=MMR_LAMBDA,
            )
        else:
            idx_sorted = candidates[:top_k]

        scores = dense
        results = []
        for idx in idx_sorted:
            row = self.meta.iloc[int(idx)]
            cats = row.get("categories")
            results.append(
                {
                    "id": int(row.get("id")),
                    "question": str(row.get("question", "")),
                    "title": str(row.get("title", "")),
                    "answer": str(row.get("answer", "")),
                    "link": str(row.get("link", "")),
                    "categories": [str(c) for c in cats] if cats is not None else [],
                    "similarity": float(scores[idx]),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Singleton + idle model unloading
# ---------------------------------------------------------------------------

_instance: Optional[FatwaRetriever] = None
_instance_lock = threading.Lock()
_last_activity = time.time()
_activity_lock = threading.Lock()


def mark_activity() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def idle_seconds() -> float:
    with _activity_lock:
        return time.time() - _last_activity


def get_retriever() -> FatwaRetriever:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = FatwaRetriever()
    return _instance


def unload_idle_model(timeout: float) -> None:
    """Unload the BGE-M3 model if the service has been idle for `timeout` seconds."""
    if _instance is not None and _instance.model_loaded and idle_seconds() >= timeout:
        if _instance.unload_model():
            print(
                f"[idle] Unloaded BGE-M3 after {int(idle_seconds())}s idle; "
                "RAM released to OS."
            )


def start_idle_monitor(timeout: int = 600, interval: int = 60) -> None:
    """Background daemon that frees the embedding model when idle.
    timeout<=0 disables it."""
    if timeout <= 0:
        print("[idle] Idle model unloading disabled (MODEL_IDLE_TIMEOUT<=0).")
        return

    def _loop():
        while True:
            time.sleep(interval)
            try:
                unload_idle_model(timeout)
            except Exception as e:  # pragma: no cover - defensive
                print(f"[idle] monitor error: {e}")

    threading.Thread(target=_loop, daemon=True, name="idle-monitor").start()
    print(f"[idle] Idle monitor started (timeout={timeout}s, interval={interval}s).")
