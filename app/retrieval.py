import os
import gc
import time
import ctypes
import platform
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel


EMB_PATH = Path("data/index/fatwas_embeddings.npy")
META_PATH = Path("data/index/fatwas_meta.parquet")

# Embedding model config (overridable via env for CPU/GPU tuning).
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
# fp16 helps on GPU but is slow/unstable on CPU-only hosts — keep it off by default.
EMB_USE_FP16 = os.getenv("EMB_USE_FP16", "0") == "1"


def _malloc_trim() -> None:
    """Best-effort: return freed heap arenas to the OS (glibc/Linux only)."""
    try:
        if platform.system() == "Linux":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class FatwaRetriever:
    """
    Holds the (cheap) prebuilt index in memory permanently, and the (expensive)
    BGE-M3 model only while it's needed. The model loads lazily on the first
    query and can be unloaded when the service is idle to free RAM.
    """

    def __init__(self):
        if not EMB_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Embeddings or meta files not found. Run build_index.py first."
            )

        # Load embeddings matrix (N, D) — normalized. ~80MB, kept resident.
        self.embeddings = np.load(EMB_PATH)
        self.meta = pd.read_parquet(META_PATH)

        if len(self.embeddings) != len(self.meta):
            raise RuntimeError(
                f"Embeddings rows ({len(self.embeddings)}) != meta rows ({len(self.meta)})"
            )

        self._model: Optional[BGEM3FlagModel] = None
        self._model_lock = threading.Lock()

    # --- model lifecycle -----------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> BGEM3FlagModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = BGEM3FlagModel(EMB_MODEL_NAME, use_fp16=EMB_USE_FP16)
        return self._model

    def unload_model(self) -> bool:
        """Drop the BGE-M3 model and return freed memory to the OS. Returns True if it was loaded."""
        with self._model_lock:
            if self._model is None:
                return False
            self._model = None
        gc.collect()
        _malloc_trim()
        return True

    # --- search --------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        model = self._ensure_model()
        outputs = model.encode(
            [text],
            batch_size=1,
            max_length=512,
        )
        vecs = outputs["dense_vecs"].astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        return vecs  # shape: (1, D)

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        mark_activity()
        query_vec = self.embed(question)[0]  # (D,)

        scores = self.embeddings @ query_vec  # (N,)

        top_k = min(top_k, len(scores))
        idx_part = np.argpartition(-scores, top_k - 1)[:top_k]
        idx_sorted = idx_part[np.argsort(-scores[idx_part])]

        results = []
        for idx in idx_sorted:
            score = float(scores[idx])
            row = self.meta.iloc[int(idx)]
            results.append(
                {
                    "id": int(row.get("id")),
                    "question": str(row.get("question", "")),
                    "title": str(row.get("title", "")),
                    "answer": str(row.get("answer", "")),
                    "link": str(row.get("link", "")),
                    "categories": row.get("categories"),
                    "similarity": score,
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
            print(f"[idle] Unloaded BGE-M3 after {int(idle_seconds())}s idle; RAM released to OS.")


def start_idle_monitor(timeout: int = 600, interval: int = 60) -> None:
    """Background daemon that frees the embedding model when idle. timeout<=0 disables it."""
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
