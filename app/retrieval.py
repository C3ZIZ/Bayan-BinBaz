from functools import lru_cache # store retriever
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel


EMB_PATH = Path("data/index/fatwas_embeddings.npy")
META_PATH = Path("data/index/fatwas_meta.parquet")


class FatwaRetriever:
    def __init__(self):
        if not EMB_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Embeddings or meta files not found!"
            )

        # Load embeddings matrix: (N, D)
        self.embeddings = np.load(EMB_PATH)
        self.meta = pd.read_parquet(META_PATH)

        if len(self.embeddings) != len(self.meta):
            raise RuntimeError(
                f"Embeddings rows ({len(self.embeddings)}) not as meta rows ({len(self.meta)})"
            )

        # Load BGE-M3 model
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed(self, text: str) -> np.ndarray:
        outputs = self.model.encode(
            [text],
            batch_size=1,
            max_length=512,
        )
        vecs = outputs["dense_vecs"].astype("float32")
        # normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        return vecs  # shape: (1, D)

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.embed(question)  # (1, D)
        query_vec = query_vec[0]          # (D,)

        # cosine similarity = dot (embeddings (N,D) · query(D,))
        scores = self.embeddings @ query_vec  # shape: (N,)

        # If N < top_k fix it
        top_k = min(top_k, len(scores))
        # Get indices of top_k highest scores
        idx_part = np.argpartition(-scores, top_k - 1)[:top_k]
        # sort descending
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


@lru_cache(maxsize=1)
def get_retriever() -> FatwaRetriever:
    return FatwaRetriever()
