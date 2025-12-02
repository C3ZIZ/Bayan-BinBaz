import numpy as np
import pandas as pd
from pathlib import Path

from FlagEmbedding import BGEM3FlagModel # might be the best model with our case?


PROCESSED_PATH = Path("data/processed/fatwas.parquet")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = INDEX_DIR / "fatwas_embeddings.npy"
META_PATH = INDEX_DIR / "fatwas_meta.parquet"


def main():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found."
        )

    df = pd.read_parquet(PROCESSED_PATH)
    texts = (df["question"].fillna("") + "\n" + df["answer"].fillna("")).tolist()

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    outputs = model.encode(
        texts,
        batch_size=32,
        max_length=512,
    )
    dense_vecs = outputs["dense_vecs"].astype("float32")

    # L2 normalization -> use dot product as cosine similarity
    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True) + 1e-12
    dense_vecs = dense_vecs / norms

    np.save(EMB_PATH, dense_vecs)
    print(f"Embeddings to {EMB_PATH}")

    
    meta_cols = ["id", "question", "title", "answer", "link", "categories"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    df_meta = df[meta_cols].copy()
    df_meta.to_parquet(META_PATH, index=False)
    print(f"Meta saved {META_PATH}")


if __name__ == "__main__":
    main()
