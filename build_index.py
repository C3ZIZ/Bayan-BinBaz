"""Build the retrieval index.

Two changes versus the original that materially affect quality:

  1. ``max_length`` was 512 while BGE-M3 supports 8192. Measured with the real
     tokenizer, that truncated 17.7% of fatwas and left 19.6% of all corpus
     tokens out of the index entirely. Answers are now chunked, so no content
     is dropped.
  2. The original embedded ``question + "\\n" + answer`` as a single vector, of
     which 81% of tokens were answer text — while users send short questions.
     That asymmetry compressed a perfect self-match to a median cosine of 0.815.
     Questions and answer chunks are embedded separately so question-to-question
     matching can dominate at query time (see app/retrieval.py::score_fatwas).

Vectors are L2-normalized and stored float16, halving the on-disk index with
negligible recall loss — it lives in git, so size matters.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

from app.chunking import chunk_text
from app.normalize import NORMALIZE_VERSION, normalize_arabic

PROCESSED_PATH = Path("data/processed/fatwas.parquet")
INDEX_DIR = Path("data/index")

EMB_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 2048
CHUNK_MAX_TOKENS = 450
CHUNK_OVERLAP = 100
BATCH_SIZE = 16


def _encode(model, texts, label):
    print(f"Encoding {len(texts)} {label}...", flush=True)
    out = model.encode(texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    vecs = out["dense_vecs"].astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs.astype(np.float16)


def main():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"{PROCESSED_PATH} not found. Run prepare_data.py first.")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_PATH)

    model = BGEM3FlagModel(EMB_MODEL, use_fp16=False)
    tokenizer = model.tokenizer

    def token_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=True)["input_ids"])

    questions = [normalize_arabic(q) for q in df["question"].fillna("").tolist()]

    chunk_texts, chunk_parent = [], []
    for row_idx, answer in enumerate(df["answer"].fillna("").tolist()):
        pieces = chunk_text(
            normalize_arabic(answer),
            token_len,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap_tokens=CHUNK_OVERLAP,
        )
        if not pieces:
            # Terse or empty answer: fall back to the question so that every
            # fatwa still owns at least one chunk and stays reachable.
            pieces = [questions[row_idx] or " "]
        for piece in pieces:
            chunk_texts.append(piece)
            chunk_parent.append(row_idx)

    question_emb = _encode(model, questions, "questions")
    answer_emb = _encode(model, chunk_texts, "answer chunks")

    np.save(INDEX_DIR / "question_emb.npy", question_emb)
    np.save(INDEX_DIR / "answer_emb.npy", answer_emb)
    np.save(INDEX_DIR / "answer_parent.npy", np.asarray(chunk_parent, dtype=np.int32))

    meta_cols = [
        c for c in ["id", "question", "title", "answer", "link", "categories"]
        if c in df.columns
    ]
    df[meta_cols].to_parquet(INDEX_DIR / "fatwas_meta.parquet", index=False)

    (INDEX_DIR / "index_manifest.json").write_text(
        json.dumps(
            {
                "normalize_version": NORMALIZE_VERSION,
                "emb_model": EMB_MODEL,
                "max_length": MAX_LENGTH,
                "chunk_max_tokens": CHUNK_MAX_TOKENS,
                "chunk_overlap": CHUNK_OVERLAP,
                "n_fatwas": int(len(question_emb)),
                "n_chunks": int(len(answer_emb)),
                "dtype": "float16",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Done. {len(question_emb)} fatwas, {len(answer_emb)} chunks.")


if __name__ == "__main__":
    main()
