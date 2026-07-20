#!/bin/sh
set -eu

# The BGE-M3 index ships prebuilt inside the image (data/index/*). Rebuild it
# ONLY if it's missing — e.g. someone mounted an empty volume over data/.
# Rebuilding downloads BGE-M3 and re-encodes ~20k fatwas (slow on CPU).
if [ ! -f data/index/fatwas_embeddings.npy ] || [ ! -f data/index/fatwas_meta.parquet ]; then
  echo "[entrypoint] Prebuilt index missing — building it (downloads BGE-M3, slow)…"
  if [ ! -f data/processed/fatwas.parquet ]; then
    if [ -f data/raw/Full_BinBaz_Data.csv ]; then
      python prepare_data.py
    else
      echo "[entrypoint] ERROR: no index, no processed parquet, and no data/raw CSV to build from." >&2
      exit 1
    fi
  fi
  python build_index.py
fi

exec "$@"
