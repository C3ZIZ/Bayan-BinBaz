"""Structured query logging.

The eval sets in eval/datasets are hand-built, which caps how well they can
represent real traffic. Logging every question with the verdict it received
turns production into a source of eval data: the false abstentions and the
wrongly-answered questions are exactly the rows worth adding.

Only the question text, the verdict, and timings are recorded — no client
identity — because the questions people ask a fatwa service are sensitive.
Logging is opt-in via QUERY_LOG_PATH.
"""

import json
import os

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from .env import env_bool, env_float, env_int, env_opt, env_str

QUERY_LOG_PATH = env_str("QUERY_LOG_PATH", "")

_lock = threading.Lock()


def log_query(
    question: str,
    verdict: str,
    premise_sound: bool,
    cited_ids: List[int],
    top_similarity: float,
    latency_ms: float,
    gate_failed_closed: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one JSONL record. Never raises — observability must not be able to
    take down the request path."""
    if not QUERY_LOG_PATH:
        return

    record = {
        "ts": time.time(),
        "question": question,
        "verdict": verdict,
        "premise_sound": premise_sound,
        "n_citations": len(cited_ids),
        "top_similarity": round(float(top_similarity), 4),
        "latency_ms": round(float(latency_ms), 1),
        "gate_failed_closed": gate_failed_closed,
    }
    if extra:
        record.update(extra)

    try:
        path = Path(QUERY_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
