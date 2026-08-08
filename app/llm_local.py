"""Local GGUF backend via llama.cpp — no hosted API, no per-question cost.

Why this exists: the hosted path burns two Inference calls per question (gate +
generation) and a depleted quota takes the whole app down. Running locally
removes that dependency entirely.

Model choice was benchmarked on the two jobs this app actually needs — structured
JSON gate verdicts and grounded Arabic generation with ``[n]`` citations — rather
than on generic leaderboards:

    Falcon-H1-3B-Instruct   gate 3/3 correct, 100% Arabic, cites, 22 tok/s, 1.9GB
    Qwen2.5-3B-Instruct     gate 2/3 correct, 100% Arabic, cites, 27 tok/s, 2.1GB
    ALLaM-7B-Instruct       gate 2/3 correct, 100% Arabic, cites, 12 tok/s, 4.3GB

Falcon-H1 (TII) is the default: it got every gate verdict right, where Qwen
wrongly abstained on an answerable question and flagged a sound premise as
false. ALLaM (SDAIA) is Arabic-native and more than twice the size, yet scored
no better on the gate, ran at half the speed, and echoed the source blocks
verbatim rather than writing an answer — so bigger and Arabic-specific did not
win here. Any of the three can be selected via LOCAL_LLM_REPO/LOCAL_LLM_FILE.

Memory: this model (~1.9GB) plus BGE-M3 (~2.3GB) exceeds a 4GB box. Set
LOCAL_UNLOAD_EMBEDDER=1 to drop the embedder after retrieval and before
generation; it reloads on the next question.
"""

import os
import threading
from typing import Any, Dict, Iterator, List, Optional

LOCAL_LLM_REPO = os.getenv("LOCAL_LLM_REPO", "tiiuae/Falcon-H1-3B-Instruct-GGUF")
LOCAL_LLM_FILE = os.getenv("LOCAL_LLM_FILE", "Falcon-H1-3B-Instruct-Q4_K_M.gguf")
LOCAL_LLM_CTX = int(os.getenv("LOCAL_LLM_CTX", "4096"))
LOCAL_LLM_THREADS = int(os.getenv("LOCAL_LLM_THREADS", str(os.cpu_count() or 4)))

# CPU generation is slow and PREFILL dominates: measured in Docker, the same
# model runs ~4 tok/s on a short prompt but ~1.2 tok/s once ~1600 tokens of
# fatwa text are prepended. So the local path gets a tighter prompt and a
# smaller answer budget than the hosted model — an answer that arrives is worth
# more than a longer one that times out.
LOCAL_CONTEXT_CHARS = int(os.getenv("LOCAL_CONTEXT_CHARS", "450"))
LOCAL_MAX_SOURCES = int(os.getenv("LOCAL_MAX_SOURCES", "3"))
LOCAL_MAX_TOKENS = int(os.getenv("LOCAL_MAX_TOKENS", "350"))
# Free the embedder before generating. Needed on ~4GB hosts where the embedder
# and the LLM cannot both be resident.
UNLOAD_EMBEDDER = os.getenv("LOCAL_UNLOAD_EMBEDDER", "0") == "1"

_llm = None
_lock = threading.Lock()


def _resolve_model_path() -> str:
    """Return a local GGUF path, downloading from the Hub on first use.

    Weight downloads are free — only the Inference API is metered — so this
    works even when the hosted quota is exhausted.
    """
    explicit = os.getenv("LOCAL_LLM_PATH")
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"LOCAL_LLM_PATH does not exist: {explicit}")
        return explicit

    from huggingface_hub import hf_hub_download

    return hf_hub_download(LOCAL_LLM_REPO, LOCAL_LLM_FILE)


def get_llm():
    """Load the GGUF model once and keep it resident."""
    global _llm
    if _llm is None:
        with _lock:
            if _llm is None:
                from llama_cpp import Llama

                _llm = Llama(
                    model_path=_resolve_model_path(),
                    n_ctx=LOCAL_LLM_CTX,
                    n_threads=LOCAL_LLM_THREADS,
                    verbose=False,
                )
    return _llm


def unload_llm() -> bool:
    """Drop the model. Returns True if it had been loaded."""
    global _llm
    with _lock:
        if _llm is None:
            return False
        _llm = None
    import gc

    gc.collect()
    return True


def _free_embedder_if_requested() -> None:
    if not UNLOAD_EMBEDDER:
        return
    try:
        from .retrieval import _instance

        if _instance is not None and _instance.model_loaded:
            _instance.unload_model()
    except Exception:  # never let a memory optimisation break a request
        pass


def _messages(system: str, user: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chat_completion(
    system: str,
    user: str,
    model: Optional[str] = None,  # accepted for interface parity; unused locally
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Blocking completion. Used by the gate."""
    result = get_llm().create_chat_completion(
        messages=_messages(system, user),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (result["choices"][0]["message"]["content"] or "").strip()


def stream_completion(
    system: str,
    user: str,
    max_tokens: int = 900,
    temperature: float = 0.1,
) -> Iterator[str]:
    """Yield text deltas so the SSE endpoint streams exactly as it does for the
    hosted backend."""
    _free_embedder_if_requested()
    stream = get_llm().create_chat_completion(
        messages=_messages(system, user),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError):
            delta = None
        if delta:
            yield delta


def describe() -> Dict[str, Any]:
    """Backend identity, surfaced by /health for operators."""
    return {
        "backend": "local",
        "repo": LOCAL_LLM_REPO,
        "file": LOCAL_LLM_FILE,
        "ctx": LOCAL_LLM_CTX,
        "threads": LOCAL_LLM_THREADS,
        "loaded": _llm is not None,
    }
