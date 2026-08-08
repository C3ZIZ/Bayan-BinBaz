"""Backend selection and the local GGUF path.

``llama_cpp`` is stubbed so these run without downloading a 1.9GB model.
"""

import sys
import types

import pytest


@pytest.fixture
def fake_llama(monkeypatch):
    """Minimal llama_cpp stand-in that records the calls it receives."""
    calls = {"init": None, "chat": []}

    class FakeLlama:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def create_chat_completion(
            self, messages, max_tokens, temperature, stream=False
        ):
            calls["chat"].append(
                {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": stream,
                }
            )
            if stream:
                return iter(
                    [
                        {"choices": [{"delta": {"content": "الحكم "}}]},
                        {"choices": [{"delta": {"content": "كذا [1]."}}]},
                        {"choices": [{"delta": {}}]},  # terminal chunk, no content
                    ]
                )
            return {"choices": [{"message": {"content": "  جواب [1].  "}}]}

    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)

    from app import llm_local

    monkeypatch.setattr(llm_local, "_llm", None)
    monkeypatch.setattr(llm_local, "_resolve_model_path", lambda: "/fake/model.gguf")
    return calls


def _set_mode(monkeypatch, mode):
    from app import llm

    monkeypatch.setattr(llm, "LLM_BACKEND", mode)
    monkeypatch.setattr(llm, "_USE_API", mode in ("api", "both"))
    monkeypatch.setattr(llm, "_USE_LOCAL", mode in ("local", "both"))
    monkeypatch.setattr(llm, "_FALLBACK", mode == "both")
    return llm


def test_local_chat_completion_returns_stripped_text(fake_llama):
    from app import llm_local

    assert llm_local.chat_completion("sys", "user") == "جواب [1]."


def test_local_streaming_yields_deltas_and_skips_empty_chunks(fake_llama):
    from app import llm_local

    assert list(llm_local.stream_completion("sys", "user")) == ["الحكم ", "كذا [1]."]


def test_local_model_is_loaded_once(fake_llama):
    from app import llm_local

    llm_local.chat_completion("s", "u")
    llm_local.chat_completion("s", "u")
    assert len(fake_llama["chat"]) == 2
    assert llm_local._llm is not None


def test_local_unload_releases_the_model(fake_llama):
    from app import llm_local

    llm_local.chat_completion("s", "u")
    assert llm_local.unload_llm() is True
    assert llm_local.unload_llm() is False


def test_local_passes_context_and_threads_to_llama(fake_llama):
    from app import llm_local

    llm_local.chat_completion("s", "u")
    assert fake_llama["init"]["n_ctx"] == llm_local.LOCAL_LLM_CTX
    assert fake_llama["init"]["n_threads"] == llm_local.LOCAL_LLM_THREADS


def test_gate_temperature_is_passed_through(fake_llama):
    from app import llm_local

    llm_local.chat_completion("s", "u", temperature=0.0, max_tokens=400)
    assert fake_llama["chat"][0]["temperature"] == 0.0
    assert fake_llama["chat"][0]["max_tokens"] == 400


def test_explicit_path_must_exist(monkeypatch):
    from app import llm_local

    monkeypatch.setenv("LOCAL_LLM_PATH", "/definitely/not/here.gguf")
    with pytest.raises(FileNotFoundError):
        llm_local._resolve_model_path()


def test_describe_reports_the_local_backend(fake_llama):
    from app import llm_local

    described = llm_local.describe()
    assert described["backend"] == "local"
    assert described["file"].endswith(".gguf")


def test_embedder_is_only_unloaded_when_requested(fake_llama, monkeypatch):
    """On a ~4GB host the embedder and the LLM cannot both be resident, but
    unloading costs a reload on the next question — so it must stay opt-in."""
    from app import llm_local

    freed = []

    class FakeRetriever:
        model_loaded = True

        def unload_model(self):
            freed.append(1)
            return True

    monkeypatch.setattr("app.retrieval._instance", FakeRetriever(), raising=False)

    monkeypatch.setattr(llm_local, "UNLOAD_EMBEDDER", False)
    list(llm_local.stream_completion("s", "u"))
    assert freed == []

    monkeypatch.setattr(llm_local, "UNLOAD_EMBEDDER", True)
    list(llm_local.stream_completion("s", "u"))
    assert freed == [1]


def test_llm_module_routes_to_local_when_selected(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "local")
    assert llm.chat_completion("sys", "user") == "جواب [1]."


def test_llm_module_streams_from_local_when_selected(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "local")
    out = "".join(llm.stream_answer("س", [], "abstain", "افتراض خاطئ"))
    assert "الحكم" in out


def test_describe_backend_reports_api_by_default(monkeypatch):
    llm = _set_mode(monkeypatch, "api")
    assert llm.describe_backend()["backend"] == "api"


def test_local_backend_needs_no_hf_token(fake_llama, monkeypatch):
    """The whole point: a depleted Inference quota must not take the app down."""
    llm = _set_mode(monkeypatch, "local")
    monkeypatch.setattr(llm, "HF_TOKEN", None)
    assert llm.chat_completion("sys", "user") == "جواب [1]."


# ------------------------- LLM_BACKEND=both: local as a FALLBACK --------------


def _break_api(monkeypatch, llm, exc=RuntimeError("402 Payment Required")):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(llm, "_api_chat", boom)
    monkeypatch.setattr(llm, "_api_stream", boom)


def test_both_prefers_the_api_when_it_works(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "both")
    monkeypatch.setattr(llm, "_api_chat", lambda *a, **k: "من الـ API")
    assert llm.chat_completion("s", "u") == "من الـ API"


def test_both_falls_back_to_local_when_the_api_fails(fake_llama, monkeypatch):
    """The whole point: a depleted quota degrades instead of failing."""
    llm = _set_mode(monkeypatch, "both")
    _break_api(monkeypatch, llm)
    assert llm.chat_completion("s", "u") == "جواب [1]."


def test_api_only_mode_does_not_fall_back(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "api")
    _break_api(monkeypatch, llm)
    with pytest.raises(RuntimeError):
        llm.chat_completion("s", "u")


def test_local_only_mode_never_calls_the_api(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "local")
    called = []
    monkeypatch.setattr(llm, "_api_chat", lambda *a, **k: called.append(1) or "api")
    assert llm.chat_completion("s", "u") == "جواب [1]."
    assert called == []


def test_both_streams_from_local_when_the_api_fails_before_any_token(
    fake_llama, monkeypatch
):
    llm = _set_mode(monkeypatch, "both")
    _break_api(monkeypatch, llm)
    out = "".join(llm.stream_answer("س", [], "abstain", "خلل"))
    assert "الحكم" in out


def test_both_does_not_splice_models_after_streaming_has_started(
    fake_llama, monkeypatch
):
    """Once text has reached the client it cannot be retracted; continuing from
    a different model would splice two answers into one."""
    llm = _set_mode(monkeypatch, "both")

    def half_then_fail(user_prompt):
        yield "بداية الجواب "
        raise RuntimeError("connection reset mid-stream")

    monkeypatch.setattr(llm, "_api_stream", half_then_fail)

    got = []
    with pytest.raises(RuntimeError):
        for part in llm.stream_answer("س", [], "derived"):
            got.append(part)
    assert got == ["بداية الجواب "]


def test_describe_backend_reports_fallback_mode(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "both")
    d = llm.describe_backend()
    assert d["backend"] == "both"
    assert d["fallback_to_local"] is True
    assert "api" in d and "local" in d
