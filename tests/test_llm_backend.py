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


# ------------------------------ the default is "both" ------------------------


def test_default_backend_is_both(monkeypatch):
    """Unset LLM_BACKEND must give API-with-local-fallback, not API-only, so a
    depleted quota degrades instead of taking the app down."""
    import importlib

    monkeypatch.delenv("LLM_BACKEND", raising=False)
    from app import llm

    reloaded = importlib.reload(llm)
    try:
        assert reloaded.LLM_BACKEND == "both"
        assert reloaded._USE_API is True
        assert reloaded._USE_LOCAL is True
        assert reloaded._FALLBACK is True
    finally:
        importlib.reload(reloaded)


def test_explicit_backend_still_overrides_the_default(monkeypatch):
    import importlib

    monkeypatch.setenv("LLM_BACKEND", "api")
    from app import llm

    reloaded = importlib.reload(llm)
    try:
        assert reloaded.LLM_BACKEND == "api"
        assert reloaded._FALLBACK is False
    finally:
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        importlib.reload(reloaded)


def test_backend_value_is_case_and_space_insensitive(monkeypatch):
    import importlib

    monkeypatch.setenv("LLM_BACKEND", "  LOCAL  ")
    from app import llm

    reloaded = importlib.reload(llm)
    try:
        assert reloaded.LLM_BACKEND == "local"
        assert reloaded._USE_API is False
    finally:
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        importlib.reload(reloaded)


# --------------- local prompt budget: prefill is the CPU bottleneck ----------


SRC = [
    {"id": i, "question": f"سؤال {i}", "title": f"ع{i}", "answer": "ن " * 400}
    for i in range(1, 9)
]


def test_local_prompt_is_smaller_than_the_api_prompt(fake_llama, monkeypatch):
    """Measured: the same local model runs ~4 tok/s on a short prompt but
    ~1.2 tok/s with ~1600 tokens of fatwa text prepended. The local path must
    therefore send less, not the same."""
    llm = _set_mode(monkeypatch, "local")
    monkeypatch.setattr(llm, "LOCAL_ALLOW_RULINGS", True)
    list(llm.stream_answer("س", SRC, "derived"))

    sent = fake_llama["chat"][-1]["messages"][-1]["content"]
    api_sized = llm.build_user_prompt("س", SRC, "derived")
    assert len(sent) < len(api_sized)


def test_local_prompt_caps_the_number_of_sources(fake_llama, monkeypatch):
    from app import llm_local

    llm = _set_mode(monkeypatch, "local")
    monkeypatch.setattr(llm, "LOCAL_ALLOW_RULINGS", True)
    list(llm.stream_answer("س", SRC, "derived"))
    sent = fake_llama["chat"][-1]["messages"][-1]["content"]
    assert f"[{llm_local.LOCAL_MAX_SOURCES}]" in sent
    assert f"[{llm_local.LOCAL_MAX_SOURCES + 1}]" not in sent


def test_local_uses_its_own_answer_budget(fake_llama, monkeypatch):
    from app import llm_local

    llm = _set_mode(monkeypatch, "local")
    monkeypatch.setattr(llm, "LOCAL_ALLOW_RULINGS", True)
    list(llm.stream_answer("س", SRC, "derived"))
    assert fake_llama["chat"][-1]["max_tokens"] == llm_local.LOCAL_MAX_TOKENS


def test_fallback_also_uses_the_compact_local_prompt(fake_llama, monkeypatch):
    """The fallback path must get the same budget as local-only, otherwise an
    outage produces an answer that never finishes."""
    llm = _set_mode(monkeypatch, "both")
    monkeypatch.setattr(llm, "LOCAL_ALLOW_RULINGS", True)
    _break_api(monkeypatch, llm)
    list(llm.stream_answer("س", SRC, "derived"))
    sent = fake_llama["chat"][-1]["messages"][-1]["content"]
    assert len(sent) < len(llm.build_user_prompt("س", SRC, "derived"))


def test_api_prompt_keeps_the_full_budget(fake_llama, monkeypatch):
    """Only the local model is constrained; the hosted one gets full context."""
    llm = _set_mode(monkeypatch, "api")
    full = llm.build_user_prompt("س", SRC, "derived")
    assert "[8]" in full


# ------------- the local model must never assert a ruling --------------------


def test_local_does_not_generate_a_ruling(fake_llama, monkeypatch):
    """Measured on the real app: the 3B model answered «هل اقدر اصلي العشاء ٥
    ركعات؟» with "permissible" — a fabricated ruling on an obligatory prayer,
    carrying valid citations. The model must not be asked at all."""
    llm = _set_mode(monkeypatch, "local")
    out = "".join(llm.stream_answer("س", SRC[:2], "derived"))
    assert fake_llama["chat"] == []          # model never invoked
    assert "تعذّر الوصول إلى النموذج الأساسي" in out


def test_local_refusal_still_lists_the_nearest_fatwas(fake_llama, monkeypatch):
    """Refusing to rule must not mean refusing to help — the user still gets
    the retrieved fatwas and their links."""
    llm = _set_mode(monkeypatch, "local")
    src = [{"id": 1, "title": "حكم الأغاني", "link": "https://binbaz.org.sa/fatwas/1",
            "question": "س", "answer": "ج"}]
    out = "".join(llm.stream_answer("س", src, "derived"))
    assert "حكم الأغاني" in out
    assert "https://binbaz.org.sa/fatwas/1" in out


def test_direct_verdict_is_blocked_locally_too(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "local")
    out = "".join(llm.stream_answer("س", SRC[:1], "direct"))
    assert fake_llama["chat"] == []
    assert "لن يصدر" in out


def test_fallback_to_local_also_refuses_to_rule(fake_llama, monkeypatch):
    """An API outage must not silently downgrade to fabricated fatwas."""
    llm = _set_mode(monkeypatch, "both")
    _break_api(monkeypatch, llm)
    out = "".join(llm.stream_answer("س", SRC[:2], "derived"))
    assert fake_llama["chat"] == []
    assert "تعذّر الوصول" in out


def test_abstain_may_still_be_written_locally(fake_llama, monkeypatch):
    """Abstentions assert no ruling, so the local model is safe to use there."""
    llm = _set_mode(monkeypatch, "local")
    out = "".join(llm.stream_answer("س", [], "abstain", "افتراض مستحيل"))
    assert fake_llama["chat"] != []
    assert "الحكم" in out


def test_override_re_enables_local_rulings(fake_llama, monkeypatch):
    llm = _set_mode(monkeypatch, "local")
    monkeypatch.setattr(llm, "LOCAL_ALLOW_RULINGS", True)
    list(llm.stream_answer("س", SRC[:1], "derived"))
    assert fake_llama["chat"] != []
