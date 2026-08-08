"""Empty env values must mean "unset".

Docker Compose cannot conditionally omit a variable, so optional settings are
written ``FOO: ${FOO:-}`` and arrive as "". Plain ``int(os.getenv(...))`` then
raises ValueError — which is exactly how /health started returning a 500 once
the compose file gained its optional passthroughs.
"""

import pytest

from app.env import env_bool, env_float, env_int, env_opt, env_str


def test_empty_string_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_THREADS", "")
    assert env_int("LOCAL_LLM_THREADS", 8) == 8


def test_whitespace_only_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("X", "   ")
    assert env_int("X", 5) == 5
    assert env_str("X", "d") == "d"
    assert env_float("X", 1.5) == 1.5
    assert env_bool("X", True) is True
    assert env_opt("X") is None


def test_unset_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("X", raising=False)
    assert env_int("X", 7) == 7
    assert env_str("X", "d") == "d"
    assert env_opt("X") is None


def test_real_values_are_used(monkeypatch):
    monkeypatch.setenv("X", "42")
    assert env_int("X", 7) == 42
    monkeypatch.setenv("Y", "0.25")
    assert env_float("Y", 1.0) == 0.25
    monkeypatch.setenv("Z", "hello")
    assert env_str("Z", "d") == "hello"


def test_values_are_stripped(monkeypatch):
    monkeypatch.setenv("X", "  42  ")
    assert env_int("X", 0) == 42
    monkeypatch.setenv("Y", "  text  ")
    assert env_str("Y", "") == "text"


def test_bool_accepts_common_spellings(monkeypatch):
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("B", truthy)
        assert env_bool("B", False) is True
    for falsy in ("0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv("B", falsy)
        assert env_bool("B", True) is False


def test_garbage_falls_back_rather_than_crashing(monkeypatch):
    """A typo in .env must not take the service down."""
    monkeypatch.setenv("X", "not-a-number")
    assert env_int("X", 9) == 9
    assert env_float("X", 1.5) == 1.5
    monkeypatch.setenv("B", "maybe")
    assert env_bool("B", True) is True


def test_zero_is_not_treated_as_empty(monkeypatch):
    monkeypatch.setenv("X", "0")
    assert env_int("X", 5) == 0
    assert env_bool("X", True) is False


def test_modules_import_with_every_optional_var_empty(monkeypatch):
    """The exact failure mode: compose passes every optional var as "" and the
    app must still start."""
    import importlib

    for name in (
        "LOCAL_LLM_THREADS", "LOCAL_LLM_PATH", "GATE_MODEL", "QUERY_LOG_PATH",
        "LLM_BACKEND", "RETRIEVAL_ALPHA", "RATE_LIMIT_REQUESTS", "LLM_MAX_TOKENS",
        "GATE_TEMPERATURE", "LOCAL_MAX_SOURCES", "CORS_ORIGINS", "EMB_USE_FP16",
    ):
        monkeypatch.setenv(name, "")

    for mod in ("app.llm_local", "app.gate", "app.llm", "app.observability"):
        importlib.reload(importlib.import_module(mod))

    from app import llm, llm_local

    assert isinstance(llm_local.LOCAL_LLM_THREADS, int)
    assert llm_local.LOCAL_LLM_THREADS > 0
    assert llm.LLM_BACKEND == "both"          # empty -> documented default


def test_describe_backend_survives_empty_optionals(monkeypatch):
    """This is what actually 500'd: /health calls describe_backend()."""
    import importlib

    monkeypatch.setenv("LOCAL_LLM_THREADS", "")
    monkeypatch.setenv("GATE_MODEL", "")
    importlib.reload(importlib.import_module("app.llm_local"))
    llm = importlib.reload(importlib.import_module("app.llm"))

    described = llm.describe_backend()
    assert described["backend"] == "both"
    assert described["local"]["threads"] > 0
