"""`.env.example` and `docker-compose.yml` must not drift from the code.

Both have drifted before: compose was missing 35 variables and still passed a
removed one, and .env.example carried a stale LLM_MAX_TOKENS. Config drift is
silent — the app runs, just not the way the docs say it does.
"""

import re
from pathlib import Path

import pytest

ENV_EXAMPLE = Path(".env.example")
COMPOSE = Path("docker-compose.yml")

# Consumed by libraries (huggingface_hub, OpenMP) rather than by our code, and
# variables that only one-off scripts read.
NOT_READ_BY_APP = {"HF_HOME", "OMP_NUM_THREADS", "PORT", "MEM_LIMIT"}
SCRIPT_ONLY = re.compile(r"^(BUILD_|EVAL_)")


def _code_defaults() -> dict:
    """Every env var the app reads through app/env.py, with its default."""
    out = {}
    pattern = re.compile(r'env_(int|float|str|bool)\(\s*"([A-Z_0-9]+)",\s*([^)]+)\)')
    for path in list(Path("app").glob("*.py")) + [Path("main.py")]:
        for kind, name, default in pattern.findall(path.read_text(encoding="utf-8")):
            default = default.strip().strip('"')
            if "env_str" in default or "os.cpu_count" in default:
                continue  # nested/computed default, nothing literal to compare
            if kind == "bool":
                default = "1" if default == "True" else "0"
            out.setdefault(name, default)
    return out


def _declared(path: Path) -> dict:
    """Active assignments only — what the file actually sets."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if path == ENV_EXAMPLE:
            m = re.match(r"^([A-Z_0-9]+)=(.*)$", line.strip())
        else:
            m = re.match(r"^\s+([A-Z_0-9]+):\s*(.*)$", line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def _mentioned(path: Path) -> set:
    """Names the file names at all, including commented-out optionals and
    alternative model choices — those count as documented."""
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*#?\s*([A-Z_0-9]{3,})=", text, re.MULTILINE))


@pytest.mark.skipif(not ENV_EXAMPLE.exists(), reason=".env.example missing")
def test_env_example_defaults_match_the_code():
    code, example = _code_defaults(), _declared(ENV_EXAMPLE)
    drift = {
        k: (example[k], v)
        for k, v in code.items()
        if k in example and example[k] != str(v)
    }
    assert not drift, f"stale defaults in .env.example: {drift}"


@pytest.mark.skipif(not ENV_EXAMPLE.exists(), reason=".env.example missing")
def test_env_example_documents_every_setting():
    documented = _mentioned(ENV_EXAMPLE)
    missing = {
        k for k in _code_defaults()
        if k not in documented and not SCRIPT_ONLY.match(k)
    }
    assert not missing, f"undocumented in .env.example: {sorted(missing)}"


@pytest.mark.skipif(not COMPOSE.exists(), reason="docker-compose.yml missing")
def test_compose_passes_every_runtime_setting():
    declared = _declared(COMPOSE)
    missing = {
        k for k in _code_defaults()
        if k not in declared and not SCRIPT_ONLY.match(k)
    }
    assert not missing, f"not passed through docker-compose.yml: {sorted(missing)}"


@pytest.mark.skipif(not COMPOSE.exists(), reason="docker-compose.yml missing")
def test_compose_passes_nothing_the_code_no_longer_reads():
    """LLM_MAX_HITS lingered here after being deleted from the code."""
    stale = {
        k for k in _declared(COMPOSE)
        if k not in _code_defaults() and k not in NOT_READ_BY_APP
        and k not in {"HF_TOKEN", "LOCAL_LLM_PATH", "LOCAL_LLM_THREADS",
                      "GATE_MODEL", "QUERY_LOG_PATH", "LLM_BACKEND"}
    }
    assert not stale, f"passed by compose but unread: {sorted(stale)}"


@pytest.mark.skipif(not COMPOSE.exists(), reason="docker-compose.yml missing")
def test_compose_does_not_make_hf_token_mandatory():
    """`${HF_TOKEN:?...}` aborted `docker compose up` without a token, even
    though the app now answers via the local fallback."""
    text = COMPOSE.read_text(encoding="utf-8")
    assert "HF_TOKEN:?" not in text and "HF_TOKEN:?" not in text.replace("${", "")
    assert re.search(r"HF_TOKEN:\s*\$\{HF_TOKEN:-", text)
