"""Environment parsing that treats an empty value as absent.

Docker Compose has no way to conditionally omit a variable, so an optional
setting is written ``FOO: ${FOO:-}`` and arrives as the empty string. Plain
``int(os.getenv("FOO", "8"))`` then raises ``ValueError: invalid literal for
int() with base 10: ''`` — which is exactly how /health started returning a 500
after the compose file gained its optional passthroughs.

Every env read goes through here so that class of failure cannot recur.
"""

import os
from typing import Optional

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_str(name: str, default: str = "") -> str:
    """Value, or ``default`` when unset OR empty."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_opt(name: str) -> Optional[str]:
    """Value, or None when unset or empty."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def env_int(name: str, default: int) -> int:
    raw = env_opt(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[env] {name}={raw!r} is not an integer; using {default}")
        return default


def env_float(name: str, default: float) -> float:
    raw = env_opt(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[env] {name}={raw!r} is not a number; using {default}")
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = env_opt(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    print(f"[env] {name}={raw!r} is not a boolean; using {default}")
    return default
