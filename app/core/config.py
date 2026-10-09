"""
Central configuration — Portkey-proxied LLM client + env settings.

All LLM calls
go through get_llm_client() so Portkey headers / rate limiting are applied
transparently. Agents never construct a client themselves.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

# ── Portkey / LLM ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
_BASE_URL: str | None = os.getenv("ANTHROPIC_BASE_URL")

LLM_MODEL: str = os.getenv("LLM_MODEL", "")
WRITER_MODEL: str = os.getenv("WRITER_MODEL", LLM_MODEL)
REVIEWER_MODEL: str = os.getenv("REVIEWER_MODEL", LLM_MODEL)
REVISER_MODEL: str = os.getenv("REVISER_MODEL", LLM_MODEL)


def _parse_custom_headers(raw: str | None) -> dict[str, str]:
    """Parse 'Key: Value' header pairs (handles real newlines and literal \\n)."""
    headers: dict[str, str] = {}
    if not raw:
        return headers
    normalised = raw.replace("\\n", "\n")
    for line in normalised.splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            headers[k.strip()] = v.strip()
    return headers


def _build_headers() -> dict[str, str]:
    """Portkey headers, Style A (single var) takes precedence over Style B."""
    raw = os.getenv("ANTHROPIC_CUSTOM_HEADERS")
    if raw:
        return _parse_custom_headers(raw)

    headers: dict[str, str] = {}
    pk_api_key = os.getenv("PORTKEY_API_KEY")
    pk_virtual_key = os.getenv("PORTKEY_VIRTUAL_KEY")
    pk_provider = os.getenv("PORTKEY_PROVIDER", "@aws-dev-us")
    if pk_api_key:
        headers["x-portkey-api-key"] = pk_api_key
    if pk_virtual_key:
        headers["x-portkey-virtual-key"] = pk_virtual_key
    if pk_provider and pk_api_key:
        headers["x-portkey-provider"] = pk_provider
    return headers


_CUSTOM_HEADERS: dict[str, str] = _build_headers()

# ── Rate limiter (gateway free tier is RPM-limited) ───────────────────────────
_RATE_LIMIT_LOCK = threading.Lock()
_LAST_CALL_TIME: float = 0.0
RATE_LIMIT_INTERVAL: float = float(os.getenv("RATE_LIMIT_SECONDS", "32"))


def rate_limit_wait() -> None:
    """Block until enough time has passed since the last LLM API call."""
    global _LAST_CALL_TIME
    with _RATE_LIMIT_LOCK:
        now = time.time()
        elapsed = now - _LAST_CALL_TIME
        if elapsed < RATE_LIMIT_INTERVAL:
            time.sleep(RATE_LIMIT_INTERVAL - elapsed)
        _LAST_CALL_TIME = time.time()


def get_llm_client() -> Any:
    """Return a configured Anthropic client (with Portkey proxy if configured)."""
    import anthropic
    import httpx2

    kwargs: dict[str, Any] = {
        "api_key": ANTHROPIC_API_KEY,
        "max_retries": 6,
    }
    timeout = httpx2.Timeout(connect=30.0, read=300.0, write=60.0, pool=10.0)
    if _BASE_URL:
        kwargs["base_url"] = _BASE_URL
        # Corporate proxy does SSL inspection with a self-signed chain
        kwargs["http_client"] = httpx2.Client(verify=False, timeout=timeout)
    else:
        kwargs["http_client"] = httpx2.Client(timeout=timeout)
    if _CUSTOM_HEADERS:
        kwargs["default_headers"] = _CUSTOM_HEADERS
    return anthropic.Anthropic(**kwargs)


# ── Glean ─────────────────────────────────────────────────────────────────────
GLEAN_SERVER_URL: str = os.getenv("GLEAN_SERVER_URL", "").rstrip("/")

# ── Knowledge base source ─────────────────────────────────────────────────────
KB_LOCAL_PATH: Path = Path(os.getenv("KB_LOCAL_PATH", ""))

# ── Workflow tuning ───────────────────────────────────────────────────────────
MAX_REVISIONS: int = int(os.getenv("MAX_REVISIONS", "3"))
MAX_RETRIEVAL_RETRIES: int = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "6"))
MIN_SCORE: float = float(os.getenv("MIN_SCORE", "0.3"))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = DATA_DIR / "index" / "chroma"
INDEX_STATE_PATH: Path = DATA_DIR / "index_state.json"
AUDIT_DB_PATH: Path = DATA_DIR / "audit.db"
