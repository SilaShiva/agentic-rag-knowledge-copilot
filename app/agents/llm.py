"""
LLM call helper — single place that applies rate limiting + JSON extraction.
Every agent node goes through complete() or complete_json().
"""
from __future__ import annotations

import json
import re

from app.core.config import LLM_MODEL, get_llm_client, rate_limit_wait
from app.core.logging import logger


def complete(
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 1200,
) -> str:
    """Rate-limited LLM completion. Returns the text response."""
    client = get_llm_client()
    rate_limit_wait()
    resp = client.messages.create(
        model=model or LLM_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text if resp.content else ""
    logger.debug("llm: %d chars returned (model=%s)", len(text), model)
    return text


def complete_json(system: str, user: str, model: str | None = None, max_tokens: int = 600) -> dict:
    """complete() + tolerant JSON extraction (first {...} block in the reply)."""
    text = complete(system, user, model=model, max_tokens=max_tokens)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in LLM reply: {text[:200]!r}")
    return json.loads(match.group(0))
