"""Smoke test 1 — LLM via Portkey gateway. Verifies the model id and connectivity."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.core import config


def main() -> int:
    client = config.get_llm_client()
    print(f"base_url: {config._BASE_URL}")
    print(f"model:    {config.LLM_MODEL}")
    print("headers:  ", list(config._CUSTOM_HEADERS.keys()))

    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": "Reply with exactly: PORTKEY-OK"}],
    )
    text = resp.content[0].text if resp.content else ""
    print(f"reply:    {text!r}")
    print("SMOKE-LLM: PASS" if "PORTKEY-OK" in text else "SMOKE-LLM: FAIL")
    return 0 if "PORTKEY-OK" in text else 1


if __name__ == "__main__":
    sys.exit(main())
