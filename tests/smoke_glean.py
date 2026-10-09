"""Smoke test 2 — Glean cookie auth + search (no API key)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.tools import glean


def main() -> int:
    print(f"configured: {glean.is_configured()}")
    if not glean.is_configured():
        print("SMOKE-GLEAN: FAIL (no cookies — close Chrome and restart, or set .env)")
        return 1
    results = glean.search("kb metric index", max_results=3)
    print(f"results:    {len(results)}")
    for r in results:
        print(f"  - {r['title'][:60]}  [{r['source']}]")
    print("SMOKE-GLEAN: PASS" if results else "SMOKE-GLEAN: PASS (0 results, but auth ok)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
