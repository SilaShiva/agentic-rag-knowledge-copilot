"""CLI: build / refresh the knowledge-base vector index.

  python ingest_cli.py            # incremental (uses index_state.json)
  python ingest_cli.py --full     # wipe + rebuild
"""
from __future__ import annotations

import argparse
import sys

from app.core.logging import logger
from app.rag import ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="knowledge-base index builder")
    parser.add_argument("--full", action="store_true", help="wipe and rebuild")
    parser.add_argument("--no-fetch", action="store_true", help="skip git fetch")
    args = parser.parse_args()

    summary = ingest.run_full() if args.full else ingest.run_incremental(
        do_fetch=not args.no_fetch
    )
    logger.info("done: %s", summary)
    print(repr(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
