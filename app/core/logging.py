"""Logging setup — concise console logs with node/agent context."""
from __future__ import annotations

import logging
import sys


class _SafeFormatter(logging.Formatter):
    """Formatter that survives console codepages without the arrow chars."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            return super().format(record)
        except UnicodeEncodeError:
            msg = record.getMessage()
            record.message = msg.encode("ascii", "replace").decode("ascii")
            record.msg = record.message
            record.args = ()
            return super().format(record)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("kb_copilot")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _SafeFormatter("%(asctime)s  %(levelname)-7s %(name)s  %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = setup_logging()
