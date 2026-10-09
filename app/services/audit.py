"""SQLite audit log — one row per chat query with its decision path."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from app.core.config import AUDIT_DB_PATH
from app.core.logging import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    question_type TEXT,
    used_glean INTEGER,
    revisions INTEGER,
    final_decision TEXT,
    latency_ms INTEGER,
    answer TEXT,
    events_json TEXT
)
"""


def _conn() -> sqlite3.Connection:
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(AUDIT_DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_query(result: dict, latency_ms: int) -> int:
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO query_log (ts, question, question_type, used_glean, "
                "revisions, final_decision, latency_ms, answer, events_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    result["question"],
                    result.get("question_type", ""),
                    1 if result.get("used_glean") else 0,
                    result.get("revisions", 0),
                    result.get("final_decision", ""),
                    latency_ms,
                    result.get("final_answer", "")[:8000],
                    json.dumps(result.get("events", []), ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)
    except Exception as exc:
        logger.warning("audit: failed to log (%s)", exc)
        return 0


def get_trace(query_id: int) -> dict | None:
    with _conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM query_log WHERE id = ?", (query_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    try:
        d["events"] = json.loads(d.pop("events_json") or "[]")
    except json.JSONDecodeError:
        d["events"] = []
    return d


def stats() -> dict:
    with _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        glean = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE used_glean = 1"
        ).fetchone()[0]
        revised = conn.execute(
            "SELECT COUNT(*) FROM query_log WHERE revisions > 0"
        ).fetchone()[0]
    return {"total_queries": total, "glean_fallbacks": glean, "revised_answers": revised}
