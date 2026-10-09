"""
Ingestion — full + incremental (content-hash diff) over the kb clone.

index_state.json maps content_hash → {path, branches, ingested_at}. A reindex
fetches the remote, re-enumerates, then:
  - new hashes   → embed + upsert
  - known hashes → skip (identical content)
  - vanished     → delete chunks

This is what the UI "Sync knowledge base" button triggers, so a fresh commit on any
kb branch is reflected in the embeddings in seconds.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from app.core.config import INDEX_STATE_PATH
from app.core.logging import logger
from app.rag import kbreader, vectorstore


def _load_state() -> dict:
    if INDEX_STATE_PATH.exists():
        try:
            return json.loads(INDEX_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("ingest: index_state.json unreadable — starting fresh")
    return {}


def _save_state(state: dict) -> None:
    INDEX_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_incremental(do_fetch: bool = True) -> dict:
    """Sync knowledge base → vector index. Returns a summary for the UI."""
    started = time.time()
    fetched = kbreader.fetch_origin() if do_fetch else False

    records = kbreader.enumerate_knowledge()
    state = _load_state()

    known = set(state.keys())
    current = {r["content_hash"]: r for r in records}

    new_hashes = [h for h in current if h not in known]
    removed_hashes = [h for h in known if h not in current]
    unchanged = len(current) - len(new_hashes)

    # Metadata may have changed (e.g. file moved to more branches) — refresh
    # records whose branch list differs, without re-embedding where possible.
    metadata_updates: list[dict] = []
    for h, rec in current.items():
        old = state.get(h)
        if old and set(old.get("branches", [])) != set(rec["branches"]):
            metadata_updates.append(rec)

    if removed_hashes:
        vectorstore.delete_documents(removed_hashes)

    to_upsert = [current[h] for h in new_hashes] + metadata_updates
    added = vectorstore.upsert_documents(to_upsert)["chunks_added"] if to_upsert else 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for h, rec in current.items():
        state[h] = {"path": rec["path"], "branches": rec["branches"], "ingested_at": now}
    _save_state(state)

    summary = {
        "fetched": fetched,
        "branches": len(kbreader.list_branches()),
        "documents_total": len(records),
        "documents_unchanged": unchanged,
        "documents_new": len(new_hashes),
        "documents_removed": len(removed_hashes),
        "chunks_added": added,
        "duration_s": round(time.time() - started, 1),
        "synced_at": now,
    }
    logger.info("ingest: %s", summary)
    return summary


def run_full() -> dict:
    """Wipe state and rebuild from scratch (first run / manual reset)."""
    state = _load_state()
    if state:
        vectorstore.delete_documents(list(state.keys()))
        INDEX_STATE_PATH.unlink(missing_ok=True)
    return run_incremental(do_fetch=False)
