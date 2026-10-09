"""API routes — chat, reindex (the Sync button), status, config, trace."""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents import workflow
from app.core import config
from app.core.logging import logger
from app.rag import ingest, vectorstore
from app.services import audit
from app.tools import glean

router = APIRouter(prefix="/api")

# Reindex runs in a background thread; status is polled from the UI.
_reindex_lock = threading.Lock()
_reindex_state: dict = {"running": False, "last": None, "error": None}


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    t0 = time.time()
    try:
        result = workflow.run(req.question)
    except Exception as exc:
        logger.exception("chat: workflow failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = int((time.time() - t0) * 1000)
    result["latency_ms"] = latency_ms
    result["query_id"] = audit.log_query(result, latency_ms)
    return result


@router.get("/trace/{query_id}")
def trace(query_id: int) -> dict:
    row = audit.get_trace(query_id)
    if row is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return row


@router.post("/reindex")
def reindex() -> dict:
    """Sync knowledge base → embeddings. Runs in background; poll /api/status."""
    if not _reindex_lock.acquire(blocking=False):
        return {"running": True, "message": "Reindex already in progress"}
    _reindex_state.update({"running": True, "error": None})

    def _job():
        try:
            _reindex_state["last"] = ingest.run_incremental(do_fetch=True)
        except Exception as exc:
            logger.exception("reindex failed")
            _reindex_state["error"] = str(exc)
        finally:
            _reindex_state["running"] = False
            _reindex_lock.release()

    threading.Thread(target=_job, daemon=True).start()
    return {"running": True, "message": "Syncing knowledge base → embeddings…"}


@router.get("/status")
def status() -> dict:
    return {
        "chunks": vectorstore.count(),
        "kb_path": str(config.KB_LOCAL_PATH),
        "kb_exists": config.KB_LOCAL_PATH.exists(),
        "glean_configured": glean.is_configured(),
        "llm_model": config.LLM_MODEL,
        "models": {
            "writer": config.WRITER_MODEL,
            "reviewer": config.REVIEWER_MODEL,
            "reviser": config.REVISER_MODEL,
        },
        "reindex": _reindex_state,
        "audit": audit.stats(),
    }


@router.get("/config")
def get_config() -> dict:
    """Non-secret runtime info for the UI header."""
    return {
        "provider": "Portkey gateway" if config._BASE_URL else "direct",
        "model": config.LLM_MODEL,
        "embedder": vectorstore.EMBED_MODEL,
        "vectorstore": "Chroma (local, persistent)",
        "glean": "cookie-based (no API key)" if glean.is_configured() else "not configured",
        "max_revisions": config.MAX_REVISIONS,
    }
