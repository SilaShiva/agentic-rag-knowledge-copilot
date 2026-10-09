"""FastAPI app entry — serves the UI and mounts /api routes."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import LLM_MODEL
from app.core.logging import logger

BASE_DIR = Path(__file__).parents[1]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Agentic RAG KB Copilot starting — model=%s", LLM_MODEL)
    yield


app = FastAPI(title="Agentic RAG KB Copilot", version="1.0.0", lifespan=_lifespan)
app.include_router(router)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
