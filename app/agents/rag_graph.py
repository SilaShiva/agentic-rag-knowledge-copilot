"""
Agentic RAG nodes — route / retrieve / grade / glean / rewrite.

Ported from the Enterprise-IT-Support-Agentic-RAG-Copilot pattern: the graph
controls *what to do next* from state instead of blindly generating. Every node
appends an event to state["events"] so the UI can render the decision path.
"""
from __future__ import annotations

import time

from app.agents.llm import complete, complete_json
from app.core.config import RETRIEVAL_K
from app.core.logging import logger
from app.rag import vectorstore
from app.tools import glean

ROUTE_SYSTEM = """You classify an incoming chat message for a knowledge copilot
over a governed analytics knowledge base (metrics, datasets, playbooks).
Reply ONLY with JSON: {"type": "greeting" | "question"}.
Use "greeting" for hellos, thanks, small talk, identity questions
("who are you"), or anything not asking for knowledge. Everything that wants an
answer about metrics, tables, definitions, processes or analysis → "question"."""


def route_node(state: dict) -> dict:
    t0 = time.time()
    q = state["question"]
    try:
        verdict = complete_json(ROUTE_SYSTEM, f"Message: {q!r}", max_tokens=60)
        qtype = verdict.get("type", "question")
    except Exception as exc:
        logger.warning("route: LLM failed (%s) — defaulting to question", exc)
        qtype = "question"
    state["question_type"] = qtype
    state["events"].append({
        "step": "route", "agent": "Router", "decision": qtype,
        "detail": "answered directly" if qtype == "greeting" else "enters RAG pipeline",
        "latency_ms": int((time.time() - t0) * 1000),
    })
    return state


RETRIEVE_SYSTEM = """You grade evidence for answering a question about KB
(analytics knowledge: metric definitions, dataset docs,
SQL playbooks). Evidence is GOOD only if it contains the substance needed to
answer correctly. Reply ONLY JSON: {"verdict": "GOOD" | "WEAK", "reason": "..."}"""


def _grade(question: str, evidence: str, state: dict, step: str, agent: str) -> str:
    t0 = time.time()
    try:
        verdict = complete_json(
            RETRIEVE_SYSTEM,
            f"Question: {question}\n\nEvidence:\n{evidence[:6000]}",
            max_tokens=120,
        )
        v = verdict.get("verdict", "WEAK").upper()
        v = v if v in ("GOOD", "WEAK") else "WEAK"
        reason = verdict.get("reason", "")
    except Exception as exc:
        logger.warning("grade: LLM failed (%s) — treating as WEAK", exc)
        v, reason = "WEAK", f"grader error: {exc}"
    state["events"].append({
        "step": step, "agent": agent, "decision": v, "detail": reason[:200],
        "latency_ms": int((time.time() - t0) * 1000),
    })
    return v


def retrieve_node(state: dict) -> dict:
    t0 = time.time()
    query = state.get("search_query") or state["question"]
    hits = vectorstore.search(query, k=RETRIEVAL_K)
    state["kb_hits"] = hits
    state["events"].append({
        "step": "retrieve", "agent": "Retriever",
        "decision": f"{len(hits)} chunks",
        "detail": ", ".join(f"{h['path']}@{h['branches'].split(',')[0]} ({h['score']})"
                            for h in hits[:3]) or "no matches in index",
        "latency_ms": int((time.time() - t0) * 1000),
    })
    if not hits:
        state["evidence_verdict"] = "WEAK"
        state["events"].append({
            "step": "grade", "agent": "Evidence Grader", "decision": "WEAK",
            "detail": "empty retrieval result", "latency_ms": 0,
        })
    return state


def grade_node(state: dict) -> dict:
    hits = state.get("kb_hits", [])
    if not hits:
        state["evidence_verdict"] = "WEAK"
        return state
    evidence = "\n\n---\n\n".join(
        f"[{h['path']} @ {h['branches']}]\n{h['text']}" for h in hits
    )
    state["evidence_verdict"] = _grade(
        state["question"], evidence, state, "grade", "Evidence Grader"
    )
    return state


def glean_node(state: dict) -> dict:
    """Backup search — cookie-based Glean (no API key)."""
    t0 = time.time()
    if not glean.is_configured():
        state["glean_results"] = []
        state["events"].append({
            "step": "glean", "agent": "Glean Backup", "decision": "SKIPPED",
            "detail": "Glean cookies not available (set GLEAN_SESSION_COOKIE or "
                      "close Chrome + restart for auto-extract)",
            "latency_ms": 0,
        })
        return state
    try:
        results = glean.search(state.get("search_query") or state["question"])
    except glean.GleanAuthError as exc:
        state["glean_results"] = []
        state["events"].append({
            "step": "glean", "agent": "Glean Backup", "decision": "ERROR",
            "detail": str(exc), "latency_ms": int((time.time() - t0) * 1000),
        })
        return state
    state["glean_results"] = results
    state["events"].append({
        "step": "glean", "agent": "Glean Backup", "decision": f"{len(results)} results",
        "detail": ", ".join(r["title"][:40] for r in results[:3]) or "nothing found",
        "latency_ms": int((time.time() - t0) * 1000),
    })
    if results:
        evidence = glean.format_results(results)
        state["glean_verdict"] = _grade(
            state["question"], evidence, state, "grade_web", "Web Evidence Grader"
        )
    else:
        state["glean_verdict"] = "WEAK"
    return state


REWRITE_SYSTEM = """You rewrite a search query to improve vector retrieval over
a corporate analytics knowledge base (metrics, datasets, SQL playbooks). Output
ONLY the rewritten query — no explanation, no quotes."""


def rewrite_node(state: dict) -> dict:
    t0 = time.time()
    retries = state.get("retrieval_retries", 0)
    try:
        rewritten = complete(
            REWRITE_SYSTEM,
            f"Original question: {state['question']}\n"
            f"Previous query that failed: {state.get('search_query') or state['question']}",
            max_tokens=80,
        ).strip().strip('"')
        state["search_query"] = rewritten or state["question"]
    except Exception as exc:
        logger.warning("rewrite: LLM failed (%s) — keeping original", exc)
    state["retrieval_retries"] = retries + 1
    state["events"].append({
        "step": "rewrite", "agent": "Query Rewriter", "decision": "RETRY",
        "detail": f"retry {state['retrieval_retries']}: {state['search_query']!r}",
        "latency_ms": int((time.time() - t0) * 1000),
    })
    return state
