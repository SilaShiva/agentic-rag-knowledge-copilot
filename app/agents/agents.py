"""
Writer / Reviewer / Reviser — self-correcting generation loop on top of RAG
(from the Self-Correcting-Multi-Agent-System reference). Reviewer output is a
structured {"decision": "PASS" | "REVISE", "feedback": str}.
"""
from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field, ValidationError

from app.agents.llm import complete
from app.core.logging import logger

WRITER_MODEL: str | None = None  # resolved from config at call time
REVIEWER_MODEL: str | None = None
REVISER_MODEL: str | None = None


class Review(BaseModel):
    decision: str = Field(pattern="^(PASS|REVISE)$")
    feedback: str = ""


def _models() -> tuple[str, str, str]:
    from app.core.config import LLM_MODEL, REVISER_MODEL, REVIEWER_MODEL, WRITER_MODEL

    return (
        WRITER_MODEL or LLM_MODEL,
        REVIEWER_MODEL or LLM_MODEL,
        REVISER_MODEL or LLM_MODEL,
    )


def _citations_block(hits: list[dict]) -> str:
    if not hits:
        return ""
    return "\n\n".join(
        f"[{i}] {h['path']} @ {h['branches']}\n{h['text']}" for i, h in enumerate(hits, 1)
    )


WRITER_SYSTEM = """You are the Writer in a RAG copilot over a governed analytics
knowledge base. Write a clear, grounded answer to the question
using ONLY the provided context. Cite sources inline as [1], [2] matching the
context blocks. If the context comes from Glean (external search), say so.
Be concise — short paragraphs or bullets. If context is insufficient, say
exactly what is missing instead of inventing facts."""


def writer_node(state: dict) -> dict:
    t0 = time.time()
    writer_model, _, _ = _models()
    hits = state.get("kb_hits", [])
    glean_results = state.get("glean_results", [])

    context_parts = []
    if hits:
        context_parts.append("Knowledge base context:\n" + _citations_block(hits))
    if glean_results:
        from app.tools import glean as glean_mod

        context_parts.append(
            "External Glean search results (mark these as external sources):\n"
            + glean_mod.format_results(glean_results)
        )
    context = "\n\n---\n\n".join(context_parts) or "No retrieved context available."

    state["context"] = context
    state["draft"] = complete(
        WRITER_SYSTEM,
        f"Question: {state['question']}\n\nContext:\n{context[:9000]}",
        model=writer_model,
    )
    state["revision_count"] = 0
    state["events"].append({
        "step": "writer", "agent": "Writer",
        "decision": "draft ready",
        "detail": f"{len(state['draft'])} chars, "
                  f"grounded in {len(hits)} knowledge-base chunks"
                  + (f" + {len(glean_results)} Glean results" if glean_results else ""),
        "latency_ms": int((time.time() - t0) * 1000),
    })
    return state


REVIEWER_SYSTEM = """You are the Reviewer in a self-correcting RAG copilot.
Judge the draft answer against: (1) grounding — every factual claim must come
from the provided context and citations [n] must match, (2) correctness for the
question, (3) clarity. Reply ONLY with JSON:
{"decision": "PASS" | "REVISE", "feedback": "<specific fix needed, empty if PASS>"}"""


def reviewer_node(state: dict) -> dict:
    t0 = time.time()
    _, reviewer_model, _ = _models()
    raw = complete(
        REVIEWER_SYSTEM,
        f"Question: {state['question']}\n\nContext given to the writer:\n"
        f"{state.get('context', '')[:6000]}\n\nDraft answer:\n{state['draft']}",
        model=reviewer_model,
        max_tokens=400,
    )
    try:
        review = Review(**json.loads(_extract_json(raw)))
        decision, feedback = review.decision, review.feedback
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("reviewer: unparseable (%s) — accepting draft", exc)
        decision, feedback = "PASS", ""
    state["decision"] = decision
    state["feedback"] = feedback
    state["events"].append({
        "step": "reviewer", "agent": "Reviewer", "decision": decision,
        "detail": feedback[:200] or "answer grounded and clear",
        "latency_ms": int((time.time() - t0) * 1000),
    })
    return state


REVISER_SYSTEM = """You are the Reviser. Improve the draft answer per the
reviewer's feedback while staying strictly grounded in the same context. Keep
inline [n] citations. Output ONLY the improved answer."""


def reviser_node(state: dict) -> dict:
    t0 = time.time()
    _, _, reviser_model = _models()
    state["draft"] = complete(
        REVISER_SYSTEM,
        f"Question: {state['question']}\n\nContext:\n{state.get('context', '')[:6000]}\n\n"
        f"Current draft:\n{state['draft']}\n\nReviewer feedback:\n{state['feedback']}",
        model=reviser_model,
    )
    state["revision_count"] = state.get("revision_count", 0) + 1
    state["events"].append({
        "step": "reviser", "agent": "Reviser",
        "decision": f"revision {state['revision_count']}",
        "detail": "applied reviewer feedback",
        "latency_ms": int((time.time() - t0) * 1000),
    })
    return state


def _extract_json(text: str) -> str:
    import re

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in reviewer reply")
    return m.group(0)
