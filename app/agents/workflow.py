"""
Combined LangGraph — agentic RAG (route → retrieve → grade → glean → rewrite)
feeding the self-correcting writer → reviewer → reviser loop.

Conditional edges (from the two reference repos):
  route     : greeting → answer directly; question → retrieve
  grade     : GOOD → writer; WEAK → glean
  grade_web : GOOD → writer (with external note); WEAK → rewrite or give up
  reviewer  : PASS → done; REVISE → reviser (max MAX_REVISIONS)
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents import agents, rag_graph
from app.core.config import MAX_REVISIONS, MAX_RETRIEVAL_RETRIES
from app.core.logging import logger


class CopilotState(TypedDict, total=False):
    question: str
    question_type: str
    search_query: str
    kb_hits: list
    evidence_verdict: str
    glean_results: list
    glean_verdict: str
    retrieval_retries: int
    context: str
    draft: str
    decision: str
    feedback: str
    revision_count: int
    greeting_answer: str
    final_answer: str
    used_glean: bool
    # Nodes mutate events in place and return the whole state — no reducer,
    # or LangGraph would re-add every prior event at each step.
    events: list


def _answer_greeting(state: dict) -> dict:
    state["final_answer"] = (
        "Hi! I'm the KB Copilot — I answer questions about Consumer Growth "
        "metrics, datasets and playbooks from the knowledge base, with "
        "Glean as backup search. Ask me things like "
        "“What is the churn definition for this domain?”"
    )
    state["used_glean"] = False
    return state


def route_decision(state: dict) -> str:
    return "direct" if state.get("question_type") == "greeting" else "retrieve"


def after_grade(state: dict) -> str:
    return "writer" if state.get("evidence_verdict") == "GOOD" else "glean"


def after_glean(state: dict) -> str:
    if state.get("glean_verdict") == "GOOD":
        return "writer"
    retries = state.get("retrieval_retries", 0)
    if retries < MAX_RETRIEVAL_RETRIES:
        return "rewrite"
    return "writer"  # give up gracefully — writer will state insufficiency


def after_reviewer(state: dict) -> str:
    if state.get("decision") == "PASS":
        return "done"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        logger.info("workflow: revision limit hit — returning latest draft")
        return "done"
    return "reviser"


def _finish(state: dict) -> dict:
    if not state.get("final_answer"):
        state["final_answer"] = state.get("draft", "")
    return state


def build_graph():
    g = StateGraph(CopilotState)

    g.add_node("route", rag_graph.route_node)
    g.add_node("greet", _answer_greeting)
    g.add_node("retrieve", rag_graph.retrieve_node)
    g.add_node("grade", rag_graph.grade_node)
    g.add_node("glean", rag_graph.glean_node)
    g.add_node("rewrite", rag_graph.rewrite_node)
    g.add_node("writer", agents.writer_node)
    g.add_node("reviewer", agents.reviewer_node)
    g.add_node("reviser", agents.reviser_node)
    g.add_node("finish", _finish)

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route", route_decision, {"direct": "greet", "retrieve": "retrieve"}
    )
    g.add_edge("greet", "finish")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", after_grade, {"writer": "writer", "glean": "glean"}
    )
    g.add_conditional_edges(
        "glean", after_glean, {"writer": "writer", "rewrite": "rewrite"}
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer", after_reviewer, {"done": "finish", "reviser": "reviser"}
    )
    g.add_edge("reviser", "reviewer")
    g.add_edge("finish", END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(question: str) -> dict:
    """Execute the full workflow. Returns the final state (serialisable subset)."""
    state = get_graph().invoke(
        {"question": question, "events": []},
        {"recursion_limit": 40},
    )
    return {
        "question": question,
        "final_answer": state.get("final_answer", ""),
        "question_type": state.get("question_type", "question"),
        "used_glean": bool(state.get("glean_results")),
        "revisions": state.get("revision_count", 0),
        "final_decision": state.get("decision", "PASS"),
        "kb_sources": [
            {"path": h["path"], "branches": h["branches"], "score": h["score"]}
            for h in state.get("kb_hits", [])
        ],
        "glean_sources": state.get("glean_results", []),
        "events": state.get("events", []),
    }
