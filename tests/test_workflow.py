"""Unit test — conditional edges of the combined workflow graph (no LLM calls).

Monkeypatches complete/complete_json to deterministic stubs, then walks:
  greeting → direct answer
  question → retrieve → grade GOOD → writer → reviewer PASS → finish
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.agents import agents, llm, rag_graph, workflow  # noqa: E402


class FakeResp:
    def __init__(self, text: str) -> None:
        self.content = [type("B", (), {"text": text})()]


def route_stub(*a, **kw):
    return {"type": "question"}  # complete_json returns parsed dict


def grade_stub(*a, **kw):
    return '{"verdict": "GOOD", "reason": "covers it"}'


def reviewer_stub(*a, **kw):
    return '{"decision": "PASS", "feedback": ""}'


def writer_stub(*a, **kw):
    return "Draft answer [1]."


def main() -> int:
    # stub LLM layer
    llm.complete = writer_stub
    llm.complete_json = route_stub
    rag_graph.complete_json = route_stub
    rag_graph.complete = writer_stub
    agents.complete = writer_stub
    rag_graph._grade = lambda q, e, s, step, agent: (
        s["events"].append({"step": step, "agent": agent, "decision": "GOOD",
                            "detail": "stub", "latency_ms": 1}) or "GOOD"
    )
    # stub retrieval + glean
    rag_graph.vectorstore.search = lambda q, k=6: [{
        "text": "NTB = no prior delivered brand order in 180 days.",
        "path": "references/glossary-metrics.md",
        "branches": "marketing", "source": "kb", "score": 0.82,
    }]
    rag_graph.glean.is_configured = lambda: False
    rag_graph.glean.search = lambda q, max_results=5: []

    wf = workflow.build_graph()
    state = wf.invoke({"question": "What is NTB?", "events": []})
    steps = [e["step"] for e in state["events"]]
    assert state["question_type"] == "question", state
    assert steps == ["route", "retrieve", "grade", "writer", "reviewer"], steps
    assert state["final_answer"].startswith("Draft answer"), state["final_answer"]

    # greeting path
    llm.complete_json = lambda *a, **kw: {"type": "greeting"}
    rag_graph.complete_json = llm.complete_json
    state2 = wf.invoke({"question": "hello!", "events": []})
    assert state2["question_type"] == "greeting"
    assert "KB Copilot" in state2["final_answer"]
    print("test_workflow: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
