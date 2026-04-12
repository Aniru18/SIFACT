"""
graph/workflow.py
Assembles the full SIFACT pipeline as a LangGraph StateGraph.

Flow:
  START → extraction → verification → synthesis → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.extraction_agent import extraction_node
from agents.synthesis_agent import synthesis_node
from agents.verification_agent import verification_node
from graph.state import SIFACTState


def build_graph() -> StateGraph:
    """Build and compile the SIFACT LangGraph workflow."""

    graph = StateGraph(SIFACTState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("extraction",   extraction_node)
    graph.add_node("verification", verification_node)
    graph.add_node("synthesis",    synthesis_node)

    # ── Define edges (linear pipeline) ────────────────────────────────────────
    graph.add_edge(START,          "extraction")
    graph.add_edge("extraction",   "verification")
    graph.add_edge("verification", "synthesis")
    graph.add_edge("synthesis",    END)

    return graph.compile()


# Singleton compiled graph (imported by main.py and tests)
sifact_graph = build_graph()