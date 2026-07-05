"""
LangGraph workflow assembly for ATS Multi-Agent Ops Assistant.

Defines the full state machine with deterministic and conditional routing.

Graph topology (happy path):
    router ──► knowledge ──► safety ──► finalize ──► END
    router ──► diagnostics ──► safety ──► finalize ──► END
    router ──► status ──► safety ──► finalize ──► END

Short-circuit paths:
    router ──► off_topic ──► END      (query outside domain)
    router ──► escalation ──► END     (explicit escalation request)
    safety ──► escalation ──► END     (safety/confidence gate failed)

Usage:
    from src.graph.workflow import run_query

    result = run_query("What is the current status of LHC_BEAM1?")
    print(result["final_response"])
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    diagnostics_node,
    escalation_node,
    finalize_node,
    knowledge_node,
    off_topic_node,
    router_node,
    safety_node,
    status_node,
)
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

#: Mapping from domain string to next node name.
_DOMAIN_TO_NODE: dict[str, str] = {
    "docs": "knowledge",
    "diagnostics": "diagnostics",
    "status": "status",
}


def route_after_router(state: AgentState) -> str:
    """Determine the next node after the router classifies the query.

    Decision priority:
    1. Off-topic → ``off_topic`` (short-circuit, no LLM cost)
    2. Explicit escalation domain → ``escalation``
    3. Domain-based routing → ``knowledge`` / ``diagnostics`` / ``status``
    4. Unknown domain → ``knowledge`` (safe default)

    Args:
        state: Current workflow state after the router node has run.

    Returns:
        Name of the next node to execute.
    """
    if state.get("is_off_topic"):
        logger.debug("route_after_router → off_topic")
        return "off_topic"

    if state.get("should_escalate") and state.get("domain") == "escalation":
        logger.debug("route_after_router → escalation (explicit)")
        return "escalation"

    domain: str = state.get("domain") or "docs"
    next_node = _DOMAIN_TO_NODE.get(domain, "knowledge")
    logger.debug("route_after_router → %s (domain=%s)", next_node, domain)
    return next_node


def route_after_safety(state: AgentState) -> str:
    """Determine the next node after the safety gate runs.

    Decision:
    - If ``should_escalate`` is True (set by the safety node or upstream),
      route to ``escalation``.
    - Otherwise route to ``finalize`` for response formatting.

    Args:
        state: Current workflow state after the safety node has run.

    Returns:
        ``'escalation'`` or ``'finalize'``.
    """
    if state.get("should_escalate"):
        logger.debug("route_after_safety → escalation")
        return "escalation"
    logger.debug("route_after_safety → finalize")
    return "finalize"


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------


def build_workflow() -> Any:
    """Assemble and compile the LangGraph state machine.

    Registers all nodes, sets the entry point, wires deterministic edges
    between domain agents and the safety node, and adds conditional edges
    from the router and safety nodes.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to invoke.
    """
    graph: StateGraph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("diagnostics", diagnostics_node)
    graph.add_node("status", status_node)
    graph.add_node("safety", safety_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("off_topic", off_topic_node)
    graph.add_node("finalize", finalize_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Conditional routing: router → domain agents / off_topic / escalation ─
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "knowledge": "knowledge",
            "diagnostics": "diagnostics",
            "status": "status",
            "escalation": "escalation",
            "off_topic": "off_topic",
        },
    )

    # ── Deterministic edges: domain agents → safety ──────────────────────────
    for domain_node in ("knowledge", "diagnostics", "status"):
        graph.add_edge(domain_node, "safety")

    # ── Conditional routing: safety → escalation | finalize ──────────────────
    graph.add_conditional_edges(
        "safety",
        route_after_safety,
        {
            "escalation": "escalation",
            "finalize": "finalize",
        },
    )

    # ── Terminal edges ───────────────────────────────────────────────────────
    graph.add_edge("escalation", END)
    graph.add_edge("off_topic", END)
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_workflow: Any | None = None
_workflow_lock_flag = False  # lightweight re-entrant guard


def get_workflow() -> Any:
    """Return the singleton compiled workflow, building it on first call.

    The graph is compiled once at startup and reused across all requests,
    avoiding repeated compilation overhead.

    Returns:
        The compiled LangGraph workflow instance.
    """
    global _workflow
    if _workflow is None:
        logger.info("Compiling LangGraph workflow (first call)…")
        _workflow = build_workflow()
    return _workflow


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

#: Default initial state values to ensure all TypedDict keys are populated.
_INITIAL_STATE_DEFAULTS: AgentState = {
    "query": "",
    "domain": None,
    "retrieved_context": None,
    "agent_response": None,
    "tool_calls": [],
    "safety_passed": False,
    "faithfulness_score": 0.0,
    "confidence_score": 0.0,
    "safety_flags": [],
    "should_escalate": False,
    "escalation_reason": None,
    "final_response": None,
    "is_off_topic": False,
    "iteration_count": 0,
    "error": None,
}


def run_query(query: str) -> AgentState:
    """Run a natural-language query through the full agentic workflow.

    This is the primary public interface for the workflow module.
    It constructs a clean initial state, invokes the compiled graph, and
    handles any top-level exceptions with a graceful error response.

    Args:
        query: The raw natural-language query from the operator or user.

    Returns:
        The final ``AgentState`` dict populated by all executed nodes.
        On error, the ``error`` field contains the exception message and
        ``final_response`` contains a safe error notice.

    Example::

        result = run_query("What is the current pressure in VACUUM_IR1?")
        print(result["final_response"])
        print(f"Domain: {result['domain']}")
        print(f"Confidence: {result['confidence_score']:.0%}")
    """
    if not query or not query.strip():
        logger.warning("run_query called with empty query")
        return {
            **_INITIAL_STATE_DEFAULTS,
            "query": query,
            "final_response": "Please provide a non-empty query.",
            "error": "Empty query",
        }

    workflow = get_workflow()

    initial_state: AgentState = {
        **_INITIAL_STATE_DEFAULTS,
        "query": query.strip(),
    }

    try:
        result: AgentState = workflow.invoke(initial_state)
        logger.info(
            "run_query completed: domain=%s confidence=%.2f safety=%s",
            result.get("domain"),
            result.get("confidence_score", 0.0),
            result.get("safety_passed"),
        )
        return result

    except Exception as exc:
        logger.exception("Unhandled workflow error for query=%r: %s", query[:80], exc)
        error_response = (
            "⚠️ A system error occurred while processing your request. "
            "Our engineering team has been notified. "
            f"Error reference: {type(exc).__name__}"
        )
        return {
            **_INITIAL_STATE_DEFAULTS,
            "query": query,
            "final_response": error_response,
            "error": str(exc),
        }
