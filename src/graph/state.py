"""
LangGraph state definitions for the ATS Multi-Agent Ops Assistant.
"""
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Central state shared across all agent nodes in the graph.

    Fields are partitioned by concern:
    - Input: raw user query
    - Routing: domain classification result
    - Agent outputs: retrieval context, generated response, tool call log
    - Safety: guardrail / faithfulness results and flags
    - Escalation: whether and why to hand off to a human
    - Final: the text ultimately returned to the caller
    - Meta: loop guard and error capture
    """

    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    """The raw natural-language query from the operator or end-user."""

    # ── Routing ────────────────────────────────────────────────────────────
    domain: Optional[str]
    """Classified domain: 'docs' | 'diagnostics' | 'status' | 'escalation' | 'off_topic'."""

    # ── Agent outputs ──────────────────────────────────────────────────────
    retrieved_context: Optional[str]
    """Raw context retrieved from MCP servers (docs, telemetry, etc.)."""

    agent_response: Optional[str]
    """The LLM-generated response produced by the active domain agent."""

    tool_calls: list[dict]
    """Ordered log of every tool invoked during this workflow run.

    Each entry is a dict with keys: node, tool, latency (seconds), and any
    extra metadata the node chooses to record.
    """

    # ── Safety ─────────────────────────────────────────────────────────────
    safety_passed: bool
    """True when the response passes all guardrail and faithfulness checks."""

    faithfulness_score: float
    """Semantic faithfulness of the response relative to retrieved_context (0–1)."""

    confidence_score: float
    """Routing or agent-level confidence estimate (0–1)."""

    safety_flags: list[str]
    """Human-readable descriptions of any safety or faithfulness violations."""

    # ── Escalation ─────────────────────────────────────────────────────────
    should_escalate: bool
    """True when the workflow determines a human engineer must be involved."""

    escalation_reason: Optional[str]
    """Short explanation of why escalation was triggered."""

    # ── Final ──────────────────────────────────────────────────────────────
    final_response: Optional[str]
    """The fully formatted response ready to be returned to the caller."""

    is_off_topic: bool
    """True when the query is outside the accelerator-operations domain."""

    # ── Meta ───────────────────────────────────────────────────────────────
    iteration_count: int
    """Number of times the workflow has looped (loop-guard counter)."""

    error: Optional[str]
    """Captured exception message if the workflow raised an unhandled error."""
