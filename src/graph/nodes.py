"""
LangGraph node implementations for ATS Multi-Agent Ops Assistant.

Each node is a pure function:  AgentState -> dict  (partial state update).

Nodes must never mutate the incoming state dict — LangGraph merges the
returned partial dict into the shared state automatically.

Node execution order (happy path):
    router -> [knowledge | diagnostics | status] -> safety -> finalize

Short-circuit paths:
    router -> off_topic  (END)
    router -> escalation (END)
    safety -> escalation (END)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.config import get_settings
from src.graph.state import AgentState
from src.llm.nvidia_client import get_llm_client
from src.mcp_servers.mcp_server_diagnostics import get_diagnostics_server
from src.mcp_servers.mcp_server_docs import get_docs_server
from src.mcp_servers.mcp_server_escalation import get_escalation_server
from src.safety.audit_logger import get_audit_logger
from src.safety.faithfulness import FaithfulnessChecker
from src.safety.guardrails import GuardrailsChecker

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Keywords that keep a query within the accelerator-operations domain.
ACCELERATOR_TOPICS: list[str] = [
    "accelerator", "LHC", "beam", "magnet", "quench", "cryogenic", "cryo",
    "RF", "cavity", "vacuum", "collimator", "luminosity", "injection",
    "dump", "interlock", "diagnostics", "telemetry", "anomaly", "health",
    "subsystem", "CERN", "proton", "physics", "operations", "status",
    "power", "converter", "cooling", "access", "emergency", "shutdown",
]

#: System prompt used by the router LLM call.
ROUTER_SYSTEM: str = (
    "You are a query router for the ATS Accelerator Operations Assistant at CERN.\n"
    "Classify the user query into exactly one domain:\n"
    "- 'docs': questions about accelerator documentation, procedures, protocols, how-to\n"
    "- 'diagnostics': questions about anomalies, system health, telemetry, failures, alerts\n"
    "- 'status': questions about current operational status of systems\n"
    "- 'escalation': requests for human help, urgent issues, emergencies\n"
    "- 'off_topic': questions unrelated to accelerator/physics operations\n\n"
    "Respond with ONLY a JSON object: "
    '{\"domain\": \"<domain>\", \"confidence\": <0.0-1.0>, \"reasoning\": \"<brief reason>\"}'
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _append_tool_call(state: AgentState, node: str, tool: str, latency: float, **extra: Any) -> list[dict]:
    """Return a new tool-call list with one entry appended (immutable pattern)."""
    entry: dict[str, Any] = {"node": node, "tool": tool, "latency": latency}
    entry.update(extra)
    return list(state.get("tool_calls") or []) + [entry]


# ---------------------------------------------------------------------------
# Node: router
# ---------------------------------------------------------------------------


def router_node(state: AgentState) -> dict:
    """Classify the incoming query into an operational domain.

    Uses a lightweight LLM call (``reasoning_effort='none'``) to keep
    routing latency minimal.  Falls back to ``'docs'`` on parse errors.

    Returns a partial state update with:
    - ``domain``
    - ``confidence_score``
    - ``is_off_topic``
    - ``should_escalate``
    - ``escalation_reason``
    - ``tool_calls`` (appended)
    """
    audit = get_audit_logger()
    llm = get_llm_client()

    start = time.perf_counter()
    raw_response = llm.simple_chat(
        state["query"],
        system=ROUTER_SYSTEM,
        reasoning_effort="none",
    )
    latency = time.perf_counter() - start

    # ── Parse JSON ──────────────────────────────────────────────────────────
    try:
        parsed: dict = json.loads(raw_response.strip())
        domain: str = str(parsed.get("domain", "docs"))
        confidence: float = float(parsed.get("confidence", 0.5))
        reasoning: str = str(parsed.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Router: failed to parse LLM JSON — defaulting to docs")
        domain = "docs"
        confidence = 0.5
        reasoning = "Parse error — defaulting to docs"

    # Normalise domain to allowed values
    valid_domains = {"docs", "diagnostics", "status", "escalation", "off_topic"}
    if domain not in valid_domains:
        logger.warning("Router: unknown domain '%s' — defaulting to docs", domain)
        domain = "docs"
        confidence = max(confidence * 0.8, 0.1)

    is_off_topic: bool = domain == "off_topic"
    low_confidence: bool = confidence < settings.confidence_threshold
    should_escalate: bool = domain == "escalation" or (low_confidence and not is_off_topic)

    escalation_reason: str | None = None
    if low_confidence and not is_off_topic and domain != "escalation":
        escalation_reason = (
            f"Low routing confidence ({confidence:.2f} < {settings.confidence_threshold})"
        )

    # ── Audit ───────────────────────────────────────────────────────────────
    audit.log_tool_call(
        node="router",
        tool="classify_query",
        input_data={"query": state["query"]},
        output_data={"domain": domain, "confidence": confidence, "reasoning": reasoning},
        latency=latency,
    )

    logger.info(
        "Router: domain=%s confidence=%.2f escalate=%s off_topic=%s",
        domain, confidence, should_escalate, is_off_topic,
    )

    return {
        "domain": domain,
        "confidence_score": confidence,
        "is_off_topic": is_off_topic,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
        "tool_calls": _append_tool_call(state, "router", "classify_query", latency),
    }


# ---------------------------------------------------------------------------
# Node: knowledge
# ---------------------------------------------------------------------------


def knowledge_node(state: AgentState) -> dict:
    """Retrieve documentation context and generate a grounded answer.

    Uses the docs MCP server for RAG retrieval, then sends the context +
    query to the LLM to produce a factual, citation-aware response.

    Returns a partial state update with:
    - ``retrieved_context``
    - ``agent_response``
    - ``confidence_score``
    - ``tool_calls`` (appended)
    """
    audit = get_audit_logger()
    llm = get_llm_client()
    docs_server = get_docs_server()

    # ── Retrieval ────────────────────────────────────────────────────────────
    ret_start = time.perf_counter()
    context: str = docs_server.search_accelerator_docs(state["query"], top_k=3)
    ret_latency = time.perf_counter() - ret_start

    # ── Generation ───────────────────────────────────────────────────────────
    system = (
        "You are the ATS Knowledge Agent at CERN. Answer questions about accelerator operations "
        "using ONLY the provided documentation context. Be precise, technical, and cite specific "
        "procedures where relevant.\n"
        "If the context does not contain a clear answer, state that explicitly — do NOT hallucinate "
        "or invent procedures.\n"
        "Always close your response with exactly one of:\n"
        "  Confidence: HIGH   (answer fully supported by context)\n"
        "  Confidence: MEDIUM (answer partially supported)\n"
        "  Confidence: LOW    (context insufficient)"
    )
    prompt = f"Documentation Context:\n{context}\n\nUser Question: {state['query']}"

    llm_start = time.perf_counter()
    response: str = llm.simple_chat(prompt, system=system)
    llm_latency = time.perf_counter() - llm_start

    # ── Derive confidence from self-reported tag ─────────────────────────────
    if "Confidence: HIGH" in response:
        confidence = 0.90
    elif "Confidence: MEDIUM" in response:
        confidence = 0.70
    else:
        confidence = 0.50

    # ── Audit ────────────────────────────────────────────────────────────────
    audit.log_tool_call(
        "knowledge", "search_accelerator_docs",
        {"query": state["query"]},
        {"context_chars": len(context)},
        ret_latency,
    )
    audit.log_tool_call(
        "knowledge", "llm_generate",
        {"prompt_chars": len(prompt)},
        {"response_chars": len(response)},
        llm_latency,
    )

    logger.info("Knowledge node: retrieved %d chars, confidence=%.2f", len(context), confidence)

    tool_calls = _append_tool_call(state, "knowledge", "search_docs", ret_latency)
    tool_calls = tool_calls + [{"node": "knowledge", "tool": "llm_generate", "latency": llm_latency}]

    return {
        "retrieved_context": context,
        "agent_response": response,
        "confidence_score": confidence,
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Node: diagnostics
# ---------------------------------------------------------------------------


def diagnostics_node(state: AgentState) -> dict:
    """Run anomaly detection and health checks for the queried subsystem.

    Extracts the subsystem name from the query using a low-cost LLM call,
    then fetches live telemetry from the diagnostics MCP server.

    Returns a partial state update with:
    - ``retrieved_context``
    - ``agent_response``
    - ``confidence_score``
    - ``tool_calls`` (appended)
    """
    audit = get_audit_logger()
    llm = get_llm_client()
    diag_server = get_diagnostics_server()

    # ── Extract subsystem name ────────────────────────────────────────────────
    available_systems = list(diag_server.get_tools().keys())
    extract_prompt = (
        f"Extract the accelerator subsystem name from this query.\n"
        f"Return ONLY the subsystem name — choose from: {available_systems} "
        f"or a name like LHC_BEAM1, CRYO_SECTOR12, RF_CAVITY_A, VACUUM_IR1, "
        f"MAGNET_ARC12, COLLIMATOR_TCP.\n"
        f"If unclear, return: LHC_BEAM1\n"
        f"Return ONLY the system name, nothing else.\n\n"
        f"Query: {state['query']}"
    )
    raw_system = llm.simple_chat(extract_prompt, reasoning_effort="none").strip()
    system_name = raw_system.split()[0].upper() if raw_system else "LHC_BEAM1"

    # ── Telemetry fetch ───────────────────────────────────────────────────────
    diag_start = time.perf_counter()
    try:
        anomaly_result: str = diag_server.run_anomaly_check(system_name, "1h")
        health_result: str = diag_server.get_system_health(system_name)
    except Exception as exc:  # graceful degradation
        logger.warning("Diagnostics server error for %s: %s", system_name, exc)
        anomaly_result = f"Anomaly check unavailable: {exc}"
        health_result = f"Health data unavailable: {exc}"
    diag_latency = time.perf_counter() - diag_start

    context = f"Anomaly Check ({system_name}):\n{anomaly_result}\n\nHealth Status:\n{health_result}"

    # ── Generation ───────────────────────────────────────────────────────────
    system = (
        "You are the ATS Diagnostics Agent at CERN. Analyse telemetry data and provide clear, "
        "actionable diagnostic assessments.\n"
        "Highlight: anomalies detected, their severity (INFO/WARNING/CRITICAL), root-cause "
        "hypotheses, and recommended immediate actions.\n"
        "Explicitly state whether escalation to a human engineer is recommended.\n"
        "Close with exactly one of:\n"
        "  Confidence: HIGH | Confidence: MEDIUM | Confidence: LOW"
    )
    prompt = f"Telemetry Data:\n{context}\n\nDiagnostic Query: {state['query']}"

    llm_start = time.perf_counter()
    response: str = llm.simple_chat(prompt, system=system)
    llm_latency = time.perf_counter() - llm_start

    if "Confidence: HIGH" in response:
        confidence = 0.90
    elif "Confidence: MEDIUM" in response:
        confidence = 0.70
    else:
        confidence = 0.50

    # ── Audit ────────────────────────────────────────────────────────────────
    audit.log_tool_call(
        "diagnostics", "run_anomaly_check",
        {"system": system_name, "window": "1h"},
        {"result_chars": len(anomaly_result)},
        diag_latency,
    )
    audit.log_tool_call(
        "diagnostics", "llm_generate",
        {"prompt_chars": len(prompt)},
        {"response_chars": len(response)},
        llm_latency,
    )

    logger.info("Diagnostics node: system=%s confidence=%.2f", system_name, confidence)

    tool_calls = _append_tool_call(state, "diagnostics", "run_anomaly_check", diag_latency, system=system_name)
    tool_calls = tool_calls + [{"node": "diagnostics", "tool": "llm_generate", "latency": llm_latency}]

    return {
        "retrieved_context": context,
        "agent_response": response,
        "confidence_score": confidence,
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Node: status
# ---------------------------------------------------------------------------


def status_node(state: AgentState) -> dict:
    """Return a full cross-system operational status report.

    Runs the diagnostics server's full-sweep diagnostic and summarises
    the results into a structured status report.

    Returns a partial state update with:
    - ``retrieved_context``
    - ``agent_response``
    - ``confidence_score``  (fixed at 0.95 — live data, high trust)
    - ``tool_calls`` (appended)
    """
    audit = get_audit_logger()
    llm = get_llm_client()
    diag_server = get_diagnostics_server()

    # ── Full diagnostic sweep ─────────────────────────────────────────────────
    diag_start = time.perf_counter()
    try:
        full_status: str = diag_server.run_full_diagnostic()
    except Exception as exc:
        logger.warning("Full diagnostic sweep failed: %s", exc)
        full_status = f"Full diagnostic unavailable: {exc}"
    diag_latency = time.perf_counter() - diag_start

    # ── Generation ───────────────────────────────────────────────────────────
    system = (
        "You are the ATS Status Agent at CERN. Provide a concise operational status summary.\n"
        "Structure your report as:\n"
        "  1. Overall system health (NOMINAL / DEGRADED / CRITICAL)\n"
        "  2. Systems in WARNING or CRITICAL state — list each with a one-line description\n"
        "  3. Immediate action items (if any)\n"
        "  4. Systems operating nominally (brief list)\n"
        "Be factual and concise. Do not add information not present in the diagnostic data.\n"
        "Close with: Confidence: HIGH"
    )
    prompt = f"Full System Diagnostic:\n{full_status}\n\nStatus Query: {state['query']}"

    llm_start = time.perf_counter()
    response: str = llm.simple_chat(prompt, system=system)
    llm_latency = time.perf_counter() - llm_start

    # ── Audit ────────────────────────────────────────────────────────────────
    audit.log_tool_call(
        "status", "run_full_diagnostic",
        {},
        {"result_chars": len(full_status)},
        diag_latency,
    )
    audit.log_tool_call(
        "status", "llm_generate",
        {"prompt_chars": len(prompt)},
        {"response_chars": len(response)},
        llm_latency,
    )

    logger.info("Status node: diagnostic_chars=%d", len(full_status))

    tool_calls = _append_tool_call(state, "status", "run_full_diagnostic", diag_latency)
    tool_calls = tool_calls + [{"node": "status", "tool": "llm_generate", "latency": llm_latency}]

    return {
        "retrieved_context": full_status,
        "agent_response": response,
        "confidence_score": 0.95,  # live data — inherently high trust
        "tool_calls": tool_calls,
    }


# ---------------------------------------------------------------------------
# Node: safety
# ---------------------------------------------------------------------------


def safety_node(state: AgentState) -> dict:
    """Evaluate the agent's response through guardrails and faithfulness checks.

    This node acts as a quality gate before the response is finalised:
    - Guardrails: detect unsafe, hallucinated, or policy-violating content.
    - Faithfulness: ensure the response is grounded in the retrieved context.

    Returns a partial state update with:
    - ``safety_passed``
    - ``faithfulness_score``
    - ``safety_flags``
    - ``should_escalate``  (may be set to True)
    - ``escalation_reason``
    """
    guardrails = GuardrailsChecker()
    faithfulness = FaithfulnessChecker()
    audit = get_audit_logger()

    response: str = state.get("agent_response") or ""
    context: str = state.get("retrieved_context") or ""

    # ── Guardrails ───────────────────────────────────────────────────────────
    safety_result: dict = guardrails.check(state["query"], response)
    flags: list[str] = list(safety_result.get("flags", []))

    # ── Faithfulness ─────────────────────────────────────────────────────────
    if context:
        faith_score: float = faithfulness.score(response, context)
    else:
        faith_score = 1.0  # no context to be unfaithful to

    if faith_score < 0.5:
        flags.append(f"Low faithfulness score: {faith_score:.2f}")

    safety_passed: bool = safety_result.get("passed", False) and faith_score >= 0.5

    # ── Escalation propagation ───────────────────────────────────────────────
    upstream_escalate: bool = state.get("should_escalate", False)
    low_confidence: bool = state.get("confidence_score", 1.0) < settings.confidence_threshold
    should_escalate: bool = upstream_escalate or not safety_passed or low_confidence

    if not safety_passed:
        escalation_reason: str | None = "Safety check failed"
    elif low_confidence and not upstream_escalate:
        escalation_reason = (
            f"Low agent confidence ({state.get('confidence_score', 0):.2f})"
        )
    else:
        escalation_reason = state.get("escalation_reason")

    # ── Audit ────────────────────────────────────────────────────────────────
    audit.log_tool_call(
        "safety", "guardrails_check",
        {"query_preview": state["query"][:120]},
        {"passed": safety_passed, "faithfulness": faith_score, "flags": flags},
        0.0,
    )

    logger.info(
        "Safety node: passed=%s faithfulness=%.2f flags=%s escalate=%s",
        safety_passed, faith_score, flags, should_escalate,
    )

    return {
        "safety_passed": safety_passed,
        "faithfulness_score": faith_score,
        "safety_flags": flags,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
    }


# ---------------------------------------------------------------------------
# Node: escalation
# ---------------------------------------------------------------------------


def escalation_node(state: AgentState) -> dict:
    """Escalate the unresolved query to an on-call human engineer.

    Creates a structured escalation ticket via the escalation MCP server
    and returns a user-facing acknowledgement with ticket details.

    Returns a partial state update with:
    - ``final_response``
    - ``tool_calls`` (appended)
    """
    audit = get_audit_logger()
    esc_server = get_escalation_server()

    reason: str = state.get("escalation_reason") or "Automatic escalation triggered"
    safety_flags: list[str] = state.get("safety_flags") or []
    confidence: float = state.get("confidence_score", 0.0)

    context = (
        f"Query: {state['query']}\n\n"
        f"Agent Response: {state.get('agent_response') or 'No response generated'}\n\n"
        f"Safety Flags: {safety_flags}\n\n"
        f"Confidence: {confidence:.2f}\n"
        f"Domain: {state.get('domain', 'unknown')}"
    )

    priority: str = "HIGH" if safety_flags else "MEDIUM"

    # ── Escalation ticket ────────────────────────────────────────────────────
    esc_start = time.perf_counter()
    try:
        esc_raw: str = esc_server.escalate_to_human(
            reason=reason, context=context, priority=priority
        )
        esc_data: dict = json.loads(esc_raw)
    except Exception as exc:
        logger.error("Escalation server error: %s", exc)
        esc_data = {
            "escalation_id": "ESC-ERROR",
            "assigned_to": {
                "name": "On-call Engineer",
                "role": "Accelerator Operations",
                "contact": "ats-oncall@cern.ch",
            },
        }
    esc_latency = time.perf_counter() - esc_start

    assigned = esc_data.get("assigned_to", {})
    response_time = "15 minutes" if priority == "HIGH" else "1 hour"

    # Truncate partial AI response to avoid overwhelming the user
    partial_ai = (state.get("agent_response") or "None")[:400]
    if len(state.get("agent_response") or "") > 400:
        partial_ai += "…"

    final_response = (
        f"⚠️ **Escalated to Human Engineer**\n\n"
        f"This query has been escalated to **{assigned.get('name', 'On-call Engineer')}** "
        f"({assigned.get('role', 'Accelerator Operations')}).\n\n"
        f"| Field | Value |\n"
        f"|-------|-------|\n"
        f"| **Escalation ID** | `{esc_data.get('escalation_id', 'N/A')}` |\n"
        f"| **Reason** | {reason} |\n"
        f"| **Priority** | {priority} |\n"
        f"| **Contact** | {assigned.get('contact', 'ats-oncall@cern.ch')} |\n"
        f"| **Expected response** | {response_time} |\n\n"
        f"*Partial AI response (unverified):*\n> {partial_ai}"
    )

    # ── Audit ────────────────────────────────────────────────────────────────
    audit.log_tool_call(
        "escalation", "escalate_to_human",
        {"reason": reason, "priority": priority},
        esc_data,
        esc_latency,
    )

    logger.info(
        "Escalation node: id=%s priority=%s assigned=%s",
        esc_data.get("escalation_id"), priority, assigned.get("name"),
    )

    return {
        "final_response": final_response,
        "tool_calls": _append_tool_call(
            state, "escalation", "escalate_to_human", esc_latency,
            escalation_id=esc_data.get("escalation_id"),
        ),
    }


# ---------------------------------------------------------------------------
# Node: off_topic
# ---------------------------------------------------------------------------


def off_topic_node(state: AgentState) -> dict:
    """Return a polite out-of-scope notice for non-accelerator queries.

    No LLM call required — this is a static, templated response.

    Returns a partial state update with:
    - ``final_response``
    - ``safety_passed``  (True — safe to return)
    - ``confidence_score`` (1.0 — routing decision is definitive)
    """
    audit = get_audit_logger()

    final_response = (
        "I'm the **ATS Accelerator Operations Assistant**, specialised in CERN accelerator operations.\n\n"
        "I can help with:\n"
        "- 📚 Accelerator documentation and procedures\n"
        "- 🔬 System diagnostics and anomaly detection\n"
        "- 📊 Operational status of accelerator subsystems\n"
        "- 🚨 Escalation to on-call engineers\n\n"
        "Your query appears to be outside my operational domain. "
        "Please rephrase to ask about accelerator operations, "
        "or contact CERN's general helpdesk for other enquiries."
    )

    audit.log_tool_call(
        "off_topic", "reject_query",
        {"query_preview": state["query"][:120]},
        {"reason": "off_topic"},
        0.0,
    )

    logger.info("Off-topic node: query rejected as out-of-scope")

    return {
        "final_response": final_response,
        "safety_passed": True,
        "confidence_score": 1.0,
    }


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------


def finalize_node(state: AgentState) -> dict:
    """Format the verified agent response for delivery to the caller.

    Appends a metadata footer with domain, confidence, and tool usage stats.
    Logs the final response via the audit logger.

    Returns a partial state update with:
    - ``final_response``
    """
    audit = get_audit_logger()

    agent_response: str = state.get("agent_response") or "No response generated."
    domain: str = state.get("domain") or "unknown"
    confidence: float = state.get("confidence_score") or 0.0
    tool_count: int = len(state.get("tool_calls") or [])
    faithfulness: float = state.get("faithfulness_score") or 0.0
    safety_flags: list[str] = state.get("safety_flags") or []

    safety_note = ""
    if safety_flags:
        safety_note = f"\n> ⚠️ Safety flags: {', '.join(safety_flags)}"

    final_response = (
        f"{agent_response}{safety_note}\n\n"
        f"---\n"
        f"*ATS AI Assistant · Domain: **{domain}** · "
        f"Confidence: **{confidence:.0%}** · "
        f"Faithfulness: **{faithfulness:.0%}** · "
        f"Tools used: **{tool_count}***"
    )

    audit.log_response(
        query=state["query"],
        domain=domain,
        response=final_response,
        confidence=confidence,
        safety_passed=state.get("safety_passed", False),
    )

    logger.info(
        "Finalize node: domain=%s confidence=%.2f faithfulness=%.2f tools=%d",
        domain, confidence, faithfulness, tool_count,
    )

    return {"final_response": final_response}
