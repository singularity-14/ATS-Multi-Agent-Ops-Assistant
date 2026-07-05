"""
Streamlit UI for ATS Multi-Agent Ops Assistant.
Provides a rich interface for querying the multi-agent system,
viewing audit logs, and running evaluations.
"""
import json
import time
import streamlit as st
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATS Multi-Agent Ops Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ATS Multi-Agent Ops Assistant — CERN Multi-Agent AI System"},
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #00d4ff;
        --secondary: #7c3aed;
        --accent: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --bg-dark: #0a0e1a;
        --bg-card: #111827;
        --bg-card2: #1f2937;
        --text-primary: #f9fafb;
        --text-secondary: #9ca3af;
        --border: #374151;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1321 50%, #0a0e1a 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1));
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }

    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #111827, #1f2937);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #00d4ff;
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,212,255,0.15);
    }

    .response-card {
        background: linear-gradient(135deg, #111827, #1a2235);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        position: relative;
        overflow: hidden;
    }

    .response-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #10b981);
    }

    .domain-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-docs        { background: rgba(0,212,255,0.15);  color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }
    .badge-diagnostics { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
    .badge-status      { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .badge-escalation  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .badge-off_topic   { background: rgba(107,114,128,0.15);color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }
    .badge-unknown     { background: rgba(107,114,128,0.15);color: #9ca3af; border: 1px solid rgba(107,114,128,0.3); }

    .stTextInput > div > div > input {
        background: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        color: #f9fafb !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 12px 16px !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,212,255,0.3) !important;
    }

    .sidebar-status {
        background: #0d1321;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
    }

    .chat-message-user {
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(0,212,255,0.03));
        border: 1px solid rgba(0,212,255,0.15);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #00d4ff;
    }

    .chat-message-assistant {
        background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(124,58,237,0.03));
        border: 1px solid rgba(124,58,237,0.15);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        border-left: 3px solid #7c3aed;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e1a, #0d1321) !important;
        border-right: 1px solid #1f2937 !important;
    }

    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f9fafb;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    .block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "query_count" not in st.session_state:
    st.session_state.query_count: int = 0
if "total_latency" not in st.session_state:
    st.session_state.total_latency: float = 0.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_domain_badge(domain: str) -> str:
    """Return an HTML badge coloured by domain."""
    domain_lower = (domain or "unknown").lower()
    return f'<span class="domain-badge badge-{domain_lower}">{domain_lower}</span>'


def get_confidence_color(score: float) -> str:
    """Map a confidence score to a traffic-light colour."""
    if score >= 0.8:
        return "#10b981"
    if score >= 0.6:
        return "#f59e0b"
    return "#ef4444"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 16px 0;">
            <div style="font-size: 2.5rem;">⚡</div>
            <h2 style="margin: 4px 0; color: #00d4ff;">ATS AI Ops</h2>
            <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">CERN Accelerator Operations</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- System status ---
    st.markdown("### 💻 System Status")

    systems = [
        ("Orchestrator",      "✅ Online",     "online"),
        ("Docs MCP Server",   "✅ Online",     "online"),
        ("Diagnostics MCP",   "✅ Online",     "online"),
        ("Escalation MCP",    "✅ Online",     "online"),
        ("Safety Layer",      "✅ Active",     "online"),
        ("Audit Logger",      "✅ Logging",    "online"),
        ("NVIDIA LLM",        "✅ Connected",  "online"),
    ]

    for name, status_text, status_type in systems:
        color = "#10b981" if status_type == "online" else "#f59e0b"
        st.markdown(
            f"""
            <div class="sidebar-status">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #9ca3af; font-size: 0.8rem;">{name}</span>
                    <span style="color: {color}; font-size: 0.75rem; font-weight: 600;">{status_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Session stats ---
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
    with col2:
        avg_latency_ms = (
            st.session_state.total_latency / max(st.session_state.query_count, 1)
        ) * 1000
        st.metric("Avg ms", f"{avg_latency_ms:.0f}")

    st.divider()

    # --- Example queries (pre-fill the text input) ---
    st.markdown("### 💡 Try These")
    examples = [
        "What is the LHC beam injection procedure?",
        "Check anomalies in RF_CAVITY_A",
        "What is the current status of all systems?",
        "Explain the magnet quench protection system",
        "Run diagnostics on CRYO_SECTOR12",
    ]
    for ex in examples:
        label = ex[:45] + ("..." if len(ex) > 45 else "")
        if st.button(label, key=f"ex_{hash(ex)}"):
            st.session_state.pending_query = ex


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ ATS Multi-Agent Ops Assistant</h1>
        <p style="color: #9ca3af; margin: 4px 0 0 0; font-size: 0.9rem;">
            Multi-agent AI system for CERN accelerator operations — powered by NVIDIA LLM + LangGraph
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "📊 Audit Logs", "🧪 Evaluation"])

# ===========================================================================
# TAB 1 — Query Interface
# ===========================================================================
with tab1:
    col1, col2 = st.columns([5, 1])
    with col1:
        # Consume any pre-filled query set by the sidebar example buttons.
        default_query = st.session_state.pop("pending_query", "")
        query = st.text_input(
            "🔎 Enter your query",
            value=default_query,
            placeholder=(
                "e.g., What is the beam injection procedure? / "
                "Check LHC_BEAM1 anomalies / System status"
            ),
            key="main_query_input",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.button("🚀 Ask", use_container_width=True)

    # --- Process query ---
    if submit and query.strip():
        with st.spinner("🤖 Multi-agent system processing…"):
            try:
                from src.graph.workflow import run_query  # local import avoids circulars

                start = time.time()
                result = run_query(query)
                latency = time.time() - start

                st.session_state.query_count += 1
                st.session_state.total_latency += latency

                st.session_state.chat_history.append(
                    {
                        "query": query,
                        "result": result,
                        "latency": latency,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

    # --- Chat history (most recent first) ---
    if st.session_state.chat_history:
        total = len(st.session_state.chat_history)
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            result = entry["result"]
            domain = result.get("domain", "unknown")
            confidence = result.get("confidence_score", 0.0)
            safety_passed = result.get("safety_passed", False)
            faithfulness = result.get("faithfulness_score", 0.0)
            tool_calls = result.get("tool_calls", [])
            safety_flags = result.get("safety_flags", [])
            escalated = result.get("should_escalate", False)

            # User bubble
            st.markdown(
                f"""
                <div class="chat-message-user">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-weight:600; color:#00d4ff;">You</span>
                        <span style="color:#6b7280; font-size:0.75rem;">{entry['timestamp']}</span>
                    </div>
                    <p style="margin:0; color:#f9fafb;">{entry['query']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Assistant response header card
            safety_indicator = (
                '<span style="color:#10b981; font-size:0.75rem;">✅ Safe</span>'
                if safety_passed
                else '<span style="color:#ef4444; font-size:0.75rem;">⚠️ Flagged</span>'
            )
            escalation_indicator = (
                '<span style="color:#ef4444; font-size:0.75rem; margin-left:8px;">🔺 Escalated</span>'
                if escalated
                else ""
            )

            st.markdown(
                f"""
                <div class="response-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <div style="display:flex; gap:8px; align-items:center;">
                            <span style="font-weight:600; color:#a78bfa;">ATS AI</span>
                            {get_domain_badge(domain)}
                            {safety_indicator}
                            {escalation_indicator}
                        </div>
                        <div style="display:flex; gap:16px; color:#6b7280; font-size:0.75rem;">
                            <span>Confidence:
                                <strong style="color:{get_confidence_color(confidence)};">
                                    {confidence:.0%}
                                </strong>
                            </span>
                            <span>Faithfulness:
                                <strong style="color:{get_confidence_color(faithfulness)};">
                                    {faithfulness:.0%}
                                </strong>
                            </span>
                            <span>Latency:
                                <strong style="color:#f9fafb;">{entry['latency']*1000:.0f}ms</strong>
                            </span>
                            <span>Tools:
                                <strong style="color:#f9fafb;">{len(tool_calls)}</strong>
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Response body rendered as markdown
            st.markdown(result.get("final_response", "_No response generated._"))

            # Expandable technical details
            with st.expander("🔍 Technical Details"):
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Faithfulness", f"{faithfulness:.0%}")
                with col_b:
                    st.metric("Confidence", f"{confidence:.0%}")
                with col_c:
                    st.metric("Safety Flags", len(safety_flags))
                with col_d:
                    st.metric("Escalated", "✅ Yes" if escalated else "❌ No")

                if safety_flags:
                    st.warning("⚠️ Safety flags: " + ", ".join(safety_flags))

                if tool_calls:
                    st.markdown("**Tool calls:**")
                    st.json(tool_calls)

            # Divider between conversations (not after the last one)
            if i < total - 1:
                st.divider()

    else:
        # Empty state
        st.markdown(
            """
            <div style="text-align:center; padding:60px 20px; color:#4b5563;">
                <div style="font-size:4rem; margin-bottom:16px;">⚡</div>
                <h3 style="color:#6b7280; font-weight:500;">Ready for your query</h3>
                <p style="color:#4b5563;">
                    Ask about accelerator operations, diagnostics, system status, or documentation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# TAB 2 — Audit Logs
# ===========================================================================
with tab2:
    st.markdown("### 📊 Audit Log Viewer")

    _col1, col_refresh = st.columns([3, 1])
    with col_refresh:
        st.button("🔄 Refresh", key="refresh_audit")  # clicking re-runs the script

    try:
        from src.safety.audit_logger import get_audit_logger

        audit = get_audit_logger()

        # Aggregate stats row
        stats = audit.get_stats()
        cols = st.columns(5)
        metrics = [
            ("Tool Calls",   stats.get("total_tool_calls", 0)),
            ("Responses",    stats.get("total_responses", 0)),
            ("Avg Confidence", f"{stats.get('avg_confidence', 0):.0%}"),
            ("Safety Rate",    f"{stats.get('safety_pass_rate', 0):.0%}"),
            ("Avg Latency",    f"{stats.get('avg_tool_latency_ms', 0):.0f}ms"),
        ]
        for col, (label, val) in zip(cols, metrics):
            with col:
                st.metric(label, val)

        st.divider()

        # Recent tool-call table
        recent = audit.get_recent_tool_calls(20)
        if recent:
            import pandas as pd

            df_raw = pd.DataFrame(recent)
            display_cols = [c for c in ["timestamp", "node", "tool", "latency_ms"] if c in df_raw.columns]
            df = df_raw[display_cols].copy()
            df.columns = [c.replace("_", " ").title() for c in display_cols]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No audit records yet. Run some queries first.")

    except Exception as e:
        st.error(f"Audit log unavailable: {e}")

# ===========================================================================
# TAB 3 — Evaluation
# ===========================================================================
with tab3:
    st.markdown("### 🧪 Evaluation Harness")
    st.markdown(
        "Run the evaluation suite against the full agent system. "
        "This will process all test cases defined in `data/test_cases/eval_suite.json`."
    )

    st.warning(
        "⚠️ Running evaluation will make multiple LLM API calls. "
        "Estimated cost: ~30 API calls."
    )

    _c1, col_run = st.columns([3, 1])
    with col_run:
        run_eval = st.button("▶️ Run Evaluation", key="run_eval_btn")

    if run_eval:
        eval_placeholder = st.empty()
        progress_bar = st.progress(0)

        try:
            from src.evaluation.evaluator import AgentEvaluator, EvalReport
            from src.graph.workflow import run_query

            test_suite_path = Path("data/test_cases/eval_suite.json")
            if not test_suite_path.exists():
                st.error(
                    "Test suite not found. "
                    "Make sure `data/test_cases/eval_suite.json` exists."
                )
            else:
                with open(test_suite_path, encoding='utf-8') as f:
                    test_suite = json.load(f)

                evaluator = AgentEvaluator()
                results = []

                for idx, task in enumerate(test_suite):
                    eval_placeholder.info(
                        f"Processing {idx + 1}/{len(test_suite)}: "
                        f"{task.get('query', '')[:60]}…"
                    )
                    progress_bar.progress((idx + 1) / len(test_suite))
                    result = evaluator.evaluate_single(task, run_query)
                    results.append(result)

                total = len(results)
                passed = sum(1 for r in results if r.domain_correct and r.safety_compliant)

                report = EvalReport(
                    total_tasks=total,
                    passed=passed,
                    failed=total - passed,
                    domain_accuracy=sum(r.domain_correct for r in results) / total,
                    avg_task_completion=sum(r.task_completion_score for r in results) / total,
                    safety_compliance_rate=sum(r.safety_compliant for r in results) / total,
                    avg_faithfulness=sum(r.faithfulness_score for r in results) / total,
                    avg_latency_ms=sum(r.latency_ms for r in results) / total,
                    escalation_rate=sum(r.escalated for r in results) / total,
                    results=results,
                )

                eval_placeholder.success(
                    f"✅ Evaluation complete: {report.passed}/{report.total_tasks} tasks passed"
                )
                progress_bar.progress(1.0)

                # Summary metrics
                cols = st.columns(5)
                with cols[0]:
                    st.metric("Domain Accuracy",   f"{report.domain_accuracy:.0%}")
                with cols[1]:
                    st.metric("Safety Rate",        f"{report.safety_compliance_rate:.0%}")
                with cols[2]:
                    st.metric("Task Completion",    f"{report.avg_task_completion:.0%}")
                with cols[3]:
                    st.metric("Avg Faithfulness",   f"{report.avg_faithfulness:.0%}")
                with cols[4]:
                    st.metric("Avg Latency",        f"{report.avg_latency_ms:.0f}ms")

                st.divider()

                # Per-task results table
                import pandas as pd

                df_data = [
                    {
                        "ID":            r.task_id,
                        "Domain OK":     "✅" if r.domain_correct else "❌",
                        "Safe":          "✅" if r.safety_compliant else "❌",
                        "Escalated":     "✅" if r.escalated else "❌",
                        "Completion":    f"{r.task_completion_score:.0%}",
                        "Faithfulness":  f"{r.faithfulness_score:.0%}",
                        "Latency (ms)":  f"{r.latency_ms:.0f}",
                    }
                    for r in results
                ]
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        except Exception as e:
            st.error(f"Evaluation error: {e}")

    else:
        st.info('Click "Run Evaluation" to start the automated evaluation suite.')
