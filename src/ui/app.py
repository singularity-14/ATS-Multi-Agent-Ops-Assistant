"""
Streamlit UI for ATS Multi-Agent Ops Assistant.
Calls the FastAPI backend at http://localhost:8000 via HTTP requests.
"""
import time
from datetime import datetime

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ATS Ops Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    background: #0d1117;
    font-family: 'Inter', sans-serif;
    color: #e6edf3;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 780px; margin: auto; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #58a6ff, #a371f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.app-header p {
    color: #8b949e;
    font-size: 0.85rem;
    margin: 0;
}

/* ── Input area ── */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}

/* ── Ask button ── */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 12px 28px !important;
    transition: opacity 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Chat bubbles ── */
.bubble-user {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 18px 0 6px 0;
}
.bubble-user .label { color: #58a6ff; font-size: 0.78rem; font-weight: 600; margin-bottom: 4px; }
.bubble-user .text  { color: #e6edf3; font-size: 0.95rem; margin: 0; }

.bubble-ai {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #a371f7;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0 4px 0;
}
.bubble-ai .label { color: #a371f7; font-size: 0.78rem; font-weight: 600; margin-bottom: 6px; }

/* ── Error bubble ── */
.bubble-error {
    background: #1c1010;
    border: 1px solid #f85149;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0 4px 0;
    color: #f85149;
    font-size: 0.9rem;
}

/* ── Divider ── */
hr { border-color: #21262d; margin: 20px 0; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 0 40px 0;
    color: #484f58;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state p { font-size: 0.9rem; margin: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

BACKEND = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>⚡ ATS Ops Assistant</h1>
        <p>CERN Multi-Agent AI · NVIDIA LLM + LangGraph</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input row
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="Ask about LHC operations, diagnostics, system status…",
        key="query_input",
    )
with col_btn:
    submit = st.button("Ask", use_container_width=True)

# ---------------------------------------------------------------------------
# Submit handler
# ---------------------------------------------------------------------------
if submit and query.strip():
    with st.spinner("Processing…"):
        try:
            resp = requests.post(
                f"{BACKEND}/query",
                json={"query": query.strip()},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            st.session_state.history.append(
                {
                    "query": query.strip(),
                    "answer": data.get("final_response", "No response."),
                    "domain": data.get("domain", "unknown"),
                    "confidence": data.get("confidence_score", 0.0),
                    "faithfulness": data.get("faithfulness_score", 0.0),
                    "safety_passed": data.get("safety_passed", False),
                    "safety_flags": data.get("safety_flags", []),
                    "escalated": data.get("escalated", False),
                    "latency_ms": data.get("latency_ms", 0),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            )
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach backend. Make sure uvicorn is running on port 8000.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Backend error {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
if st.session_state.history:
    for i, entry in enumerate(reversed(st.session_state.history)):
        # User bubble
        st.markdown(
            f"""
            <div class="bubble-user">
                <div class="label">You · {entry['timestamp']}</div>
                <p class="text">{entry['query']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI answer
        st.markdown(
            '<div class="bubble-ai"><div class="label">⚡ ATS AI</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(entry["answer"])

        # Collapsed details
        with st.expander("Details"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Confidence", f"{entry['confidence']:.0%}")
            c2.metric("Faithfulness", f"{entry['faithfulness']:.0%}")
            c3.metric("Latency", f"{entry['latency_ms']:.0f}ms")
            c4.metric("Safety", "✅ Pass" if entry["safety_passed"] else "⚠️ Flag")
            if entry["safety_flags"]:
                st.warning("Flags: " + ", ".join(entry["safety_flags"]))

        if i < len(st.session_state.history) - 1:
            st.markdown("<hr>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">⚡</div>
            <p>Type a question above and press <strong>Ask</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
