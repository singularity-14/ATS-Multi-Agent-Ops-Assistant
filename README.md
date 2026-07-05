# ⚡ ATS Multi-Agent Ops Assistant

**Multi-agent AI system for CERN accelerator operations** — built with LangGraph, MCP, FastAPI, Streamlit, and NVIDIA NIM LLMs. Implements safe agent deployment patterns including guardrails, faithfulness checking, audit logging, and automated evaluation.

[![CI](https://github.com/singularity-14/ATS-Multi-Agent-Ops-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/singularity-14/ATS-Multi-Agent-Ops-Assistant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5.1-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-76b900.svg)](https://developer.nvidia.com/nim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Is

The **ATS Multi-Agent Ops Assistant** is a production-grade, multi-agent AI system designed to support operators at CERN's Accelerator and Technology Sector (ATS). It provides intelligent, grounded answers to questions about accelerator documentation, subsystem diagnostics, and real-time operational status — with a mandatory human-in-the-loop escalation path for safety-critical situations.

### Problems It Solves

| Problem | Solution |
|---------|----------|
| Operators need instant answers from thousands of pages of accelerator documentation | RAG-based Knowledge Agent with BM25 search over structured docs corpus |
| Shift engineers manually check dozens of subsystems for anomalies | Diagnostics Agent with automated anomaly detection across 10 LHC subsystems |
| AI responses can hallucinate in high-stakes environments | Faithfulness checker + confidence thresholds + mandatory escalation gates |
| No audit trail for AI decisions in safety-critical contexts | Full SQLite audit log of every tool call, LLM invocation, and final response |
| Hard to know when to trust the AI vs. call a human | Graduated confidence scoring with automatic escalation below threshold |
| Hard to evaluate multi-agent system quality systematically | Built-in evaluation harness measuring 6 independent metrics across a test suite |

### Design Philosophy

This system is built around three principles drawn from the CERN ATS AI Core Team mandate:

1. **Safety-first**: Every response passes through guardrails and faithfulness checks before reaching the user. Low-confidence or policy-violating responses are escalated to human engineers, never silently returned.

2. **Reusable patterns**: Each component (MCP server pattern, safety layer, evaluation harness) is designed to be extracted and reused across other CERN systems or accelerator facilities.

3. **Full observability**: Every LLM call, tool invocation, and routing decision is persisted to a queryable audit log, enabling post-hoc review and compliance reporting.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User / Operator Interface                        │
│                   (Streamlit UI  ·  FastAPI REST  ·  CLI)               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  query
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     LangGraph State Machine                             │
│                                                                         │
│   ┌──────────┐   off_topic ──────────────────────────────► END         │
│   │  ROUTER  │   escalation ─────────────────────────────► END         │
│   │  (LLM)   │                                                          │
│   └────┬─────┘                                                          │
│        │  domain: docs / diagnostics / status                           │
│        ▼                                                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐                      │
│  │  KNOWLEDGE  │  │ DIAGNOSTICS  │  │  STATUS   │  ← Domain Agents     │
│  │    AGENT    │  │    AGENT     │  │   AGENT   │                      │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘                      │
│         └────────────────┼────────────────┘                             │
│                          │  agent_response + retrieved_context           │
│                          ▼                                              │
│                   ┌─────────────┐                                       │
│                   │   SAFETY    │  ← Guardrails + Faithfulness Check    │
│                   │    GATE     │                                        │
│                   └──────┬──────┘                                       │
│                          │                                              │
│              ┌───────────┴───────────┐                                  │
│              │ safety_passed         │ !safety_passed / low confidence   │
│              ▼                       ▼                                  │
│        ┌──────────┐          ┌────────────────┐                         │
│        │ FINALIZE │          │   ESCALATION   │ → Human Engineer        │
│        └────┬─────┘          └───────┬────────┘                         │
│             │                        │                                  │
└─────────────┼────────────────────────┼──────────────────────────────────┘
              │                        │
              ▼                        ▼
         final_response          escalation ticket
              │                  + engineer contact
              ▼                        │
    ┌─────────────────┐               ▼
    │  Audit Logger   │◄─────  All paths logged
    │   (SQLite)      │        to data/audit.db
    └─────────────────┘

MCP Tool Servers (called by domain agents):
┌──────────────────┐  ┌────────────────────┐  ┌─────────────────────┐
│  Docs Server     │  │ Diagnostics Server │  │  Escalation Server  │
│ ─ search_docs    │  │ ─ run_anomaly_check│  │ ─ escalate_to_human │
│ ─ get_doc_by_id  │  │ ─ get_sys_health   │  │ ─ get_oncall_eng    │
│ ─ list_categories│  │ ─ get_telemetry    │  │ ─ create_incident   │
└──────────────────┘  │ ─ run_full_diag    │  │ ─ get_esc_history   │
                      └────────────────────┘  └─────────────────────┘
```

### Graph Topology

The LangGraph state machine has exactly three types of paths:

```
Happy path:    router → knowledge   → safety → finalize → END
               router → diagnostics → safety → finalize → END
               router → status      → safety → finalize → END

Short-circuit: router → off_topic  → END   (non-accelerator query)
               router → escalation → END   (explicit human request)
               safety → escalation → END   (safety gate failed)
```

---

## 📦 Project Structure

```
ats-agentic-ops-assistant/
│
├── 📄 .env.example              # Template for all environment variables
├── 📄 .gitignore                # Standard Python gitignore
├── 📄 pyproject.toml            # Project metadata, linting (Black, Ruff, Mypy), pytest config
├── 📄 requirements.txt          # Pinned dependencies
├── 📄 README.md                 # This file
├── 📄 ARCHITECTURE.md           # Deep-dive architecture reference
│
└── src/                         # Main Python package
    ├── __init__.py
    ├── config.py                # Pydantic-settings configuration (all env vars)
    │
    ├── llm/                     # LLM client layer
    │   ├── __init__.py
    │   └── nvidia_client.py     # NvidiaLLMClient — NIM REST wrapper with retry logic
    │
    ├── graph/                   # LangGraph state machine
    │   ├── __init__.py
    │   ├── state.py             # AgentState TypedDict — shared state schema
    │   ├── nodes.py             # All 7 node implementations (router, knowledge, …)
    │   └── workflow.py          # Graph assembly, conditional routing, run_query()
    │
    ├── mcp_servers/             # Model Context Protocol tool servers
    │   ├── __init__.py
    │   ├── mcp_server_docs.py       # DocumentationServer — RAG search over docs corpus
    │   ├── mcp_server_diagnostics.py # DiagnosticsServer — anomaly detection, telemetry
    │   └── mcp_server_escalation.py  # EscalationServer — human handoff, incident reports
    │
    ├── safety/                  # Safety layer (all checks happen here)
    │   ├── __init__.py
    │   ├── guardrails.py        # GuardrailsChecker — topic + content allowlists
    │   ├── faithfulness.py      # FaithfulnessChecker — TF-based grounding score
    │   └── audit_logger.py      # AuditLogger — SQLite persistence for all events
    │
    └── evaluation/              # Evaluation harness
        ├── __init__.py
        ├── evaluator.py         # AgentEvaluator — 6-metric evaluation pipeline
        └── reports.py           # ReportGenerator — rich terminal + JSON reports

data/                            # Runtime data (git-ignored)
    ├── audit.db                 # SQLite audit database
    ├── escalations.db           # SQLite escalation records
    ├── docs/
    │   └── accelerator_docs.json  # Documentation corpus (JSON)
    └── eval_reports/
        └── eval_YYYYMMDD_HHMMSS.json  # Evaluation run snapshots
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11 or 3.12
- An [NVIDIA NIM API key](https://build.nvidia.com/) (free tier available)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/singularity-14/ATS-Multi-Agent-Ops-Assistant.git
cd ATS-Multi-Agent-Ops-Assistant
```

### 2. Create your environment file

```bash
cp .env.example .env
```

Open `.env` and set your NVIDIA API key:

```bash
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
```

### 3. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell

# Install all dependencies
pip install -r requirements.txt

# Or install as a package (editable mode for development)
pip install -e .
```

### 4. Run the FastAPI backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 5. Run the Streamlit UI

```bash
# In a second terminal
streamlit run src/ui/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

### 6. Run a quick test query (Python)

```python
from src.graph.workflow import run_query

result = run_query("What is the current status of LHC_BEAM1?")
print(result["final_response"])
print(f"Domain: {result['domain']}")
print(f"Confidence: {result['confidence_score']:.0%}")
print(f"Safety passed: {result['safety_passed']}")
```

### 7. Run the evaluation harness

```bash
python scripts/run_evaluation.py
```

---

## 🔧 Configuration

All configuration is managed via environment variables (loaded from `.env`). The application uses **pydantic-settings** for type-safe validation at startup — missing required variables raise an informative error immediately.

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | *(required)* | Your NVIDIA NIM API key — get one at [build.nvidia.com](https://build.nvidia.com/) |
| `NVIDIA_MODEL` | `mistralai/mistral-medium-3.5-128b` | NVIDIA-hosted model identifier. Any NIM-compatible model works. |
| `NVIDIA_API_URL` | `https://integrate.api.nvidia.com/v1/chat/completions` | NIM chat completions endpoint |
| `APP_ENV` | `development` | Runtime environment: `development` \| `staging` \| `production` |
| `LOG_LEVEL` | `INFO` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `AUDIT_DB_PATH` | `data/audit.db` | Path to the SQLite audit database (created automatically) |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence score `[0.0–1.0]` to accept a response without escalation |
| `MAX_TOKENS` | `16384` | Maximum completion tokens per LLM call |
| `TEMPERATURE` | `0.70` | LLM sampling temperature `[0.0–2.0]` |
| `FASTAPI_PORT` | `8000` | Port for the FastAPI REST service |
| `STREAMLIT_PORT` | `8501` | Port for the Streamlit UI |

> **Tip**: Set `CONFIDENCE_THRESHOLD=0.5` during local development to reduce escalations while testing. Raise it back to `0.7`+ for any production or demo deployment.

---

## 🤖 Agent Architecture

The system uses **7 LangGraph nodes**, each implemented as a pure function `AgentState → dict`. Nodes return only the fields they update — LangGraph merges partial updates into the shared state automatically, making each node independently testable.

### `router_node` — Query Classifier

**Purpose**: Classify the incoming query into one of five domains using a fast, low-cost LLM call.

**How it works**:
1. Sends the query to the LLM with `reasoning_effort='low'` to minimise routing latency.
2. Expects a structured JSON response: `{"domain": "...", "confidence": 0.0-1.0, "reasoning": "..."}`.
3. Falls back to `docs` domain on any JSON parse failure.
4. Triggers `should_escalate=True` if confidence < `CONFIDENCE_THRESHOLD`.

**Domain taxonomy**:

| Domain | Triggers | Next node |
|--------|----------|-----------|
| `docs` | How-to questions, procedures, protocol lookups | `knowledge` |
| `diagnostics` | Anomaly reports, health checks, telemetry queries | `diagnostics` |
| `status` | Operational status sweeps, overall system health | `status` |
| `escalation` | Explicit human help requests, emergencies | `escalation` (direct) |
| `off_topic` | Anything outside accelerator/physics operations | `off_topic` (direct) |

---

### `knowledge_node` — Documentation Agent

**Purpose**: Answer questions grounded in the accelerator documentation corpus using RAG.

**How it works**:
1. Calls `DocumentationServer.search_accelerator_docs(query, top_k=3)` — BM25-style keyword search.
2. Sends retrieved context + query to the LLM with a strict system prompt: *"answer ONLY from context, cite procedures, never hallucinate"*.
3. Extracts a self-reported confidence tag (`Confidence: HIGH/MEDIUM/LOW`) from the LLM response and converts it to a numeric score (0.90 / 0.70 / 0.50).

**System prompt design** (excerpt):
> *"If the context does not contain a clear answer, state that explicitly — do NOT hallucinate or invent procedures."*

---

### `diagnostics_node` — Anomaly Detection Agent

**Purpose**: Analyse telemetry data and generate actionable diagnostic assessments for a specific subsystem.

**How it works**:
1. Uses a low-cost LLM call to extract the subsystem name from the query (e.g., `"LHC_BEAM1"`, `"CRYO_SECTOR12"`).
2. Calls `DiagnosticsServer.run_anomaly_check(system, "1h")` — simulates 20 data points with 8% anomaly probability and Gaussian noise.
3. Calls `DiagnosticsServer.get_system_health(system)` — returns health score, live parameters, and active alerts.
4. Sends combined telemetry to LLM for narrative assessment: severity, root-cause hypothesis, recommended actions.

**Supported subsystems**: `LHC_BEAM1`, `LHC_BEAM2`, `CRYO_SECTOR12`, `CRYO_SECTOR34`, `RF_CAVITY_A`, `RF_CAVITY_B`, `VACUUM_IR1`, `VACUUM_IR5`, `MAGNET_ARC12`, `COLLIMATOR_TCP`

---

### `status_node` — Operational Status Agent

**Purpose**: Generate a cross-system operational status report covering all monitored subsystems.

**How it works**:
1. Calls `DiagnosticsServer.run_full_diagnostic()` — sweeps all 10 subsystems and computes aggregate health.
2. Sends the full sweep result to the LLM with a structured report template: overall health → degraded systems → action items → nominal systems.
3. Returns `confidence_score=0.95` — live data has inherently high trust.

---

### `safety_node` — Quality Gate

**Purpose**: Evaluate the domain agent's response through two independent safety checks before it reaches the user.

**Check 1 — Guardrails** (`GuardrailsChecker`):
- Scans query and response for blocked content patterns (violence, off-topic domains, harmful content).
- Verifies topic relevance via keyword allowlist overlap.
- Checks response length sanity (rejects responses < 10 characters).

**Check 2 — Faithfulness** (`FaithfulnessChecker`):
- Computes a weighted token-overlap score between the LLM response and the retrieved context.
- Score formula: `0.7 × precision + 0.3 × F1` where precision = fraction of response tokens found in context.
- Flags any response with score < 0.5 as potentially hallucinated.

**Escalation logic**:
- `should_escalate = upstream_escalate OR !safety_passed OR low_confidence`
- If escalation is triggered, sets `escalation_reason` for the escalation node.

---

### `escalation_node` — Human Handoff

**Purpose**: Create a structured escalation ticket and return a user-facing acknowledgement with engineer contact details.

**How it works**:
1. Determines priority: `HIGH` if safety flags exist, otherwise `MEDIUM`.
2. Calls `EscalationServer.escalate_to_human(reason, context, priority)`.
3. Persists the escalation to `data/escalations.db` with full context.
4. Assigns an on-call engineer from a round-robin roster of 4 domain specialists.
5. Returns a formatted markdown response with escalation ID, engineer name, contact, and SLA.

**On-call roster** (simulated, production would integrate with PagerDuty/CERN HSS):
- Dr. Elena Marchetti — LHC Operations Engineer (beam, injection, dump)
- Dr. Kai Hofmann — Cryogenics Specialist (helium, cryostat, quench recovery)
- Dr. Priya Sharma — RF Systems Engineer (cavity tuning, klystron)
- Dr. James Okafor — Beam Diagnostics Expert (BPM, beam loss monitors)

---

### `off_topic_node` — Scope Boundary

**Purpose**: Return a polite, informative out-of-scope notice without any LLM invocation.

This is a static node — no LLM call, no tool call, zero marginal cost. It tells the user what the system *can* help with and redirects them appropriately.

---

### `finalize_node` — Response Formatter

**Purpose**: Format the verified agent response with a metadata footer and persist it to the audit log.

The footer appended to every successful response:
```
---
*ATS AI Assistant · Domain: **diagnostics** · Confidence: **90%** · Faithfulness: **85%** · Tools used: **4***
```

---

## 🔧 MCP Tool Servers

The system uses the **Model Context Protocol (MCP)** pattern to decouple tool implementations from agent logic. Each server is a Python class with a `get_tools()` registry and a module-level singleton accessor. This makes them independently deployable and testable.

### `DocumentationServer` (`mcp_server_docs.py`)

Provides RAG-based search over the accelerator documentation corpus stored in `data/docs/accelerator_docs.json`.

| Tool | Signature | Description |
|------|-----------|-------------|
| `search_accelerator_docs` | `(query: str, top_k: int = 3) → str` | BM25-style keyword search; returns top-k ranked documents |
| `get_document_by_id` | `(doc_id: str) → str` | Fetch a specific document by its unique ID |
| `list_doc_categories` | `() → str` | List all categories with document counts |

**Example call**:
```python
from src.mcp_servers.mcp_server_docs import get_docs_server

server = get_docs_server()
results = server.search_accelerator_docs("beam injection energy ramp", top_k=3)
print(results)
```

---

### `DiagnosticsServer` (`mcp_server_diagnostics.py`)

Provides simulated but physically plausible telemetry for 10 LHC-style subsystems. In a production deployment, this would connect to CERN's SCADA (WinCC OA) or the Logging and Archiving (NXCALS) system.

| Tool | Signature | Description |
|------|-----------|-------------|
| `run_anomaly_check` | `(system: str, time_window: str = "1h") → str` | Generates 20 synthetic data points; flags 3–6σ deviations as anomalies |
| `get_system_health` | `(subsystem: str) → str` | Returns health score (85–100%), live parameters, and active alerts |
| `get_telemetry_history` | `(system: str, metric: str, hours: int = 24) → str` | Historical telemetry as a drifting time series |
| `run_full_diagnostic` | `(include_all: bool = True) → str` | Cross-system sweep; returns per-subsystem status and aggregate summary |

**Example call**:
```python
from src.mcp_servers.mcp_server_diagnostics import get_diagnostics_server

server = get_diagnostics_server()
anomalies = server.run_anomaly_check("LHC_BEAM1", "1h")
health = server.get_system_health("RF_CAVITY_A")
full_status = server.run_full_diagnostic()
```

---

### `EscalationServer` (`mcp_server_escalation.py`)

Manages human handoff workflows. Persists all escalations and incident reports to `data/escalations.db`. Production integration points: CERN ServiceNow (SNOW), PagerDuty.

| Tool | Signature | Description |
|------|-----------|-------------|
| `escalate_to_human` | `(reason, context, priority="MEDIUM") → str` | Creates escalation ticket, assigns on-call engineer, persists to DB |
| `get_escalation_history` | `(limit: int = 10) → str` | Retrieves recent escalations ordered by creation time |
| `get_oncall_engineer` | `(specialty: str = None) → str` | Returns best-matched on-call engineer for given specialty |
| `create_incident_report` | `(title, description, severity, affected_systems) → str` | Formal incident report with auto-generated next steps |

**Priority SLAs**:

| Priority | Expected Response |
|----------|------------------|
| `CRITICAL` | 10 minutes |
| `HIGH` | 15 minutes |
| `MEDIUM` | 1 hour |
| `LOW` | 4 hours |

**Example call**:
```python
from src.mcp_servers.mcp_server_escalation import get_escalation_server

server = get_escalation_server()
ticket = server.escalate_to_human(
    reason="Quench event detected in CRYO_SECTOR12",
    context="3 HIGH severity anomalies at t-5min, t-3min, t-1min.",
    priority="HIGH"
)
print(ticket)  # JSON with escalation ID, engineer contact, SLA
```

---

## 🛡️ Safety Framework

Safety is a first-class architectural concern, not an afterthought. Every non-trivial response passes through a two-stage safety gate before reaching the user.

### Stage 1: Topic Guardrails (`GuardrailsChecker`)

**Allowlist-based topic filtering**: The query must contain at least one term from a 50-term accelerator domain vocabulary (beam, magnet, quench, cryogenic, RF, luminosity, etc.). Queries with no domain overlap and more than 5 words are flagged.

**Blocklist pattern matching**: Both the query and the agent response are scanned against 6 regex patterns covering: food/cooking, politics/elections, financial markets, relationships, violence, and illegal activity. A match in either the input or output triggers a flag.

**Response sanity**: Responses shorter than 10 characters are flagged as likely errors.

### Stage 2: Faithfulness Checking (`FaithfulnessChecker`)

**Algorithm** (no external model required):
1. Tokenize both the response and the retrieved context (lowercase, strip stopwords and punctuation).
2. Compute `precision = |response_tokens ∩ context_tokens| / |response_tokens|`
3. Compute `recall = |response_tokens ∩ context_tokens| / |context_tokens|`
4. Compute `F1 = 2 × precision × recall / (precision + recall)`
5. Final score: **`0.7 × precision + 0.3 × F1`** (precision-weighted to penalise hallucinated terms)

| Score Range | Assessment | Action |
|-------------|------------|--------|
| ≥ 0.7 | HIGH — fully grounded | Accept |
| 0.5–0.69 | MEDIUM — mostly grounded | Accept with flag |
| < 0.5 | LOW — potential hallucination | Escalate |

### Stage 3: Confidence Threshold Gate

The `safety_node` enforces an aggregate check:

```python
should_escalate = (
    upstream_escalate          # Router already flagged it
    or not safety_passed       # Guardrails or faithfulness failed
    or confidence < THRESHOLD  # Agent self-reported low confidence
)
```

The default threshold is `0.7`, configurable via `CONFIDENCE_THRESHOLD`.

### Escalation Priority

When escalation is triggered by safety failures (safety flags present), the ticket is automatically set to `HIGH` priority (15-minute SLA). Confidence-only escalations default to `MEDIUM` (1-hour SLA).

---

## 🧪 Evaluation Harness

The `AgentEvaluator` provides a systematic, repeatable way to measure system quality across 6 independent metrics. This enables tracking regressions and validating improvements as the system evolves.

### Running Evaluations

```bash
# Run full evaluation suite
python scripts/run_evaluation.py

# Run with verbose per-task output
python -c "
from src.evaluation.evaluator import AgentEvaluator
from src.graph.workflow import run_query

evaluator = AgentEvaluator()
test_suite = [
    {
        'id': 'T001',
        'query': 'What is the LHC injection energy?',
        'expected_domain': 'docs',
        'expected_keywords': ['injection', 'energy', 'GeV'],
        'expected_safe': True,
    },
    {
        'id': 'T002',
        'query': 'Is there an anomaly in LHC_BEAM1?',
        'expected_domain': 'diagnostics',
        'expected_keywords': ['anomaly', 'beam'],
        'expected_safe': True,
    },
    {
        'id': 'T003',
        'query': 'What is the weather like today?',
        'expected_domain': 'off_topic',
        'expected_safe': True,  # off_topic queries are gracefully declined, not safety violations
    },
]
report = evaluator.run_full_eval(test_suite, run_query, verbose=True)
print(report.to_dict())
"
```

### Metrics Measured

| Metric | Definition | Target |
|--------|------------|--------|
| **Domain Accuracy** | Fraction of queries routed to the correct domain | ≥ 85% |
| **Task Completion Score** | Keyword coverage × response quality `[0.0–1.0]` | ≥ 0.70 |
| **Safety Compliance Rate** | Fraction of responses passing all safety checks | ≥ 95% |
| **Average Faithfulness** | Mean faithfulness score across all responses | ≥ 0.60 |
| **Average Latency** | Mean end-to-end wall-clock time in milliseconds | Informational |
| **Escalation Rate** | Fraction of queries that triggered human escalation | Informational |

### Test Case Schema

Each test case in the suite is a plain Python dict:

```python
{
    "id": "T042",                          # Unique test identifier
    "query": "What caused the quench?",    # Natural language query
    "expected_domain": "diagnostics",      # Expected routing target
    "expected_keywords": ["quench", "temperature", "sector"],  # Response must contain these
    "expected_safe": True,                 # True = should be answered, False = should be rejected
}
```

### Adding New Test Cases

1. Add entries to your test suite list following the schema above.
2. Set `expected_safe=False` for queries that should be rejected (off-topic or harmful).
3. Leave `expected_keywords` empty to use length-based scoring instead of keyword coverage.
4. Run the evaluator and check the per-task table for failures.

### Report Output

The `ReportGenerator` produces two outputs:
- **Rich terminal table** — colour-coded ✅/❌ per metric with pass thresholds.
- **JSON file** — saved to `data/eval_reports/eval_YYYYMMDD_HHMMSS.json` for CI integration or long-term tracking.

---

## 🚀 API Reference

The FastAPI service exposes a REST interface over the full agent workflow.

### `POST /query`

Run a natural-language query through the full agent pipeline.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the current status of RF_CAVITY_A?"}'
```

**Response**:
```json
{
  "final_response": "## RF Cavity A — Operational Status\n...",
  "domain": "status",
  "confidence_score": 0.95,
  "faithfulness_score": 0.82,
  "safety_passed": true,
  "should_escalate": false,
  "tool_calls": [
    {"node": "router", "tool": "classify_query", "latency": 0.312},
    {"node": "status", "tool": "run_full_diagnostic", "latency": 0.004},
    {"node": "status", "tool": "llm_generate", "latency": 1.847}
  ]
}
```

### `GET /health`

Health check — verifies API is up and LLM connectivity.

```bash
curl http://localhost:8000/health
```

### `GET /audit/recent`

Return the 50 most recent tool calls from the audit log.

```bash
curl http://localhost:8000/audit/recent
```

### `GET /audit/stats`

Return aggregate audit statistics.

```bash
curl http://localhost:8000/audit/stats
```

**Response**:
```json
{
  "total_tool_calls": 1284,
  "total_responses": 214,
  "avg_confidence": 0.847,
  "safety_pass_rate": 0.963,
  "avg_tool_latency_ms": 423.7
}
```

### `GET /escalations`

Return recent escalation history from `data/escalations.db`.

```bash
curl "http://localhost:8000/escalations?limit=10"
```

---

## 📊 Monitoring & Audit

### Audit Database Schema

All agent activity is persisted to `data/audit.db` (SQLite). The database is created automatically on first run.

**`tool_calls` table** — one row per tool or LLM invocation:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `session_id` | TEXT | Groups calls within a single query run |
| `node` | TEXT | Agent node name (router, knowledge, safety, …) |
| `tool` | TEXT | Tool name (classify_query, run_anomaly_check, llm_generate, …) |
| `input_data` | TEXT (JSON) | Serialised input arguments |
| `output_data` | TEXT (JSON) | Serialised output / result summary |
| `latency_ms` | REAL | Wall-clock latency in milliseconds |
| `timestamp` | TEXT (ISO 8601) | UTC timestamp |

**`responses` table** — one row per completed query:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `session_id` | TEXT | Session identifier |
| `query` | TEXT | Original user query |
| `domain` | TEXT | Classified domain |
| `response` | TEXT | Final formatted response |
| `confidence` | REAL | Agent confidence score |
| `safety_passed` | INTEGER | 1 = passed, 0 = failed |
| `timestamp` | TEXT (ISO 8601) | UTC timestamp |

**`sessions` table** — one row per session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Session ID |
| `started_at` | TEXT | Session start time |
| `query_count` | INTEGER | Number of queries in session |

### Querying the Audit Log

```python
import sqlite3

conn = sqlite3.connect("data/audit.db")
conn.row_factory = sqlite3.Row

# Most common tools used
for row in conn.execute("""
    SELECT node, tool, COUNT(*) as calls, AVG(latency_ms) as avg_ms
    FROM tool_calls
    GROUP BY node, tool
    ORDER BY calls DESC
"""):
    print(dict(row))

# Safety pass rate by domain
for row in conn.execute("""
    SELECT domain, AVG(safety_passed) as pass_rate, AVG(confidence) as avg_conf
    FROM responses
    GROUP BY domain
"""):
    print(dict(row))
```

```bash
# Quick stats from the CLI
sqlite3 data/audit.db "
SELECT
  (SELECT COUNT(*) FROM tool_calls)  AS total_tool_calls,
  (SELECT COUNT(*) FROM responses)   AS total_responses,
  (SELECT AVG(safety_passed) FROM responses) AS safety_pass_rate;
"
```

---

## 🐳 Docker Deployment

### Build and run with Docker

> The `Dockerfile` is already included in the repository root. No changes needed.

```bash
docker build -t ats-ops-assistant .
docker run -p 8000:8000 -p 8501:8501 \
  -e NVIDIA_API_KEY=nvapi-xxxx \
  -v $(pwd)/data:/app/data \
  ats-ops-assistant
```

### Docker Compose (API + UI)

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - APP_ENV=production
      - CONFIDENCE_THRESHOLD=0.7
    volumes:
      - ./data:/app/data
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ui:
    build: .
    ports:
      - "8501:8501"
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
      - API_BASE_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy
    command: streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0
```

```bash
# Start all services
NVIDIA_API_KEY=nvapi-xxxx docker compose up -d

# View logs
docker compose logs -f api

# Tear down
docker compose down
```

---

## 🤝 Contributing

### Adding a New MCP Server

1. Create `src/mcp_servers/mcp_server_<domain>.py`.
2. Implement a class with a `name` attribute, tool methods, and a `get_tools() → dict` registry.
3. Add a module-level singleton: `_server = None` + `def get_server() → YourServer`.
4. Import and call your server from the appropriate node in `src/graph/nodes.py`.
5. Add test cases to the evaluation suite targeting the new domain.

### Adding a New Agent Node

1. Implement your node as `def my_node(state: AgentState) → dict` in `src/graph/nodes.py`.
2. Register it: `graph.add_node("my_node", my_node)` in `src/graph/workflow.py`.
3. Wire edges: `graph.add_edge("my_node", "safety")` for domain agents, or `graph.add_edge("my_node", END)` for terminal nodes.
4. Update `route_after_router` or add a new routing function + `add_conditional_edges` call.

### Extending the Safety Layer

- **New guardrail rule**: Add a regex to `BLOCKED_PATTERNS` in `src/safety/guardrails.py`, or add a new check method to `GuardrailsChecker.check()`.
- **Better faithfulness scoring**: Replace the TF-based scorer in `FaithfulnessChecker.score()` with a sentence-transformer cosine similarity for higher accuracy.
- **New safety check type**: Implement a new checker class and invoke it inside `safety_node` in `src/graph/nodes.py`.

### Adding Evaluation Test Cases

Edit or extend the test suite dict list in your evaluation script. Follow the schema:
```python
{"id": "T099", "query": "...", "expected_domain": "...", "expected_keywords": [...], "expected_safe": True}
```

### Code Quality

```bash
# Format
black src/ --line-length 100

# Lint
ruff check src/

# Type check
mypy src/

# Test
pytest tests/ -v --cov=src
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
