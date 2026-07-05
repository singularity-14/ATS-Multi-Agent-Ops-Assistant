# ⚡ ATS Multi-Agent Ops Assistant

**Multi-agent AI system for CERN accelerator operations** — built with LangGraph, MCP, FastAPI, Streamlit, and NVIDIA NIM LLMs. Implements safe agent deployment patterns including guardrails, faithfulness checking, audit logging, and automated evaluation.

[![CI](https://github.com/singularity-14/ATS-Multi-Agent-Ops-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/singularity-14/ATS-Multi-Agent-Ops-Assistant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5.1-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-76b900.svg)](https://developer.nvidia.com/nim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Directly mirrors the CERN ATS AI Core Team mission: *reusable patterns, evaluation frameworks, and safe agent deployment practices across the accelerator sector.*

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
│              Streamlit UI (port 8501)  ·  FastAPI REST (port 8000)      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  HTTP POST /query
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
         Final Response          Escalation Alert
         (with audit log)        (with audit log)
```

**Key architectural decision**: The Streamlit UI communicates with the LangGraph backend **exclusively through the FastAPI REST API** (`POST /query`). This means both services must be running simultaneously, and the system can be extended with any frontend (web app, CLI, mobile) without touching the agent logic.

---

## 📁 Project Structure

```
ATS-Multi-Agent-Ops-Assistant/
├── src/
│   ├── api/              # FastAPI REST service (port 8000)
│   │   └── main.py       # /query, /health, /audit, /systems endpoints
│   ├── graph/            # LangGraph workflow
│   │   ├── workflow.py   # Graph definition & run_query()
│   │   ├── nodes.py      # All agent node functions
│   │   └── state.py      # AgentState TypedDict
│   ├── agents/           # Agent logic (router, knowledge, diagnostics, status)
│   ├── llm/
│   │   └── nvidia_client.py   # NVIDIA NIM API client (non-reasoning model safe)
│   ├── mcp_servers/      # MCP tool servers (docs, diagnostics, escalation)
│   ├── safety/           # Guardrails, faithfulness checker, audit logger
│   ├── evaluation/       # Evaluation harness & report generator
│   ├── config.py         # Pydantic-settings configuration
│   └── ui/
│       └── app.py        # Streamlit frontend → calls FastAPI via HTTP
├── scripts/
│   ├── run_evaluation.py # CLI evaluation runner
│   ├── start_api.py      # API launcher helper
│   └── start_ui.py       # UI launcher helper
├── tests/                # Pytest unit tests (no real LLM calls)
├── data/
│   ├── docs/             # Accelerator documentation corpus
│   ├── test_cases/
│   │   └── eval_suite.json   # Evaluation test cases
│   └── eval_reports/     # Auto-generated evaluation JSON reports
├── .github/workflows/
│   ├── ci.yml            # Lint → Unit Tests → Docker build → Security scan
│   └── eval.yml          # Weekly automated evaluation (needs NVIDIA_API_KEY secret)
├── .env.example          # Environment variable template
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or 3.12
- An [NVIDIA NIM API key](https://build.nvidia.com) (free tier available)

### 1. Clone & Install

```bash
git clone https://github.com/singularity-14/ATS-Multi-Agent-Ops-Assistant.git
cd ATS-Multi-Agent-Ops-Assistant
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your key:

```env
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
NVIDIA_MODEL=mistralai/mistral-medium-3.5-128b
NVIDIA_API_URL=https://integrate.api.nvidia.com/v1/chat/completions
APP_ENV=development
LOG_LEVEL=INFO
```

> **Important**: The `.env` file must be in the project root (`ATS-Multi-Agent-Ops-Assistant/`), which is also where you run all commands from.

### 3. Start the Backend (FastAPI)

Open **Terminal 1** and run:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify it works: http://localhost:8000/health

### 4. Start the Frontend (Streamlit)

Open **Terminal 2** and run:

```bash
streamlit run src/ui/app.py --server.port 8501
```

Open **http://localhost:8501** in your browser. Type a question and click **Ask**.

> ⚠️ **Both terminals must be running at the same time.** The Streamlit UI sends all queries to the FastAPI backend at `http://localhost:8000/query`.

---

## 🌐 API Reference

The FastAPI backend exposes a full REST API. Interactive docs at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API metadata |
| `GET` | `/health` | Health check for all subsystems |
| `POST` | `/query` | Submit a query to the multi-agent system |
| `GET` | `/audit/recent` | Recent tool calls from audit log |
| `GET` | `/audit/stats` | Aggregate audit statistics |
| `GET` | `/systems/status` | Full diagnostic status for all subsystems |
| `GET` | `/systems/{name}/health` | Health for a specific subsystem |
| `GET` | `/systems/{name}/anomalies` | Anomaly detection for a subsystem |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the LHC beam injection procedure?"}'
```

```json
{
  "query": "What is the LHC beam injection procedure?",
  "domain": "docs",
  "final_response": "The LHC beam injection procedure involves...",
  "confidence_score": 0.87,
  "safety_passed": true,
  "faithfulness_score": 0.82,
  "safety_flags": [],
  "escalated": false,
  "tool_calls_count": 2,
  "latency_ms": 1843.5
}
```

---

## 🤖 Agent Domains

Queries are automatically routed to the appropriate agent:

| Domain | Trigger Examples | Agent Behaviour |
|--------|-----------------|-----------------|
| `docs` | "What is the beam injection procedure?", "Explain magnet quench protection" | BM25 search over documentation corpus → LLM synthesis |
| `diagnostics` | "Check anomalies in RF_CAVITY_A", "Run diagnostics on CRYO_SECTOR12" | Reads live telemetry, runs anomaly detection |
| `status` | "What is the current status of all systems?", "Is LHC_BEAM1 nominal?" | Aggregates real-time subsystem health |
| `escalation` | Queries about beam dumps, emergency stops, safety interlocks | Routes directly to human-in-the-loop |
| `off_topic` | Non-accelerator questions | Politely declined |

---

## 🛡️ Safety Architecture

Every response passes through a two-stage safety gate before being returned:

1. **Guardrails Checker** — keyword and pattern-based filter for unsafe, out-of-scope, or policy-violating content
2. **Faithfulness Checker** — scores the response against the retrieved context to detect hallucination

If either check fails, or if confidence falls below the threshold (default: 0.7), the query is **escalated** — the system flags it for human review rather than returning a potentially incorrect answer.

---

## 🧪 Running Tests

Unit tests do **not** require a real NVIDIA API key:

```bash
pytest tests/test_safety.py tests/test_mcp_servers.py tests/test_evaluation.py \
  -v --override-ini="addopts=" -k "not llm and not nvidia"
```

Run all tests (some may be skipped without a key):

```bash
pytest -v --override-ini="addopts="
```

---

## 📊 Evaluation Harness

Run the offline evaluation suite against the full test case library:

```bash
# Run all test cases
python scripts/run_evaluation.py

# Run a subset of 5 cases (faster, good for development)
python scripts/run_evaluation.py --subset 5
```

The script exits with code `1` if domain accuracy falls below 70%. Reports are saved to `data/eval_reports/`.

> Requires a valid `NVIDIA_API_KEY` in `.env` — this makes real LLM calls.

---

## ⚙️ CI / CD

### GitHub Actions Workflows

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Every push to `main`/`develop`, all PRs | Lint → Unit Tests (3.11 & 3.12) → Docker build → Security scan |
| `eval.yml` | Every Monday 2am UTC, or manually | Runs evaluation suite against live NVIDIA API |

### Setting up CI

For CI to work, add your NVIDIA key as a GitHub secret:

1. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `NVIDIA_API_KEY`, Value: your key

> The `ci.yml` unit tests use a fake key (`test_key_for_unit_tests`) and mock LLM calls — they work without the secret. Only `eval.yml` requires the real key.

---

## 🐳 Docker

```bash
# Build
docker build -t ats-agentic-ops .

# Run API backend
docker run -p 8000:8000 \
  -e NVIDIA_API_KEY=your_key_here \
  ats-agentic-ops

# Or with docker-compose (starts both API and UI)
docker-compose up
```

---

## 🔧 Configuration Reference

All settings are loaded from `.env` (or environment variables). See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | *(required)* | Your NVIDIA NIM API key |
| `NVIDIA_MODEL` | `mistralai/mistral-medium-3.5-128b` | Model identifier |
| `NVIDIA_API_URL` | `https://integrate.api.nvidia.com/v1/chat/completions` | NIM endpoint |
| `APP_ENV` | `development` | Runtime environment |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CONFIDENCE_THRESHOLD` | `0.7` | Minimum score before escalation |
| `MAX_TOKENS` | `4096` | Max LLM completion tokens |
| `TEMPERATURE` | `0.70` | LLM sampling temperature |
| `FASTAPI_PORT` | `8000` | Backend port |
| `STREAMLIT_PORT` | `8501` | Frontend port |
| `AUDIT_DB_PATH` | `data/audit.db` | SQLite audit database path |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
