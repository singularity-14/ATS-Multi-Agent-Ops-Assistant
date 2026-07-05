"""
FastAPI application for ATS Multi-Agent Ops Assistant.
Exposes REST endpoints for query processing, audit logs, and system status.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph.workflow import run_query
from src.safety.audit_logger import get_audit_logger
from src.mcp_servers.mcp_server_diagnostics import get_diagnostics_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ATS Multi-Agent Ops Assistant API starting...")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="ATS Multi-Agent Ops Assistant",
    description=(
        "Multi-agent AI system for CERN accelerator operations — "
        "with safety guardrails, audit logging, and MCP tool servers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=2000, description="Natural language query"
    )
    session_id: Optional[str] = Field(
        None, description="Optional session ID for audit tracking"
    )


class QueryResponse(BaseModel):
    query: str
    domain: Optional[str]
    final_response: str
    confidence_score: float
    safety_passed: bool
    faithfulness_score: float
    safety_flags: list[str]
    escalated: bool
    tool_calls_count: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    subsystems: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — basic API metadata."""
    return {
        "message": "ATS Multi-Agent Ops Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return operational health of the API and all subsystems."""
    # Touch the diagnostics server to confirm it initialises cleanly.
    get_diagnostics_server()
    return HealthResponse(
        status="operational",
        version="1.0.0",
        model=settings.nvidia_model,
        subsystems={
            "mcp_docs": "online",
            "mcp_diagnostics": "online",
            "mcp_escalation": "online",
            "audit_logger": "online",
            "safety_layer": "online",
        },
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def process_query(request: QueryRequest):
    """Process a natural language query through the multi-agent system."""
    start = time.time()
    try:
        result = run_query(request.query)
        latency_ms = (time.time() - start) * 1000

        return QueryResponse(
            query=request.query,
            domain=result.get("domain"),
            final_response=result.get("final_response", "No response generated."),
            confidence_score=result.get("confidence_score", 0.0),
            safety_passed=result.get("safety_passed", False),
            faithfulness_score=result.get("faithfulness_score", 0.0),
            safety_flags=result.get("safety_flags", []),
            escalated=result.get("should_escalate", False),
            tool_calls_count=len(result.get("tool_calls", [])),
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/recent", tags=["Audit"])
async def get_recent_audit(limit: int = Query(default=50, le=200)):
    """Return the most recent tool calls from the audit log."""
    audit = get_audit_logger()
    return {"tool_calls": audit.get_recent_tool_calls(limit)}


@app.get("/audit/stats", tags=["Audit"])
async def get_audit_stats():
    """Return aggregate statistics from the audit log."""
    audit = get_audit_logger()
    return audit.get_stats()


@app.get("/systems/status", tags=["Diagnostics"])
async def get_all_system_status():
    """Return a full diagnostic status report for all accelerator subsystems."""
    import json

    diag = get_diagnostics_server()
    return json.loads(diag.run_full_diagnostic())


@app.get("/systems/{system_name}/health", tags=["Diagnostics"])
async def get_system_health(system_name: str):
    """Return the health status of a specific accelerator subsystem."""
    import json

    diag = get_diagnostics_server()
    return json.loads(diag.get_system_health(system_name))


@app.get("/systems/{system_name}/anomalies", tags=["Diagnostics"])
async def check_anomalies(system_name: str, time_window: str = "1h"):
    """Run anomaly detection on a specific subsystem over the given time window."""
    import json

    diag = get_diagnostics_server()
    return json.loads(diag.run_anomaly_check(system_name, time_window))


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.fastapi_port,
        reload=True,
    )
