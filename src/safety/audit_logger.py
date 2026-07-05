"""
Audit logger — records every tool call, LLM call, and response to SQLite.
Provides full traceability for safe agent deployment.
"""
import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import get_settings

logger = logging.getLogger(__name__)

class AuditLogger:
    """Thread-safe SQLite audit logger for all agent actions."""
    
    def __init__(self, db_path: Optional[Path] = None):
        settings = get_settings()
        self.db_path = db_path or settings.audit_db_path
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f'AuditLogger initialized at {self.db_path}')
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    node TEXT,
                    tool TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    latency_ms REAL,
                    timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    domain TEXT,
                    response TEXT,
                    confidence REAL,
                    safety_passed INTEGER,
                    timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT,
                    query_count INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
                CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_node ON tool_calls(node);
            ''')
            conn.commit()
    
    def new_session(self) -> str:
        """Start a new audit session."""
        session_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO sessions VALUES (?,?,?)',
                (session_id, datetime.utcnow().isoformat() + 'Z', 0)
            )
            conn.commit()
        return session_id
    
    def log_tool_call(
        self,
        node: str,
        tool: str,
        input_data: dict,
        output_data: Any,
        latency: float,
        session_id: Optional[str] = None
    ) -> str:
        """Log a tool call with full input/output and latency."""
        record_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?)',
                (
                    record_id,
                    session_id or 'default',
                    node,
                    tool,
                    json.dumps(input_data, default=str),
                    json.dumps(output_data, default=str),
                    round(latency * 1000, 2),  # convert to ms
                    datetime.utcnow().isoformat() + 'Z'
                )
            )
            conn.commit()
        return record_id
    
    def log_response(
        self,
        query: str,
        domain: str,
        response: str,
        confidence: float,
        safety_passed: bool,
        session_id: Optional[str] = None
    ) -> str:
        """Log the final agent response."""
        record_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)',
                (
                    record_id,
                    session_id or 'default',
                    query,
                    domain,
                    response,
                    confidence,
                    1 if safety_passed else 0,
                    datetime.utcnow().isoformat() + 'Z'
                )
            )
            conn.commit()
        return record_id
    
    def get_recent_tool_calls(self, limit: int = 50) -> list[dict]:
        """Get recent tool calls from the audit log."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM tool_calls ORDER BY timestamp DESC LIMIT ?', (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
    
    def get_stats(self) -> dict:
        """Get aggregate audit statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total_calls = conn.execute('SELECT COUNT(*) FROM tool_calls').fetchone()[0]
            total_responses = conn.execute('SELECT COUNT(*) FROM responses').fetchone()[0]
            avg_confidence = conn.execute('SELECT AVG(confidence) FROM responses').fetchone()[0] or 0
            safety_rate = conn.execute('SELECT AVG(safety_passed) FROM responses').fetchone()[0] or 0
            avg_latency = conn.execute('SELECT AVG(latency_ms) FROM tool_calls').fetchone()[0] or 0
        
        return {
            'total_tool_calls': total_calls,
            'total_responses': total_responses,
            'avg_confidence': round(avg_confidence, 3),
            'safety_pass_rate': round(safety_rate, 3),
            'avg_tool_latency_ms': round(avg_latency, 2)
        }

_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    global _logger
    if _logger is None:
        _logger = AuditLogger()
    return _logger
