"""
MCP Escalation Server — human handoff and notification tools.
Exposes tools: escalate_to_human, get_escalation_history, get_oncall_engineer, create_incident_report
"""
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path('data/escalations.db')

# ---------------------------------------------------------------------------
# On-call engineer roster (simulated; production would pull from PagerDuty/CERN HSS)
# ---------------------------------------------------------------------------

ONCALL_ROSTER: list[dict[str, str]] = [
    {
        'name': 'Dr. Elena Marchetti',
        'role': 'LHC Operations Engineer',
        'contact': 'e.marchetti@cern.ch',
        'phone': '+41-22-767-1234',
        'expertise': 'beam operations, injection, dump',
    },
    {
        'name': 'Dr. Kai Hofmann',
        'role': 'Cryogenics Specialist',
        'contact': 'k.hofmann@cern.ch',
        'phone': '+41-22-767-5678',
        'expertise': 'helium cooling, cryostat maintenance, quench recovery',
    },
    {
        'name': 'Dr. Priya Sharma',
        'role': 'RF Systems Engineer',
        'contact': 'p.sharma@cern.ch',
        'phone': '+41-22-767-9012',
        'expertise': 'RF cavity tuning, klystron, beam loading compensation',
    },
    {
        'name': 'Dr. James Okafor',
        'role': 'Beam Diagnostics Expert',
        'contact': 'j.okafor@cern.ch',
        'phone': '+41-22-767-3456',
        'expertise': 'BPM, beam loss monitors, emittance measurement',
    },
]

# Map priority codes to expected response time strings
PRIORITY_RESPONSE_TIMES: dict[str, str] = {
    'CRITICAL': '10 minutes',
    'HIGH': '15 minutes',
    'MEDIUM': '1 hour',
    'LOW': '4 hours',
}


class EscalationServer:
    """MCP-compatible escalation and human handoff server.

    Persists all escalations to a local SQLite database so that history
    survives agent restarts. In a production deployment this would integrate
    with CERN's SNOW (ServiceNow) or a PagerDuty webhook.
    """

    name = 'escalation'

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info('EscalationServer initialized. Database: %s', self.db_path)

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the escalations table if it does not already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS escalations (
                    id              TEXT PRIMARY KEY,
                    reason          TEXT NOT NULL,
                    context         TEXT,
                    priority        TEXT DEFAULT 'MEDIUM',
                    engineer_name   TEXT,
                    engineer_role   TEXT,
                    engineer_contact TEXT,
                    status          TEXT DEFAULT 'OPEN',
                    created_at      TEXT NOT NULL,
                    resolved_at     TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS incident_reports (
                    report_id       TEXT PRIMARY KEY,
                    title           TEXT NOT NULL,
                    description     TEXT,
                    severity        TEXT,
                    affected_systems TEXT,
                    status          TEXT DEFAULT 'OPEN',
                    created_at      TEXT NOT NULL,
                    assigned_team   TEXT
                )
            ''')
            conn.commit()

    # ------------------------------------------------------------------
    # Tool: escalate_to_human
    # ------------------------------------------------------------------

    def escalate_to_human(
        self,
        reason: str,
        context: str,
        priority: str = 'MEDIUM',
    ) -> str:
        """Escalate an issue to an on-call engineer with full context.

        Args:
            reason: Short description of why escalation is needed.
            context: Full context string (agent reasoning, diagnostic output, etc.).
            priority: One of CRITICAL | HIGH | MEDIUM | LOW.

        Returns:
            JSON string with escalation ID, assigned engineer, and SLA info.
        """
        priority = priority.upper()
        if priority not in PRIORITY_RESPONSE_TIMES:
            priority = 'MEDIUM'

        escalation_id = str(uuid.uuid4())[:8].upper()
        created_at = datetime.utcnow().isoformat() + 'Z'

        # Select engineer — simple round-robin based on minute parity; a
        # production system would query a rotation schedule.
        engineer_index = datetime.utcnow().minute % len(ONCALL_ROSTER)
        engineer = ONCALL_ROSTER[engineer_index]

        # Truncate context for DB storage (full context kept in memory/logs)
        context_summary = (
            context[:1000] + '… [truncated]' if len(context) > 1000 else context
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT INTO escalations
                   (id, reason, context, priority, engineer_name, engineer_role,
                    engineer_contact, status, created_at, resolved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (
                    escalation_id,
                    reason,
                    context_summary,
                    priority,
                    engineer['name'],
                    engineer['role'],
                    engineer['contact'],
                    'OPEN',
                    created_at,
                    None,
                ),
            )
            conn.commit()

        response_time = PRIORITY_RESPONSE_TIMES[priority]
        result: dict[str, Any] = {
            'escalation_id': escalation_id,
            'status': 'ESCALATED',
            'priority': priority,
            'assigned_to': engineer,
            'reason': reason,
            'context_summary': context_summary,
            'created_at': created_at,
            'expected_response_time': response_time,
            'message': (
                f'Escalation {escalation_id} created successfully. '
                f'{engineer["name"]} ({engineer["role"]}) has been notified at '
                f'{engineer["contact"]}. '
                f'Expected response time: {response_time}. '
                f'Priority: {priority}.'
            ),
        }

        logger.info(
            'Escalation %s created [%s]: %s', escalation_id, priority, reason[:120]
        )
        return json.dumps(result, indent=2)

    # ------------------------------------------------------------------
    # Tool: get_escalation_history
    # ------------------------------------------------------------------

    def get_escalation_history(self, limit: int = 10) -> str:
        """Get recent escalation history from the database.

        Args:
            limit: Maximum number of records to return (1–100).

        Returns:
            JSON string with count and list of escalation records.
        """
        limit = max(1, min(limit, 100))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM escalations ORDER BY created_at DESC LIMIT ?',
                (limit,),
            ).fetchall()

        history = [dict(row) for row in rows]
        return json.dumps(
            {'count': len(history), 'escalations': history},
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool: get_oncall_engineer
    # ------------------------------------------------------------------

    def get_oncall_engineer(self, specialty: Optional[str] = None) -> str:
        """Get the current on-call engineer, optionally filtered by specialty.

        Args:
            specialty: Optional keyword to filter by role/expertise
                       (e.g. 'cryo', 'RF', 'beam').

        Returns:
            JSON string with the best-matched on-call engineer and full roster.
        """
        roster = ONCALL_ROSTER

        if specialty:
            keyword = specialty.lower()
            filtered = [
                e for e in roster
                if keyword in e['role'].lower() or keyword in e.get('expertise', '').lower()
            ]
            if filtered:
                roster = filtered

        return json.dumps(
            {
                'on_call': roster[0],
                'full_roster': ONCALL_ROSTER,
                'filtered_by': specialty or 'none',
            },
            indent=2,
        )

    # ------------------------------------------------------------------
    # Tool: create_incident_report
    # ------------------------------------------------------------------

    def create_incident_report(
        self,
        title: str,
        description: str,
        severity: str,
        affected_systems: str,
    ) -> str:
        """Create a formal incident report and persist it to the database.

        Args:
            title: Short incident title.
            description: Full incident description with observed symptoms.
            severity: One of CRITICAL | HIGH | MEDIUM | LOW.
            affected_systems: Comma-separated list of affected subsystem names.

        Returns:
            JSON string with the complete incident report record.
        """
        severity = severity.upper()
        report_id = (
            f'INC-{datetime.utcnow().strftime("%Y%m%d")}-'
            f'{str(uuid.uuid4())[:4].upper()}'
        )
        created_at = datetime.utcnow().isoformat() + 'Z'
        systems_list = [s.strip() for s in affected_systems.split(',') if s.strip()]

        next_steps = [
            'Notify affected system operators and shift coordinator.',
            'Initiate automated diagnostic sweep on affected subsystems.',
            'Review SCADA audit logs for root cause identification.',
            'Prepare status update for CERN operations centre.',
            'Document all remediation actions taken in this report.',
        ]
        if severity == 'CRITICAL':
            next_steps.insert(0, '⚠️  CRITICAL: Alert LHC Run Coordinator immediately.')

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                '''INSERT INTO incident_reports
                   (report_id, title, description, severity, affected_systems,
                    status, created_at, assigned_team)
                   VALUES (?,?,?,?,?,?,?,?)''',
                (
                    report_id,
                    title,
                    description,
                    severity,
                    json.dumps(systems_list),
                    'OPEN',
                    created_at,
                    'ATS Operations',
                ),
            )
            conn.commit()

        report: dict[str, Any] = {
            'report_id': report_id,
            'title': title,
            'description': description,
            'severity': severity,
            'affected_systems': systems_list,
            'created_at': created_at,
            'status': 'OPEN',
            'assigned_team': 'ATS Operations',
            'next_steps': next_steps,
        }

        logger.info(
            'Incident report %s created [%s]: %s', report_id, severity, title
        )
        return json.dumps(report, indent=2)

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def get_tools(self) -> dict[str, Any]:
        """Return a mapping of tool names to callable methods."""
        return {
            'escalate_to_human': self.escalate_to_human,
            'get_escalation_history': self.get_escalation_history,
            'get_oncall_engineer': self.get_oncall_engineer,
            'create_incident_report': self.create_incident_report,
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_server: EscalationServer | None = None


def get_escalation_server() -> EscalationServer:
    """Return the module-level singleton EscalationServer instance."""
    global _server
    if _server is None:
        _server = EscalationServer()
    return _server


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = get_escalation_server()
    print(server.get_oncall_engineer('cryo'))
    print(server.escalate_to_human(
        reason='Unexplained quench event in CRYO_SECTOR12',
        context='Anomaly detector flagged 3 HIGH severity spikes at t-5min, t-3min, t-1min.',
        priority='HIGH',
    ))
    print(server.get_escalation_history(5))
