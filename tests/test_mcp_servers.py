"""
Tests for MCP tool servers: docs, diagnostics, escalation.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from src.mcp_servers.mcp_server_diagnostics import DiagnosticsServer, SUBSYSTEMS
from src.mcp_servers.mcp_server_escalation import EscalationServer

class TestDiagnosticsServer:
    def setup_method(self):
        self.server = DiagnosticsServer()
    
    def test_run_anomaly_check_known_system(self):
        result = self.server.run_anomaly_check('LHC_BEAM1', '1h')
        data = json.loads(result)
        assert data['system'] == 'LHC_BEAM1'
        assert 'anomalies_detected' in data
        assert 'status' in data
        assert data['status'] in ['NOMINAL', 'WARNING', 'ALERT']
    
    def test_run_anomaly_check_unknown_system(self):
        result = self.server.run_anomaly_check('UNKNOWN_SYS', '1h')
        assert 'Unknown system' in result
    
    def test_get_system_health_known(self):
        result = self.server.get_system_health('RF_CAVITY_A')
        data = json.loads(result)
        assert data['subsystem'] == 'RF_CAVITY_A'
        assert 'health' in data
        assert 'health_score' in data['health']
    
    def test_get_system_health_unknown(self):
        result = self.server.get_system_health('NONEXISTENT')
        data = json.loads(result)
        assert 'error' in data
    
    def test_get_telemetry_history(self):
        result = self.server.get_telemetry_history('LHC_BEAM1', 'current', 24)
        data = json.loads(result)
        assert data['system'] == 'LHC_BEAM1'
        assert len(data['data']) == 24
    
    def test_run_full_diagnostic(self):
        result = self.server.run_full_diagnostic()
        data = json.loads(result)
        assert 'overall_health' in data
        assert 'systems_ok' in data
        assert 'subsystems' in data
        assert len(data['subsystems']) == len(SUBSYSTEMS)
    
    def test_get_tools_returns_all(self):
        tools = self.server.get_tools()
        assert 'run_anomaly_check' in tools
        assert 'get_system_health' in tools
        assert 'get_telemetry_history' in tools
        assert 'run_full_diagnostic' in tools

class TestEscalationServer:
    def setup_method(self):
        import tempfile
        from pathlib import Path
        self.tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.server = EscalationServer(db_path=Path(self.tmp_db.name))
    
    def test_escalate_to_human(self):
        result = self.server.escalate_to_human(
            reason='High anomaly detected',
            context='LHC_BEAM1 showing 5-sigma deviation',
            priority='HIGH'
        )
        data = json.loads(result)
        assert data['status'] == 'ESCALATED'
        assert 'escalation_id' in data
        assert 'assigned_to' in data
    
    def test_get_oncall_engineer(self):
        result = self.server.get_oncall_engineer()
        data = json.loads(result)
        assert 'on_call' in data
        assert 'name' in data['on_call']
    
    def test_create_incident_report(self):
        result = self.server.create_incident_report(
            title='LHC Beam Loss',
            description='Unexpected beam loss in sector 3',
            severity='HIGH',
            affected_systems='LHC_BEAM1,MAGNET_ARC12'
        )
        data = json.loads(result)
        assert 'report_id' in data
        assert data['report_id'].startswith('INC-')
        assert 'next_steps' in data
