"""
MCP Diagnostics Server — anomaly detection and system health tools.
Exposes tools: run_anomaly_check, get_system_health, get_telemetry_history, run_full_diagnostic
"""
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accelerator subsystem definitions with realistic nominal parameters
# ---------------------------------------------------------------------------

SUBSYSTEMS: dict[str, dict[str, Any]] = {
    'LHC_BEAM1': {
        'nominal_current': 0.582,   # A
        'nominal_energy': 6800,     # GeV
        'max_temp': 1.9,            # K (superconducting threshold)
        'status': 'operational',
    },
    'LHC_BEAM2': {
        'nominal_current': 0.582,
        'nominal_energy': 6800,
        'max_temp': 1.9,
        'status': 'operational',
    },
    'CRYO_SECTOR12': {
        'nominal_temp': 1.9,        # K
        'max_temp': 4.5,            # K (He-II lambda point)
        'pressure': 1e-3,           # mbar
        'status': 'operational',
    },
    'CRYO_SECTOR34': {
        'nominal_temp': 1.9,
        'max_temp': 4.5,
        'pressure': 1e-3,
        'status': 'operational',
    },
    'RF_CAVITY_A': {
        'frequency': 400.789,       # MHz
        'voltage': 16.0,            # MV
        'phase': 0.0,               # degrees
        'status': 'operational',
    },
    'RF_CAVITY_B': {
        'frequency': 400.789,
        'voltage': 16.0,
        'phase': 0.0,
        'status': 'operational',
    },
    'VACUUM_IR1': {
        'pressure': 1e-10,          # mbar (interaction region)
        'status': 'operational',
    },
    'VACUUM_IR5': {
        'pressure': 1e-10,
        'status': 'operational',
    },
    'MAGNET_ARC12': {
        'current': 11850,           # A (dipole main circuit)
        'resistance': 0.0,          # Ω (superconducting)
        'status': 'operational',
    },
    'COLLIMATOR_TCP': {
        'gap': 5.7,                 # mm (primary collimator half-gap)
        'temperature': 25.0,        # °C
        'status': 'operational',
    },
}


class DiagnosticsServer:
    """MCP-compatible diagnostics and anomaly detection server.

    Provides simulated but physically plausible telemetry for LHC-style
    accelerator subsystems. Intended for use in agentic ops pipelines where
    real SCADA access is unavailable.
    """

    name = 'diagnostics'

    def __init__(self):
        self.anomaly_log: list[dict[str, Any]] = []
        logger.info('DiagnosticsServer initialized with %d subsystems', len(SUBSYSTEMS))

    # ------------------------------------------------------------------
    # Tool: run_anomaly_check
    # ------------------------------------------------------------------

    def run_anomaly_check(self, system: str, time_window: str = '1h') -> str:
        """Run anomaly detection on system telemetry (simulated).

        Args:
            system: Subsystem name (e.g. 'LHC_BEAM1', 'RF_CAVITY_A').
            time_window: Observation window string (e.g. '1h', '30m', '6h').

        Returns:
            JSON string containing anomaly report with severity and recommendations.
        """
        system_upper = system.upper()
        if system_upper not in SUBSYSTEMS:
            available = ', '.join(sorted(SUBSYSTEMS.keys()))
            return json.dumps({
                'error': f'Unknown system: {system}',
                'available_systems': available,
            }, indent=2)

        anomalies: list[dict[str, Any]] = []

        # Generate 20 synthetic data points over the time_window
        for i in range(20):
            noise = random.gauss(0, 0.02)       # baseline ±2% Gaussian noise
            if random.random() < 0.08:          # 8% probability of anomalous sample
                spike_factor = random.uniform(3, 6)
                noise *= spike_factor           # 3–6 σ spike
                anomalies.append({
                    'time': f't-{20 - i}min',
                    'deviation': f'{abs(noise):.2%}',
                    'severity': 'HIGH' if abs(noise) > 0.1 else 'MEDIUM',
                })

        has_high = any(a['severity'] == 'HIGH' for a in anomalies)
        status = 'ALERT' if has_high else ('WARNING' if anomalies else 'NOMINAL')

        result: dict[str, Any] = {
            'system': system_upper,
            'time_window': time_window,
            'data_points_analyzed': 20,
            'anomalies_detected': len(anomalies),
            'status': status,
            'anomalies': anomalies,
            'recommendation': self._get_recommendation(system_upper, anomalies),
        }

        # Persist to in-memory log for session history
        self.anomaly_log.append(result)

        return json.dumps(result, indent=2)

    # ------------------------------------------------------------------
    # Tool: get_system_health
    # ------------------------------------------------------------------

    def get_system_health(self, subsystem: str) -> str:
        """Get current health status of an accelerator subsystem.

        Args:
            subsystem: Subsystem identifier (case-insensitive).

        Returns:
            JSON string with health score, live parameters, and any active alerts.
        """
        sub_upper = subsystem.upper()
        if sub_upper not in SUBSYSTEMS:
            return json.dumps({
                'error': f'Unknown subsystem: {subsystem}',
                'available': sorted(SUBSYSTEMS.keys()),
            }, indent=2)

        params = SUBSYSTEMS[sub_upper].copy()

        # Simulate live health score between 85–100 %
        health_score = round(random.uniform(0.85, 1.0), 3)
        params['health_score'] = health_score
        params['health_percent'] = f'{health_score * 100:.1f}%'
        params['last_checked'] = datetime.utcnow().isoformat() + 'Z'
        params['alerts'] = []

        if health_score < 0.90:
            params['alerts'].append({
                'level': 'WARNING',
                'msg': 'Health score below optimal threshold (90%).',
                'action': 'Schedule preventive inspection within 24 h.',
            })
        if health_score < 0.87:
            params['alerts'].append({
                'level': 'CRITICAL',
                'msg': 'Health score critically low.',
                'action': 'Immediate operator review required.',
            })

        return json.dumps({'subsystem': sub_upper, 'health': params}, indent=2)

    # ------------------------------------------------------------------
    # Tool: get_telemetry_history
    # ------------------------------------------------------------------

    def get_telemetry_history(self, system: str, metric: str, hours: int = 24) -> str:
        """Get historical telemetry data for a specific system metric.

        Args:
            system: Subsystem name.
            metric: Metric name to retrieve (used as label; data is simulated).
            hours: Number of hours of history to return (1 data point per hour).

        Returns:
            JSON string with timestamped data series.
        """
        if hours < 1 or hours > 720:
            return json.dumps({'error': 'hours must be between 1 and 720.'}, indent=2)

        data_points: list[dict[str, Any]] = []
        now = datetime.utcnow()

        # Simulate a slowly drifting signal with Gaussian noise
        base_value = 1.0
        drift = 0.0
        for i in range(hours):
            ts = (now - timedelta(hours=hours - i)).isoformat() + 'Z'
            drift += random.gauss(0, 0.001)     # slow random walk
            value = base_value + drift + random.gauss(0, 0.01)
            data_points.append({
                'timestamp': ts,
                'value': round(value, 4),
                'unit': 'normalized',
            })

        return json.dumps({
            'system': system.upper(),
            'metric': metric,
            'hours': hours,
            'points': len(data_points),
            'data': data_points,
        }, indent=2)

    # ------------------------------------------------------------------
    # Tool: run_full_diagnostic
    # ------------------------------------------------------------------

    def run_full_diagnostic(self, include_all: bool = True) -> str:
        """Run a full diagnostic sweep across all accelerator subsystems.

        Args:
            include_all: If True, sweep all known subsystems. Reserved for
                         future filtering support.

        Returns:
            JSON string with per-subsystem results and aggregate summary.
        """
        results: dict[str, Any] = {}
        for sys_name in SUBSYSTEMS:
            health = round(random.uniform(0.80, 1.0), 3)
            if health > 0.90:
                status = 'OK'
            elif health > 0.80:
                status = 'WARNING'
            else:
                status = 'CRITICAL'

            results[sys_name] = {
                'health_score': health,
                'status': status,
                'checked_at': datetime.utcnow().isoformat() + 'Z',
            }

        overall = round(
            sum(v['health_score'] for v in results.values()) / len(results), 3
        )
        n_ok = sum(1 for v in results.values() if v['status'] == 'OK')
        n_warn = sum(1 for v in results.values() if v['status'] == 'WARNING')
        n_crit = sum(1 for v in results.values() if v['status'] == 'CRITICAL')

        summary: dict[str, Any] = {
            'overall_health': overall,
            'overall_status': (
                'OK' if overall > 0.90 else
                ('WARNING' if overall > 0.80 else 'CRITICAL')
            ),
            'systems_ok': n_ok,
            'systems_warning': n_warn,
            'systems_critical': n_crit,
            'total_systems': len(results),
            'swept_at': datetime.utcnow().isoformat() + 'Z',
            'subsystems': results,
        }
        return json.dumps(summary, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_recommendation(self, system: str, anomalies: list[dict]) -> str:
        """Generate an actionable recommendation based on detected anomalies."""
        if not anomalies:
            return 'System operating within nominal parameters. No action required.'

        high_sev = [a for a in anomalies if a['severity'] == 'HIGH']
        if high_sev:
            return (
                f'HIGH severity anomalies detected in {system}. '
                'Recommend immediate inspection and consider escalation to on-call engineer. '
                'Do NOT continue operations without engineer sign-off.'
            )
        return (
            f'Medium-level deviations detected in {system}. '
            'Monitor closely over next hour. '
            'Investigate root cause if pattern persists beyond two consecutive windows.'
        )

    def get_tools(self) -> dict[str, Any]:
        """Return a mapping of tool names to callable methods."""
        return {
            'run_anomaly_check': self.run_anomaly_check,
            'get_system_health': self.get_system_health,
            'get_telemetry_history': self.get_telemetry_history,
            'run_full_diagnostic': self.run_full_diagnostic,
        }


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_server: DiagnosticsServer | None = None


def get_diagnostics_server() -> DiagnosticsServer:
    """Return the module-level singleton DiagnosticsServer instance."""
    global _server
    if _server is None:
        _server = DiagnosticsServer()
    return _server


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = get_diagnostics_server()
    print(server.run_full_diagnostic())
    print(server.run_anomaly_check('LHC_BEAM1', '1h'))
    print(server.get_system_health('RF_CAVITY_A'))
