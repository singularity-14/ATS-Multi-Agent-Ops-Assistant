"""
Pytest configuration and shared fixtures for ATS Multi-Agent Ops Assistant tests.
Sets up a mock environment so unit tests don't require a real NVIDIA_API_KEY.
"""
import os
import tempfile
from pathlib import Path

import pytest

# Inject dummy env vars before any src imports happen
os.environ.setdefault("NVIDIA_API_KEY", "test_key_for_unit_tests_only")
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Temporary data directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "docs").mkdir()
        (p / "test_cases").mkdir()
        (p / "eval_reports").mkdir()
        yield p


@pytest.fixture(scope="session")
def sample_accelerator_doc():
    """A minimal accelerator doc for testing search."""
    return {
        "id": "doc-001",
        "title": "LHC Beam Injection Procedure",
        "category": "Operations",
        "subsystem": "LHC_BEAM",
        "content": (
            "The LHC beam injection procedure involves transferring proton bunches "
            "from the SPS into the LHC ring. The superconducting dipole magnets are "
            "cooled to 1.9K before injection begins. RF cavities at 400 MHz guide the "
            "beam along the correct trajectory. Beam diagnostics monitors confirm "
            "nominal beam current of 0.582A. The injection takes approximately 3 minutes "
            "per beam and is supervised by the operations team."
        ),
        "keywords": ["injection", "beam", "SPS", "LHC", "proton", "RF", "dipole"],
    }


@pytest.fixture(scope="function")
def audit_db_path(tmp_path):
    """Fresh SQLite DB path for each test function."""
    return tmp_path / "test_audit.db"
