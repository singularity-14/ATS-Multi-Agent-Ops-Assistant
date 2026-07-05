"""
Configuration management for ATS Multi-Agent Ops Assistant.

Uses pydantic-settings to load configuration from environment variables
and an optional .env file. All settings are validated at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── NVIDIA LLM ────────────────────────────────────────────────────────────
    nvidia_api_key: str = Field(..., description="NVIDIA NIM API key")
    nvidia_model: str = Field(
        default="mistralai/mistral-medium-3.5-128b",
        description="NVIDIA-hosted model identifier",
    )
    nvidia_api_url: str = Field(
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        description="NVIDIA NIM chat completions endpoint",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_env: str = Field(
        default="development",
        description="Runtime environment: development | staging | production",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    audit_db_path: Path = Field(
        default=Path("data/audit.db"),
        description="Path to the SQLite audit database",
    )

    # ── Inference Defaults ────────────────────────────────────────────────────
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to accept an agent response",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens for LLM completion",
    )
    temperature: float = Field(
        default=0.70,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses",
    )

    # ── Server Ports ──────────────────────────────────────────────────────────
    fastapi_port: int = Field(
        default=8000,
        description="Port for the FastAPI REST service",
    )
    streamlit_port: int = Field(
        default=8501,
        description="Port for the Streamlit UI",
    )

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.app_env.lower() == "production"

    @property
    def audit_db_dir(self) -> Path:
        """Parent directory of the audit database (created on demand)."""
        return self.audit_db_path.parent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (cached) application settings singleton.

    The cache is intentionally limited to a single instance so the
    environment is only parsed once per process.  Call
    ``get_settings.cache_clear()`` in tests to reset between runs.
    """
    return Settings()  # type: ignore[call-arg]
