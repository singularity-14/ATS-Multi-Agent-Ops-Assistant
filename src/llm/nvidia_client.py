"""
NVIDIA NIM LLM Client for ATS Multi-Agent Ops Assistant.

Wraps the NVIDIA NIM chat-completions REST API with retry logic,
structured logging, and a simple singleton accessor.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import get_settings

logger = logging.getLogger(__name__)

# Models that support the `reasoning_effort` parameter.
# Standard chat/instruct models will return 400 if this field is included.
_REASONING_MODELS: frozenset[str] = frozenset({
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/nemotron-4-340b-instruct",
})


# ---------------------------------------------------------------------------
# Retry strategy — only on transient HTTP errors, NOT on read timeouts.
# LLM read-timeouts mean the server already started generating; retrying
# immediately just burns quota and hits the timeout again.
# ---------------------------------------------------------------------------
_RETRY_STRATEGY = Retry(
    total=3,
    connect=3,      # retry on connection-level failures
    read=0,         # do NOT retry on ReadTimeoutError
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
    raise_on_status=False,
)


def _build_session() -> requests.Session:
    """Return a requests.Session with the retry adapter mounted."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class NvidiaLLMClient:
    """Thread-safe client for the NVIDIA NIM chat-completions API.

    Example usage::

        client = NvidiaLLMClient()
        answer = client.simple_chat("What is a magnet quench?")
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._session = _build_session()
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "medium",
        top_p: float = 1.0,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Send a structured chat request to the NVIDIA NIM endpoint.

        Args:
            messages:        OpenAI-style message list (role/content dicts).
            temperature:     Override sampling temperature (default from settings).
            max_tokens:      Override max completion tokens (default from settings).
            reasoning_effort: Effort hint for reasoning models ('low'|'medium'|'high').
            top_p:           Nucleus sampling probability mass.
            stream:          If True, streaming mode is requested (not supported here).

        Returns:
            A dict with keys:
            - ``content``  (str)  — the assistant's reply text.
            - ``latency``  (float) — wall-clock seconds for the API round-trip.
            - ``usage``    (dict)  — token usage reported by the API.
            - ``model``    (str)  — model identifier echoed by the API.

        Raises:
            requests.exceptions.HTTPError: On non-2xx responses after retries.
            requests.exceptions.Timeout:   If the request exceeds 120 s.
        """
        payload: dict[str, Any] = {
            "model": self.settings.nvidia_model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "top_p": top_p,
            "stream": stream,
        }
        # Only reasoning models accept the `reasoning_effort` field.
        # Sending it to standard models causes a 400 Bad Request.
        if self.settings.nvidia_model in _REASONING_MODELS:
            payload["reasoning_effort"] = reasoning_effort

        logger.debug(
            "NVIDIA NIM request | model=%s | messages=%d | max_tokens=%d",
            payload["model"],
            len(messages),
            payload["max_tokens"],
        )

        start_ts = time.perf_counter()
        try:
            response = self._session.post(
                self.settings.nvidia_api_url,
                headers=self._headers,
                json=payload,
                timeout=240,  # 240 s — LLM responses can be slow under load
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "NVIDIA API HTTP error | status=%s | body=%s",
                exc.response.status_code if exc.response else "N/A",
                exc.response.text[:500] if exc.response else "",
            )
            raise
        except requests.exceptions.RequestException as exc:
            logger.error("NVIDIA API request failed: %s", exc)
            raise

        latency = time.perf_counter() - start_ts
        raw: dict[str, Any] = response.json()

        content: str = raw["choices"][0]["message"]["content"]
        usage: dict[str, Any] = raw.get("usage", {})
        model_echo: str = raw.get("model", self.settings.nvidia_model)

        logger.info(
            "NVIDIA NIM response | latency=%.2fs | prompt_tokens=%s | completion_tokens=%s",
            latency,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )

        return {
            "content": content,
            "latency": latency,
            "usage": usage,
            "model": model_echo,
        }

    def simple_chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: str = "medium",
    ) -> str:
        """Convenience wrapper — build messages list and return only the text.

        Args:
            prompt:          User prompt text.
            system:          Optional system-role preamble.
            temperature:     Override sampling temperature.
            max_tokens:      Override max completion tokens.
            reasoning_effort: Reasoning effort hint.

        Returns:
            The assistant reply as a plain string.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return result["content"]

    def health_check(self) -> bool:
        """Perform a minimal API call to verify connectivity and credentials.

        Returns:
            True if the API responds successfully, False otherwise.
        """
        try:
            self.simple_chat(
                prompt="Reply with exactly one word: OK",
                system="You are a health-check endpoint. Reply with exactly one word: OK.",
                max_tokens=5,
            )
            logger.info("NVIDIA NIM health check: PASSED")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("NVIDIA NIM health check: FAILED — %s", exc)
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[NvidiaLLMClient] = None


def get_llm_client() -> NvidiaLLMClient:
    """Return the module-level :class:`NvidiaLLMClient` singleton.

    The instance is created lazily on first call and reused thereafter,
    which avoids repeated settings parsing and session construction.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        _client = NvidiaLLMClient()
    return _client


def reset_llm_client() -> None:
    """Reset the singleton (useful in tests or after config changes)."""
    global _client  # noqa: PLW0603
    _client = None
