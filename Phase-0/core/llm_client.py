"""
core.llm_client — Shared LLM client with provider auto-detection and retry.

Extracted from ``preprocessing.build_index`` so that every module that needs
LLM calls (index builder, router, generator) uses one consistent code path.

Supported Providers
-------------------
- **Google Gemini** (FREE tier) — set ``LLM_PROVIDER=gemini``
- **Groq** (FREE tier) — set ``LLM_PROVIDER=groq``
- **OpenAI** — set ``LLM_PROVIDER=openai``
- **Anthropic** — set ``LLM_PROVIDER=anthropic``
- **Any OpenAI-compatible** — set ``LLM_PROVIDER=openai`` + ``LLM_BASE_URL``

Auto-detection: if ``LLM_PROVIDER`` is not set, the provider is guessed
from the model name (``gemini`` → Gemini, ``claude`` → Anthropic,
``llama``/``mixtral`` with Groq key → Groq, else → OpenAI).

Features
--------
- **Retry with backoff**: transient failures are retried up to ``max_retries``
  times with exponential back-off (1 s, 2 s, 4 s …).
- **Graceful degradation**: returns ``None`` on permanent failure so callers can
  fall back to non-LLM logic without crashing.

Public API
----------
call_llm(prompt, ...) -> str | None
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

from config.settings import LLM_API_KEY, LLM_MODEL_NAME, LLM_PROVIDER, LLM_BASE_URL


# ═══════════════════════════════════════════════════════════════════════════
# Provider detection
# ═══════════════════════════════════════════════════════════════════════════

# Base URLs for providers that use OpenAI-compatible endpoints.
_PROVIDER_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq":   "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",  # SDK default, listed for clarity
}


def _detect_provider(model: str, provider_hint: str) -> str:
    """Determine which provider to use from explicit setting or model name.

    Priority:
    1. Explicit ``LLM_PROVIDER`` env var (``gemini``, ``groq``, ``openai``, ``anthropic``)
    2. Model name heuristics:
       - Contains ``gemini`` → ``"gemini"``
       - Contains ``claude`` → ``"anthropic"``
       - Contains ``llama`` or ``mixtral`` → ``"groq"``
       - Everything else → ``"openai"``

    Returns one of: ``"gemini"``, ``"groq"``, ``"openai"``, ``"anthropic"``.
    """
    if provider_hint:
        return provider_hint.lower()

    m = model.lower()
    if "gemini" in m:
        return "gemini"
    if "claude" in m:
        return "anthropic"
    if any(name in m for name in ("llama", "mixtral", "deepseek", "qwen")):
        return "groq"
    return "openai"


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def call_llm(
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 500,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
) -> str | None:
    """Send a prompt to the configured LLM and return the response text.

    Parameters
    ----------
    prompt : str
        The user/system message to send.
    temperature : float
        Sampling temperature.  Use ``0`` for deterministic routing,
        ``0.3`` for creative-ish tasks like abstract generation.
    max_tokens : int
        Maximum tokens in the response.
    model : str | None
        Override the model name.  Defaults to ``LLM_MODEL_NAME`` from
        settings / ``.env``.
    api_key : str | None
        Override the API key.  Defaults to ``LLM_API_KEY`` from settings.
    max_retries : int
        How many times to retry on transient errors (e.g. rate-limit,
        network timeout).  Set to ``0`` for no retries.

    Returns
    -------
    str | None
        The LLM's response text, or ``None`` if the call failed after
        all retries (logged via ``warnings.warn``).
    """
    resolved_key = api_key or LLM_API_KEY
    resolved_model = model or LLM_MODEL_NAME

    if not resolved_key or not resolved_model:
        warnings.warn(
            "[LLM_CLIENT] No API key or model name configured — "
            "set LLM_API_KEY and LLM_MODEL_NAME in .env",
            stacklevel=2,
        )
        return None

    provider = _detect_provider(resolved_model, LLM_PROVIDER)

    for attempt in range(1 + max_retries):
        try:
            if provider == "anthropic":
                return _call_anthropic(
                    prompt, resolved_model, resolved_key, temperature, max_tokens
                )
            else:
                # Gemini, Groq, and OpenAI all use OpenAI-compatible endpoints.
                # The only difference is the base_url.
                base_url = LLM_BASE_URL or _PROVIDER_BASE_URLS.get(provider)
                return _call_openai_compatible(
                    prompt, resolved_model, resolved_key, temperature, max_tokens,
                    base_url=base_url,
                )
        except _PermanentError as exc:
            # Auth errors, invalid model, etc. — don't retry
            warnings.warn(
                f"[LLM_CLIENT] Permanent error (attempt {attempt + 1}): {exc}",
                stacklevel=2,
            )
            return None
        except Exception as exc:
            # Transient errors — retry with exponential backoff
            if attempt < max_retries:
                backoff = 2 ** attempt  # 1s, 2s, 4s ...
                warnings.warn(
                    f"[LLM_CLIENT] Transient error (attempt {attempt + 1}/{1 + max_retries}), "
                    f"retrying in {backoff}s: {exc}",
                    stacklevel=2,
                )
                time.sleep(backoff)
            else:
                warnings.warn(
                    f"[LLM_CLIENT] All {1 + max_retries} attempts failed: {exc}",
                    stacklevel=2,
                )
                return None

    return None  # unreachable, but keeps mypy happy


# ═══════════════════════════════════════════════════════════════════════════
# Internal — provider-specific callers
# ═══════════════════════════════════════════════════════════════════════════

class _PermanentError(Exception):
    """Raised for errors that should NOT be retried (auth, bad model, etc.)."""


def _call_openai_compatible(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
) -> str:
    """Call any OpenAI-compatible chat completions endpoint.

    Works with: OpenAI, Google Gemini, Groq, Together AI, Ollama, etc.
    The ``base_url`` parameter controls which provider's endpoint is hit.

    Raises ``_PermanentError`` for auth/model errors; lets transient
    errors (rate-limit, timeout) propagate for retry.
    """
    try:
        from openai import OpenAI, AuthenticationError, NotFoundError
    except ImportError:
        raise _PermanentError(
            "'openai' package not installed — run: pip install openai>=1.0.0"
        )

    try:
        # Build client kwargs — only include base_url if explicitly set
        # (lets the SDK use its default for plain OpenAI calls).
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except (AuthenticationError, NotFoundError) as exc:
        raise _PermanentError(str(exc)) from exc
    # All other exceptions (RateLimitError, APIConnectionError, etc.)
    # propagate as transient → caller retries.


def _call_anthropic(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call the Anthropic Messages API.

    Raises ``_PermanentError`` for auth/model errors; lets transient
    errors propagate for retry.
    """
    try:
        import anthropic
    except ImportError:
        raise _PermanentError(
            "'anthropic' package not installed — run: pip install anthropic"
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except anthropic.AuthenticationError as exc:
        raise _PermanentError(str(exc)) from exc
    # All other exceptions propagate as transient → caller retries.
