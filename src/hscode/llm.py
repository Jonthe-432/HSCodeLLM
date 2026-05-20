"""
Model-agnostic LLM access for HSCode.

This module is a thin wrapper around :func:`langchain.chat_models.init_chat_model`
that:

  * accepts our environment-variable conventions (``HSCODE_PROVIDER`` /
    ``HSCODE_MODEL``) and applies sensible defaults when they are unset,
  * normalises a handful of historical provider aliases (``gemini`` ->
    ``google-genai``) to whatever LangChain expects today,
  * returns a ``BaseChatModel`` ready to be paired with
    :meth:`~langchain_core.language_models.chat_models.BaseChatModel.with_structured_output`.

The rest of the codebase only ever depends on the LangChain
``BaseChatModel`` interface, so adding/removing/upgrading providers is a
single-line change to :data:`_PROVIDER_DEFAULTS` plus a corresponding
extra in :file:`pyproject.toml`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hscode.config import SETTINGS

# LangChain providers we know how to spin up.  Maps the user-facing name
# (what callers pass / what HSCODE_PROVIDER is set to) to:
#   * the LangChain provider prefix used by ``init_chat_model``,
#   * the install-extra name (for nice error messages on ImportError),
#   * a sensible default model slug.
_PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    # Cloud-hosted, OpenAI-compatible
    "openai": {
        "lc_provider": "openai",
        "extra": "openai",
        "default_model": "gpt-5.4-nano",
    },
    "azure": {
        "lc_provider": "azure_openai",
        "extra": "azure",
        # No truly-universal default — Azure uses deployment names. Caller
        # must supply ``model`` (or set HSCODE_MODEL).
        "default_model": "",
    },
    # OpenRouter — proxies hundreds of models through one OpenAI-compatible API.
    # LangChain ships a built-in "openrouter:" prefix that handles auth via
    # OPENROUTER_API_KEY and the right base_url.
    "openrouter": {
        "lc_provider": "openrouter",
        "extra": "openrouter",
        "default_model": "openai/gpt-5.4-nano",
    },
    # Native vendors
    "anthropic": {
        "lc_provider": "anthropic",
        "extra": "anthropic",
        "default_model": "claude-haiku-4.5",
    },
    "google": {
        "lc_provider": "google_genai",
        "extra": "google",
        "default_model": "gemini-3.1-flash-lite",
    },
    "gemini": {  # alias for "google"
        "lc_provider": "google_genai",
        "extra": "google",
        "default_model": "gemini-3.1-flash-lite",
    },
    # Local
    "ollama": {
        "lc_provider": "ollama",
        "extra": "ollama",
        "default_model": "llama3.1",
    },
}


def list_providers() -> List[str]:
    """Return all provider names recognised by :func:`get_chat_model`."""
    return sorted(_PROVIDER_DEFAULTS.keys())


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    **extra_kwargs: Any,
):
    """Build a LangChain ``BaseChatModel`` from a provider name and model slug.

    Args:
        provider: One of :func:`list_providers`. Defaults to
            ``$HSCODE_PROVIDER`` or ``"openai"``.
        model: Model slug. Defaults to ``$HSCODE_MODEL`` or the
            provider-specific default in :data:`_PROVIDER_DEFAULTS`.
        temperature: Sampling temperature (default 0.0 for deterministic
            classification).
        **extra_kwargs: Forwarded to ``init_chat_model`` (e.g. ``timeout``,
            ``max_tokens``, or anything provider-specific the LangChain
            integration accepts).

    Raises:
        ValueError: If ``provider`` is unknown.
        ImportError: If the LangChain integration package for ``provider``
            is not installed. The error message names the right extra.
    """
    try:
        from langchain.chat_models import init_chat_model  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "LangChain is not installed. Run: pip install 'hscode[openai]' "
            "(or any other provider extra)."
        ) from exc

    resolved_provider = (provider or SETTINGS.provider or "openai").lower()
    if resolved_provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown provider {resolved_provider!r}. "
            f"Known providers: {list_providers()}"
        )

    info = _PROVIDER_DEFAULTS[resolved_provider]
    resolved_model = model or SETTINGS.model or info["default_model"]
    if not resolved_model:
        raise ValueError(
            f"Provider {resolved_provider!r} has no default model — please pass "
            f"`model=` or set HSCODE_MODEL."
        )

    # LangChain accepts "<provider>:<model>" as a single string. We use the
    # explicit ``model_provider`` kwarg so callers can pass model slugs that
    # contain colons (e.g. OpenRouter free-tier suffixes like ``:free``).
    try:
        return init_chat_model(
            model=resolved_model,
            model_provider=info["lc_provider"],
            temperature=temperature,
            **extra_kwargs,
        )
    except ImportError as exc:
        raise ImportError(
            f"Provider {resolved_provider!r} requires an extra LangChain integration. "
            f"Install it with:  pip install 'hscode[{info['extra']}]'\n"
            f"Original error: {exc}"
        ) from exc
