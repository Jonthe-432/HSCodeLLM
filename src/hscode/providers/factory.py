"""Provider factory — instantiates the right provider by name."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from hscode.config import SETTINGS
from hscode.providers.base import LLMProvider, ProviderError

_PROVIDER_REGISTRY: Dict[str, str] = {
    # name -> "module.path:ClassName"
    "openai": "hscode.providers.openai_provider:OpenAIProvider",
    "azure": "hscode.providers.azure_provider:AzureOpenAIProvider",
    "anthropic": "hscode.providers.anthropic_provider:AnthropicProvider",
    "google": "hscode.providers.google_provider:GoogleProvider",
    "gemini": "hscode.providers.google_provider:GoogleProvider",
    "ollama": "hscode.providers.ollama_provider:OllamaProvider",
    "openrouter": "hscode.providers.openrouter_provider:OpenRouterProvider",
}


def list_providers() -> List[str]:
    """Return the list of provider names registered with the factory."""
    return sorted(set(_PROVIDER_REGISTRY.keys()))


def get_provider(
    name: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """Instantiate an LLM provider by name.

    Args:
        name: One of :func:`list_providers` (e.g. ``"openai"``). Defaults to
            the value of ``HSCODE_PROVIDER`` or ``"openai"``.
        model: Model name. Defaults to ``HSCODE_MODEL`` or a provider-specific
            sensible default.
        **kwargs: Forwarded to the provider constructor.

    Raises:
        ProviderError: If the provider is unknown or its SDK is not installed.
    """
    resolved_name = (name or SETTINGS.provider or "openai").lower()
    resolved_model = model or SETTINGS.model

    if resolved_name not in _PROVIDER_REGISTRY:
        raise ProviderError(
            f"Unknown provider {resolved_name!r}. Known providers: {list_providers()}"
        )

    target = _PROVIDER_REGISTRY[resolved_name]
    module_path, class_name = target.split(":")

    try:
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except ImportError as exc:
        raise ProviderError(
            f"Provider {resolved_name!r} requires an optional dependency. "
            f"Install it with:  pip install 'hscode[{resolved_name}]'\n"
            f"Original error: {exc}"
        ) from exc

    return cls(model=resolved_model, **kwargs)
