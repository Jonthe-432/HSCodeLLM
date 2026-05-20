"""High-level convenience API for the HSCode package."""

from __future__ import annotations

from typing import Any, Optional

from hscode.classifier import HSCodeClassifier
from hscode.models import ClassificationResult
from hscode.providers import LLMProvider, get_provider


def classify(
    description: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    llm_provider: Optional[LLMProvider] = None,
    max_retries: Optional[int] = None,
    **provider_kwargs: Any,
) -> ClassificationResult:
    """Classify a single product description into an 8-digit EU CN code.

    This is the simplest entry point. For high-throughput use cases that
    reuse the same LLM client across many classifications, prefer
    instantiating :class:`HSCodeClassifier` directly.

    Args:
        description: Free-text product description.
        provider: Provider name (``"openai"``, ``"azure"``, ``"anthropic"``,
            ``"google"``, ``"ollama"``). Defaults to ``$HSCODE_PROVIDER`` or
            ``"openai"``.
        model: Model name passed to the provider. Defaults to
            ``$HSCODE_MODEL`` or a provider-specific default.
        llm_provider: Pre-built :class:`LLMProvider` instance. Takes
            precedence over ``provider``/``model``/``provider_kwargs``.
        max_retries: Maximum number of hierarchical passes.
        **provider_kwargs: Extra keyword arguments forwarded to the
            provider constructor (e.g. ``temperature``, ``base_url``).

    Returns:
        A :class:`ClassificationResult` describing the chosen CN code.
    """
    if llm_provider is None:
        llm_provider = get_provider(name=provider, model=model, **provider_kwargs)

    classifier = HSCodeClassifier(provider=llm_provider, max_retries=max_retries)
    return classifier.classify(description)
