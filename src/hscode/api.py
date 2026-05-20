"""High-level convenience API for the HSCode package."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from hscode.classifier import HSCodeClassifier
from hscode.llm import get_chat_model
from hscode.models import ClassificationResult


def classify(
    description: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
    max_retries: Optional[int] = None,
    **llm_kwargs: Any,
) -> ClassificationResult:
    """Classify a single product description into an 8-digit EU CN code.

    This is the simplest entry point. For high-throughput use cases that
    reuse the same LLM client across many classifications, prefer
    instantiating :class:`HSCodeClassifier` directly.

    Args:
        description: Free-text product description.
        provider: Provider name (``"openai"``, ``"azure"``, ``"anthropic"``,
            ``"google"``, ``"ollama"``, ``"openrouter"``). Defaults to
            ``$HSCODE_PROVIDER`` or ``"openai"``.
        model: Model slug passed to the provider. Defaults to
            ``$HSCODE_MODEL`` or a provider-specific default.
        llm: Pre-built LangChain ``BaseChatModel``. Takes precedence over
            ``provider``/``model``/``llm_kwargs``.
        max_retries: Maximum number of hierarchical passes.
        **llm_kwargs: Extra keyword arguments forwarded to
            :func:`hscode.llm.get_chat_model` (e.g. ``temperature``,
            ``timeout``, ``max_tokens``).

    Returns:
        A :class:`ClassificationResult` describing the chosen CN code.
    """
    if llm is None:
        llm = get_chat_model(provider=provider, model=model, **llm_kwargs)

    classifier = HSCodeClassifier(llm=llm, max_retries=max_retries)
    return classifier.classify(description)
