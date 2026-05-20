"""
LLM provider abstractions for HSCode.

The classifier is decoupled from any specific LLM SDK: it only depends on
:class:`LLMProvider`, which exposes a single ``generate_structured`` method.

Provider implementations live in submodules and are imported lazily so users
only need to install the SDKs they actually use.

Usage::

    from hscode.providers import get_provider
    provider = get_provider("openai", model="gpt-5.4-nano")

    # or directly:
    from hscode.providers.openai_provider import OpenAIProvider
    provider = OpenAIProvider(model="gpt-5.4-nano")
"""

from hscode.providers.base import (
    LLMProvider,
    ProviderError,
    StructuredOutput,
)
from hscode.providers.factory import get_provider, list_providers

__all__ = [
    "LLMProvider",
    "ProviderError",
    "StructuredOutput",
    "get_provider",
    "list_providers",
]
