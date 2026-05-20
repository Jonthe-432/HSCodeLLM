"""Tests for the LangChain-powered LLM accessor (hscode.llm)."""

from __future__ import annotations

import pytest

from hscode.llm import list_providers, _PROVIDER_DEFAULTS, get_chat_model


def test_list_providers_contains_expected_names() -> None:
    names = set(list_providers())
    expected = {"openai", "azure", "openrouter", "anthropic", "google", "gemini", "ollama"}
    assert expected.issubset(names)


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_chat_model(provider="nonexistent_provider")


def test_provider_defaults_have_required_keys() -> None:
    for name, info in _PROVIDER_DEFAULTS.items():
        assert "lc_provider" in info, f"{name}: missing lc_provider"
        assert "extra" in info, f"{name}: missing extra"
        assert "default_model" in info, f"{name}: missing default_model"


def test_azure_requires_explicit_model() -> None:
    """Azure has no universal default (deployment names vary per tenant)."""
    # Sanity check: the catalogue records an empty default_model for azure.
    assert _PROVIDER_DEFAULTS["azure"]["default_model"] == ""
    with pytest.raises(ValueError, match="no default model"):
        get_chat_model(provider="azure")
