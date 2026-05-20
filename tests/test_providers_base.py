"""Tests for the provider base class & factory."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from hscode.providers import get_provider, list_providers
from hscode.providers.base import LLMProvider, ProviderError


class Schema(BaseModel):
    code: str
    note: str


def test_list_providers_contains_expected_names() -> None:
    names = list_providers()
    for name in (
        "openai",
        "azure",
        "anthropic",
        "google",
        "gemini",
        "ollama",
        "openrouter",
    ):
        assert name in names


def test_unknown_provider_raises() -> None:
    with pytest.raises(ProviderError):
        get_provider(name="nonexistent_provider")


def test_parse_json_response_plain() -> None:
    parsed = LLMProvider.parse_json_response('{"code": "85", "note": "hi"}', Schema)
    assert parsed.code == "85"
    assert parsed.note == "hi"


def test_parse_json_response_with_code_fence() -> None:
    parsed = LLMProvider.parse_json_response(
        '```json\n{"code": "85", "note": "hi"}\n```', Schema
    )
    assert parsed.code == "85"


def test_parse_json_response_embedded_in_prose() -> None:
    raw = 'Sure! Here you go:\n{"code": "85", "note": "hi"}\nLet me know if you need more.'
    parsed = LLMProvider.parse_json_response(raw, Schema)
    assert parsed.code == "85"


def test_parse_json_response_empty_raises() -> None:
    with pytest.raises(ProviderError):
        LLMProvider.parse_json_response("", Schema)


def test_parse_json_response_garbage_raises() -> None:
    with pytest.raises(ProviderError):
        LLMProvider.parse_json_response("not json at all", Schema)


def test_schema_instructions_includes_schema() -> None:
    text = LLMProvider.schema_instructions(Schema)
    assert "code" in text and "note" in text
    assert "JSON" in text
