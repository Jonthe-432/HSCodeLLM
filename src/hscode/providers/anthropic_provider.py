"""Anthropic Claude provider.

Uses tool-use to obtain structured output: we declare a single tool whose
input schema matches the desired output schema and force the model to call
it. The tool's ``input`` dict is then validated against the Pydantic schema.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from hscode.providers.base import LLMProvider, ProviderError, StructuredOutput


class AnthropicProvider(LLMProvider):
    """LLM provider for Anthropic Claude models."""

    name = "anthropic"
    # Cheapest model in the current Claude family (haiku tier, latest
    # generation). Override with `model=` or the ``HSCODE_MODEL`` env var.
    DEFAULT_MODEL = "claude-haiku-4.5"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "Anthropic SDK not installed. Install with: pip install 'hscode[anthropic]'"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set.")

        self._client = Anthropic(api_key=resolved_key)

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        schema_json = schema.model_json_schema()
        tool_name = f"return_{schema.__name__.lower()}"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": tool_name,
                    "description": f"Return a {schema.__name__} object matching the input schema.",
                    "input_schema": schema_json,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return schema.model_validate(block.input)  # type: ignore[arg-type]

        # Fallback: if Claude returned plain text, try to parse it as JSON.
        text_parts = [
            getattr(b, "text", "")
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        raw = "\n".join(t for t in text_parts if t)
        return self.parse_json_response(raw, schema)
