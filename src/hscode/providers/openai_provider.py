"""OpenAI provider — uses the ``openai`` Python SDK (>=1.40)."""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from hscode.providers.base import LLMProvider, ProviderError, StructuredOutput


class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI's chat completions API.

    Uses the native structured-output endpoint ``client.chat.completions.parse``
    which constrains the model's output to a JSON schema derived from a
    Pydantic model.
    """

    name = "openai"
    # Cheapest model in the latest GPT-5 family (per the OpenRouter
    # catalogue, which is treated as the source of truth for what models
    # are currently available). Override with `model=` or the
    # ``HSCODE_MODEL`` environment variable.
    DEFAULT_MODEL = "gpt-5.4-nano"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.temperature = temperature

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "OpenAI SDK not installed. Install with: pip install 'hscode[openai]'"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. Export your API key as an environment variable."
            )

        self._client = OpenAI(api_key=resolved_key, base_url=base_url)

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        # Native structured outputs (openai>=1.40). The schema is enforced
        # server-side via JSON-schema strict mode, so the response is
        # guaranteed to parse.
        try:
            completion = self._client.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema,
            )
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise ProviderError(f"OpenAI refused: {message.refusal}")
            if message.parsed is None:
                raise ProviderError("OpenAI returned no parsed structured output")
            return message.parsed  # type: ignore[return-value]
        except AttributeError:
            # Older SDK without `.parse()` — fall back to JSON mode + manual parsing.
            instructions = self.schema_instructions(schema)
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{system_prompt}\n\n{instructions}"},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = completion.choices[0].message.content or ""
            return self.parse_json_response(raw, schema)
