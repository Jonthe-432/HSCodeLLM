"""Ollama provider — for fully local classification."""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from hscode.providers.base import LLMProvider, ProviderError, StructuredOutput


class OllamaProvider(LLMProvider):
    """LLM provider for models served via Ollama.

    Uses Ollama's ``format=<json_schema>`` parameter to constrain the
    response to a JSON schema derived from the target Pydantic model.
    """

    name = "ollama"
    # Reasonable default — small enough to run on a laptop, capable enough
    # for classification. Override with `model=` or HSCODE_MODEL.
    DEFAULT_MODEL = "llama3.1"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.temperature = temperature
        self.host = host or os.environ.get("OLLAMA_HOST", self.DEFAULT_HOST)

        try:
            from ollama import Client  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "Ollama SDK not installed. Install with: pip install 'hscode[ollama]'"
            ) from exc

        self._client = Client(host=self.host)

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        schema_json = schema.model_json_schema()

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=schema_json,
                options={"temperature": self.temperature},
            )
            raw = response["message"]["content"] or ""
            return schema.model_validate_json(raw)
        except Exception:
            # Older Ollama versions only support format='json'.
            instructions = self.schema_instructions(schema)
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\n\n{instructions}"},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": self.temperature},
            )
            raw = response["message"]["content"] or ""
            return self.parse_json_response(raw, schema)
