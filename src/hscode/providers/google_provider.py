"""Google Gemini provider — uses the unified ``google-genai`` Python SDK.

The legacy ``google-generativeai`` package is deprecated in favour of
``google-genai`` (the unified GenAI SDK that works for both the Gemini API
and Vertex AI).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from hscode.providers.base import LLMProvider, ProviderError, StructuredOutput


class GoogleProvider(LLMProvider):
    """LLM provider for Google Gemini models."""

    name = "google"
    # Cheapest model in the current Gemini family (flash-lite tier, latest
    # generation). Override with `model=` or the ``HSCODE_MODEL`` env var.
    DEFAULT_MODEL = "gemini-3.1-flash-lite"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.temperature = temperature

        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "google-genai not installed. Install with: pip install 'hscode[google]'"
            ) from exc

        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise ProviderError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.")

        self._client = genai.Client(api_key=resolved_key)
        self._types = genai_types

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        # The unified SDK accepts a Pydantic class directly as ``response_schema``
        # together with ``response_mime_type='application/json'``.
        config = self._types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )

        # New SDK exposes the already-parsed Pydantic object on `.parsed`.
        parsed = getattr(response, "parsed", None)
        if parsed is not None and isinstance(parsed, schema):
            return parsed  # type: ignore[return-value]

        raw = getattr(response, "text", "") or ""
        return self.parse_json_response(raw, schema)
