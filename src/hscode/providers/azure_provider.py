"""Azure OpenAI provider."""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from hscode.providers.base import LLMProvider, ProviderError, StructuredOutput


class AzureOpenAIProvider(LLMProvider):
    """LLM provider for Azure OpenAI deployments."""

    name = "azure"
    DEFAULT_API_VERSION = "2024-08-01-preview"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # On Azure, "model" is really the deployment name unless caller provides one.
        super().__init__(model=model, **kwargs)
        self.temperature = temperature

        try:
            from openai import AzureOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "OpenAI SDK not installed. Install with: pip install 'hscode[azure]'"
            ) from exc

        resolved_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        resolved_deployment = (
            deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model
        )
        resolved_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or self.DEFAULT_API_VERSION
        )

        missing = [
            name
            for name, val in [
                ("AZURE_OPENAI_API_KEY", resolved_key),
                ("AZURE_OPENAI_ENDPOINT", resolved_endpoint),
                ("AZURE_OPENAI_DEPLOYMENT (or model=)", resolved_deployment),
            ]
            if not val
        ]
        if missing:
            raise ProviderError(
                "Azure OpenAI is missing required configuration: " + ", ".join(missing)
            )

        self.deployment = resolved_deployment
        self.model = resolved_deployment  # for logging/visibility

        self._client = AzureOpenAI(
            api_key=resolved_key,
            azure_endpoint=resolved_endpoint,
            api_version=resolved_version,
        )

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        try:
            completion = self._client.beta.chat.completions.parse(
                model=self.deployment,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=schema,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ProviderError("Azure OpenAI returned no parsed structured output")
            return parsed  # type: ignore[return-value]
        except AttributeError:
            pass

        instructions = self.schema_instructions(schema)
        completion = self._client.chat.completions.create(
            model=self.deployment,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\n{instructions}"},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = completion.choices[0].message.content or ""
        return self.parse_json_response(raw, schema)
