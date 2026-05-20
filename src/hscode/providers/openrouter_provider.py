"""OpenRouter provider — uses the OpenAI-compatible OpenRouter API.

OpenRouter (https://openrouter.ai) exposes a single endpoint that is
wire-compatible with the OpenAI Chat Completions API, but lets you pick
from hundreds of models across providers (e.g. ``openai/gpt-5.4-nano``,
``anthropic/claude-haiku-4.5``, ``google/gemini-3.1-flash-lite``, etc.).

Because the wire protocol is OpenAI-compatible, we reuse the ``openai``
Python SDK and simply point it at ``https://openrouter.ai/api/v1``.

Configuration (env vars):
    OPENROUTER_API_KEY      Your OpenRouter API key (required).
    OPENROUTER_BASE_URL     Override base URL (default:
                            ``https://openrouter.ai/api/v1``).
    OPENROUTER_HTTP_REFERER Optional. Sent as ``HTTP-Referer`` header so your
                            app is listed on openrouter.ai/rankings.
    OPENROUTER_APP_TITLE    Optional. Sent as ``X-Title`` header for the
                            same ranking purposes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type
from urllib import request as _urlrequest
from urllib.error import HTTPError, URLError

from hscode.providers.base import LLMProvider, ProviderError, StructuredOutput


class OpenRouterProvider(LLMProvider):
    """LLM provider for OpenRouter's OpenAI-compatible chat completions API."""

    name = "openrouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    # OpenRouter requires a fully-qualified model slug (``vendor/model``).
    # Cheapest model in the newest GPT-5 family at the time of writing.
    DEFAULT_MODEL = "openai/gpt-5.4-nano"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or self.DEFAULT_MODEL, **kwargs)
        self.temperature = temperature

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "OpenAI SDK not installed. Install with: pip install 'hscode[openrouter]'"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Export your API key as an environment variable."
            )

        resolved_base_url = (
            base_url
            or os.environ.get("OPENROUTER_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        # Optional ranking headers — see https://openrouter.ai/docs/quick-start
        self._default_headers: Dict[str, str] = {}
        referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        title = app_title or os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            self._default_headers["HTTP-Referer"] = referer
        if title:
            self._default_headers["X-Title"] = title

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            default_headers=self._default_headers or None,
        )

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        # Try native structured outputs first. Some OpenRouter-routed models
        # (notably OpenAI's own) support JSON-schema strict mode through the
        # OpenAI SDK's ``.parse()`` helper. We treat any failure here as a
        # signal to fall back to JSON mode, since many models routed via
        # OpenRouter don't support strict schema enforcement.
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
                raise ProviderError(f"OpenRouter refused: {message.refusal}")
            if message.parsed is not None:
                return message.parsed  # type: ignore[return-value]
            raw = message.content or ""
            if raw:
                return self.parse_json_response(raw, schema)
        except ProviderError:
            raise
        except Exception:
            # AttributeError (older SDK without ``.parse()``), 4xx from the
            # provider rejecting strict JSON-schema mode, etc. Fall back.
            pass

        # Fallback: JSON mode + manual parsing. Works for virtually every
        # chat model routed by OpenRouter.
        instructions = self.schema_instructions(schema)
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"{system_prompt}\n\n{instructions}"},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            # Some models on OpenRouter reject ``response_format``. Retry
            # one more time without it — the schema instructions in the
            # system prompt still nudge the model toward valid JSON.
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": f"{system_prompt}\n\n{instructions}"},
                    {"role": "user", "content": user_prompt},
                ],
            )

        raw = completion.choices[0].message.content or ""
        return self.parse_json_response(raw, schema)

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    @classmethod
    def list_models(
        cls,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        structured_only: bool = False,
        timeout: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """Fetch the live OpenRouter model catalogue.

        Calls ``GET {base_url}/models`` and returns the ``data`` array.
        Each entry includes ``id`` (the slug to pass as ``model=``),
        ``name``, ``context_length``, ``pricing``, ``supported_parameters``,
        and more — see https://openrouter.ai/docs/api-reference/list-available-models.

        Args:
            api_key: Optional. If omitted, falls back to ``OPENROUTER_API_KEY``.
                The endpoint is technically public, but sending the key is
                recommended (and required for some account-scoped fields).
            base_url: Override base URL (defaults to ``OPENROUTER_BASE_URL``
                or the OpenRouter production URL).
            structured_only: If True, keep only models whose
                ``supported_parameters`` advertise ``response_format`` or
                ``structured_outputs`` — i.e. models suitable for the
                schema-constrained output the HSCode classifier requires.
            timeout: HTTP timeout in seconds.

        Raises:
            ProviderError: On HTTP, JSON, or transport failures.
        """
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        resolved_base_url = (
            base_url
            or os.environ.get("OPENROUTER_BASE_URL")
            or cls.DEFAULT_BASE_URL
        ).rstrip("/")

        url = f"{resolved_base_url}/models"
        req = _urlrequest.Request(url, headers={"Accept": "application/json"})
        if resolved_key:
            req.add_header("Authorization", f"Bearer {resolved_key}")

        try:
            with _urlrequest.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise ProviderError(
                f"OpenRouter /models returned HTTP {exc.code}: {exc.reason}"
            ) from exc
        except URLError as exc:
            raise ProviderError(
                f"Could not reach OpenRouter /models: {exc.reason}"
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError(
                f"OpenRouter /models returned invalid JSON: {exc}"
            ) from exc

        models = payload.get("data") or []
        if not isinstance(models, list):
            raise ProviderError("OpenRouter /models response missing 'data' array")

        if structured_only:
            def _supports_structured(m: Dict[str, Any]) -> bool:
                params = m.get("supported_parameters") or []
                return "response_format" in params or "structured_outputs" in params

            models = [m for m in models if _supports_structured(m)]

        return models
