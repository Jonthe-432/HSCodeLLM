"""OpenRouter helpers — model discovery against the live ``/models`` endpoint.

LangChain treats OpenRouter as just another chat-model provider, but it
doesn't expose a way to list the ~350 models available through OpenRouter
at any given moment. This module is a tiny, dependency-free wrapper
around the public ``GET /api/v1/models`` endpoint so callers can:

  * discover what model slugs are valid (``OpenRouter.list_models()``),
  * filter to models that support structured outputs (recommended for
    this project, which constrains every LLM reply to a Pydantic schema),
  * compare prices, context lengths, and supported parameters.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib import request as _urlrequest
from urllib.error import HTTPError, URLError


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter ``/models`` endpoint can't be reached
    or returns unexpected data."""


def list_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    structured_only: bool = False,
    timeout: float = 15.0,
) -> List[Dict[str, Any]]:
    """Fetch the live OpenRouter model catalogue.

    Calls ``GET {base_url}/models`` and returns the ``data`` array.
    Each entry includes ``id`` (the slug to pass as the model name),
    ``name``, ``context_length``, ``pricing``, ``supported_parameters``,
    and more — see https://openrouter.ai/docs/api-reference/list-available-models.

    Args:
        api_key: Optional. Falls back to ``OPENROUTER_API_KEY``. The
            endpoint is technically public, but sending the key is
            recommended (and required for some account-scoped fields).
        base_url: Override base URL (defaults to ``OPENROUTER_BASE_URL``
            or the OpenRouter production URL).
        structured_only: If True, keep only models whose
            ``supported_parameters`` advertise ``response_format`` or
            ``structured_outputs`` — i.e. models suitable for the
            schema-constrained output the HSCode classifier requires.
        timeout: HTTP timeout in seconds.

    Raises:
        OpenRouterError: On HTTP, JSON, or transport failures.
    """
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    resolved_base_url = (
        base_url
        or os.environ.get("OPENROUTER_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")

    url = f"{resolved_base_url}/models"
    req = _urlrequest.Request(url, headers={"Accept": "application/json"})
    if resolved_key:
        req.add_header("Authorization", f"Bearer {resolved_key}")

    try:
        with _urlrequest.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise OpenRouterError(
            f"OpenRouter /models returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except URLError as exc:
        raise OpenRouterError(
            f"Could not reach OpenRouter /models: {exc.reason}"
        ) from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise OpenRouterError(
            f"OpenRouter /models returned invalid JSON: {exc}"
        ) from exc

    models = payload.get("data") or []
    if not isinstance(models, list):
        raise OpenRouterError("OpenRouter /models response missing 'data' array")

    if structured_only:
        def _supports_structured(m: Dict[str, Any]) -> bool:
            params = m.get("supported_parameters") or []
            return "response_format" in params or "structured_outputs" in params

        models = [m for m in models if _supports_structured(m)]

    return models
