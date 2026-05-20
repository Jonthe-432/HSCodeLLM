"""
Base interface for LLM providers.

Every provider must implement :meth:`LLMProvider.generate_structured`,
which takes a system prompt, a user prompt, and a Pydantic model class
defining the expected output schema. The provider is responsible for:

  * sending the prompts to the model,
  * constraining or coercing the model's response to the schema,
  * returning a validated Pydantic instance.

This is the *only* coupling between the classifier and the underlying
model SDK, which makes the system fully model-agnostic.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hscode.logging_config import get_logger

logger = get_logger("providers")

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised when an LLM provider fails to produce a usable response."""


class LLMProvider(ABC):
    """Abstract base class for all LLM providers used by HSCode."""

    name: str = "abstract"

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # Public API — subclasses override `_call`
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(ProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        """Generate a response conforming to ``schema``.

        Retries with exponential backoff on transient provider errors.
        """
        try:
            return self._call(system_prompt, user_prompt, schema)
        except ProviderError:
            raise
        except ValidationError as exc:
            raise ProviderError(
                f"{self.name}: response did not match schema {schema.__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"{self.name}: {type(exc).__name__}: {exc}") from exc

    @abstractmethod
    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[StructuredOutput],
    ) -> StructuredOutput:
        """Provider-specific implementation. Subclasses must override."""

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def parse_json_response(raw: str, schema: Type[StructuredOutput]) -> StructuredOutput:
        """Best-effort parsing of a model response into a Pydantic model.

        Handles common patterns:
            * Plain JSON object
            * JSON wrapped in ```json ... ``` fences
            * JSON object embedded in surrounding prose
        """
        if not raw:
            raise ProviderError("Empty response from model")

        # Strip code fences
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Try direct parse
        try:
            return schema.model_validate_json(cleaned)
        except (ValidationError, ValueError):
            pass

        # Try to extract first JSON object via brace matching
        first_brace = cleaned.find("{")
        if first_brace >= 0:
            depth = 0
            for i in range(first_brace, len(cleaned)):
                ch = cleaned[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[first_brace : i + 1]
                        try:
                            return schema.model_validate_json(candidate)
                        except (ValidationError, ValueError):
                            try:
                                return schema.model_validate(json.loads(candidate))
                            except Exception:
                                break

        raise ProviderError(
            f"Could not parse model response into {schema.__name__}: {raw[:300]}"
        )

    @staticmethod
    def schema_instructions(schema: Type[BaseModel]) -> str:
        """Generate a short instruction block describing a JSON schema.

        Used as a fallback when the provider doesn't natively support
        constrained / structured output.
        """
        schema_json = schema.model_json_schema()
        return (
            "Respond with a single JSON object matching this schema. "
            "Do not include any prose, explanation, or markdown — only the JSON object.\n"
            f"Schema:\n{json.dumps(schema_json, indent=2)}"
        )
