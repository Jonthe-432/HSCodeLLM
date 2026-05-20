"""Pytest fixtures shared across tests.

We never hit the real SPARQL endpoint or any LLM provider in tests — each
fixture provides an isolated, in-memory replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Type

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from hscode.cn_retriever import CNCodeRetriever

# ---------------------------------------------------------------------------
# Tiny synthetic CN nomenclature for tests
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_cn_df() -> pd.DataFrame:
    """A miniature CN dataset covering one full chain plus extras."""
    rows = [
        # Chapter 85 — Electrical machinery
        ("85", 2, "", "Electrical machinery and equipment"),
        # Heading 8518 — Microphones / loudspeakers / headphones
        ("8518", 4, "", "Microphones, loudspeakers, headphones and earphones"),
        # Subheading 851830 — Headphones and earphones
        ("851830", 6, "", "Headphones and earphones"),
        # CN 85183000 — final 8-digit code
        ("85183000", 8, "PST", "Headphones and earphones, whether or not combined with microphone"),
        # Sibling 85181010 just so we test "pick among multiple"
        ("851810", 6, "", "Microphones"),
        ("85181010", 8, "PST", "Microphones having a frequency range of 300 Hz to 3.4 kHz"),
        # Unrelated chapter to exercise chapter selection
        ("61", 2, "", "Articles of apparel, knitted or crocheted"),
        ("6109", 4, "", "T-shirts, singlets and other vests, knitted or crocheted"),
        ("610910", 6, "", "Of cotton"),
        ("61091000", 8, "PST", "T-shirts, singlets and other vests of cotton, knitted or crocheted"),
    ]
    return pd.DataFrame(
        [
            {
                "cn_code": code,
                "cn_notation": code,
                "level": level,
                "supplementary_unit": unit,
                "description": desc,
                "cn_year": 2026,
            }
            for code, level, unit, desc in rows
        ]
    )


@pytest.fixture
def cn_retriever(synthetic_cn_df: pd.DataFrame, tmp_path: Path) -> CNCodeRetriever:
    """A CNCodeRetriever pre-loaded with the synthetic dataset.

    No network is contacted because ``_cn_data`` is set directly.
    """
    retriever = CNCodeRetriever(
        target_year=2026, target_month=6, cache_dir=tmp_path / "cache"
    )
    retriever._cn_data = synthetic_cn_df.copy()
    retriever._build_lookups()
    return retriever


# ---------------------------------------------------------------------------
# Scripted LangChain chat model — returns canned Pydantic objects in order
# ---------------------------------------------------------------------------


class ScriptedChatModel(FakeListChatModel):
    """A LangChain BaseChatModel test double that returns canned structured outputs.

    ``with_structured_output(schema)`` is overridden to return a runnable
    that pops the next canned ``BaseModel`` instance from ``structured_responses``
    on each ``invoke``. The conversation history passed to ``invoke`` is
    captured in ``calls`` so tests can assert on it.

    The plain ``invoke()`` path (no structured output) falls back to the
    parent ``FakeListChatModel`` behaviour using ``string_responses``.
    """

    structured_responses: List[BaseModel] = []
    calls: List[List[BaseMessage]] = []

    @classmethod
    def script(
        cls,
        structured_responses: Sequence[BaseModel],
        string_responses: Sequence[str] = (),
    ) -> "ScriptedChatModel":
        """Construct a scripted model with canned structured replies."""
        # FakeListChatModel needs at least one string response to be valid.
        responses = list(string_responses) or ["unused"]
        instance = cls(responses=responses)
        # Pydantic v2 / langchain BaseModel: assign via __dict__ to skip
        # frozen-field protection while still keeping the same instance.
        object.__setattr__(instance, "structured_responses", list(structured_responses))
        object.__setattr__(instance, "calls", [])
        return instance

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Type[BaseModel],
        *,
        include_raw: bool = False,
        **kwargs,
    ) -> Runnable:
        captured_self = self

        def _runner(messages):
            # Normalise: invoke can receive a string, a single message, or a list.
            if isinstance(messages, list):
                history = list(messages)
            else:
                history = [messages]
            captured_self.calls.append(history)
            if not captured_self.structured_responses:
                raise AssertionError(
                    "ScriptedChatModel ran out of canned structured responses"
                )
            nxt = captured_self.structured_responses.pop(0)
            if not isinstance(nxt, schema):
                raise AssertionError(
                    f"Scripted response type mismatch: expected {schema.__name__}, "
                    f"got {type(nxt).__name__}"
                )
            return nxt

        return RunnableLambda(_runner)


@pytest.fixture
def scripted_chat_model_factory():
    """Returns a callable that builds a ScriptedChatModel from a list of responses."""
    return ScriptedChatModel.script
