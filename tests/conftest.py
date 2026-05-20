"""Pytest fixtures shared across tests.

We never hit the real SPARQL endpoint or any LLM provider in tests — each
fixture provides an isolated, in-memory replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Type

import pandas as pd
import pytest

from hscode.cn_retriever import CNCodeRetriever
from hscode.providers.base import LLMProvider, StructuredOutput

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
# Scripted LLM provider — returns canned responses in order
# ---------------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    """A test double LLMProvider that returns canned objects in order.

    Each item in ``responses`` must be a Pydantic model instance matching
    the expected schema for that call.
    """

    name = "scripted"

    def __init__(self, responses: List[StructuredOutput]) -> None:  # type: ignore[type-var]
        super().__init__(model="scripted")
        self.responses = list(responses)
        self.calls: List[Tuple[str, str, Type[StructuredOutput]]] = []

    def _call(self, system_prompt: str, user_prompt: str, schema):  # type: ignore[override]
        self.calls.append((system_prompt, user_prompt, schema))
        if not self.responses:
            raise AssertionError("ScriptedProvider ran out of canned responses")
        nxt = self.responses.pop(0)
        if not isinstance(nxt, schema):
            raise AssertionError(
                f"Scripted response type mismatch: expected {schema.__name__}, got {type(nxt).__name__}"
            )
        return nxt


@pytest.fixture
def scripted_provider_factory():
    """Returns a callable that builds a ScriptedProvider from a list of responses."""
    return ScriptedProvider
