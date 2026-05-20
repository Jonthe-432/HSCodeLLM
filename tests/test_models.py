"""Tests for the public data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hscode.models import ClassificationResult, CommodityCode, HSCodeLevel


def test_classification_result_is_valid_flag() -> None:
    r = ClassificationResult(
        hs_code="85183000",
        description="Headphones",
        confidence=0.9,
        reasoning="",
        validated=True,
    )
    assert r.is_valid is True

    r2 = r.model_copy(update={"validated": False})
    assert r2.is_valid is False

    r3 = r.model_copy(update={"hs_code": "N/A"})
    assert r3.is_valid is False


def test_commodity_code_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        CommodityCode(hs_code="12345678", description="x", confidence=1.5, reasoning="x")
    with pytest.raises(ValidationError):
        CommodityCode(hs_code="12345678", description="x", confidence=-0.1, reasoning="x")


def test_hscode_level_backtrack_signal() -> None:
    lvl = HSCodeLevel(code="BACKTRACK", reasoning="wrong chapter", backtrack_to_level=1)
    assert lvl.code == "BACKTRACK"
    assert lvl.backtrack_to_level == 1
