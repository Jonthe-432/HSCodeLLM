"""End-to-end tests for the classifier — using a scripted LLM provider."""

from __future__ import annotations

import pytest

from hscode.classifier import HSCodeClassifier
from hscode.models import CommodityCode, HSCodeLevel


def test_regex_fastpath_skips_llm(cn_retriever, scripted_provider_factory) -> None:
    """If a valid CN code is embedded in the description, the LLM is not called."""
    provider = scripted_provider_factory([])  # zero canned answers; calling the LLM would fail
    classifier = HSCodeClassifier(provider=provider, retriever=cn_retriever)

    result = classifier.classify("Headphones, commodity code: 85183000")

    assert result.hs_code == "85183000"
    assert result.status == "regex_extracted"
    assert result.validated is True
    assert result.supplementary_unit == "PST"
    assert provider.calls == []  # no LLM calls were made
    assert result.cn_year == 2026


def test_unclear_description_skipped(cn_retriever, scripted_provider_factory) -> None:
    provider = scripted_provider_factory([])
    classifier = HSCodeClassifier(provider=provider, retriever=cn_retriever)

    result = classifier.classify("12345")
    assert result.status == "description_unclear"
    assert result.hs_code == "N/A"
    assert provider.calls == []


def test_full_hierarchical_classification(cn_retriever, scripted_provider_factory) -> None:
    """Headphones description traverses 85 → 8518 → 851830 → 85183000."""
    responses = [
        HSCodeLevel(code="85", description="Electrical machinery and equipment",
                    reasoning="Headphones are electrical equipment"),
        HSCodeLevel(code="8518", description="Microphones, loudspeakers, headphones and earphones",
                    reasoning="Heading covers headphones"),
        HSCodeLevel(code="851830", description="Headphones and earphones",
                    reasoning="Subheading is exactly headphones"),
        CommodityCode(hs_code="85183000",
                      description="Headphones and earphones, ...",
                      confidence=0.93,
                      reasoning="Best match for the product"),
    ]
    provider = scripted_provider_factory(responses)
    classifier = HSCodeClassifier(provider=provider, retriever=cn_retriever, max_retries=3)

    result = classifier.classify("Wireless bluetooth headphones with ANC")

    assert result.hs_code == "85183000"
    assert result.status == "ok"
    assert result.validated is True
    assert result.chapter == "85"
    assert result.heading == "8518"
    assert result.subheading == "851830"
    assert result.confidence == pytest.approx(0.93)
    assert result.supplementary_unit == "PST"
    assert len(provider.calls) == 4
    assert "Headphones" in result.reasoning


def test_backtracking_at_heading_level(cn_retriever, scripted_provider_factory) -> None:
    """First attempt picks chapter 61 (wrong) → backtracks → picks 85."""
    responses = [
        # Attempt 1: wrong chapter 61
        HSCodeLevel(code="61", description="Articles of apparel", reasoning="(wrong)"),
        # Heading inside chapter 61 → BACKTRACK
        HSCodeLevel(code="BACKTRACK", description="", reasoning="No fit",
                    backtrack_to_level=1),
        # Attempt 2: correct chapter 85
        HSCodeLevel(code="85", description="Electrical machinery", reasoning="ok"),
        HSCodeLevel(code="8518", description="Headphones heading", reasoning="ok"),
        HSCodeLevel(code="851830", description="Headphones subheading", reasoning="ok"),
        CommodityCode(hs_code="85183000", description="Headphones",
                      confidence=0.9, reasoning="ok"),
    ]
    provider = scripted_provider_factory(responses)
    classifier = HSCodeClassifier(provider=provider, retriever=cn_retriever, max_retries=3)

    result = classifier.classify("Wireless bluetooth headphones")

    assert result.hs_code == "85183000"
    assert result.attempts >= 2
    assert result.status == "ok"


def test_classification_fails_after_max_retries(cn_retriever, scripted_provider_factory) -> None:
    """Every attempt requests backtrack → eventually we give up."""
    responses = []
    for _ in range(20):
        responses.append(
            HSCodeLevel(code="85", description="x", reasoning="ok")
        )
        responses.append(
            HSCodeLevel(code="BACKTRACK", description="", reasoning="never satisfied",
                        backtrack_to_level=1)
        )
    provider = scripted_provider_factory(responses)
    classifier = HSCodeClassifier(provider=provider, retriever=cn_retriever, max_retries=3)

    result = classifier.classify("Some valid product description here")
    assert result.status == "classification_failed"
    assert result.hs_code == "N/A"
    assert result.attempts == 3


def test_invalid_cn_code_marked_unvalidated(cn_retriever, scripted_provider_factory) -> None:
    """LLM returns an 8-digit code with the right prefix but not in our DB."""
    responses = [
        HSCodeLevel(code="85", description="Electrical machinery", reasoning="ok"),
        HSCodeLevel(code="8518", description="Headphones", reasoning="ok"),
        HSCodeLevel(code="851830", description="Headphones", reasoning="ok"),
        CommodityCode(hs_code="85183099", description="Made-up code",
                      confidence=0.5, reasoning="invented"),
    ]
    provider = scripted_provider_factory(responses)
    classifier = HSCodeClassifier(provider=provider, retriever=cn_retriever, max_retries=2)

    result = classifier.classify("Wireless bluetooth headphones")
    # 85183099 starts with subheading 851830 so prefix check passes,
    # but it's NOT in the synthetic DB → not validated.
    assert result.hs_code == "85183099"
    assert result.status == "invalid_code"
    assert result.validated is False
