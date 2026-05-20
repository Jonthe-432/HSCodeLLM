"""Tests for the CN retriever — purely offline using the synthetic dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from hscode.cn_retriever import CNCodeRetriever


def test_validate_known_code(cn_retriever: CNCodeRetriever) -> None:
    is_valid, desc, unit, unit_desc = cn_retriever.validate_and_get_info("85183000")
    assert is_valid is True
    assert "Headphones" in desc
    assert unit == "PST"
    assert "iece" in unit_desc.lower() or "items" in unit_desc.lower()


def test_validate_unknown_code(cn_retriever: CNCodeRetriever) -> None:
    is_valid, desc, unit, unit_desc = cn_retriever.validate_and_get_info("99999999")
    assert is_valid is False
    assert desc is None and unit is None and unit_desc is None


def test_validate_garbage_input(cn_retriever: CNCodeRetriever) -> None:
    for bad in ["", "abc", "123", "12345678901"]:
        is_valid, *_ = cn_retriever.validate_and_get_info(bad)
        assert is_valid is False


def test_clean_code_normalises_separators(cn_retriever: CNCodeRetriever) -> None:
    # Dots, dashes, spaces should be stripped.
    assert cn_retriever.is_valid("8518.30.00")
    assert cn_retriever.is_valid("8518 30 00")
    assert cn_retriever.is_valid("8518-30-00")


def test_hierarchy_navigation(cn_retriever: CNCodeRetriever) -> None:
    chapters = cn_retriever.get_chapters()
    chapter_codes = {c for c, _ in chapters}
    assert "85" in chapter_codes and "61" in chapter_codes

    headings = cn_retriever.get_headings("85")
    assert ("8518", "Microphones, loudspeakers, headphones and earphones") in headings
    # All headings must start with "85"
    assert all(c.startswith("85") for c, _ in headings)

    subheadings = cn_retriever.get_subheadings("8518")
    sub_codes = {c for c, _ in subheadings}
    assert "851830" in sub_codes and "851810" in sub_codes

    cn_codes = cn_retriever.get_cn_codes("851830")
    cn_set = {c for c, _ in cn_codes}
    assert cn_set == {"85183000"}


def test_extract_embedded_code_with_label(cn_retriever: CNCodeRetriever) -> None:
    assert cn_retriever.extract_embedded_code("Product XYZ commodity code: 85183000") == "85183000"
    assert cn_retriever.extract_embedded_code("Cotton t-shirt CN 61091000") == "61091000"


def test_extract_embedded_code_eu_format(cn_retriever: CNCodeRetriever) -> None:
    assert cn_retriever.extract_embedded_code("Article 6109 10 00 cotton") == "61091000"


def test_extract_embedded_code_ignored_when_not_in_db(cn_retriever: CNCodeRetriever) -> None:
    # 12345678 isn't a valid CN code in our synthetic dataset
    assert cn_retriever.extract_embedded_code("part number 12345678") is None


def test_extract_embedded_code_handles_empty_text(cn_retriever: CNCodeRetriever) -> None:
    assert cn_retriever.extract_embedded_code("") is None
    assert cn_retriever.extract_embedded_code(None) is None  # type: ignore[arg-type]
