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


# ---------------------------------------------------------------------------
# Regression: ~33% of EU CN headings emit no level-6 rows from SPARQL.
# get_subheadings() must synthesise them from level-8 prefixes so the
# hierarchical classifier can still narrow down. Models a heading shaped
# like the real 8518 entry in CN 2026.
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_heading_retriever(tmp_path) -> CNCodeRetriever:
    rows = [
        # Heading 8518 — only level-4 and level-8 (no level-6 in source data)
        ("85", 2, "", "Electrical machinery and equipment"),
        ("8518", 4, "", "Microphones, loudspeakers, headphones and earphones"),
        ("85181000", 8, "NO_SU", "Microphones and stands therefor"),
        ("85182100", 8, "PST", "Single loudspeakers, mounted in their enclosures"),
        ("85182200", 8, "PST", "Multiple loudspeakers, mounted in the same enclosure"),
        ("85182900", 8, "PST", "Other"),
        ("85183000", 8, "NO_SU", "Headphones and earphones"),
        ("85184000", 8, "PST", "Audio-frequency electric amplifiers"),
        ("85185000", 8, "PST", "Electric sound amplifier sets"),
        ("85189000", 8, "NO_SU", "Parts"),
    ]
    df = pd.DataFrame(
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
    retriever = CNCodeRetriever(target_year=2026, target_month=6, cache_dir=tmp_path)
    retriever._cn_data = df
    retriever._build_lookups()
    return retriever


def test_subheadings_synthesised_when_missing(
    flat_heading_retriever: CNCodeRetriever,
) -> None:
    """If SPARQL emits no level-6 rows, derive them from level-8 prefixes."""
    subs = flat_heading_retriever.get_subheadings("8518")
    codes = {c for c, _ in subs}
    # One synthesised subheading per distinct 6-digit prefix.
    assert codes == {
        "851810",
        "851821",
        "851822",
        "851829",
        "851830",
        "851840",
        "851850",
        "851890",
    }
    # Single-child prefixes reuse the level-8 description verbatim.
    desc_by_code = dict(subs)
    assert desc_by_code["851830"] == "Headphones and earphones"
    assert desc_by_code["851810"] == "Microphones and stands therefor"


def test_subheadings_explicit_take_precedence_over_synthesis(
    cn_retriever: CNCodeRetriever,
) -> None:
    """When real level-6 rows exist, they must be returned as-is (no synthesis)."""
    subs = cn_retriever.get_subheadings("8518")
    codes = {c for c, _ in subs}
    # The synthetic conftest dataset provides explicit level-6 rows.
    assert "851830" in codes and "851810" in codes
    desc_by_code = dict(subs)
    # Explicit level-6 description, NOT the synthesised "Subheading 851830 (covers …)" form.
    assert desc_by_code["851830"] == "Headphones and earphones"


def test_subheadings_empty_when_heading_has_no_children(
    flat_heading_retriever: CNCodeRetriever,
) -> None:
    """A heading that doesn't exist in the data must yield no subheadings."""
    assert flat_heading_retriever.get_subheadings("9999") == []
