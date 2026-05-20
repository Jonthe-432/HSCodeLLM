"""
Data models for the HSCode classifier.

All models are Pydantic-based so they double as structured-output schemas
for any LLM provider that supports JSON-schema constrained generation.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured outputs the LLM is asked to produce
# ---------------------------------------------------------------------------


class HSCodeLevel(BaseModel):
    """Output schema for one level of the hierarchical classification.

    The LLM can either return a real classification (``code`` + ``description``)
    OR request a backtrack by setting ``code='BACKTRACK'`` and
    ``backtrack_to_level`` to the level it wants to retry from.
    """

    code: str = Field(
        ...,
        description=(
            "The selected code at this level (2, 4 or 6 digits depending on "
            "level), OR the literal string 'BACKTRACK' to ask the caller to "
            "revisit an earlier level."
        ),
    )
    description: str = Field(
        "",
        description="Official description for the selected code (empty when backtracking).",
    )
    reasoning: str = Field(
        ...,
        description="Brief justification for the choice, or for the backtrack request.",
    )
    backtrack_to_level: Optional[int] = Field(
        None,
        description=(
            "When ``code='BACKTRACK'``, indicates which level to restart from: "
            "1 = Chapter, 2 = Heading, 3 = Subheading."
        ),
    )


class CommodityCode(BaseModel):
    """Final 8-digit CN/HS code returned by the LLM."""

    hs_code: str = Field(
        ...,
        description="8-digit Combined Nomenclature code (digits only, no spaces).",
    )
    description: str = Field(..., description="Official EU description for this code.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Self-reported confidence between 0.0 and 1.0."
    )
    reasoning: str = Field(..., description="Brief justification.")


# ---------------------------------------------------------------------------
# Public result returned to the caller
# ---------------------------------------------------------------------------


ClassificationStatus = Literal[
    "ok",
    "regex_extracted",
    "invalid_code",
    "description_unclear",
    "classification_failed",
]


class ClassificationResult(BaseModel):
    """Result returned by :func:`hscode.classify`."""

    hs_code: str = Field(
        ...,
        description="8-digit CN code, or 'N/A' / '99999999' when classification was not possible.",
    )
    description: str = Field(..., description="Official EU description for the code.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Reasoning chain for the final code.")

    supplementary_unit: Optional[str] = Field(
        None,
        description="EU statistical unit (e.g. 'PST', 'M2', 'NO_SU') for Intrastat reporting.",
    )
    supplementary_unit_description: Optional[str] = Field(
        None, description="Human-readable description of the supplementary unit."
    )

    chapter: Optional[str] = Field(None, description="Chosen 2-digit chapter.")
    heading: Optional[str] = Field(None, description="Chosen 4-digit heading.")
    subheading: Optional[str] = Field(None, description="Chosen 6-digit subheading.")

    status: ClassificationStatus = Field(
        "ok", description="Outcome category for programmatic handling."
    )
    validated: bool = Field(
        False, description="Whether the final code was validated against the EU CN database."
    )
    attempts: int = Field(1, ge=1, description="Number of hierarchical passes performed.")
    cn_year: Optional[int] = Field(None, description="CN nomenclature year used.")

    @property
    def is_valid(self) -> bool:
        """True if a real 8-digit CN code was returned and validated."""
        return self.validated and self.hs_code.isdigit() and len(self.hs_code) == 8
