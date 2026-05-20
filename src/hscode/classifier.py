"""
Hierarchical HS / CN code classifier.

The classifier walks the EU Combined Nomenclature tree
(Chapter → Heading → Subheading → CN code) and, at each level, asks the LLM
to pick from the *actual* set of valid codes at that level. The LLM is also
allowed to **backtrack** — i.e. tell us that a higher level was wrong — and
the classifier retries with the requested level cleared.

This module is fully model-agnostic: it only depends on the
:class:`~hscode.providers.LLMProvider` interface.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from hscode.cn_retriever import CNCodeRetriever, get_cn_retriever
from hscode.config import SETTINGS
from hscode.logging_config import get_logger
from hscode.models import (
    ClassificationResult,
    CommodityCode,
    HSCodeLevel,
)
from hscode.providers import LLMProvider, ProviderError, get_provider

logger = get_logger("classifier")


# ---------------------------------------------------------------------------
# Heuristics for skipping LLM calls
# ---------------------------------------------------------------------------


_MIN_WORDS = 2
_MIN_LETTERS = 5


def _description_is_classifiable(description: str) -> bool:
    """Lightweight heuristic — does this description have enough signal?"""
    import re

    if not description or not description.strip():
        return False
    clean = re.sub(r"[^A-Za-z\s]", "", description)
    words = clean.split()
    if len(words) < _MIN_WORDS:
        return False
    if len(clean.replace(" ", "")) < _MIN_LETTERS:
        return False
    if re.match(r"^[\d\-\.\s]+$", description.strip()):
        return False
    return True


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class HSCodeClassifier:
    """Production-ready HS code classifier.

    Parameters
    ----------
    provider:
        An :class:`LLMProvider` instance. If omitted, a provider is built
        from environment variables via :func:`hscode.providers.get_provider`.
    retriever:
        A :class:`CNCodeRetriever`. Defaults to the global singleton.
    max_retries:
        Maximum number of hierarchical passes before giving up. Each pass
        may be triggered by a backtrack request from the LLM.
    """

    SYSTEM_PROMPT = (
        "You are an expert in EU Combined Nomenclature (CN) and Harmonized "
        "System (HS) classification. You select the correct code based on "
        "the official EU nomenclature data provided in each prompt. "
        "If none of the options provided actually fit the product, you set "
        "code='BACKTRACK' and indicate which level to retry."
    )

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        retriever: Optional[CNCodeRetriever] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self._provider = provider  # lazy
        self.retriever = retriever or get_cn_retriever()
        self.max_retries = max_retries if max_retries is not None else SETTINGS.max_retries

    # ------------------------------------------------------------------
    # Provider lazy-loading
    # ------------------------------------------------------------------

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider()
        return self._provider

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def classify(self, description: str) -> ClassificationResult:
        """Classify a single product description into an 8-digit CN code."""
        if not description or not description.strip():
            return self._fail("Empty description provided", status="description_unclear")

        # Ensure nomenclature data is loaded — this also gives us cn_year.
        df = self.retriever.load()
        cn_year: Optional[int] = None
        if not df.empty and "cn_year" in df.columns:
            try:
                cn_year = int(df["cn_year"].iloc[0])
            except (ValueError, TypeError):
                cn_year = None

        # ----- 1. Regex fast-path ----------------------------------------
        embedded = self.retriever.extract_embedded_code(description)
        if embedded:
            is_valid, desc, supp, supp_desc = self.retriever.validate_and_get_info(embedded)
            if is_valid:
                logger.info("Regex extracted valid CN code %s from description", embedded)
                return ClassificationResult(
                    hs_code=embedded,
                    description=desc or "",
                    confidence=0.95,
                    reasoning=(
                        f"The 8-digit CN code {embedded} was extracted directly from the "
                        f"product description and validated against the EU CN database."
                    ),
                    supplementary_unit=supp,
                    supplementary_unit_description=supp_desc,
                    chapter=embedded[:2],
                    heading=embedded[:4],
                    subheading=embedded[:6],
                    status="regex_extracted",
                    validated=True,
                    attempts=1,
                    cn_year=cn_year,
                )

        # ----- 2. Quality gate -------------------------------------------
        if not _description_is_classifiable(description):
            logger.info("Description is unclear, refusing to classify: %r", description)
            return self._fail(
                f"Description {description!r} does not contain enough information "
                "to determine a commodity code (only codes/numbers/short text).",
                status="description_unclear",
                cn_year=cn_year,
            )

        # ----- 3. Hierarchical LLM classification ------------------------
        return self._classify_hierarchically(description, cn_year=cn_year)

    # ------------------------------------------------------------------
    # Hierarchical traversal with backtracking
    # ------------------------------------------------------------------

    def _classify_hierarchically(
        self, description: str, cn_year: Optional[int]
    ) -> ClassificationResult:
        chapter: Optional[HSCodeLevel] = None
        heading: Optional[HSCodeLevel] = None
        subheading: Optional[HSCodeLevel] = None

        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            logger.info("Hierarchical classification attempt %d/%d", attempt, self.max_retries)

            # ----- Level 1: Chapter (2-digit) ----------------------------
            if chapter is None:
                chapters = self.retriever.get_chapters()
                if not chapters:
                    return self._fail(
                        "No chapters loaded from CN database", status="classification_failed",
                        cn_year=cn_year, attempts=attempt,
                    )
                chapter = self._pick_level(
                    description=description,
                    level=1,
                    level_name="2-digit HS Chapter",
                    options=chapters,
                    parent_path=[],
                )
                # Chapter cannot legitimately backtrack — nothing above it.
                if chapter.code == "BACKTRACK" or len(chapter.code) != 2 or not chapter.code.isdigit():
                    logger.warning("Invalid chapter %r returned, retrying", chapter.code)
                    chapter = None
                    continue

            # ----- Level 2: Heading (4-digit) ----------------------------
            if heading is None:
                headings = self.retriever.get_headings(chapter.code)
                if not headings:
                    logger.warning("No headings for chapter %s, backtracking", chapter.code)
                    chapter = None
                    continue
                heading = self._pick_level(
                    description=description,
                    level=2,
                    level_name="4-digit HS Heading",
                    options=headings,
                    parent_path=[("Chapter", chapter)],
                )
                if heading.code == "BACKTRACK":
                    chapter = None
                    heading = None
                    continue
                if len(heading.code) != 4 or not heading.code.startswith(chapter.code):
                    logger.warning("Invalid heading %r, backtracking", heading.code)
                    chapter = None
                    heading = None
                    continue

            # ----- Level 3: Subheading (6-digit) -------------------------
            if subheading is None:
                subheadings = self.retriever.get_subheadings(heading.code)
                if not subheadings:
                    logger.warning("No subheadings for heading %s, backtracking", heading.code)
                    heading = None
                    continue
                subheading = self._pick_level(
                    description=description,
                    level=3,
                    level_name="6-digit HS Subheading",
                    options=subheadings,
                    parent_path=[("Chapter", chapter), ("Heading", heading)],
                )
                if subheading.code == "BACKTRACK":
                    bt = subheading.backtrack_to_level or 2
                    subheading = None
                    if bt == 1:
                        chapter = None
                        heading = None
                    else:
                        heading = None
                    continue
                if len(subheading.code) != 6 or not subheading.code.startswith(heading.code):
                    logger.warning("Invalid subheading %r, backtracking", subheading.code)
                    heading = None
                    subheading = None
                    continue

            # ----- Level 4: CN code (8-digit) ----------------------------
            cn_codes = self.retriever.get_cn_codes(subheading.code)
            if not cn_codes:
                logger.warning("No CN codes for subheading %s, backtracking", subheading.code)
                subheading = None
                continue

            cn_result = self._pick_cn_code(
                description=description,
                options=cn_codes,
                chapter=chapter,
                heading=heading,
                subheading=subheading,
            )

            if cn_result.hs_code == "BACKTRACK":
                # The model used a CommodityCode to signal backtrack — find which level.
                # We re-ask at the highest reasonable level (subheading) for simplicity.
                subheading = None
                continue

            if (
                len(cn_result.hs_code) != 8
                or not cn_result.hs_code.isdigit()
                or not cn_result.hs_code.startswith(subheading.code)
            ):
                logger.warning(
                    "Invalid CN code %r at attempt %d, retrying", cn_result.hs_code, attempt
                )
                subheading = None
                continue

            # ----- Validate against SPARQL CN database -------------------
            is_valid, official_desc, supp, supp_desc = self.retriever.validate_and_get_info(
                cn_result.hs_code
            )

            return ClassificationResult(
                hs_code=cn_result.hs_code,
                description=official_desc or cn_result.description,
                confidence=cn_result.confidence,
                reasoning=self._compose_reasoning(chapter, heading, subheading, cn_result),
                supplementary_unit=supp,
                supplementary_unit_description=supp_desc,
                chapter=chapter.code,
                heading=heading.code,
                subheading=subheading.code,
                status="ok" if is_valid else "invalid_code",
                validated=is_valid,
                attempts=attempt,
                cn_year=cn_year,
            )

        # Exhausted retries
        logger.error("Classification failed after %d attempts", self.max_retries)
        return self._fail(
            f"Could not classify after {self.max_retries} hierarchical passes.",
            status="classification_failed",
            cn_year=cn_year,
            attempts=self.max_retries,
        )

    # ------------------------------------------------------------------
    # Per-level LLM calls
    # ------------------------------------------------------------------

    def _pick_level(
        self,
        description: str,
        level: int,
        level_name: str,
        options: List[Tuple[str, str]],
        parent_path: List[Tuple[str, HSCodeLevel]],
    ) -> HSCodeLevel:
        options_text = "\n".join(f"- {code}: {desc}" for code, desc in options)
        parent_block = ""
        if parent_path:
            parent_block = "\nCurrent path:\n" + "\n".join(
                f"- {name}: {lvl.code} — {lvl.description}" for name, lvl in parent_path
            )

        backtrack_hint = ""
        if level > 1:
            backtrack_hint = (
                "\n\nIf NONE of the options below correctly describe the product, set "
                "code='BACKTRACK' and `backtrack_to_level` to the level you want to retry "
                "from (1=Chapter, 2=Heading, 3=Subheading). Explain why in `reasoning`."
            )

        user_prompt = (
            f"Determine the {level_name} for the following product.\n\n"
            f"Product description: {description}\n"
            f"{parent_block}\n\n"
            f"Available options at this level:\n{options_text}\n"
            f"{backtrack_hint}\n\n"
            "Return the selected code, its description, and a brief reasoning."
        )

        try:
            return self.provider.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=HSCodeLevel,
            )
        except ProviderError as exc:
            logger.error("Provider failed at level %d: %s", level, exc)
            raise

    def _pick_cn_code(
        self,
        description: str,
        options: List[Tuple[str, str]],
        chapter: HSCodeLevel,
        heading: HSCodeLevel,
        subheading: HSCodeLevel,
    ) -> CommodityCode:
        options_text = "\n".join(f"- {code}: {desc}" for code, desc in options)
        user_prompt = (
            "Determine the final 8-digit EU Combined Nomenclature (CN) code for this product.\n\n"
            f"Product description: {description}\n\n"
            "Current path:\n"
            f"- Chapter:    {chapter.code} — {chapter.description}\n"
            f"- Heading:    {heading.code} — {heading.description}\n"
            f"- Subheading: {subheading.code} — {subheading.description}\n\n"
            f"Available 8-digit CN codes within subheading {subheading.code}:\n"
            f"{options_text}\n\n"
            "Pick the most appropriate 8-digit CN code (which MUST start with "
            f"{subheading.code}). Provide:\n"
            "  - hs_code: the 8-digit code, digits only\n"
            "  - description: the official description\n"
            "  - confidence: 0.0–1.0 self-rated confidence\n"
            "  - reasoning: brief justification\n\n"
            "If none of these codes is appropriate, set hs_code='BACKTRACK' and explain why."
        )

        try:
            return self.provider.generate_structured(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=CommodityCode,
            )
        except ProviderError as exc:
            logger.error("Provider failed at CN code level: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_reasoning(
        chapter: HSCodeLevel,
        heading: HSCodeLevel,
        subheading: HSCodeLevel,
        cn_result: CommodityCode,
    ) -> str:
        return (
            f"Chapter {chapter.code} ({chapter.description}): {chapter.reasoning}\n"
            f"Heading {heading.code} ({heading.description}): {heading.reasoning}\n"
            f"Subheading {subheading.code} ({subheading.description}): {subheading.reasoning}\n"
            f"CN code {cn_result.hs_code}: {cn_result.reasoning}"
        )

    @staticmethod
    def _fail(
        reason: str,
        status: str,
        cn_year: Optional[int] = None,
        attempts: int = 1,
    ) -> ClassificationResult:
        return ClassificationResult(
            hs_code="N/A",
            description=reason,
            confidence=0.0,
            reasoning=reason,
            supplementary_unit=None,
            supplementary_unit_description=None,
            chapter=None,
            heading=None,
            subheading=None,
            status=status,  # type: ignore[arg-type]
            validated=False,
            attempts=attempts,
            cn_year=cn_year,
        )
