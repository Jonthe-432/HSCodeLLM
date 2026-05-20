"""
Hierarchical HS / CN code classifier.

Built on top of LangChain: every level of the hierarchy is asked of the
same chat model in **one ongoing conversation**, so the model retains
full memory of its own prior choices and reasoning. This avoids the
common failure mode of stateless per-level calls where the model
re-picks a chapter it just ruled out.

The classifier walks the EU Combined Nomenclature tree
(Chapter → Heading → Subheading → CN code) and, at each level, asks the
LLM to pick from the *actual* set of valid codes at that level. The LLM
is also allowed to **backtrack** — i.e. say a higher level was wrong —
and the classifier rewinds the relevant state while keeping the
conversation history intact, so the model knows what it already ruled
out.

This module depends only on the LangChain ``BaseChatModel`` interface
plus its ``with_structured_output`` method, which makes the system fully
model-agnostic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from hscode.cn_retriever import CNCodeRetriever, get_cn_retriever
from hscode.config import SETTINGS
from hscode.llm import get_chat_model
from hscode.logging_config import get_logger
from hscode.models import (
    ClassificationResult,
    CommodityCode,
    HSCodeLevel,
)

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
    llm:
        A LangChain :class:`BaseChatModel`. If omitted, one is built from
        environment variables via :func:`hscode.llm.get_chat_model`.
    retriever:
        A :class:`CNCodeRetriever`. Defaults to the global singleton.
    max_retries:
        Maximum number of hierarchical passes before giving up. Each pass
        may be triggered by a backtrack request from the LLM.
    """

    SYSTEM_PROMPT = (
        "You are an expert in EU Combined Nomenclature (CN) and Harmonized "
        "System (HS) classification. You walk the official EU nomenclature "
        "tree level by level: Chapter (2 digits) → Heading (4 digits) → "
        "Subheading (6 digits) → CN code (8 digits). At each step you pick "
        "from the candidate set the user provides — never invent codes. "
        "If you realise an earlier choice was wrong, set code='BACKTRACK' "
        "and indicate the level to retry; the full conversation is "
        "preserved between turns, so remember your prior reasoning."
    )

    # Hard ceiling: even with backtracking, never exceed this many LLM
    # turns in a single classify() call. Prevents runaway cost if a model
    # gets stuck flipping between two chapters.
    MAX_TURNS_HARD_CAP = 30

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        retriever: Optional[CNCodeRetriever] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self._llm = llm  # lazy
        self.retriever = retriever or get_cn_retriever()
        self.max_retries = max_retries if max_retries is not None else SETTINGS.max_retries

    # ------------------------------------------------------------------
    # LLM lazy-loading
    # ------------------------------------------------------------------

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = get_chat_model()
        return self._llm

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
    # Hierarchical traversal as a single conversation
    # ------------------------------------------------------------------

    def _classify_hierarchically(
        self, description: str, cn_year: Optional[int]
    ) -> ClassificationResult:
        # Single, ever-growing conversation. Every prompt and every
        # parsed model reply is appended here, so the model can see
        # (and reason over) everything it has already said.
        history: List[BaseMessage] = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Classify this product into the 8-digit EU CN code.\n\n"
                    f"Product description: {description}\n\n"
                    "I will walk you through the hierarchy one level at a time. "
                    "Wait for the option set for each level before responding."
                )
            ),
        ]

        chapter: Optional[HSCodeLevel] = None
        heading: Optional[HSCodeLevel] = None
        subheading: Optional[HSCodeLevel] = None

        attempt = 0
        turns = 0
        while attempt < self.max_retries:
            attempt += 1
            logger.info("Hierarchical classification attempt %d/%d", attempt, self.max_retries)

            # ----- Level 1: Chapter (2-digit) ----------------------------
            if chapter is None:
                chapters = self.retriever.get_chapters()
                if not chapters:
                    return self._fail(
                        "No chapters loaded from CN database",
                        status="classification_failed",
                        cn_year=cn_year,
                        attempts=attempt,
                    )
                chapter = self._ask_level(
                    history=history,
                    level=1,
                    level_name="2-digit HS Chapter",
                    options=chapters,
                )
                turns += 1
                if turns >= self.MAX_TURNS_HARD_CAP:
                    return self._fail(
                        "Exceeded hard turn cap during chapter selection.",
                        status="classification_failed",
                        cn_year=cn_year,
                        attempts=attempt,
                    )
                if (
                    chapter.code == "BACKTRACK"
                    or len(chapter.code) != 2
                    or not chapter.code.isdigit()
                ):
                    logger.warning("Invalid chapter %r returned, retrying", chapter.code)
                    history.append(
                        HumanMessage(
                            content=(
                                f"Your previous response {chapter.code!r} is not a valid "
                                "2-digit chapter code. Please pick a different option from "
                                "the list I just showed you."
                            )
                        )
                    )
                    chapter = None
                    continue

            # ----- Level 2: Heading (4-digit) ----------------------------
            if heading is None:
                headings = self.retriever.get_headings(chapter.code)
                if not headings:
                    logger.warning("No headings for chapter %s, backtracking", chapter.code)
                    history.append(
                        HumanMessage(
                            content=(
                                f"Chapter {chapter.code} has no 4-digit headings in the EU "
                                "CN database. Please pick a different chapter."
                            )
                        )
                    )
                    chapter = None
                    continue

                heading = self._ask_level(
                    history=history,
                    level=2,
                    level_name="4-digit HS Heading",
                    options=headings,
                    parent_summary=f"You chose chapter {chapter.code}: {chapter.description}.",
                )
                turns += 1
                if turns >= self.MAX_TURNS_HARD_CAP:
                    return self._fail(
                        "Exceeded hard turn cap during heading selection.",
                        status="classification_failed",
                        cn_year=cn_year,
                        attempts=attempt,
                    )
                if heading.code == "BACKTRACK":
                    logger.info(
                        "Model requested backtrack from heading level (chapter=%s)",
                        chapter.code,
                    )
                    chapter = None
                    heading = None
                    continue
                if len(heading.code) != 4 or not heading.code.startswith(chapter.code):
                    logger.warning("Invalid heading %r, asking model to retry", heading.code)
                    history.append(
                        HumanMessage(
                            content=(
                                f"Your previous response {heading.code!r} is not a valid "
                                f"4-digit heading under chapter {chapter.code}. Please pick a "
                                "different option, or say BACKTRACK if the chapter itself "
                                "was wrong."
                            )
                        )
                    )
                    heading = None
                    continue

            # ----- Level 3: Subheading (6-digit) -------------------------
            if subheading is None:
                subheadings = self.retriever.get_subheadings(heading.code)
                if not subheadings:
                    logger.warning(
                        "No subheadings for heading %s, asking model to backtrack",
                        heading.code,
                    )
                    history.append(
                        HumanMessage(
                            content=(
                                f"Heading {heading.code} has no 6-digit subheadings in the "
                                "EU CN database. Please pick a different heading (or "
                                "BACKTRACK to the chapter level if no heading fits)."
                            )
                        )
                    )
                    heading = None
                    continue

                subheading = self._ask_level(
                    history=history,
                    level=3,
                    level_name="6-digit HS Subheading",
                    options=subheadings,
                    parent_summary=(
                        f"You chose chapter {chapter.code} and heading {heading.code}: "
                        f"{heading.description}."
                    ),
                )
                turns += 1
                if turns >= self.MAX_TURNS_HARD_CAP:
                    return self._fail(
                        "Exceeded hard turn cap during subheading selection.",
                        status="classification_failed",
                        cn_year=cn_year,
                        attempts=attempt,
                    )
                if subheading.code == "BACKTRACK":
                    bt = subheading.backtrack_to_level or 2
                    logger.info(
                        "Model requested backtrack to level %d from subheading (heading=%s)",
                        bt,
                        heading.code,
                    )
                    subheading = None
                    if bt == 1:
                        chapter = None
                        heading = None
                    else:
                        heading = None
                    continue
                if len(subheading.code) != 6 or not subheading.code.startswith(heading.code):
                    logger.warning("Invalid subheading %r, asking model to retry", subheading.code)
                    history.append(
                        HumanMessage(
                            content=(
                                f"Your previous response {subheading.code!r} is not a valid "
                                f"6-digit subheading under heading {heading.code}. Please pick "
                                "a different option, or say BACKTRACK if no option fits."
                            )
                        )
                    )
                    subheading = None
                    continue

            # ----- Level 4: CN code (8-digit) ----------------------------
            cn_codes = self.retriever.get_cn_codes(subheading.code)
            if not cn_codes:
                logger.warning(
                    "No CN codes for subheading %s, asking model to backtrack",
                    subheading.code,
                )
                history.append(
                    HumanMessage(
                        content=(
                            f"Subheading {subheading.code} has no 8-digit CN codes in the "
                            "EU CN database. Please pick a different subheading."
                        )
                    )
                )
                subheading = None
                continue

            cn_result = self._ask_cn_code(
                history=history,
                options=cn_codes,
                chapter=chapter,
                heading=heading,
                subheading=subheading,
            )
            turns += 1
            if turns >= self.MAX_TURNS_HARD_CAP:
                return self._fail(
                    "Exceeded hard turn cap during final CN code selection.",
                    status="classification_failed",
                    cn_year=cn_year,
                    attempts=attempt,
                )

            if cn_result.hs_code == "BACKTRACK":
                logger.info(
                    "Model requested backtrack from CN code level (subheading=%s)",
                    subheading.code,
                )
                subheading = None
                continue

            if (
                len(cn_result.hs_code) != 8
                or not cn_result.hs_code.isdigit()
                or not cn_result.hs_code.startswith(subheading.code)
            ):
                logger.warning(
                    "Invalid CN code %r at attempt %d, asking model to retry",
                    cn_result.hs_code,
                    attempt,
                )
                history.append(
                    HumanMessage(
                        content=(
                            f"Your previous response {cn_result.hs_code!r} is not a valid "
                            f"8-digit CN code under subheading {subheading.code}. Please "
                            "pick a different option from the list above."
                        )
                    )
                )
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
    # Per-level LLM calls — append prompt to history, then append reply
    # ------------------------------------------------------------------

    def _ask_level(
        self,
        history: List[BaseMessage],
        level: int,
        level_name: str,
        options: List[Tuple[str, str]],
        parent_summary: Optional[str] = None,
    ) -> HSCodeLevel:
        options_text = "\n".join(f"- {code}: {desc}" for code, desc in options)

        backtrack_hint = ""
        if level > 1:
            backtrack_hint = (
                "\n\nIf NONE of the options below correctly describe the product, set "
                "code='BACKTRACK' and `backtrack_to_level` to the level you want to "
                "retry from (1=Chapter, 2=Heading, 3=Subheading). Explain why in "
                "`reasoning`."
            )

        prompt_parts = [f"Now pick the {level_name}."]
        if parent_summary:
            prompt_parts.append(parent_summary)
        prompt_parts.append(f"Available options at this level:\n{options_text}")
        prompt_parts.append(
            "Return the selected code, its description, and a brief reasoning."
            + backtrack_hint
        )

        history.append(HumanMessage(content="\n\n".join(prompt_parts)))
        structured = self.llm.with_structured_output(HSCodeLevel)
        result: HSCodeLevel = structured.invoke(history)  # type: ignore[assignment]
        # Echo the model's structured reply back into the conversation
        # so subsequent turns can reference it.
        history.append(AIMessage(content=result.model_dump_json()))
        return result

    def _ask_cn_code(
        self,
        history: List[BaseMessage],
        options: List[Tuple[str, str]],
        chapter: HSCodeLevel,
        heading: HSCodeLevel,
        subheading: HSCodeLevel,
    ) -> CommodityCode:
        options_text = "\n".join(f"- {code}: {desc}" for code, desc in options)
        prompt = (
            "Now pick the final 8-digit EU Combined Nomenclature (CN) code.\n\n"
            f"You chose chapter {chapter.code}, heading {heading.code}, and subheading "
            f"{subheading.code}: {subheading.description}.\n\n"
            f"Available 8-digit CN codes within subheading {subheading.code}:\n"
            f"{options_text}\n\n"
            "Pick the most appropriate 8-digit CN code (which MUST start with "
            f"{subheading.code}). Provide:\n"
            "  - hs_code: the 8-digit code, digits only\n"
            "  - description: the official description\n"
            "  - confidence: 0.0–1.0 self-rated confidence\n"
            "  - reasoning: brief justification\n\n"
            "If none of these codes is appropriate, set hs_code='BACKTRACK' and "
            "explain why."
        )
        history.append(HumanMessage(content=prompt))
        structured = self.llm.with_structured_output(CommodityCode)
        result: CommodityCode = structured.invoke(history)  # type: ignore[assignment]
        history.append(AIMessage(content=result.model_dump_json()))
        return result

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
