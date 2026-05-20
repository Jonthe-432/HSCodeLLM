"""
CN Code Retriever — fetches the EU Combined Nomenclature from the EU
Publications Office SPARQL endpoint, caches it locally, and exposes:

  * validation of 8-digit CN codes
  * hierarchy navigation (chapter / heading / subheading / CN code)
  * regex-based extraction of codes embedded in free text
  * supplementary-unit lookup

The retriever is intentionally self-contained — it has no dependency on
LLMs or providers, so it can be unit-tested in isolation.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

from hscode.config import SETTINGS
from hscode.logging_config import get_logger

logger = get_logger("cn_retriever")


class CNCodeRetriever:
    """Retrieves CN codes and supplementary units from the EU SPARQL endpoint."""

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    EMBEDDED_CODE_PATTERNS: Tuple[str, ...] = (
        # Explicit labels followed by code
        r"(?:commodity|hs|cn|tariff|goods)\s*(?:code)?[:\s.\-=]+(\d{8})",
        r"(?:code|nummer|kode|c[oó]digo)[:\s.\-=]+(\d{8})",
        # Code followed by a context word
        r"\b(\d{8})\b\s*[-–]\s*(?:commodity|tariff|cn|hs)",
        # 8-digit code in parentheses
        r"\((\d{8})\)",
        # EU printed format: 4 digits + 2 digits + 2 digits
        r"(\d{4})\s+(\d{2})\s+(\d{2})",
    )

    UNIT_DESCRIPTIONS: Dict[str, str] = {
        "NO_SU": "No supplementary unit required",
        "PST": "Pieces / items (number)",
        "M2": "Square metres",
        "M3": "Cubic metres",
        "L": "Litres",
        "PA": "Pairs",
        "G": "Grams",
        "KG": "Kilograms",
        "M": "Metres",
        "1000_PST": "Thousands of pieces",
        "1000_KWH": "Thousands of kilowatt-hours",
        "L_ALC_100PCT": "Litres of pure alcohol",
        "TJ": "Terajoules",
        "KG_90PCT_SDT": "Kilograms of 90% dry matter",
        "KG_N": "Kilograms of nitrogen",
        "KG_NET_EDA": "Kilograms net drained weight",
        "KG_P2O5": "Kilograms of phosphorus pentoxide",
        "KG_K2O": "Kilograms of potassium oxide",
        "KG_U": "Kilograms of uranium",
        "KG_NAOH": "Kilograms of sodium hydroxide",
        "KG_KOH": "Kilograms of potassium hydroxide",
        "GI_FS": "Grams of fissile isotopes",
        "CTL": "Carrying capacity in tonnes",
        "CEEL": "Number of cells",
        "CK": "Carats",
    }

    # Single global lock for the singleton accessor
    _instance_lock: Lock = Lock()
    _instance: Optional["CNCodeRetriever"] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        target_year: Optional[int] = None,
        target_month: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        sparql_endpoint: Optional[str] = None,
        sparql_timeout: Optional[int] = None,
    ) -> None:
        now = datetime.now()
        self.target_month = int(target_month) if target_month else now.month

        if target_year is not None:
            self.target_year = int(target_year)
        elif SETTINGS.cn_year:
            self.target_year = int(SETTINGS.cn_year)
        elif self.target_month == 1:
            self.target_year = now.year - 1
            logger.info("January detected — using previous year's CN codes: %d", self.target_year)
        else:
            self.target_year = now.year

        self.cache_dir = Path(cache_dir) if cache_dir else SETTINGS.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.sparql_endpoint = sparql_endpoint or SETTINGS.sparql_endpoint
        self.sparql_timeout = sparql_timeout or SETTINGS.sparql_timeout

        self._cn_data: Optional[pd.DataFrame] = None
        self._valid_codes: Optional[Set[str]] = None
        self._unit_mapping: Optional[Dict[str, str]] = None
        self._description_mapping: Optional[Dict[str, str]] = None
        self._lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.cache_dir / f"cn_codes_{self.target_year}.csv"

    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            now = datetime.now()
            cache_valid = (
                self.target_year == now.year and self.target_month == now.month
            ) or (self.target_year == now.year - 1 and now.month == 1)
            if not cache_valid:
                logger.info(
                    "CN cache for year %d is stale (now %d-%02d), will refresh",
                    self.target_year,
                    now.year,
                    now.month,
                )
                # We still load it as a fallback in case the network is unavailable.
            df = pd.read_csv(path, dtype=str)
            logger.info("Loaded %d CN entries from cache: %s", len(df), path)
            return df if cache_valid else None
        except Exception as exc:  # pragma: no cover - cache corruption is rare
            logger.warning("Failed to read CN cache %s: %s", path, exc)
            return None

    def _load_stale_cache(self) -> Optional[pd.DataFrame]:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, dtype=str)
            logger.warning("Falling back to stale CN cache: %s (%d entries)", path, len(df))
            return df
        except Exception:  # pragma: no cover
            return None

    def _save_to_cache(self, df: pd.DataFrame) -> None:
        path = self._cache_path()
        try:
            df.to_csv(path, index=False)
            logger.info("Saved %d CN entries to cache: %s", len(df), path)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write CN cache %s: %s", path, exc)

    # ------------------------------------------------------------------
    # SPARQL fetching
    # ------------------------------------------------------------------

    def _get_latest_cn_year(self) -> int:
        query = (
            "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
            "SELECT DISTINCT ?scheme WHERE { "
            "?scheme a skos:ConceptScheme . "
            'FILTER(CONTAINS(STR(?scheme), "data.europa.eu/xsp/cn")) '
            "}"
        )
        try:
            resp = requests.get(
                self.sparql_endpoint,
                params={"query": query, "format": "json"},
                timeout=60,
            )
            resp.raise_for_status()
            if not resp.text.strip():
                return self.target_year
            data = resp.json()
            years = []
            for row in data.get("results", {}).get("bindings", []):
                scheme = row.get("scheme", {}).get("value", "")
                m = re.search(r"cn(\d{4})", scheme)
                if m:
                    years.append(int(m.group(1)))
            if years:
                latest = max(years)
                logger.info("Available CN years: %s, latest: %d", sorted(set(years)), latest)
                return latest
            return self.target_year
        except Exception as exc:
            logger.warning("Could not determine latest CN year: %s", exc)
            return self.target_year

    def _fetch_from_sparql(self) -> pd.DataFrame:
        available_year = self._get_latest_cn_year()
        cn_year = self.target_year if self.target_year <= available_year else available_year

        logger.info("Fetching CN %d data from %s ...", cn_year, self.sparql_endpoint)

        query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX m8g: <http://data.europa.eu/m8g/>
        SELECT ?notation ?label ?unit
        FROM <http://data.europa.eu/xsp/cn{cn_year}/cn{cn_year}>
        WHERE {{
          ?concept skos:notation ?notation .
          OPTIONAL {{ ?concept m8g:statUnitMeasure ?unit . }}
          OPTIONAL {{ ?concept skos:prefLabel ?label . FILTER(LANG(?label) = "en") }}
        }}
        ORDER BY ?notation
        """
        try:
            resp = requests.get(
                self.sparql_endpoint,
                params={"query": query, "format": "json"},
                timeout=self.sparql_timeout,
            )
            resp.raise_for_status()
            if not resp.text.strip():
                logger.error("SPARQL endpoint returned an empty body")
                return _empty_cn_df()
            data = resp.json()
            results = data.get("results", {}).get("bindings", [])
            logger.info("Retrieved %d CN entries from SPARQL", len(results))

            records = []
            for row in results:
                notation = row.get("notation", {}).get("value", "")
                cn_code = notation.replace(" ", "")
                if not cn_code.isdigit():
                    continue
                unit_uri = row.get("unit", {}).get("value", "")
                unit_code = unit_uri.split("/")[-1] if unit_uri else ""
                label = (row.get("label", {}).get("value", "") or "")[:300]
                records.append(
                    {
                        "cn_code": cn_code,
                        "cn_notation": notation,
                        "level": len(cn_code),
                        "supplementary_unit": unit_code,
                        "description": label,
                        "cn_year": cn_year,
                    }
                )
            return pd.DataFrame.from_records(records) if records else _empty_cn_df()
        except Exception as exc:
            logger.error("SPARQL fetch failed: %s", exc)
            return _empty_cn_df()

    # ------------------------------------------------------------------
    # Public data loading
    # ------------------------------------------------------------------

    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Return the CN DataFrame, loading from cache or SPARQL as needed."""
        with self._lock:
            if self._cn_data is not None and not force_refresh:
                return self._cn_data

            if not force_refresh:
                cached = self._load_from_cache()
                if cached is not None and not cached.empty:
                    self._cn_data = cached
                    self._build_lookups()
                    return self._cn_data

            df = self._fetch_from_sparql()
            if df.empty:
                stale = self._load_stale_cache()
                if stale is not None and not stale.empty:
                    df = stale
                else:
                    raise CNDataUnavailableError(
                        "Could not load CN data from SPARQL or cache. "
                        "Check your network connection."
                    )
            else:
                self._save_to_cache(df)

            self._cn_data = df
            self._build_lookups()
            return self._cn_data

    def _build_lookups(self) -> None:
        assert self._cn_data is not None
        df = self._cn_data.copy()
        df["cn_code"] = df["cn_code"].astype(str).str.strip()
        if "level" in df.columns:
            df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(0).astype(int)
        else:
            df["level"] = df["cn_code"].str.len()

        cn8 = df[df["level"] == 8]
        self._valid_codes = set(cn8["cn_code"])
        self._unit_mapping = dict(
            zip(cn8["cn_code"], cn8["supplementary_unit"].astype(str).str.strip())
        )
        self._description_mapping = dict(
            zip(cn8["cn_code"], cn8["description"].astype(str).str.strip())
        )
        self._cn_data = df
        logger.info(
            "Indexed %d 8-digit CN codes (%d total entries across all levels)",
            len(self._valid_codes),
            len(df),
        )

    # ------------------------------------------------------------------
    # Hierarchy navigation
    # ------------------------------------------------------------------

    def _level_rows(self, level: int) -> List[Tuple[str, str]]:
        df = self.load()
        rows = (
            df[df["level"] == level][["cn_code", "description"]]
            .drop_duplicates("cn_code")
            .fillna("")
        )
        return [(str(r["cn_code"]), str(r["description"])) for _, r in rows.iterrows()]

    def get_chapters(self) -> List[Tuple[str, str]]:
        return self._level_rows(2)

    def get_headings(self, chapter: str) -> List[Tuple[str, str]]:
        chapter = chapter.strip()
        return [(c, d) for c, d in self._level_rows(4) if c.startswith(chapter)]

    def get_subheadings(self, heading: str) -> List[Tuple[str, str]]:
        heading = heading.strip()
        return [(c, d) for c, d in self._level_rows(6) if c.startswith(heading)]

    def get_cn_codes(self, subheading: str) -> List[Tuple[str, str]]:
        subheading = subheading.strip()
        return [(c, d) for c, d in self._level_rows(8) if c.startswith(subheading)]

    # ------------------------------------------------------------------
    # Validation / lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_code(code: str) -> str:
        if not code:
            return ""
        clean = re.sub(r"[.\s\-]", "", str(code)).strip()
        if clean.isdigit() and len(clean) < 8:
            clean = clean.zfill(8)
        return clean

    def is_valid(self, code: str) -> bool:
        self.load()
        assert self._valid_codes is not None
        return self._clean_code(code) in self._valid_codes

    def get_description(self, code: str) -> Optional[str]:
        self.load()
        assert self._description_mapping is not None
        return self._description_mapping.get(self._clean_code(code))

    def get_supplementary_unit(self, code: str) -> Optional[str]:
        self.load()
        assert self._unit_mapping is not None
        return self._unit_mapping.get(self._clean_code(code))

    def get_unit_description(self, unit_code: Optional[str]) -> Optional[str]:
        if not unit_code:
            return None
        return self.UNIT_DESCRIPTIONS.get(unit_code, unit_code)

    def validate_and_get_info(
        self, code: str
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Return ``(is_valid, description, supp_unit, supp_unit_description)``."""
        clean = self._clean_code(code)
        if not clean or len(clean) != 8 or not clean.isdigit():
            return False, None, None, None
        if not self.is_valid(clean):
            return False, None, None, None
        desc = self.get_description(clean)
        supp = self.get_supplementary_unit(clean)
        return True, desc, supp, self.get_unit_description(supp)

    # ------------------------------------------------------------------
    # Embedded-code extraction
    # ------------------------------------------------------------------

    def extract_embedded_code(self, text: str) -> Optional[str]:
        """Return a valid 8-digit CN code mentioned in ``text``, or ``None``."""
        if not text:
            return None
        self.load()
        assert self._valid_codes is not None

        text_low = text.lower()
        for pattern in self.EMBEDDED_CODE_PATTERNS:
            for match in re.findall(pattern, text_low):
                code = "".join(match) if isinstance(match, tuple) else match
                clean = self._clean_code(code)
                if len(clean) == 8 and clean.isdigit() and clean in self._valid_codes:
                    logger.debug("Extracted embedded CN code %s via pattern %r", clean, pattern)
                    return clean
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_cn_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["cn_code", "cn_notation", "level", "supplementary_unit", "description", "cn_year"]
    )


class CNDataUnavailableError(RuntimeError):
    """Raised when CN data cannot be retrieved from SPARQL or cache."""


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


def get_cn_retriever() -> CNCodeRetriever:
    """Return a process-wide CN retriever singleton."""
    if CNCodeRetriever._instance is None:
        with CNCodeRetriever._instance_lock:
            if CNCodeRetriever._instance is None:
                CNCodeRetriever._instance = CNCodeRetriever()
    return CNCodeRetriever._instance


def preload_cn_data() -> bool:
    """Pre-load CN data into the global cache. Returns True on success."""
    try:
        df = get_cn_retriever().load()
        return not df.empty
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to preload CN data: %s", exc)
        return False
