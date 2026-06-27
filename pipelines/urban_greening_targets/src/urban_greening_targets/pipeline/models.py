"""
pipeline/models.py — Pydantic models for the Urban Greening Targets Pipeline.

These enforce strict JSON schemas via OpenAI Structured Outputs and serve as
the canonical data representation throughout the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────


class GoalType(str, Enum):
    MANDATORY = "Mandatory"
    ASPIRATIONAL = "Aspirational"
    DRAFT = "Draft"
    UNKNOWN = "Unknown"


class SourceType(str, Enum):
    PRIMARY = "Primary"
    SECONDARY = "Secondary"


class JurisdictionLevel(str, Enum):
    CITY = "city"
    NEIGHBORHOOD = "neighborhood"
    DISTRICT = "district"
    METRO = "metro"
    STATE = "state"


class TargetCategory(str, Enum):
    CANOPY_COVER = "canopy_cover"
    TREE_PLANTING = "tree_planting"
    GREEN_INFRASTRUCTURE = "green_infrastructure"
    EQUITY_TARGET = "equity_target"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    QUOTE_MISMATCH = "quote_mismatch"
    URL_INVALID = "url_invalid"


class FinalStatus(str, Enum):
    PRIMARY_TARGET_FOUND = "primary_target_found"
    SECONDARY_TARGET_FOUND = "secondary_target_found"
    UNOFFICIAL_TARGET_FOUND = "unofficial_target_found"
    STATE_FALLBACK_USED = "state_fallback_used"
    NO_TARGET_FOUND = "no_target_found"
    UNPROCESSED = "unprocessed"


# ── LLM Extraction Response Models ───────────────────────
# These match the JSON‑schema passed to OpenAI Structured Outputs.


class ExtractedRecord(BaseModel):
    """A single target extracted from one source document."""

    goal_description_exact: str = Field(
        ..., description="Exact quote from source — no paraphrasing."
    )
    target_value: Optional[str] = Field(
        None, description="Numeric target value, e.g. '30', '90000'."
    )
    target_years: Optional[str] = Field(
        None, description="Deadline year(s), e.g. '2030', '2025-2035'."
    )
    metric_type: Optional[str] = Field(
        None,
        description="E.g. 'Tree canopy cover (%)', 'Number of trees', 'Acres of green space'.",
    )
    baseline_value: Optional[str] = Field(
        None, description="Starting value if stated, e.g. '22% (2017)'."
    )
    goal_type: Optional[str] = Field(
        None, description="Mandatory | Aspirational | Draft | Unknown"
    )
    jurisdiction_level: Optional[str] = Field(
        None, description="city | neighborhood | district | metro | state"
    )
    target_category: Optional[str] = Field(
        None,
        description="canopy_cover | tree_planting | green_infrastructure | equity_target",
    )
    relevance: Optional[str] = Field(
        None,
        description="relevant | marginal | off_topic — whether the target is directly about urban greening",
    )
    notes: Optional[str] = Field(
        None, description="Equity focus, context, additional details."
    )


class ExtractionResponse(BaseModel):
    """Full LLM response for one source document."""

    source_title: str
    source_url: str
    publication_date: Optional[str] = Field(
        None, description="YYYY-MM format or null."
    )
    source_type: str = Field(
        default="Primary", description="Primary | Secondary"
    )
    records: List[ExtractedRecord] = Field(default_factory=list)


# ── Discovery Response ───────────────────────────────────


class DiscoveryResponse(BaseModel):
    """LLM response listing candidate URLs for a city."""

    official_candidate_urls: List[str] = Field(default_factory=list)
    secondary_candidate_urls: List[str] = Field(default_factory=list)
    notes: str = ""


# ── Verification Response ────────────────────────────────


class VerificationResponse(BaseModel):
    """LLM response for no-target verification."""

    final_status: str
    reason: str
    coverage_ok: bool
    missing_source_categories: List[str] = Field(default_factory=list)


# ── Scored Target Record (pipeline-internal) ─────────────


class TargetRecord(BaseModel):
    """A scored, validated target record for a single city."""

    city: str
    state: str
    goal_description_exact: str
    target_value: Optional[str] = None
    target_years: Optional[str] = None
    metric_type: Optional[str] = None
    baseline_value: Optional[str] = None
    goal_type: str = "Unknown"
    source_title: str = ""
    source_url: str = ""
    source_type: str = "Primary"
    publication_date: Optional[str] = None
    notes: Optional[str] = None
    jurisdiction_level: str = "city"
    target_category: str = "canopy_cover"
    evidence_strength: float = 0.0
    verification_status: str = "unverified"
    preferred: bool = False
    # Access / availability of source URL
    access_status: str = "open"  # open | restricted | broken
    # How many independent source domains corroborate this specific target
    corroboration_count: int = 1
    # True when this record comes from state-level fallback (no city target found)
    is_state_fallback: bool = False
    # Source reliability: official | unofficial | unknown
    source_reliability: str = "official"
    # Relevance flag: relevant | marginal | off_topic
    relevance_flag: str = "relevant"
    # True when data may exist in figures/images that text extraction missed
    figure_data_flag: bool = False


# ── State Fallback ────────────────────────────────────────


class StateFallback(BaseModel):
    """State-level target used when city has no target."""

    goal_description_exact: str = ""
    source_title: str = ""
    source_url: str = ""
    source_type: str = "Secondary - State"


# ── Per-City Result (final output) ───────────────────────


class CityResult(BaseModel):
    """Consolidated result for one city, including all targets found."""

    city: str
    state: str
    target_records: List[TargetRecord] = Field(default_factory=list)
    final_status: str = "unprocessed"
    official_sources_checked: List[str] = Field(default_factory=list)
    secondary_sources_checked: List[str] = Field(default_factory=list)
    state_fallback: Optional[StateFallback] = None
    notes: str = ""


# ── Source Document (internal) ────────────────────────────


class SourceDocument(BaseModel):
    """Fetched text content from a URL (HTML or PDF)."""

    url: str
    title: str = ""
    content_type: str = "html"  # html | pdf
    text: str = ""
    fetch_timestamp: str = ""
    http_status: int = 0
    is_official: bool = False
