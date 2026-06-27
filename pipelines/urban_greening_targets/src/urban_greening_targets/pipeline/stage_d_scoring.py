"""
pipeline/stage_d_scoring.py — Evidence Scoring

Scores each extracted record on a 0.0–1.0 scale based on source authority,
content quality, and goal specificity.  The formula rewards:
  +5  Primary source (official .gov)
  +4  Exact quote present
  +3  Numeric target value present
  +2  Target year present
  +3  Mandatory / ordinance / resolution
  -1  Press-release only
  -2  Draft plan
  -2  Secondary-source only (already accounted via +2 instead of +5)

Max possible raw score = 17 → normalised to 1.0.
"""

from __future__ import annotations

from typing import List

from config import log
from pipeline.models import ExtractionResponse, TargetRecord


def score_record(
    source_type: str,
    record: dict,
    source_title: str,
) -> float:
    """Calculate evidence strength score for one extracted record."""
    score = 0.0

    # Source authority
    if source_type == "Primary":
        score += 5
    else:
        score += 2

    # Content quality
    if record.get("goal_description_exact"):
        score += 4
    if record.get("target_value"):
        score += 3
    if record.get("target_years"):
        score += 2

    # Goal authority
    goal_type = (record.get("goal_type") or "").lower()
    title_lower = source_title.lower()
    if any(w in goal_type for w in ("mandatory",)) or any(
        w in title_lower for w in ("ordinance", "resolution", "code")
    ):
        score += 3
    if "draft" in goal_type:
        score -= 2
    if "press release" in title_lower:
        score -= 1

    # Relevance penalty
    relevance = (record.get("relevance") or "relevant").lower()
    if relevance == "marginal":
        score -= 5
    elif relevance == "off_topic":
        score -= 10

    # Normalise to 0–1
    max_possible = 17.0
    return round(min(max(score / max_possible, 0.0), 1.0), 2)


def score_extraction(
    extraction: ExtractionResponse,
) -> List[TargetRecord]:
    """
    Score all records from one ExtractionResponse and return TargetRecords.

    Each record gets an evidence_strength score, source metadata propagated,
    and city/state placeholders (filled later during consolidation).
    """
    scored: List[TargetRecord] = []
    for rec in extraction.records:
        rec_dict = rec.model_dump()
        s = score_record(
            source_type=extraction.source_type,
            record=rec_dict,
            source_title=extraction.source_title,
        )
        relevance = (rec.relevance or "relevant").lower()
        scored.append(
            TargetRecord(
                city="",  # filled during consolidation
                state="",
                goal_description_exact=rec.goal_description_exact,
                target_value=rec.target_value,
                target_years=rec.target_years,
                metric_type=rec.metric_type,
                baseline_value=rec.baseline_value,
                goal_type=rec.goal_type or "Unknown",
                source_title=extraction.source_title,
                source_url=extraction.source_url,
                source_type=extraction.source_type,
                publication_date=extraction.publication_date,
                notes=rec.notes,
                jurisdiction_level=rec.jurisdiction_level or "city",
                target_category=rec.target_category or "canopy_cover",
                evidence_strength=s,
                verification_status="unverified",
                relevance_flag=relevance,
            )
        )
    return scored


def score_all(
    extractions: List[ExtractionResponse],
) -> List[TargetRecord]:
    """Score all extractions and return a flat list of TargetRecords."""
    all_records: List[TargetRecord] = []
    for ext in extractions:
        scored = score_extraction(ext)
        all_records.extend(scored)
    # Sort by score descending
    all_records.sort(key=lambda r: r.evidence_strength, reverse=True)
    log.info(f"[Scoring] Scored {len(all_records)} record(s)")
    return all_records
