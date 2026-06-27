"""
pipeline/stage_e_consolidation.py — City-Level Consolidation

Merges all scored records for a single city into a CityResult:
  1. De-duplicates records describing the same target.
  2. Marks the strongest source per target as preferred.
  3. Determines preliminary final_status.
  4. Separates official vs secondary sources checked.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List

from config import log
from pipeline.models import CityResult, ExtractionResponse, TargetRecord


# ── Duplicate Detection ───────────────────────────────────


def _similarity(a: str, b: str) -> float:
    """Fuzzy similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _are_duplicates(rec_a: TargetRecord, rec_b: TargetRecord) -> bool:
    """
    Two records are duplicates if they share very similar quotes AND
    matching target value + target year.
    """
    quote_sim = _similarity(
        rec_a.goal_description_exact, rec_b.goal_description_exact
    )
    if quote_sim >= 0.85:
        return True

    # Also match if same target_value + target_years + metric_type
    if (
        rec_a.target_value
        and rec_a.target_value == rec_b.target_value
        and rec_a.target_years == rec_b.target_years
        and _similarity(rec_a.metric_type or "", rec_b.metric_type or "") > 0.7
    ):
        return True

    return False


# ── Dedup & Mark Preferred ────────────────────────────────

# Priority order for source preference (higher = better)
_SOURCE_PRIORITY = {
    "ordinance": 5,
    "resolution": 5,
    "code": 5,
    "plan": 4,
    "dashboard": 4,
    "report": 3,
    "press release": 2,
    "press_release": 2,
}


def _source_priority(rec: TargetRecord) -> int:
    title = rec.source_title.lower()
    for keyword, priority in _SOURCE_PRIORITY.items():
        if keyword in title:
            return priority
    return 1


def _dedup_records(records: List[TargetRecord]) -> List[TargetRecord]:
    """
    Group duplicates together.  For each group keep all records but mark
    the one with highest (evidence_strength, source_priority) as preferred.
    """
    if not records:
        return []

    groups: List[List[TargetRecord]] = []
    used = set()

    for i, rec in enumerate(records):
        if i in used:
            continue
        group = [rec]
        used.add(i)
        for j in range(i + 1, len(records)):
            if j in used:
                continue
            if _are_duplicates(rec, records[j]):
                group.append(records[j])
                used.add(j)
        groups.append(group)

    deduped: List[TargetRecord] = []
    for group in groups:
        group.sort(
            key=lambda r: (r.evidence_strength, _source_priority(r)),
            reverse=True,
        )
        group[0].preferred = True
        deduped.extend(group)

    return deduped


# ── Public API ────────────────────────────────────────────


def consolidate(
    city: str,
    state: str,
    extractions: List[ExtractionResponse],
    scored_records: List[TargetRecord],
) -> CityResult:
    """
    Merge all extraction data into a single CityResult for a city.
    """
    # Collect source URLs
    official_sources: List[str] = []
    secondary_sources: List[str] = []
    for ext in extractions:
        url = ext.source_url
        if ext.source_type == "Primary":
            official_sources.append(url)
        else:
            secondary_sources.append(url)

    # Fill city/state on records
    for rec in scored_records:
        rec.city = city
        rec.state = state

    # Dedup
    deduped = _dedup_records(scored_records)

    # Determine preliminary status
    has_primary = any(
        r.source_type == "Primary" and r.preferred for r in deduped
    )
    has_any = len(deduped) > 0

    if has_primary:
        status = "primary_target_found"
    elif has_any:
        status = "secondary_target_found"
    else:
        status = "no_target_found"

    result = CityResult(
        city=city,
        state=state,
        target_records=deduped,
        final_status=status,
        official_sources_checked=list(dict.fromkeys(official_sources)),
        secondary_sources_checked=list(dict.fromkeys(secondary_sources)),
    )

    log.info(
        f"[Consolidation] {city}, {state} — {len(deduped)} record(s), "
        f"status={status}"
    )
    return result
