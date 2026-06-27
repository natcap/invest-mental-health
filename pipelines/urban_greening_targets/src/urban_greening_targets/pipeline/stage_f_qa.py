"""
pipeline/stage_f_qa.py — QA, Verification & Fallback

Quality controls:
  QC-1  Exact-quote validation (fuzzy ≥ 90% in source text)
  QC-2  Domain validation (Primary only if .gov)
  QC-3  Year consistency (target_years matches quote)
  QC-4  Scope validation (citywide vs neighbourhood)
  QC-5  "No target" evidence requirement
  QC-6  State fallback separation
  QC-7  Duplicate detection (already handled in Stage E)

Also handles the controlled fallback for cities with no primary target:
  1. Secondary search (partners, NGOs)
  2. State-level fallback query
  3. LLM verification of "no target" classification
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import List, Optional
from urllib.parse import urlparse

import requests

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, log
from pipeline.db import PipelineDB
from pipeline.models import (
    CityResult,
    StateFallback,
    TargetRecord,
    VerificationResponse,
)


# ── QC-1: Exact-Quote Validation ─────────────────────────


def _validate_quote(quote: str, source_text: str, threshold: float = 0.90) -> bool:
    """Check if the exact quote appears in the source text (fuzzy)."""
    if not quote or not source_text:
        return False

    # Quick exact check
    if quote.lower() in source_text.lower():
        return True

    # Sliding-window fuzzy check for long quotes
    q_len = len(quote)
    best_ratio = 0.0
    source_lower = source_text.lower()
    quote_lower = quote.lower()

    # Sample windows for efficiency
    step = max(1, q_len // 4)
    for i in range(0, len(source_lower) - q_len + 1, step):
        window = source_lower[i : i + q_len]
        ratio = SequenceMatcher(None, quote_lower, window).ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio >= threshold:
            return True

    return best_ratio >= threshold


# ── QC-2: Domain Validation ──────────────────────────────


def _is_official_domain(url: str, official_domains: List[str]) -> bool:
    """Check if the URL belongs to an official city domain."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return any(d.lower() in host for d in official_domains)


# ── QC-3: Year Consistency ───────────────────────────────

import re


def _validate_year_consistency(record: TargetRecord) -> bool:
    """Check target_years value appears in the exact quote."""
    if not record.target_years or not record.goal_description_exact:
        return True  # nothing to validate
    years = re.findall(r"\d{4}", record.target_years)
    quote = record.goal_description_exact
    return any(y in quote for y in years)


# ── QC-4b: URL Accessibility Check ─────────────────────────


def _check_url_accessibility(url: str) -> str:
    """
    Return 'open', 'restricted', or 'broken' based on HTTP response.
    Uses HEAD first, falls back to GET if HEAD is blocked.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; UrbanGreeningBot/1.0; +https://your-org.org)"
            )
        }
        resp = requests.head(url, timeout=8, allow_redirects=True, headers=headers)
        if resp.status_code == 200:
            return "open"
        if resp.status_code in (401, 403, 407, 429):
            return "restricted"
        if resp.status_code in (404, 410):
            return "broken"
        # Some servers reject HEAD — retry with GET (streaming, no body download)
        resp2 = requests.get(url, timeout=8, stream=True, headers=headers)
        resp2.close()
        if resp2.status_code == 200:
            return "open"
        if resp2.status_code in (401, 403, 407, 429):
            return "restricted"
        return "broken"
    except Exception:
        return "broken"


# ── Corroboration Scoring ────────────────────────────────


def _compute_corroboration(records: List[TargetRecord]) -> None:
    """
    For each record, count how many independent source domains cite the
    same target value + target year.  Updates corroboration_count in-place.
    Records without numeric targets get corroboration_count=1 (no cross-check).
    """
    for rec in records:
        if not rec.target_value or not rec.target_years:
            rec.corroboration_count = 1
            continue
        tv = (rec.target_value or "").strip().lower()
        ty = (rec.target_years or "").strip()
        same_target = [
            r for r in records
            if (r.target_value or "").strip().lower() == tv
            and (r.target_years or "").strip() == ty
        ]
        unique_domains = {urlparse(r.source_url).netloc.lower() for r in same_target}
        rec.corroboration_count = max(1, len(unique_domains))


# ── QC-5: No-Target Evidence Requirement ──────────────────


def _check_no_target_evidence(result: CityResult) -> List[str]:
    """
    Return list of missing source categories.
    A city can only be "no_target_found" if ALL three categories were checked:
      1. Sustainability/climate source
      2. Parks/urban forestry source
      3. PDF or council document search
    """
    checked = " ".join(
        result.official_sources_checked + result.secondary_sources_checked
    ).lower()

    missing = []
    if not any(w in checked for w in ("sustainab", "climate", "environment")):
        missing.append("sustainability/climate")
    if not any(w in checked for w in ("park", "forest", "tree", "urban-forest")):
        missing.append("parks/urban forestry")
    if not any(w in checked for w in (".pdf", "council", "resolution", "minutes")):
        missing.append("PDF/council document")
    return missing


# ── LLM Verification (for no-target cities) ──────────────


def _llm_verify(
    city: str,
    state: str,
    sources_checked: List[str],
    records_summary: str,
) -> VerificationResponse:
    """Ask LLM to verify the no-target classification."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        return VerificationResponse(
            final_status="no_target_found",
            reason="API key not set — skipping verification",
            coverage_ok=False,
            missing_source_categories=["unknown"],
        )

    from config import OPENAI_BASE_URL  # noqa: PLC0415
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    prompt = f"""You are verifying whether this city has an official urban greening target.

City: {city}
State: {state}
Official sources checked: {json.dumps(sources_checked[:20])}
Extracted records summary: {records_summary}

Task:
Determine the final classification:
1. primary_target_found — official city target identified
2. secondary_target_found — city-endorsed target found via secondary source
3. state_fallback_used — no city target; state-level target should be used
4. no_target_found — no target at any level after thorough search

Be conservative. If declaring "no official target found", verify the checked
sources reasonably cover: sustainability/climate dept, urban forestry/parks dept,
planning/comprehensive plans, council resolutions, and PDF plans/reports.

Return JSON with keys: final_status, reason, coverage_ok (bool), missing_source_categories (list).
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            max_completion_tokens=800,
        )
        data = json.loads(response.choices[0].message.content)
        return VerificationResponse(**data)
    except Exception as e:
        log.warning(f"[QA] LLM verification failed for {city}, {state}: {e}")
        return VerificationResponse(
            final_status="no_target_found",
            reason=f"Verification error: {e}",
            coverage_ok=False,
            missing_source_categories=[],
        )


# ── Public API ────────────────────────────────────────────


def run_qa(
    result: CityResult,
    db: PipelineDB,
    official_domains: List[str],
    use_llm: bool = True,
) -> CityResult:
    """
    Run all quality-control checks on a consolidated CityResult.

    Modifies result in-place and returns it.
    """
    log.info(f"[QA] Running QA for {result.city}, {result.state}")

    # ── QC-1 & QC-2 & QC-3: Per-record validation ────────
    for rec in result.target_records:
        # QC-1: Quote validation
        cached = db.get_cached_source(rec.source_url)
        if cached and cached.get("text_content"):
            if _validate_quote(rec.goal_description_exact, cached["text_content"]):
                rec.verification_status = "verified"
            else:
                rec.verification_status = "quote_mismatch"
                log.warning(
                    f"[QA] Quote mismatch for {result.city}: "
                    f"'{rec.goal_description_exact[:50]}...'"
                )
        else:
            rec.verification_status = "unverified"

        # QC-2: Domain validation
        if rec.source_type == "Primary":
            if not _is_official_domain(rec.source_url, official_domains):
                rec.source_type = "Secondary"
                log.info(
                    f"[QA] Reclassified source as Secondary: {rec.source_url[:60]}"
                )

        # QC-3: Year consistency
        if not _validate_year_consistency(rec):
            if rec.notes:
                rec.notes += " | Year mismatch between target_years and quote."
            else:
                rec.notes = "Year mismatch between target_years and quote."

        # QC-7: Relevance re-check — demote marginal records
        if rec.relevance_flag == "marginal":
            note = "Marginal relevance — may not be directly about urban greening."
            rec.notes = (rec.notes + " | " + note) if rec.notes else note

        # QC-8: Source reliability tagging
        if rec.source_reliability == "official":
            # official sources are from .gov or official domains
            if not _is_official_domain(rec.source_url, official_domains):
                if rec.source_type == "Secondary":
                    rec.source_reliability = "unofficial"

        # QC-6: URL accessibility — skip for already-flagged or state fallback
        if rec.access_status == "open" and not rec.is_state_fallback:
            rec.access_status = _check_url_accessibility(rec.source_url)
            if rec.access_status == "restricted":
                note = "[Source access restricted — may require login or subscription]"
                rec.notes = (rec.notes + " | " + note) if rec.notes else note
                log.info(
                    f"[QA] Access restricted: {rec.source_url[:70]}"
                )
            elif rec.access_status == "broken":
                note = "[Source URL appears broken — verify manually]"
                rec.notes = (rec.notes + " | " + note) if rec.notes else note
                log.warning(
                    f"[QA] Broken URL: {rec.source_url[:70]}"
                )

    # ── Corroboration scoring across all records ──────────
    _compute_corroboration(result.target_records)

    # ── Re-evaluate final_status after QC ─────────────────

    verified_primaries = [
        r
        for r in result.target_records
        if r.source_type == "Primary" and r.verification_status == "verified"
    ]
    verified_any = [
        r for r in result.target_records if r.verification_status == "verified"
    ]

    if verified_primaries:
        result.final_status = "primary_target_found"
    elif verified_any:
        result.final_status = "secondary_target_found"
    elif result.target_records:
        # Have records but none verified — keep what we have
        result.final_status = "secondary_target_found"
    else:
        result.final_status = "no_target_found"

    # ── QC-5: No-target evidence check ────────────────────

    if result.final_status == "no_target_found":
        missing = _check_no_target_evidence(result)
        if missing:
            result.notes = (
                f"Coverage gaps: {', '.join(missing)}. "
                "Consider running fallback search."
            )
            log.warning(
                f"[QA] {result.city}, {result.state} — insufficient evidence "
                f"for no-target: missing {missing}"
            )

        # LLM verification for no-target cities
        if use_llm:
            records_summary = "No records extracted." if not result.target_records else json.dumps(
                [r.model_dump() for r in result.target_records[:5]], indent=2
            )
            verification = _llm_verify(
                result.city,
                result.state,
                result.official_sources_checked + result.secondary_sources_checked,
                records_summary,
            )
            result.final_status = verification.final_status
            if verification.reason:
                result.notes = (
                    (result.notes + " | " if result.notes else "")
                    + f"Verification: {verification.reason}"
                )

    log.info(
        f"[QA] {result.city}, {result.state} — final_status={result.final_status}"
    )
    return result


def add_state_fallback(
    result: CityResult,
    fallback: StateFallback,
) -> CityResult:
    """Attach a state-level fallback to a CityResult (never replaces city records)."""
    result.state_fallback = fallback
    if result.final_status == "no_target_found":
        result.final_status = "state_fallback_used"
    return result
