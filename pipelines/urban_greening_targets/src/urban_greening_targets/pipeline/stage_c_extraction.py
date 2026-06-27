"""
pipeline/stage_c_extraction.py — LLM Evidence Extraction

Uses OpenAI Structured Outputs (Responses API) to extract urban greening
targets from source documents.  Each source is processed individually to
maximise accuracy and allow per-source scoring.
"""

from __future__ import annotations

import json
from typing import List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, log
from pipeline.models import ExtractionResponse, ExtractedRecord, SourceDocument


# ── JSON Schema for Structured Outputs ────────────────────

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "source_title": {"type": "string"},
        "source_url": {"type": "string"},
        "publication_date": {"type": ["string", "null"]},
        "source_type": {"type": "string"},
        "figure_data_hints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Descriptions of data that appears to be in figures, images, charts, or infographics that could not be extracted as text.",
        },
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal_description_exact": {"type": "string"},
                    "target_value": {"type": ["string", "null"]},
                    "target_years": {"type": ["string", "null"]},
                    "metric_type": {"type": ["string", "null"]},
                    "baseline_value": {"type": ["string", "null"]},
                    "goal_type": {"type": ["string", "null"]},
                    "jurisdiction_level": {"type": ["string", "null"]},
                    "target_category": {"type": ["string", "null"]},
                    "relevance": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
                "required": [
                    "goal_description_exact",
                    "target_value",
                    "target_years",
                    "metric_type",
                    "baseline_value",
                    "goal_type",
                    "jurisdiction_level",
                    "target_category",
                    "relevance",
                    "notes",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "source_title",
        "source_url",
        "publication_date",
        "source_type",
        "figure_data_hints",
        "records",
    ],
    "additionalProperties": False,
}


# ── Prompt Builder ────────────────────────────────────────


def _build_extraction_prompt(
    city: str, state: str, source_title: str, source_url: str, text: str
) -> str:
    return f"""You are extracting urban greening targets from a single source document.

City: {city}
State: {state}
Source title: {source_title}
Source URL: {source_url}

Rules:
- Extract ONLY targets explicitly stated in the provided text.
- Quote EXACT wording from the source — do NOT paraphrase.
- If no numeric target is present, return an empty records list.
- Record multiple targets as separate entries.
- Distinguish citywide targets from neighborhood-specific targets.
- Distinguish mandatory/adopted vs aspirational/strategy vs draft.
- If the target is not city-adopted, mark it clearly.

STRICT RELEVANCE RULES — What IS an urban greening target:
- Tree canopy cover goals (e.g., "achieve 40% canopy cover by 2030")
- Tree planting goals (e.g., "plant 1 million trees by 2035")
- Green infrastructure goals involving VEGETATION or GREEN SPACE: expand green
  roofs, increase park acres, add green corridors, biodiversity plans, new open
  space (expressed in acres/square feet/area)
- Shade structure targets directly tied to tree planting programs

What is NOT an urban greening target — do NOT extract these:
- Greenhouse gas (GHG) or carbon emission reduction targets (e.g., "reduce emissions 45%")
- Stormwater VOLUME targets (e.g., "capture 36 million gallons") — UNLESS
  they are explicitly tied to green infrastructure AREA (e.g., "143 acres of
  green infrastructure to manage stormwater")
- Generic sustainability goals not tied to trees, canopy, or green space
- Energy efficiency or renewable energy targets
- Air quality targets not tied to vegetation
- Past/completed one-time events that are not forward-looking goals
  (e.g., "last year we planted 200 trees" with no future target)
- Stormwater percentages about volume of water (e.g., "increase stormwater capture
  by 10% annually") — unless expressed as AREA of green infrastructure

For goal_type use: Mandatory | Aspirational | Draft | Unknown
For jurisdiction_level use: city | neighborhood | district | metro | state
For target_category use: canopy_cover | tree_planting | green_infrastructure | equity_target
For source_type use: Primary | Secondary

RELEVANCE ASSESSMENT — For each record set the "relevance" field:
- "relevant" — directly about trees, canopy cover, tree planting, or green space area
- "marginal" — tangentially related (e.g., stormwater plan mentioning green
  infrastructure but target expressed as volume not area, or shade structures
  without tree counts)
- "off_topic" — not about urban greening (GHG, energy, stormwater volume,
  generic sustainability)

FIGURE/IMAGE DATA — If the source text contains references to figures, charts,
maps, or infographics that seem to contain numeric targets or data that could NOT
be extracted as text (e.g., "See Figure 3", "as shown in the chart below"),
describe what data might be in those figures in the "figure_data_hints" field.
This helps identify data that requires manual review.

CRITICAL — Fill EVERY field as completely as possible:

1. **target_years** — ALWAYS provide a value:
   - If the document states a deadline year, use it (e.g., "2030", "2025-2035").
   - If the document is a municipal code/ordinance with no deadline, use "ongoing".
   - If the document says "by 2040" or "over the next decade", compute the year.
   - If the document was published in year YYYY and says "over N years", set target_years = YYYY + N.
   - Search the ENTIRE document for any timeline, deadline, or year — even if mentioned far from the target.
   - As a last resort, if no year can be determined at all, use "not specified".
   - NEVER leave target_years as null or empty string.

2. **metric_type** — Be specific and descriptive. Examples:
   - "percentage" → for canopy cover %, impervious surface %
   - "count" → for number of trees
   - "CSO volume reduction" → for gallons/volume
   - "funding commitment" → for dollar amounts
   - "square feet" / "acres" → for area-based targets
   - "trees per year" → for annual planting rates
   - NEVER leave null — if the target has a value, it has a metric type.

3. **baseline_value** — Look carefully for any current/starting values:
   - "currently has 22% canopy" → baseline = "22%"
   - "up from 32%" → baseline = "32%"
   - If a document mentions both a goal and a current state, capture the current state.

4. **target_value** — Include units (e.g., "40%", "1 million trees", "$3.5 billion").

Source text (truncated to fit context):
{text}"""


# ── LLM Call ──────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
def _call_llm(prompt: str) -> dict:
    """Call OpenAI with structured output and return parsed JSON."""
    from config import OPENAI_BASE_URL  # noqa: PLC0415
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    try:
        # Try Responses API first (newer)
        response = client.responses.create(
            model=LLM_MODEL,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "greening_extract",
                    "schema": EXTRACTION_SCHEMA,
                    "strict": True,
                }
            },
            temperature=LLM_TEMPERATURE,
        )
        return json.loads(response.output_text)
    except AttributeError:
        # Fallback to Chat Completions API
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            max_completion_tokens=4000,
        )
        return json.loads(response.choices[0].message.content)


# ── Public API ────────────────────────────────────────────


def extract_from_source(
    city: str,
    state: str,
    doc: SourceDocument,
) -> ExtractionResponse:
    """
    Extract greening targets from a single source document using LLM.

    Returns ExtractionResponse with zero or more records.
    """
    if not doc.text.strip():
        log.debug(f"[Extraction] Skipping empty document: {doc.url[:60]}")
        return ExtractionResponse(
            source_title=doc.title,
            source_url=doc.url,
            source_type="Primary" if doc.is_official else "Secondary",
        )

    # Skip obvious 404 / error pages that slipped through Stage B
    title_lower = (doc.title or "").lower()
    if any(
        marker in title_lower
        for marker in (
            "page not found",
            "404 error",
            "404 |",
            "| 404",
            "page cannot be found",
            "error page",
        )
    ):
        log.info(f"[Extraction] Skipping error page: {doc.title[:80]}")
        return ExtractionResponse(
            source_title=doc.title,
            source_url=doc.url,
            source_type="Primary" if doc.is_official else "Secondary",
        )

    log.info(f"[Extraction] Processing: {doc.title or doc.url[:60]}")

    prompt = _build_extraction_prompt(
        city=city,
        state=state,
        source_title=doc.title,
        source_url=doc.url,
        text=doc.text,
    )

    try:
        data = _call_llm(prompt)
        resp = ExtractionResponse(**data)

        # Store figure_data_hints if present (for manual review)
        figure_hints = data.get("figure_data_hints", [])
        if figure_hints:
            log.info(
                f"[Extraction] {doc.url[:60]} — figure data hints: {figure_hints}"
            )
            # Attach hints to each record's notes for downstream processing
            for rec in resp.records:
                hint_note = "Figure/image data may contain additional targets: " + "; ".join(figure_hints[:3])
                rec.notes = (rec.notes + " | " + hint_note) if rec.notes else hint_note

        # Filter out off_topic records (keep relevant + marginal)
        original_count = len(resp.records)
        resp.records = [
            r for r in resp.records
            if (r.relevance or "relevant") != "off_topic"
        ]
        filtered_count = original_count - len(resp.records)
        if filtered_count:
            log.info(
                f"[Extraction] {doc.url[:60]} — filtered {filtered_count} off-topic record(s)"
            )

        log.info(
            f"[Extraction] {doc.url[:60]} → {len(resp.records)} record(s)"
        )
        return resp
    except Exception as e:
        log.error(f"[Extraction] LLM extraction failed for {doc.url[:60]}: {e}")
        return ExtractionResponse(
            source_title=doc.title,
            source_url=doc.url,
            source_type="Primary" if doc.is_official else "Secondary",
        )


def extract_all(
    city: str,
    state: str,
    documents: List[SourceDocument],
) -> List[ExtractionResponse]:
    """
    Run LLM extraction on all source documents for a city.

    Returns list of ExtractionResponse objects.
    """
    results: List[ExtractionResponse] = []
    for doc in documents:
        resp = extract_from_source(city, state, doc)
        results.append(resp)
    return results


# ── Gap-Fill : enrich records with missing fields ────────

GAP_FILL_SCHEMA = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "target_years": {"type": ["string", "null"]},
                    "metric_type": {"type": ["string", "null"]},
                    "baseline_value": {"type": ["string", "null"]},
                    "target_value": {"type": ["string", "null"]},
                },
                "required": ["index", "target_years", "metric_type",
                             "baseline_value", "target_value"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["patches"],
    "additionalProperties": False,
}


def gap_fill_records(
    city: str,
    state: str,
    records: List[dict],
) -> List[dict]:
    """
    Review extracted records that have missing target_years, metric_type,
    baseline_value, or target_value.  Uses a single LLM call to patch
    all gaps.  Returns *new* list of records with patches applied.

    ``records`` is a list of dicts with at least the keys from
    ExtractedRecord + source_title + source_url.
    """
    # Identify which records need gap-filling
    gaps = []
    for i, r in enumerate(records):
        missing = []
        if not r.get("target_years"):
            missing.append("target_years")
        if not r.get("metric_type"):
            missing.append("metric_type")
        if not r.get("baseline_value"):
            missing.append("baseline_value")
        if not r.get("target_value"):
            missing.append("target_value")
        if missing:
            gaps.append({"index": i, "missing": missing, "record": r})

    if not gaps:
        log.info(f"[GapFill] {city}, {state} — all fields complete, nothing to fill")
        return records

    log.info(
        f"[GapFill] {city}, {state} — {len(gaps)}/{len(records)} records need gap-fill"
    )

    # Build a concise prompt listing only the gapped records
    items_text = ""
    for g in gaps:
        r = g["record"]
        items_text += (
            f"\n  Record #{g['index']}:\n"
            f"    quote: \"{r.get('goal_description_exact', '')}\"\n"
            f"    source: {r.get('source_title', '')}\n"
            f"    current target_value: {r.get('target_value') or '(empty)'}\n"
            f"    current target_years: {r.get('target_years') or '(empty)'}\n"
            f"    current metric_type:  {r.get('metric_type') or '(empty)'}\n"
            f"    current baseline_value: {r.get('baseline_value') or '(empty)'}\n"
            f"    missing fields: {', '.join(g['missing'])}\n"
        )

    prompt = f"""You are reviewing urban greening target records for {city}, {state}.
Some records have missing fields. Your job is to infer the correct values
from the exact quote text and source context.

RULES:
- target_years: If the target comes from a municipal code or ordinance with no
  deadline, use "ongoing". If the source says "by 20XX", use that year.
  If the quote says "over the next N years" and you know the publication year,
  compute the target year.  If truly unknowable, use "not specified".
  NEVER leave as null or empty.
- metric_type: Always provide a descriptive type matching the target_value.
  Examples: "percentage", "count", "square feet", "acres", "CSO volume reduction",
  "funding commitment", "trees per year".  NEVER leave as null if target_value exists.
- baseline_value: Only fill if the quote or source mentions a starting/current value.
  If not mentioned, leave as null.
- target_value: Only fill if a numeric value can be extracted from the quote.
  Include units (e.g., "40%", "1 million trees").

Records needing gap-fill:
{items_text}

Return a JSON object with a "patches" array. Each patch has:
  index (int), target_years, metric_type, baseline_value, target_value
Only include records that had missing fields. Keep existing non-empty values unchanged.
"""

    try:
        from config import OPENAI_BASE_URL
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        try:
            response = client.responses.create(
                model=LLM_MODEL,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "gap_fill",
                        "schema": GAP_FILL_SCHEMA,
                        "strict": True,
                    }
                },
                temperature=0.0,
            )
            data = json.loads(response.output_text)
        except AttributeError:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_completion_tokens=4000,
            )
            data = json.loads(response.choices[0].message.content)

        # Apply patches
        patched = 0
        for patch in data.get("patches", []):
            idx = patch.get("index")
            if idx is None or idx < 0 or idx >= len(records):
                continue
            rec = records[idx]
            for field in ("target_years", "metric_type", "baseline_value", "target_value"):
                new_val = patch.get(field)
                if new_val and not rec.get(field):
                    rec[field] = new_val
                    patched += 1

        log.info(f"[GapFill] {city}, {state} — patched {patched} field(s)")
        return records

    except Exception as e:
        log.warning(f"[GapFill] LLM gap-fill failed for {city}, {state}: {e}")
        return records
