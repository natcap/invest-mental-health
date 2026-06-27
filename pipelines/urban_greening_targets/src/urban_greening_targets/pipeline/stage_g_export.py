"""
pipeline/stage_g_export.py — Output Export

Generates:
  • urban_greening_targets.json  — Full structured data
  • urban_greening_targets.md    — Markdown table
  • urban_greening_targets.csv   — CSV for spreadsheets
  • summary_statistics.md        — Coverage stats
  • sources_checked.json         — All URLs checked per city
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import List

from config import OUTPUT_DIR, log
from pipeline.models import CityResult


# ── JSON Export ───────────────────────────────────────────


def export_json(results: List[CityResult], outdir: Path = OUTPUT_DIR) -> Path:
    """Export full structured data as JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "urban_greening_targets.json"
    data = [r.model_dump() for r in results]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"[Export] JSON → {path}")
    return path


# ── Markdown Table Export ─────────────────────────────────

_MD_COLUMNS = [
    "City",
    "Goal Description (Exact Quote)",
    "Target Value",
    "Target Year(s)",
    "Metric Type",
    "Baseline Value",
    "Goal Type",
    "Source Title",
    "Source URL",
    "Source Type",
    "Date",
    "Jurisdiction",
    "Target Category",
    "Verification",
    "Corroboration",
    "Strength",
    "Access",
    "Source Reliability",
    "Relevance",
    "Notes",
]


def _escape_md(text: str) -> str:
    """Escape pipe characters for Markdown tables."""
    return (text or "").replace("|", "\\|").replace("\n", " ")


def export_markdown(results: List[CityResult], outdir: Path = OUTPUT_DIR) -> Path:
    """Export results as a Markdown table."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "urban_greening_targets.md"

    header = "| " + " | ".join(_MD_COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * len(_MD_COLUMNS)) + "|"
    rows = [header, sep]

    for cr in results:
        if cr.target_records:
            for tr in cr.target_records:
                row = (
                    f"| {_escape_md(tr.city)}, {_escape_md(tr.state)} "
                    f"| {_escape_md(tr.goal_description_exact)} "
                    f"| {_escape_md(tr.target_value or '—')} "
                    f"| {_escape_md(tr.target_years or '—')} "
                    f"| {_escape_md(tr.metric_type or '—')} "
                    f"| {_escape_md(tr.baseline_value or '—')} "
                    f"| {_escape_md(tr.goal_type or '—')} "
                    f"| {_escape_md(tr.source_title)} "
                    f"| {_escape_md(tr.source_url)} "
                    f"| {_escape_md(tr.source_type)} "
                    f"| {_escape_md(tr.publication_date or '—')} "
                    f"| {_escape_md(tr.jurisdiction_level)} "
                    f"| {_escape_md(tr.target_category or '—')} "
                    f"| {_escape_md(tr.verification_status)} "
                    f"| {tr.corroboration_count} "
                    f"| {tr.evidence_strength} "
                    f"| {_escape_md(tr.access_status)} "
                    f"| {_escape_md(tr.source_reliability)} "
                    f"| {_escape_md(tr.relevance_flag)} "
                    f"| {_escape_md(tr.notes or '')} |"
                )
                rows.append(row)
        else:
            checked = "; ".join(cr.official_sources_checked[:3])
            fallback_note = ""
            if cr.state_fallback:
                fallback_note = (
                    f" State fallback: {cr.state_fallback.source_title} — "
                    f"{_escape_md(cr.state_fallback.goal_description_exact[:100])}"
                )
            row = (
                f"| {_escape_md(cr.city)}, {_escape_md(cr.state)} "
                f"| No official target found "
                f"| — | — | — | — | — "
                f"| Sources checked | {_escape_md(checked)} "
                f"| — | — | city | — | verified_no_target | 1 | 0.0 | open "
                f"| — | — "
                f"| {_escape_md(cr.notes)}{fallback_note} |"
            )
            rows.append(row)

    path.write_text("\n".join(rows), encoding="utf-8")
    log.info(f"[Export] Markdown → {path}")
    return path


# ── CSV Export ────────────────────────────────────────────


def export_csv(results: List[CityResult], outdir: Path = OUTPUT_DIR) -> Path:
    """Export results as CSV."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "urban_greening_targets.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_MD_COLUMNS)

        for cr in results:
            if cr.target_records:
                for tr in cr.target_records:
                    writer.writerow(
                        [
                            f"{tr.city}, {tr.state}",
                            tr.goal_description_exact,
                            tr.target_value or "",
                            tr.target_years or "",
                            tr.metric_type or "",
                            tr.baseline_value or "",
                            tr.goal_type or "",
                            tr.source_title,
                            tr.source_url,
                            tr.source_type,
                            tr.publication_date or "",
                            tr.jurisdiction_level,
                            tr.target_category or "",
                            tr.verification_status,
                            tr.corroboration_count,
                            tr.evidence_strength,
                            tr.access_status,
                            tr.source_reliability,
                            tr.relevance_flag,
                            tr.notes or "",
                        ]
                    )
            else:
                checked = "; ".join(cr.official_sources_checked[:3])
                writer.writerow(
                    [
                        f"{cr.city}, {cr.state}",
                        "No official target found",
                        "", "", "", "", "",
                        "Sources checked",
                        checked,
                        "", "", "city", "",
                        "verified_no_target",
                        1,
                        0.0,
                        "open",
                        "",
                        "",
                        cr.notes,
                    ]
                )

    log.info(f"[Export] CSV → {path}")
    return path


# ── Sources-Checked Export ────────────────────────────────


def export_sources(results: List[CityResult], outdir: Path = OUTPUT_DIR) -> Path:
    """Export all checked URLs per city."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "sources_checked.json"
    data = {}
    for cr in results:
        key = f"{cr.city}, {cr.state}"
        data[key] = {
            "official": cr.official_sources_checked,
            "secondary": cr.secondary_sources_checked,
            "final_status": cr.final_status,
        }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"[Export] Sources → {path}")
    return path


# ── Summary Statistics ────────────────────────────────────


def export_summary(results: List[CityResult], outdir: Path = OUTPUT_DIR) -> Path:
    """Generate summary statistics in Markdown."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "summary_statistics.md"

    total = len(results)
    status_counts = Counter(r.final_status for r in results)
    state_counts = Counter(r.state for r in results)

    # Cities with targets
    cities_with = [r for r in results if r.target_records]
    cities_without = [r for r in results if not r.target_records]

    # Average evidence strength
    all_scores = [
        rec.evidence_strength
        for r in results
        for rec in r.target_records
    ]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # States with most targets
    state_target_counts = Counter()
    for r in results:
        if r.target_records:
            state_target_counts[r.state] += 1

    lines = [
        "# Urban Greening Targets — Summary Statistics\n",
        f"**Total cities processed:** {total}\n",
        "## Final Status Breakdown\n",
        "| Status | Count | % |",
        "|--------|------:|---:|",
    ]
    for status, count in status_counts.most_common():
        pct = count / total * 100 if total else 0
        lines.append(f"| {status} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        f"**Cities with at least one target:** {len(cities_with)}",
        f"**Cities without any target:** {len(cities_without)}",
        f"**Total target records:** {len(all_scores)}",
        f"**Average evidence strength:** {avg_score:.2f}",
        "",
        "## By State\n",
        "| State | Cities | With Targets |",
        "|-------|-------:|-------------:|",
    ])
    for st in sorted(state_counts.keys()):
        lines.append(
            f"| {st} | {state_counts[st]} | {state_target_counts.get(st, 0)} |"
        )

    if cities_without:
        lines.extend([
            "",
            "## Cities Without Targets\n",
        ])
        for r in sorted(cities_without, key=lambda x: x.city):
            lines.append(f"- {r.city}, {r.state} — {r.final_status}")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"[Export] Summary → {path}")
    return path


# ── Public API ────────────────────────────────────────────


def export_all(
    results: List[CityResult], outdir: Path = OUTPUT_DIR
) -> dict:
    """Run all exports and return paths."""
    paths = {
        "json": str(export_json(results, outdir)),
        "markdown": str(export_markdown(results, outdir)),
        "csv": str(export_csv(results, outdir)),
        "sources": str(export_sources(results, outdir)),
        "summary": str(export_summary(results, outdir)),
    }
    log.info(f"[Export] All exports complete → {outdir}")
    return paths
