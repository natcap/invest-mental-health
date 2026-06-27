"""
CLI entry point for the Urban Greening Targets Pipeline.

Usage:
  PYTHONPATH=src python -m urban_greening_targets.cli run --all
  PYTHONPATH=src python -m urban_greening_targets.cli run --all --resume
  PYTHONPATH=src python -m urban_greening_targets.cli run --cities "New York,NY"
  PYTHONPATH=src python -m urban_greening_targets.cli export
  PYTHONPATH=src python -m urban_greening_targets.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from cities import City, get_all_cities, get_city
from config import MAX_RETRIES, OUTPUT_DIR, log
from pipeline.db import PipelineDB
from pipeline.models import CityResult, ExtractedRecord, StateFallback
from pipeline.stage_a_discovery import discover_sources
from pipeline.stage_b_collection import collect_sources
from pipeline.stage_c_extraction import extract_all, gap_fill_records
from pipeline.stage_d_scoring import score_all
from pipeline.stage_e_consolidation import consolidate
from pipeline.stage_f_qa import add_state_fallback, run_qa
from pipeline.stage_g_export import export_all


# ── Orchestrator ──────────────────────────────────────────


def process_city(
    city_obj: City,
    db: PipelineDB,
    use_llm: bool = True,
    stop_after_stage: Optional[str] = None,
) -> CityResult:
    """
    Run the full pipeline (Stages A → F) for a single city.

    Stages:
      A — Source Discovery
      B — Page/PDF Collection
      C — LLM Evidence Extraction
      D — Evidence Scoring
      E — City-Level Consolidation
      F — QA / Verification / Fallback
    """
    city = city_obj.name
    state = city_obj.state
    domains = city_obj.domains

    log.info(f"{'='*60}")
    log.info(f"Processing: {city}, {state}")
    log.info(f"{'='*60}")

    # ── Stage A: Source Discovery ─────────────────────────
    db.update_city_stage(city, state, "discovery", "in_progress")
    try:
        urls = discover_sources(city, state, domains, use_llm=use_llm, phase="official")
        official_urls = urls["official"]
        secondary_urls = urls["secondary"]

        # Save discovered URLs
        db.save_discovered_urls(city, state, official_urls, "official")
        db.save_discovered_urls(city, state, secondary_urls, "secondary")

        db.update_city_stage(city, state, "discovery", "done")
    except Exception as e:
        db.update_city_stage(city, state, "discovery", "failed", str(e))
        log.error(f"[{city}] Discovery failed: {e}")
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes=f"Discovery failed: {e}")

    if stop_after_stage == "discovery":
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes="Stopped after discovery")

    # ── Stage B: Page/PDF Collection ──────────────────────
    db.update_city_stage(city, state, "collection", "in_progress")
    try:
        all_urls = official_urls + secondary_urls
        documents = collect_sources(
            city, state, all_urls, db, official_domains=domains
        )
        db.update_city_stage(city, state, "collection", "done")
    except Exception as e:
        db.update_city_stage(city, state, "collection", "failed", str(e))
        log.error(f"[{city}] Collection failed: {e}")
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes=f"Collection failed: {e}")

    if stop_after_stage == "collection":
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes="Stopped after collection")

    if not documents:
        log.warning(f"[{city}] No documents collected — marking as no_target_found")
        result = CityResult(
            city=city, state=state,
            final_status="no_target_found",
            official_sources_checked=official_urls[:10],
            secondary_sources_checked=secondary_urls[:10],
            notes="No fetchable documents found from discovered URLs.",
        )
        db.save_city_result(city, state, result.model_dump())
        db.update_city_stage(city, state, "complete", "complete")
        return result

    # ── Stage C: LLM Evidence Extraction ──────────────────
    db.update_city_stage(city, state, "extraction", "in_progress")
    try:
        extractions = extract_all(city, state, documents)
        db.update_city_stage(city, state, "extraction", "done")
    except Exception as e:
        db.update_city_stage(city, state, "extraction", "failed", str(e))
        log.error(f"[{city}] Extraction failed: {e}")
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes=f"Extraction failed: {e}")

    if stop_after_stage == "extraction":
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes="Stopped after extraction")

    # ── Stage C½: Gap-Fill missing fields ─────────────────
    for ext_resp in extractions:
        if ext_resp.records:
            rec_dicts = [r.model_dump() for r in ext_resp.records]
            patched = gap_fill_records(city, state, rec_dicts)
            ext_resp.records = [ExtractedRecord(**d) for d in patched]

    # ── Stage D: Scoring ──────────────────────────────────
    db.update_city_stage(city, state, "scoring", "in_progress")
    scored_records = score_all(extractions)
    db.update_city_stage(city, state, "scoring", "done")

    # Save individual records to DB
    for rec in scored_records:
        db.save_extracted_record(
            city, state,
            rec.source_url,
            rec.model_dump(),
            rec.evidence_strength,
            rec.verification_status,
        )

    if stop_after_stage == "scoring":
        return CityResult(city=city, state=state, final_status="unprocessed",
                          notes="Stopped after scoring")

    # ── Stage E: Consolidation ────────────────────────────
    db.update_city_stage(city, state, "consolidation", "in_progress")
    result = consolidate(city, state, extractions, scored_records)
    db.update_city_stage(city, state, "consolidation", "done")

    if stop_after_stage == "consolidation":
        db.save_city_result(city, state, result.model_dump())
        return result

    # ── Stage F: QA / Verification ────────────────────────
    db.update_city_stage(city, state, "qa", "in_progress")
    result = run_qa(result, db, official_domains=domains, use_llm=use_llm)

    # ── Fallback for no-target cities ─────────────────────
    if result.final_status == "no_target_found" and use_llm:
        log.info(f"[{city}] No target found — running secondary search fallback")
        try:
            secondary_results = discover_sources(
                city, state, domains, use_llm=use_llm, phase="secondary"
            )
            if secondary_results["secondary"]:
                extra_docs = collect_sources(
                    city, state,
                    secondary_results["secondary"][:10],
                    db, official_domains=domains,
                )
                if extra_docs:
                    extra_extractions = extract_all(city, state, extra_docs)
                    # Gap-fill missing fields
                    for ext_resp in extra_extractions:
                        if ext_resp.records:
                            rec_dicts = [r.model_dump() for r in ext_resp.records]
                            patched = gap_fill_records(city, state, rec_dicts)
                            ext_resp.records = [ExtractedRecord(**d) for d in patched]
                    extra_scored = score_all(extra_extractions)
                    if extra_scored:
                        # Merge into existing result
                        result.target_records.extend(extra_scored)
                        result.secondary_sources_checked.extend(
                            secondary_results["secondary"][:10]
                        )
                        result.final_status = "secondary_target_found"
                        for rec in extra_scored:
                            rec.city = city
                            rec.state = state
        except Exception as e:
            log.warning(f"[{city}] Secondary fallback failed: {e}")

    # ── State-level fallback ──────────────────────────────
    if result.final_status == "no_target_found" and use_llm:
        log.info(f"[{city}] Running state-level fallback for {state}")
        try:
            state_results = discover_sources(
                city, state, domains, use_llm=use_llm, phase="state"
            )
            state_urls = state_results.get("official", []) + state_results.get("secondary", [])
            if state_urls:
                state_docs = collect_sources(
                    city, state, state_urls[:5], db, official_domains=domains
                )
                if state_docs:
                    state_extractions = extract_all(city, state, state_docs)
                    # Gap-fill missing fields
                    for ext_resp in state_extractions:
                        if ext_resp.records:
                            rec_dicts = [r.model_dump() for r in ext_resp.records]
                            patched = gap_fill_records(city, state, rec_dicts)
                            ext_resp.records = [ExtractedRecord(**d) for d in patched]
                    state_scored = score_all(state_extractions)
                    if state_scored:
                        # Add state records as full rows so they appear in export
                        for rec in state_scored:
                            rec.city = city
                            rec.state = state
                            rec.jurisdiction_level = "state"
                            rec.is_state_fallback = True
                            rec.notes = (
                                (rec.notes + " | " if rec.notes else "")
                                + f"State-level fallback target ({state})."
                            )
                            result.target_records.append(rec)
                        result.secondary_sources_checked.extend(state_urls[:5])
                        # Also keep the lightweight StateFallback metadata
                        best = state_scored[0]
                        fallback = StateFallback(
                            goal_description_exact=best.goal_description_exact,
                            source_title=best.source_title,
                            source_url=best.source_url,
                        )
                        result = add_state_fallback(result, fallback)
        except Exception as e:
            log.warning(f"[{city}] State fallback failed: {e}")

    # ── Unofficial source fallback ────────────────────────
    if result.final_status in ("no_target_found", "state_fallback_used") and use_llm:
        log.info(f"[{city}] No official/state target — running unofficial source search")
        try:
            unofficial_results = discover_sources(
                city, state, domains, use_llm=use_llm, phase="unofficial"
            )
            unofficial_urls = (
                unofficial_results.get("official", [])
                + unofficial_results.get("secondary", [])
            )
            if unofficial_urls:
                unofficial_docs = collect_sources(
                    city, state, unofficial_urls[:15], db, official_domains=domains
                )
                if unofficial_docs:
                    unofficial_extractions = extract_all(city, state, unofficial_docs)
                    # Gap-fill missing fields
                    for ext_resp in unofficial_extractions:
                        if ext_resp.records:
                            rec_dicts = [r.model_dump() for r in ext_resp.records]
                            patched = gap_fill_records(city, state, rec_dicts)
                            ext_resp.records = [ExtractedRecord(**d) for d in patched]
                    unofficial_scored = score_all(unofficial_extractions)
                    if unofficial_scored:
                        for rec in unofficial_scored:
                            rec.city = city
                            rec.state = state
                            rec.source_reliability = "unofficial"
                            rec.notes = (
                                (rec.notes + " | " if rec.notes else "")
                                + "Unofficial source — not from city government."
                            )
                            result.target_records.append(rec)
                        result.secondary_sources_checked.extend(unofficial_urls[:15])
                        result.final_status = "unofficial_target_found"
                        log.info(
                            f"[{city}] Found {len(unofficial_scored)} unofficial target(s)"
                        )
        except Exception as e:
            log.warning(f"[{city}] Unofficial fallback failed: {e}")

    db.update_city_stage(city, state, "qa", "done")

    # ── Save final result ─────────────────────────────────
    db.save_city_result(city, state, result.model_dump())
    db.update_city_stage(city, state, "complete", "complete")

    log.info(
        f"[{city}] Complete — status={result.final_status}, "
        f"records={len(result.target_records)}"
    )
    return result


# ── CLI Commands ──────────────────────────────────────────


def cmd_run(args):
    """Run the pipeline for selected cities."""
    db = PipelineDB()

    # Build city list
    if args.all:
        cities = get_all_cities()
    elif args.cities:
        cities = []
        for spec in args.cities:
            parts = spec.rsplit(",", 1)
            if len(parts) == 2:
                name, st = parts[0].strip(), parts[1].strip()
                c = get_city(name, st)
                if c:
                    cities.append(c)
                else:
                    log.warning(f"City not found: {spec}")
            else:
                log.warning(f"Invalid city spec '{spec}' — use 'Name,ST' format")
    else:
        log.error("Specify --all or --cities")
        return

    # Initialise progress rows
    for c in cities:
        db.init_city(c.name, c.state)

    # Resume support: skip completed cities
    if args.resume:
        completed = {(r["city"], r["state"]) for r in db.get_complete_cities()}
        before = len(cities)
        cities = [c for c in cities if (c.name, c.state) not in completed]
        skipped = before - len(cities)
        if skipped:
            log.info(f"Resuming — skipping {skipped} already-complete cities")

    # Run ID for this batch
    run_id = str(uuid.uuid4())[:8]
    db.start_run(run_id, len(cities), {"llm": not args.no_llm})

    log.info(f"Run {run_id}: processing {len(cities)} cities")

    results: List[CityResult] = []
    for city_obj in tqdm(cities, desc="Cities", unit="city"):
        try:
            result = process_city(
                city_obj, db,
                use_llm=not args.no_llm,
                stop_after_stage=args.stage,
            )
            results.append(result)
        except Exception as e:
            log.error(f"[{city_obj.name}] Unhandled error: {e}")
            db.update_city_stage(
                city_obj.name, city_obj.state, "error", "failed", str(e)
            )

    db.finish_run(run_id)

    # Auto-export if not stopping at a specific stage
    if not args.stage:
        # Gather all results (including previously completed)
        all_result_dicts = db.get_all_results()
        all_results = [CityResult(**d) for d in all_result_dicts]
        if all_results:
            export_all(all_results, OUTPUT_DIR)

    # Print summary
    summary = db.summary()
    log.info(
        f"\nRun {run_id} complete: "
        f"{summary['complete']}/{summary['total']} done, "
        f"{summary['failed']} failed, {summary['pending']} pending"
    )
    db.close()


def cmd_export(args):
    """Export results to files."""
    db = PipelineDB()
    all_dicts = db.get_all_results()
    if not all_dicts:
        log.error("No results to export. Run the pipeline first.")
        db.close()
        return

    results = [CityResult(**d) for d in all_dicts]
    outdir = Path(args.outdir) if args.outdir else OUTPUT_DIR
    paths = export_all(results, outdir)
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")
    db.close()


def cmd_status(args):
    """Show pipeline progress."""
    db = PipelineDB()
    progress = db.get_all_progress()
    summary = db.summary()

    print(f"\nPipeline Status")
    print(f"  Total cities:  {summary['total']}")
    print(f"  Complete:      {summary['complete']}")
    print(f"  Failed:        {summary['failed']}")
    print(f"  Pending:       {summary['pending']}")

    if args.verbose:
        print(f"\n{'City':<25} {'State':<6} {'Stage':<15} {'Status':<12} {'Attempts'}")
        print("-" * 75)
        for p in progress:
            print(
                f"{p['city']:<25} {p['state']:<6} {p['stage']:<15} "
                f"{p['status']:<12} {p['attempts']}"
            )

    db.close()


# ── Argument Parsing ──────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Urban Greening Targets Pipeline — "
        "Collect tree canopy & greening goals for US cities"
    )
    sub = parser.add_subparsers(dest="command", help="Pipeline commands")

    # run
    run_p = sub.add_parser("run", help="Run the pipeline")
    run_p.add_argument("--all", action="store_true", help="Process all 100 cities")
    run_p.add_argument(
        "--cities", nargs="*", metavar="CITY,ST",
        help="Specific cities, e.g. 'New York,NY' 'Chicago,IL'"
    )
    run_p.add_argument("--resume", action="store_true", help="Skip already-complete cities")
    run_p.add_argument("--no-llm", action="store_true", help="Disable LLM-assisted discovery")
    run_p.add_argument(
        "--stage", choices=["discovery", "collection", "extraction", "scoring", "consolidation"],
        help="Stop after this stage (for debugging)"
    )
    run_p.set_defaults(func=cmd_run)

    # export
    exp_p = sub.add_parser("export", help="Export results to files")
    exp_p.add_argument("--outdir", help="Output directory (default: data/output/)")
    exp_p.set_defaults(func=cmd_export)

    # status
    stat_p = sub.add_parser("status", help="Show pipeline progress")
    stat_p.add_argument("-v", "--verbose", action="store_true", help="Show per-city detail")
    stat_p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
