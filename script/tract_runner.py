import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import Resampling
from shapely.geometry import mapping
from datetime import datetime
import matplotlib

# Suppress noisy but harmless warnings from shapefile I/O and geopandas
warnings.filterwarnings("ignore", message=".*unary_union.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Column names longer than 10.*")
warnings.filterwarnings("ignore", message=".*Normalized/laundered field name.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import shutil

# Import everything needed from the main analysis module
from merged_analysis import (
    run_pd_analysis,
    fit_sigmoid_adaptive,
    sigmoid_function,
    reproject_raster,
    clip_raster,
    resample_raster_to_target,
    calculate_weighted_ndvi,
    calculate_weighted_treecover,
    reproject_shapefile,
)


# =============== Helper Functions ===============

def mean_continuous_raster(raster_path, band=1, extra_nodata_values=None):
    """Compute mean of a continuous raster, excluding nodata values."""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(band).astype("float32")
            nodata = src.nodata

            valid_mask = np.ones(data.shape, dtype=bool)
            if nodata is not None:
                valid_mask &= (data != nodata)
            if extra_nodata_values:
                for val in extra_nodata_values:
                    valid_mask &= (data != val)

            valid_data = data[valid_mask]
            if len(valid_data) == 0:
                return {"mean": np.nan, "n_valid_pixels": 0}

            return {"mean": float(np.mean(valid_data)), "n_valid_pixels": len(valid_data)}

    except Exception:
        return {"mean": np.nan, "n_valid_pixels": 0}


def fit_city_sigmoid(city, city_folder, tract_path, pop_path, ndvi_path, tree_path,
                     output_dir, target_crs="EPSG:5070"):
    """
    For a single city, compute population-weighted NDVI and tree cover per tract,
    then fit an adaptive sigmoid curve.

    This replicates the fitting portion of run_ndvi_tree_analysis() but is
    lightweight -- it only computes what is needed for the sigmoid fit and
    does not produce maps or extra CSV files.

    Returns:
        fit_result (dict): output of fit_sigmoid_adaptive(), containing:
                           x_curve, y_curve, steepness, midpoint, r_squared,
                           success, category, y_min, y_max
        aoi_adm2_clipped (GeoDataFrame): tracts with weighted_ndvi / weighted_treecover
        pop_dst_clip (str): path to the reprojected+clipped population raster
                            (reused by run_pd_analysis to avoid duplicate work)
        ndvi_resampled_path (str): path to the processed NDVI raster
    """
    # --- 1. Load and reproject tracts ---
    aoi = reproject_shapefile(tract_path, target_crs)
    aoi_geometry = [aoi.geometry.union_all()]
    aoi = aoi[aoi.area > 100]
    n_tracts = len(aoi)

    # --- 2. Population raster ---
    pop_dst_path = pop_path.replace("_setnull", "").replace(".tif", "_reprojected.tif")
    pop_dst_clip = pop_dst_path.replace(".tif", "_clipped.tif")
    reproject_raster(pop_path, target_crs, pop_dst_path)
    clip_raster(pop_dst_path, aoi_geometry, pop_dst_clip)

    with rasterio.open(pop_dst_clip) as pop_src:
        target_meta = {
            "crs": pop_src.crs,
            "transform": pop_src.transform,
            "width": pop_src.width,
            "height": pop_src.height
        }

    # --- 3. NDVI raster ---
    ndvi_dst_path = ndvi_path.replace(".tif", "_prj.tif")
    ndvi_dst_clip = ndvi_dst_path.replace(".tif", "_clipped.tif")
    ndvi_resampled_path = ndvi_dst_clip.replace(".tif", "_100m.tif")
    reproject_raster(ndvi_path, target_crs, ndvi_dst_path)
    clip_raster(ndvi_dst_path, aoi_geometry, ndvi_dst_clip)

    with rasterio.open(pop_dst_clip) as pop_src:
        ndvi_resampled = resample_raster_to_target(ndvi_dst_clip, target_meta, Resampling.bilinear)
        with rasterio.open(ndvi_resampled_path, 'w', **pop_src.meta) as dst:
            dst.write(ndvi_resampled)

    # NDVI scale check (divide by 100 if scaled)
    with rasterio.open(ndvi_resampled_path) as src:
        check_data = src.read(1)
        valid_check = check_data[np.isfinite(check_data)]
        if src.nodata is not None:
            valid_check = valid_check[valid_check != src.nodata]
        if len(valid_check) > 0 and (abs(valid_check.max()) > 10 or abs(valid_check.min()) > 10):
            check_data = check_data / 100.0
            meta = src.meta.copy()
            converted_path = ndvi_resampled_path.replace('.tif', '_std.tif')
            with rasterio.open(converted_path, 'w', **meta) as dst:
                dst.write(check_data, 1)
            ndvi_resampled_path = converted_path

    # --- 4. Population-weighted NDVI per tract ---
    weighted_ndvi_list = [
        calculate_weighted_ndvi(row.geometry, ndvi_resampled_path, pop_dst_clip)
        for _, row in aoi.iterrows()
    ]
    aoi["weighted_ndvi"] = weighted_ndvi_list

    # --- 5. Tree cover raster + population-weighted tree cover per tract ---
    tree_prj_path = tree_path.replace(".tif", "_prj.tif")
    tree_clip_path = tree_prj_path.replace(".tif", "_clipped.tif")
    reproject_raster(tree_path, target_crs, tree_prj_path)
    clip_raster(tree_prj_path, aoi_geometry, tree_clip_path)

    tree_resampled_path = tree_clip_path.replace('.tif', '_100m.tif')
    with rasterio.open(pop_dst_clip) as pop_src:
        tree_resampled = resample_raster_to_target(tree_clip_path, target_meta, Resampling.bilinear)
        with rasterio.open(tree_resampled_path, 'w', **pop_src.meta) as dst:
            dst.write(tree_resampled)

    weighted_tree_list = [
        calculate_weighted_treecover(row.geometry, tree_resampled_path, pop_dst_clip)
        for _, row in aoi.iterrows()
    ]
    aoi["weighted_treecover"] = weighted_tree_list

    # --- 6. Fit adaptive sigmoid ---
    x_data = aoi["weighted_treecover"].values
    y_data = aoi["weighted_ndvi"].values

    # Use the actual data range (plus margin) instead of a fixed (0, 100).
    # This prevents midpoint from being pushed to the boundary in low-cover
    # cities (e.g. desert areas with only 1-5% tree cover).
    valid_x = x_data[~np.isnan(x_data)]
    if len(valid_x) > 0:
        x_margin = max((np.nanmax(valid_x) - np.nanmin(valid_x)) * 0.2, 5.0)
        x_lo = max(0.0,   np.nanmin(valid_x) - x_margin)
        x_hi = min(100.0, np.nanmax(valid_x) + x_margin)
    else:
        x_lo, x_hi = 0.0, 100.0

    fit_result = fit_sigmoid_adaptive(
        x_data, y_data, n_tracts=n_tracts,
        extend_to_range=(x_lo, x_hi), verbose=False
    )

    return fit_result, aoi, pop_dst_clip, ndvi_resampled_path




def save_city_sigmoid_plot(fit_result, aoi_tracts, city, city_plot_dir):
    """
    Save the sigmoid fit scatter plot for a city.
    x = weighted_treecover per tract, y = weighted_ndvi per tract.
    """
    x_data = aoi_tracts["weighted_treecover"].values
    y_data = aoi_tracts["weighted_ndvi"].values
    valid  = ~(np.isnan(x_data) | np.isnan(y_data))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_data[valid], y_data[valid],
               alpha=0.5, s=40, color='steelblue', label='Tracts (pop-weighted)')
    ax.plot(fit_result['x_curve'], fit_result['y_curve'],
            'r-', linewidth=2,
            label=f"Sigmoid fit  R²={fit_result['r_squared']:.3f}")

    params_text = (
        f"steepness : {fit_result['steepness']:.5f}\n"
        f"midpoint  : {fit_result['midpoint']:.2f}\n"
        f"category  : {fit_result['category']}"
    )
    ax.text(0.02, 0.97, params_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel("Population-weighted Tree Cover (%)")
    ax.set_ylabel("Population-weighted NDVI")
    ax.set_title(f"City {city}  —  Adaptive Sigmoid Fit  (n={len(x_data[valid])} tracts)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(city_plot_dir, "sigmoid_fit.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def save_city_pd_map(city_results_df, aoi_tracts, city, mult, city_plot_dir):
    """
    Draw a choropleth of total_pd_cases per tract for one scenario and save it.
    city_results_df: subset of all_results for this city + this mult.
    """
    # Merge pd cases back onto the tract GeoDataFrame
    plot_gdf = aoi_tracts.copy()
    merge_df = city_results_df[["tract", "total_pd_cases", "cost_savings"]].copy()
    merge_df = merge_df.rename(columns={"tract": "GEOID"})
    plot_gdf["GEOID"] = plot_gdf["GEOID"].astype(str)
    merge_df["GEOID"] = merge_df["GEOID"].astype(str)
    plot_gdf = plot_gdf.merge(merge_df, on="GEOID", how="left")
    plot_gdf["total_pd_cases"] = plot_gdf["total_pd_cases"].fillna(0)

    total_cases = plot_gdf["total_pd_cases"].sum()

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_gdf.plot(
        column="total_pd_cases",
        cmap="YlOrRd",
        linewidth=0.5,
        edgecolor="grey",
        legend=True,
        legend_kwds={"label": "Preventable Depression Cases", "orientation": "vertical"},
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "No data / skipped"}
    )
    ax.set_title(
        f"City {city}  —  Preventable Depression Cases\n"
        f"+{mult}% Tree Cover  |  Total: {total_cases:,.0f} cases",
        fontsize=11
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    fig.tight_layout()

    out_path = os.path.join(city_plot_dir, f"pd_cases_{mult}pct.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path

# =============== Main Function ===============

def run_tract_level_analysis(base_dir, cost_excel, effect_excel, output_dir,
                             save_plots=False, cleanup_temp=True, batch_size=20,
                             start_city=None, multiply_percents=None):
    """
    For each city, fit the sigmoid curve from raw raster data, then run
    tract-level preventable depression case analysis for each tree cover
    increase scenario.

    No intermediate CSV is required -- everything is computed from scratch
    per city using the raw input rasters.

    Args:
        base_dir:           Root directory containing per-city sub-folders
        cost_excel:         Excel table with health economic costs
        effect_excel:       Excel table with health effect sizes
        output_dir:         Directory where all outputs are saved
        save_plots:         Whether to save figures (False saves memory)
        cleanup_temp:       Whether to delete per-tract temp directories immediately
        batch_size:         Tracts between forced garbage collection calls
        start_city:         City ID string to resume from (None = start from beginning)
        multiply_percents:  List of tree cover increase percentages to simulate
                            (default: [10, 20, 30, 40, 50])
    """
    if multiply_percents is None:
        multiply_percents = [10, 20, 30, 40, 50]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    failed_tracts = []
    city_fit_summary = []   # one row per city: sigmoid params + fit quality

    log_file = os.path.join(output_dir, f"processing_log_{timestamp}.txt")

    def log_print(msg, to_file=True):
        print(msg)
        if to_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

    # === Load cost value ===
    try:
        cost_df = pd.read_excel(cost_excel)
        cost_value = cost_df.loc[cost_df["region"] == "USA", "cost_value"].values[0]
        log_print(f"Health cost: ${cost_value}/case")
    except Exception as e:
        log_print(f"[WARNING] Could not read cost value, using default 11000: {e}")
        cost_value = 11000

    # === Build city list ===
    city_list = sorted([
        c for c in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, c))
    ])

    if start_city:
        if start_city in city_list:
            start_idx = city_list.index(start_city)
            city_list = city_list[start_idx:]
            log_print(f"Resuming from city {start_city} (skipping first {start_idx})")
        else:
            log_print(f"[WARNING] start_city '{start_city}' not found; processing all cities")

    log_print(f"Cities to process: {len(city_list)}")
    log_print(f"Tree cover scenarios: {multiply_percents}")
    log_print(f"Memory settings: save_plots={save_plots}, cleanup_temp={cleanup_temp}\n")

    for city_idx, city in enumerate(city_list, 1):
        log_print(f"\n{'=' * 60}")
        log_print(f"[{city_idx}/{len(city_list)}] City {city}")
        log_print(f"{'=' * 60}")

        city_folder = os.path.join(base_dir, city)

        # ==== File paths ====
        ndvi_path  = os.path.join(city_folder, f"NDVI_median_landsat_30m_2021_{city}_scaled100.tif")
        pop_path   = os.path.join(city_folder, f"ppp_{city}.tif")
        risk_path  = os.path.join(city_folder, f"depress_{city}.tif")
        tree_path  = os.path.join(city_folder, f"treecover_{city}.tif")
        tract_path = os.path.join(city_folder, f"city_{city}_tract.shp")

        missing_files = [f for f in [ndvi_path, pop_path, risk_path, tree_path, tract_path]
                         if not os.path.exists(f)]
        if missing_files:
            log_print(f"[SKIP] Missing {len(missing_files)} file(s):")
            for f in missing_files:
                log_print(f"   - {os.path.basename(f)}")
            continue

        # ================================================================
        # STEP 1: Fit sigmoid from raw raster data
        # ================================================================
        log_print("Step 1: Fitting sigmoid from raw raster data...")
        try:
            fit_result, aoi_tracts, pop_dst_clip, ndvi_resampled_path = fit_city_sigmoid(
                city=city,
                city_folder=city_folder,
                tract_path=tract_path,
                pop_path=pop_path,
                ndvi_path=ndvi_path,
                tree_path=tree_path,
                output_dir=city_folder,     # write intermediate rasters alongside source data
            )
        except Exception as e:
            log_print(f"[ERROR] Sigmoid fitting failed: {e}")
            continue

        x_curve    = fit_result['x_curve']
        y_curve    = fit_result['y_curve']
        steepness  = fit_result['steepness']
        midpoint   = fit_result['midpoint']
        r_squared  = fit_result['r_squared']
        n_tracts   = len(aoi_tracts)

        log_print(f"  Tracts: {n_tracts} | "
                  f"steepness={steepness:.4f}, midpoint={midpoint:.2f}, R²={r_squared:.4f} "
                  f"[{fit_result['category']}]")

        # Create per-city plot subfolder
        city_plot_dir = os.path.join(output_dir, city)
        os.makedirs(city_plot_dir, exist_ok=True)

        # Save sigmoid fit plot
        try:
            sig_plot_path = save_city_sigmoid_plot(fit_result, aoi_tracts, city, city_plot_dir)
            log_print(f"  Sigmoid plot saved: {os.path.basename(sig_plot_path)}")
        except Exception as e:
            log_print(f"  [WARNING] Could not save sigmoid plot: {e}")

        # Save per-city sigmoid summary
        city_fit_summary.append({
            "city":       city,
            "n_tracts":   n_tracts,
            "steepness":  steepness,
            "midpoint":   midpoint,
            "r_squared":  r_squared,
            "category":   fit_result['category'],
            "success":    fit_result['success'],
        })

        # ================================================================
        # STEP 2: Compute per-tract tree cover means
        # ================================================================
        log_print("Step 2: Computing mean tree cover per tract...")
        records = []
        valid_tract_indices = []

        for idx, row in aoi_tracts.iterrows():
            geom  = row.geometry
            geoid = row.get("GEOID", str(idx))

            try:
                with rasterio.open(tree_path) as src:
                    out_image, out_transform = mask(
                        src, [mapping(geom)], crop=True, all_touched=True
                    )
                    if out_image.size == 0:
                        continue

                    out_image = out_image.astype("float32")
                    meta = src.meta.copy()
                    meta.update({
                        "height": out_image.shape[1],
                        "width":  out_image.shape[2],
                        "transform": out_transform
                    })

                    tmp_path = os.path.join(output_dir, f"_tmp_{city}_{idx}.tif")
                    with rasterio.open(tmp_path, "w", **meta) as dst:
                        dst.write(out_image)

                result = mean_continuous_raster(tmp_path, extra_nodata_values=[254, 255, -9999])

                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            except Exception:
                continue

            if not np.isnan(result["mean"]) and result["n_valid_pixels"] >= 10:
                records.append({
                    "city":           city,
                    "GEOID":          geoid,
                    "treecover_mean": result["mean"],
                    "n_valid_pixels": result["n_valid_pixels"]
                })
                valid_tract_indices.append(idx)

            plt.close('all')
            gc.collect()

        if not records:
            log_print("[SKIP] No valid tract tree cover data computed")
            continue

        df_tree = pd.DataFrame(records)
        log_print(f"  Valid tracts with tree cover data: {len(df_tree)}/{n_tracts}")

        tracts_valid = aoi_tracts.loc[valid_tract_indices].copy()

        # ================================================================
        # STEP 3: Compute NDVI deltas for each scenario
        # ================================================================
        for mult in multiply_percents:
            df_tree[f"target_treecover_{mult}pct"] = np.minimum(
                df_tree["treecover_mean"] * (1 + mult / 100), 100
            )
            df_tree[f"ndvi_current_{mult}pct"] = np.interp(
                df_tree["treecover_mean"].values, x_curve, y_curve
            )
            df_tree[f"ndvi_target_{mult}pct"] = np.interp(
                df_tree[f"target_treecover_{mult}pct"].values, x_curve, y_curve
            )
            df_tree[f"ndvi_delta_{mult}pct"] = (
                df_tree[f"ndvi_target_{mult}pct"] - df_tree[f"ndvi_current_{mult}pct"]
            )

        df_tree.to_csv(
            os.path.join(output_dir, f"treecover_by_tract_{city}.csv"), index=False
        )

        # ================================================================
        # STEP 4: Run PD analysis per tract per scenario
        # ================================================================
        log_print("Step 3: Running tract-level PD analysis...")

        for mult in multiply_percents:
            log_print(f"\n  Scenario: +{mult}% tree cover")
            tract_success_count = 0

            for tract_row_idx, tract_row in df_tree.iterrows():
                geoid      = tract_row["GEOID"]
                current_tc = tract_row["treecover_mean"]
                target_tc  = tract_row[f"target_treecover_{mult}pct"]
                ndvi_goal  = tract_row[f"ndvi_target_{mult}pct"]
                ndvi_delta = tract_row[f"ndvi_delta_{mult}pct"]

                if (tract_row_idx + 1) % 10 == 0 or tract_row_idx == 0:
                    log_print(
                        f"    [{tract_row_idx + 1}/{len(df_tree)}] {geoid}: "
                        f"TC {current_tc:.1f}% -> {target_tc:.1f}%, "
                        f"NDVI delta={ndvi_delta:.4f}"
                    )

                # Skip negligible NDVI increases
                if ndvi_delta < 0.001:
                    failed_tracts.append({
                        'city': city, 'tract': geoid, 'mult': mult,
                        'reason': f'NDVI delta too small ({ndvi_delta:.5f})'
                    })
                    continue

                tract_temp_dir = os.path.join(
                    output_dir, f"tract_temp_{city}_{geoid}_{mult}"
                )
                os.makedirs(tract_temp_dir, exist_ok=True)

                try:
                    tract_gdf = tracts_valid[
                        tracts_valid["GEOID"].astype(str) == str(geoid)
                    ].copy()

                    if len(tract_gdf) == 0:
                        failed_tracts.append({
                            'city': city, 'tract': geoid, 'mult': mult,
                            'reason': 'tract GeoDataFrame is empty'
                        })
                        continue

                    # Use GeoPackage to avoid ESRI Shapefile 10-char column name limit
                    tract_shp_path = os.path.join(tract_temp_dir, f"tract_{geoid}.gpkg")
                    tract_gdf.to_file(tract_shp_path, driver="GPKG")

                    fig1, fig2, fig_hist, fig_cost_curve, total_pd_cases = run_pd_analysis(
                        aoi_adm1_path=tract_shp_path,
                        aoi_adm2_path=tract_shp_path,
                        pop_path=pop_path,
                        ndvi_path=ndvi_path,
                        tree_path=tree_path,
                        risk_path=risk_path,
                        excel_file=effect_excel,
                        output_dir=tract_temp_dir,
                        NE_goal=ndvi_goal,
                        aoi_adm2_clipped=tract_gdf,
                        x_curve=x_curve,
                        y_curve=y_curve,
                        cost_value=cost_value
                    )

                    all_results.append({
                        "city":             city,
                        "tract":            geoid,
                        "multiply_percent": mult,
                        "current_treecover": current_tc,
                        "target_treecover": target_tc,
                        "current_ndvi":     tract_row[f"ndvi_current_{mult}pct"],
                        "target_ndvi":      ndvi_goal,
                        "ndvi_delta":       ndvi_delta,
                        "steepness":        steepness,
                        "midpoint":         midpoint,
                        "r_squared":        r_squared,
                        "total_pd_cases":   total_pd_cases,
                        "cost_savings":     total_pd_cases * cost_value
                    })

                    tract_success_count += 1

                    for fig in [fig1, fig2, fig_hist, fig_cost_curve]:
                        try:
                            plt.close(fig)
                        except Exception:
                            pass
                    plt.close('all')

                except Exception as e:
                    error_msg = str(e)[:100]
                    if tract_row_idx < 3:
                        log_print(f"    [ERROR] {geoid}: {error_msg}")
                    failed_tracts.append({
                        'city': city, 'tract': geoid, 'mult': mult, 'reason': error_msg
                    })

                finally:
                    plt.close('all')

                    if cleanup_temp:
                        try:
                            if os.path.exists(tract_temp_dir):
                                shutil.rmtree(tract_temp_dir)
                        except Exception:
                            pass

                    if (tract_row_idx + 1) % batch_size == 0:
                        gc.collect(); gc.collect(); gc.collect()
                        try:
                            import psutil
                            mem_gb = psutil.Process().memory_info().rss / 1024 ** 3
                            log_print(f"    [Memory: {mem_gb:.2f} GB]", to_file=False)
                        except Exception:
                            pass

            log_print(f"  Completed +{mult}%: {tract_success_count}/{len(df_tree)} tracts OK")

            # Save city-level PD cases map for this scenario
            city_mult_results = [
                r for r in all_results
                if r["city"] == city and r["multiply_percent"] == mult
            ]
            if city_mult_results:
                try:
                    pd_map_path = save_city_pd_map(
                        pd.DataFrame(city_mult_results), aoi_tracts, city, mult, city_plot_dir
                    )
                    log_print(f"  PD map saved: {os.path.basename(pd_map_path)}")
                except Exception as e:
                    log_print(f"  [WARNING] Could not save PD map (+{mult}%): {e}")

            # Save intermediate results after each scenario
            if all_results:
                pd.DataFrame(all_results).to_csv(
                    os.path.join(output_dir, f"temp_results_{city}_{mult}pct.csv"),
                    index=False
                )

        gc.collect(); gc.collect(); gc.collect()

    # ==== Save final outputs ====

    # Per-city sigmoid fit summary
    if city_fit_summary:
        fit_summary_path = os.path.join(output_dir, f"city_sigmoid_summary_{timestamp}.csv")
        pd.DataFrame(city_fit_summary).to_csv(fit_summary_path, index=False)
        log_print(f"\nSigmoid fit summary saved to: {fit_summary_path}")

    # Main results
    if all_results:
        df_all = pd.DataFrame(all_results)
        summary_path = os.path.join(output_dir, f"tract_level_pd_summary_{timestamp}.csv")
        df_all.to_csv(summary_path, index=False)

        log_print(f"\n{'=' * 60}")
        log_print("All cities processed")
        log_print(f"{'=' * 60}")
        log_print(f"Successful tracts:       {len(df_all)}")
        log_print(f"Total preventable cases: {df_all['total_pd_cases'].sum():,.0f}")
        log_print(f"Total cost savings:      ${df_all['cost_savings'].sum() / 1e9:.2f}B")
        log_print(f"Results saved to:        {summary_path}")
    else:
        log_print("\n[WARNING] No successful results generated")

    if failed_tracts:
        failed_df = pd.DataFrame(failed_tracts)
        failed_path = os.path.join(output_dir, f"failed_tracts_{timestamp}.csv")
        failed_df.to_csv(failed_path, index=False)
        log_print(f"\nFailed tracts: {len(failed_tracts)} -- saved to: {failed_path}")

    log_print(f"\nLog: {log_file}")


# =============== Entry Point ===============

if __name__ == "__main__":
    base_dir     = r"S:\Shared drives\invest-health\City500\City_Folder_By_Num\City"
    cost_excel   = r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_cost_table.xlsx"
    effect_excel = r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_effect_size_table.xlsx"
    output_dir   = os.path.join(base_dir, "tract_level_run_output_10_50V5")

    # Resume from a specific city (None = start from the beginning)
    start_from_city = None   # e.g. "4805000"

    run_tract_level_analysis(
        base_dir=base_dir,
        cost_excel=cost_excel,
        effect_excel=effect_excel,
        output_dir=output_dir,
        save_plots=False,
        cleanup_temp=True,
        batch_size=10,
        start_city=start_from_city,
        multiply_percents=[10, 20, 30, 40, 50],
    )