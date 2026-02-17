import os
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
import rasterio.mask
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterstats import zonal_stats

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.widgets import Slider, Button

from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.api as sm

import plotly.express as px
import plotly.io as pio


# ============================================================================
# ADAPTIVE SIGMOID STRATEGY
# ============================================================================

def adaptive_sigmoid_strategy(n_tracts):
    """
    Select adaptive sigmoid fitting parameters based on tract count.
    Fewer tracts -> higher smoothness (tighter steepness bounds).
    More tracts  -> lower smoothness (wider steepness bounds).
    All cases enforce steepness > 0 to guarantee monotonic increase.
    """
    if n_tracts < 14:
        return {'steepness_bounds': (0.001, 0.05), 'midpoint_bounds': (0, 100),
                'initial_k': 0.02, 'maxfev': 20000, 'note': 'very_small_very_smooth'}
    elif n_tracts < 16:
        return {'steepness_bounds': (0.001, 0.1), 'midpoint_bounds': (0, 100),
                'initial_k': 0.05, 'maxfev': 15000, 'note': 'small_smooth'}
    elif n_tracts < 23:
        return {'steepness_bounds': (0.001, 0.3), 'midpoint_bounds': (0, 100),
                'initial_k': 0.1, 'maxfev': 10000, 'note': 'medium_moderate'}
    elif n_tracts < 36:
        return {'steepness_bounds': (0.001, 0.6), 'midpoint_bounds': (0, 100),
                'initial_k': 0.1, 'maxfev': 10000, 'note': 'large_standard'}
    elif n_tracts < 81:
        return {'steepness_bounds': (0.001, 1.0), 'midpoint_bounds': (0, 100),
                'initial_k': 0.15, 'maxfev': 10000, 'note': 'very_large_flexible'}
    else:
        return {'steepness_bounds': (0.001, 2.0), 'midpoint_bounds': (0, 100),
                'initial_k': 0.2, 'maxfev': 10000, 'note': 'extremely_large_most_flexible'}


def sigmoid_function(x, steepness, midpoint):
    """Standard logistic function. Monotonically increasing when steepness > 0."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def build_sigmoid_curve(steepness, midpoint, x_range=(0, 100), n_points=200):
    """
    Reconstruct a sigmoid curve directly from known steepness/midpoint parameters,
    without re-fitting any data.

    Used in tract_runner.py: after loading fitted parameters from CSV,
    call this function to restore the curve and pass it to run_pd_analysis
    as the x_curve/y_curve arguments.

    Parameters:
        steepness (float): sigmoid slope parameter k (from 'steepness' column in fitting_results.csv)
        midpoint  (float): sigmoid midpoint x0 (from 'midpoint' column in fitting_results.csv)
        x_range   (tuple): x range of the curve, default (0, 100)
        n_points  (int):   number of sample points, default 200

    Returns:
        x_curve (np.ndarray), y_curve (np.ndarray)
    """
    x_curve = np.linspace(x_range[0], x_range[1], n_points)
    y_curve = sigmoid_function(x_curve, steepness, midpoint)
    return x_curve, y_curve


def fit_sigmoid_adaptive(x_data, y_data, n_tracts, extend_to_range=(0, 100),
                         verbose=True, custom_params=None):
    """
    Fit a sigmoid curve with adaptive smoothness based on tract count.

    Parameters:
        x_data, y_data:   Input data (weighted_treecover, weighted_ndvi)
        n_tracts:         Number of tracts (determines smoothness level)
        extend_to_range:  Range to extend the curve, default (0, 100)
        verbose:          Whether to print fitting info
        custom_params:    Custom parameter dict to override adaptive defaults

    Returns:
        dict containing:
            'x_curve':   Fitted curve x values (np.ndarray, 200 points)
            'y_curve':   Fitted curve y values (np.ndarray, 200 points)
            'steepness': Slope parameter k
            'midpoint':  Midpoint parameter x0
            'r_squared': Goodness of fit R-squared
            'success':   Whether fitting succeeded (bool)
            'category':  Adaptive category name (str)
            'y_min':     Original y minimum value
            'y_max':     Original y maximum value
    """
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    n_points = len(x_clean)

    adaptive_params = custom_params if custom_params is not None else adaptive_sigmoid_strategy(n_tracts)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"ADAPTIVE SIGMOID FITTING")
        print(f"Number of tracts: {n_tracts} | Data points: {n_points}")
        print(f"Category: {adaptive_params['note']}")
        print(f"Steepness bounds: {adaptive_params['steepness_bounds']}")
        print(f"{'=' * 60}")

    x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)

    # Insufficient data: return constant mean curve
    if n_points < 2:
        if verbose:
            print("ERROR: Not enough data points")
        return {
            'x_curve': x_extended,
            'y_curve': np.full(200, np.nanmean(y_data) if len(y_data) > 0 else 0.5),
            'steepness': np.nan, 'midpoint': np.nan,
            'r_squared': np.nan, 'success': False,
            'category': adaptive_params['note'], 'y_min': np.nan, 'y_max': np.nan
        }

    # Normalize y to [0, 1]
    y_min, y_max = np.min(y_clean), np.max(y_clean)
    y_range = y_max - y_min

    if y_range < 1e-6:
        if verbose:
            print("WARNING: No variation in y data")
        return {
            'x_curve': x_extended,
            'y_curve': np.full(200, y_min),
            'steepness': 0.0, 'midpoint': 50.0,
            'r_squared': 0.0, 'success': False,
            'category': adaptive_params['note'], 'y_min': y_min, 'y_max': y_max
        }

    y_normalized = (y_clean - y_min) / y_range

    try:
        x_mid = (np.min(x_clean) + np.max(x_clean)) / 2
        p0 = [adaptive_params['initial_k'], x_mid]
        bounds = (
            [adaptive_params['steepness_bounds'][0], adaptive_params['midpoint_bounds'][0]],
            [adaptive_params['steepness_bounds'][1], adaptive_params['midpoint_bounds'][1]]
        )

        popt, _ = curve_fit(sigmoid_function, x_clean, y_normalized,
                            p0=p0, bounds=bounds, maxfev=adaptive_params['maxfev'])
        steepness_fit, midpoint_fit = popt

        y_curve = sigmoid_function(x_extended, steepness_fit, midpoint_fit) * y_range + y_min

        # R-squared
        y_pred = sigmoid_function(x_clean, steepness_fit, midpoint_fit) * y_range + y_min
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if verbose:
            print(f"  Steepness: {steepness_fit:.6f} | Midpoint: {midpoint_fit:.2f} | R2: {r_squared:.4f}")

        return {
            'x_curve': x_extended, 'y_curve': y_curve,
            'steepness': steepness_fit, 'midpoint': midpoint_fit,
            'r_squared': r_squared, 'success': True,
            'category': adaptive_params['note'], 'y_min': y_min, 'y_max': y_max
        }

    except Exception as e:
        if verbose:
            print(f"  Sigmoid fitting failed: {e}, trying linear fallback...")

        try:
            slope, intercept, r_value, _, _ = stats.linregress(x_clean, y_normalized)
            if slope < 0:
                slope, intercept = 0.0, np.mean(y_normalized)
            y_curve = (slope * x_extended + intercept) * y_range + y_min
            r_squared_linear = r_value ** 2 if slope > 0 else 0

            if verbose:
                print(f"  Linear fallback: slope={slope:.6f}, R2={r_squared_linear:.4f}")

            return {
                'x_curve': x_extended, 'y_curve': y_curve,
                'steepness': slope, 'midpoint': 50.0,
                'r_squared': r_squared_linear, 'success': True,
                'category': f"{adaptive_params['note']}_linear_fallback",
                'y_min': y_min, 'y_max': y_max
            }

        except Exception as e2:
            if verbose:
                print(f"  Linear fallback also failed: {e2}")
            return {
                'x_curve': x_extended,
                'y_curve': np.full(200, np.mean(y_clean)),
                'steepness': np.nan, 'midpoint': np.nan,
                'r_squared': np.nan, 'success': False,
                'category': f"{adaptive_params['note']}_failed",
                'y_min': y_min, 'y_max': y_max
            }


# ============================================================================
# MAIN NDVI + TREE COVER ANALYSIS
# ============================================================================

def run_ndvi_tree_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path,
                           risk_path, excel_file, output_dir):
    """
    Main analysis function: compute population-weighted NDVI and tree cover,
    then fit an adaptive sigmoid curve.

    Returns:
        NE_goal (float):                NDVI target placeholder (0.3)
        ndvi_fig (Figure):              Population-weighted NDVI map
        tree_fig (Figure):              Tree cover map (zonal stats)
        slider_fig (Figure):            Sigmoid fit plot
        fit_result (dict):              Full dict from fit_sigmoid_adaptive, containing:
                                          'x_curve', 'y_curve',
                                          'steepness', 'midpoint', 'r_squared',
                                          'success', 'category', 'y_min', 'y_max'
        aoi_adm2_clipped (GeoDataFrame): Contains weighted_ndvi and weighted_treecover columns
        ndvi_resampled_path (str):      Path to the processed NDVI raster
    """
    target_crs = "EPSG:5070"

    # 1. Load AOI
    aoi_adm1 = reproject_shapefile(aoi_adm1_path, target_crs)
    aoi_adm1_geometry = [aoi_adm1.geometry.unary_union]

    aoi_adm2 = gpd.read_file(aoi_adm2_path)
    if aoi_adm2.crs != target_crs:
        aoi_adm2 = aoi_adm2.to_crs(target_crs)
    aoi_adm2_clipped = gpd.clip(aoi_adm2, aoi_adm1)
    aoi_adm2_clipped = aoi_adm2_clipped[aoi_adm2_clipped.area > 100]

    n_tracts = len(aoi_adm2_clipped)
    print(f"  Number of tracts: {n_tracts}")

    # 2. Process population raster
    pop_dst_path = pop_path.replace("_setnull", "").replace(".tif", "_reprojected.tif")
    pop_dst_clip = pop_dst_path.replace(".tif", "_clipped.tif")
    reproject_raster(pop_path, target_crs, pop_dst_path)
    clip_raster(pop_dst_path, aoi_adm1_geometry, pop_dst_clip)

    # 3. Process NDVI raster
    ndvi_dst_path = ndvi_path.replace(".tif", "_prj.tif")
    ndvi_dst_clip = ndvi_dst_path.replace(".tif", "_clipped.tif")
    ndvi_resampled_path = ndvi_dst_clip.replace(".tif", "_100m.tif")
    reproject_raster(ndvi_path, target_crs, ndvi_dst_path)
    clip_raster(ndvi_dst_path, aoi_adm1_geometry, ndvi_dst_clip)

    with rasterio.open(pop_dst_clip) as pop_src:
        target_meta = {
            "crs": pop_src.crs, "transform": pop_src.transform,
            "width": pop_src.width, "height": pop_src.height
        }
        ndvi_resampled = resample_raster_to_target(ndvi_dst_clip, target_meta, Resampling.bilinear)
        with rasterio.open(ndvi_resampled_path, 'w', **pop_src.meta) as dst:
            dst.write(ndvi_resampled)

    # NDVI range check (convert scaled values if needed)
    with rasterio.open(ndvi_resampled_path) as src:
        check_data = src.read(1)
        valid_check = check_data[np.isfinite(check_data)]
        if src.nodata is not None:
            valid_check = valid_check[valid_check != src.nodata]
        if len(valid_check) > 0 and (abs(valid_check.max()) > 10 or abs(valid_check.min()) > 10):
            print(f"  Converting NDVI from scaled range to standard (-1 to 1)...")
            check_data = check_data / 100.0
            meta = src.meta.copy()
            converted_path = ndvi_resampled_path.replace('.tif', '_std.tif')
            with rasterio.open(converted_path, 'w', **meta) as dst:
                dst.write(check_data, 1)
            ndvi_resampled_path = converted_path

    # 4. Compute population-weighted NDVI
    weighted_ndvi_list = [
        calculate_weighted_ndvi(row.geometry, ndvi_resampled_path, pop_dst_clip)
        for _, row in aoi_adm2_clipped.iterrows()
    ]
    aoi_adm2_clipped["weighted_ndvi"] = weighted_ndvi_list

    ndvi_fig, ax = plt.subplots(figsize=(5.4, 4.5))
    aoi_adm2_clipped.plot(column="weighted_ndvi", cmap="Greens", linewidth=0.8, ax=ax,
                          edgecolor="black", legend=True, legend_kwds={"orientation": "vertical"})
    ax.set_title("Population-Weighted NDVI by Census Tract")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ndvi_fig.tight_layout()

    # 5. Process tree cover raster
    raster_reproject_path = tree_path.replace(".tif", "_prj.tif")
    raster_clipped_path = raster_reproject_path.replace(".tif", "_clipped.tif")
    reproject_raster(tree_path, target_crs, raster_reproject_path)
    clip_raster(raster_reproject_path, aoi_adm1_geometry, raster_clipped_path)

    # 5a. Zonal stats tree cover (for map visualization)
    aoi_adm2_raw = gpd.read_file(aoi_adm2_path)
    if aoi_adm2_raw.crs != target_crs:
        aoi_adm2_raw = aoi_adm2_raw.to_crs(target_crs)
    aoi_adm2_raw_clipped = gpd.clip(aoi_adm2_raw, aoi_adm1)
    aoi_adm2_raw_clipped = aoi_adm2_raw_clipped[aoi_adm2_raw_clipped.area > 100]

    df_stats = compute_zonal_statistics(aoi_adm2_raw_clipped, raster_clipped_path)
    aoi_joined = aoi_adm2_raw_clipped.merge(df_stats, on="GEOID")

    tree_fig, ax = plt.subplots(figsize=(8, 4.5))
    aoi_joined.plot(column="cover_10", cmap="viridis_r", linewidth=0.8, ax=ax,
                    edgecolor="black", legend=True, legend_kwds={"orientation": "vertical"})
    ax.set_title("Tree cover (%) by census tract")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    tree_fig.tight_layout()

    # 5b. Population-weighted tree cover (for sigmoid fitting)
    print("\n  Calculating population-weighted Tree Cover...")
    tree_resampled_for_weighting = raster_clipped_path.replace('.tif', '_100m.tif')
    with rasterio.open(pop_dst_clip) as pop_src:
        tree_data_resampled = resample_raster_to_target(
            raster_clipped_path, target_meta, Resampling.bilinear
        )
        with rasterio.open(tree_resampled_for_weighting, 'w', **pop_src.meta) as dst:
            dst.write(tree_data_resampled)

    weighted_tree_list = [
        calculate_weighted_treecover(row.geometry, tree_resampled_for_weighting, pop_dst_clip)
        for _, row in aoi_adm2_clipped.iterrows()
    ]
    aoi_adm2_clipped["weighted_treecover"] = weighted_tree_list
    print(f"    Weighted Tree Cover: "
          f"{np.nanmin(weighted_tree_list):.1f}% - {np.nanmax(weighted_tree_list):.1f}% "
          f"(mean {np.nanmean(weighted_tree_list):.1f}%)")

    # 6. Save CSV outputs
    ndvi_csv_path = os.path.join(output_dir, "population_weighted_ndvi_by_adm2.csv")
    landcover_csv_path = os.path.join(output_dir, "landcover_percentages_adm2.csv")
    weighted_tc_csv_path = os.path.join(output_dir, "weighted_treecover_data.csv")

    aoi_adm2_clipped[["GEOID", "weighted_ndvi"]].to_csv(ndvi_csv_path, index=False)
    df_stats.to_csv(landcover_csv_path, index=False)
    aoi_adm2_clipped[["GEOID", "weighted_treecover", "weighted_ndvi"]].to_csv(
        weighted_tc_csv_path, index=False
    )

    # 7. Adaptive sigmoid fit (x = weighted_treecover, y = weighted_ndvi)
    x_data = aoi_adm2_clipped["weighted_treecover"].values
    y_data = aoi_adm2_clipped["weighted_ndvi"].values

    fit_result = fit_sigmoid_adaptive(x_data, y_data, n_tracts=n_tracts, extend_to_range=(0, 100))

    # 8. Plot fit results
    slider_fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_data, y_data, 'o', alpha=0.4, label='Original Data', color='skyblue')
    ax.plot(fit_result['x_curve'], fit_result['y_curve'], 'r-', linewidth=2,
            label=f"Adaptive Sigmoid (R2={fit_result['r_squared']:.3f})")
    ax.axvline(np.nanmin(x_data), color='gray', linestyle=':', alpha=0.7, label='Data Range')
    ax.axvline(np.nanmax(x_data), color='gray', linestyle=':', alpha=0.7)

    params_text = (f"Steepness: {fit_result['steepness']:.6f}\n"
                   f"Midpoint:  {fit_result['midpoint']:.2f}\n"
                   f"Category:  {fit_result['category']}")
    ax.text(0.02, 0.98, params_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), family='monospace')

    ax.set_xlabel("Weighted Tree Cover (%)")
    ax.set_ylabel("Weighted NDVI")
    ax.set_title(f"Adaptive Sigmoid Fit: Weighted Tree Cover vs NDVI (n={n_tracts} tracts)")
    ax.set_xlim(0, 100)
    ax.grid(True)
    ax.legend()
    slider_fig.tight_layout()

    NE_goal = 0.3  # placeholder

    return (NE_goal, ndvi_fig, tree_fig, slider_fig,
            fit_result,           # dict: x_curve/y_curve/steepness/midpoint/r_squared/...
            aoi_adm2_clipped,
            ndvi_resampled_path)


# ============================================================================
# PD ANALYSIS
# ============================================================================

def run_pd_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path,
                    excel_file, output_dir, NE_goal, aoi_adm2_clipped,
                    x_curve, y_curve, cost_value):
    """
    Compute preventable depression cases and economic cost savings.

    Parameters:
        x_curve, y_curve (np.ndarray): Sigmoid curve arrays for cost-curve visualization.
            - From run_ndvi_tree_analysis: pass fit_result['x_curve'] / fit_result['y_curve']
            - From tract_runner.py: reconstruct with build_sigmoid_curve(steepness, midpoint)

    Returns:
        fig1, fig2, fig_hist, fig_cost_curve, total_pd_cases
    """
    target_crs = "EPSG:5070"

    aoi_adm1 = gpd.read_file(aoi_adm1_path)
    if aoi_adm1.crs != target_crs:
        aoi_adm1 = aoi_adm1.to_crs(target_crs)

    pop_dst_path = pop_path.replace("_setnull", "").replace(".tif", "_reprojected.tif")
    pop_dst_clip = pop_dst_path.replace(".tif", "_clipped.tif")

    risk_dst_path = risk_path.replace(".tif", "_prj.tif")
    risk_dst_clip = risk_dst_path.replace(".tif", "_clipped.tif")
    risk_resampled_path = risk_dst_clip.replace(".tif", "_100m.tif")

    reproject_raster(risk_path, target_crs, risk_dst_path)
    clip_raster(risk_dst_path, [aoi_adm1.geometry.unary_union], risk_dst_clip)

    with rasterio.open(pop_dst_clip) as pop_src:
        target_meta = {
            "crs": pop_src.crs, "transform": pop_src.transform,
            "width": pop_src.width, "height": pop_src.height
        }
        risk_resampled = resample_raster_to_target(risk_dst_clip, target_meta, Resampling.bilinear)
        with rasterio.open(risk_resampled_path, 'w', **pop_src.meta) as dst:
            dst.write(risk_resampled)

    with rasterio.open(risk_resampled_path) as risk_src:
        baseline_risk_raster = risk_src.read(1)
        max_val = np.nanmax(baseline_risk_raster[baseline_risk_raster > 0])
        if max_val > 1.5:
            baseline_risk_raster = baseline_risk_raster / 100.0
        baseline_risk_raster = np.where(
            (baseline_risk_raster < 0) | (baseline_risk_raster > 1), 0.15, baseline_risk_raster
        )

    result = load_health_effects(
        excel_file=excel_file, health_indicator_i="depression",
        baseline_risk=0.15, NE_goal=NE_goal
    )

    ndvi_raster_path = ndvi_path.replace(".tif", "_prj_clipped_100m_std.tif")
    if not os.path.exists(ndvi_raster_path):
        ndvi_raster_path = ndvi_path.replace(".tif", "_prj_clipped_100m.tif")
        print(f"  Warning: Using non-converted NDVI file")

    with rasterio.open(ndvi_raster_path) as ndvi_src:
        NDVI_array = ndvi_src.read(1)

    with rasterio.open(pop_dst_clip) as pop_src:
        Pop_array = pop_src.read(1)

    results = calculate_pd_layer(
        ndvi_raster_path=ndvi_raster_path, pop_raster_path=pop_dst_clip,
        rr=result["risk_ratio"], NE_goal=NE_goal,
        baseline_risk_rate=baseline_risk_raster, output_dir=output_dir
    )

    PD_raster_path = os.path.join(output_dir, "PD_i.tif")
    with rasterio.open(PD_raster_path) as src:
        PD_masked, PD_transform = rasterio.mask.mask(src, aoi_adm1.geometry, crop=True, nodata=src.nodata)
        PD_meta = src.meta.copy()
        PD_meta.update({"transform": PD_transform,
                        "width": PD_masked.shape[2], "height": PD_masked.shape[1]})

    fig1 = plot_pd_map_v1(PD_raster_path, aoi_adm2_clipped, return_fig=True)

    aoi_adm1_outline = gpd.GeoDataFrame(geometry=[aoi_adm1.unary_union], crs=aoi_adm1.crs)
    fig2 = plot_pd_map_v3(PD_masked=PD_masked, PD_meta=PD_meta, aoi_gdf=aoi_adm1_outline,
                          figures_dir=output_dir, return_fig=True)

    with rasterio.open(PD_raster_path) as src:
        PD_data = src.read(1)
        meta = src.meta.copy()

    PD_data_clean = np.where(PD_data < 0, np.nan, PD_data)
    meta.update({"nodata": np.nan})

    cleaned_raster_path = os.path.join(output_dir, "Prev_case.tif")
    with rasterio.open(cleaned_raster_path, "w", **meta) as dst:
        dst.write(PD_data_clean, 1)

    PD_cost = PD_data_clean * (cost_value / 1000)
    cost_raster_path = os.path.join(output_dir, "Prev_cost_pixel.tif")
    with rasterio.open(cost_raster_path, "w", **meta) as dst:
        dst.write(PD_cost, 1)

    zonal_result = zonal_stats(aoi_adm2_clipped, cost_raster_path, stats="sum", nodata=np.nan)
    sum_values = [round(s["sum"], 2) if s["sum"] is not None else 0 for s in zonal_result]
    aoi_adm2_clipped["PD_i_sum_cost"] = sum_values

    vmin = aoi_adm2_clipped["PD_i_sum_cost"].min()
    vmax = aoi_adm2_clipped["PD_i_sum_cost"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.OrRd

    fig_hist, ax = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    ax.set_aspect('equal')
    aoi_adm2_clipped.plot(column="PD_i_sum_cost", cmap=cmap, norm=norm,
                          linewidth=0.8, ax=ax, edgecolor="black", legend=False)
    sm_obj = ScalarMappable(cmap=cmap, norm=norm)
    sm_obj.set_array([])
    cbar = plt.colorbar(sm_obj, ax=ax, fraction=0.046, pad=0.04)
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{val:.0f}" for val in tick_values])
    ax.set_xlim(aoi_adm2_clipped.total_bounds[[0, 2]])
    ax.set_ylim(aoi_adm2_clipped.total_bounds[[1, 3]])
    ax.set_title("Total Preventable Cost (1000 USD) by Tract", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")

    # Interpolate x_curve/y_curve to find tree cover value corresponding to NE_goal
    selected_cover = float(np.interp(NE_goal, y_curve, x_curve))
    fig_cost_curve = simulate_cost_vs_treecover(
        x_curve, y_curve, Pop_array, NDVI_array, baseline_risk_raster,
        result["risk_ratio"], PD_masked, selected_cover=selected_cover, cost_value=cost_value
    )

    total_pd_cases = np.nansum(PD_masked[0])

    zonal_cases = zonal_stats(aoi_adm2_clipped, PD_raster_path, stats="sum", nodata=np.nan)
    tract_cases = []
    for idx, (geom, stat) in enumerate(zip(aoi_adm2_clipped.geometry, zonal_cases)):
        cases_sum = stat["sum"] if stat["sum"] is not None else 0.0
        cases_sum = round(max(0, cases_sum), 2)
        tract_cases.append({
            'GEOID': aoi_adm2_clipped.iloc[idx]['GEOID'],
            'preventable_cases': cases_sum,
            'preventable_cost_usd': round(cases_sum * cost_value, 2)
        })

    df_tract_cases = pd.DataFrame(tract_cases)
    df_tract_cases.to_csv(os.path.join(output_dir, "preventable_cases_by_tract.csv"), index=False)

    return fig1, fig2, fig_hist, fig_cost_curve, total_pd_cases


# ============================================================================
# Helper Functions
# ============================================================================

def simulate_cost_vs_treecover(x_curve, y_curve, Pop_array, baseline_ndvi_raster,
                               baseline_risk_raster, risk_ratio, PD_masked=None,
                               selected_cover=None, cost_value=11000):
    """Simulate total preventable cost savings under different tree cover targets."""
    start = (min(x_curve) // 5) * 5
    end = ((max(x_curve) + 4) // 5) * 5 + 5
    tree_cover_range = list(np.arange(start, end + 1, 5))
    cost_savings = []
    highlight_cost = None

    for tree_cover in tree_cover_range:
        ndvi_goal = np.interp(tree_cover, x_curve, y_curve)
        delta_ndvi = ndvi_goal - baseline_ndvi_raster
        delta_positive = np.where(delta_ndvi > 0, delta_ndvi, 0)
        RR_i = np.exp(np.log(risk_ratio) * 10 * delta_positive)
        PD_i = np.where((1 - RR_i) * baseline_risk_raster * Pop_array > 0,
                        (1 - RR_i) * baseline_risk_raster * Pop_array, 0)
        total_cost_savings = np.nansum(PD_i) * (cost_value / 1000)
        cost_savings.append(total_cost_savings)
        if selected_cover is not None and abs(tree_cover - selected_cover) < 2.5:
            highlight_cost = total_cost_savings

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(tree_cover_range, cost_savings, marker='o', color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel("Tree Cover Target (%)", fontsize=10)
    ax.set_ylabel("Total Preventable Cost Savings (1000 USD)", fontsize=10)
    ax.set_title("Cost Savings by Tree Cover Target", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    if selected_cover is not None and highlight_cost is not None:
        ax.plot(selected_cover, highlight_cost, 'ro', markersize=10, label="Selected Target")
        ax.legend(fontsize=9)
    ax.set_xlim(min(tree_cover_range) - 2, max(tree_cover_range) + 2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def process_shapefile(shp_path, target_crs, aoi_boundary):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    gdf_clipped = gpd.clip(gdf, aoi_boundary)
    gdf_clipped = gdf_clipped[gdf_clipped.area > 100]
    gdf_clipped.reset_index(drop=True, inplace=True)
    return gdf_clipped


def compute_zonal_statistics(aoi_shapefile, raster_path):
    with rasterio.open(raster_path) as src:
        nodata_val = src.nodata
        scale = (src.scales[0] if src.scales else 1.0)
        offset = (src.offsets[0] if src.offsets else 0.0)
        data = src.read(1, masked=True).astype(float) * scale + offset

    use_conversion = False
    conversion_factor = 1.0
    if scale == 1.0 and offset == 0.0:
        if data.max() > 100 and data.max() <= 253:
            use_conversion = True
            conversion_factor = 100.0 / 255.0

    stats_result = zonal_stats(aoi_shapefile, raster_path,
                               stats=['mean', 'median', 'std', 'count'], nodata=nodata_val)
    percent_stats = []
    for s in stats_result:
        mean_val = s['mean']
        if mean_val is None:
            cover_percent = 0.0
        else:
            cover = mean_val * scale + offset
            if use_conversion:
                cover *= conversion_factor
            cover_percent = max(0.0, min(100.0, cover))
        percent_stats.append({
            'cover_10': cover_percent, 'cover_mean': cover_percent,
            'cover_raw': mean_val if mean_val is not None else 0.0,
            'cover_count': s['count'] if s['count'] is not None else 0
        })

    df = pd.DataFrame(percent_stats)
    df["GEOID"] = aoi_shapefile["GEOID"].values
    return df


def save_landcover_csv(dataframe, aoi_shapefile, output_path):
    if "GEOID" in aoi_shapefile.columns:
        dataframe.insert(0, "GEOID", aoi_shapefile["GEOID"])
    dataframe.to_csv(output_path, index=False)


def plot_landcover(dataframe, aoi_shapefile, column, title, return_fig=False):
    if column in dataframe.columns:
        aoi_shapefile = aoi_shapefile.copy()
        aoi_shapefile[column] = dataframe[column]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        aoi_shapefile.plot(column=column, cmap="viridis", legend=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        if return_fig:
            return fig
        else:
            plt.show()


def reproject_shapefile(shp_path, target_crs, dst_path=None):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    if dst_path:
        gdf.to_file(dst_path)
    return gdf


def reproject_raster(src_path, target_crs, dst_path, resampling_method=Resampling.nearest):
    with rasterio.open(src_path) as src:
        if src.crs == target_crs:
            with rasterio.open(dst_path, 'w', **src.meta) as dst:
                dst.write(src.read())
            return
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        dst_meta = src.meta.copy()
        dst_meta.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
        with rasterio.open(dst_path, 'w', **dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=target_crs,
                    resampling=resampling_method
                )


def clip_raster(raster_path, clipping_geom, output_path=None, nodata_value=None):
    with rasterio.open(raster_path) as src:
        out_nodata = nodata_value if nodata_value is not None else src.nodata
        out_image, out_transform = mask(src, clipping_geom, crop=True, nodata=out_nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": out_image.shape[1],
            "width": out_image.shape[2], "transform": out_transform, "nodata": out_nodata
        })
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)
    return out_image, out_meta


def resample_raster_to_target(src_path, target_meta, resampling_method=Resampling.nearest):
    with rasterio.open(src_path) as src:
        dest_data = np.empty((src.count, target_meta["height"], target_meta["width"]),
                             dtype=src.meta['dtype'])
        reproject(
            source=src.read(), destination=dest_data,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_meta["transform"], dst_crs=target_meta["crs"],
            resampling=resampling_method
        )
    return dest_data


def calculate_weighted_ndvi(polygon, ndvi_path, pop_path):
    """Calculate population-weighted NDVI for a given polygon."""
    with rasterio.open(ndvi_path) as ndvi_src:
        ndvi_bounds = ndvi_src.bounds
        ndvi_nodata = ndvi_src.nodata
    with rasterio.open(pop_path) as pop_src:
        pop_nodata = pop_src.nodata

    if isinstance(polygon, (gpd.GeoDataFrame, gpd.GeoSeries)):
        polygon = polygon.iloc[0]

    if not polygon.intersects(box(*ndvi_bounds)):
        return np.nan

    ndvi_clip_data, _ = clip_raster(ndvi_path, [polygon.__geo_interface__], nodata_value=ndvi_nodata)
    pop_clip_data, _ = clip_raster(pop_path, [polygon.__geo_interface__], nodata_value=pop_nodata)

    ndvi_clip = ndvi_clip_data[0].astype(np.float32)
    pop_clip = pop_clip_data[0].astype(np.float32)

    ndvi_clip[ndvi_clip <= -1e4] = np.nan
    ndvi_clip[ndvi_clip > 1] = np.nan
    pop_clip[pop_clip <= 0] = np.nan
    pop_clip[pop_clip > 1e8] = np.nan

    if ndvi_nodata is not None:
        ndvi_clip[ndvi_clip == ndvi_nodata] = np.nan
    if pop_nodata is not None:
        pop_clip[pop_clip == pop_nodata] = np.nan

    valid_mask = ~np.isnan(ndvi_clip) & ~np.isnan(pop_clip)
    total_pop = np.nansum(pop_clip[valid_mask])
    weighted_sum = np.nansum(ndvi_clip[valid_mask] * pop_clip[valid_mask])
    return weighted_sum / total_pop if total_pop > 0 else np.nan


def calculate_weighted_treecover(polygon, treecover_path, pop_path):
    """Calculate population-weighted tree cover for a given polygon."""
    with rasterio.open(treecover_path) as tree_src:
        tree_bounds = tree_src.bounds
        tree_nodata = tree_src.nodata
    with rasterio.open(pop_path) as pop_src:
        pop_nodata = pop_src.nodata

    if isinstance(polygon, (gpd.GeoDataFrame, gpd.GeoSeries)):
        polygon = polygon.iloc[0]

    if not polygon.intersects(box(*tree_bounds)):
        return np.nan

    tree_clip_data, _ = clip_raster(treecover_path, [polygon.__geo_interface__], nodata_value=tree_nodata)
    pop_clip_data, _ = clip_raster(pop_path, [polygon.__geo_interface__], nodata_value=pop_nodata)

    tree_clip = tree_clip_data[0].astype(np.float32)
    pop_clip = pop_clip_data[0].astype(np.float32)

    if tree_nodata is not None:
        tree_clip[tree_clip == tree_nodata] = np.nan
    if pop_nodata is not None:
        pop_clip[pop_clip == pop_nodata] = np.nan

    tree_clip[tree_clip < 0] = np.nan
    tree_clip[tree_clip > 100] = np.nan
    pop_clip[pop_clip <= 0] = np.nan
    pop_clip[pop_clip > 1e8] = np.nan

    valid_mask = ~np.isnan(tree_clip) & ~np.isnan(pop_clip)
    if np.sum(valid_mask) == 0:
        return np.nan

    total_pop = np.nansum(pop_clip[valid_mask])
    weighted_sum = np.nansum(tree_clip[valid_mask] * pop_clip[valid_mask])
    return weighted_sum / total_pop if total_pop > 0 else np.nan


def plot_ndvi_vs_negoal_gradient(ndvi_resampled_path, aoi_gdf, ne_goal_value):
    with rasterio.open(ndvi_resampled_path) as src:
        ndvi_crs = src.crs
    if aoi_gdf.crs != ndvi_crs:
        aoi_gdf = aoi_gdf.to_crs(ndvi_crs)

    ndvi_means = []
    with rasterio.open(ndvi_resampled_path) as src:
        for geom in aoi_gdf.geometry:
            try:
                out_image, _ = rasterio.mask.mask(src, [geom], crop=True, filled=True, nodata=np.nan)
                valid = out_image[0][np.isfinite(out_image[0])]
                ndvi_means.append(np.nanmean(valid) if valid.size > 0 else np.nan)
            except Exception:
                ndvi_means.append(np.nan)

    aoi_gdf = aoi_gdf.copy()
    aoi_gdf["ndvi_delta"] = ne_goal_value - np.array(ndvi_means)

    max_abs = max(abs(np.nanmin(aoi_gdf["ndvi_delta"])), abs(np.nanmax(aoi_gdf["ndvi_delta"])))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(aoi_gdf.total_bounds[[0, 2]])
    ax.set_ylim(aoi_gdf.total_bounds[[1, 3]])
    aoi_gdf.plot(ax=ax, column="ndvi_delta", cmap=cmap, norm=norm, edgecolor="black", linewidth=0.8)
    sm_obj = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm_obj, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(-max_abs, max_abs, 7))
    cbar.ax.set_yticklabels([f"{t:+.1f}" for t in np.linspace(-max_abs, max_abs, 7)], fontsize=9)
    ax.set_title("NDVI_target - NDVI_baseline", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False); ax.set_facecolor("white")
    return fig


def merge_ndvi_landcover_data(ndvi_csv_path, landcover_csv_path):
    if not os.path.exists(ndvi_csv_path):
        raise FileNotFoundError(f"NDVI CSV file not found: {ndvi_csv_path}")
    if not os.path.exists(landcover_csv_path):
        raise FileNotFoundError(f"Land Cover CSV file not found: {landcover_csv_path}")
    return pd.merge(pd.read_csv(ndvi_csv_path), pd.read_csv(landcover_csv_path), on="GEOID", how="inner")


def plot_ndvi_vs_treecover(df, adm_level):
    ndvi_threshold = df["weighted_ndvi"].quantile(0.75)
    df["NDVI_Level"] = df["weighted_ndvi"].apply(
        lambda x: "Above 75th Percentile" if x > ndvi_threshold else "Below 75th Percentile"
    )
    df["marker_size"] = df["cover_10"].apply(lambda x: 20 if x > 30 else 5)
    fig = px.scatter(
        df, x="cover_10", y="weighted_ndvi", color="NDVI_Level", size="marker_size",
        category_orders={"NDVI_Level": ["Above 75th Percentile", "Below 75th Percentile"]},
        color_discrete_map={"Above 75th Percentile": "#5ab4ac", "Below 75th Percentile": "#d8b365"},
        trendline="rolling", trendline_color_override="red", trendline_scope="overall",
        hover_data={"GEOID": True, "cover_10": True, "weighted_ndvi": True}
    )
    fig.update_layout(
        title=f"Tree Cover vs. Mean NDVI (by adm {adm_level})",
        xaxis_title="Tree Cover (%)", yaxis_title="Mean NDVI (Population-Weighted)",
        template="plotly"
    )
    fig.show()


def load_health_effects(excel_file, health_indicator_i, baseline_risk=None, NE_goal=0.3):
    es = pd.read_excel(excel_file, sheet_name="Sheet1", usecols="A:D")
    es_selected = es[es["health_indicator"] == health_indicator_i]
    effect_size_i = _validate_effect_size(es_selected)
    effect_indicator_i = _validate_effect_indicator(es_selected)
    baseline_risk = 0.15 if baseline_risk is None else baseline_risk
    risk_ratio = _calculate_risk_ratio(effect_size_i, effect_indicator_i, baseline_risk)
    return {
        "effect_size_i": effect_size_i, "effect_indicator_i": effect_indicator_i,
        "risk_ratio": risk_ratio, "NE_goal": NE_goal, "baseline_risk": baseline_risk,
    }


def _validate_effect_size(es_data):
    values = np.unique(es_data["effect_size"].values)
    if len(values) == 1:
        return values[0]
    raise ValueError("Multiple effect_size values found")


def _validate_effect_indicator(es_data):
    indicators = np.unique(es_data["effect_indicator"].values)
    if len(indicators) == 1:
        return indicators[0]
    raise ValueError("Multiple effect_indicators found")


def _calculate_risk_ratio(effect_size, effect_indicator, baseline_risk):
    if effect_indicator == "risk ratio":
        return effect_size if isinstance(baseline_risk, float) else np.full_like(baseline_risk, effect_size)
    if effect_indicator == "odd ratio":
        return effect_size / (1 - baseline_risk + baseline_risk * effect_size)
    raise ValueError("effect_indicator must be 'risk ratio' or 'odd ratio'")


def calculate_pd_layer(ndvi_raster_path, pop_raster_path, rr, NE_goal,
                       baseline_risk_rate, output_dir="."):
    with rasterio.open(ndvi_raster_path) as src:
        ndvi_data = src.read(1).astype(np.float32)
        ndvi_meta = src.meta.copy()
        ndvi_meta.update(dtype="float32", nodata=np.nan)

    ndvi_data[ndvi_data < 0] = np.nan
    if baseline_risk_rate.shape != ndvi_data.shape:
        raise ValueError(f"Risk shape {baseline_risk_rate.shape} != NDVI {ndvi_data.shape}")

    delta_NE_i = NE_goal - ndvi_data
    delta_positive = np.where(delta_NE_i > 0, delta_NE_i, 0)
    RR_i = np.exp(np.log(rr) * 10 * delta_positive)
    PF_i = 1 - RR_i

    with rasterio.open(pop_raster_path) as src:
        pop_data = src.read(1).astype(np.float32)
        if pop_data.shape != ndvi_data.shape:
            pop_resampled = np.empty_like(ndvi_data, dtype=np.float32)
            reproject(source=pop_data, destination=pop_resampled,
                      src_transform=src.meta["transform"], src_crs=src.meta["crs"],
                      dst_transform=ndvi_meta["transform"], dst_crs=ndvi_meta["crs"],
                      resampling=Resampling.bilinear)
            pop_data = pop_resampled

    pop_data[pop_data <= 0] = np.nan
    PD_i = np.maximum(0, PF_i * baseline_risk_rate * pop_data)

    os.makedirs(output_dir, exist_ok=True)
    output_paths = {}
    for name, data in [("ndvi_masked", ndvi_data), ("delta_NE_i", delta_NE_i),
                       ("delta_NE_i_positive", delta_positive), ("PD_i", PD_i)]:
        path = os.path.join(output_dir, f"{name}.tif")
        with rasterio.open(path, "w", **ndvi_meta) as dst:
            dst.write(data, 1)
        output_paths[name] = path

    return {"PD_i": PD_i, "PF_i": PF_i, "RR_i": RR_i, "delta_NE_i": delta_NE_i,
            "delta_NE_i_positive": delta_positive, "output_paths": output_paths}


def plot_pd_map_v1(PD_raster_path, aoi_gdf, return_fig=False):
    with rasterio.open(PD_raster_path) as src:
        PD_masked, PD_transform = rasterio.mask.mask(src, aoi_gdf.geometry, crop=True, nodata=src.nodata)
        PD_meta = src.meta.copy()
        PD_meta.update({"transform": PD_transform,
                        "width": PD_masked.shape[2], "height": PD_masked.shape[1]})

    PD_flat = PD_masked[0].ravel()
    PD_flat = PD_flat[np.isfinite(PD_flat)]
    if PD_flat.size == 0:
        return None

    min_val, max_val = PD_flat.min(), PD_flat.max()
    vcenter = 0 if min_val < 0 < max_val else (min_val + max_val) / 2
    if min_val >= max_val:
        max_val = min_val + 1e-6

    norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=vcenter, vmax=max_val)
    cmap = plt.cm.RdBu
    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(PD_masked[0], cmap=cmap, norm=norm, extent=pd_extent, origin="upper")
    aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)
    ticks = np.linspace(min_val, max_val, 6)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PD_i Values (red=negative, blue=positive)", fontsize=9)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks])
    ax.set_title("Preventable Depression Risk Map", fontsize=11)
    ax.set_xlabel("Longitude", fontsize=9); ax.set_ylabel("Latitude", fontsize=9)
    ax.set_facecolor("white"); ax.grid(False)
    if return_fig:
        return fig
    else:
        plt.show()


def plot_pd_map_v3(PD_masked, PD_meta, aoi_gdf, figures_dir,
                   output_name="PD_risk_map_v8_discrete.png", return_fig=False):
    PD_masked[0][PD_masked[0] < 0] = 0
    PD_flat = PD_masked[0].ravel()
    PD_flat = PD_flat[np.isfinite(PD_flat)]
    if PD_flat.size == 0:
        raise ValueError("No valid PD_i data in the AOI.")

    quantiles = [0.01, 0.1, 0.25, 0.50, 0.6, 0.7, 0.9, 0.99]
    breakpoints = np.unique(np.round(np.concatenate([np.quantile(PD_flat, quantiles), [0]]), 1))
    if len(breakpoints) == 1:
        breakpoints = [breakpoints[0] - 1e-6, breakpoints[0] + 1e-6]

    vmin, vmax = np.min(breakpoints), np.max(breakpoints)
    norm = Normalize(vmin=vmin, vmax=vmax)
    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.set_aspect('equal')
    if hasattr(aoi_gdf, 'unary_union'):
        gpd.GeoSeries([aoi_gdf.unary_union], crs=aoi_gdf.crs).boundary.plot(
            ax=ax, edgecolor="black", linewidth=1.0)
    else:
        aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

    im = ax.imshow(PD_masked[0], cmap=plt.cm.Blues, norm=norm, extent=pd_extent, origin="upper")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    tick_values = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.1f}" for v in tick_values])
    ax.set_title("Preventable Depression Cases", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.grid(False); ax.set_facecolor("white")
    if return_fig:
        return fig


def safe_raster_write(path, meta, data):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        path = path.replace(".tif", "_new.tif")
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(data, 1)


# ============================================================================
# Legacy Fitting Methods (kept for reference; not called by run_ndvi_tree_analysis)
# ============================================================================

def create_sigmoid_fit(x_data, y_data, extend_to_range=(0, 100)):
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean, y_clean = x_data[valid_mask], y_data[valid_mask]

    def sigmoid(x, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    try:
        popt, _ = curve_fit(sigmoid, x_clean, y_clean,
                            p0=[0.1, 50.0], bounds=([1e-3, -1e6], [10.0, 1e6]))
        x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)
        return x_extended, sigmoid(x_extended, *popt)
    except Exception:
        return create_extended_polynomial_constrained(x_data, y_data, extend_to_range)


def create_extended_lowess_smooth(x_data, y_data, extend_to_range=(0, 100)):
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean, y_clean = x_data[valid_mask], y_data[valid_mask]
    n_points = len(x_clean)
    frac = 0.6 if n_points > 100 else (0.5 if n_points > 50 else (0.4 if n_points > 20 else 0.3))

    try:
        lowess_result = sm.nonparametric.lowess(y_clean, x_clean, frac=frac, it=3)
        x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]
    except Exception:
        coeffs = np.polyfit(x_clean, y_clean, deg=1)
        x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)
        return x_extended, np.clip(np.polyval(coeffs, x_extended), 0.0, 1.0)

    x_min_data, x_max_data = np.min(x_clean), np.max(x_clean)
    n_edge = min(5, max(1, len(x_lowess) // 10))
    slope_low = (y_lowess[n_edge] - y_lowess[0]) / (x_lowess[n_edge] - x_lowess[0])
    slope_high = (y_lowess[-1] - y_lowess[-n_edge - 1]) / (x_lowess[-1] - x_lowess[-n_edge - 1])
    y_min_fit, y_max_fit = y_lowess[0], y_lowess[-1]

    x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)
    y_extended = np.array([
        y_min_fit + slope_low * (x - x_min_data) if x < x_min_data else
        (y_max_fit + slope_high * (x - x_max_data) if x > x_max_data else np.interp(x, x_lowess, y_lowess))
        for x in x_extended
    ])
    for i in range(1, len(y_extended)):
        if y_extended[i] < y_extended[i - 1]:
            y_extended[i] = y_extended[i - 1]
    return x_extended, np.clip(y_extended, 0.0, 1.0)


def create_extended_polynomial_monotonic(x_data, y_data, extend_to_range=(0, 100), degree=2):
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean, y_clean = x_data[valid_mask], y_data[valid_mask]
    actual_degree = 1 if len(x_clean) < 20 else degree
    coefficients = np.polyfit(x_clean, y_clean, deg=actual_degree)
    poly_function = np.poly1d(coefficients)
    x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)
    y_extended = poly_function(x_extended)
    for i in range(1, len(y_extended)):
        if y_extended[i] < y_extended[i - 1]:
            y_extended[i] = y_extended[i - 1]
    return x_extended, np.clip(y_extended, 0.0, 1.0)


def create_extended_polynomial_constrained(x_data, y_data, extend_to_range=(0, 100)):
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean, y_clean = x_data[valid_mask], y_data[valid_mask]

    def power_func_shifted(x, a, b, k):
        return a + b * np.power(np.clip(x, 0, None) / 100.0, k)

    try:
        y_min, y_max = np.nanmin(y_clean), np.nanmax(y_clean)
        p0 = [y_min, max(y_max - y_min, 1e-3), 1.0]
        popt, _ = curve_fit(power_func_shifted, x_clean, y_clean,
                            p0=p0, bounds=([-np.inf, 0.0, 0.1], [np.inf, np.inf, 5.0]),
                            maxfev=20000)
        x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)
        return x_extended, power_func_shifted(x_extended, *popt)
    except Exception:
        x_extended = np.linspace(extend_to_range[0], extend_to_range[1], 200)
        return x_extended, np.full_like(x_extended, np.nanmean(y_data))


def create_extended_polynomial(x_data, y_data, extend_to_range=(0, 100), degree=2,
                               monotonic=True, constrained=False, method='auto'):
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    n_points = np.sum(valid_mask)
    if constrained:
        return create_extended_polynomial_constrained(x_data, y_data, extend_to_range)
    if method == 'sigmoid':
        return create_sigmoid_fit(x_data, y_data, extend_to_range)
    if method == 'auto':
        method = 'lowess' if n_points > 50 else ('sigmoid' if n_points > 20 else 'polynomial')
    if method == 'lowess':
        return create_extended_lowess_smooth(x_data, y_data, extend_to_range)
    elif method == 'sigmoid':
        return create_sigmoid_fit(x_data, y_data, extend_to_range)
    else:
        return create_extended_polynomial_monotonic(x_data, y_data, extend_to_range, degree)