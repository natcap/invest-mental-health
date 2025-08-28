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
import statsmodels.api as sm

import plotly.express as px
import plotly.io as pio


def run_ndvi_tree_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, excel_file, output_dir):
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import rasterio
    import statsmodels.api as sm
    from rasterio.enums import Resampling

    # from NDVI_PW import reproject_shapefile, reproject_raster, clip_raster, resample_raster_to_target, calculate_weighted_ndvi
    # from Tree_Cover import compute_zonal_statistics
    # from Result import merge_ndvi_landcover_data

    target_crs = "EPSG:5070"

    # 1. Load AOI
    aoi_adm1 = reproject_shapefile(aoi_adm1_path, target_crs)
    aoi_adm1_geometry = [aoi_adm1.geometry.unary_union]

    aoi_adm2 = gpd.read_file(aoi_adm2_path)
    if aoi_adm2.crs != target_crs:
        aoi_adm2 = aoi_adm2.to_crs(target_crs)
    aoi_adm2_clipped = gpd.clip(aoi_adm2, aoi_adm1)
    aoi_adm2_clipped = aoi_adm2_clipped[aoi_adm2_clipped.area > 100]

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
            "crs": pop_src.crs,
            "transform": pop_src.transform,
            "width": pop_src.width,
            "height": pop_src.height
        }
        ndvi_resampled = resample_raster_to_target(ndvi_dst_clip, target_meta, Resampling.bilinear)
        with rasterio.open(ndvi_resampled_path, 'w', **pop_src.meta) as dst:
            dst.write(ndvi_resampled)

    # 4. Compute weighted NDVI
    weighted_ndvi_list = [
        calculate_weighted_ndvi(row.geometry, ndvi_resampled_path, pop_dst_clip)
        for _, row in aoi_adm2_clipped.iterrows()
    ]
    aoi_adm2_clipped["weighted_ndvi"] = weighted_ndvi_list

    ndvi_fig, ax = plt.subplots(figsize=(5.4, 4.5))
    aoi_adm2_clipped.plot(
        column="weighted_ndvi", cmap="Greens", linewidth=0.8, ax=ax, edgecolor="black",
        legend=True,
        legend_kwds={"orientation": "vertical"}
    )
    ax.set_title("Population-Weighted NDVI by Census Tract")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ndvi_fig.tight_layout()

    # Step 6: Tree cover zonal stats (reload shapefile to match GEOID )
    raster_reproject_path = tree_path.replace(".tif", "_prj.tif")
    raster_clipped_path = raster_reproject_path.replace(".tif", "_clipped.tif")
    reproject_raster(tree_path, target_crs, raster_reproject_path)
    clip_raster(raster_reproject_path, aoi_adm1_geometry, raster_clipped_path)

    # reload shapefile to match GEOID
    aoi_adm2_raw = gpd.read_file(aoi_adm2_path)
    if aoi_adm2_raw.crs != target_crs:
        aoi_adm2_raw = aoi_adm2_raw.to_crs(target_crs)
    aoi_adm2_raw_clipped = gpd.clip(aoi_adm2_raw, aoi_adm1)
    aoi_adm2_raw_clipped = aoi_adm2_raw_clipped[aoi_adm2_raw_clipped.area > 100]


    # calculate zonal stats
    df_stats = compute_zonal_statistics(aoi_adm2_raw_clipped, raster_clipped_path)
    aoi_joined = aoi_adm2_raw_clipped.merge(df_stats, on="GEOID")

    tree_fig, ax = plt.subplots(figsize=(8, 4.5))
    aoi_joined.plot(
        column="cover_10", cmap="viridis_r", linewidth=0.8, ax=ax, edgecolor="black",
        legend=True,
        legend_kwds={ "orientation": "vertical"}
    )
    ax.set_title("Tree cover (%) by census tract")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    tree_fig.tight_layout()

    # 6. Merge CSV and LOWESS
    ndvi_csv_path = os.path.join(output_dir, "population_weighted_ndvi_by_adm2.csv")
    landcover_csv_path = os.path.join(output_dir, "landcover_percentages_adm2.csv")
    aoi_adm2_clipped[["GEOID", "weighted_ndvi"]].to_csv(ndvi_csv_path, index=False)
    df_stats.to_csv(landcover_csv_path, index=False)

    df_merged = merge_ndvi_landcover_data(ndvi_csv_path, landcover_csv_path)
    df_sorted = df_merged.sort_values("cover_10")
    x = df_sorted["cover_10"].values
    y = df_sorted["weighted_ndvi"].values
    # z = sm.nonparametric.lowess(y, x, frac=0.4)
    # x_lowess = z[:, 0]
    # y_lowess = z[:, 1]
    x_lowess, y_lowess = create_extended_lowess(x, y, extend_to_range=(0, 100))

    # slider_fig, ax = plt.subplots(figsize=(12, 4))
    # ax.plot(x, y, 'o', alpha=0.4, label='Data', color='skyblue')
    # ax.plot(x_lowess, y_lowess, 'r-', linewidth=2, label='LOWESS')
    # ax.set_xlabel("Tree Cover (%)")
    # ax.set_ylabel("NDVI")
    # ax.set_title("Select Tree Cover to Set NE_goal (NDVI)")
    # ax.grid(True)
    # ax.legend()
    slider_fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, 'o', alpha=0.4, label='Original Data', color='skyblue')
    ax.plot(x_lowess, y_lowess, 'r-', linewidth=2, label='Extended LOWESS (0-100%)')

    # 标记原始数据范围
    ax.axvline(min(x), color='gray', linestyle=':', alpha=0.7, label='Data Range')
    ax.axvline(max(x), color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel("Tree Cover (%)")
    ax.set_ylabel("NDVI")
    ax.set_title("Select Tree Cover to Set NE_goal (NDVI) - Extended Range")
    ax.set_xlim(0, 100)  # 显示完整0-100%范围
    ax.grid(True)
    ax.legend()

    slider_fig.tight_layout()

    NE_goal = 0.3  # placeholder
    return NE_goal, ndvi_fig, tree_fig, slider_fig, x_lowess, y_lowess, aoi_adm2_clipped, ndvi_resampled_path


def run_pd_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, excel_file, output_dir,
                    NE_goal, aoi_adm2_clipped, x_lowess, y_lowess, cost_value):
    target_crs = "EPSG:5070"

    # Load AOI (adm1)
    aoi_adm1 = gpd.read_file(aoi_adm1_path)
    if aoi_adm1.crs != target_crs:
        aoi_adm1 = aoi_adm1.to_crs(target_crs)

    # === Fix: create population raster output path ===
    pop_dst_path = pop_path.replace("_setnull", "").replace(".tif", "_reprojected.tif")
    pop_dst_clip = pop_dst_path.replace(".tif", "_clipped.tif")

    # === Process risk raster: reproject, clip, resample ===
    risk_dst_path = risk_path.replace(".tif", "_prj.tif")
    risk_dst_clip = risk_dst_path.replace(".tif", "_clipped.tif")
    risk_resampled_path = risk_dst_clip.replace(".tif", "_100m.tif")

    reproject_raster(risk_path, target_crs, risk_dst_path)
    clip_raster(risk_dst_path, [aoi_adm1.geometry.unary_union], risk_dst_clip)

    # Resample risk raster to 100m resolution (aligned with population raster)
    with rasterio.open(pop_dst_clip) as pop_src:
        target_meta = {
            "crs": pop_src.crs,
            "transform": pop_src.transform,
            "width": pop_src.width,
            "height": pop_src.height
        }
        risk_resampled = resample_raster_to_target(risk_dst_clip, target_meta, Resampling.bilinear)
        with rasterio.open(risk_resampled_path, 'w', **pop_src.meta) as dst:
            dst.write(risk_resampled)

    # Read risk raster and replace invalid values (<0) with 0.15
    with rasterio.open(risk_resampled_path) as risk_src:
        baseline_risk_raster = risk_src.read(1)
        baseline_risk_raster = np.where(baseline_risk_raster < 0, 0.15, baseline_risk_raster)

    # Load health effect parameters using scalar baseline risk
    result = load_health_effects(
        excel_file=excel_file,
        health_indicator_i="depression",
        baseline_risk=0.15,
        NE_goal=NE_goal
    )

    # Read NDVI and population data
    ndvi_raster_path = ndvi_path.replace(".tif", "_prj_clipped_100m.tif")
    with rasterio.open(ndvi_raster_path) as ndvi_src:
        NDVI_array = ndvi_src.read(1)
        ndvi_meta = ndvi_src.meta

    with rasterio.open(pop_dst_clip) as pop_src:
        Pop_array = pop_src.read(1)

    # Compute PD_i raster and outputs
    results = calculate_pd_layer(
        ndvi_raster_path=ndvi_raster_path,
        pop_raster_path=pop_dst_clip,
        rr=result["risk_ratio"],
        NE_goal=NE_goal,
        baseline_risk_rate=baseline_risk_raster,
        output_dir=output_dir
    )

    PD_raster_path = os.path.join(output_dir, "PD_i.tif")
    with rasterio.open(PD_raster_path) as src:
        PD_masked, PD_transform = rasterio.mask.mask(src, aoi_adm1.geometry, crop=True, nodata=src.nodata)
        PD_meta = src.meta.copy()
        PD_meta.update({"transform": PD_transform, "width": PD_masked.shape[2], "height": PD_masked.shape[1]})

    # Plot continuous PD_i map with colorbar
    fig1 = plot_pd_map_v1(PD_raster_path, aoi_adm2_clipped, return_fig=True)

    # Get external boundary of adm1
    aoi_adm1_outline = gpd.GeoDataFrame(geometry=[aoi_adm1.unary_union], crs=aoi_adm1.crs)

    # Plot discrete PD_i map (boundary only)
    fig2 = plot_pd_map_v3(
        PD_masked=PD_masked,
        PD_meta=PD_meta,
        aoi_gdf=aoi_adm1_outline,
        figures_dir=output_dir,
        return_fig=True
    )

    # Clean PD_i: replace negative values with NaN
    with rasterio.open(PD_raster_path) as src:
        PD_data = src.read(1)
        meta = src.meta.copy()

    PD_data_clean = np.where(PD_data < 0, np.nan, PD_data)
    meta.update({"nodata": np.nan})

    # Save cleaned raster
    cleaned_raster_path = os.path.join(output_dir, "Prev_case.tif")
    with rasterio.open(cleaned_raster_path, "w", **meta) as dst:
        dst.write(PD_data_clean, 1)

    # Compute economic cost per pixel and save
    PD_cost = PD_data_clean * (cost_value / 1000)
    cost_raster_path = os.path.join(output_dir, "Prev_cost_pixel.tif")
    with rasterio.open(cost_raster_path, "w", **meta) as dst:
        dst.write(PD_cost, 1)

    # Zonal cost summary for each tract
    zonal_result = zonal_stats(aoi_adm2_clipped, cost_raster_path, stats="sum", nodata=np.nan)
    sum_values = [round(stat["sum"], 2) if stat["sum"] is not None else 0 for stat in zonal_result]
    aoi_adm2_clipped["PD_i_sum_cost"] = sum_values

    # Plot cost choropleth
    vmin = aoi_adm2_clipped["PD_i_sum_cost"].min()
    vmax = aoi_adm2_clipped["PD_i_sum_cost"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.OrRd

    fig_hist, ax = plt.subplots(figsize=(6.3, 4.5), constrained_layout=True)
    ax.set_aspect('equal')

    aoi_adm2_clipped.plot(
        column="PD_i_sum_cost",
        cmap=cmap,
        norm=norm,
        linewidth=0.8,
        ax=ax,
        edgecolor="black",
        legend=False
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{val:.0f}" for val in tick_values])

    ax.set_xlim(aoi_adm2_clipped.total_bounds[[0, 2]])
    ax.set_ylim(aoi_adm2_clipped.total_bounds[[1, 3]])
    ax.set_title("Total Preventable Cost (1000 USD) by Tract", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    # Simulate cost curve (x = tree cover)
    selected_cover = float(np.interp(NE_goal, y_lowess, x_lowess))
    fig_cost_curve = simulate_cost_vs_treecover(
        x_lowess, y_lowess,
        Pop_array, NDVI_array, baseline_risk_raster,
        result["risk_ratio"], PD_masked, selected_cover=selected_cover, cost_value=cost_value
    )

    # Calculate total preventable cases
    per_pixel_cases = PD_masked[0]
    total_pd_cases = np.nansum(per_pixel_cases)

    return fig1, fig2, fig_hist, fig_cost_curve, total_pd_cases


def simulate_cost_vs_treecover(x_lowess, y_lowess, Pop_array, baseline_ndvi_raster,
                               baseline_risk_raster, risk_ratio, PD_masked=None,
                               selected_cover=None, cost_value=11000):
    """Simulate total preventable cost savings under different tree cover targets."""

    start = (min(x_lowess) // 5) * 5
    end = ((max(x_lowess) + 4) // 5) * 5 + 5
    tree_cover_range = list(np.arange(start, end + 1, 5))

    cost_savings = []  # Total preventable cost savings
    highlight_cost = None

    print(f"=== Cost Curve Calculation ===")
    print(f"Tree cover range: {start}% - {end}%")
    print(f"Risk ratio: {risk_ratio}")
    print(f"Cost per case: ${cost_value}")

    for i, tree_cover in enumerate(tree_cover_range):
        # Get corresponding NDVI target
        ndvi_goal = np.interp(tree_cover, x_lowess, y_lowess)

        # Calculate NDVI improvement
        delta_ndvi = ndvi_goal - baseline_ndvi_raster

        # Only positive improvements contribute to health benefits
        delta_positive = np.where(delta_ndvi > 0, delta_ndvi, 0)

        # Health impact calculation
        RR_i = np.exp(np.log(risk_ratio) * 10 * delta_positive)
        PF_i = 1 - RR_i
        PD_i = PF_i * baseline_risk_raster * Pop_array
        PD_i = np.where(PD_i > 0, PD_i, 0)

        # Total preventable cases at this tree cover level
        total_preventable_cases = np.nansum(PD_i)

        # Total cost savings (preventable cases × cost per case)
        total_cost_savings = total_preventable_cases * (cost_value / 1000)  # In thousands USD

        cost_savings.append(total_cost_savings)

        # Debug output for first few points
        if i < 3:
            print(f"Tree cover {tree_cover}%:")
            print(f"  NDVI target: {ndvi_goal:.3f}")
            print(f"  Pixels with improvement: {np.sum(delta_positive > 0):,}")
            print(f"  Preventable cases: {total_preventable_cases:,.0f}")
            print(f"  Cost savings: ${total_cost_savings:,.0f}k")

        # Mark selected target
        if selected_cover is not None and abs(tree_cover - selected_cover) < 2.5:
            highlight_cost = total_cost_savings

    # Plotting
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(tree_cover_range, cost_savings, marker='o', color='steelblue', linewidth=2, markersize=6)

    # Formatting
    ax.set_xlabel("Tree Cover Target (%)", fontsize=10)
    ax.set_ylabel("Total Preventable Cost Savings (1000 USD)", fontsize=10)
    ax.set_title("Cost Savings by Tree Cover Target", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # Highlight selected target
    if selected_cover is not None and highlight_cost is not None:
        ax.plot(selected_cover, highlight_cost, 'ro', markersize=10, label="Selected Target")
        ax.legend(fontsize=9)

    # Set axis limits
    ax.set_xlim(min(tree_cover_range) - 2, max(tree_cover_range) + 2)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


### tree cover###

def process_shapefile(shp_path, target_crs, aoi_boundary):
    """Load, reproject, and clip a shapefile."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf_clipped = gpd.clip(gdf, aoi_boundary)
    gdf_clipped = gdf_clipped[gdf_clipped.area > 100]  # Filter small polygons
    gdf_clipped.reset_index(drop=True, inplace=True)

    return gdf_clipped


def compute_zonal_statistics(aoi_shapefile, raster_path):
    with rasterio.open(raster_path) as src:
        nodata_val = src.nodata
        scale = (src.scales[0] if src.scales else 1.0)
        offset = (src.offsets[0] if src.offsets else 0.0)

        data = src.read(1, masked=True)  # generate mask
        data = data.astype(float) * scale + offset

        print("=== Tree Cover Debug ===")
        print(f"dtype: {src.dtypes[0]}, nodata: {nodata_val}, scale: {scale}, offset: {offset}")
        print(f"Raster min/max (masked): {data.min():.1f} / {data.max():.1f}")

        unique_vals = np.unique(data.compressed())[:10]
        print(f"Sample values: {unique_vals}")


    use_conversion = False
    conversion_factor = 1.0
    if scale == 1.0 and offset == 0.0:

        if data.max() > 100 and data.max() <= 253:
            use_conversion = True
            conversion_factor = 100.0 / 255.0

    stats = zonal_stats(
        aoi_shapefile, raster_path,
        stats=['mean', 'median', 'std', 'count'],
        nodata=nodata_val
    )

    percent_stats = []
    for s in stats:
        mean_val = s['mean']
        if mean_val is None:
            cover_percent = 0.0
        else:
            # apply scale/offset
            cover = mean_val * scale + offset

            if use_conversion:
                cover *= conversion_factor
            cover_percent = max(0.0, min(100.0, cover))

        percent_stats.append({
            'cover_10': cover_percent,
            'cover_mean': cover_percent,
            'cover_raw': mean_val if mean_val is not None else 0.0,
            'cover_count': s['count'] if s['count'] is not None else 0
        })

    df = pd.DataFrame(percent_stats)
    df["GEOID"] = aoi_shapefile["GEOID"].values

    print("Tree cover results:")
    print(f"  Range: {df['cover_10'].min():.1f}% - {df['cover_10'].max():.1f}%")
    print(f"  Mean: {df['cover_10'].mean():.1f}%")
    print(f"  >50%: {(df['cover_10'] > 50).sum()}, >70%: {(df['cover_10'] > 70).sum()}")
    print(f"  Sample values: {df['cover_10'].head().tolist()}")

    return df


def save_landcover_csv(dataframe, aoi_shapefile, output_path):
    """Save the land cover percentage data to a CSV file."""
    if "GEOID" in aoi_shapefile.columns:
        dataframe.insert(0, "GEOID", aoi_shapefile["GEOID"])

    dataframe.to_csv(output_path, index=False)
    print(f"CSV file saved to: {output_path}")


def plot_landcover(dataframe, aoi_shapefile, column, title, return_fig=False):
    """Plot a map of land cover percentages for a given class. Return figure if needed."""
    import matplotlib.pyplot as plt

    if column in dataframe.columns:
        aoi_shapefile = aoi_shapefile.copy()
        aoi_shapefile[column] = dataframe[column]

        fig, ax = plt.subplots(figsize=(5.5, 5))
        aoi_shapefile.plot(column=column, cmap="viridis", legend=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if return_fig:
            return fig
        else:
            plt.show()


### NDVI_PW ###

def reproject_shapefile(shp_path, target_crs, dst_path=None):
    """Reproject a shapefile to a target CRS."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    if dst_path:
        gdf.to_file(dst_path)
    return gdf


def reproject_raster(src_path, target_crs, dst_path, resampling_method=Resampling.nearest):
    """Reproject a raster to a target CRS."""
    with rasterio.open(src_path) as src:
        if src.crs == target_crs:
            with rasterio.open(dst_path, 'w', **src.meta) as dst:
                dst.write(src.read())
            return

        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )


def clip_raster(raster_path, clipping_geom, output_path=None, nodata_value=None):
    """Clip a raster to a given geometry."""
    with rasterio.open(raster_path) as src:
        out_nodata = nodata_value if nodata_value is not None else src.nodata
        out_image, out_transform = mask(src, clipping_geom, crop=True, nodata=out_nodata)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": out_nodata
        })
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)
    return out_image, out_meta


def resample_raster_to_target(src_path, target_meta, resampling_method=Resampling.nearest):
    """
    Resample a raster to match the resolution and transform of a target raster.

    Parameters:
        src_path (str): Path to the source raster.
        target_meta (dict): Metadata of the target raster (must include 'crs', 'transform', 'width', 'height').
        resampling_method: Resampling method to use (default: Resampling.nearest).

    Returns:
        numpy.ndarray: The resampled raster data.
    """
    with rasterio.open(src_path) as src:
        # Create an empty array with target dimensions
        dest_data = np.empty((src.count, target_meta["height"], target_meta["width"]), dtype=src.meta['dtype'])

        # Reproject the source raster to match the target
        reproject(
            source=src.read(),
            destination=dest_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            resampling=resampling_method
        )
    return dest_data


def calculate_weighted_ndvi(polygon, ndvi_path, pop_path):
    """Calculate population-weighted NDVI for a given polygon."""
    with rasterio.open(ndvi_path) as ndvi_src:
        ndvi_crs = ndvi_src.crs
        ndvi_bounds = ndvi_src.bounds
        ndvi_nodata = ndvi_src.nodata

    with rasterio.open(pop_path) as pop_src:
        pop_crs = pop_src.crs
        pop_nodata = pop_src.nodata

    if isinstance(polygon, (gpd.GeoDataFrame, gpd.GeoSeries)):
        polygon = polygon.iloc[0]

    raster_box = box(*ndvi_bounds)
    if not polygon.intersects(raster_box):
        return np.nan

    ndvi_clip_data, _ = clip_raster(ndvi_path, [polygon.__geo_interface__], nodata_value=ndvi_nodata)
    pop_clip_data, _ = clip_raster(pop_path, [polygon.__geo_interface__], nodata_value=pop_nodata)

    ndvi_clip = ndvi_clip_data[0].astype(np.float32)
    pop_clip = pop_clip_data[0].astype(np.float32)


    ndvi_clip[ndvi_clip <= -1e4] = np.nan
    ndvi_clip[ndvi_clip > 1] = np.nan


    pop_clip[pop_clip <= 0] = np.nan
    pop_clip[pop_clip > 1e8] = np.nan

    # print("NDVI nodata:", ndvi_nodata)
    # print("POP nodata:", pop_nodata)
    # print("NDVI min/max:", np.nanmin(ndvi_clip), np.nanmax(ndvi_clip))
    # print("POP min/max:", np.nanmin(pop_clip), np.nanmax(pop_clip))
    if ndvi_nodata is not None:
        ndvi_clip[ndvi_clip == ndvi_nodata] = np.nan
    if pop_nodata is not None:
        pop_clip[pop_clip == pop_nodata] = np.nan

    valid_mask = ~np.isnan(ndvi_clip) & ~np.isnan(pop_clip)
    total_pop = np.nansum(pop_clip[valid_mask])
    weighted_sum = np.nansum(ndvi_clip[valid_mask] * pop_clip[valid_mask])

    return weighted_sum / total_pop if total_pop > 0 else np.nan


def plot_ndvi_vs_negoal_gradient(ndvi_resampled_path, aoi_gdf, ne_goal_value):
    """Plot AOI blocks colored by NDVI deviation from NE_goal (including negative values)."""

    with rasterio.open(ndvi_resampled_path) as src:
        ndvi_crs = src.crs

    if aoi_gdf.crs != ndvi_crs:
        aoi_gdf = aoi_gdf.to_crs(ndvi_crs)

    ndvi_means = []
    with rasterio.open(ndvi_resampled_path) as src:
        for geom in aoi_gdf.geometry:
            try:
                out_image, _ = rasterio.mask.mask(src, [geom], crop=True, filled=True, nodata=np.nan)
                values = out_image[0]
                valid = values[np.isfinite(values)]
                mean_ndvi = np.nanmean(valid) if valid.size > 0 else np.nan
                ndvi_means.append(mean_ndvi)
            except Exception:
                ndvi_means.append(np.nan)

    aoi_gdf = aoi_gdf.copy()
    # Show full difference (positive AND negative)
    aoi_gdf["ndvi_delta"] = ne_goal_value - np.array(ndvi_means)

    delta_min = np.nanmin(aoi_gdf["ndvi_delta"])
    delta_max = np.nanmax(aoi_gdf["ndvi_delta"])
    max_abs = max(abs(delta_min), abs(delta_max))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(aoi_gdf.total_bounds[[0, 2]])
    ax.set_ylim(aoi_gdf.total_bounds[[1, 3]])

    aoi_gdf.plot(ax=ax, column="ndvi_delta", cmap=cmap, norm=norm,
                 edgecolor="black", linewidth=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.linspace(-max_abs, max_abs, 7))
    cbar.ax.set_yticklabels([f"{t:+.1f}" for t in np.linspace(-max_abs, max_abs, 7)], fontsize=9)

    ax.set_title("NDVI_target – NDVI_baseline", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_facecolor("white")

    return fig

### result.py ###

def merge_ndvi_landcover_data(ndvi_csv_path, landcover_csv_path):
    """
    Merge NDVI and Land Cover data on GEOID.

    Parameters:
        ndvi_csv_path (str): Path to the NDVI CSV file.
        landcover_csv_path (str): Path to the land cover CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing NDVI and Land Cover statistics.
    """
    print(f"Looking for NDVI CSV at: {os.path.abspath(ndvi_csv_path)}")
    print(f"Looking for Land Cover CSV at: {os.path.abspath(landcover_csv_path)}")

    if not os.path.exists(ndvi_csv_path):
        raise FileNotFoundError(f"NDVI CSV file not found: {ndvi_csv_path}")
    if not os.path.exists(landcover_csv_path):
        raise FileNotFoundError(f"Land Cover CSV file not found: {landcover_csv_path}")

    df_ndvi = pd.read_csv(ndvi_csv_path)
    df_landcover = pd.read_csv(landcover_csv_path)

    # Merge datasets on GEOID
    df_merged = pd.merge(df_ndvi, df_landcover, on="GEOID", how="inner")
    return df_merged


def plot_ndvi_vs_treecover(df, adm_level):
    """
    Create an interactive Plotly scatter plot of Tree Cover (%) vs. Population-Weighted NDVI.
    Includes a LOWESS trendline to visualize the overall relationship.
    """
    import statsmodels.api as sm  # Required for LOWESS trendline
    print("Generating Plotly scatter plot...")

    # Categorize NDVI level using the 75th percentile as threshold
    ndvi_threshold = df["weighted_ndvi"].quantile(0.75)
    df["NDVI_Level"] = df["weighted_ndvi"].apply(
        lambda x: "Above 75th Percentile" if x > ndvi_threshold else "Below 75th Percentile"
    )

    # Set marker size based on tree cover percentage
    df["marker_size"] = df["cover_10"].apply(lambda x: 20 if x > 30 else 5)

    # Create interactive scatter plot with LOWESS trendline
    fig = px.scatter(
        df,
        x="cover_10",
        y="weighted_ndvi",
        color="NDVI_Level",
        size="marker_size",
        category_orders={"NDVI_Level": ["Above 75th Percentile", "Below 75th Percentile"]},
        color_discrete_map={"Above 75th Percentile": "#5ab4ac", "Below 75th Percentile": "#d8b365"},
        trendline="rolling",                      # Add LOWESS trendline
        trendline_color_override="red",          # Trendline color
        trendline_scope="overall",               # Apply trendline to all data
        hover_data={"GEOID": True, "cover_10": True, "weighted_ndvi": True}
    )

    # Customize plot layout
    fig.update_layout(
        title=f"Tree Cover vs. Mean NDVI (by adm {adm_level})",
        xaxis_title="Tree Cover (%)",
        yaxis_title="Mean NDVI (Population-Weighted)",
        template="plotly"
    )

    fig.show()

def plot_ndvi_vs_treecover_popup(df):
    """
    Use matplotlib to create a popup scatter plot window with LOWESS trendline.
    """

    x = df["cover_10"]
    y = df["weighted_ndvi"]

    # Filter NaN
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]

    # Fit LOWESS
    lowess = sm.nonparametric.lowess
    z = lowess(y, x, frac=0.4)  # frac controls smoothing

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label="Data", c="lightblue", edgecolors="k")
    plt.plot(z[:, 0], z[:, 1], color="red", linewidth=2, label="LOWESS trend")

    plt.xlabel("Tree Cover (%)")
    plt.ylabel("Mean NDVI (Population-Weighted)")
    plt.title("Tree Cover vs. NDVI with LOWESS Trendline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def select_ne_goal_with_slider_combined(df, return_fig=False):
    """
    Combined LOWESS trendline + interactive slider + confirm button.
    Used to select NE_goal based on Tree Cover (%).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import statsmodels.api as sm
    import numpy as np

    # Prepare data
    df_sorted = df.sort_values("cover_10")
    x = df_sorted["cover_10"].values
    y = df_sorted["weighted_ndvi"].values

    # Fit LOWESS curve
    lowess = sm.nonparametric.lowess
    z = lowess(y, x, frac=0.4)
    x_lowess = z[:, 0]
    y_lowess = z[:, 1]

    init_val = 30.0
    selected = {"value": None}

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    ax.plot(x, y, 'o', alpha=0.4, label='Data', color='skyblue', markersize=6)
    ax.plot(x_lowess, y_lowess, 'r-', linewidth=2, label='LOWESS')

    vline = ax.axvline(init_val, color='gray', linestyle='--', label='Selected')
    ndvi_val = np.interp(init_val, x_lowess, y_lowess)
    ndvi_text = ax.text(0.05, 0.95, f"NDVI = {ndvi_val:.3f}",
                        transform=ax.transAxes, fontsize=12, va='top')

    ax.set_xlabel("Tree Cover (%)")
    ax.set_ylabel("NDVI")
    ax.set_title("Select Tree Cover to Set NE_goal (NDVI)")
    ax.legend()
    ax.grid(True)

    ax_slider = plt.axes([0.1, 0.15, 0.75, 0.05])
    slider = Slider(ax_slider, "Tree Cover", min(x), max(x), valinit=init_val)

    ax_button = plt.axes([0.4, 0.025, 0.2, 0.05])
    button = Button(ax_button, "Confirm")

    def update(val):
        tree_cover = slider.val
        ndvi_est = np.interp(tree_cover, x_lowess, y_lowess)
        vline.set_xdata([tree_cover])
        ndvi_text.set_text(f"NDVI = {ndvi_est:.3f}")
        fig.canvas.draw_idle()

    def on_confirm(event):
        tree_cover = slider.val
        selected["value"] = float(np.interp(tree_cover, x_lowess, y_lowess))
        print(f"Confirmed Tree Cover = {tree_cover:.2f} → NDVI = {selected['value']:.3f}")
        plt.close(fig)  #  Close only the figure created

    slider.on_changed(update)
    button.on_clicked(on_confirm)

    if return_fig:
        #  Wait until user confirms selection
        while selected["value"] is None:
            plt.pause(0.1)

        return selected["value"], fig
    else:
        plt.show()
        return selected["value"] if selected["value"] is not None else 0.3


def load_health_effects(
        excel_file: str,
        health_indicator_i: str,
        baseline_risk=None,
        NE_goal: float = 0.3
        # aoi_gdf: gpd.GeoDataFrame = None,
        # baseline_risk_gdf: gpd.GeoDataFrame = None,
        # risk_col: str = "DEPRESS"
) -> dict:
    """
    Calculate health effects from exposure data, supporting both uniform and geographic risk inputs.

    Parameters:
        excel_file: Path to Excel file containing health effect sizes
        health_indicator_i: Target health indicator (e.g., 'depression')
        baseline_risk: Optional uniform risk value (default: None)
        NE_goal: Target NDVI exposure value (default: 0.3)
        aoi_gdf: GeoDataFrame of study area (required for geographic risk)
        baseline_risk_gdf: GeoDataFrame containing geographic risk values
        risk_col: Column name for risk values in baseline_risk_gdf (default: "DEPRESS")

    Returns:
        Dictionary containing:
        - effect_size_i: Extracted effect size
        - effect_indicator_i: Effect metric type
        - risk_ratio: Calculated risk ratio(s)
        - NE_goal: Target NDVI value
        - baseline_risk: Risk value(s) used
        - aoi_gdf: Study area geometry

    Raises:
        ValueError: For invalid inputs or inconsistent health effect data
    """
    # Load and filter health effect data
    es = pd.read_excel(excel_file, sheet_name="Sheet1", usecols="A:D")
    es_selected = es[es["health_indicator"] == health_indicator_i]

    # Validate health effect data consistency
    effect_size_i = _validate_effect_size(es_selected)
    effect_indicator_i = _validate_effect_indicator(es_selected)

    # Process risk inputs (priority to geographic data)
    baseline_risk = 0.15 if baseline_risk is None else baseline_risk
    risk_ratio = _calculate_risk_ratio(effect_size_i, effect_indicator_i, baseline_risk)
    # === 修改结束 ===

    return {
        "effect_size_i": effect_size_i,
        "effect_indicator_i": effect_indicator_i,
        "risk_ratio": risk_ratio,
        "NE_goal": NE_goal,
        "baseline_risk": baseline_risk,
    }

def _validate_effect_size(es_data: pd.DataFrame) -> float:
    """Validate and extract unique effect size value."""
    values = np.unique(es_data["effect_size"].values)
    if len(values) == 1:
        return values[0]
    raise ValueError("Multiple effect_size values found - specify exact indicator")


def _validate_effect_indicator(es_data: pd.DataFrame) -> str:
    """Validate and extract unique effect indicator type."""
    indicators = np.unique(es_data["effect_indicator"].values)
    if len(indicators) == 1:
        return indicators[0]
    raise ValueError("Multiple effect_indicators found - specify exact metric")


def _calculate_risk_ratio(
        effect_size: float,
        effect_indicator: str,
        baseline_risk
) -> float:
    """
    Calculate risk ratio(s) based on effect metric type.

    Supports both scalar and array inputs for baseline_risk.
    """
    if effect_indicator == "risk ratio":
        return effect_size if isinstance(baseline_risk, float) else np.full_like(baseline_risk, effect_size)

    if effect_indicator == "odd ratio":
        if isinstance(baseline_risk, float):
            return effect_size / (1 - baseline_risk + baseline_risk * effect_size)
        return effect_size / (1 - baseline_risk + baseline_risk * effect_size)

    raise ValueError("effect_indicator must be either 'risk ratio' or 'odd ratio'")

### calculate_pd_layer ###


def calculate_pd_layer(ndvi_raster_path: str, pop_raster_path: str, rr: float,
                       NE_goal: float, baseline_risk_rate: np.ndarray, output_dir: str = ".") -> dict:
    """Calculate PD_i raster. Only considers NDVI improvements (positive delta)."""

    # Load and process NDVI data
    with rasterio.open(ndvi_raster_path) as src:
        ndvi_data = src.read(1).astype(np.float32)
        ndvi_meta = src.meta.copy()
        ndvi_meta.update(dtype="float32", nodata=np.nan)

    ndvi_data[ndvi_data < 0] = np.nan
    if baseline_risk_rate.shape != ndvi_data.shape:
        raise ValueError(f"Risk shape {baseline_risk_rate.shape} != NDVI {ndvi_data.shape}")

    # Calculate NDVI improvement (only positive deltas)
    delta_NE_i = NE_goal - ndvi_data
    delta_positive = np.where(delta_NE_i > 0, delta_NE_i, 0)  # Zero out NDVI decreases

    print(f"NDVI deltas - All: [{np.nanmin(delta_NE_i):.3f}, {np.nanmax(delta_NE_i):.3f}], "
          f"Positive: [{np.nanmin(delta_positive):.3f}, {np.nanmax(delta_positive):.3f}]")
    print(f"Pixels: Improved={np.sum(delta_positive > 0):,}, Decreased={np.sum(delta_NE_i < 0):,}")

    # Risk calculation using only positive improvements
    RR_i = np.exp(np.log(rr) * 10 * delta_positive)
    PF_i = 1 - RR_i

    # Load and process population data
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

    # Calculate health impact (force non-negative)
    PD_i = np.maximum(0, PF_i * baseline_risk_rate * pop_data)

    print(f"Health impact: {np.sum(PD_i > 0):,} pixels, {np.nansum(PD_i):,.0f} total cases")

    # Save outputs
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


### viz_pd_1 ###

def plot_pd_map_v1(PD_raster_path, aoi_gdf, return_fig=False):
    """
    Plot a continuous color-scaled PD_i map with diverging color (red to blue).
    Returns figure if `return_fig=True`.
    """
    # Load and mask PD_i raster
    with rasterio.open(PD_raster_path) as src:
        PD_masked, PD_transform = rasterio.mask.mask(src, aoi_gdf.geometry, crop=True, nodata=src.nodata)
        PD_meta = src.meta.copy()
        PD_meta.update({
            "transform": PD_transform,
            "width": PD_masked.shape[2],
            "height": PD_masked.shape[1]
        })

    PD_flat = PD_masked[0].ravel()
    PD_flat = PD_flat[np.isfinite(PD_flat)]

    if PD_flat.size == 0:
        print("No valid data in PD_i after masking.")
        return None

    min_val = PD_flat.min()
    max_val = PD_flat.max()
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
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_facecolor("white")
    ax.grid(False)


    if return_fig:
        return fig
    else:
        plt.show()

### v3 ###

def plot_pd_map_v3(PD_masked, PD_meta, aoi_gdf, figures_dir, output_name="PD_risk_map_v8_discrete.png",
                   return_fig=False):
    """
    Plot PD_i raster with discrete color bins using quantile-based breakpoints.
    Only shows the outer boundary of the AOI.
    """
    # Flatten and remove NaNs
    PD_masked[0][PD_masked[0] < 0] = 0
    PD_flat = PD_masked[0].ravel()
    PD_flat = PD_flat[np.isfinite(PD_flat)]
    if PD_flat.size == 0:
        raise ValueError("No valid PD_i data in the AOI. Exiting.")

    # Compute quantile-based breakpoints
    quantiles = [0.01, 0.1, 0.25, 0.50, 0.6, 0.7, 0.9, 0.99]
    breakpoints = np.quantile(PD_flat, quantiles)
    breakpoints = np.concatenate([breakpoints, [0]])
    breakpoints = np.unique(np.round(breakpoints, 1))

    if len(breakpoints) == 1:
        breakpoints = [breakpoints[0] - 1e-6, breakpoints[0] + 1e-6]

    num_bins = len(breakpoints) - 1
    vmin = np.min(breakpoints)
    vmax = np.max(breakpoints)
    ncolors = num_bins + 2
    cmap = plt.cm.Blues
    norm = Normalize(vmin=np.min(breakpoints), vmax=np.max(breakpoints))

    # Define extent
    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.set_aspect('equal')

    if hasattr(aoi_gdf, 'unary_union'):

        outline = gpd.GeoSeries([aoi_gdf.unary_union], crs=aoi_gdf.crs)
        outline.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
    else:

        aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

    im = ax.imshow(PD_masked[0], cmap=cmap, norm=norm, extent=pd_extent, origin="upper")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    tick_values = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.1f}" for v in tick_values])

    # format
    ax.set_title("Preventable Depression Cases", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_facecolor("white")

    if return_fig:
        return fig


def safe_raster_write(path, meta, data):

    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed existing file: {path}")
    except Exception as e:
        print(f"Failed to remove existing file: {e}")
        path = path.replace(".tif", "_new.tif")
        print(f"Switching to new output path: {path}")

    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(data, 1)


def create_extended_lowess(x_data, y_data, extend_to_range=(0, 100)):
    """
    Create LOWESS curve extended to full 0-100% tree cover range

    Parameters:
    x_data: original tree cover data
    y_data: original NDVI data
    extend_to_range: tuple (min, max) for extended range

    Returns:
    x_extended, y_extended: extended x and y arrays
    """
    import numpy as np
    import statsmodels.api as sm
    from scipy import interpolate

    # Original LOWESS fit
    lowess_result = sm.nonparametric.lowess(y_data, x_data, frac=0.4)
    x_lowess = lowess_result[:, 0]
    y_lowess = lowess_result[:, 1]

    # Create extended range
    min_extend, max_extend = extend_to_range

    # Handle extrapolation beyond data range
    x_min_data, x_max_data = min(x_data), max(x_data)
    y_min_fit, y_max_fit = y_lowess[0], y_lowess[-1]

    # Estimate slopes at boundaries for linear extrapolation
    if len(x_lowess) >= 2:
        # Slope at lower end
        slope_low = (y_lowess[1] - y_lowess[0]) / (x_lowess[1] - x_lowess[0])
        # Slope at upper end
        slope_high = (y_lowess[-1] - y_lowess[-2]) / (x_lowess[-1] - x_lowess[-2])
    else:
        slope_low = slope_high = 0

    # Create extended x range
    x_extended = np.linspace(min_extend, max_extend, 200)

    # Interpolate/extrapolate y values
    y_extended = np.zeros_like(x_extended)

    for i, x_val in enumerate(x_extended):
        if x_val < x_min_data:
            # Linear extrapolation below data range
            y_extended[i] = y_min_fit + slope_low * (x_val - x_min_data)
        elif x_val > x_max_data:
            # Linear extrapolation above data range
            y_extended[i] = y_max_fit + slope_high * (x_val - x_max_data)
        else:
            # Interpolation within data range
            y_extended[i] = np.interp(x_val, x_lowess, y_lowess)

    # Ensure NDVI values stay within reasonable bounds (0 to 1)
    y_extended = np.clip(y_extended, 0.0, 1.0)

    return x_extended, y_extended