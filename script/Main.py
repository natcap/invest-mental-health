import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable ,get_cmap
import rasterio
import rasterio.mask
from rasterstats import zonal_stats
from rasterio.features import rasterize
from scipy.interpolate import UnivariateSpline

from NDVI_PW import *
from Tree_Cover import *
from Result import *
from calculate_pd_load_input import (
    load_health_effects,
    load_baseline_risk_from_shapefile
)
from calculate_pd_layer import calculate_pd_layer
from viz_pd_1 import plot_pd_map_v1
from viz_pd_2 import plot_pd_map_v2
from viz_pd_3 import plot_pd_map_v3


def run_ndvi_tree_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, excel_file, output_dir):
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import rasterio
    import statsmodels.api as sm
    from rasterio.enums import Resampling

    from NDVI_PW import reproject_shapefile, reproject_raster, clip_raster, resample_raster_to_target, calculate_weighted_ndvi
    from Tree_Cover import compute_zonal_statistics
    from Result import merge_ndvi_landcover_data

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
    pop_dst_path = pop_path.replace("_setnull", "")
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

    ndvi_fig, ax = plt.subplots(figsize=(6, 5))
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
    df_stats["GEOID"] = aoi_adm2_raw_clipped["GEOID"].values  # 保证对齐
    aoi_joined = aoi_adm2_raw_clipped.merge(df_stats, on="GEOID")

    tree_fig, ax = plt.subplots(figsize=(8, 4))
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
    z = sm.nonparametric.lowess(y, x, frac=0.4)
    x_lowess = z[:, 0]
    y_lowess = z[:, 1]

    slider_fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, y, 'o', alpha=0.4, label='Data', color='skyblue')
    ax.plot(x_lowess, y_lowess, 'r-', linewidth=2, label='LOWESS')
    ax.set_xlabel("Tree Cover (%)")
    ax.set_ylabel("NDVI")
    ax.set_title("Select Tree Cover to Set NE_goal (NDVI)")
    ax.grid(True)
    ax.legend()
    slider_fig.tight_layout()

    NE_goal = 0.3  # placeholder
    return NE_goal, ndvi_fig, tree_fig, slider_fig, x_lowess, y_lowess, aoi_adm2_clipped, ndvi_resampled_path

def run_pd_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, excel_file, output_dir,
                    NE_goal, aoi_adm2_clipped, x_lowess, y_lowess):
    target_crs = "EPSG:5070"

    # Load inputs
    aoi_adm1 = gpd.read_file(aoi_adm1_path)
    if aoi_adm1.crs != target_crs:
        aoi_adm1 = aoi_adm1.to_crs(target_crs)

    risk_gdf = load_baseline_risk_from_shapefile(shapefile_path=risk_path, risk_col="DEPRESS")
    result = load_health_effects(excel_file=excel_file, health_indicator_i="depression",
                                 baseline_risk_gdf=risk_gdf, aoi_gdf=aoi_adm1, NE_goal=NE_goal)

    ndvi_raster_path = ndvi_path.replace(".tif", "_prj_clipped_100m.tif")
    pop_raster_path = pop_path.replace("_setnull.tif", "_clipped.tif")

    with rasterio.open(ndvi_raster_path) as ndvi_src:
        NDVI_array = ndvi_src.read(1)
        ndvi_meta = ndvi_src.meta

    with rasterio.open(pop_raster_path) as pop_src:
        Pop_array = pop_src.read(1)

    baseline_risk_raster = np.full(NDVI_array.shape, 0.15, dtype=np.float32)
    if hasattr(risk_gdf, 'geometry'):
        baseline_risk_raster = rasterize(
            shapes=[(geom, value) for geom, value in zip(risk_gdf.geometry, risk_gdf["DEPRESS"])],
            out_shape=NDVI_array.shape,
            transform=ndvi_meta["transform"],
            fill=0.0,
            dtype=np.float32
        )

    # Calculate initial PD layer
    results = calculate_pd_layer(
        ndvi_raster_path=ndvi_raster_path,
        pop_raster_path=pop_raster_path,
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

    fig1 = plot_pd_map_v1(PD_raster_path, aoi_adm2_clipped, return_fig=True)
    fig2 = plot_pd_map_v3(PD_masked=PD_masked, PD_meta=PD_meta, aoi_gdf=aoi_adm1, figures_dir=output_dir,
                          return_fig=True)

    # Histogram plotting
    pd_values = PD_masked[0].ravel()
    pd_values = pd_values[np.isfinite(pd_values)]  # remove NaNs
    pd_values[pd_values < 0] = 0  # If PD value < 0, set to 0
    pd_values_scaled = np.round(pd_values * 1000).astype(int)
    unique, counts = np.unique(pd_values_scaled, return_counts=True)
    histogram_dict = dict(zip(unique, counts))

    zonal_result = zonal_stats(aoi_adm2_clipped, PD_raster_path, stats="mean", nodata=0)
    mean_values = [round(stat["mean"] * 1000, 2) if stat["mean"] is not None else 0 for stat in zonal_result]
    aoi_adm2_clipped["PD_i_avg_x1000"] = mean_values

    vmin = aoi_adm2_clipped["PD_i_avg_x1000"].min()
    vmax = aoi_adm2_clipped["PD_i_avg_x1000"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.OrRd

    fig_hist, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.set_aspect('equal')

    aoi_adm2_clipped.plot(
        column="PD_i_avg_x1000",
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
    cbar.set_ticklabels([f"{val / 1000:.0f}" for val in tick_values])
    # cbar.set_label("Cost(1000 USD)", fontsize=10)
    ax.set_xlim(aoi_adm2_clipped.total_bounds[[0, 2]])
    ax.set_ylim(aoi_adm2_clipped.total_bounds[[1, 3]])

    ax.set_title("Total Preventable Cost by Tract", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")

    # # Prepare for simulate_cost_vs_treecover
    # with rasterio.open(ndvi_path.replace(".tif", "_prj_clipped_100m.tif")) as ndvi_src:
    #     raster_meta = ndvi_src.meta.copy()
    #
    # Pop_array = rasterio.open(pop_path.replace("_setnull.tif", "_clipped.tif")).read(1)
    # baseline_risk_raster = baseline_risk_raster  # already calculated
    # risk_ratio = result["risk_ratio"]

    # read raster_meta
    with rasterio.open(ndvi_path.replace(".tif", "_prj_clipped_100m.tif")) as ndvi_src:
        raster_meta = ndvi_src.meta.copy()

    # read population
    with rasterio.open(pop_path.replace("_setnull.tif", "_clipped.tif")) as pop_src:
        Pop_array = pop_src.read(1)
        pop_nodata = pop_src.nodata

        # ingore negative value
        if pop_nodata is not None:
            Pop_array = np.where(Pop_array == pop_nodata, 0, Pop_array)
        else:
            Pop_array = np.where(Pop_array < -1e30, 0, Pop_array)


        Pop_array = np.where(Pop_array > 0, Pop_array, 0)

    # verify
    print("Cleaned Pop_array stats:", Pop_array.min(), Pop_array.max())
    risk_ratio = result["risk_ratio"]
    # Create cost vs tree cover curve
    # fig_cost_curve = simulate_cost_vs_treecover(
    #     x_lowess, y_lowess,
    #     Pop_array, baseline_risk_raster, raster_meta, risk_ratio,
    #     PD_masked=PD_masked
    # )
    fig_cost_curve = simulate_cost_vs_treecover(
        x_lowess, y_lowess,
        Pop_array, NDVI_array, baseline_risk_raster,
        risk_ratio, PD_masked
    )

    return fig1, fig2, fig_hist, fig_cost_curve

def simulate_cost_vs_treecover(x_lowess, y_lowess, Pop_array, baseline_ndvi_raster, baseline_risk_raster, risk_ratio, PD_masked=None):
    """
    Simulate total cost under different Tree Cover (%) choices, using updated PD_i formula:
    PD_i = (1 - exp(ln(RR) * 10 * (NDVI_goal - NDVI_actual))) * baseline_risk * population
    """
    import numpy as np
    import matplotlib.pyplot as plt

    tree_cover_range = np.linspace(min(x_lowess), max(x_lowess), 50)
    total_costs = []
    cumulative_cost = 0

    for tree_cover in tree_cover_range:
        # print(tree_cover)
        ndvi_goal = np.interp(tree_cover, x_lowess, y_lowess)
        delta_ndvi = ndvi_goal - baseline_ndvi_raster
        RR_i = np.exp(np.log(risk_ratio) * 10 * delta_ndvi)
        PF_i = 1 - RR_i
        PD_i = PF_i * baseline_risk_raster * Pop_array
        PD_i = np.where(PD_i > 0, PD_i, 0)
        cumulative_cost += np.nansum(PD_i)
        total_costs.append(cumulative_cost/1000)
        # print(total_costs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(tree_cover_range, total_costs, marker='o', color='steelblue')
    ax.set_xlabel("Tree Cover (%)", fontsize=10)
    ax.set_ylabel("Total Preventable Depression (Sum × Pop)", fontsize=10)

    # Optional annotation from final PD_i
    subtitle = ""
    if PD_masked is not None:
        PD_values = PD_masked[0]
        mask = PD_values > 0
        total_PD_gt0 = np.nansum(PD_values[mask] * Pop_array[mask])
        # subtitle = f"\nActual Total for PD_i > 0: {total_PD_gt0:,.0f}"
        print(f"PD_i > 0 的像素数量: {np.count_nonzero(mask)}")
        print(f"PD_i > 0 的总值（×Pop）: {total_PD_gt0:,.0f}")

    ax.set_title(f"Total Preventable Cost (1000 USD) by Tree Cover Gradient{subtitle}", fontsize=12)
    ax.grid(True)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    return fig

