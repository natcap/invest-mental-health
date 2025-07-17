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
    df_stats["GEOID"] = aoi_adm2_raw_clipped["GEOID"].values
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
    z = sm.nonparametric.lowess(y, x, frac=0.4)
    x_lowess = z[:, 0]
    y_lowess = z[:, 1]

    slider_fig, ax = plt.subplots(figsize=(12, 4))
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

def run_pd_analysis(aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, excel_file,output_dir,
                    NE_goal, aoi_adm2_clipped, x_lowess, y_lowess, cost_value):
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

    # Clean PD_i raster: set values < 0 to NaN and save to a temporary file
    with rasterio.open(PD_raster_path) as src:
        PD_data = src.read(1)
        meta = src.meta.copy()

    # Set negative values to NaN
    PD_data_clean = np.where(PD_data < 0, np.nan, PD_data)
    meta.update({"nodata": np.nan})

    # Save the cleaned raster to disk
    cleaned_raster_path = os.path.join(output_dir, "Prev_case.tif")
    with rasterio.open(cleaned_raster_path, "w", **meta) as dst:
        dst.write(PD_data_clean, 1)

    PD_cost = PD_data_clean * (cost_value / 1000)

    cost_raster_path = os.path.join(output_dir, "Prev_cost_pixel.tif")
    with rasterio.open(cost_raster_path, "w", **meta) as dst:
        dst.write(PD_cost, 1)


    # Perform zonal statistics (sum of PD_cost within each tract)
    zonal_result = zonal_stats(aoi_adm2_clipped, cost_raster_path, stats="sum", nodata=np.nan)

    # Multiply sum by cost value to get preventable cost per tract
    sum_values = [round(stat["sum"], 2) if stat["sum"] is not None else 0 for stat in zonal_result]
    aoi_adm2_clipped["PD_i_sum_cost"] = sum_values

    # Set color normalization and color map
    vmin = aoi_adm2_clipped["PD_i_sum_cost"].min()
    vmax = aoi_adm2_clipped["PD_i_sum_cost"].max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.OrRd

    # Plotting figure
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

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    tick_values = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{val:.0f}" for val in tick_values])

    # Set figure bounds and appearance
    ax.set_xlim(aoi_adm2_clipped.total_bounds[[0, 2]])
    ax.set_ylim(aoi_adm2_clipped.total_bounds[[1, 3]])
    ax.set_title("Total Preventable Cost (1000 USD) by Tract", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
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
    selected_cover = float(np.interp(NE_goal, y_lowess, x_lowess))
    fig_cost_curve = simulate_cost_vs_treecover(
        x_lowess, y_lowess,
        Pop_array, NDVI_array, baseline_risk_raster,
        risk_ratio, PD_masked,selected_cover=selected_cover,cost_value=cost_value
    )
    per_pixel_cases = PD_masked[0]  # already population-weighted preventable depression per pixel
    total_pd_cases = np.nansum(per_pixel_cases)  # sum across valid pixels

    return fig1, fig2, fig_hist, fig_cost_curve, total_pd_cases

def simulate_cost_vs_treecover(x_lowess, y_lowess, Pop_array, baseline_ndvi_raster, baseline_risk_raster, risk_ratio, PD_masked=None, selected_cover=None, cost_value=11000):
    """
    Simulate total cost under different Tree Cover (%) choices, using updated PD_i formula:
    PD_i = (1 - exp(ln(RR) * 10 * (NDVI_goal - NDVI_actual))) * baseline_risk * population
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # tree_cover_range = np.linspace(min(x_lowess), max(x_lowess), 50)
    start = (min(x_lowess) // 5) * 5
    end = ((max(x_lowess) + 4) // 5) * 5 + 5
    tree_cover_range = list(np.arange(start, end + 1, 5))

    total_costs = []

    cumulative_cost = 0
    highlight_cost = None  # for selected_cover red dot

    for tree_cover in tree_cover_range:
        ndvi_goal = np.interp(tree_cover, x_lowess, y_lowess)
        delta_ndvi = ndvi_goal - baseline_ndvi_raster
        RR_i = np.exp(np.log(risk_ratio) * 10 * delta_ndvi)
        PF_i = 1 - RR_i
        PD_i = PF_i * baseline_risk_raster * Pop_array
        PD_i = np.where(PD_i > 0, PD_i, 0)



        cost = np.nansum(PD_i) * (cost_value / 1000)
        if selected_cover == tree_cover:
            highlight_cost=cost
        # cumulative_cost += cost
        # total_costs.append(cumulative_cost)
        total_costs.append(cost)

        # If this point is closest to selected_cover, record it
        if selected_cover is not None and abs(tree_cover - selected_cover) < (tree_cover_range[1] - tree_cover_range[0]) / 2:
            highlight_cost = cost

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.plot(tree_cover_range, total_costs, marker='o', color='steelblue')
    ax.set_xlabel("Tree Cover (%)", fontsize=10)
    ax.set_ylabel("Total Preventable Depression (Sum × Pop)", fontsize=10)

    # Optional annotation from final PD_i
    if PD_masked is not None:
        PD_values = PD_masked[0]
        mask = PD_values > 0
        total_PD_gt0 = np.nansum(PD_values[mask])
        # total_PD_gt0 = np.nansum(PD_values[mask] * Pop_array[mask])
        print(f"PD_i > 0 pixel number: {np.count_nonzero(mask)}")
        print(f"PD_i > 0 total value（×Pop）: {total_PD_gt0:,.0f}")

    ax.set_title("Total Preventable Cost(1000 USD) by Tree Cover Gradient", fontsize=12)
    ax.grid(True)
    ax.tick_params(labelsize=9)

    if selected_cover is not None and highlight_cost is not None:
        ax.plot(selected_cover, highlight_cost, 'ro', markersize=8, label="Selected Target")
        ax.legend()

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
    """Compute zonal statistics (categorical) for each census tract."""
    stats = zonal_stats(aoi_shapefile, raster_path, categorical=True, nodata=-9999)
    # aoi_shapefile = aoi_shapefile.rename(columns={"geoid10": "GEOID"})
    # Get unique land cover classes
    unique_classes = sorted(set(key for stat in stats for key in stat.keys()))

    # Compute percentages for each census tract
    percent_stats = []
    for stat in stats:
        total_pixels = sum(stat.values())
        tract_percent = {
            f"cover_{lc}": (stat.get(lc, 0) / total_pixels * 100) if total_pixels > 0 else 0
            for lc in unique_classes
        }
        percent_stats.append(tract_percent)

    df = pd.DataFrame(percent_stats)

    print("Columns in aoi shapefile:", aoi_shapefile.columns)

    df["GEOID"] = aoi_shapefile["GEOID"].values  # Ensure merge compatibility
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
    """
    Plot AOI blocks colored by NDVI deviation from NE_goal.
    Uses smooth Red→White→Green continuous colormap.

    Args:
        ndvi_resampled_path (str): Path to NDVI raster (already resampled).
        aoi_gdf (GeoDataFrame): Census tracts (adm2) geometry.
        ne_goal_value (float): NE_goal threshold for NDVI.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """

    # Open NDVI raster
    with rasterio.open(ndvi_resampled_path) as src:
        ndvi_crs = src.crs

    # Reproject AOI if needed
    if aoi_gdf.crs != ndvi_crs:
        aoi_gdf = aoi_gdf.to_crs(ndvi_crs)

    # Calculate NDVI mean for each block
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

    # Add deviation from NE_goal
    aoi_gdf = aoi_gdf.copy()
    aoi_gdf["ndvi_delta"] = ne_goal_value - np.array(ndvi_means)

    # Set color normalization centered at 0 (deviation = 0)
    delta_min = np.nanmin(aoi_gdf["ndvi_delta"])
    delta_max = np.nanmax(aoi_gdf["ndvi_delta"])
    max_abs = max(abs(delta_min), abs(delta_max))

    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    cmap = plt.cm.RdYlGn  # Red-White-Green gradient, smoother

    fig, ax = plt.subplots(figsize=(6,4.5), constrained_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(aoi_gdf.total_bounds[[0, 2]])
    ax.set_ylim(aoi_gdf.total_bounds[[1, 3]])

    aoi_gdf.plot(
        ax=ax,
        column="ndvi_delta",
        cmap=cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.8
    )

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("NDVI Deviation from NE_goal", fontsize=10)
    cbar.set_ticks(np.linspace(-max_abs, max_abs, 7))
    cbar.ax.set_yticklabels([f"{t:+.1f}" for t in np.linspace(-max_abs, max_abs, 7)], fontsize=9)

    # style
    ax.set_title("NDVI_target – NDVI_baseline", fontsize=12)
    # ax.set_title(f"{ne_goal_value:.2f} = NE_goal vs NDVI", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.set_facecolor("white")
    print("NDVI Plot Bounds:", aoi_gdf.total_bounds)
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

### calculate_pd_load_input ###

def load_baseline_risk_from_shapefile(
        shapefile_path: str,
        geoid_col: str = "GEOID",
        risk_col: str = "DEPRESS",
        default_risk: float = 0.15,
        target_crs: str = "EPSG:5070"
) -> gpd.GeoDataFrame:
    """
    Load baseline risk data from a shapefile with geographic attributes.

    Args:
        shapefile_path: Path to the shapefile containing risk data
        geoid_col: Column name for geographic identifiers (default: "GEOID")
        risk_col: Column name containing risk values (default: "DEPRESS")
        default_risk: Default value to fill missing risk values (default: 0.15)
        target_crs: Target coordinate reference system (default: "EPSG:5070")

    Returns:
        GeoDataFrame containing processed GEOID and baseline risk data

    Raises:
        ValueError: If required columns are missing in the shapefile
    """
    # Read shapefile into GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Validate required columns
    if geoid_col not in gdf.columns:
        raise ValueError(f"Shapefile missing required column: {geoid_col}")
    if risk_col not in gdf.columns:
        raise ValueError(f"Shapefile missing required column: {risk_col}")

    # Standardize coordinate reference system
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    # Ensure GEOID is string type for consistent merging
    gdf[geoid_col] = gdf[geoid_col].astype(str)

    # Fill missing risk values with default
    gdf[risk_col] = gdf[risk_col].fillna(default_risk)

    return gdf[[geoid_col, risk_col]]


def load_health_effects(
        excel_file: str,
        health_indicator_i: str,
        baseline_risk=None,
        NE_goal: float = 0.3,
        aoi_gdf: gpd.GeoDataFrame = None,
        baseline_risk_gdf: gpd.GeoDataFrame = None,
        risk_col: str = "DEPRESS"
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
    if baseline_risk_gdf is not None:
        if aoi_gdf is None:
            raise ValueError("aoi_gdf required for geographic risk matching")
        risk_values = _match_risk_values(aoi_gdf, baseline_risk_gdf, risk_col)
        risk_ratio = _calculate_risk_ratio(effect_size_i, effect_indicator_i, risk_values)
    else:
        baseline_risk = 0.15 if baseline_risk is None else baseline_risk
        risk_ratio = _calculate_risk_ratio(effect_size_i, effect_indicator_i, baseline_risk)

    return {
        "effect_size_i": effect_size_i,
        "effect_indicator_i": effect_indicator_i,
        "risk_ratio": risk_ratio,
        "NE_goal": NE_goal,
        "baseline_risk": baseline_risk if baseline_risk_gdf is None else risk_values,
        "aoi_gdf": aoi_gdf
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


def _match_risk_values(
        aoi_gdf: gpd.GeoDataFrame,
        risk_gdf: gpd.GeoDataFrame,
        risk_col: str
) -> np.ndarray:
    """Match risk values to study areas using GEOID."""
    aoi_gdf["GEOID"] = aoi_gdf["GEOID"].astype(str)
    merged = aoi_gdf.merge(risk_gdf, on="GEOID", how="left")
    return merged[risk_col].fillna(0.15).values  # Fill unmatched areas with default


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

def calculate_pd_layer(
        ndvi_raster_path: str,
        pop_raster_path: str,
        rr: float,
        NE_goal: float,
        baseline_risk_rate: np.ndarray,
        output_dir: str = "."
) -> dict:
    """
    Calculate Preventable Disease Impact (PD_i) raster from environmental and population data.

    Parameters:
        ndvi_raster_path: Path to NDVI raster (geotiff)
        pop_raster_path: Path to population density raster (geotiff)
        rr: Risk ratio (RR_0.1NE) from health studies
        NE_goal: Target NDVI value for health benefit
        baseline_risk_rate: Baseline disease prevalence raster (must match NDVI dimensions)
        output_dir: Output directory path (default: current directory)

    Returns:
        Dictionary containing:
        - PD_i: Preventable Disease Impact raster (numpy array)
        - PF_i: Preventable Fraction raster
        - RR_i: Risk Ratio raster
        - delta_NE_i: NDVI deficit raster
        - output_paths: Dictionary of saved raster paths

    Raises:
        ValueError: If input rasters have incompatible dimensions
        RuntimeError: If raster processing fails
    """

    # 1. NDVI Data Preparation
    try:
        with rasterio.open(ndvi_raster_path) as src:
            ndvi_data = src.read(1).astype(np.float32)
            ndvi_meta = src.meta.copy()
            ndvi_meta.update(dtype="float32", nodata=np.nan)

        # Mask invalid NDVI values (<0 typically indicates no data)
        ndvi_data[ndvi_data < 0] = np.nan

        # Validate baseline risk dimensions
        if baseline_risk_rate.shape != ndvi_data.shape:
            raise ValueError(
                f"Baseline risk shape {baseline_risk_rate.shape} "
                f"doesn't match NDVI {ndvi_data.shape}"
            )
    except Exception as e:
        raise RuntimeError(f"NDVI processing failed: {str(e)}")

    # 2. NDVI Deficit Calculation
    delta_NE_i = NE_goal - ndvi_data

    # 3. Risk Reduction Modeling
    RR_i = np.exp(np.log(rr) * 10 * delta_NE_i)  # Convert RR per 0.1 NDVI to actual delta
    PF_i = 1 - RR_i  # Preventable fraction

    # 4. Population Data Processing
    try:
        with rasterio.open(pop_raster_path) as src:
            population_data = src.read(1).astype(np.float32)
            pop_meta = src.meta.copy()

        # Resample population if needed (using bilinear interpolation)
        if population_data.shape != ndvi_data.shape:
            population_resampled = np.empty_like(ndvi_data, dtype=np.float32)
            reproject(
                source=population_data,
                destination=population_resampled,
                src_transform=pop_meta["transform"],
                src_crs=pop_meta["crs"],
                dst_transform=ndvi_meta["transform"],
                dst_crs=ndvi_meta["crs"],
                resampling=Resampling.bilinear,
            )
            population_data = population_resampled

        # Mask non-populated areas
        population_data[population_data <= 0] = np.nan
    except Exception as e:
        raise RuntimeError(f"Population data processing failed: {str(e)}")

    # 5. Health Impact Calculation
    PD_i = PF_i * baseline_risk_rate * population_data

    # 6. Output Generation
    output_paths = {}
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Save intermediate rasters
        for name, data in [
            ("ndvi_masked", ndvi_data),
            ("delta_NE_i", delta_NE_i),
            ("PD_i", PD_i)
        ]:
            path = os.path.join(output_dir, f"{name}.tif")
            with rasterio.open(path, "w", **ndvi_meta) as dst:
                dst.write(data, 1)
            output_paths[name] = path
            print(f"Saved {name} raster: {path}")

    except Exception as e:
        raise RuntimeError(f"Output saving failed: {str(e)}")

    return {
        "PD_i": PD_i,
        "PF_i": PF_i,
        "RR_i": RR_i,
        "delta_NE_i": delta_NE_i,
        "output_paths": output_paths
    }


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

def plot_pd_map_v3(PD_masked, PD_meta, aoi_gdf, figures_dir, output_name="PD_risk_map_v8_discrete.png", return_fig=False):
    """
    Plot PD_i raster with discrete color bins using quantile-based breakpoints.

    Parameters:
        PD_masked (np.ndarray): Masked PD_i raster array.
        PD_meta (dict): Raster metadata.
        aoi_gdf (GeoDataFrame): AOI geometry for overlay.
        figures_dir (str): Directory to save the figure.
        output_name (str): Output filename.
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
    print("Breakpoints:", breakpoints)

    if len(breakpoints) == 1:
        breakpoints = [breakpoints[0] - 1e-6, breakpoints[0] + 1e-6]

    num_bins = len(breakpoints) - 1
    vmin = np.min(breakpoints)
    vmax = np.max(breakpoints)
    ncolors = num_bins + 2  # Ensure enough colors for 'extend="both"'
    cmap = plt.cm.Blues
    norm = Normalize(vmin=np.min(breakpoints), vmax=np.max(breakpoints))
    print(PD_flat)

    # Define extent
    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(6,4.5), constrained_layout=True)
    ax.set_aspect('equal')
    im = ax.imshow(PD_masked[0], cmap=cmap, norm=norm, extent=pd_extent, origin="upper")
    aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
    print("PD Plot Extent:", pd_extent)

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

    plt.grid(False)

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, output_name)
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