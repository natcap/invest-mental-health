# # Function for NDVI_Population_Weighted calculation

import os
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box


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
    aoi_gdf["ndvi_delta"] = np.array(ndvi_means) - ne_goal_value

    # Set color normalization centered at 0 (deviation = 0)
    delta_min = np.nanmin(aoi_gdf["ndvi_delta"])
    delta_max = np.nanmax(aoi_gdf["ndvi_delta"])
    max_abs = max(abs(delta_min), abs(delta_max))

    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    cmap = plt.cm.RdYlGn  # Red-White-Green gradient, smoother

    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
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