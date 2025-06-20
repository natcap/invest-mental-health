import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as ResampleEnum
from rasterio.transform import array_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from typing import Tuple, Dict, Union, List
from Treecover_approach import load_health_effects


class NDVI_LULC_Analyzer:
    def __init__(self, aoi_path: str, output_dir: str = "output_ndvi_lulc"):
        """
        Initialize the NDVI-LULC analyzer with AOI path and output directory.

        Args:
            aoi_path: Path to the Area of Interest (AOI) shapefile
            output_dir: Directory to save output files
        """
        self.aoi_path = aoi_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures = []

        # Load AOI boundary
        self.aoi_gdf = gpd.read_file(self.aoi_path).to_crs("EPSG:5070")

    def clip_raster(self, raster_path: str, boundary_geom) -> Tuple[np.ndarray, dict]:
        """
        Clip a raster to the specified boundary geometry.

        Args:
            raster_path: Path to input raster file
            boundary_geom: Shapely geometry to use for clipping

        Returns:
            Tuple of (clipped array, metadata)
        """
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rio_mask(src, [boundary_geom], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
        return out_image.squeeze(), out_meta

    def align_ndvi_to_nlcd(self, ndvi_path: str, nlcd_meta: dict) -> np.ndarray:
        """
        Reproject and align NDVI raster to match NLCD raster specifications.

        Args:
            ndvi_path: Path to NDVI raster
            nlcd_meta: Metadata dictionary from NLCD raster

        Returns:
            Reprojected NDVI array
        """
        with rasterio.open(ndvi_path) as src:
            ndvi_reprojected = np.zeros((nlcd_meta['height'], nlcd_meta['width']), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=ndvi_reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=nlcd_meta['transform'],
                dst_crs = "EPSG:5070",
                resampling=ResampleEnum.bilinear,
                dst_nodata=np.nan
            )
        return ndvi_reprojected

    def calculate_mean_ndvi(self, ndvi: np.ndarray, nlcd_data: np.ndarray, nlcd_nodata: int) -> Dict[int, float]:
        """
        Calculate mean NDVI for each NLCD class.

        Args:
            ndvi: NDVI array
            nlcd_data: NLCD land cover array
            nlcd_nodata: NLCD no-data value

        Returns:
            Dictionary mapping NLCD codes to mean NDVI values
        """
        valid_classes = np.unique(nlcd_data[(nlcd_data != nlcd_nodata) & (nlcd_data > 0)])
        results = {}
        for cls in valid_classes:
            mask = (nlcd_data == cls) & (~np.isnan(ndvi))
            results[cls] = np.nanmean(ndvi[mask]) if np.any(mask) else np.nan
        return results

    def process_nlcd_with_ndvi(self, nlcd_file: str, ndvi_file: str, output_name: str) -> Tuple[
        pd.DataFrame, np.ndarray, dict, gpd.GeoDataFrame]:
        """
        Process NLCD and NDVI data to calculate mean NDVI by land cover class.

        Args:
            nlcd_file: Path to NLCD raster
            ndvi_file: Path to NDVI raster
            output_name: Base name for output files

        Returns:
            Tuple containing:
            - DataFrame with mean NDVI by class
            - Clipped NLCD array
            - NLCD metadata
            - Projected AOI GeoDataFrame
        """
        with rasterio.open(nlcd_file) as nlcd_src:
            boundary_proj = self.aoi_gdf.to_crs(nlcd_src.crs)
            boundary_geom = boundary_proj.geometry.values[0]

        clipped_nlcd, nlcd_meta = self.clip_raster(nlcd_file, boundary_geom)
        nlcd_nodata = nlcd_meta.get('nodata', 0)
        ndvi_aligned = self.align_ndvi_to_nlcd(ndvi_file, nlcd_meta)
        mean_ndvi = self.calculate_mean_ndvi(ndvi_aligned, clipped_nlcd, nlcd_nodata)

        df = pd.DataFrame({'NLCD_Code': list(mean_ndvi.keys()),
                           'Mean_NDVI': list(mean_ndvi.values())})
        csv_path = os.path.join(self.output_dir, f'mean_ndvi_by_{output_name}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved NDVI by NLCD CSV: {csv_path}")
        return df, clipped_nlcd, nlcd_meta, boundary_proj

    def generate_ndvi_by_nlcd_raster(self, clipped_nlcd: np.ndarray, mean_ndvi_df: pd.DataFrame,
                                     meta: dict, exclude_classes: list = [11, 12, 90, 95]) -> Tuple[np.ndarray, dict]:
        """
        Generate a raster where each pixel has the mean NDVI value for its NLCD class.

        Args:
            clipped_nlcd: Clipped NLCD array
            mean_ndvi_df: DataFrame with mean NDVI values
            meta: Raster metadata
            exclude_classes: List of NLCD classes to exclude

        Returns:
            Tuple of (NDVI raster array, updated metadata)
        """
        ndvi_array = np.full_like(clipped_nlcd, np.nan, dtype=np.float32)
        mean_ndvi = dict(zip(mean_ndvi_df['NLCD_Code'], mean_ndvi_df['Mean_NDVI']))

        for cls, val in mean_ndvi.items():
            if cls not in exclude_classes:
                ndvi_array[clipped_nlcd == cls] = val

        ndvi_array = np.where(np.isnan(ndvi_array), -9999, ndvi_array)
        meta.update({"dtype": "float32", "count": 1, "nodata": -9999})
        return ndvi_array, meta

    def plot_raw_ndvi(self, ndvi_path: str, output_name: str = "raw_ndvi",
                      vmin: float = -1, vmax: float = 1) -> str:

        """
        Load, clip and plot raw NDVI data (before LULC analysis)

        Args:
            ndvi_path: Path to NDVI raster file
            output_name: Base name for output files
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale

        Returns:
            Path to the saved figure
        """

        from matplotlib.colors import Normalize

        # Generate output paths
        fig_path = os.path.join(self.output_dir, f"{output_name}.png")

        # Step 1: Load the NDVI raster
        with rasterio.open(ndvi_path) as ndvi_src:
            ndvi_crs = ndvi_src.crs
            ndvi_meta = ndvi_src.meta.copy()

            # Reproject AOI if needed
            if self.aoi_gdf.crs != ndvi_crs:
                print(f"Reprojecting AOI from {self.aoi_gdf.crs} to {ndvi_crs}")
                aoi_proj = self.aoi_gdf.to_crs(ndvi_crs)
            else:
                aoi_proj = self.aoi_gdf

            # Clip the NDVI
            aoi_geometry = [aoi_proj.geometry.unary_union]
            ndvi_clipped, transform = rio_mask(ndvi_src, aoi_geometry, crop=True)

            # Update metadata
            ndvi_meta.update({
                "height": ndvi_clipped.shape[1],
                "width": ndvi_clipped.shape[2],
                "transform": transform
            })

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Calculate extent
        x_min = transform[2]
        x_max = transform[2] + transform[0] * ndvi_clipped.shape[2]
        y_min = transform[5] + transform[4] * ndvi_clipped.shape[1]
        y_max = transform[5]

        # Plot clipped NDVI
        img = ax.imshow(ndvi_clipped[0],
                        cmap='RdYlGn',
                        norm=norm,
                        extent=[x_min, x_max, y_min, y_max])

        # Add boundary and colorbar
        aoi_proj.boundary.plot(ax=ax, edgecolor='red',
                               linewidth=1, linestyle='--',
                               label='AOI Boundary')
        plt.colorbar(img, label='NDVI')

        # Customize plot
        ax.set_title(f'NDVI within AOI Boundary - {output_name.replace("_", " ").title()}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.axis('off')
        ax.legend()

        # Save and show
        plt.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        self.figures.append(fig)
        print(f"Saved raw NDVI plot: {fig_path}")

        return fig_path

    def save_and_plot_ndvi(self, ndvi_array: np.ndarray, meta: dict, boundary_proj: gpd.GeoDataFrame,
                           filename: str, title: str, cmap: str = 'RdYlGn', vmin: float = -0.8,
                           vmax: float = 0.8) -> str:
        """
        Save NDVI raster and create a visualization plot.

        Args:
            ndvi_array: NDVI array to plot
            meta: Raster metadata
            boundary_proj: Projected boundary GeoDataFrame
            filename: Base name for output files
            title: Plot title
            cmap: Color map to use
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale

        Returns:
            Path to the saved figure
        """

        from matplotlib.colors import Normalize

        # Save raster
        raster_path = os.path.join(self.output_dir, filename + ".tif")
        with rasterio.open(raster_path, "w", **meta) as dst:
            dst.write(ndvi_array, 1)
        print(f"Saved raster: {raster_path}")

        # Create plot
        left, bottom, right, top = array_bounds(meta["height"], meta["width"], meta["transform"])
        fig, ax = plt.subplots(figsize=(10, 6))

        # Mask nodata values for plotting
        ndvi_plot = np.ma.masked_equal(ndvi_array, -9999)

        norm = Normalize(vmin=vmin, vmax=vmax)
        img = ax.imshow(ndvi_plot, cmap=cmap, norm=norm,
                        extent=[left, right, bottom, top], origin='upper')

        boundary_proj.boundary.plot(ax=ax, edgecolor='black',
                                    linewidth=0.5, linestyle='--', label='AOI Boundary')

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Mean NDVI', rotation=270, labelpad=20)
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.axis('off')
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, filename + ".png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig)
        print(f"Saved figure: {fig_path}")

        return fig_path

    def calculate_ndvi_difference(self, ndvi_array1: np.ndarray, ndvi_array2: np.ndarray,
                                  meta: dict) -> Tuple[np.ndarray, dict]:
        """
        Calculate difference between two NDVI rasters.

        Args:
            ndvi_array1: First NDVI array (baseline)
            ndvi_array2: Second NDVI array (scenario)
            meta: Raster metadata

        Returns:
            Tuple of (difference array, updated metadata)
        """
        # Calculate difference while handling nodata values
        ndvi_diff = np.where(
            (ndvi_array1 == -9999) | (ndvi_array2 == -9999),
            -9999,
            ndvi_array2 - ndvi_array1
        )

        meta_diff = meta.copy()
        meta_diff.update({"dtype": "float32", "nodata": -9999})
        return ndvi_diff, meta_diff

    def plot_ndvi_difference(self, ndvi_diff: np.ndarray, meta: dict, boundary_proj: gpd.GeoDataFrame,
                             filename: str, title: str) -> str:
        """
        Create a specialized plot for NDVI difference visualization.

        Args:
            ndvi_diff: NDVI difference array
            meta: Raster metadata
            boundary_proj: Projected boundary GeoDataFrame
            filename: Base name for output files
            title: Plot title

        Returns:
            Path to the saved figure
        """
        # Save difference raster
        diff_path = os.path.join(self.output_dir, filename + ".tif")
        with rasterio.open(diff_path, "w", **meta) as dst:
            dst.write(ndvi_diff, 1)
        print(f"Saved difference raster: {diff_path}")

        # Create plot
        left, bottom, right, top = array_bounds(meta["height"], meta["width"], meta["transform"])
        fig, ax = plt.subplots(figsize=(10, 6))

        # Mask nodata values
        ndvi_diff_plot = np.ma.masked_equal(ndvi_diff, -9999)

        # Calculate symmetric color range
        max_abs = max(abs(np.nanmin(ndvi_diff_plot)), abs(np.nanmax(ndvi_diff_plot)))
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

        img = ax.imshow(ndvi_diff_plot, cmap='RdYlGn', norm=norm,
                        extent=[left, right, bottom, top], origin='upper')

        boundary_proj.boundary.plot(ax=ax, edgecolor='black',
                                    linewidth=0.5, linestyle='--', label='AOI Boundary')

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('NDVI Change', rotation=270, labelpad=20)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.axis('off')
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, filename + ".png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(fig)
        print(f"Saved difference figure: {fig_path}")

        return fig_path

    def show_all_plots(self):
        """Display all generated plots."""
        plt.show()


def compute_pd_from_ndvi_change(delta_ndvi: np.ndarray,
                                 pop_raster_path: str,
                                 tract_shapefile_path: str,
                                 risk_shapefile_path: str,
                                 health_effect_path: str,
                                 output_dir: str,
                                aoi_path: str,
                                 health_indicator_i: str = "depression",cost_value=10):
    os.makedirs(output_dir, exist_ok=True)
    print(">>> output_dir33333 =", output_dir)

    effects = load_health_effects(health_effect_path, health_indicator_i)
    if isinstance(effects, dict):
        rr0 = effects["risk_ratio"]
    else:
        raise TypeError("Expected effects to be a dict returned by load_health_effects")

    risk_gdf = gpd.read_file(risk_shapefile_path)
    if risk_gdf.crs != "EPSG:5070":
        risk_gdf = risk_gdf.to_crs("EPSG:5070")

    with rasterio.open(pop_raster_path) as src:
        pop_array = src.read(1)
        meta = src.meta.copy()
        meta.update({"crs": "EPSG:5070"})
        transform = src.transform
        crs = src.crs

    delta_meta = meta
    delta_ndvi_aligned = resample_ndvi_to_match(
        delta_ndvi=delta_ndvi,
        delta_meta=delta_meta,
        target_shape=pop_array.shape,
        target_transform=transform,
        target_crs=crs
    )

    common_mask = (pop_array > 0) & (delta_ndvi_aligned != -9999)
    delta_ndvi = np.where(common_mask, delta_ndvi_aligned, np.nan)
    pop_array = np.where(common_mask, pop_array, 0)

    rr_i = np.exp(np.log(rr0) * 10 * delta_ndvi)
    pd_i = (rr_i - 1) / rr_i * pop_array * 0.15
    pd_i = np.nan_to_num(pd_i, nan=0.0, posinf=0.0, neginf=0.0)

    pd_raster_path = os.path.join(output_dir, "pd_i_temp.tif")
    save_raster(pd_i, pd_raster_path, meta, transform, crs)

    tract_gdf = gpd.read_file(tract_shapefile_path).to_crs("EPSG:5070")
    aoi_gdf = gpd.read_file(aoi_path).to_crs("EPSG:5070")
    tract_clipped = gpd.overlay(tract_gdf, aoi_gdf, how="intersection")
    print(">>> output_dir44444444 =", output_dir)
    fig1_path = plot_pd_map_v1(pd_raster_path, risk_gdf, output_dir=output_dir, output_name="Figure_7.png")

    with rasterio.open(pd_raster_path) as src:
        PD_masked, _ = rio_mask(src, tract_clipped.geometry, crop=True, nodata=src.nodata, filled=True)
        PD_masked = np.where(PD_masked == src.nodata, np.nan, PD_masked)
        PD_meta = src.meta.copy()

    # fig2_path = plot_pd_map_v3(PD_masked, PD_meta, tract_clipped, output_dir)
    fig_tract_avg = plot_pd_by_tract_sum(pd_raster_path, tract_clipped, output_dir, output_name="Figure_6.png",cost_value = cost_value)
    print("Saved fig1:", fig1_path)
    # print("Saved fig2:", fig2_path)
    print("CRS of population raster:", crs)
    print("CRS of NDVI delta:", delta_meta['crs'])
    print("CRS of tract shapefile:", tract_gdf.crs)
    print("CRS of AOI shapefile:", aoi_gdf.crs)
    print("CRS of baseline risk shapefile:", risk_gdf.crs)

def save_raster(array, output_path, meta, transform, crs):
    meta.update({
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "transform": transform,
        "crs": crs,
        "dtype": rasterio.float32,
        "count": 1
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(array.astype(np.float32), 1)

def resample_ndvi_to_match(delta_ndvi, delta_meta, target_shape, target_transform, target_crs):
    aligned_ndvi = np.empty(target_shape, dtype=np.float32)
    reproject(
        source=delta_ndvi,
        destination=aligned_ndvi,
        src_transform=delta_meta['transform'],
        src_crs=delta_meta['crs'],
        dst_transform=target_transform,
        dst_crs="EPSG:5070",
        resampling=ResampleEnum.bilinear
    )
    return aligned_ndvi


def plot_pd_map_v1(PD_raster_path, aoi_gdf, return_fig=False, output_dir=None, output_name="PD_by_tract_avg.png"):
    """
    Plot PD_i raster with colormap starting from 0, masking all negative values.
    Only outer AOI boundary is shown.
    Saves or returns figure depending on `return_fig`.
    """
    print(">>> output_dir55555 =", output_dir)
    # import rasterio
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors
    # import geopandas as gpd
    # from shapely.geometry import MultiPolygon
    # from rasterio.mask import mask as rio_mask
    # import os
    print(">> Generating Figure_7 using:", PD_raster_path)
    print(">>> output_dir =", output_dir)
    with rasterio.open(PD_raster_path) as src:
        PD_masked, PD_transform = rio_mask(src, aoi_gdf.geometry, crop=True, nodata=src.nodata, filled=True)
        PD_masked = np.where(PD_masked == src.nodata, np.nan, PD_masked)
        # PD_masked = np.where(PD_masked == 0, 999, PD_masked)
        # PD_masked = np.where(PD_masked < 0, np.nan, PD_masked)




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

    vmin = PD_flat.min()
    vmax = PD_flat.max()
    vcenter =0
    if vmax <= 0:
        vmax = 1e-6

    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

    # redefine set center as white FFFFFFF
    cmap = LinearSegmentedColormap.from_list("custom_RdBu", ["blue", "white", "red"])

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(PD_masked[0], cmap=cmap, norm=norm, extent=pd_extent, origin="upper")


    outer_boundary = aoi_gdf.unary_union
    gpd.GeoSeries(outer_boundary).boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

    ticks = np.linspace(vmin, vmax, 6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PD_i Values", fontsize=9)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks])

    ax.set_title("Preventable Depression Cases", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    # if return_fig:
    #     print("222")
    #     return fig
    if output_dir:
        # print("222")
        os.makedirs(output_dir, exist_ok=True)
        # print("223")
        fig_path = os.path.join(output_dir, output_name)
        # print("222")
        plt.savefig(fig_path, dpi=300)
        print("✅ Saved figure:", fig_path)
        plt.close(fig)
        return fig_path
    else:
        plt.show()

def plot_pd_by_tract_sum(pd_raster_path, tract_gdf, output_dir, output_name="PD_by_tract_sum.png", cost_value=10 ):
    """
    Plot tract-level **sum** of PD_i values using a white-red gradient.
    Tract boundaries are shown, and values start from 0 upward.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import os

    # Compute the sum of values within each tract
    sums = []
    with rasterio.open(pd_raster_path) as src:
        for geom in tract_gdf.geometry:
            try:
                out_image, _ = rio_mask(src, [geom], crop=True)
                data = out_image[0]

                nodata_val = src.nodata
                if nodata_val is not None:
                    data = data[data != nodata_val]


                # if data.size > 0:
                    # print("Min value in this tract:", np.min(data))
                # sums.append(np.nansum(data) if data.size > 0 else np.nan)
                sums.append(np.sum(data)*cost_value/1000)
            except:

                sums.append(np.nan)

    tract_gdf = tract_gdf.copy()
    tract_gdf["PD_sum"] = sums

    # Set color range from 0 to the max value
    vmin = np.nanmin(tract_gdf["PD_sum"])
    vmax = np.nanmax(tract_gdf["PD_sum"])
    if vmax <= 0 or np.isnan(vmax):
        vmax = 1e-6  # fallback to avoid blank map
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    tract_gdf.plot(
        column="PD_sum",
        cmap="coolwarm",
        linewidth=0,
        edgecolor=None,
        norm=norm,
        ax=ax
    )

    # Overlay tract boundaries (in black)
    tract_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Total PD_i per Tract", fontsize=9)

    # Style adjustments
    ax.set_title("Total Preventable Cost (1000 USD) by Tract", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, output_name)
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def force_align_raster_to_reference(input_array: np.ndarray, input_meta: dict,
                                    reference_shape: Tuple[int, int],
                                    reference_transform, reference_crs) -> np.ndarray:
    """
    Force resample input_array to match the spatial layout of a reference raster.

    Args:
        input_array: The input raster array (e.g., NDVI diff)
        input_meta: Metadata of input raster
        reference_shape: Shape (height, width) of the reference raster (e.g., population)
        reference_transform: Affine transform of the reference raster
        reference_crs: CRS of the reference raster

    Returns:
        Resampled array that matches the reference raster
    """
    aligned_array = np.empty(reference_shape, dtype=np.float32)

    reproject(
        source=input_array,
        destination=aligned_array,
        src_transform=input_meta["transform"],
        src_crs=input_meta["crs"],
        dst_transform=reference_transform,
        dst_crs=reference_crs,
        resampling=ResampleEnum.nearest
    )
    return aligned_array

def run_ndvi_option3_pipeline(aoi_path, ndvi_2011_path, ndvi_2021_path,
                               lc_2011_path, lc_2021_path,
                               pop_raster_path, tract_path,
                               risk_path, health_effect_path, output_dir, cost_value):

        analyzer = NDVI_LULC_Analyzer(aoi_path, output_dir)

        # fig 1 & fig 2
        analyzer.plot_raw_ndvi(ndvi_2011_path, "Figure_1")
        analyzer.plot_raw_ndvi(ndvi_2021_path, "Figure_2")

        # fig 3
        df_2011, nlcd_2011, meta_2011, aoi_proj = analyzer.process_nlcd_with_ndvi(lc_2011_path, ndvi_2011_path, "2011")
        ndvi_2011_raster, meta_2011 = analyzer.generate_ndvi_by_nlcd_raster(nlcd_2011, df_2011, meta_2011)
        analyzer.save_and_plot_ndvi(ndvi_2011_raster, meta_2011, aoi_proj, "Figure_3", "NDVI by NLCD - 2011")

        # fig 4
        df_2021, nlcd_2021, meta_2021, _ = analyzer.process_nlcd_with_ndvi(lc_2021_path, ndvi_2021_path, "2021")
        ndvi_2021_raster, meta_2021 = analyzer.generate_ndvi_by_nlcd_raster(nlcd_2021, df_2021, meta_2021)
        analyzer.save_and_plot_ndvi(ndvi_2021_raster, meta_2021, aoi_proj, "Figure_4", "NDVI by NLCD - 2021")

        # fig 5
        ndvi_diff_raw, meta_diff = analyzer.calculate_ndvi_difference(ndvi_2011_raster, ndvi_2021_raster, meta_2011)
        analyzer.plot_ndvi_difference(ndvi_diff_raw, meta_diff, aoi_proj, "Figure_5", "NDVI Change (2021 - 2011)")

        # Align diff to population raster shape
        with rasterio.open(pop_raster_path) as src:
            pop_meta = src.meta.copy()
            ref_shape = (pop_meta["height"], pop_meta["width"])
            ref_transform = src.transform
            ref_crs = src.crs

        ndvi_diff = force_align_raster_to_reference(
            input_array=ndvi_diff_raw,
            input_meta=meta_diff,
            reference_shape=ref_shape,
            reference_transform=ref_transform,
            reference_crs=ref_crs
        )

        # fig 6 & fig 7
        # print(">>> output_dir2222222 =", output_dir)
        compute_pd_from_ndvi_change(
            delta_ndvi=ndvi_diff,
            pop_raster_path=pop_raster_path,
            tract_shapefile_path=tract_path,
            risk_shapefile_path=risk_path,
            health_effect_path=health_effect_path,
            output_dir=output_dir,
            aoi_path=aoi_path,
            health_indicator_i="depression",
            cost_value=cost_value
        )
        # Show all plots
        # analyzer.show_all_plots()

def run_ndvi_option2_pipeline(aoi_path: str,
                              nlcd_2011_path: str,
                              nlcd_2021_path: str,
                              nlcd_attr_table_path: str,
                              pop_raster_path: str,
                              tract_path: str,
                              risk_path: str,
                              health_effect_path: str,
                              output_dir: str,
                              cost_value: float = 10):
    """
    Pipeline for Option 2: Generate NDVI maps from NLCD + attribute table,
    calculate NDVI change, and run PD/cost analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    analyzer = NDVI_LULC_Analyzer(aoi_path, output_dir)

    # read table
    attr_df = pd.read_excel(nlcd_attr_table_path)
    code_to_ndvi = {}
    code_to_mask = {}
    for _, row in attr_df.iterrows():
        code = row["lc_code"]
        ndvi_val = row["lc_ndvi"]
        exclude = row.get("lc_exclude", 0)
        if pd.notna(ndvi_val):
            code_to_ndvi[code] = ndvi_val
            code_to_mask[code] = 0 if exclude == 1 else 1

    #  2011
    with rasterio.open(nlcd_2011_path) as src:
        boundary_proj = analyzer.aoi_gdf.to_crs(src.crs)
        geometry = boundary_proj.geometry.values[0]
        nlcd_2011_clip, meta_2011 = analyzer.clip_raster(nlcd_2011_path, geometry)

    ndvi_2011 = np.full_like(nlcd_2011_clip, -9999, dtype=np.float32)
    mask_2011 = np.zeros_like(nlcd_2011_clip, dtype=np.uint8)
    for code, ndvi_val in code_to_ndvi.items():
        ndvi_2011[nlcd_2011_clip == code] = ndvi_val
    for code, mask_val in code_to_mask.items():
        mask_2011[nlcd_2011_clip == code] = mask_val

    meta_2011.update({"dtype": "float32", "nodata": -9999})
    fig3_path = analyzer.save_and_plot_ndvi(np.where(mask_2011 == 1, ndvi_2011, np.nan), meta_2011, boundary_proj,
                                            "op2_Figure_3", "NDVI from NLCD - 2011")

    # 2021
    with rasterio.open(nlcd_2021_path) as src:
        nlcd_2021_clip, meta_2021 = analyzer.clip_raster(nlcd_2021_path, geometry)

    ndvi_2021 = np.full_like(nlcd_2021_clip, -9999, dtype=np.float32)
    mask_2021 = np.zeros_like(nlcd_2021_clip, dtype=np.uint8)
    for code, ndvi_val in code_to_ndvi.items():
        ndvi_2021[nlcd_2021_clip == code] = ndvi_val
    for code, mask_val in code_to_mask.items():
        mask_2021[nlcd_2021_clip == code] = mask_val

    meta_2021.update({"dtype": "float32", "nodata": -9999})
    fig4_path = analyzer.save_and_plot_ndvi(np.where(mask_2021 == 1, ndvi_2021, np.nan), meta_2021, boundary_proj,
                                            "op2_Figure_4", "NDVI from NLCD - 2021")

    # NDVI diff
    ndvi_diff_raw, meta_diff = analyzer.calculate_ndvi_difference(ndvi_2011, ndvi_2021, meta_2011)
    mask_common = (mask_2011 == 1) & (mask_2021 == 1)
    fig5_path = analyzer.plot_ndvi_difference(np.where(mask_common, ndvi_diff_raw, np.nan), meta_diff, boundary_proj,
                                              "op2_Figure_5", "NDVI Change (2021 - 2011)")

    # population
    with rasterio.open(pop_raster_path) as src:
        pop_meta = src.meta.copy()
        ref_shape = (pop_meta["height"], pop_meta["width"])
        ref_transform = src.transform
        ref_crs = src.crs

    ndvi_diff = force_align_raster_to_reference(
        input_array=ndvi_diff_raw,
        input_meta=meta_diff,
        reference_shape=ref_shape,
        reference_transform=ref_transform,
        reference_crs=ref_crs
    )

    # PD analysis
    compute_pd_from_ndvi_change(
        delta_ndvi=ndvi_diff,
        pop_raster_path=pop_raster_path,
        tract_shapefile_path=tract_path,
        risk_shapefile_path=risk_path,
        health_effect_path=health_effect_path,
        output_dir=output_dir,
        aoi_path=aoi_path,
        health_indicator_i="depression",
        cost_value=cost_value
    )

if __name__ == "__main__":
    run_ndvi_option3_pipeline()


