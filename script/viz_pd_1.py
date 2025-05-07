import rasterio
import rasterio.mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

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

    fig, ax = plt.subplots(figsize=(6, 5))
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
