import rasterio
import rasterio.mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

def plot_pd_map_v2(
    PD_raster_path,
    aoi_gdf,
    figures_dir,
    percentile_upper=99,
    percentile_lower=1,
    pos_steps=4,
    neg_steps=1,
    output_name="PD_risk_map_v7.1.png"
):
    """
    Plot Preventable Depression (PD_i) with discrete color breaks using quantiles.

    Parameters:
        PD_raster_path (str): Path to the PD_i raster file.
        aoi_gdf (GeoDataFrame): Area of interest geometry.
        figures_dir (str): Output directory to save the plot.
        percentile_upper (int): Upper percentile cap for positive values (default: 99).
        percentile_lower (int): Lower percentile cap for negative values (default: 1).
        pos_steps (int): Number of positive quantile ticks.
        neg_steps (int): Number of negative linear intervals.
        output_name (str): Output filename.
    """
    with rasterio.open(PD_raster_path) as src:
        PD_masked, PD_transform = rasterio.mask.mask(src, aoi_gdf.geometry, crop=True, nodata=src.nodata)
        PD_meta = src.meta.copy()
        PD_meta.update({"transform": PD_transform, "width": PD_masked.shape[2], "height": PD_masked.shape[1]})

    # Flatten and clean
    PD_flat = PD_masked[0].ravel()
    PD_flat = PD_flat[np.isfinite(PD_flat)]
    if PD_flat.size == 0:
        print("No valid PD_i data found.")
        return

    # Compute percentile-based range
    min_val = np.percentile(PD_flat, percentile_lower)
    max_val = np.percentile(PD_flat, percentile_upper)

    vcenter = 0 if min_val < 0 < max_val else (min_val + max_val) / 2
    if min_val >= max_val:
        max_val = min_val + 1e-6

    norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=vcenter, vmax=max_val)
    cmap = plt.cm.RdBu

    # Extent for plotting
    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(PD_masked[0], cmap=cmap, norm=norm, extent=pd_extent, origin="upper")
    aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, label="AOI Boundary")

    # Generate ticks
    pos_vals = PD_flat[PD_flat > 0]
    if len(pos_vals) > 0:
        q = np.linspace(0, percentile_upper, pos_steps + 1)
        pos_ticks = np.percentile(pos_vals, q)
        pos_ticks = np.unique(np.concatenate(([0], pos_ticks, [max_val])))
    else:
        pos_ticks = np.array([0])

    neg_ticks = np.linspace(min_val, 0, neg_steps + 1, endpoint=False)
    ticks = np.concatenate((neg_ticks, pos_ticks))
    ticks = np.unique(np.round(ticks, 2))

    print("Tick values:", ticks)

    cbar = plt.colorbar(im, ax=ax, extend="both")
    cbar.set_label("PD_i Values (red=negative, blue=positive)", fontsize=10)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks])

    # Labels
    plt.title("Preventable Depression (PD) - Discrete Binning", fontsize=12)
    plt.xlabel("Longitude", fontsize=10)
    plt.ylabel("Latitude", fontsize=10)
    ax.set_facecolor("white")
    plt.grid(False)
    plt.legend()

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

    print(f"Plot saved at: {output_path}")
