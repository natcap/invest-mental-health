import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import numpy as np
import os

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
    ncolors = num_bins + 2  # Ensure enough colors for 'extend="both"'
    cmap = plt.cm.get_cmap("RdBu", ncolors)
    norm = mcolors.BoundaryNorm(breakpoints, ncolors=ncolors, extend="both")

    # Define extent
    pd_extent = [
        PD_meta["transform"][2],
        PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
        PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
        PD_meta["transform"][5]
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.set_aspect('equal')
    im = ax.imshow(PD_masked[0], cmap=cmap, norm=norm, extent=pd_extent, origin="upper")
    aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
    print("PD Plot Extent:", pd_extent)
    # color bar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, extend='neither')
    cbar.set_label("PD_i (Quantile Bins)", fontsize=10)
    cbar.set_ticks(breakpoints)
    cbar.set_ticklabels([f"{b:.1f}" for b in breakpoints])
    cbar.ax.tick_params(labelsize=9)

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
    # plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    # plt.show()
    # print(f"Plot saved at: {output_path}")
    if return_fig:
        return fig
