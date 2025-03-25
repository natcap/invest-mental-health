import rasterio
import rasterio.mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

# --- Paths and data ---
PD_raster_path = "PD_i.tif"
aoi_gdf = aoi_adm1  # Must be a valid GeoDataFrame


# Flatten and filter
PD_flat = PD_masked[0].ravel()
PD_flat = PD_flat[np.isfinite(PD_flat)]
if PD_flat.size == 0:
    raise ValueError("No valid PD_i data in the AOI. Exiting.")

# --- Compute quantile breakpoints ---
quantiles = [0.01, 0.1, 0.25, 0.50, 0.6, 0.7, 0.9, 0.99] ## remove 0 and 1 as they are way smaller or larger than other values
breakpoints = np.quantile(PD_flat, quantiles)
breakpoints = np.concatenate([breakpoints, [0]])
breakpoints = np.unique(np.round(breakpoints, 1))
print(breakpoints)

# Fallback if all breakpoints are same
if len(breakpoints) == 1:
    breakpoints = [breakpoints[0] - 1e-6, breakpoints[0] + 1e-6]

num_bins = len(breakpoints) - 1

# IMPORTANT FIX:
# 'extend="both"' demands that ncolors >= num_bins + 2
# So we set:
ncolors = num_bins + 2

# Build a discrete colormap with enough colors
cmap = plt.cm.get_cmap("RdBu", ncolors)
norm = mcolors.BoundaryNorm(breakpoints, ncolors=ncolors, extend="both")

# --- Extent for plotting ---
pd_extent = [
    PD_meta["transform"][2],
    PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
    PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
    PD_meta["transform"][5]
]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(
    PD_masked[0],
    cmap=cmap,
    norm=norm,
    extent=pd_extent,
    origin="upper"
)

aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, label="AOI Boundary")

# Add colorbar with 'spacing="proportional"' to reflect bins proportionally
# cbar = plt.colorbar(im, ax=ax, spacing="proportional")
cbar = plt.colorbar(im, ax=ax)
# cbar.ax.set_yscale('linear')
cbar.set_label("PD_i Values (Binned by Quantile)")

# Ticks automatically come from the boundaries
cbar.set_ticks(breakpoints)
cbar.set_ticklabels([f"{b:.1f}" for b in breakpoints])

# Title and labels
plt.title("Preventable Depression (PD_i) - Discrete Bins")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(False)
ax.set_facecolor("white")
plt.legend()

output_plot_path = os.path.join(figures_dir, "PD_risk_map_v8_discrete.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight", transparent=True)
plt.show()

print(f"Plot saved at: {output_plot_path}")
