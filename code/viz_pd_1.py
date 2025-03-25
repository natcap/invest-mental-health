import rasterio
import rasterio.mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

# Define paths
PD_raster_path = "PD_i.tif"

# Load AOI boundary
aoi_gdf = aoi_adm1  # Ensure 'aoi_adm1' is a valid GeoDataFrame

# Load and mask PD_i raster with the AOI
with rasterio.open(PD_raster_path) as src:
    PD_masked, PD_transform = rasterio.mask.mask(src, aoi_gdf.geometry, crop=True, nodata=src.nodata)
    PD_meta = src.meta.copy()
    PD_meta.update({"transform": PD_transform, "width": PD_masked.shape[2], "height": PD_masked.shape[1]})

# Flatten and remove non-finite values for color scale computation
PD_flat = PD_masked[0].ravel()
PD_flat = PD_flat[np.isfinite(PD_flat)]

if PD_flat.size == 0:
    print("No valid data in PD_i after masking. Exiting.")
    exit()

min_val = PD_flat.min()
max_val = PD_flat.max()

# Anchor at zero if there's negative and positive data
if min_val < 0 and max_val > 0:
    vcenter = 0
else:
    vcenter = (min_val + max_val) / 2

# Avoid degenerate color range
if min_val >= max_val:
    max_val = min_val + 1e-6

# Build color norm: negative=red, positive=blue
norm = mcolors.TwoSlopeNorm(vmin=min_val, vcenter=vcenter, vmax=max_val)
color_map = plt.cm.RdBu

# Define plotting extent
pd_extent = [
    PD_meta["transform"][2],
    PD_meta["transform"][2] + PD_meta["width"] * PD_meta["transform"][0],
    PD_meta["transform"][5] + PD_meta["height"] * PD_meta["transform"][4],
    PD_meta["transform"][5]
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the masked PD_i data
im = ax.imshow(
    PD_masked[0],
    cmap=color_map,
    norm=norm,
    extent=pd_extent,
    origin="upper"
)

# Overlay the AOI boundary
aoi_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5, label="AOI Boundary")

# -------------------------------
# Enhanced Colorbar Definition
# -------------------------------
# We'll create a set of ticks that includes min_val, 0 (if in range), and max_val,
# plus extra ticks for intermediate values.

ticks = [min_val]
# If zero is in range, add it
if min_val < 0 < max_val:
    ticks.append(0)
# Then add the max_val
ticks.append(max_val)

# Optionally, add intermediate steps. For example, 3 extra steps between min->0 and 3 extra steps between 0->max.
# You can fine-tune how many intermediate ticks you want.
num_intermediates = 3
if min_val < 0 < max_val:
    neg_ticks = np.linspace(min_val, 0, num_intermediates + 0)  # exclude endpoints
    pos_ticks = np.linspace(0, max_val, num_intermediates + 4)
    ticks = np.concatenate([neg_ticks, [0], pos_ticks])
else:
    # If purely negative or purely positive, just do a simple linear space
    ticks = np.linspace(min_val, max_val, num_intermediates + 4)

ticks = sorted(list(set(ticks)))  # Ensure sorted and unique

print(ticks)


# Create and configure colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_yscale('linear')

cbar.set_label("PD_i Values (red=negative, blue=positive)", fontsize=10)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels([f"{t:.1f}" for t in ticks])

# Title and labels
plt.title("Preventable Depression (PD)", fontsize=12)
plt.xlabel("Longitude", fontsize=10)
plt.ylabel("Latitude", fontsize=10)

# Remove grid and set white background
plt.grid(False)
ax.set_facecolor("white")

# Add legend
plt.legend()

# Save the figure
output_plot_path = os.path.join(figures_dir, "PD_risk_map_v6.png")
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(output_plot_path, dpi=300, bbox_inches="tight", transparent=True)

# Show the plot
plt.show()

print(f"Plot saved at: {output_plot_path}")
