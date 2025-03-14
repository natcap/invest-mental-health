# Re-import necessary libraries since execution state was reset
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# -----------------------
# Step 1: Read NDVI Raster and Apply Masking
# -----------------------
ndvi_raster_path = "ndvi_s2_075_2019_10m_v2_prj_clipped_100m.tif"
with rasterio.open(ndvi_raster_path) as src:
    ndvi_data = src.read(1).astype(np.float32)  # Convert to float32
    ndvi_meta = src.meta.copy()
    ndvi_meta.update(dtype="float32", nodata=np.nan)

# Mask NDVI values < 0 as NoData (NaN)
ndvi_data[ndvi_data < 0] = np.nan

# Save masked NDVI raster
ndvi_masked_raster_path = "ndvi_masked.tif"
with rasterio.open(ndvi_masked_raster_path, "w", **ndvi_meta) as dst:
    dst.write(ndvi_data, 1)
print(f"Masked NDVI raster saved: {ndvi_masked_raster_path}")

# -----------------------
# Step 2: Calculate Delta NE_i (Difference from Goal)
# -----------------------
NE_goal = 0.4  # Example NE goal value
delta_NE_i = NE_goal - ndvi_data

# Save Delta NE_i raster
delta_NE_raster_path = "delta_NE_i.tif"
with rasterio.open(delta_NE_raster_path, "w", **ndvi_meta) as dst:
    dst.write(delta_NE_i, 1)
print(f"Delta NE_i raster saved: {delta_NE_raster_path}")

# -----------------------
# Step 3: Calculate RR_i
# -----------------------
RR_0_1NE = rr  # Example value for RR_0.1NE
RR_i = np.exp(np.log(RR_0_1NE) * 10 * delta_NE_i)

# -----------------------
# Step 4: Calculate PF_i
# -----------------------
PF_i = 1 - RR_i

# -----------------------
# Step 5: Read and Reproject Population Layer
# -----------------------
pop_raster_path = "usa_ppp_2020_UNadj_constrained_SF_prj_clipped.tif"
with rasterio.open(pop_raster_path) as src:
    population_data = src.read(1).astype(np.float32)  # Convert to float32
    pop_meta = src.meta.copy()

# Ensure population raster aligns with NDVI raster
if population_data.shape != ndvi_data.shape:
    print("Resampling population raster to match NDVI resolution...")
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

# Mask population data (set non-positive values to NaN)
population_data[population_data <= 0] = np.nan

# -----------------------
# Step 6: Calculate PD_i (Preventable Disease Impact)
# -----------------------
baseline_risk = 0.15  # Example baseline risk
PD_i = PF_i * baseline_risk * population_data

# -----------------------
# Step 7: Save PD_i as Raster Output
# -----------------------
PD_raster_path = "PD_i.tif"
with rasterio.open(PD_raster_path, "w", **ndvi_meta) as dst:
    dst.write(PD_i, 1)
print(f"PD_i raster saved: {PD_raster_path}")

