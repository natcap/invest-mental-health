
import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import mapping
from tqdm import tqdm

# Load NLCD 2020 land cover raster

dir_lc = 'E:/_data/LULC/NLCD_USA/'


nlcd_path = os.path.join(dir_lc, 'nlcd_2019_land_cover_l48_20210604', 'nlcd_2019_land_cover_l48_20210604.img')
nlcd = rasterio.open(nlcd_path)

# Get and print current working directory
cwd = os.path.dirname(os.getcwd())
print("Current working directory:", cwd)
# Load places data
f = os.path.join(cwd, "data", "aoi", "places_in_or_touching_metros_487.shp")
places_in_met_unique_mainland = gpd.read_file(f)

# Ensure CRS matches
if places_in_met_unique_mainland.crs != nlcd.crs:
    places_proj = places_in_met_unique_mainland.to_crs(nlcd.crs)
else:
    places_proj = places_in_met_unique_mainland.copy()

# Develop land classes from NLCD
developed_classes = [21, 22, 23, 24]

# Create a new column to store % developed land
places_proj["pct_developed"] = np.nan

# Loop through each polygon and calculate % developed
# Loop through each polygon and calculate % developed
for idx, row in tqdm(places_proj.iterrows(), total=len(places_proj)):
    geom = [mapping(row.geometry)]
    
    try:
        out_image, out_transform = rasterio.mask.mask(nlcd, geom, crop=True)
        out_image = out_image[0]  # first band

        if out_image.size > 0:
            valid_pixels = out_image[out_image != nlcd.nodata]
            total_pixels = len(valid_pixels)
            if total_pixels > 0:
                developed_pixels = np.isin(valid_pixels, developed_classes).sum()
                percent_developed = (developed_pixels / total_pixels) * 100
                places_proj.at[idx, "pct_developed"] = round(percent_developed, 2)
    except Exception as e:
        print(f"Failed at index {idx}: {e}")
        continue

# Done
print("✅ Completed calculation.")
print(places_proj[["NAME", "pct_developed"]].head())