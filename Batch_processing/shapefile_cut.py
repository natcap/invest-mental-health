import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# ==== Input paths ====
root_folder = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\City"  # City folders
big_shapefile = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\prevalence_rate_usa_2021.shp"  # large baseline shapefile

# Load the big shapefile once
gdf_big = gpd.read_file(big_shapefile)

# Loop through each city folder
for subdir in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subdir)
    if not os.path.isdir(subfolder_path):
        continue

    # Prefer aoi_2_cityname.shp, fallback to aoi_cityname.shp
    city_shp = next(
        (os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.startswith("aoi_2_") and f.endswith(".shp")),
        None
    )
    if not city_shp:
        city_shp = next((os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.startswith("aoi_") and f.endswith(".shp")), None)

    if not city_shp:
        print(f"[Skipped] No shapefile found in {subfolder_path}")
        continue

    # Read the city shapefile
    gdf_city = gpd.read_file(city_shp)
    if gdf_city.empty:
        print(f"[Skipped] Empty shapefile in {subdir}")
        continue

    # === 1. Strict intersection: clip big shapefile by city boundary ===
    city_boundary = gdf_city.union_all()  # merge to one polygon
    gdf_big_clipped = gdf_big.copy()
    gdf_big_clipped["geometry"] = gdf_big_clipped.geometry.intersection(city_boundary)

    # Drop empty and non-polygon geometries
    gdf_big_clipped = gdf_big_clipped[~gdf_big_clipped.geometry.is_empty & gdf_big_clipped.geometry.notna()]
    gdf_big_clipped = gdf_big_clipped[gdf_big_clipped.geometry.apply(lambda g: isinstance(g, (Polygon, MultiPolygon)))]

    if gdf_big_clipped.empty:
        print(f"[Warning] No polygon geometry left after clipping for {subdir}, skipping.")
        continue

    # Save clipped shapefile
    out_shp = os.path.join(subfolder_path, f"baseline_{subdir}.shp")
    gdf_big_clipped.to_file(out_shp, driver="ESRI Shapefile")

    print(f"[OK] {subdir}: saved {out_shp} | Features before: {len(gdf_big)} → after: {len(gdf_big_clipped)}")
