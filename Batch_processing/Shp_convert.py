import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import numpy as np

# ===== User configuration =====
base_dir = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\City"
usa_shp = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\prevalence_rate_usa_2021.shp"
target_field = "DEPRESS"
cell_size = 100
crs_epsg = 5070  # NAD83 / Conus Albers

# ===== Load and reproject USA prevalence shapefile =====
usa_gdf = gpd.read_file(usa_shp).to_crs(epsg=crs_epsg)

# ===== Loop through each city folder =====
for city in os.listdir(base_dir):
    city_folder = os.path.join(base_dir, city)
    if not os.path.isdir(city_folder):
        continue

    aoi_path = os.path.join(city_folder, f"aoi_{city}.shp")
    if not os.path.exists(aoi_path):
        print(f"AOI not found for city: {city}")
        continue

    # Load AOI and reproject
    aoi = gpd.read_file(aoi_path).to_crs(epsg=crs_epsg)

    # Clip USA prevalence data to AOI
    clipped = gpd.overlay(usa_gdf, aoi, how="intersection")

    if clipped.empty:
        print(f"No intersection found for {city}, skipping.")
        continue

    # Round DEPRESS values
    clipped[target_field] = clipped[target_field].round(1)

    # Compute bounds and raster shape
    minx, miny, maxx, maxy = clipped.total_bounds
    width = int((maxx - minx) / cell_size)
    height = int((maxy - miny) / cell_size)
    transform = from_origin(minx, maxy, cell_size, cell_size)

    # Geometry-value tuples
    shapes = [
        (geom, value)
        for geom, value in zip(clipped.geometry, clipped[target_field])
        if not np.isnan(value)
    ]

    # Rasterize
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype='float32'
    )

    # Output path
    output_tif = os.path.join(city_folder, f"depress_{city}.tif")

    # Write raster
    with rasterio.open(
        output_tif, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=f"EPSG:{crs_epsg}",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(raster, 1)

    print(f"{city}: Raster saved to {output_tif}")
