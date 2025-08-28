import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape

# ==== Input paths ====
root_folder = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\City"  # 15 city folders
big_shapefile = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\prevalence_rate_usa_2021.shp"  # large baseline shapefile
big_tif = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\nlcd_tcc_conus_wgs84_v2023-5_20210101_20211231.tif"  # large treecover raster
ppp_tif = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\usa_ppp_2020_UNadj_constrained.tif"

# Load the big shapefile once
gdf_big = gpd.read_file(big_shapefile)

# Loop through each city folder
for subdir in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subdir)
    if os.path.isdir(subfolder_path):
        # Find the city's shapefile
        city_shp = None
        for file in os.listdir(subfolder_path):
            if file.endswith(".shp"):
                city_shp = os.path.join(subfolder_path, file)
                break

        if not city_shp:
            print(f"No shapefile found in {subfolder_path}")
            continue

        # Read the city shapefile
        gdf_city = gpd.read_file(city_shp)

        # === 1. Clip strictly by city boundary, preserve big shapefile attributes ===
        # Merge all geometries in the small shapefile to a single polygon
        city_boundary = gdf_city.union_all()  # (use .unary_union if older geopandas)

        # Perform intersection manually for each feature in big shapefile
        gdf_big_clipped = gdf_big.copy()
        gdf_big_clipped["geometry"] = gdf_big_clipped["geometry"].intersection(city_boundary)
        # Drop rows with empty geometry (outside the city)
        gdf_big_clipped = gdf_big_clipped[~gdf_big_clipped.is_empty]

        out_shp = os.path.join(subfolder_path, f"baseline_{subdir}.shp")
        gdf_big_clipped.to_file(out_shp)
        print(f"Strictly clipped shapefile saved: {out_shp}")


        # # === 2. Clip the big raster (treecover) ===
        # with rasterio.open(big_tif) as src:
        #     # Reproject city shapefile to match raster CRS
        #     gdf_city_proj = gdf_city.to_crs(src.crs)
        #
        #     # Clip raster
        #     out_image, out_transform = mask(src, gdf_city_proj.geometry, crop=True)
        #     out_meta = src.meta.copy()
        #     out_meta.update({
        #         "height": out_image.shape[1],
        #         "width": out_image.shape[2],
        #         "transform": out_transform
        #     })
        #
        #     out_tif = os.path.join(subfolder_path, f"treecover_{subdir}.tif")
        #     with rasterio.open(out_tif, "w", **out_meta) as dest:
        #         dest.write(out_image)
        #     print(f"Clipped raster saved: {out_tif}")
        #
        # # === 3. Clip the PPP raster ===
        # with rasterio.open(ppp_tif) as src:
        #     gdf_city_proj = gdf_city.to_crs(src.crs)  # reproject to raster CRS
        #     out_image, out_transform = mask(src, gdf_city_proj.geometry, crop=True)
        #     out_meta = src.meta.copy()
        #     out_meta.update({
        #         "height": out_image.shape[1],
        #         "width": out_image.shape[2],
        #         "transform": out_transform
        #     })
        #
        #     out_ppp = os.path.join(subfolder_path, f"ppp_{subdir}.tif")
        #     with rasterio.open(out_ppp, "w", **out_meta) as dest:
        #         dest.write(out_image)
        #     print(f"Clipped PPP raster saved: {out_ppp}")
