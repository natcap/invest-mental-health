import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

root_folder =  r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\City"

for subdir, _, files in os.walk(root_folder):
    for file in files:
        if file.startswith("aoi_") and file.endswith(".shp") and not file.startswith("aoi_2_"):
            shp_path = os.path.join(subdir, file)
            try:
                gdf = gpd.read_file(shp_path)
                if gdf.empty:
                    print(f"[Skipped] Empty shapefile: {shp_path}")
                    continue

                # Merge all geometries into one
                merged_geom = gdf.unary_union

                cleaned_polygons = []
                if merged_geom.geom_type == "Polygon":
                    cleaned_polygons.append(Polygon(merged_geom.exterior))
                elif merged_geom.geom_type == "MultiPolygon":
                    for poly in merged_geom.geoms:
                        cleaned_polygons.append(Polygon(poly.exterior))
                else:
                    print(f"[Warning] Unsupported geometry in {shp_path}")
                    continue

                # Create new GeoDataFrame with all cleaned polygons
                new_gdf = gpd.GeoDataFrame(geometry=cleaned_polygons, crs=gdf.crs)

                # Build new file name: aoi_2_cityname.shp
                new_name = "aoi_2_" + file.split("aoi_")[1]
                new_path = os.path.join(subdir, new_name)

                # Save to new file
                new_gdf.to_file(new_path)
                print(f"[OK] Saved outer boundaries (multi): {new_path}")

            except Exception as e:
                print(f"[Error] Failed for {shp_path}: {e}")
