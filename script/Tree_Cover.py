# Function for Tree Cover calculation

import os
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import NDVI_PW  # Import core geospatial functions

def process_shapefile(shp_path, target_crs, aoi_boundary):
    """Load, reproject, and clip a shapefile."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf_clipped = gpd.clip(gdf, aoi_boundary)
    gdf_clipped = gdf_clipped[gdf_clipped.area > 100]  # Filter small polygons
    gdf_clipped.reset_index(drop=True, inplace=True)

    return gdf_clipped


def compute_zonal_statistics(aoi_shapefile, raster_path):
    """Compute zonal statistics (categorical) for each census tract."""
    stats = zonal_stats(aoi_shapefile, raster_path, categorical=True, nodata=-9999)

    # Get unique land cover classes
    unique_classes = sorted(set(key for stat in stats for key in stat.keys()))

    # Compute percentages for each census tract
    percent_stats = []
    for stat in stats:
        total_pixels = sum(stat.values())
        tract_percent = {
            f"cover_{lc}": (stat.get(lc, 0) / total_pixels * 100) if total_pixels > 0 else 0
            for lc in unique_classes
        }
        percent_stats.append(tract_percent)

    df = pd.DataFrame(percent_stats)
    df["GEOID"] = aoi_shapefile["GEOID"].values  # Ensure merge compatibility
    return df


def save_landcover_csv(dataframe, aoi_shapefile, output_path):
    """Save the land cover percentage data to a CSV file."""
    if "GEOID" in aoi_shapefile.columns:
        dataframe.insert(0, "GEOID", aoi_shapefile["GEOID"])

    dataframe.to_csv(output_path, index=False)
    print(f"CSV file saved to: {output_path}")


def plot_landcover(dataframe, aoi_shapefile, column, title, return_fig=False):
    """Plot a map of land cover percentages for a given class. Return figure if needed."""
    import matplotlib.pyplot as plt

    if column in dataframe.columns:
        aoi_shapefile = aoi_shapefile.copy()
        aoi_shapefile[column] = dataframe[column]

        fig, ax = plt.subplots(figsize=(6, 5))
        aoi_shapefile.plot(column=column, cmap="viridis", legend=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if return_fig:
            return fig
        else:
            plt.show()
