
## Function to Reproject a Shapefile
import geopandas as gpd

def reproject_shapefile(shp_path, target_crs, dst_path=None):
    """
    Reproject a shapefile to a target CRS if needed.
    
    Parameters:
      shp_path (str): File path to the input shapefile.
      target_crs (dict or str): The target CRS. Can be an EPSG string (e.g., "EPSG:4326") or a CRS object.
      dst_path (str, optional): If provided, the reprojected shapefile is saved to this path.
    
    Returns:
      geopandas.GeoDataFrame: The reprojected GeoDataFrame.
    """
    # Read the shapefile
    gdf = gpd.read_file(shp_path)
    
    # Check if the shapefile's CRS matches the target CRS
    if gdf.crs != target_crs:
        print(f"Reprojecting shapefile from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    else:
        print("Shapefile CRS already matches the target CRS.")
    
    # If an output path is provided, save the reprojected GeoDataFrame.
    if dst_path:
        gdf.to_file(dst_path)
        print(f"Reprojected shapefile saved to: {dst_path}")
    
    return gdf

# Example usage for the shapefile function:
# target_crs can be specified as an EPSG code string, e.g., "EPSG:4326"
# reprojected_gdf = reproject_shapefile("path/to/input_shapefile.shp", "EPSG:4326", "path/to/output_shapefile.shp")




## ###################################################################################################################################
##      reproject_raster
## ###################################################################################################################################
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


## Function to Reproject a Raster
def reproject_raster(src_path, target_crs, dst_path, resampling_method=Resampling.nearest):
    """
    Reproject a raster to a target CRS if needed.
    
    Parameters:
      src_path (str): Path to the source raster.
      target_crs (dict or CRS): The target coordinate reference system (CRS). 
                                This can be provided as a dict (e.g., from rasterio.crs.CRS) or an EPSG string.
      dst_path (str): Path where the reprojected raster will be saved.
      resampling_method: Resampling method to use (default is Resampling.nearest).
                         For continuous data, consider Resampling.bilinear.
    
    Returns:
      None. The reprojected raster is written to dst_path.
    """
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        # Check if the source CRS matches the target CRS
        if src_crs == target_crs:
            print("Raster CRS already matches the target CRS. Copying file to destination.")
            # Optionally, copy the file without reprojecting
            with rasterio.open(dst_path, 'w', **src.meta) as dst:
                dst.write(src.read())
            return
        
        # Compute the transform, width, and height of the reprojected raster.
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        # Update the metadata for the output raster.
        dst_meta = src.meta.copy()
        dst_meta.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create an empty destination array for all bands.
        destination = None
        with rasterio.open(dst_path, 'w', **dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )
    print(f"Reprojected raster saved to: {dst_path}")
