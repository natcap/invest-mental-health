


def reproject_and_align(src_lc, src_tc, resampling_method=Resampling.nearest):
    """
    Reproject and align the treeCanopy raster to match the landcover raster's grid.
    """

    # Manually set the CRS if it's undefined or incorrect (replace with the correct EPSG codes)
    if src_lc.crs is None:
        src_lc_crs = "EPSG:5070"  # Example: Albers Equal Area
    else:
        src_lc_crs = src_lc.crs
    
    if src_tc.crs is None:
        src_tc_crs = "EPSG:4326"  # Example: WGS 84
    else:
        src_tc_crs = src_tc.crs

    # Calculate the transform and dimensions of the aligned treeCanopy raster
    transform, width, height = calculate_default_transform(
        src_tc_crs, src_lc_crs, src_lc.width, src_lc.height, *src_lc.bounds
    )
    
    # Initialize an array to hold the reprojected treeCanopy raster
    aligned_tc = np.empty((src_tc.count, height, width), dtype=src_tc.dtypes[0])
    
    # Reproject the treeCanopy raster to match the landcover raster's grid
    reproject(
        source=rasterio.band(src_tc, 1),
        destination=aligned_tc,
        src_transform=src_tc.transform,
        src_crs=src_tc_crs,
        dst_transform=transform,
        dst_crs=src_lc_crs,
        resampling=resampling_method
    )
    
    # Read the landcover data (no need to reproject, just read)
    aligned_lc = src_lc.read(1)

    return aligned_lc, aligned_tc[0], transform