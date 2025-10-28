import geopandas as gpd
import os
from pathlib import Path


def convert_shapefile_to_geojson(shp_path, output_path):
    """
    Convert a single shapefile to GeoJSON format

    Args:
        shp_path (str): Path to the input shapefile
        output_path (str): Path where the GeoJSON will be saved

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Read the shapefile
        gdf = gpd.read_file(shp_path)
        print(f"  Reading shapefile: {os.path.basename(shp_path)}")
        print(f"  Columns found: {list(gdf.columns)}")
        print(f"  Shape: {gdf.shape}")

        # Check for GEOID column
        if 'GEOID' in gdf.columns:
            print(f"  GEOID column found with sample values: {gdf['GEOID'].head().tolist()}")
        else:
            print(f"  Warning: No GEOID column found")

        # Convert to GeoJSON
        gdf.to_file(output_path, driver='GeoJSON')

        # Verify the conversion
        gdf_test = gpd.read_file(output_path)
        print(f"  Converted columns: {list(gdf_test.columns)}")

        if 'GEOID' in gdf.columns and 'GEOID' not in gdf_test.columns:
            print(f"  Error: GEOID column lost during conversion!")
            return False
        elif 'GEOID' in gdf_test.columns:
            print(f"  Success: GEOID preserved with values: {gdf_test['GEOID'].head().tolist()}")

        print(f"  Conversion successful: {output_path}")
        return True

    except Exception as e:
        print(f"  Error converting {shp_path}: {e}")
        return False


def batch_convert_shapefiles(root_directory):
    """
    Batch convert all shapefiles in subdirectories to GeoJSON format

    Args:
        root_directory (str): Root directory to search for shapefiles
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_directory}")
        return

    print(f"Starting batch conversion in: {root_directory}")
    print("=" * 80)

    total_found = 0
    total_converted = 0
    failed_conversions = []

    # Walk through all subdirectories
    for subdir in root_path.iterdir():
        if not subdir.is_dir():
            continue

        print(f"\nProcessing directory: {subdir.name}")
        print("-" * 40)

        # Find all shapefile (.shp) files in the subdirectory
        shp_files = list(subdir.glob("*.shp"))

        if not shp_files:
            print(f"  No shapefiles found in {subdir.name}")
            continue

        print(f"  Found {len(shp_files)} shapefile(s)")

        # Convert each shapefile to GeoJSON
        for shp_file in shp_files:
            total_found += 1

            # Create output GeoJSON path (same directory, different extension)
            geojson_file = shp_file.with_suffix('.geojson')

            print(f"\n  Converting: {shp_file.name}")

            # Perform the conversion
            if convert_shapefile_to_geojson(str(shp_file), str(geojson_file)):
                total_converted += 1
                print(f"  Output saved: {geojson_file.name}")
            else:
                failed_conversions.append(str(shp_file))

    # Print summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total shapefiles found: {total_found}")
    print(f"Successfully converted: {total_converted}")
    print(f"Failed conversions: {len(failed_conversions)}")

    if failed_conversions:
        print("\nFailed files:")
        for failed_file in failed_conversions:
            print(f"  - {failed_file}")

    if total_converted > 0:
        print(f"\nConversion completed! {total_converted} GeoJSON files created.")
    else:
        print("\nNo files were converted.")


if __name__ == "__main__":
    # Set the root directory path
    root_directory = r"G:\Shared drives\invest-health\City"

    # Run the batch conversion
    batch_convert_shapefiles(root_directory)