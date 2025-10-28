import os
import shutil
import re
from pathlib import Path


def extract_code_from_city_file(filename):
    """Extract xxxx from city_xxxx.xxx format"""
    match = re.match(r'city_(.+?)\.', filename)
    return match.group(1) if match else None


def extract_code_from_ndvi_file(filename):
    """Extract xxxx from NDVI_median_landsat_30m_2021_xxxx_scaled100.tif format"""
    match = re.match(r'NDVI_median_landsat_30m_2021_(.+?)_scaled100\.tif', filename)
    return match.group(1) if match else None


def organize_files(folder1_path, folder2_path, output_path):
    """
    Organize files from two folders into new folders based on matching codes

    Args:
        folder1_path: Path to first folder (containing city_xxxx.xxx files)
        folder2_path: Path to second folder (containing NDVI_median_landsat_30m_2021_xxxx_scaled100.tif files)
        output_path: Path to output folder
    """

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Scan first folder and build code -> list of filenames mapping
    folder1_files = {}
    for filename in os.listdir(folder1_path):
        if os.path.isfile(os.path.join(folder1_path, filename)):
            code = extract_code_from_city_file(filename)
            if code:
                if code not in folder1_files:
                    folder1_files[code] = []
                folder1_files[code].append(filename)

    # Scan second folder and build code -> filename mapping
    folder2_files = {}
    for filename in os.listdir(folder2_path):
        if os.path.isfile(os.path.join(folder2_path, filename)):
            code = extract_code_from_ndvi_file(filename)
            if code:
                folder2_files[code] = filename

    # Find matching codes
    common_codes = set(folder1_files.keys()) & set(folder2_files.keys())

    print(f"Found {len(folder1_files)} files in folder 1")
    print(f"Found {len(folder2_files)} files in folder 2")
    print(f"Found {len(common_codes)} matching codes\n")

    # Create folders and copy files for each matching code
    for code in sorted(common_codes):
        # Create new folder
        new_folder = os.path.join(output_path, code)
        os.makedirs(new_folder, exist_ok=True)

        # Copy all files from folder 1 with this code
        for filename in folder1_files[code]:
            src = os.path.join(folder1_path, filename)
            dst = os.path.join(new_folder, filename)
            shutil.copy2(src, dst)

        # Copy file from folder 2
        src2 = os.path.join(folder2_path, folder2_files[code])
        dst2 = os.path.join(new_folder, folder2_files[code])
        shutil.copy2(src2, dst2)

        print(f"✓ Created folder: {code}")
        for filename in folder1_files[code]:
            print(f"  - {filename}")
        print(f"  - {folder2_files[code]}")

    # Report unmatched files
    unmatched_folder1 = set(folder1_files.keys()) - common_codes
    unmatched_folder2 = set(folder2_files.keys()) - common_codes

    if unmatched_folder1:
        print(f"\n⚠ {len(unmatched_folder1)} codes in folder 1 have no match:")
        for code in sorted(unmatched_folder1):
            print(f"  - code: {code} ({len(folder1_files[code])} files)")

    if unmatched_folder2:
        print(f"\n⚠ {len(unmatched_folder2)} files in folder 2 have no match:")
        for code in sorted(unmatched_folder2):
            print(f"  - {folder2_files[code]} (code: {code})")

    print(f"\n✓ Done! Created {len(common_codes)} folders")

# Usage example
if __name__ == "__main__":
    # Modify these paths to your actual paths
    folder1 = r"S:\Shared drives\invest-health\City500\City_Folder_By_Num\aoi_each"  # Folder containing city_xxxx.xxx files
    folder2 = r"S:\Shared drives\invest-health\City500\City_Folder_By_Num\ndvi_us500"  # Folder containing NDVI files
    output = r"S:\Shared drives\invest-health\City500\City_Folder_By_Num\City"  # Output folder

    organize_files(folder1, folder2, output)