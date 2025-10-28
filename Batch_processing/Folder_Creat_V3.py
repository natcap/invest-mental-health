import os
import shutil
import re
from pathlib import Path


def extract_code_from_tract_file(filename):
    """Extract xxxx from city_xxxx_tract.xxx format"""
    match = re.match(r'city_(.+?)_tract\.', filename)
    return match.group(1) if match else None


def organize_tract_files(tract_folder, city_base_folder):
    """
    Organize tract files into existing city folders

    Args:
        tract_folder: Folder containing city_xxxx_tract.xxx files
        city_base_folder: Base folder containing subfolders named by xxxx codes
    """

    # Get all tract files
    tract_files = {}
    for filename in os.listdir(tract_folder):
        file_path = os.path.join(tract_folder, filename)
        if os.path.isfile(file_path):
            code = extract_code_from_tract_file(filename)
            if code:
                if code not in tract_files:
                    tract_files[code] = []
                tract_files[code].append(filename)

    print(f"Found {len(tract_files)} unique codes in tract folder")
    print(f"Total tract files: {sum(len(files) for files in tract_files.values())}\n")

    # Get existing city folders
    existing_folders = set()
    for item in os.listdir(city_base_folder):
        item_path = os.path.join(city_base_folder, item)
        if os.path.isdir(item_path):
            existing_folders.add(item)

    print(f"Found {len(existing_folders)} existing city folders\n")

    # Copy files to corresponding folders
    copied_count = 0
    missing_folders = set()

    for code, filenames in sorted(tract_files.items()):
        target_folder = os.path.join(city_base_folder, code)

        if code not in existing_folders:
            missing_folders.add(code)
            print(f"⚠ Folder not found for code: {code} ({len(filenames)} files)")
            continue

        # Copy all files with this code
        for filename in filenames:
            src = os.path.join(tract_folder, filename)
            dst = os.path.join(target_folder, filename)
            shutil.copy2(src, dst)
            copied_count += 1

        print(f"✓ Copied to {code}/ ({len(filenames)} files)")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Total files copied: {copied_count}")
    print(f"  Successful codes: {len(tract_files) - len(missing_folders)}")
    print(f"  Missing folders: {len(missing_folders)}")

    if missing_folders:
        print(f"\n⚠ Codes without matching folders:")
        for code in sorted(missing_folders):
            print(f"  - {code} ({len(tract_files[code])} files)")


# Usage example
if __name__ == "__main__":
    # Modify these paths to your actual paths
    tract_folder = r"S:\Shared drives\invest-health\City500\aoi_each_with_tract"  # Folder containing city_xxxx_tract.xxx files
    city_base_folder = r"S:\Shared drives\invest-health\City500\City_Folder_By_Num\City"  # Folder containing xxxx subfolders

    organize_tract_files(tract_folder, city_base_folder)