import os
import shutil


source_folder = r'C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\City'


for filename in os.listdir(source_folder):
    # if filename.startswith('aoi_') and '.' in filename:
    #     name_part = filename.split('.')[0]
    #     suffix = filename.split('.')[-1]
    #     keyword = name_part[4:]
    if filename.startswith('NDVI_median_landsat_30m_') and '.' in filename:
        name_part = filename.split('.')[0]  # Remove extension

        # Extract city name: remove the prefix "NDVI_median_landsat_30m_2021_"
        prefix = "NDVI_median_landsat_30m_2021_"
        if name_part.startswith(prefix):
            keyword = name_part[len(prefix):]  # Everything after the prefix


        target_folder = os.path.join(source_folder, keyword)
        os.makedirs(target_folder, exist_ok=True)


        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(target_folder, filename)
        shutil.move(src_path, dst_path)

print("Files have been categorized")
