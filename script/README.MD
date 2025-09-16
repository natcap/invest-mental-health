# How to Run This Project

## 1. Environment Setup
- Requires Python >= 3.10
- Install dependencies:

    pip install -r requirements.txt

## 2. Run the GUI

    python Gui.py

At startup, a flowchart screen appears with three options:

Option 1 → NDVI + Tree Cover + PD analysis

Option 2 → LULC-driven NDVI scenario (simplified)

Option 3 → LULC + NDVI integrated scenario (enhanced)

## 3. Code Structure

Main GUI

Gui.py

Entry point. Shows flowchart and launches Option 1/2/3

Option 1: 
GUI: GUI.py Function:Treecover_approach.py

Load AOI, NDVI, Tree Cover, Population, Risk rasters + effect table

Fit LOWESS (Tree cover % → NDVI)

User selects tree cover target → converted into NE_goal NDVI

Run PD analysis → PD_i map, cost maps, summary

Option 2: 

GUI: Run_option2.py Function:lulc_ndvi_analysis.py

Generate NDVI maps from NLCD codes using the attribute table

Apply mask from lc_exclude to filter invalid classes

Compute ΔNDVI (2021–2011), apply common mask

Align to population raster grid

Run PD + cost analysis

Option 3: 
GUI: Run_option3.py Function:lulc_ndvi_analysis.py


Derive per-class NDVI values by combining LULC with actual NDVI rasters

Apply mask from lc_exclude to filter invalid classes

Compute ΔNDVI (2021–2011), apply common mask

Align to population raster grid

Run PD + cost analysis

## 4. Preset/Cost table read/Output

Presets:


default_paths_option1 → Example inputs for Option 1

AOI shapefile(s)

NDVI raster

Tree cover raster

Population raster

Risk raster

Health effect Excel table

Health cost Excel table


default_paths_option2 → Example inputs for Option 2

LULC raster (baseline + scenario)

LULC attribute table (Excel)

Population raster

Tract shapefile

Risk shapefile

Health effect Excel table

Output folder


default_paths_option3 → Example inputs for Option 3

LULC raster (baseline + scenario)

NDVI raster (baseline + scenario)

LULC attribute table

Population raster

Tract shapefile

Risk shapefile

Health effect Excel table

Output folder


CostTableSelector widget:

Browse and select health_cost_table.xlsx

Dropdown for region → loads cost_value

Outputs:

NDVI maps (baseline, scenario, ΔNDVI)

Tree cover / NDVI relationship plots

PD_i maps (pixel-level and tract-level)

PD histograms and cost curves

Summary text: total preventable cases + monetary savings

## 5. Batch Processing

Scripts:

cut.py → clip NDVI & population rasters

Shp_convert.py → convert shapefile to raster

Treecover_approach.py → main functions

Batch.py → orchestrates multi-city pipeline

Folder Structure

Batch/

  ├─ City/

  │   ├─ Chicago/

  │   │   ├─ aoi.shp

  │   │   ├─ ndvi_Chicago.tif

  │   │   ├─ treecover_Chicago.tif

  │   │   ├─ pop_Chicago.tif

  │   │   └─ depress_Chicago.tif

  │   └─ OtherCity/...

  ├─ health_cost_table.xlsx

  └─ health_effect_size_table.xlsx


Outputs

City-specific PD maps, ΔNDVI maps, histograms

summary_message.txt

