import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
from Treecover_approach import run_ndvi_tree_analysis, run_pd_analysis, plot_ndvi_vs_negoal_gradient


# ==== Logging Setup ====
def setup_logging(output_dir):
    """Setup logging to both console and file with filtering"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"batch_processing_log_{timestamp}.txt")

    # Keywords to filter out from logs
    filter_keywords = [
        "Sample values:",
        "Looking for NDVI CSV at:",
        "Looking for Land Cover CSV at:",
        "NDVI deltas - All:",
        "=== Cost Curve Calculation ===",
        "Tree cover range:",
        "Risk ratio:",
        "Cost per case:",
        "Tree cover 0.0%:",
        "Tree cover 5.0%:",
        "Tree cover 10.0%:",
        "NDVI target:",
        "Pixels with improvement:",
        "Preventable cases:",
        "Cost savings:"
    ]

    class Logger:
        def __init__(self, log_file, filter_keywords):
            self.terminal = sys.stdout
            self.log = open(log_file, "w", encoding="utf-8")
            self.filter_keywords = filter_keywords
            self.current_line = ""

        def write(self, message):
            self.current_line += message

            # Check if we have complete lines
            if '\n' in message:
                lines = self.current_line.split('\n')
                self.current_line = lines[-1]  # Keep the last incomplete line

                for line in lines[:-1]:  # Process complete lines
                    should_filter = any(keyword in line for keyword in self.filter_keywords)

                    if not should_filter:
                        self.terminal.write(line + '\n')

                    # Always write to log file (unfiltered)
                    self.log.write(line + '\n')

            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    logger = Logger(log_file, filter_keywords)
    sys.stdout = logger
    return logger, log_file


def mean_continuous_raster(raster_path, band=1, extra_nodata_values=None, return_ci95=False):
    """Calculate mean value of a continuous raster, excluding nodata values"""
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(band)
            nodata = src.nodata

            # Create mask for valid data
            valid_mask = np.ones(data.shape, dtype=bool)

            # Exclude nodata values
            if nodata is not None:
                valid_mask = valid_mask & (data != nodata)

            # Exclude extra nodata values
            if extra_nodata_values:
                for val in extra_nodata_values:
                    valid_mask = valid_mask & (data != val)

            valid_data = data[valid_mask]

            if len(valid_data) == 0:
                return {"mean": np.nan, "std": np.nan, "n_valid_pixels": 0}

            result = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "n_valid_pixels": len(valid_data)
            }

            if return_ci95:
                se = result["std"] / np.sqrt(len(valid_data))
                result["ci95_lower"] = result["mean"] - 1.96 * se
                result["ci95_upper"] = result["mean"] + 1.96 * se

            return result

    except Exception as e:
        print(f"Error reading raster {raster_path}: {e}")
        return {"mean": np.nan, "std": np.nan, "n_valid_pixels": 0}


def get_max_tract_cases(fig2):
    """Extract maximum tract cases from the PD_by_pixel plot"""
    try:
        # Get the data from the figure
        ax = fig2.get_axes()[0]
        images = ax.get_images()
        if images:
            data = images[0].get_array()
            # Remove masked/invalid values
            valid_data = data[~np.isnan(data) & (data > 0)]
            if len(valid_data) > 0:
                return float(np.max(valid_data))
    except Exception as e:
        print(f"Error extracting max tract cases: {e}")
    return np.nan


def save_to_excel(results_data, excel_path):
    """Save results to Excel, appending if file exists"""
    try:
        # Try to read existing data
        if os.path.exists(excel_path):
            try:
                existing_df = pd.read_excel(excel_path, engine='openpyxl')
            except Exception as read_error:
                # File is corrupted, rename it and start fresh
                backup_path = excel_path.replace('.xlsx', '_corrupted_backup.xlsx')
                try:
                    os.rename(excel_path, backup_path)
                    print(f"  Warning: Corrupted Excel file moved to {backup_path}")
                except:
                    os.remove(excel_path)
                    print(f"  Warning: Corrupted Excel file removed, starting fresh")
                existing_df = None

            if existing_df is not None:
                # Check if this city-percentage combination already exists
                new_df = pd.DataFrame([results_data])

                # Remove existing entry for same city and percentage if it exists
                mask = (existing_df['city'] == results_data['city']) & \
                       (existing_df['tree_cover_increase_pct'] == results_data['tree_cover_increase_pct']) & \
                       (existing_df['calculation_method'] == results_data['calculation_method'])
                existing_df = existing_df[~mask]

                # Append new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Create new dataframe if file was corrupted
                combined_df = pd.DataFrame([results_data])
        else:
            # Create new dataframe
            combined_df = pd.DataFrame([results_data])

        # Sort by city, method and percentage for better readability
        combined_df = combined_df.sort_values(['city', 'calculation_method', 'tree_cover_increase_pct'])

        # Save to Excel with explicit engine
        combined_df.to_excel(excel_path, index=False, engine='openpyxl')

    except Exception as e:
        print(f"Error saving to Excel: {e}")
        # Fallback: save as CSV
        csv_path = excel_path.replace('.xlsx', '.csv')
        pd.DataFrame([results_data]).to_csv(csv_path, mode='a',
                                            header=not os.path.exists(csv_path), index=False)


# ==== Global settings ====
base_dir = r"S:\Shared drives\invest-health\City500\City_Folder_By_Num\City"
health_cost_excel = r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_cost_table.xlsx"
effect_excel = r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_effect_size_table.xlsx"

# ==== User Configuration ====
# Method 1: Percentage Add - directly add percentage
TREE_COVER_PERCENTAGES_ADD = list(range(1, 21))  # 1% to 20%

# Method 2: Percentage Multiply - multiply by (1 + percentage)
TREE_COVER_PERCENTAGES_MULTIPLY = list(range(5, 55, 5))  # 5%, 10%, 15%, ..., 50%

START_FROM_CITY = "0137000"  # Start processing from this city
ENABLE_PERCENTAGE_ADD = True  # Enable method 1 (Percentage Add)
ENABLE_PERCENTAGE_MULTIPLY = True  # Enable method 2 (Percentage Multiply)

# Use non-interactive backend to save memory
import matplotlib

matplotlib.use('Agg')

# Setup logging
logger, log_file = setup_logging(base_dir)
print(f"Log file: {log_file}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Processing methods enabled:")
if ENABLE_PERCENTAGE_ADD:
    print(f"  - Percentage Add: {TREE_COVER_PERCENTAGES_ADD[0]}-{TREE_COVER_PERCENTAGES_ADD[-1]}%")
if ENABLE_PERCENTAGE_MULTIPLY:
    print(f"  - Percentage Multiply: {TREE_COVER_PERCENTAGES_MULTIPLY[0]}-{TREE_COVER_PERCENTAGES_MULTIPLY[-1]}%")
print(f"Starting from city: {START_FROM_CITY}")
print("=" * 60)

# Setup Excel file paths for both methods
excel_results_path_add = os.path.join(base_dir, 'tree_cover_batch_results_percentage_add.xlsx')
excel_results_path_multiply = os.path.join(base_dir, 'tree_cover_batch_results_percentage_multiply.xlsx')

# ==== Load national health cost (USD per case) ====
try:
    health_cost_df = pd.read_excel(health_cost_excel)
    cost_value = health_cost_df.loc[health_cost_df["region"] == "USA", "cost_value"].values[0]
    print(f"Health cost: ${cost_value}/case")
except Exception as e:
    print(f"Error loading health cost: {e}")
    cost_value = 0

# ==== Collect tree cover statistics ====
base_path = Path(base_dir)
treecover_stats = []

print("\n[1/2] Calculating current tree cover...")
print("-" * 60)

for city_dir in sorted([p for p in base_path.iterdir() if p.is_dir()]):
    city_name = city_dir.name

    # Look for tree cover raster files
    treecover_candidates = list(city_dir.glob("treecover_*.tif"))
    if not treecover_candidates:
        treecover_candidates = list(city_dir.glob("treecover.tif"))

    if not treecover_candidates:
        print(f"  {city_name}: No tree cover raster")
        continue

    treecover_path = treecover_candidates[0]

    # Calculate mean tree cover
    stats = mean_continuous_raster(treecover_path, band=1,
                                   extra_nodata_values=(254, 255),
                                   return_ci95=True)

    current_treecover = stats['mean']
    if np.isnan(current_treecover):
        print(f"  {city_name}: Invalid data")
        continue

    treecover_stats.append({
        "city": city_name,
        "current_treecover": current_treecover,
        "n_valid_pixels": stats['n_valid_pixels'],
        "std": stats['std']
    })

    print(f"  {city_name}: {current_treecover:.1f}%")

# Save tree cover statistics
stats_df = pd.DataFrame(treecover_stats)
stats_csv = os.path.join(base_dir, 'treecover_stats.csv')
stats_df.to_csv(stats_csv, index=False)
print(f"\nSaved tree cover stats: {stats_csv}")

# Filter to start from specified city
start_idx = stats_df[stats_df['city'] == START_FROM_CITY].index
if len(start_idx) > 0:
    stats_df = stats_df.iloc[start_idx[0]:]
    print(f"Starting from city {START_FROM_CITY} (index {start_idx[0]})")
else:
    print(f"Warning: City {START_FROM_CITY} not found, processing all cities")

# Calculate total scenarios based on enabled methods
total_scenarios_add = len(stats_df) * len(TREE_COVER_PERCENTAGES_ADD) if ENABLE_PERCENTAGE_ADD else 0
total_scenarios_multiply = len(stats_df) * len(TREE_COVER_PERCENTAGES_MULTIPLY) if ENABLE_PERCENTAGE_MULTIPLY else 0
total_scenarios = total_scenarios_add + total_scenarios_multiply

print(f"\n[2/2] Processing {len(stats_df)} cities with {total_scenarios} total scenarios...")
print(f"  - Add method: {total_scenarios_add} scenarios")
print(f"  - Multiply method: {total_scenarios_multiply} scenarios")
print("-" * 60)

# ==== Main processing loop ====
all_results = []
failed_cities = []
processed_count = 0

# Loop through each city
for city_idx, row in enumerate(stats_df.iterrows(), 1):
    _, row_data = row
    city = row_data['city']
    current_treecover = row_data['current_treecover']

    city_folder = os.path.join(base_dir, city)

    print(f"\n[{city_idx}/{len(stats_df)}] {city} (current: {current_treecover:.1f}%)")

    # Construct file paths (matching your original code pattern)
    aoi_adm1_path = os.path.join(city_folder, f"city_{city}_tract.shp")
    aoi_adm2_path = os.path.join(city_folder, f"city_{city}_tract.shp")
    pop_path = os.path.join(city_folder, f"ppp_{city}.tif")
    ndvi_path = os.path.join(city_folder, f"NDVI_median_landsat_30m_2021_{city}_scaled100.tif")
    tree_path = os.path.join(city_folder, f"treecover_{city}.tif")
    risk_path = os.path.join(city_folder, f"depress_{city}.tif")
    output_dir = city_folder

    # Check file existence
    required_files = [aoi_adm1_path, pop_path, ndvi_path, tree_path, risk_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"  Missing files: {len(missing_files)}")
        if ENABLE_PERCENTAGE_ADD:
            failed_cities.extend([f"{city}_add_{pct}" for pct in TREE_COVER_PERCENTAGES_ADD])
        if ENABLE_PERCENTAGE_MULTIPLY:
            failed_cities.extend([f"{city}_multiply_{pct}" for pct in TREE_COVER_PERCENTAGES_MULTIPLY])
        continue

    try:
        # Clean up previous output files
        output_files_to_clean = [
            "ndvi_map.png", "tree_cover_map.png", "ndvi_tree_relationship.png",
            "pd_map_v1.png", "PD_by_pixel.png", "PD_histogram.png",
            "NDVI_vs_NE_goal.png", "summary_message.txt"
        ]

        for filename in output_files_to_clean:
            file_path = os.path.join(output_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

        # Step 1: Run NDVI analysis ONCE to get LOWESS relationship
        ne_goal_initial, ndvi_fig, tree_fig, slider_fig, x_lowess, y_lowess, aoi_adm2, ndvi_resampled_path = run_ndvi_tree_analysis(
            aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, effect_excel, output_dir
        )

        # Save the NDVI-Tree relationship curve in the city folder
        import gc

        gc.collect()  # Clean before saving

        slider_fig.savefig(os.path.join(output_dir, "ndvi_tree_relationship.png"), dpi=300, bbox_inches='tight')

        # Close figures to save memory
        plt.close(ndvi_fig)
        plt.close(tree_fig)
        plt.close(slider_fig)

        # Clear figure objects
        del ndvi_fig, tree_fig, slider_fig

        # Force garbage collection
        gc.collect()
        gc.collect()
        gc.collect()

        # Process each method
        city_success = False
        city_results_add = []
        city_results_multiply = []

        # Method 1: Percentage Add - Direct addition
        if ENABLE_PERCENTAGE_ADD:
            print(f"  Processing Method 1 (Percentage Add):")

            # Create subfolder for percentage add method
            results_dir_add = os.path.join(output_dir, "tree_cover_analysis_results", "percentage_add")
            os.makedirs(results_dir_add, exist_ok=True)

            for pct_idx, tree_cover_percent in enumerate(TREE_COVER_PERCENTAGES_ADD, 1):
                target_treecover = current_treecover + tree_cover_percent

                try:
                    # Calculate target NDVI using interpolation
                    target_ndvi = float(np.interp(target_treecover, x_lowess, y_lowess))
                    target_ndvi = max(0.0, min(1.0, target_ndvi))  # Clamp to valid NDVI range

                    # Run PD analysis with the calculated target NDVI
                    fig1, fig2, fig_hist, fig_cost_curve, total_cases = run_pd_analysis(
                        aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path,
                        effect_excel, output_dir, target_ndvi, aoi_adm2, x_lowess, y_lowess, cost_value
                    )

                    tract_csv_path = os.path.join(output_dir, "preventable_cases_by_tract.csv")
                    if os.path.exists(tract_csv_path):
                        try:
                            df_tract = pd.read_csv(tract_csv_path)
                            df_tract['tree_cover_increase_pct'] = tree_cover_percent
                            df_tract['current_treecover_pct'] = current_treecover
                            df_tract['target_treecover_pct'] = target_treecover
                            df_tract['target_ndvi'] = target_ndvi
                            df_tract['calculation_method'] = 'percentage_add'

                            # save results
                            tract_result_path = os.path.join(results_dir_add,
                                                             f"tract_results_{city}_{tree_cover_percent}pct.csv")
                            df_tract.to_csv(tract_result_path, index=False)
                        except Exception as e:
                            print(f"    Warning: Failed to save tract results: {e}")

                    # Get maximum tract cases
                    max_tract_cases = get_max_tract_cases(fig2)

                    # Calculate results
                    money_saved = total_cases * cost_value

                    # Save only the required plots with descriptive names
                    pd_pixel_filename = f"PD_by_pixel_{city}_{tree_cover_percent}pct_increase.png"
                    pd_hist_filename = f"PD_histogram_{city}_{tree_cover_percent}pct_increase.png"

                    # Force garbage collection before saving large files
                    gc.collect()

                    fig2.savefig(os.path.join(results_dir_add, pd_pixel_filename), dpi=300, bbox_inches='tight')
                    fig_hist.savefig(os.path.join(results_dir_add, pd_hist_filename), dpi=300, bbox_inches='tight')

                    # Close figures immediately after saving
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig_hist)
                    plt.close(fig_cost_curve)

                    # Clear figure objects
                    del fig1, fig2, fig_hist, fig_cost_curve

                    # Aggressive garbage collection after each scenario
                    gc.collect()
                    gc.collect()
                    gc.collect()

                    # Prepare data for Excel
                    excel_data = {
                        'city': city,
                        'calculation_method': 'percentage_add',
                        'current_treecover_pct': current_treecover,
                        'tree_cover_increase_pct': tree_cover_percent,
                        'target_treecover_pct': target_treecover,
                        'target_ndvi': target_ndvi,
                        'preventable_cases': total_cases,
                        'preventable_cost_usd': money_saved,
                        'cost_per_case_usd': cost_value,
                        'max_tract_cases': max_tract_cases,
                        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    # Store results
                    all_results.append(excel_data)
                    city_results_add.append({
                        'pct': tree_cover_percent,
                        'cases': total_cases,
                        'savings': money_saved,
                        'max_tract': max_tract_cases
                    })

                    # Save to Excel immediately (in case of crashes)
                    save_to_excel(excel_data, excel_results_path_add)

                    processed_count += 1
                    city_success = True

                except Exception as e:
                    print(f"  Error at +{tree_cover_percent}%: {e}")
                    failed_cities.append(f"{city}_add_{tree_cover_percent}")

                finally:
                    # Close all matplotlib figures to save memory
                    plt.close('all')

                    # Aggressive memory cleanup after each percentage
                    gc.collect()

                    # Every 5 scenarios, do extra cleanup
                    if pct_idx % 5 == 0:
                        gc.collect()
                        gc.collect()

                        # Try to release system memory
                        try:
                            import ctypes

                            ctypes.CDLL("msvcrt").malloc_trim(0)
                        except:
                            try:
                                ctypes.CDLL("libc.so.6").malloc_trim(0)
                            except:
                                pass

            # Print summary for Method 1
            if city_results_add:
                total_city_cases = sum(r['cases'] for r in city_results_add)
                total_city_savings = sum(r['savings'] for r in city_results_add)
                max_tract_result = max(city_results_add, key=lambda x: x['max_tract'])
                print(
                    f"   Add: {len(city_results_add)} scenarios | Cases: {total_city_cases:,.0f} | Savings: ${total_city_savings / 1e6:.1f}M | Max tract: {max_tract_result['max_tract']:.1f} (at +{max_tract_result['pct']}%)")

        # Method 2: Percentage Multiply - Multiply by (1 + percentage)
        if ENABLE_PERCENTAGE_MULTIPLY:
            print(f"  Processing Method 2 (Percentage Multiply):")

            # Create subfolder for percentage multiply method
            results_dir_multiply = os.path.join(output_dir, "tree_cover_analysis_results", "percentage_multiply")
            os.makedirs(results_dir_multiply, exist_ok=True)

            for pct_idx, tree_cover_percent in enumerate(TREE_COVER_PERCENTAGES_MULTIPLY, 1):
                # Calculate target as n * (1 + percentage/100)
                multiplier = 1 + (tree_cover_percent / 100)
                target_treecover = current_treecover * multiplier

                try:
                    # Calculate target NDVI using interpolation
                    target_ndvi = float(np.interp(target_treecover, x_lowess, y_lowess))
                    target_ndvi = max(0.0, min(1.0, target_ndvi))  # Clamp to valid NDVI range

                    # Run PD analysis with the calculated target NDVI
                    fig1, fig2, fig_hist, fig_cost_curve, total_cases = run_pd_analysis(
                        aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path,
                        effect_excel, output_dir, target_ndvi, aoi_adm2, x_lowess, y_lowess, cost_value
                    )

                    tract_csv_path = os.path.join(output_dir, "preventable_cases_by_tract.csv")
                    if os.path.exists(tract_csv_path):
                        try:
                            df_tract = pd.read_csv(tract_csv_path)
                            df_tract['tree_cover_increase_pct'] = tree_cover_percent
                            df_tract['current_treecover_pct'] = current_treecover
                            df_tract['target_treecover_pct'] = target_treecover
                            df_tract['target_ndvi'] = target_ndvi
                            df_tract['calculation_method'] = 'percentage_multiply'

                            # save results
                            tract_result_path = os.path.join(results_dir_multiply,
                                                             f"tract_results_{city}_{tree_cover_percent}pct_multiply.csv")
                            df_tract.to_csv(tract_result_path, index=False)
                        except Exception as e:
                            print(f"    Warning: Failed to save tract results: {e}")

                    # Get maximum tract cases
                    max_tract_cases = get_max_tract_cases(fig2)

                    # Calculate results
                    money_saved = total_cases * cost_value

                    # Save only the required plots with descriptive names
                    pd_pixel_filename = f"PD_by_pixel_{city}_{tree_cover_percent}pct_multiply.png"
                    pd_hist_filename = f"PD_histogram_{city}_{tree_cover_percent}pct_multiply.png"

                    # Force garbage collection before saving large files
                    gc.collect()

                    fig2.savefig(os.path.join(results_dir_multiply, pd_pixel_filename), dpi=300, bbox_inches='tight')
                    fig_hist.savefig(os.path.join(results_dir_multiply, pd_hist_filename), dpi=300, bbox_inches='tight')

                    # Close figures immediately after saving
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig_hist)
                    plt.close(fig_cost_curve)

                    # Clear figure objects
                    del fig1, fig2, fig_hist, fig_cost_curve

                    # Aggressive garbage collection after each scenario
                    gc.collect()
                    gc.collect()
                    gc.collect()

                    # Prepare data for Excel
                    excel_data = {
                        'city': city,
                        'calculation_method': 'percentage_multiply',
                        'current_treecover_pct': current_treecover,
                        'tree_cover_increase_pct': tree_cover_percent,
                        'target_treecover_pct': target_treecover,
                        'target_ndvi': target_ndvi,
                        'preventable_cases': total_cases,
                        'preventable_cost_usd': money_saved,
                        'cost_per_case_usd': cost_value,
                        'max_tract_cases': max_tract_cases,
                        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }

                    # Store results
                    all_results.append(excel_data)
                    city_results_multiply.append({
                        'pct': tree_cover_percent,
                        'cases': total_cases,
                        'savings': money_saved,
                        'max_tract': max_tract_cases
                    })

                    # Save to Excel immediately (in case of crashes)
                    save_to_excel(excel_data, excel_results_path_multiply)

                    processed_count += 1
                    city_success = True

                except Exception as e:
                    print(f"  Error at multiply {tree_cover_percent}%: {e}")
                    failed_cities.append(f"{city}_multiply_{tree_cover_percent}")

                finally:
                    # Close all matplotlib figures to save memory
                    plt.close('all')

                    # Aggressive memory cleanup after each percentage
                    gc.collect()

                    # Every 5 scenarios, do extra cleanup
                    if pct_idx % 5 == 0:
                        gc.collect()
                        gc.collect()

                        # Try to release system memory
                        try:
                            import ctypes

                            ctypes.CDLL("msvcrt").malloc_trim(0)
                        except:
                            try:
                                ctypes.CDLL("libc.so.6").malloc_trim(0)
                            except:
                                pass

            # Print summary for Method 2
            if city_results_multiply:
                total_city_cases = sum(r['cases'] for r in city_results_multiply)
                total_city_savings = sum(r['savings'] for r in city_results_multiply)
                max_tract_result = max(city_results_multiply, key=lambda x: x['max_tract'])
                print(
                    f"    ✓ Multiply: {len(city_results_multiply)} scenarios | Cases: {total_city_cases:,.0f} | Savings: ${total_city_savings / 1e6:.1f}M | Max tract: {max_tract_result['max_tract']:.1f} (at ×{max_tract_result['pct']}%)")

        # Overall progress
        progress_pct = (processed_count / total_scenarios) * 100
        print(f"  Progress: [{progress_pct:.1f}%]")

    except Exception as e:
        print(f"  Initial analysis error: {e}")
        if ENABLE_PERCENTAGE_ADD:
            failed_cities.extend([f"{city}_add_{pct}" for pct in TREE_COVER_PERCENTAGES_ADD])
        if ENABLE_PERCENTAGE_MULTIPLY:
            failed_cities.extend([f"{city}_multiply_{pct}" for pct in TREE_COVER_PERCENTAGES_MULTIPLY])

    finally:
        # Force cleanup after each city - AGGRESSIVE MEMORY RELEASE
        plt.close('all')

        # Clear matplotlib cache
        try:
            from matplotlib import font_manager

            font_manager._rebuild()
        except:
            pass

        # Force garbage collection multiple times
        gc.collect()
        gc.collect()
        gc.collect()

        # Try to release numpy memory
        try:
            import ctypes

            ctypes.CDLL("msvcrt").malloc_trim(0)  # Windows
        except:
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)  # Linux
            except:
                pass

        # Print memory usage (optional, for monitoring)
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / 1024 ** 3
            if city_idx % 5 == 0:  # Print every 5 cities
                print(f"  [Memory: {mem_gb:.2f} GB]")
        except:
            pass

# ==== Final Summary ====
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Cities processed: {len(stats_df)}")
print(f"Total scenarios: {total_scenarios}")
print(f"Successful: {len(all_results)} ({len(all_results) / total_scenarios * 100:.1f}%)")
print(f"Failed: {len(failed_cities)}")

if all_results:
    results_df = pd.DataFrame(all_results)

    # Summary by method
    for method in results_df['calculation_method'].unique():
        method_df = results_df[results_df['calculation_method'] == method]
        total_cases_all = method_df['preventable_cases'].sum()
        total_savings_all = method_df['preventable_cost_usd'].sum()

        method_name = "Percentage Add" if method == "percentage_add" else "Percentage Multiply"
        print(f"\n{method_name} Results:")
        print(f"  Total cases prevented: {total_cases_all:,.0f}")
        print(f"  Total savings: ${total_savings_all / 1e9:.2f}B")
        print(f"  Cases per scenario (avg): {method_df['preventable_cases'].mean():,.0f}")
        print(
            f"  Cases per scenario (range): {method_df['preventable_cases'].min():,.0f} - {method_df['preventable_cases'].max():,.0f}")

print(f"\nOutput files:")
if ENABLE_PERCENTAGE_ADD:
    print(f"  Percentage Add: {excel_results_path_add}")
if ENABLE_PERCENTAGE_MULTIPLY:
    print(f"  Percentage Multiply: {excel_results_path_multiply}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Close the logger
logger.close()
sys.stdout = logger.terminal