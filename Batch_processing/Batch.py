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


# ==== Global settings ====
base_dir = r"C:\Users\74007\Downloads\Stanford University\0_input_data\Batch\City"
health_cost_excel = r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_cost_table.xlsx"
effect_excel = r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_effect_size_table.xlsx"

# ==== User Configuration ====
TREE_COVER_INCREASE_PERCENT = 10.0  ############# user input

# Setup logging
logger, log_file = setup_logging(base_dir)
print(f"Logging to: {log_file}")
print(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Tree cover increase target: {TREE_COVER_INCREASE_PERCENT}%")
print("=" * 60)

# ==== Load national health cost (USD per case) ====
try:
    health_cost_df = pd.read_excel(health_cost_excel)
    cost_value = health_cost_df.loc[health_cost_df["region"] == "USA", "cost_value"].values[0]
    print(f"Health cost per case: ${cost_value}")
except Exception as e:
    print(f"Error loading health cost data: {e}")
    cost_value = 0

# ==== Collect tree cover statistics ====
base_path = Path(base_dir)
treecover_stats = []

print("\nStep 1: Calculating current tree cover for each city...")
print("-" * 50)

for city_dir in sorted([p for p in base_path.iterdir() if p.is_dir()]):
    city_name = city_dir.name

    # Look for tree cover raster files
    treecover_candidates = list(city_dir.glob("treecover_*.tif"))
    if not treecover_candidates:
        # Try alternative patterns
        treecover_candidates = list(city_dir.glob("treecover.tif"))

    if not treecover_candidates:
        print(f"Skip {city_name}: No tree cover raster found")
        continue

    treecover_path = treecover_candidates[0]

    # Calculate mean tree cover
    stats = mean_continuous_raster(treecover_path, band=1,
                                   extra_nodata_values=(254, 255),
                                   return_ci95=True)

    current_treecover = stats['mean']
    if np.isnan(current_treecover):
        print(f"Skip {city_name}: Invalid tree cover data")
        continue

    # Calculate target tree cover
    target_treecover = current_treecover + TREE_COVER_INCREASE_PERCENT

    treecover_stats.append({
        "city": city_name,
        "current_treecover": current_treecover,
        "target_treecover": target_treecover,
        "treecover_increase": TREE_COVER_INCREASE_PERCENT,
        "n_valid_pixels": stats['n_valid_pixels'],
        "std": stats['std']
    })

    print(
        f"{city_name:20} → Current: {current_treecover:.2f}%, Target: {target_treecover:.2f}% (+{TREE_COVER_INCREASE_PERCENT:.1f}%)")

# Save tree cover statistics
stats_df = pd.DataFrame(treecover_stats)
stats_csv = os.path.join(base_dir, 'treecover_stats.csv')
stats_df.to_csv(stats_csv, index=False)
print(f"\nTree cover statistics saved to: {stats_csv}")

print("\nStep 2: Processing cities with health impact analysis...")
print("-" * 60)

# ==== Main processing loop ====
processed_cities = []
failed_cities = []

for _, row in stats_df.iterrows():
    city = row['city']
    current_treecover = row['current_treecover']
    target_treecover = row['target_treecover']

    city_folder = os.path.join(base_dir, city)

    print(f"\nProcessing {city}")
    print(f"Current tree cover: {current_treecover:.2f}%")
    print(f"Target tree cover: {target_treecover:.2f}%")

    # Construct file paths
    aoi_adm1_path = os.path.join(city_folder, f"aoi_{city}.shp")
    aoi_adm2_path = os.path.join(city_folder, f"aoi_{city}.shp")
    pop_path = os.path.join(city_folder, f"ppp_{city}.tif")
    ndvi_path = os.path.join(city_folder, f"NDVI_median_landsat_30m_2021_{city}.tif")
    tree_path = os.path.join(city_folder, f"treecover_{city}.tif")
    risk_path = os.path.join(city_folder, f"depress_{city}.tif")
    output_dir = city_folder

    # Check file existence
    required_files = [aoi_adm1_path, pop_path, ndvi_path, tree_path, risk_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Missing files in {city}: {missing_files}")
        failed_cities.append(city)
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

        # Step 1: NDVI + tree cover analysis
        print(f"  Running NDVI and tree cover analysis...")
        ne_goal, ndvi_fig, tree_fig, slider_fig, x_lowess, y_lowess, aoi_adm2, ndvi_resampled_path = run_ndvi_tree_analysis(
            aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path, effect_excel, output_dir
        )

        # Save plots from Step 1
        ndvi_fig.savefig(os.path.join(output_dir, "ndvi_map.png"), dpi=300, bbox_inches='tight')
        tree_fig.savefig(os.path.join(output_dir, "tree_cover_map.png"), dpi=300, bbox_inches='tight')
        slider_fig.savefig(os.path.join(output_dir, "ndvi_tree_relationship.png"), dpi=300, bbox_inches='tight')

        # Step 2: Health impact analysis with custom target
        print(f"  Running health impact analysis...")
        fig1, fig2, fig_hist, fig_cost_curve, total_cases = run_pd_analysis(
            aoi_adm1_path, aoi_adm2_path, pop_path, ndvi_path, tree_path, risk_path,
            effect_excel, output_dir, ne_goal, aoi_adm2, x_lowess, y_lowess, cost_value
        )

        # Step 3: Plot NDVI deviation from NE_goal
        print(f"  Generating NDVI deviation plot...")
        fig3 = plot_ndvi_vs_negoal_gradient(ndvi_resampled_path, aoi_adm2, ne_goal)

        # Save all result plots
        fig1.savefig(os.path.join(output_dir, "pd_map_v1.png"), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(output_dir, "PD_by_pixel.png"), dpi=300, bbox_inches='tight')
        fig_hist.savefig(os.path.join(output_dir, "PD_histogram.png"), dpi=300, bbox_inches='tight')
        fig3.savefig(os.path.join(output_dir, "NDVI_vs_NE_goal.png"), dpi=300, bbox_inches='tight')

        # Calculate results
        money_saved = total_cases * cost_value

        # Create detailed summary message
        summary_lines = [
            f"City: {city}",
            f"Current tree cover: {current_treecover:.2f}%",
            f"Target tree cover: {target_treecover:.2f}% (+{TREE_COVER_INCREASE_PERCENT:.1f}%)",
            f"Cases prevented: {total_cases:,.0f}",
            f"Money saved: ${money_saved:,.0f}",
            f"Cost per case: ${cost_value:,.0f}"
        ]

        summary_message = "\n".join(summary_lines)

        # Save summary message
        with open(os.path.join(output_dir, "summary_message.txt"), "w", encoding="utf-8") as f:
            f.write(summary_message)

        print(f"  Results: {total_cases:,.0f} cases prevented, ${money_saved:,.0f} saved")
        processed_cities.append({
            'city': city,
            'current_treecover': current_treecover,
            'target_treecover': target_treecover,
            'cases_prevented': total_cases,
            'money_saved': money_saved
        })

    except Exception as e:
        print(f"  Error processing {city}: {e}")
        failed_cities.append(city)

    finally:
        # Clear all matplotlib figures to save memory
        plt.close('all')

# ==== Final Summary ====
print("\n" + "=" * 60)
print("PROCESSING SUMMARY")
print("=" * 60)
print(f"Total cities analyzed: {len(treecover_stats)}")
print(f"Successfully processed: {len(processed_cities)}")
print(f"Failed: {len(failed_cities)}")

if failed_cities:
    print(f"\nFailed cities: {', '.join(failed_cities)}")

if processed_cities:
    results_df = pd.DataFrame(processed_cities)
    total_cases = results_df['cases_prevented'].sum()
    total_savings = results_df['money_saved'].sum()
    avg_current_treecover = results_df['current_treecover'].mean()
    avg_target_treecover = results_df['target_treecover'].mean()

    print(f"\nOverall Results:")
    print(f"Average current tree cover: {avg_current_treecover:.2f}%")
    print(f"Average target tree cover: {avg_target_treecover:.2f}%")
    print(f"Total cases prevented: {total_cases:,.0f}")
    print(f"Total money saved: ${total_savings:,.0f}")

    # Save final results
    results_csv = os.path.join(base_dir, 'processing_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")

print(f"\nProcessing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}")

# Close the logger
logger.close()
sys.stdout = logger.terminal