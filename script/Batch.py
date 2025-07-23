import os
import matplotlib.pyplot as plt
from Treecover_approach import run_ndvi_tree_analysis, run_pd_analysis, plot_ndvi_vs_negoal_gradient
import pandas as pd




default_paths_option1 = [
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\data_Erik\Chicago_Boundary\Chicago_aoi_boundary.shp",
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\data_Erik\Chicago_Boundary\chicago_aoi_censustract_reproj.shp",
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\data_Erik\usa_UNadj_Chi_pp_proj_clipped.tif",
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\data_Erik\chicago_NDVI_2013_prj_clipped.tif",
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\data_Erik\TreeCover\Tree_Cover_Clipped.tif",
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\data_Erik\BaselineRisk\chicago_aoi_baselinerisk_final.shp",
            r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_effect_size_table.xlsx",
            r"C:\Users\74007\Downloads\Stanford University\output_result"
        ]
# default_paths_option1 = [
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\aoi\cb_2019_us_county_500k_06075_clip.shp",
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\aoi\cb_2019_06_tract_500k.shp",
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\population\usa_ppp_2020_UNadj_constrained_SF_proj_setnull.tif",
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\ndvi\ndvi_s2_075_2019_10m_v2.tif",
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\tree_cover\ESA_WorldCover_10m_2021_v200_N36W123_Map.tif",
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\risk\baseline_incidence_rate_06075.shp",
#             r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_effect_size_table.xlsx",
#             r"C:\Users\74007\Downloads\Stanford University\output_result"
#         ]
tree_cover_percent = 30.0

# Step 1: Calculate NDVI and ne_goal
ne_goal, ndvi_fig, tree_fig, slider_fig, x_lowess, y_lowess, aoi_adm2, ndvi_resampled_path = run_ndvi_tree_analysis(*default_paths_option1)

# Step 2: PD analysis
health_cost_df = pd.read_excel(r"C:\Users\74007\Downloads\Stanford University\0_input_data\health_cost_table.xlsx")

print(health_cost_df.columns)

cost_value = health_cost_df.loc[health_cost_df["region"] == "USA", "cost_value"].values[0]

print(health_cost_df.columns)

fig1, fig2, fig_hist, fig_cost_curve, total_cases = run_pd_analysis(
    *default_paths_option1, ne_goal, aoi_adm2, x_lowess, y_lowess, cost_value
)
fig3 = plot_ndvi_vs_negoal_gradient(ndvi_resampled_path, aoi_adm2, ne_goal)

# Save figure
output_dir = default_paths_option1[7]
os.makedirs(output_dir, exist_ok=True)
fig2.savefig(os.path.join(output_dir, "PD_by_pixel.png"), dpi=300, bbox_inches='tight')
fig_hist.savefig(os.path.join(output_dir, "PD_histogram.png"), dpi=300, bbox_inches='tight')

# save summary
money_saved = total_cases * cost_value
message = f"{tree_cover_percent:.1f}% tree cover → {total_cases:,.0f} cases prevented, ${money_saved:,.0f} saved."
print(message)
# with open(os.path.join(output_dir, "summary_message.txt"), "w") as f:
#     f.write(message)

#
for fig in [fig1, fig2, fig_hist, fig3]:
    fig.show()
plt.show()
