import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd

default_paths_option1 = [
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_us_county_500k_06075_clip.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_06_tract_500k.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\population\usa_ppp_2020_UNadj_constrained_SF_proj_setnull.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\ndvi\ndvi_s2_075_2019_10m_v2.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\tree_cover\ESA_WorldCover_10m_2021_v200_N36W123_Map.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\risk\baseline_incidence_rate_06075_2019.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\health_effect_size_table.xlsx",
            r"G:\Shared drives\invest-health\data\output_result"
        ]

# default_paths_option1 = [
#             r"G:\Shared drives\invest-health\data\0_input_data\data_Erik\Chicago_Boundary\Chicago_aoi_boundary.shp",
#             r"G:\Shared drives\invest-health\data\0_input_data\data_Erik\Chicago_Boundary\chicago_aoi_censustract_reproj.shp",
#             r"G:\Shared drives\invest-health\data\0_input_data\population\usa_ppp_2020_UNadj_constrained_Chicago.tif",
#             r"G:\Shared drives\invest-health\data\0_input_data\data_Erik\chicago_NDVI_2013_prj_clipped.tif",
#             r"G:\Shared drives\invest-health\data\0_input_data\data_Erik\TreeCover\Tree_Cover_Clipped.tif",
#             r"G:\Shared drives\invest-health\data\0_input_data\data_Erik\BaselineRisk\chicago_aoi_baselinerisk_final.shp",
#             r"G:\Shared drives\invest-health\data\0_input_data\health_effect_size_table.xlsx",
#             r"G:\Shared drives\invest-health\data\0_input_data\data_Erik\output"
#         ]

default_paths_option2 = [
    r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_us_county_500k_06075_clip.shp",
    r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_06_tract_500k.shp",
    r"G:\Shared drives\invest-health\data\0_input_data\population\usa_ppp_2020_UNadj_constrained_SF_proj_setnull.tif",
    r"G:\Shared drives\invest-health\data\0_input_data\lc\nlcd_2011_land_cover.tif",
    r"G:\Shared drives\invest-health\data\0_input_data\lc\nlcd_2021_land_cover.tif",
    r"G:\Shared drives\invest-health\data\0_input_data\lc\_lulc_attribute_table.xlsx",
    r"G:\Shared drives\invest-health\data\0_input_data\risk\baseline_incidence_rate_06075_2019.shp",
    r"G:\Shared drives\invest-health\data\0_input_data\health_effect_size_table.xlsx",
    r"G:\Shared drives\invest-health\data\output_result"
]

default_paths_option3 = [
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_us_county_500k_06075_clip.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_06_tract_500k.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\population\usa_ppp_2020_UNadj_constrained_SF_proj_setnull.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\ndvi\NDVI_landsat_30m_06075_2011_median.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\ndvi\NDVI_landsat_30m_06075_2021_median.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\lc\nlcd_2011_land_cover.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\lc\nlcd_2021_land_cover.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\risk\baseline_incidence_rate_06075_2019.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\health_effect_size_table.xlsx",
            r"G:\Shared drives\invest-health\data\output_result"
        ]

class CostTableSelector:
    def __init__(self, parent):
        self.frame = tk.Frame(parent)

        # Label
        tk.Label(self.frame, text="Health Cost Table:", font=("Arial", 12)).grid(row=0, column=0, sticky="w")

        # Entry for path
        self.entry = tk.Entry(self.frame, width=70, font=("Arial", 12))
        self.entry.grid(row=0, column=1, padx=5)
        self.entry.insert(0, r"G:\Shared drives\invest-health\data\0_input_data\health_cost_table.xlsx")

        # Browse button
        browse_btn = tk.Button(self.frame, text="Browse", font=("Arial", 12), command=self.browse)
        browse_btn.grid(row=0, column=2, padx=5)

        # Input button
        input_btn = tk.Button(self.frame, text="Input", font=("Arial", 12), command=self.load_table)
        input_btn.grid(row=0, column=3, padx=5)

        # Dropdown
        self.combo_var = tk.StringVar()
        self.combo = ttk.Combobox(self.frame, textvariable=self.combo_var, font=("Arial", 12), state="readonly")
        self.combo.grid(row=1, column=1, columnspan=2, pady=5, sticky="w")
        self.combo.bind("<<ComboboxSelected>>", self.display_cost)

        # Label to show cost
        self.cost_label = tk.Label(self.frame, text="", font=("Arial", 12))
        self.cost_label.grid(row=2, column=1, columnspan=2, sticky="w")

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if path:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, path)

    def load_table(self):
        path = self.entry.get()
        try:
            df = pd.read_excel(path)
            if "region" in df.columns:
                self.df = df
                self.combo['values'] = df['region'].dropna().unique().tolist()
                self.combo.current(0)
                self.display_cost()
        except Exception as e:
            print("Failed to load file:", e)

    def display_cost(self, event=None):
        selected = self.combo_var.get()
        try:
            value = self.df.loc[self.df['region'] == selected, 'cost_value'].values[0]
            self.cost_label.config(text=f"cost = {value}")
        except:
            self.cost_label.config(text="")
