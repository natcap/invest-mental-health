import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from lulc_ndvi_analysis import run_ndvi_option2_pipeline
from Health_cost import CostTableSelector
import os

class Option2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Option 2 Display")
        self.root.geometry("1920x1080")
        self.run_count = 0

        self.left_frame = tk.Frame(root, width=300, height=600)
        self.left_frame.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self.left_frame.grid_propagate(False)

        self.separator = tk.Frame(root, width=2, bg="gray", relief="sunken")
        self.separator.grid(row=0, column=1, sticky="ns")

        self.right_canvas = tk.Canvas(root)
        self.right_canvas.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)

        self.v_scrollbar = tk.Scrollbar(root, orient="vertical", command=self.right_canvas.yview)
        self.v_scrollbar.grid(row=0, column=3, sticky="ns")

        self.h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=self.right_canvas.xview)
        self.h_scrollbar.grid(row=1, column=2, sticky="ew", padx=(5, 10))

        self.right_canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.right_frame = tk.Frame(self.right_canvas)
        self.right_canvas.create_window((0, 0), window=self.right_frame, anchor="nw")
        self.right_frame.bind("<Configure>", lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))

        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self.log_label = tk.Label(self.left_frame, text="Status: Waiting to run", wraplength=280, anchor="w", justify="left")
        self.log_label.pack(fill="x", pady=(0, 20))

        self.run_button = tk.Button(self.left_frame, text="RUN", command=self.handle_run, bg="green", fg="white", font=("Arial", 14))
        self.run_button.pack(side="bottom", pady=(10, 5), fill="x")

        self.back_button = tk.Button(self.left_frame, text="Back", command=self.handle_back, bg="orange", fg="black", font=("Arial", 12))
        self.back_button.pack(side="bottom", pady=(0, 10), fill="x")

        self.right_plot_frame = tk.Frame(self.right_frame)
        self.right_plot_frame.pack(fill="both", expand=True)

        self.right_plot_frame_top = tk.Frame(self.right_plot_frame)
        self.right_plot_frame_top.pack(fill="both", expand=True)

        self.right_plot_frame_middle = tk.Frame(self.right_plot_frame)
        self.right_plot_frame_middle.pack(fill="both", expand=True)

        self.right_plot_frame_bottom = tk.Frame(self.right_plot_frame)
        self.right_plot_frame_bottom.pack(fill="both", expand=True)

        # Input panel on the right of first figure area
        self.input_panel = tk.Frame(self.right_plot_frame_top)
        self.input_panel.pack(fill="x", pady=10)

        self.input_labels = [
            "AOI County Shapefile:", "AOI Tract Shapefile:", "Population Raster:",
            "Land Cover Raster Baseline:", "Land Cover Raster Scenario:",
            "NLCD Attribute Table:", "Baseline Risk Shapefile:",
            "Health Effect Excel Table:", "Output Folder:"
        ]
        self.input_entries = []
        default_paths = [
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_us_county_500k_06075_clip.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_06_tract_500k.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\population\usa_ppp_2020_UNadj_constrained_SF_proj_setnull.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\lc\nlcd_2011_land_cover.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\lc\nlcd_2021_land_cover.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\lc\_lulc_attribute_table.xlsx",
            r"G:\Shared drives\invest-health\data\0_input_data\risk\baseline_incidence_rate_06075_2019.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\health_effect_size_table.xlsx",
            r"G:\Shared drives\invest-health\data\0_input_data\output"
        ]

        for i, label_text in enumerate(self.input_labels):
            row = tk.Frame(self.input_panel)
            row.pack(fill="x", pady=2)
            label = tk.Label(row, text=label_text, width=25, anchor="w", font=("Arial", 12))
            label.pack(side="left")
            entry = tk.Entry(row, width=130, font=("Arial", 12))
            entry.insert(0, default_paths[i])
            entry.pack(side="left", padx=(5, 0))
            browse_btn = tk.Button(row, text="Browse", command=lambda i=i: self.browse_file(i), font=("Arial", 12))
            browse_btn.pack(side="left", padx=5)
            self.input_entries.append(entry)

        for _ in range(2):
            spacer = tk.Label(self.input_panel, text="", height=1)
            spacer.pack()

        self.cost_selector = CostTableSelector(self.right_plot_frame_bottom)
        self.cost_selector.frame.pack(anchor="w", padx=10, pady=10)

    def browse_file(self, index):
        if "Folder" in self.input_labels[index]:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename()
        if path:
            self.input_entries[index].delete(0, tk.END)
            self.input_entries[index].insert(0, path)

    def handle_run(self):
        self.run_count += 1
        self.log_label.config(text=f"Status: Running step {self.run_count}...")

        paths = [entry.get() for entry in self.input_entries]
        (aoi_path, tract_path, pop_raster_path,
         lc_2011_path, lc_2021_path,
         attr_table_path, risk_path,
         health_effect_path, output_dir) = paths

        img_dir = output_dir

        selected_cost = self.cost_selector.df.loc[
            self.cost_selector.df["region"] == self.cost_selector.combo_var.get(), "cost_value"
        ].values[0]

        print(selected_cost)
        if self.run_count == 1:
            self.input_panel.pack_forget()
            run_ndvi_option2_pipeline(
                aoi_path=aoi_path,
                nlcd_2011_path=lc_2011_path,
                nlcd_2021_path=lc_2021_path,
                nlcd_attr_table_path=attr_table_path,
                pop_raster_path=pop_raster_path,
                tract_path=tract_path,
                risk_path=risk_path,
                health_effect_path=health_effect_path,
                output_dir=output_dir,
                cost_value=selected_cost
            )

            for frame in [self.right_plot_frame_top, self.right_plot_frame_middle, self.right_plot_frame_bottom]:
                for widget in frame.winfo_children():
                    if widget != self.input_panel:
                        widget.destroy()

            paths = ["op2_Figure_3.png", "op2_Figure_4.png"]
            locations = [
                (self.right_plot_frame_top, "left"), (self.right_plot_frame_top, "right"),
                (self.right_plot_frame_middle, "left"), (self.right_plot_frame_middle, "right")
            ]
        else:
            for frame in [self.right_plot_frame_top, self.right_plot_frame_middle, self.right_plot_frame_bottom]:
                for widget in frame.winfo_children():
                    if widget != self.input_panel:
                        widget.destroy()

            paths = ["op2_Figure_5.png", "Figure_7.png", "Figure_6.png"]
            locations = [
                (self.right_plot_frame_top, "left"), (self.right_plot_frame_top, "right"),
                (self.right_plot_frame_middle, "left")
            ]

        for (frame, side), filename in zip(locations, paths):
            path = os.path.join(img_dir, filename)
            if os.path.exists(path):
                img = Image.open(path)
                max_width = 600
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    if "Figure_5" in filename:
                        img = img.resize((600, 500), Image.Resampling.LANCZOS)
                    elif "Figure_7" in filename:
                        img = img.resize((700, 500), Image.Resampling.LANCZOS)
                    elif "Figure_6" in filename:
                        img = img.resize((650, 500), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                label = tk.Label(frame, image=photo)
                label.image = photo
                label.pack(side=side, padx=10, pady=10)

    def handle_back(self):
        if self.run_count == 2:
            self.run_count = 1
            self.handle_run()

if __name__ == "__main__":
    root = tk.Tk()
    app = Option2GUI(root)
    root.mainloop()
