import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from Main import run_ndvi_tree_analysis, run_pd_analysis
import numpy as np
from Treecover_approach import*
# from NDVI_PW import plot_ndvi_vs_negoal_gradient
from PIL import Image, ImageTk


class FlowchartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Invest_Mental_Health Flowchart Entry")
        self.root.geometry("1920x1080")
        self.display_flowchart_screen()

    def display_flowchart_screen(self):
        self.flow_frame = tk.Frame(self.root)
        self.flow_frame.pack(fill="both", expand=True)

        # Load and show flowchart image
        img_path = r"D:/natcap/invest-mental-health/figs/_flow_chart.png"
        image = Image.open(img_path)
        image = image.resize((1300, 910), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)

        img_label = tk.Label(self.flow_frame, image=self.photo)
        img_label.pack(pady=20)

        # Option buttons
        button_frame = tk.Frame(self.flow_frame)
        button_frame.pack()

        tk.Button(button_frame, text="Option 1", font=("Arial", 14), command=self.launch_option_1).grid(row=0, column=0, padx=20)
        tk.Button(button_frame, text="Option 2", font=("Arial", 14), command=lambda: self.show_not_implemented(2)).grid(row=0, column=1, padx=20)
        tk.Button(button_frame, text="Option 3", font=("Arial", 14), command=lambda: self.show_not_implemented(3)).grid(row=0, column=2, padx=20)

    def launch_option_1(self):
        self.flow_frame.destroy()
        InvestGUI(self.root)

    def show_not_implemented(self, option_num):
        tk.messagebox.showinfo("Option", f"Option {option_num} is not implemented yet.")


class InvestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Invest_Mental_Health model")
        self.root.geometry("1920x1080")
        self.run_count = 0

        self.x_lowess = None
        self.y_lowess = None
        self.vline = None
        self.slider_canvas = None

        # Left panel
        self.left_frame = tk.Frame(root, width=300, height=600)
        self.left_frame.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)
        self.left_frame.grid_propagate(False)  # Prevent resizing by content



        self.separator = tk.Frame(root, width=2, bg="gray", relief="sunken")
        self.separator.grid(row=0, column=1, sticky="ns")

        # Right canvas and scrollbars
        self.right_canvas = tk.Canvas(root)
        self.right_canvas.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)

        self.v_scrollbar = tk.Scrollbar(root, orient="vertical", command=self.right_canvas.yview)
        self.v_scrollbar.grid(row=0, column=3, sticky="ns")

        self.h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=self.right_canvas.xview)
        self.h_scrollbar.grid(row=1, column=2, sticky="ew", padx=(5, 10))

        self.right_canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.right_frame = tk.Frame(self.right_canvas)
        self.right_canvas.create_window((0, 0), window=self.right_frame, anchor="nw")

        # Scroll region updates
        self.right_frame.bind("<Configure>", lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")))

        root.grid_columnconfigure(0, weight=0)
        root.grid_columnconfigure(1, weight=0)
        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # Run status and button
        # self.log_label = tk.Label(self.left_frame, text="Status: Waiting to run", anchor="w", justify="left")
        self.log_label = tk.Label(self.left_frame, text="Status: Waiting to run", wraplength=280, anchor="w", justify="left")
        self.log_label.pack(fill="x", pady=(0, 20))

        # RUN button first (appears at the very bottom)
        self.run_button = tk.Button(
            self.left_frame, text="RUN", command=self.handle_run,
            bg="green", fg="white", font=("Arial", 14)
        )
        self.run_button.pack(side="bottom", pady=(10, 5), fill="x")

        # Back button above RUN
        self.back_button = tk.Button(
            self.left_frame, text="Back", command=self.handle_back,
            bg="orange", fg="black", font=("Arial", 12)
        )
        self.back_button.pack(side="bottom", pady=(0, 10), fill="x")


        # Input section
        self.right_input_frame = tk.Frame(self.right_frame)
        self.right_input_frame.pack(fill="both", expand=True)
        self.right_canvas.bind_all("<MouseWheel>", lambda event: self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        self.input_entries = []
        self.input_labels = [
            "AOI County Shapefile:",
            "AOI Tract Shapefile:",
            "Population Raster:",
            "NDVI Raster:",
            "Tree Cover Raster (ESA WorldCover):",
            "Baseline Risk Shapefile:",
            "Health Effect Excel Table:",
            "Health Cost Table:",
            "Output Folder:"
        ]

        default_paths = [
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_us_county_500k_06075_clip.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\aoi\cb_2019_06_tract_500k.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\population\usa_ppp_2020_UNadj_constrained_SF_proj_setnull.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\ndvi\ndvi_s2_075_2019_10m_v2.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\tree_cover\ESA_WorldCover_10m_2021_v200_N36W123_Map.tif",
            r"G:\Shared drives\invest-health\data\0_input_data\risk\baseline_incidence_rate_06075_2019.shp",
            r"G:\Shared drives\invest-health\data\0_input_data\health_effect_size_table.xlsx",
            r"G:\Shared drives\invest-health\data\0_input_data\health_cost_table.xlsx",
            r"G:\Shared drives\invest-health\data\output_result"
        ]

        for i in range(9):
            label = tk.Label(self.right_input_frame, text=self.input_labels[i], anchor="e",font=("Arial", 12))
            label.grid(row=i, column=0, sticky="e", padx=5, pady=5)

######### Edit entry dialog length##########
            entry = tk.Entry(self.right_input_frame, width=130,font=("Arial", 12))
######### Edit entry dialog length##########
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="we")
            entry.insert(0, default_paths[i])
            self.input_entries.append(entry)

            browse_btn = tk.Button(self.right_input_frame, text="Browse", font=("Arial", 12),command=lambda e=entry, i=i: self.browse_path(e, self.input_labels[i]))
            browse_btn.grid(row=i, column=2, padx=5, pady=5)

        self.right_input_frame.grid_columnconfigure(1, weight=1)

    def browse_path(self, entry_widget, label_text):
        if "Folder" in label_text:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename()
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def handle_run(self):
        self.run_count += 1
        self.log_label.config(text=f"Status: Running step {self.run_count}...")
        # self.log_label = tk.Label(self.left_frame, text=f"Status: Running step {self.run_count}...", wraplength=280, anchor="w",
        #                           justify="left")

        if self.run_count == 1:
            self.paths = [e.get() for e in self.input_entries]

            if self.right_input_frame:
                self.right_input_frame.destroy()
                self.right_input_frame = None

            self.right_plot_frame = tk.Frame(self.right_frame)
            self.right_plot_frame.pack(fill="both", expand=True)

            self.right_plot_frame_top = tk.Frame(self.right_plot_frame)
            self.right_plot_frame_top.pack(fill="both", expand=True)

            self.right_plot_frame_middle = tk.Frame(self.right_plot_frame)
            self.right_plot_frame_middle.pack(fill="both", expand=True)

            self.right_plot_frame_bottom = tk.Frame(self.right_plot_frame)
            self.right_plot_frame_bottom.pack(fill="x", expand=False)

            ne_goal, ndvi_fig, tree_fig, slider_fig, self.x_lowess, self.y_lowess, self.aoi_adm2,self.ndvi_resampled_path = run_ndvi_tree_analysis(*self.paths)
            self.vline = slider_fig.axes[0].axvline(30.0, color='gray', linestyle='--')

            self.display_figure(ndvi_fig, location="top", side="left")
            self.display_figure(tree_fig, location="top", side="right")
            self.display_figure(slider_fig, location="middle")
            self.right_plot_frame.configure(bg="white")
            self.right_plot_frame_top.configure(bg="white")
            self.right_plot_frame_middle.configure(bg="white")
            self.right_plot_frame_bottom.configure(bg="white")

            self.setup_slider_area(slider_fig)

            self.ne_goal = ne_goal
            self.log_label.config(text=f"NE goal selected: {ne_goal:.3f}")
            # self.log_label = tk.Label(self.left_frame, text=f"NE goal selected: {ne_goal:.3f}", wraplength=280,
            #                           anchor="w",
            #                           justify="left")

        elif self.run_count == 2:
            # Clear old figures in all three sections
            for widget in self.right_plot_frame_top.winfo_children():
                widget.destroy()
            for widget in self.right_plot_frame_middle.winfo_children():
                widget.destroy()
            for widget in self.right_plot_frame_bottom.winfo_children():
                widget.destroy()

            health_cost_df = pd.read_excel(self.paths[7])
            cost_value = health_cost_df.loc[health_cost_df["region"] == "USA", "cost_value"].values[0]
            fig1, fig2, fig_hist, fig_cost_curve, total_cases = run_pd_analysis(
                *self.paths, self.ne_goal, self.aoi_adm2, self.x_lowess, self.y_lowess,cost_value
            )
            fig3 = plot_ndvi_vs_negoal_gradient(self.ndvi_resampled_path, self.aoi_adm2, self.ne_goal)
            self.display_figure(fig3, location="top", side="left")
            self.display_figure(fig2, location="top", side="right")
            self.display_figure(fig_hist, location="middle", side="left")
            self.display_figure(fig_cost_curve, location="middle", side="right")

            self.right_plot_frame.configure(bg="white")
            self.right_plot_frame_top.configure(bg="white")
            self.right_plot_frame_middle.configure(bg="white")
            self.right_plot_frame_bottom.configure(bg="white")

            # Calculate summary message
            cover = self.slider_var.get()

            health_cost_df = pd.read_excel(self.paths[7])
            city_cost = health_cost_df.loc[health_cost_df["region"] == "USA", "cost_value"].values[0]

            money_saved = total_cases * city_cost

            import textwrap
            message = f"{cover:.1f}% tree cover → {total_cases:,.0f} cases prevented, ${money_saved:,.0f} saved."
            wrapped_msg = textwrap.fill(message, width=35)
            self.log_label.config(text=wrapped_msg)


    def setup_slider_area(self, fig):
        self.slider_canvas = FigureCanvasTkAgg(fig, master=self.right_plot_frame_middle)
        self.slider_canvas.draw()

        self.slider_ax = fig.axes[0]
        initial_cover = self.slider_var.get() if hasattr(self, "slider_var") else 30.0
        self.vline = self.slider_ax.axvline(initial_cover, color='gray', linestyle='--')


        slider_label = tk.Label(self.right_plot_frame_bottom, text="Tree Cover (%)")
        slider_label.pack(anchor="w")

        self.slider_var = tk.DoubleVar(value=30.0)
        self.slider = tk.Scale(
            self.right_plot_frame_bottom,
            from_=min(self.x_lowess),
            to=max(self.x_lowess),
            resolution=0.5,
            orient="horizontal",
            variable=self.slider_var,
            command=self.update_ndvi_value
        )


        self.slider.pack()

        def update_slider_width(event):
            fig_widget = self.slider_canvas.get_tk_widget()
            fig_width = fig_widget.winfo_width()
            if fig_width > 100:
                self.slider.config(length=fig_width - 2)


        self.right_plot_frame_middle.after(100, self.update_slider_length)


        self.ndvi_display = tk.Label(self.right_plot_frame_bottom, text="Drag the slider", font=("Arial", 12))
        self.ndvi_display.pack(pady=5)

        confirm_btn = tk.Button(self.right_plot_frame_bottom, text="Confirm", command=self.confirm_ndvi)
        confirm_btn.pack()

    def update_slider_length(self):
        try:
            fig_widget = self.slider_canvas.get_tk_widget()
            fig_width = fig_widget.winfo_width()

            if fig_width > 200:
                self.slider.config(length=int(fig_width * 0.9))
            else:

                fallback_width = self.right_plot_frame_middle.winfo_width()
                self.slider.config(length=int(fallback_width * 0.8))
        except Exception as e:
            print("Slider width update error:", e)


    def update_ndvi_value(self, val):
        cover = float(val)
        ndvi_val = float(np.interp(cover, self.x_lowess, self.y_lowess))
        if hasattr(self, "vline"):
            self.vline.set_xdata([cover])
            self.slider_canvas.draw()
        self.ndvi_display.config(text=f"Tree cover target: {cover:.1f}% → Corresponding NDVI: {ndvi_val:.3f}")


    def confirm_ndvi(self):
        cover = self.slider_var.get()
        self.ne_goal = float(np.interp(cover, self.x_lowess, self.y_lowess))
        self.log_label.config(text=f"Confirmed NE_goal: {self.ne_goal:.3f}")
        # self.log_label = tk.Label(self.left_frame, text=f"Confirmed NE_goal: {self.ne_goal:.3f}", wraplength=280,
        #                           anchor="w",
        #                           justify="left")

    def display_figure(self, fig, location="middle", side="left"):
        frame = {
            "top": self.right_plot_frame_top,
            "middle": self.right_plot_frame_middle
        }.get(location, self.right_plot_frame)


        col = 0 if side == "left" else 1

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")


    def handle_back(self):
        if self.run_count == 2:
            self.run_count = 1
            # clear images
            for section in [self.right_plot_frame_top, self.right_plot_frame_middle, self.right_plot_frame_bottom]:
                for widget in section.winfo_children():
                    widget.destroy()

            # Return to last step
            ne_goal, ndvi_fig, tree_fig, slider_fig, self.x_lowess, self.y_lowess, self.aoi_adm2, self.ndvi_resampled_path = run_ndvi_tree_analysis(
                *self.paths)
            self.vline = slider_fig.axes[0].axvline(30.0, color='gray', linestyle='--')
            self.display_figure(ndvi_fig, location="top", side="left")
            self.display_figure(tree_fig, location="top", side="right")
            self.display_figure(slider_fig, location="middle")
            self.setup_slider_area(slider_fig)
            self.ne_goal = ne_goal
            self.log_label.config(text="Returned to NE_goal selection.")
            # self.log_label = tk.Label(self.left_frame, text="Returned to NE_goal selection view.", wraplength=280,
            #                           anchor="w",
            #                           justify="left")

if __name__ == "__main__":
    root = tk.Tk()
    # root.geometry("1400x1200")
    # root.state("zoomed")
    app = FlowchartApp(root)
    # app = InvestGUI(root)
    root.mainloop()
