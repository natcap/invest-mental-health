import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import statsmodels.api as sm
import numpy as np
from scipy.interpolate import UnivariateSpline

# Set Plotly renderer for Jupyter Lab
pio.renderers.default = "browser"

def merge_ndvi_landcover_data(ndvi_csv_path, landcover_csv_path):
    """
    Merge NDVI and Land Cover data on GEOID.

    Parameters:
        ndvi_csv_path (str): Path to the NDVI CSV file.
        landcover_csv_path (str): Path to the land cover CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing NDVI and Land Cover statistics.
    """
    print(f"Looking for NDVI CSV at: {os.path.abspath(ndvi_csv_path)}")
    print(f"Looking for Land Cover CSV at: {os.path.abspath(landcover_csv_path)}")

    if not os.path.exists(ndvi_csv_path):
        raise FileNotFoundError(f"NDVI CSV file not found: {ndvi_csv_path}")
    if not os.path.exists(landcover_csv_path):
        raise FileNotFoundError(f"Land Cover CSV file not found: {landcover_csv_path}")

    df_ndvi = pd.read_csv(ndvi_csv_path)
    df_landcover = pd.read_csv(landcover_csv_path)

    # Merge datasets on GEOID
    df_merged = pd.merge(df_ndvi, df_landcover, on="GEOID", how="inner")
    return df_merged


def plot_ndvi_vs_treecover(df, adm_level):
    """
    Create an interactive Plotly scatter plot of Tree Cover (%) vs. Population-Weighted NDVI.
    Includes a LOWESS trendline to visualize the overall relationship.
    """
    import statsmodels.api as sm  # Required for LOWESS trendline
    print("Generating Plotly scatter plot...")

    # Categorize NDVI level using the 75th percentile as threshold
    ndvi_threshold = df["weighted_ndvi"].quantile(0.75)
    df["NDVI_Level"] = df["weighted_ndvi"].apply(
        lambda x: "Above 75th Percentile" if x > ndvi_threshold else "Below 75th Percentile"
    )

    # Set marker size based on tree cover percentage
    df["marker_size"] = df["cover_10"].apply(lambda x: 20 if x > 30 else 5)

    # Create interactive scatter plot with LOWESS trendline
    fig = px.scatter(
        df,
        x="cover_10",
        y="weighted_ndvi",
        color="NDVI_Level",
        size="marker_size",
        category_orders={"NDVI_Level": ["Above 75th Percentile", "Below 75th Percentile"]},
        color_discrete_map={"Above 75th Percentile": "#5ab4ac", "Below 75th Percentile": "#d8b365"},
        trendline="rolling",                      # Add LOWESS trendline
        trendline_color_override="red",          # Trendline color
        trendline_scope="overall",               # Apply trendline to all data
        hover_data={"GEOID": True, "cover_10": True, "weighted_ndvi": True}
    )

    # Customize plot layout
    fig.update_layout(
        title=f"Tree Cover vs. Mean NDVI (by adm {adm_level})",
        xaxis_title="Tree Cover (%)",
        yaxis_title="Mean NDVI (Population-Weighted)",
        template="plotly"
    )

    fig.show()

def plot_ndvi_vs_treecover_popup(df):
    """
    Use matplotlib to create a popup scatter plot window with LOWESS trendline.
    """

    x = df["cover_10"]
    y = df["weighted_ndvi"]

    # Filter NaN
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]

    # Fit LOWESS
    lowess = sm.nonparametric.lowess
    z = lowess(y, x, frac=0.4)  # frac controls smoothing

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label="Data", c="lightblue", edgecolors="k")
    plt.plot(z[:, 0], z[:, 1], color="red", linewidth=2, label="LOWESS trend")

    plt.xlabel("Tree Cover (%)")
    plt.ylabel("Mean NDVI (Population-Weighted)")
    plt.title("Tree Cover vs. NDVI with LOWESS Trendline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def select_ne_goal_with_slider_combined(df, return_fig=False):
    """
    Combined LOWESS trendline + interactive slider + confirm button.
    Used to select NE_goal based on Tree Cover (%).
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import statsmodels.api as sm
    import numpy as np

    # Prepare data
    df_sorted = df.sort_values("cover_10")
    x = df_sorted["cover_10"].values
    y = df_sorted["weighted_ndvi"].values

    # Fit LOWESS curve
    lowess = sm.nonparametric.lowess
    z = lowess(y, x, frac=0.4)
    x_lowess = z[:, 0]
    y_lowess = z[:, 1]

    init_val = 30.0
    selected = {"value": None}

    fig, ax = plt.subplots(figsize=(20, 6))
    plt.subplots_adjust(left=0.1, bottom=0.3)

    ax.plot(x, y, 'o', alpha=0.4, label='Data', color='skyblue', markersize=6)
    ax.plot(x_lowess, y_lowess, 'r-', linewidth=2, label='LOWESS')

    vline = ax.axvline(init_val, color='gray', linestyle='--', label='Selected')
    ndvi_val = np.interp(init_val, x_lowess, y_lowess)
    ndvi_text = ax.text(0.05, 0.95, f"NDVI = {ndvi_val:.3f}",
                        transform=ax.transAxes, fontsize=12, va='top')

    ax.set_xlabel("Tree Cover (%)")
    ax.set_ylabel("NDVI")
    ax.set_title("Select Tree Cover to Set NE_goal (NDVI)")
    ax.legend()
    ax.grid(True)

    ax_slider = plt.axes([0.1, 0.15, 0.75, 0.05])
    slider = Slider(ax_slider, "Tree Cover", min(x), max(x), valinit=init_val)

    ax_button = plt.axes([0.4, 0.025, 0.2, 0.05])
    button = Button(ax_button, "Confirm")

    def update(val):
        tree_cover = slider.val
        ndvi_est = np.interp(tree_cover, x_lowess, y_lowess)
        vline.set_xdata([tree_cover])
        ndvi_text.set_text(f"NDVI = {ndvi_est:.3f}")
        fig.canvas.draw_idle()

    def on_confirm(event):
        tree_cover = slider.val
        selected["value"] = float(np.interp(tree_cover, x_lowess, y_lowess))
        print(f"Confirmed Tree Cover = {tree_cover:.2f} → NDVI = {selected['value']:.3f}")
        plt.close(fig)  #  Close only the figure created

    slider.on_changed(update)
    button.on_clicked(on_confirm)

    if return_fig:
        #  Wait until user confirms selection
        while selected["value"] is None:
            plt.pause(0.1)

        return selected["value"], fig
    else:
        plt.show()
        return selected["value"] if selected["value"] is not None else 0.3
