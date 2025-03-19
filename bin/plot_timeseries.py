import os
import pandas as pd
import matplotlib.pyplot as plt
import math


# Helper function: Load and preprocess CSV
def load_and_preprocess_csv(filename, required_columns):
    """Loads CSV, checks required columns, converts date column, and sorts by date."""
    if not os.path.exists(filename):
        print(f"Skipping: File not found - {filename}")
        return None

    df = pd.read_csv(filename)

    if not required_columns.issubset(df.columns):
        print(f"Skipping: Missing required columns {required_columns} in {filename}")
        return None

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(by="date")


# Helper function: Highlight fire phases on a plot
def highlight_fire_phases(ax, pre_fire, fire, post_fire):
    """Highlights pre-fire, active fire, and post-fire periods on a plot."""
    pre_fire_start, pre_fire_end = map(pd.to_datetime, pre_fire)
    fire_start, fire_end = map(pd.to_datetime, fire)
    post_fire_start, post_fire_end = map(pd.to_datetime, post_fire)

    ax.axvspan(pre_fire_start, pre_fire_end, color="blue", alpha=0.15, label="Pre-Fire")
    ax.axvspan(fire_start, fire_end, color="red", alpha=0.3, label="Active Fire")
    ax.axvspan(post_fire_start, post_fire_end, color="green", alpha=0.15, label="Post-Fire")


# Function to plot NDVI timeseries with fire phases highlighted
def plot_ndvi_timeseries(filename, experiment_path, pre_fire, fire, post_fire, figsize=(8, 5)):
    """Plots NDVI timeseries for each parcel with fire phases highlighted."""
    
    df = load_and_preprocess_csv(filename, {"date", "feature_index", "band_unnamed"})
    if df is None:
        return

    plots_dir = os.path.join(experiment_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    for feature_id, group in df.groupby("feature_index"):
        ax.plot(group["date"], group["band_unnamed"], linestyle="-", label=f"Parcel {feature_id}")

    highlight_fire_phases(ax, pre_fire, fire, post_fire)

    ax.set(title="NDVI Timeseries (Fire Phases Highlighted)", xlabel="Date", ylabel="NDVI", ylim=(0, 1))
    ax.legend(title="Parcel ID", loc="upper left", ncol=2)
    plt.xticks(rotation=45)

    plot_path = os.path.join(plots_dir, "ndvi_fire_phases.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    print(f"Plot saved at: {plot_path}")


# Function to plot Sentinel-5P timeseries with fire phases highlighted
def plot_s5p_timeseries(filename, experiment_path, pre_fire, fire, post_fire, figsize=(12, 8), use_log_scale=True):
    """Plots Sentinel-5P timeseries with fire phases highlighted."""
    
    df = load_and_preprocess_csv(filename, {"date"})
    if df is None:
        return

    df.interpolate(method="linear", inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].map(lambda x: max(x, 1e-6))

    plots_dir = os.path.join(experiment_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    gas_bands = [col for col in df.columns if col != "date"]
    rows, cols = math.ceil(len(gas_bands) / 2), min(2, len(gas_bands))
    colors = plt.get_cmap("tab10").colors  

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=100, constrained_layout=True)
    axes = axes.flatten()

    for i, band in enumerate(gas_bands):
        ax = axes[i]
        ax.plot(df["date"], df[band], linestyle="-", linewidth=2, color=colors[i % len(colors)], label=band)
        highlight_fire_phases(ax, pre_fire, fire, post_fire)

        ax.set(title=f"{band} Concentration", xlabel="Date", ylabel="Scaled Value")
        ax.legend(loc="upper left")
        if use_log_scale:
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plot_path = os.path.join(plots_dir, "s5p_fire_phases.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
    print(f"Plot saved at: {plot_path}")

