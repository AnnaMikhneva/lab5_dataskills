

import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import gaussian_kde


#  Style

FORMATS      = ["float64", "float32", "float16", "int8"]
FORMAT_LABEL = {
    "float64": "float64\n(reference)",
    "float32": "float32",
    "float16": "float16",
    "int8":    "int8\n(quantised)",
}
COLORS = {
    "float64": "#2196F3",   # blue
    "float32": "#4CAF50",   # green
    "float16": "#FF9800",   # orange
    "int8":    "#F44336",   # red
}
INTRA_COLOR = "#1565C0"    # dark blue  (intra-speaker)
INTER_COLOR = "#B71C1C"    # dark red   (inter-speaker)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def kde_plot(ax, data, color, label, bw_adjust=0.8):
    data = data[np.isfinite(data)]
    if len(data) < 5:
        ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center")
        return
    try:
        kde = gaussian_kde(data, bw_method=bw_adjust)
        x   = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, kde(x), color=color, label=label, linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)
    except Exception:
        ax.hist(data, bins=30, color=color, alpha=0.4, label=label, density=True)



# Figure 1: KDE distributions

def plot_kde(distances_dir: Path, plots_dir: Path, params: dict):
    bw = params.get("kde_bw_adjust", 0.8)
    dpi = params.get("figure_dpi", 150)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax, fmt in zip(axes, FORMATS):
        intra_path = distances_dir / f"intra_dists_{fmt}.npy"
        inter_path = distances_dir / f"inter_dists_{fmt}.npy"

        if not intra_path.exists():
            ax.set_title(f"{fmt} — data missing")
            continue

        intra = np.load(intra_path)
        inter = np.load(inter_path)

        kde_plot(ax, intra, INTRA_COLOR, "Intra-speaker", bw_adjust=bw)
        kde_plot(ax, inter, INTER_COLOR, "Inter-speaker", bw_adjust=bw)

        ax.set_title(FORMAT_LABEL[fmt], fontsize=12, fontweight="bold",
                     color=COLORS[fmt])
        ax.set_xlabel("Cosine distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Distribution of Cosine Distances by Precision Level\n"
        "(Intra = same speaker / same word / different recording;\n"
        " Inter = different speakers / same word)",
        fontsize=11,
    )
    fig.tight_layout()
    out = plots_dir / "distances_kde.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# Figure 2: mean distances with error bars

def plot_bar(summary_df: pd.DataFrame, plots_dir: Path, params: dict):
    dpi = params.get("figure_dpi", 150)

    x    = np.arange(len(FORMATS))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_intra = ax.bar(
        x - w/2,
        summary_df["mean_intra"],
        w,
        yerr=summary_df["std_intra"],
        label="Intra-speaker",
        color=INTRA_COLOR,
        alpha=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.5},
    )
    bars_inter = ax.bar(
        x + w/2,
        summary_df["mean_inter"],
        w,
        yerr=summary_df["std_inter"],
        label="Inter-speaker",
        color=INTER_COLOR,
        alpha=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.5},
    )

    # Annotate bars with values
    for bar in list(bars_intra) + list(bars_inter):
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.002,
                f"{h:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([FORMAT_LABEL[f].replace("\n", " ") for f in FORMATS])
    ax.set_ylabel("Mean cosine distance")
    ax.set_title("Mean Intra- vs Inter-Speaker Cosine Distances\nby Numerical Precision (error bars = ±1 std)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = plots_dir / "distances_bar.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")



# Figure 3: Intra/inter ratio evolution

def plot_ratio(summary_df: pd.DataFrame, plots_dir: Path, params: dict):
    dpi = params.get("figure_dpi", 150)

    fig, ax = plt.subplots(figsize=(8, 4))

    ratios = summary_df["ratio"].values
    x      = np.arange(len(FORMATS))

    ax.plot(x, ratios, marker="o", linewidth=2.5, markersize=8,
            color="#5C4033", label="Intra/Inter ratio")

    for xi, yi, fmt in zip(x, ratios, FORMATS):
        if np.isfinite(yi):
            ax.annotate(f"{yi:.4f}", (xi, yi),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=COLORS[fmt], fontweight="bold")

    # Reference line at ratio = 1
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2,
               label="ratio = 1 (intra = inter)")

    ax.set_xticks(x)
    ax.set_xticklabels([FORMAT_LABEL[f].replace("\n", " ") for f in FORMATS])
    ax.set_ylabel("Intra / Inter distance ratio")
    ax.set_title(
        "Evolution of Intra/Inter-Speaker Distance Ratio\n"
        "as Numerical Precision Decreases\n"
        "(ratio < 1 ⟹ same-speaker words are closer → structure preserved)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1.5, ratios[np.isfinite(ratios)].max() * 1.2) if len(ratios[np.isfinite(ratios)]) > 0 else 2)

    fig.tight_layout()
    out = plots_dir / "ratio_evolution.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# Figure 4: Deviation from float64 reference

def plot_deviation(summary_df: pd.DataFrame, plots_dir: Path, params: dict):
    dpi = params.get("figure_dpi", 150)

    # Exclude float64 itself (deviation = 0 by definition)
    df_plot = summary_df[summary_df["format"] != "float64"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, col, title, ylabel in [
        (axes[0], "mean_dev_from_f64", "Mean Absolute Deviation\nfrom float64 reference", "Mean |Δdistance|"),
        (axes[1], "max_dev_from_f64",  "Maximum Absolute Deviation\nfrom float64 reference", "Max |Δdistance|"),
    ]:
        colors = [COLORS[f] for f in df_plot["format"]]
        bars   = ax.bar(df_plot["format"], df_plot[col], color=colors, alpha=0.85)

        for bar, val in zip(bars, df_plot[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f"{val:.2e}", ha="center", va="bottom", fontsize=9)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Numerical Error Introduced by Precision Reduction", fontsize=11)
    fig.tight_layout()
    out = plots_dir / "deviation_from_ref.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# Figure 5: Memory and time

def plot_memory_time(
    sizes: dict, timing: dict, plots_dir: Path, params: dict
):
    dpi = params.get("figure_dpi", 150)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    fmts   = [f for f in FORMATS if f in sizes]
    colors = [COLORS[f] for f in fmts]

    # Memory
    mem_vals = [sizes[f] for f in fmts]
    bars = ax1.bar([FORMAT_LABEL[f].replace("\n"," ") for f in fmts],
                   mem_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, mem_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{val:.1f} MB", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Disk Space per Precision Format")
    ax1.set_ylabel("Size (MB)")
    ax1.grid(True, axis="y", alpha=0.3)

    # Time
    time_vals = [timing.get(f, np.nan) for f in fmts]
    bars = ax2.bar([FORMAT_LABEL[f].replace("\n"," ") for f in fmts],
                   time_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, time_vals):
        if np.isfinite(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                     f"{val:.2f}s", ha="center", va="bottom", fontsize=9)
    ax2.set_title("Distance Matrix Computation Time")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Efficiency Trade-Off: Memory vs Compute Time", fontsize=11)
    fig.tight_layout()
    out = plots_dir / "memory_time.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Produce all visualisations")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)

    distances_dir = Path("data/distances")
    features_dir  = Path("data/features")
    plots_dir     = Path("data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load summary stats
    summary_path = distances_dir / "summary_stats.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run stage 4 first: python src/04_compute_distances.py")
    summary_df = pd.read_csv(summary_path)

    # Load timing and sizes
    with open(distances_dir / "timing.json") as f:
        timing = json.load(f)
    with open(features_dir / "precision_sizes.json") as f:
        sizes = json.load(f)

    print("Generating figures…")
    plot_kde(distances_dir, plots_dir, params)
    plot_bar(summary_df, plots_dir, params)
    plot_ratio(summary_df, plots_dir, params)
    plot_deviation(summary_df, plots_dir, params)
    plot_memory_time(sizes, timing, plots_dir, params)

    print(f"\nAll figures saved under {plots_dir}/")


if __name__ == "__main__":
    main()
