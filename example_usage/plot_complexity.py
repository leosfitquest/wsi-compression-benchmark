"""
plot_complexity.py
------------------
Visualizes the results from results_complexity.csv in two scientific subplots:
  1. Grouped bar chart (log scale): Encoding vs. Decoding throughput (MPix/s)
  2. Pareto scatter plot: Reconstruction quality (PSNR) vs. Decoding speed (MPix/s)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_path = Path("results_complexity.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"File not found: {csv_path}")

df = pd.read_csv(csv_path)

# Color scheme consistent with plot_rd_curves.py
COLORS = {
    "JPEG 4:4:4": "#7f7f7f",
    "JPEG 4:2:0": "#bcbd22",
    "JPEG 2000": "#1f77b4",
    "WebP": "#17becf",
    "JPEG XL": "#2ca02c",
    "AVIF": "#9467bd",
    "bmshj2018-hyperprior": "#ff7f0e",
    "cheng2020-attn": "#d62728",
}

MARKERS = {
    "JPEG 4:4:4": "o",
    "JPEG 4:2:0": "X",
    "JPEG 2000": "s",
    "WebP": "p",
    "JPEG XL": "^",
    "AVIF": "D",
    "bmshj2018-hyperprior": "o",
    "cheng2020-attn": "v",
}


def _fmt_val(val: float) -> str:
    if val >= 10.0:
        return f"{val:.0f}"
    if val >= 1.0:
        return f"{val:.1f}"
    return f"{val:.2f}"


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# ==============================================================================
# 1. Left: Grouped bar chart (Throughput MPix/s on log scale)
# ==============================================================================
labels = df["codec_or_model"].tolist()
x = np.arange(len(labels))
width = 0.36

rects1 = ax1.bar(x - width / 2, df["enc_mpix_s"], width, label="Encoding [MPix/s]", color="#4c72b0", alpha=0.85)
rects2 = ax1.bar(x + width / 2, df["dec_mpix_s"], width, label="Decoding [MPix/s]", color="#55a868", alpha=0.85)

ax1.set_yscale("log")
ax1.set_ylabel("Throughput [MPix/s] (Log Scale)", fontsize=11)
ax1.set_title("Computational Throughput: Encoding vs. Decoding", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)
ax1.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=10)

# Annotate values above bars without scientific notation collisions
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(
        _fmt_val(h),
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=7.5, rotation=0
    )
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(
        _fmt_val(h),
        xy=(rect.get_x() + rect.get_width() / 2, h),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center", va="bottom", fontsize=7.5, rotation=0
    )

ax1.set_ylim(top=ax1.get_ylim()[1] * 3.0)

# ==============================================================================
# 2. Right: Pareto Trade-off (Decoding Speed vs. PSNR)
# ==============================================================================
for _, row in df.iterrows():
    name = row["codec_or_model"]
    color = COLORS.get(name, "black")
    marker = MARKERS.get(name, "o")
    is_neural = row["category"] == "neural"

    ax2.scatter(
        row["dec_mpix_s"],
        row["psnr_db"],
        color=color,
        marker=marker,
        s=110,
        edgecolors="black" if is_neural else "none",
        linewidths=1.2 if is_neural else 0.0,
        label=f"{name} ({row['bpp']:.2f} bpp)",
        zorder=4,
    )

    # Place text label slightly offset next to each point
    ax2.annotate(
        name,
        xy=(row["dec_mpix_s"], row["psnr_db"]),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold" if is_neural or name == "AVIF" else "normal",
    )

ax2.set_xscale("log")
ax2.set_xlabel("Decoding Throughput [MPix/s] (Log Scale -> faster is further right)", fontsize=11)
ax2.set_ylabel("Reconstruction Quality RGB-PSNR [dB]", fontsize=11)
ax2.set_title("Pareto Front: Quality vs. Decoding Speed", fontsize=13, fontweight="bold")
ax2.grid(True, which="both", linestyle=":", alpha=0.6)
ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

# Leave space on the right for annotations
ax2.set_xlim(right=ax2.get_xlim()[1] * 4.0)

plt.suptitle("WSI Computational Complexity Benchmark (256x256 Tiles, CPU Single-Pass)", fontsize=14, y=0.98)
plt.tight_layout()

out_plot = Path("complexity_comparison.png")
plt.savefig(out_plot, dpi=300)
print(f"Complexity plot successfully saved: {out_plot.resolve()}")