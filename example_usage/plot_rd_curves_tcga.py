"""
plot_rd_curves_tcga.py
Erstellt R-D-Kurven (bpp vs. PSNR / MS-SSIM) fuer den vorkomprimierten
TCGA-Datensatz (results_tcga256_conv.csv und results_tcga256_ai.csv).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

conv_path = Path("results_tcga256_conv.csv")
ai_path = Path("results_tcga256_ai.csv")

if not conv_path.exists() or not ai_path.exists():
    raise FileNotFoundError("Erforderliche CSV-Dateien nicht gefunden (results_tcga256_conv.csv / results_tcga256_ai.csv).")

df_conv = pd.read_csv(conv_path)
df_ai = pd.read_csv(ai_path)

# Mittelwerte ueber alle 5 Kacheln berechnen
conv_grouped = df_conv.groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "ms_ssim"]].mean().reset_index()
ai_grouped = df_ai.groupby(["model", "quality"])[["bpp_real", "psnr_db", "ms_ssim"]].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Farb-, Marker- und Linien-Styles konsistent zu plot_rd_curves.py
styles = {
    "AVIF":       {"color": "#9467bd", "marker": "D", "label": "AVIF (libavif)"},
    "JPEG XL":    {"color": "#2ca02c", "marker": "^", "label": "JPEG XL (libjxl)"},
    "WebP":       {"color": "#17becf", "marker": "p", "label": "WebP (libwebp)"},
    "JPEG 2000":  {"color": "#1f77b4", "marker": "s", "label": "JPEG 2000 (OpenJPEG)"},
    "JPEG 4:4:4": {"color": "#7f7f7f", "marker": "o", "label": "JPEG 4:4:4"},
    "JPEG 4:2:0": {"color": "#bcbd22", "marker": "x", "label": "JPEG 4:2:0", "linestyle": ":"},
}

ai_styles = {
    "bmshj2018-hyperprior": {"color": "#ff7f0e", "marker": "o", "label": "Hyperprior (Ballé 2018)"},
    "cheng2020-attn":       {"color": "#d62728", "marker": "v", "label": "Cheng2020-Attention"},
}

# 1. Konventionelle Kurven (gesaettigte Duplikat-Punkte am unteren Rand deduplizieren)
for codec, group in conv_grouped.groupby("codec"):
    group = group.sort_values("bpp")
    group = group.loc[group["bpp"].round(3).drop_duplicates(keep="last").index]
    st = styles.get(codec, {"color": "gray", "marker": "x", "label": codec})
    ax1.plot(group["bpp"], group["psnr_db"], **st, linewidth=1.8)
    ax2.plot(group["bpp"], group["ms_ssim"], **st, linewidth=1.8)

# 2. CompressAI Kurven
for model, group in ai_grouped.groupby("model"):
    group = group.sort_values("bpp_real")
    st = ai_styles.get(model, {"color": "black", "marker": "x", "label": model})
    ax1.plot(group["bpp_real"], group["psnr_db"], **st, linewidth=1.8, linestyle="--")
    ax2.plot(group["bpp_real"], group["ms_ssim"], **st, linewidth=1.8, linestyle="--")

# 3. Klinische Scanner-Baseline (Q=70, ~0.28 bpp) auf beiden Achsen
SCANNER_BASELINE_BPP = 0.28
for ax in (ax1, ax2):
    ax.axvline(
        x=SCANNER_BASELINE_BPP,
        color="#444444",
        linestyle="--",
        linewidth=1.3,
        alpha=0.75,
        label="Scanner Baseline (Q=70, ~0.28 bpp)"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Bitrate [bpp]", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

# Achsenbeschriftungen & Layout
ax1.set_title("Rate-Distortion: PSNR (TCGA Pre-Compressed)", fontsize=13, fontweight="bold")
ax1.set_ylabel("RGB-PSNR [dB]", fontsize=11)

ax2.set_title("Rate-Distortion: MS-SSIM (TCGA Pre-Compressed)", fontsize=13, fontweight="bold")
ax2.set_ylabel("MS-SSIM", fontsize=11)

plt.suptitle("WSI Pilot Benchmark (TCGA-LUAD, Scanner JPEG Q=70 Baseline)", fontsize=14, y=0.98)
plt.tight_layout()

out_plot = Path("rd_curves_tcga_precompressed.png")
plt.savefig(out_plot, dpi=300)
print(f"TCGA R-D Kurven erfolgreich generiert: {out_plot.resolve()}")