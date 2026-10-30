"""
plot_rd_curves.py
Liest results_rms_conv.csv und results_rms_ai.csv ein, berechnet die
Mittelwerte pro Bitrate/Qualitaetsstufe und plottet R-D-Kurven (bpp vs. PSNR / MS-SSIM).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

conv_path = Path("results_rms_conv.csv")
ai_path = Path("results_rms_ai.csv")

if not conv_path.exists() or not ai_path.exists():
    raise FileNotFoundError("Bitte stelle sicher, dass results_rms_conv.csv und results_rms_ai.csv existieren.")

df_conv = pd.read_csv(conv_path)
df_ai = pd.read_csv(ai_path)

# Mittelwerte ueber alle Kacheln berechnen
# Konventionell: gruppiert nach Codec und Ziel-Bitrate
conv_grouped = df_conv.groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "ms_ssim"]].mean().reset_index()

# CompressAI: gruppiert nach Modell und Quality-Level
ai_grouped = df_ai.groupby(["model", "quality"])[["bpp_real", "psnr_db", "ms_ssim"]].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Farb- und Marker-Mapping
styles = {
    "JPEG 4:4:4": {"color": "#1f77b4", "marker": "o", "label": "JPEG 4:4:4"},
    "JPEG 4:2:0": {"color": "#aec7e8", "marker": "s", "label": "JPEG 4:2:0"},
    "JPEG 2000":  {"color": "#2ca02c", "marker": "^", "label": "JPEG 2000 (OpenJPEG)"},
}

ai_styles = {
    "bmshj2018-hyperprior": {"color": "#ff7f0e", "marker": "D", "label": "Hyperprior (Ballé 2018)"},
    "cheng2020-attn":       {"color": "#d62728", "marker": "v", "label": "Cheng2020-Attention"},
}

# 1. Konventionelle Kurven plotten
for codec, group in conv_grouped.groupby("codec"):
    group = group.sort_values("bpp")
    st = styles.get(codec, {"color": "gray", "marker": "x", "label": codec})
    ax1.plot(group["bpp"], group["psnr_db"], **st, linewidth=1.8)
    ax2.plot(group["bpp"], group["ms_ssim"], **st, linewidth=1.8)

# 2. CompressAI Kurven plotten
for model, group in ai_grouped.groupby("model"):
    group = group.sort_values("bpp_real")
    st = ai_styles.get(model, {"color": "black", "marker": "x", "label": model})
    ax1.plot(group["bpp_real"], group["psnr_db"], **st, linewidth=1.8, linestyle="--")
    ax2.plot(group["bpp_real"], group["ms_ssim"], **st, linewidth=1.8, linestyle="--")

# Achsenbeschriftungen & Formatierung
ax1.set_title("Rate-Distortion: PSNR", fontsize=13, fontweight="bold")
ax1.set_xlabel("Bitrate [bpp]", fontsize=11)
ax1.set_ylabel("RGB-PSNR [dB]", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
# ==============================================================================
# NEU: Gestrichelte Referenzlinie fuer die klinische Scanner-Baseline (Q=70)
# ==============================================================================
SCANNER_BASELINE_BPP = 0.28
ax1.axvline(
    x=SCANNER_BASELINE_BPP,
    color="#555555",
    linestyle="--",
    linewidth=1.2,
    alpha=0.75,
    label="Scanner Baseline (Q=70, ~0.28 bpp)"
)

# ==============================================================================
ax1.legend(loc="lower right")

ax2.set_title("Rate-Distortion: MS-SSIM", fontsize=13, fontweight="bold")
ax2.set_xlabel("Bitrate [bpp]", fontsize=11)
ax2.set_ylabel("MS-SSIM", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right")

plt.suptitle("WSI Ground-Truth Benchmark (IDC RMS Cohort, JPEG 2000 Lossless .90)", fontsize=14, y=0.98)
plt.tight_layout()  

out_plot = Path("rd_curves_rms_groundtruth.png")
plt.savefig(out_plot, dpi=300)
print(f"R-D Kurven erfolgreich generiert und gespeichert: {out_plot.resolve()}")