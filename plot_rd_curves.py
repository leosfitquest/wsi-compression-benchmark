"""
plot_rd_curves.py
-----------------
Liest results_rms_conv.csv und results_rms_ai.csv ein, berechnet die
Mittelwerte pro Bitrate/Qualitaetsstufe und plottet R-D-Kurven (bpp vs. PSNR / MS-SSIM)
fuer alle 6 konventionellen Codecs, neuronale Modelle und die Scanner-Baseline.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

conv_path = Path("results/pilot/results_rms_conv.csv")
ai_path = Path("results/pilot/results_rms_ai.csv")

if not conv_path.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {conv_path}")

df_conv = pd.read_csv(conv_path)
conv_grouped = df_conv.groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "ms_ssim"]].mean().reset_index()

# Farb-, Marker- und Linien-Styles fuer alle 6 konventionellen Codecs
styles = {
    "AVIF":       {"color": "#9467bd", "marker": "D", "label": "AVIF (libavif)"},
    "JPEG XL":    {"color": "#2ca02c", "marker": "^", "label": "JPEG XL (libjxl)"},
    "WebP":       {"color": "#17becf", "marker": "p", "label": "WebP (libwebp)"},
    "JPEG 2000":  {"color": "#1f77b4", "marker": "s", "label": "JPEG 2000 (OpenJPEG)"},
    "JPEG 4:4:4": {"color": "#7f7f7f", "marker": "o", "label": "JPEG 4:4:4"},
    "JPEG 4:2:0": {"color": "#bcbd22", "marker": "x", "label": "JPEG 4:2:0", "linestyle": ":"},
}

# Styles fuer neuronale CompressAI-Modelle
ai_styles = {
    "bmshj2018-hyperprior": {"color": "#ff7f0e", "marker": "o", "label": "Hyperprior (Ballé 2018)"},
    "cheng2020-attn":       {"color": "#d62728", "marker": "v", "label": "Cheng2020-Attention"},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# 1. Konventionelle Kurven einzeichnen (gesaettigte Duplikat-Punkte am unteren Rand deduplizieren)
for codec, group in conv_grouped.groupby("codec"):
    group = group.sort_values("bpp")
    group = group.loc[group["bpp"].round(3).drop_duplicates(keep="last").index]
    st = styles.get(codec, {"color": "black", "marker": ".", "label": codec})
    ax1.plot(group["bpp"], group["psnr_db"], **st, linewidth=1.8)
    ax2.plot(group["bpp"], group["ms_ssim"], **st, linewidth=1.8)

# 2. CompressAI Kurven einzeichnen (falls vorhanden)
if ai_path.exists():
    df_ai = pd.read_csv(ai_path)
    ai_grouped = df_ai.groupby(["model", "quality"])[["bpp_real", "psnr_db", "ms_ssim"]].mean().reset_index()
    for model, group in ai_grouped.groupby("model"):
        group = group.sort_values("bpp_real")
        st = ai_styles.get(model, {"color": "black", "marker": "x", "label": model})
        ax1.plot(group["bpp_real"], group["psnr_db"], **st, linewidth=1.8, linestyle="--")
        ax2.plot(group["bpp_real"], group["ms_ssim"], **st, linewidth=1.8, linestyle="--")
else:
    print(f"[info] {ai_path} nicht gefunden – es werden nur konventionelle Codecs dargestellt.")

# 3. Klinische Scanner-Baseline (Q=70, ~0.28 bpp)
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

# Achsenbeschriftungen & Formatierung
ax1.set_title("Rate-Distortion: PSNR", fontsize=13, fontweight="bold")
ax1.set_ylabel("RGB-PSNR [dB]", fontsize=11)

ax2.set_title("Rate-Distortion: MS-SSIM", fontsize=13, fontweight="bold")
ax2.set_ylabel("MS-SSIM", fontsize=11)

plt.suptitle("WSI Ground-Truth Benchmark (IDC RMS Cohort, JPEG 2000 Lossless .90)", fontsize=14, y=0.98)
plt.tight_layout()

out_plot = Path("results/pilot/rd_curves_rms_groundtruth.png")
plt.savefig(out_plot, dpi=300)
print(f"R-D Kurven erfolgreich generiert und gespeichert: {out_plot.resolve()}")