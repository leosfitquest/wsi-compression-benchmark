"""
plot_rd_curves_curated.py
-------------------------
Plottet die Rate-Distortion-Kurven des Curated Test Sets v1 (Track A - IDC RMS):
  1. rd_curves_track_a_curated.png: 3 Panels (RGB-PSNR, Y-PSNR, MS-SSIM)
  2. rd_curves_track_a_s1_vs_s2.png: Stratifizierter Vergleich S1 (Pink) vs. S2 (Blau)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

conv_path = Path("results/curated_v1/results_rms_conv.csv")
ai_path = Path("results/curated_v1/results_rms_ai.csv")
manifest_path = Path("results/curated_v1/manifest_track_a.csv")

if not conv_path.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {conv_path}")

df_conv = pd.read_csv(conv_path)
df_ai = pd.read_csv(ai_path) if ai_path.exists() else pd.DataFrame()

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

REF_TARGET_BPP = 0.28


def _plot_three_panels(df_c: pd.DataFrame, df_a: pd.DataFrame, title: str, out_file: Path) -> None:
    conv_grouped = (
        df_c.groupby(["codec", "target_bpp"])
        .filter(lambda g: abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15)
        .groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "psnr_y_db", "ms_ssim"]]
        .mean()
        .reset_index()
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)

    for codec, group in conv_grouped.groupby("codec"):
        group = group.sort_values("bpp")
        group = group.loc[group["bpp"].round(3).drop_duplicates(keep="last").index]
        st = styles.get(str(codec), {"color": "black", "marker": ".", "label": str(codec)})
        ax1.plot(group["bpp"], group["psnr_db"], **st, linewidth=1.8)
        ax2.plot(group["bpp"], group["psnr_y_db"], **st, linewidth=1.8)
        ax3.plot(group["bpp"], group["ms_ssim"], **st, linewidth=1.8)

    if not df_a.empty:
        ai_grouped = df_a.groupby(["model", "quality"])[["bpp_real", "psnr_db", "psnr_y_db", "ms_ssim"]].mean().reset_index()
        for model, group in ai_grouped.groupby("model"):
            group = group.sort_values("bpp_real")
            st = ai_styles.get(str(model), {"color": "black", "marker": "x", "label": str(model)})
            ax1.plot(group["bpp_real"], group["psnr_db"], **st, linewidth=1.8, linestyle="--")
            ax2.plot(group["bpp_real"], group["psnr_y_db"], **st, linewidth=1.8, linestyle="--")
            ax3.plot(group["bpp_real"], group["ms_ssim"], **st, linewidth=1.8, linestyle="--")

    for ax in (ax1, ax2, ax3):
        ax.axvline(
            x=REF_TARGET_BPP,
            color="#444444",
            linestyle="--",
            linewidth=1.3,
            alpha=0.75,
            label="Reference Target (0.28 bpp, ~85:1)",
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("Bitrate [bpp]", fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax1.set_title("Rate-Distortion: RGB-PSNR", fontsize=12.5, fontweight="bold")
    ax1.set_ylabel("RGB-PSNR [dB]", fontsize=11)

    ax2.set_title("Rate-Distortion: Luminance Y-PSNR", fontsize=12.5, fontweight="bold")
    ax2.set_ylabel("Y-PSNR (BT.601) [dB]", fontsize=11)

    ax3.set_title("Rate-Distortion: MS-SSIM", fontsize=12.5, fontweight="bold")
    ax3.set_ylabel("MS-SSIM", fontsize=11)

    plt.suptitle(title, fontsize=13.5, y=0.99)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Plot gespeichert: {out_file.resolve()}")


def _plot_s1_vs_s2(df_c: pd.DataFrame, df_a: pd.DataFrame, mf: pd.DataFrame, out_file: Path) -> None:
    s1_ids = set(mf[mf["stain_stratum"] == "S1_eosinophilic"]["tile_id"].astype(str))
    s2_ids = set(mf[mf["stain_stratum"] == "S2_basophilic"]["tile_id"].astype(str))

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300, sharex=True)
    strata = [
        ("S1_eosinophilic (Pink Stroma/Cytoplasm Dominant)", s1_ids, axes[0, 0], axes[0, 1]),
        ("S2_basophilic (Blue Hematoxylin/Nuclei Dominant)", s2_ids, axes[1, 0], axes[1, 1]),
    ]

    for label_str, id_set, ax_psnr, ax_ssim in strata:
        sub_c = df_c[df_c["tile_id"].isin(id_set)]
        conv_grouped = (
            sub_c.groupby(["codec", "target_bpp"])
            .filter(lambda g: abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15)
            .groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "ms_ssim"]]
            .mean()
            .reset_index()
        )
        for codec, group in conv_grouped.groupby("codec"):
            group = group.sort_values("bpp")
            group = group.loc[group["bpp"].round(3).drop_duplicates(keep="last").index]
            st = styles.get(str(codec), {"color": "black", "marker": ".", "label": str(codec)})
            ax_psnr.plot(group["bpp"], group["psnr_db"], **st, linewidth=1.8)
            ax_ssim.plot(group["bpp"], group["ms_ssim"], **st, linewidth=1.8)

        if not df_a.empty:
            sub_a = df_a[df_a["tile_id"].isin(id_set)]
            if not sub_a.empty:
                ai_grouped = sub_a.groupby(["model", "quality"])[["bpp_real", "psnr_db", "ms_ssim"]].mean().reset_index()
                for model, group in ai_grouped.groupby("model"):
                    group = group.sort_values("bpp_real")
                    st = ai_styles.get(str(model), {"color": "black", "marker": "x", "label": str(model)})
                    ax_psnr.plot(group["bpp_real"], group["psnr_db"], **st, linewidth=1.8, linestyle="--")
                    ax_ssim.plot(group["bpp_real"], group["ms_ssim"], **st, linewidth=1.8, linestyle="--")

        for ax in (ax_psnr, ax_ssim):
            ax.axvline(x=REF_TARGET_BPP, color="#444444", linestyle="--", linewidth=1.2, alpha=0.75, label="Reference Target (0.28 bpp)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_xlabel("Bitrate [bpp]", fontsize=10.5)
            ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

        ax_psnr.set_title(f"{label_str} — RGB-PSNR", fontsize=11.5, fontweight="bold")
        ax_psnr.set_ylabel("RGB-PSNR [dB]", fontsize=10.5)
        ax_ssim.set_title(f"{label_str} — MS-SSIM", fontsize=11.5, fontweight="bold")
        ax_ssim.set_ylabel("MS-SSIM", fontsize=10.5)

    plt.suptitle("Ruifrok H&E Stain Stratification: S1 (Eosinophilic) vs. S2 (Basophilic)", fontsize=14, y=0.99)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Stratifizierter S1/S2-Plot gespeichert: {out_file.resolve()}")


if __name__ == "__main__":
    n_tiles = df_conv["tile_id"].nunique()
    _plot_three_panels(
        df_conv,
        df_ai,
        f"Curated Track A Benchmark (IDC RMS Lossless .90, n={n_tiles} Level-0 Tiles)",
        Path("results/curated_v1/rd_curves_track_a_curated.png"),
    )
    if manifest_path.exists():
        mf_df = pd.read_csv(manifest_path)
        _plot_s1_vs_s2(
            df_conv,
            df_ai,
            mf_df,
            Path("results/curated_v1/rd_curves_track_a_s1_vs_s2.png"),
        )