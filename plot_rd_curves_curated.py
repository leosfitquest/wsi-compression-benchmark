"""
plot_rd_curves_curated.py
-------------------------
Plots Rate-Distortion curves for Track A (legacy curated_v1 or unified 12-codec
manifest runs):
- Removes legacy 0.28 bpp vertical reference line
- Fixes MS-SSIM right-edge / legend clipping via explicit xlim + bbox_inches="tight"
- Automatically plots 2x2 (RGB-PSNR, Y-PSNR BT.709, YUV-PSNR 6:1:1, MS-SSIM) when
  `psnr_yuv_db` is available, or clean 1x3 on legacy CSVs
- Supports `--csvs` + `--manifest` to generate:
  (1) Overall R-D, (2) Lossless vs. Lossy R-D, (3) Organ & Stain R-D
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from bdrate_summary import build_curves_from_df, load_codec_csvs, load_unified_manifest

CODEC_STYLES: dict[str, dict[str, object]] = {
    "AVIF":                 {"color": "#9467bd", "marker": "D", "label": "AVIF (libavif, 4:4:4)", "linestyle": "-"},
    "JPEG XL":              {"color": "#2ca02c", "marker": "^", "label": "JPEG XL (libjxl)", "linestyle": "-"},
    "WebP":                 {"color": "#17becf", "marker": "p", "label": "WebP (libwebp, 4:2:0)", "linestyle": "-"},
    "JPEG 2000":            {"color": "#1f77b4", "marker": "s", "label": "JPEG 2000 (OpenJPEG)", "linestyle": "-"},
    "JPEG 4:4:4":           {"color": "#7f7f7f", "marker": "o", "label": "JPEG 4:4:4", "linestyle": "-"},
    "JPEG 4:2:0":           {"color": "#bcbd22", "marker": "x", "label": "JPEG 4:2:0", "linestyle": ":"},
    "HM All-Intra":         {"color": "#8c564b", "marker": "P", "label": "HM All-Intra (HEVC 4:4:4)", "linestyle": "-."},
    "VTM All-Intra":        {"color": "#1b9e77", "marker": "*", "label": "VTM All-Intra (VVC 4:4:4)", "linestyle": "-."},
    "JPEG AI VM (BOP)":     {"color": "#e377c2", "marker": "H", "label": "JPEG AI VM (BOP)", "linestyle": "--"},
    "MLIC++":               {"color": "#000000", "marker": "h", "label": "MLIC++ (Neural)", "linestyle": "--"},
    "bmshj2018-hyperprior": {"color": "#ff7f0e", "marker": "o", "label": "Hyperprior (Ball? 2018)", "linestyle": "--"},
    "cheng2020-attn":       {"color": "#d62728", "marker": "v", "label": "Cheng2020-Attention", "linestyle": "--"},
}


def _style_for(codec: str) -> dict[str, object]:
    return CODEC_STYLES.get(codec, {"color": "#333333", "marker": ".", "label": codec, "linestyle": "-"})


def plot_curves_panel(curves: dict[str, pd.DataFrame], title: str, out_file: Path) -> None:
    if not curves:
        return
    has_yuv = any("psnr_yuv_db" in df.columns and df["psnr_yuv_db"].notna().any() for df in curves.values())
    max_bpp = max(float(df["bpp"].max()) for df in curves.values())
    legend_cols = 2 if len(curves) > 8 else 1

    if has_yuv:
        fig, axes_grid = plt.subplots(2, 2, figsize=(15.5, 11.0), dpi=300)
        panels = [
            (axes_grid[0, 0], "psnr_db", "Rate-Distortion: RGB-PSNR", "RGB-PSNR [dB]"),
            (axes_grid[0, 1], "psnr_y_db", "Rate-Distortion: Luminance Y-PSNR (BT.709)", "Y-PSNR (BT.709) [dB]"),
            (axes_grid[1, 0], "psnr_yuv_db", "Rate-Distortion: Weighted YUV-PSNR (6:1:1)", "YUV-PSNR (BT.709 6:1:1) [dB]"),
            (axes_grid[1, 1], "ms_ssim", "Rate-Distortion: MS-SSIM", "MS-SSIM"),
        ]
    else:
        fig, axes_row = plt.subplots(1, 3, figsize=(19.5, 6.0), dpi=300)
        panels = [
            (axes_row[0], "psnr_db", "Rate-Distortion: RGB-PSNR", "RGB-PSNR [dB]"),
            (axes_row[1], "psnr_y_db", "Rate-Distortion: Luminance Y-PSNR", "Y-PSNR [dB]"),
            (axes_row[2], "ms_ssim", "Rate-Distortion: MS-SSIM", "MS-SSIM"),
        ]

    for codec, grp in sorted(curves.items()):
        st = _style_for(codec)
        for ax, col, _, _ in panels:
            if col in grp.columns and grp[col].notna().any():
                ax.plot(grp["bpp"], grp[col], **st, linewidth=1.8, markersize=5.5)

    for ax, _, ptitle, ylabel in panels:
        ax.set_title(ptitle, fontsize=12.0, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11.0)
        ax.set_xlabel("Bitrate [bpp]", fontsize=11.0)
        ax.set_xlim(left=0.0, right=max_bpp * 1.07)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92, ncol=legend_cols)

    fig.suptitle(title, fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Plot gespeichert: {out_file.resolve()}")


def plot_s1_vs_s2(df_all: pd.DataFrame, mf: pd.DataFrame, out_file: Path) -> None:
    if "stain_stratum" not in mf.columns:
        return
    s1_ids = set(mf[mf["stain_stratum"] == "S1_eosinophilic"]["tile_id"].astype(str))
    s2_ids = set(mf[mf["stain_stratum"] == "S2_basophilic"]["tile_id"].astype(str))
    c_s1 = build_curves_from_df(df_all, allowed_tiles=s1_ids, strict_pairing=True)
    c_s2 = build_curves_from_df(df_all, allowed_tiles=s2_ids, strict_pairing=True)
    if not c_s1 or not c_s2:
        return

    max_bpp = max(max(float(d["bpp"].max()) for d in c_s1.values()), max(float(d["bpp"].max()) for d in c_s2.values()))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0), dpi=300, sharex=True)
    strata = [
        ("S1_eosinophilic (Pink Stroma/Cytoplasm Dominant)", c_s1, axes[0, 0], axes[0, 1]),
        ("S2_basophilic (Blue Hematoxylin/Nuclei Dominant)", c_s2, axes[1, 0], axes[1, 1]),
    ]
    for label_str, curves, ax_psnr, ax_ssim in strata:
        for codec, grp in sorted(curves.items()):
            st = _style_for(codec)
            ax_psnr.plot(grp["bpp"], grp["psnr_db"], **st, linewidth=1.8, markersize=5.5)
            ax_ssim.plot(grp["bpp"], grp["ms_ssim"], **st, linewidth=1.8, markersize=5.5)
        for ax in (ax_psnr, ax_ssim):
            ax.set_xlim(left=0.0, right=max_bpp * 1.07)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_xlabel("Bitrate [bpp]", fontsize=10.5)
            ax.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
        ax_psnr.set_title(f"{label_str} ? RGB-PSNR", fontsize=11.5, fontweight="bold")
        ax_psnr.set_ylabel("RGB-PSNR [dB]", fontsize=10.5)
        ax_ssim.set_title(f"{label_str} ? MS-SSIM", fontsize=11.5, fontweight="bold")
        ax_ssim.set_ylabel("MS-SSIM", fontsize=10.5)

    n_paired = len(s1_ids & set(df_all["tile_id"])) + len(s2_ids & set(df_all["tile_id"]))
    fig.suptitle(f"Ruifrok H&E Stain Stratification: S1 vs. S2 (n={n_paired} Tiles)", fontsize=14.0, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Stratifizierter S1/S2-Plot gespeichert: {out_file.resolve()}")


def plot_unified_suite(csv_paths: Sequence[Path], manifest_csv: Path, out_dir: Path) -> None:
    mf = load_unified_manifest(manifest_csv)
    df_all = load_codec_csvs(csv_paths)
    meta_cols = [c for c in ("tile_id", "gt_type", "organ", "stain") if c in mf.columns]
    joined = df_all.merge(mf[meta_cols], on="tile_id", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("Manifest join yielded 0 rows.")

    tile_meta = joined[["tile_id", "gt_type"]].drop_duplicates()
    n_ll = int((tile_meta["gt_type"] == "lossless").sum())
    n_lo = int((tile_meta["gt_type"] == "lossy").sum())

    # 1. Overall
    plot_curves_panel(
        build_curves_from_df(joined, strict_pairing=True),
        f"Track A Overall R-D (Tile-Pooled, n_lossless={n_ll}, n_lossy={n_lo})",
        out_dir / "rd_curves_overall.png",
    )

    # 2. Lossless vs. Lossy
    for gt_type, grp in joined.groupby("gt_type", sort=True):
        n_t = int(grp["tile_id"].nunique())
        plot_curves_panel(
            build_curves_from_df(grp, strict_pairing=True),
            f"Track A R-D: Ground-Truth = {gt_type} (Paired n={n_t} Tiles)",
            out_dir / f"rd_curves_gt_{gt_type}.png",
        )

    # 3. Organ & Stain
    for (gt_type, organ, stain), grp in joined.groupby(["gt_type", "organ", "stain"], sort=True):
        n_t = int(grp["tile_id"].nunique())
        slug = f"{gt_type}_{organ}_{stain}".lower().replace("&", "").replace(" ", "_").replace("/", "_")
        plot_curves_panel(
            build_curves_from_df(grp, strict_pairing=True),
            f"Track A R-D: {organ} | {stain} ({gt_type}, Paired n={n_t} Tiles)",
            out_dir / f"rd_curves_stratum_{slug}.png",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Track A R-D curves (legacy or unified manifest suite).")
    p.add_argument("--csvs", type=Path, nargs="+", help="Codec CSVs for unified plotting")
    p.add_argument("--manifest", type=Path, help="Unified manifest CSV")
    p.add_argument("--out-dir", type=Path, default=Path("results/unified_v1"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.csvs and args.manifest:
        plot_unified_suite(args.csvs, args.manifest, args.out_dir)
    else:
        conv_path = Path("results/curated_v1/results_rms_conv.csv")
        ai_path = Path("results/curated_v1/results_rms_ai.csv")
        manifest_path = Path("results/curated_v1/manifest_track_a.csv")
        if not conv_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {conv_path}")
        paths = [conv_path] + ([ai_path] if ai_path.exists() else [])
        df_all = load_codec_csvs(paths)
        curves = build_curves_from_df(df_all, strict_pairing=True)
        n_tiles = int(df_all[df_all["codec"] == next(iter(curves))]["tile_id"].nunique()) if not ai_path.exists() else int(pd.read_csv(ai_path)["tile_id"].nunique())
        plot_curves_panel(
            curves,
            f"Curated Track A Benchmark (IDC RMS Lossless .90 / .2.1, Paired n={n_tiles} Level-0 Tiles)",
            Path("results/curated_v1/rd_curves_track_a_curated.png"),
        )
        if manifest_path.exists():
            plot_s1_vs_s2(df_all, pd.read_csv(manifest_path), Path("results/curated_v1/rd_curves_track_a_s1_vs_s2.png"))
