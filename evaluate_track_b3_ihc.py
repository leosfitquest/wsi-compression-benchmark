"""
evaluate_track_b3_ihc.py
------------------------
Fuehrt den Track-B3-Benchmark (IHC / DAB Color & Chroma-Subsampling Stress Test) durch:
- Vergleicht 4:4:4 (JPEG 4:4:4, JPEG 2000, AVIF 4:4:4), 4:2:0 (JPEG 4:2:0, WebP),
  XYB-Farbraum (JPEG XL) und neuronale RGB-Autoencoder (Hyperprior, Cheng2020)
- Berechnet pro Kachel und Bitrate:
  - RGB-PSNR, Luminanz Y-PSNR und Chrominanz CbCr-PSNR (ITU-R BT.601)
  - Ruifrok (2001) DAB-Stain-PSNR (ungesaettigte physikalische DAB-OD ohne Clipping bei OD=2.125)
  - DAB-Positivity-Dice (feste physikalische Schwelle DAB OD >= 0.15)
  - Perzeptueller Farbabstand CIE Delta-E (Delta E*ab im L*a*b*-Farbraum)
- Erzeugt Summary-CSV, ungerundete PCHIP-BD-Rate-Chroma-Ablationstabelle und rd_curves_track_b3_ihc.png.
- Unterstuetzt `--recompute-from-csv` fuer 1-Sekunden-PCHIP-Reaggregation ohne neuen Codec-Lauf.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bdrate_summary import bd_psnr, bd_rate
from benchmark_compressai import (
    QUALITY_RANGES,
    load_model,
    pad_to_multiple,
    real_coding,
    tensor_to_tile,
    tile_to_tensor,
)
from benchmark_conventional import (
    DEFAULT_TARGETS_BPP,
    RATE_TOLERANCE,
    build_codecs,
    match_target_bpp,
)
from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb, psnr_y
from prepare_track_b3_anhir_ihc import _INV_HDAB_MAT

# Exakte physikalische Obergrenze der Ruifrok-DAB-OD bei uint8 >= 1: log10(255) / 0.776 ~= 3.1012 OD
DAB_PEAK_OD = float(math.log10(255.0) / 0.776)


@dataclass(frozen=True)
class B3ResultRow:
    tile_id: str
    profile: str
    category: str
    codec_or_model: str
    chroma_mode: str
    setting: float
    target_bpp: float
    bpp: float
    within_tolerance: bool
    psnr_rgb_db: float
    psnr_y_db: float
    psnr_chroma_db: float
    psnr_dab_db: float
    dab_dice: float
    cie_delta_e: float
    ms_ssim: float


CHROMA_MODES = {
    "JPEG 4:4:4": "4:4:4 (Full Chroma)",
    "JPEG 4:2:0": "4:2:0 (Subsampled)",
    "JPEG 2000": "4:4:4 (ICT Wavelet)",
    "JPEG XL": "XYB (Opsin Perceptual)",
    "WebP": "4:2:0 (YUV420p)",
    "AVIF": "4:4:4 (AV1 Intra)",
    "bmshj2018-hyperprior": "Neural RGB Latent",
    "cheng2020-attn": "Neural RGB Attention",
}


def psnr_cbcr(ref: np.ndarray, rec: np.ndarray) -> float:
    """Chrominance PSNR in dB (ITU-R BT.601 Cb and Cr channels, peak value 255)."""
    r1, g1, b1 = ref[..., 0].astype(np.float64), ref[..., 1].astype(np.float64), ref[..., 2].astype(np.float64)
    r2, g2, b2 = rec[..., 0].astype(np.float64), rec[..., 1].astype(np.float64), rec[..., 2].astype(np.float64)
    cb1 = -0.168736 * r1 - 0.331264 * g1 + 0.500000 * b1
    cr1 = 0.500000 * r1 - 0.418688 * g1 - 0.081312 * b1
    cb2 = -0.168736 * r2 - 0.331264 * g2 + 0.500000 * b2
    cr2 = 0.500000 * r2 - 0.418688 * g2 - 0.081312 * b2
    mse = 0.5 * float(np.mean((cb1 - cb2) ** 2) + np.mean((cr1 - cr2) ** 2))
    if mse == 0.0:
        return float("nan")
    return float(10.0 * math.log10(255.0**2 / mse))


def extract_dab_od(tile: np.ndarray) -> np.ndarray:
    """Extrahiert die ungesaettigte Ruifrok (2001) DAB-Optische-Dichte ohne Clipping starker DAB-Pixel."""
    od = -np.log10(np.clip(tile.astype(np.float64), 1.0, 255.0) / 255.0)
    return np.maximum((od @ _INV_HDAB_MAT)[..., 1], 0.0)


def dab_psnr_and_dice(dab_ref: np.ndarray, rec: np.ndarray) -> tuple[float, float]:
    dab_rec = extract_dab_od(rec)
    mse = float(np.mean((dab_ref - dab_rec) ** 2))
    if mse == 0.0:
        psnr_val = float("nan")
    else:
        psnr_val = float(10.0 * math.log10(DAB_PEAK_OD**2 / mse))

    # Feste physikalische Positivitaetsschwelle (OD >= 0.15, konsistent mit prepare_track_b3_anhir_ihc.py)
    m_ref = dab_ref >= 0.15
    m_rec = dab_rec >= 0.15
    inter = float(np.logical_and(m_ref, m_rec).sum())
    tot = float(m_ref.sum() + m_rec.sum())
    dice_val = (2.0 * inter / tot) if tot > 0.0 else float("nan")
    return psnr_val, dice_val


def _rgb_to_lab(img: np.ndarray) -> np.ndarray:
    srgb = img.astype(np.float64) / 255.0
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    m_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = (lin @ m_xyz.T) / np.array([0.95047, 1.00000, 1.08883])
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0.0)), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def cie_delta_e(lab_ref: np.ndarray, rec: np.ndarray) -> float:
    lab_rec = _rgb_to_lab(rec)
    return float(np.mean(np.linalg.norm(lab_ref - lab_rec, axis=-1)))


def _aggregate_and_save_b3(
    df: pd.DataFrame,
    summary_csv: Path,
    bdrate_csv: Path,
    plot_path: Path,
) -> None:
    df = df.copy()
    df["chroma_mode"] = df["codec_or_model"].map(lambda x: CHROMA_MODES.get(str(x), "Other"))

    df_valid = df.groupby(["codec_or_model", "target_bpp"]).filter(
        lambda g: (g["category"].iloc[0] == "neural")
        or (abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15)
    )

    bucket_key = df_valid["target_bpp"].where(df_valid["category"] == "conventional", df_valid["setting"])

    raw_summary_rows = []
    rounded_summary_rows = []
    for (name, bucket), grp in df_valid.groupby(["codec_or_model", bucket_key]):
        mean_bpp = float(grp["bpp"].mean())
        mean_rgb = float(grp["psnr_rgb_db"].mean())
        mean_y = float(grp["psnr_y_db"].mean())
        mean_chr = float(grp["psnr_chroma_db"].mean())
        mean_dab = float(grp["psnr_dab_db"].mean())
        mean_dice = float(grp["dab_dice"].mean())
        mean_de = float(grp["cie_delta_e"].mean())

        raw_summary_rows.append({
            "Codec / Model": str(name),
            "Chroma Mode": CHROMA_MODES.get(str(name), "Other"),
            "Bucket": float(bucket),
            "Mean bpp": mean_bpp,
            "RGB-PSNR [dB]": mean_rgb,
            "Y-PSNR [dB]": mean_y,
            "Chroma-PSNR [dB]": mean_chr,
            "DAB-Stain-PSNR [dB]": mean_dab,
            "DAB-Dice": mean_dice,
            "CIE Delta-E": mean_de,
        })
        rounded_summary_rows.append({
            "Codec / Model": str(name),
            "Chroma Mode": CHROMA_MODES.get(str(name), "Other"),
            "Bucket": float(bucket),
            "Mean bpp": round(mean_bpp, 3),
            "RGB-PSNR [dB]": round(mean_rgb, 2),
            "Y-PSNR [dB]": round(mean_y, 2),
            "Chroma-PSNR [dB]": round(mean_chr, 2),
            "DAB-Stain-PSNR [dB]": round(mean_dab, 2),
            "DAB-Dice": round(mean_dice, 4),
            "CIE Delta-E": round(mean_de, 2),
        })

    raw_sum_df = pd.DataFrame(raw_summary_rows).sort_values(["Codec / Model", "Mean bpp"])
    sum_df = pd.DataFrame(rounded_summary_rows).sort_values(["Codec / Model", "Mean bpp"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(summary_csv, index=False)

    # PCHIP BD-Rate Chroma-Ablation direkt auf den UNGERUNDETEN Gruppenmitteln (relativ zu JPEG 4:4:4)
    anchor = "JPEG 4:4:4"
    ref_grp = raw_sum_df[raw_sum_df["Codec / Model"] == anchor].sort_values("Mean bpp")
    ref_grp = ref_grp.loc[ref_grp["Mean bpp"].round(3).drop_duplicates(keep="last").index]
    r_ref = ref_grp["Mean bpp"].to_numpy()
    prgb_ref = ref_grp["RGB-PSNR [dB]"].to_numpy()
    py_ref = ref_grp["Y-PSNR [dB]"].to_numpy()
    pchr_ref = ref_grp["Chroma-PSNR [dB]"].to_numpy()
    pdab_ref = ref_grp["DAB-Stain-PSNR [dB]"].to_numpy()

    bdr_rows = []
    for name, grp in raw_sum_df.groupby("Codec / Model"):
        grp = grp.sort_values("Mean bpp")
        grp = grp.loc[grp["Mean bpp"].round(3).drop_duplicates(keep="last").index]
        r_t = grp["Mean bpp"].to_numpy()
        if name == anchor:
            bdr_rgb, bdr_y, bdr_chr, bdr_dab, bdp_dab = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            bdr_rgb = bd_rate(r_ref, prgb_ref, r_t, grp["RGB-PSNR [dB]"].to_numpy())
            bdr_y = bd_rate(r_ref, py_ref, r_t, grp["Y-PSNR [dB]"].to_numpy())
            bdr_chr = bd_rate(r_ref, pchr_ref, r_t, grp["Chroma-PSNR [dB]"].to_numpy())
            bdr_dab = bd_rate(r_ref, pdab_ref, r_t, grp["DAB-Stain-PSNR [dB]"].to_numpy())
            bdp_dab = bd_psnr(r_ref, pdab_ref, r_t, grp["DAB-Stain-PSNR [dB]"].to_numpy())
        bdr_rows.append({
            "Codec / Model": str(name),
            "Chroma Mode": CHROMA_MODES.get(str(name), "Other"),
            "BD-Rate RGB-PSNR [%]": round(bdr_rgb, 2),
            "BD-Rate Y-PSNR [%]": round(bdr_y, 2),
            "BD-Rate Chroma-PSNR [%]": round(bdr_chr, 2),
            "BD-Rate DAB-Stain-PSNR [%]": round(bdr_dab, 2),
            "BD-DAB-PSNR [dB]": round(bdp_dab, 2),
        })

    bdr_df = pd.DataFrame(bdr_rows).sort_values("BD-Rate DAB-Stain-PSNR [%]")
    bdr_df.to_csv(bdrate_csv, index=False)
    print(f"\n=== Track B3: IHC / DAB Chroma-Subsampling Ablation (Anchor: {anchor}, Method: PCHIP) ===")
    print(bdr_df.to_string(index=False))
    print(f"Saved to: {bdrate_csv.resolve()}")

    n_tiles = int(df["tile_id"].nunique())
    _plot_track_b3(sum_df, n_tiles, plot_path)


def run_track_b3(
    data_dir: Path,
    manifest_path: Path,
    out_csv: Path,
    summary_csv: Path,
    bdrate_csv: Path,
    plot_path: Path,
    fast_only: bool = True,
    limit: int | None = None,
    recompute_from_csv: bool = False,
) -> None:
    if recompute_from_csv:
        if not out_csv.exists():
            raise FileNotFoundError(f"Raw-CSV nicht gefunden: {out_csv}")
        df = pd.read_csv(out_csv)
        df["chroma_mode"] = df["codec_or_model"].map(lambda x: CHROMA_MODES.get(str(x), "Other"))
        df.to_csv(out_csv, index=False)
        _aggregate_and_save_b3(df, summary_csv, bdrate_csv, plot_path)
        return

    npy_path = data_dir / "ihc_tiles_u8.npy"
    if not npy_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Bitte zuerst `python prepare_track_b3_anhir_ihc.py` ausfuehren.")

    images = np.load(npy_path, mmap_mode="r")
    mf = pd.read_csv(manifest_path)
    if fast_only:
        mf = mf[mf["is_fast_subset"].astype(str).str.lower() == "true"].copy()
    if limit is not None:
        mf = mf.head(limit).copy()

    print(f"[Track B3] Starte IHC / DAB Chroma-Stress-Test auf {len(mf)} Kacheln ({mf['subset_or_profile'].nunique()} IHC-Profile)...")
    codecs = build_codecs(jxl_effort=7, verbose=False)

    ref_cache: dict[str, tuple[str, np.ndarray, np.ndarray, np.ndarray]] = {}
    for _, r in mf.iterrows():
        tid = str(r["tile_id"])
        idx = int(r["tile_idx"])
        prof = str(r["subset_or_profile"])
        tile = np.asarray(images[idx], dtype=np.uint8).copy()
        dab_ref = extract_dab_od(tile)
        lab_ref = _rgb_to_lab(tile)
        ref_cache[tid] = (prof, tile, dab_ref, lab_ref)

    rows: list[B3ResultRow] = []

    def _score_decoded(
        tid: str,
        category: str,
        name: str,
        setting: float,
        target_bpp: float,
        bpp: float,
        within_tol: bool,
        decoded: np.ndarray,
    ) -> B3ResultRow:
        prof, ref_tile, dab_ref, lab_ref = ref_cache[tid]
        p_dab, d_dice = dab_psnr_and_dice(dab_ref, decoded)
        de_val = cie_delta_e(lab_ref, decoded)
        return B3ResultRow(
            tile_id=tid,
            profile=prof,
            category=category,
            codec_or_model=name,
            chroma_mode=CHROMA_MODES.get(name, "Other"),
            setting=round(setting, 4),
            target_bpp=target_bpp,
            bpp=round(bpp, 4),
            within_tolerance=within_tol,
            psnr_rgb_db=round(psnr_rgb(ref_tile, decoded), 2),
            psnr_y_db=round(psnr_y(ref_tile, decoded), 2),
            psnr_chroma_db=round(psnr_cbcr(ref_tile, decoded), 2),
            psnr_dab_db=round(p_dab, 2) if np.isfinite(p_dab) else float("nan"),
            dab_dice=round(d_dice, 4) if np.isfinite(d_dice) else float("nan"),
            cie_delta_e=round(de_val, 3),
            ms_ssim=round(ms_ssim_rgb(ref_tile, decoded), 4),
        )

    print("[1/2] Evaluierung der 6 konventionellen Codecs (inkl. 4:4:4 vs. 4:2:0 vs. XYB)...")
    for i, (tid, (_, ref_tile, _, _)) in enumerate(ref_cache.items(), start=1):
        h, w = ref_tile.shape[:2]
        for codec in codecs:
            t_cache = {}
            for target in DEFAULT_TARGETS_BPP:
                param, enc = match_target_bpp(codec, ref_tile, target, cache=t_cache)
                bpp = bits_per_pixel(enc.num_bytes, h, w)
                tol = abs(bpp - target) / target <= RATE_TOLERANCE
                rows.append(_score_decoded(tid, "conventional", codec.name, param, target, bpp, tol, enc.decoded))
        if i % 10 == 0 or i == len(ref_cache):
            print(f"  Konventionell: [{i}/{len(ref_cache)}] Kacheln fertig.")

    print("[2/2] Evaluierung der CompressAI-Modelle (bmshj2018-hyperprior & cheng2020-attn)...")
    cpu = torch.device("cpu")
    for model_name in ("bmshj2018-hyperprior", "cheng2020-attn"):
        lo, hi = QUALITY_RANGES[model_name]
        for q in range(lo, min(hi, 6) + 1):
            model = load_model(model_name, q, "mse", cpu)
            for tid, (_, ref_tile, _, _) in ref_cache.items():
                x, hw = pad_to_multiple(tile_to_tensor(ref_tile))
                num_bytes, x_hat, _, _ = real_coding(model, x, hw)
                bpp = bits_per_pixel(num_bytes, *hw)
                rec = tensor_to_tile(x_hat)
                rows.append(_score_decoded(tid, "neural", model_name, float(q), round(bpp, 2), bpp, True, rec))
            print(f"  CompressAI: {model_name} q={q} fertig.")
            del model

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(B3ResultRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    df = pd.DataFrame([asdict(r) for r in rows])
    _aggregate_and_save_b3(df, summary_csv, bdrate_csv, plot_path)


def _plot_track_b3(sum_df: pd.DataFrame, n_tiles: int, plot_path: Path) -> None:
    styles = {
        "AVIF":                 {"color": "#9467bd", "marker": "D", "label": "AVIF (libavif, 4:4:4)", "linestyle": "-"},
        "JPEG XL":              {"color": "#2ca02c", "marker": "^", "label": "JPEG XL (XYB space)", "linestyle": "-"},
        "WebP":                 {"color": "#17becf", "marker": "p", "label": "WebP (4:2:0)", "linestyle": "-"},
        "JPEG 2000":            {"color": "#1f77b4", "marker": "s", "label": "JPEG 2000 (4:4:4 ICT)", "linestyle": "-"},
        "JPEG 4:4:4":           {"color": "#7f7f7f", "marker": "o", "label": "JPEG 4:4:4 (Anchor)", "linestyle": "-"},
        "JPEG 4:2:0":           {"color": "#bcbd22", "marker": "x", "label": "JPEG 4:2:0 (Subsampled)", "linestyle": ":"},
        "bmshj2018-hyperprior": {"color": "#ff7f0e", "marker": "o", "label": "Hyperprior (Ballé 2018)", "linestyle": "--"},
        "cheng2020-attn":       {"color": "#d62728", "marker": "v", "label": "Cheng2020-Attention", "linestyle": "--"},
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)
    for name, grp in sum_df.groupby("Codec / Model"):
        grp = grp.sort_values("Mean bpp")
        st = styles.get(str(name), {"color": "black", "marker": ".", "label": str(name), "linestyle": "-"})
        ax1.plot(grp["Mean bpp"], grp["Chroma-PSNR [dB]"], **st, linewidth=1.8)
        ax2.plot(grp["Mean bpp"], grp["DAB-Stain-PSNR [dB]"], **st, linewidth=1.8)
        ax3.plot(grp["Mean bpp"], grp["CIE Delta-E"], **st, linewidth=1.8)

    for ax in (ax1, ax2, ax3):
        ax.axvline(x=0.28, color="#444444", linestyle="--", linewidth=1.2, alpha=0.75, label="Reference Target (0.28 bpp)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("Bitrate [bpp]", fontsize=11)

    ax1.set_title("Chroma Subsampling Impact: CbCr-PSNR", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Chroma CbCr-PSNR [dB]", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax2.set_title("IHC Deconvolution Fidelity: DAB-Stain-PSNR", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Ruifrok DAB Optical Density PSNR [dB]", fontsize=11)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax3.set_title("Perceptual Color Error: CIE Delta-E (lower = better)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Mean CIE Delta-E (L*a*b*)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.suptitle(f"Curated Track B3: IHC / DAB Color & Chroma-Subsampling Stress Test (n={n_tiles} Lossless Tiles)", fontsize=13.5, y=0.99)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.unlink(missing_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Track-B3-Plot gespeichert: {plot_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Track B3 (IHC / DAB Color & Chroma-Subsampling Stress Test).")
    p.add_argument("--data-dir", type=Path, default=Path("anhir_data"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b3_ihc.csv"))
    p.add_argument("--out", type=Path, default=Path("results/curated_v1/results_track_b3_ihc.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/curated_v1/results_track_b3_summary.csv"))
    p.add_argument("--bdrate", type=Path, default=Path("results/curated_v1/results_bdrate_track_b3_ihc.csv"))
    p.add_argument("--plot", type=Path, default=Path("results/curated_v1/rd_curves_track_b3_ihc.png"))
    p.add_argument("--all-tiles", action="store_true", help="Run all 2,000 tiles instead of the 40-tile fast subset")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--recompute-from-csv", action="store_true", help="Re-aggregate summary, PCHIP BD-Rate & plot from existing raw CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_track_b3(
        args.data_dir, args.manifest, args.out, args.summary, args.bdrate, args.plot,
        fast_only=not args.all_tiles, limit=args.limit,
        recompute_from_csv=args.recompute_from_csv,
    )