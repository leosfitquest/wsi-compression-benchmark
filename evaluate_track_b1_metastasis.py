"""
evaluate_track_b1_metastasis.py
-------------------------------
Fuehrt den Track-B1-Benchmark (CAMELYON16 Metastasis Classification & Deep Feature Drift) durch:
  - 6 konventionelle Codecs (JPEG 4:4:4, JPEG 4:2:0, JPEG 2000, JPEG XL, WebP, AVIF)
  - 2 CompressAI-Modelle (bmshj2018-hyperprior, cheng2020-attn)
  - Kalibriert einen eingefrorenen Linear-Probe-Klassifikator (ResNet-50 2048-D Features
    + Ruifrok Haematoxylin-OD-Merkmale) auf den unkomprimierten Referenz-Kacheln und misst:
      * Classification AUROC und Delta-AUROC vs. Reference
      * Tumor Probability Drift |P_dec(tumor) - P_ref(tumor)| (niedriger = stabiler)
      * Classification Decision Agreement [%]
      * Deep Feature Cosine Similarity cos(f(ref), f(dec))
  - Erzeugt Summary-CSV und den 3-Panel-Plot rd_curves_track_b1_metastasis.png.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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
from evaluate_track_b2_pannuke import DeepFeatureExtractor
from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb, psnr_y
from prepare_track_a_rms import _INV_STAIN_MAT


@dataclass(frozen=True)
class B1ResultRow:
    tile_id: str
    label: int
    class_name: str
    category: str
    codec_or_model: str
    setting: float
    target_bpp: float
    bpp: float
    within_tolerance: bool
    psnr_db: float
    psnr_y_db: float
    ms_ssim: float
    feature_cos_sim: float
    prob_ref: float
    prob_dec: float
    abs_prob_drift: float
    decision_match: int


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exakte Mann-Whitney-U AUROC-Berechnung in reinem NumPy."""
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_score, dtype=np.float64)
    pos = y_t == 1
    neg = y_t == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    diff = y_s[pos][:, None] - y_s[neg][None, :]
    wins = float(np.sum(diff > 0)) + 0.5 * float(np.sum(diff == 0))
    return wins / float(n_pos * n_neg)


def _extract_combined_descriptor(tile: np.ndarray, feat_vec: torch.Tensor) -> np.ndarray:
    od = -np.log10(np.clip(tile.astype(np.float64), 1.0, 255.0) / 255.0)
    stains = od @ _INV_STAIN_MAT
    h_ch = np.maximum(stains[..., 0], 0.0)
    e_ch = np.maximum(stains[..., 1], 0.0)

    h_mean = float(np.mean(h_ch))
    h_p90 = float(np.percentile(h_ch, 90.0))
    h_std = float(np.std(h_ch))
    e_mean = float(np.mean(e_ch))
    he_ratio = h_mean / (h_mean + e_mean + 1e-8)

    morph_vec = np.array([h_mean, h_p90, h_std, e_mean, he_ratio, 1.0], dtype=np.float64)
    cnn_np = feat_vec.squeeze(0).cpu().numpy().astype(np.float64)
    return np.concatenate([cnn_np, morph_vec], axis=0)


def _fit_linear_probe(X: np.ndarray, y: np.ndarray, l2_reg: float = 2.0) -> np.ndarray:
    targets = np.where(y == 1, 2.2, -2.2).astype(np.float64)
    d = X.shape[1]
    reg_mat = l2_reg * np.eye(d, dtype=np.float64)
    reg_mat[-1, -1] = 1e-4
    return np.linalg.solve(X.T @ X + reg_mat, X.T @ targets)


def _predict_prob(desc: np.ndarray, w: np.ndarray) -> float:
    logit = max(-20.0, min(20.0, float(np.dot(desc, w))))
    return 1.0 / (1.0 + float(np.exp(-logit)))


def run_track_b1(
    data_dir: Path,
    manifest_path: Path,
    out_csv: Path,
    summary_csv: Path,
    plot_path: Path,
    fast_only: bool = True,
    limit: int | None = None,
) -> None:
    npy_path = data_dir / "b1_tiles_u8.npy"
    if not npy_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Bitte zuerst `python prepare_track_b1_camelyon.py` ausfuehren.")

    images = np.load(npy_path, mmap_mode="r")
    mf = pd.read_csv(manifest_path)
    if fast_only:
        mf = mf[mf["is_fast_subset"].astype(str).str.lower() == "true"].copy()
    if limit is not None:
        mf = mf.head(limit).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Track B1] Starte CAMELYON16 Metastasis & Feature Drift auf {len(mf)} Kacheln (Device={device})...")
    feat_extractor = DeepFeatureExtractor(device)
    codecs = build_codecs(jxl_effort=7, verbose=False)

    ref_items: list[tuple[str, int, str, np.ndarray, torch.Tensor, np.ndarray]] = []
    for _, r in mf.iterrows():
        tid = str(r["tile_id"])
        idx = int(r["tile_idx"])
        label = int(r["label"])
        cname = str(r["class_name"])
        tile = np.asarray(images[idx], dtype=np.uint8).copy()
        f_ref = feat_extractor.embed(tile)
        desc_ref = _extract_combined_descriptor(tile, f_ref)
        ref_items.append((tid, label, cname, tile, f_ref, desc_ref))

    X_ref = np.stack([item[5] for item in ref_items], axis=0)
    y_ref = np.array([item[1] for item in ref_items], dtype=np.int32)
    probe_w = _fit_linear_probe(X_ref, y_ref, l2_reg=2.0)

    ref_cache: dict[str, tuple[int, str, np.ndarray, torch.Tensor, float]] = {}
    ref_probs = []
    for tid, label, cname, tile, f_ref, desc_ref in ref_items:
        p_ref = _predict_prob(desc_ref, probe_w)
        ref_probs.append(p_ref)
        ref_cache[tid] = (label, cname, tile, f_ref, p_ref)

    ref_auroc = compute_auroc(y_ref, np.array(ref_probs))
    print(f"  Kalibrierter Referenz-Klassifikator (Original CAMELYON16): AUROC = {ref_auroc:.4f}")

    rows: list[B1ResultRow] = []

    def _score_decoded(
        tid: str,
        category: str,
        name: str,
        setting: float,
        target_bpp: float,
        bpp: float,
        within_tol: bool,
        decoded: np.ndarray,
    ) -> B1ResultRow:
        label, cname, ref_tile, f_ref, p_ref = ref_cache[tid]
        f_rec = feat_extractor.embed(decoded)
        cos_sim = feat_extractor.cosine_sim(f_ref, f_rec)
        desc_rec = _extract_combined_descriptor(decoded, f_rec)
        p_dec = _predict_prob(desc_rec, probe_w)
        drift = abs(p_dec - p_ref)
        match = int((p_dec >= 0.5) == (p_ref >= 0.5))
        return B1ResultRow(
            tile_id=tid,
            label=label,
            class_name=cname,
            category=category,
            codec_or_model=name,
            setting=round(setting, 4),
            target_bpp=target_bpp,
            bpp=round(bpp, 4),
            within_tolerance=within_tol,
            psnr_db=round(psnr_rgb(ref_tile, decoded), 2),
            psnr_y_db=round(psnr_y(ref_tile, decoded), 2),
            ms_ssim=round(ms_ssim_rgb(ref_tile, decoded), 4),
            feature_cos_sim=round(cos_sim, 5),
            prob_ref=round(p_ref, 4),
            prob_dec=round(p_dec, 4),
            abs_prob_drift=round(drift, 4),
            decision_match=match,
        )

    print("[1/2] Evaluierung der 6 konventionellen Codecs...")
    for i, (tid, (_, _, ref_tile, _, _)) in enumerate(ref_cache.items(), start=1):
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
            for tid, (_, _, ref_tile, _, _) in ref_cache.items():
                x, hw = pad_to_multiple(tile_to_tensor(ref_tile))
                num_bytes, x_hat, _, _ = real_coding(model, x, hw)
                bpp = bits_per_pixel(num_bytes, *hw)
                rec = tensor_to_tile(x_hat)
                rows.append(_score_decoded(tid, "neural", model_name, float(q), round(bpp, 2), bpp, True, rec))
            print(f"  CompressAI: {model_name} q={q} fertig.")
            del model

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(B1ResultRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    df = pd.DataFrame([asdict(r) for r in rows])
    df_valid = df.groupby(["codec_or_model", "target_bpp"]).filter(
        lambda g: (g["category"].iloc[0] == "neural")
        or (abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15)
    )

    summary_list = []
    for (name, bucket), grp in df_valid.groupby(
        ["codec_or_model", df_valid["target_bpp"].where(df_valid["category"] == "conventional", df_valid["setting"])]
    ):
        auroc_val = compute_auroc(grp["label"].to_numpy(), grp["prob_dec"].to_numpy())
        summary_list.append({
            "Codec / Model": name,
            "Bucket": bucket,
            "Mean bpp": round(float(grp["bpp"].mean()), 3),
            "RGB-PSNR [dB]": round(float(grp["psnr_db"].mean()), 2),
            "MS-SSIM": round(float(grp["ms_ssim"].mean()), 4),
            "Feature CosSim": round(float(grp["feature_cos_sim"].mean()), 4),
            "Mean |Delta P(tumor)|": round(float(grp["abs_prob_drift"].mean()), 4),
            "Decision Match [%]": round(100.0 * float(grp["decision_match"].mean()), 2),
            "AUROC": round(auroc_val, 4),
            "Delta AUROC": round(auroc_val - ref_auroc, 4),
        })

    sum_df = pd.DataFrame(summary_list).sort_values(["Codec / Model", "Mean bpp"])
    sum_df.to_csv(summary_csv, index=False)
    print(f"\n=== Track B1 (CAMELYON16 Metastasis Classification & Feature Drift) Summary ===")
    print(sum_df.to_string(index=False))

    _plot_track_b1(sum_df, len(ref_cache), plot_path)


def _plot_track_b1(sum_df: pd.DataFrame, n_tiles: int, plot_path: Path) -> None:
    styles = {
        "AVIF":                 {"color": "#9467bd", "marker": "D", "label": "AVIF (libavif)", "linestyle": "-"},
        "JPEG XL":              {"color": "#2ca02c", "marker": "^", "label": "JPEG XL (libjxl)", "linestyle": "-"},
        "WebP":                 {"color": "#17becf", "marker": "p", "label": "WebP (libwebp)", "linestyle": "-"},
        "JPEG 2000":            {"color": "#1f77b4", "marker": "s", "label": "JPEG 2000 (OpenJPEG)", "linestyle": "-"},
        "JPEG 4:4:4":           {"color": "#7f7f7f", "marker": "o", "label": "JPEG 4:4:4", "linestyle": "-"},
        "JPEG 4:2:0":           {"color": "#bcbd22", "marker": "x", "label": "JPEG 4:2:0", "linestyle": ":"},
        "bmshj2018-hyperprior": {"color": "#ff7f0e", "marker": "o", "label": "Hyperprior (Ballé 2018)", "linestyle": "--"},
        "cheng2020-attn":       {"color": "#d62728", "marker": "v", "label": "Cheng2020-Attention", "linestyle": "--"},
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)
    for name, grp in sum_df.groupby("Codec / Model"):
        grp = grp.sort_values("Mean bpp")
        st = styles.get(str(name), {"color": "black", "marker": ".", "label": str(name), "linestyle": "-"})
        ax1.plot(grp["Mean bpp"], grp["AUROC"], **st, linewidth=1.8)
        ax2.plot(grp["Mean bpp"], grp["Mean |Delta P(tumor)|"], **st, linewidth=1.8)
        ax3.plot(grp["Mean bpp"], grp["Feature CosSim"], **st, linewidth=1.8)

    for ax in (ax1, ax2, ax3):
        ax.axvline(x=0.28, color="#444444", linestyle="--", linewidth=1.2, alpha=0.75, label="Reference Target (0.28 bpp)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("Bitrate [bpp]", fontsize=11)

    ax1.set_title("Classification Stability: Tumor vs. Normal AUROC", fontsize=12, fontweight="bold")
    ax1.set_ylabel("AUROC", fontsize=11)
    ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax2.set_title("Prediction Drift: Mean |Delta P(tumor)| (lower = better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Absolute Probability Drift", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    ax3.set_title("Deep Feature Cosine Similarity (ResNet-50)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Cosine Similarity vs. Uncompressed", fontsize=11)
    ax3.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle(f"Curated Track B1: CAMELYON16 Metastasis & Feature Drift (n={n_tiles} Balanced Tiles)", fontsize=13.5, y=0.99)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Track-B1-Plot gespeichert: {plot_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Track B1 (CAMELYON16 Metastasis & Feature Drift).")
    p.add_argument("--data-dir", type=Path, default=Path("camelyon_data"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b1_tumor.csv"))
    p.add_argument("--out", type=Path, default=Path("results/curated_v1/results_track_b1_metastasis.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/curated_v1/results_track_b1_summary.csv"))
    p.add_argument("--plot", type=Path, default=Path("results/curated_v1/rd_curves_track_b1_metastasis.png"))
    p.add_argument("--all-tiles", action="store_true", help="Run all extracted tiles instead of the 40-tile fast subset")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_track_b1(
        args.data_dir, args.manifest, args.out, args.summary, args.plot,
        fast_only=not args.all_tiles, limit=args.limit,
    )