"""
evaluate_track_b2_pannuke.py
----------------------------
Fuehrt den Track-B2-Benchmark auf PanNuke Fold 3 (native 256x256 Tiles, 19 Organe) durch:
  - 6 konventionelle Codecs (JPEG 4:4:4, JPEG 4:2:0, JPEG 2000, JPEG XL, WebP, AVIF)
  - 2 CompressAI-Modelle (bmshj2018-hyperprior, cheng2020-attn)
  - Berechnet pro Kachel und Bitrate:
      * RGB-PSNR, Y-PSNR, MS-SSIM sowie getrenntes PSNR innerhalb der Ground-Truth-Zellkerne (psnr_nuclei_db)
      * Deep Feature Cosine Similarity (ResNet-50 Embeddings: cos(f(ref), f(dec)))
      * Downstream Nuclei Segmentation Drift: Delta Dice, Delta bPQ (binary Panoptic Quality)
        und Delta mPQ (5-Klassen Multi-Class Panoptic Quality)
  - Erzeugt Summary-CSVs und den 3-Panel-Plot rd_curves_track_b2_pannuke.png.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

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
from prepare_track_a_rms import _INV_STAIN_MAT


@dataclass(frozen=True)
class B2ResultRow:
    tile_id: str
    organ: str
    category: str
    codec_or_model: str
    setting: float
    target_bpp: float
    bpp: float
    within_tolerance: bool
    psnr_db: float
    psnr_y_db: float
    psnr_nuclei_db: float
    ms_ssim: float
    feature_cos_sim: float
    dice_vs_ref: float
    bpq_vs_ref: float
    mpq_vs_ref: float


class DeepFeatureExtractor:
    """Extrahiert L2-normierte 2048-D Feature-Vektoren (ResNet-50 ImageNet/Pathology-Backbone)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.model = backbone.eval().to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed(self, tile: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)
        x = (x - self.mean) / self.std
        feat = self.model(x)
        return feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def cosine_sim(f_ref: torch.Tensor, f_rec: torch.Tensor) -> float:
        return float(torch.sum(f_ref * f_rec).item())


def _fast_connected_components_2d(binary_mask: np.ndarray) -> np.ndarray:
    """Schnelles 4-Nachbarschafts-Labeling fuer 2D-Binaermasken (reines NumPy, keine Zusatzpakete)."""
    h, w = binary_mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]
    next_label = 1

    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up == 0 and left == 0:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            elif up != 0 and left == 0:
                labels[y, x] = up
            elif up == 0 and left != 0:
                labels[y, x] = left
            else:
                r1, r2 = up, left
                while parent[r1] != r1:
                    r1 = parent[r1]
                while parent[r2] != r2:
                    r2 = parent[r2]
                root = min(r1, r2)
                labels[y, x] = root
                parent[r1] = root
                parent[r2] = root

    for i in range(1, len(parent)):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r

    remap = np.zeros(len(parent), dtype=np.int32)
    cur = 1
    for i in range(1, len(parent)):
        root = parent[i]
        if remap[root] == 0:
            remap[root] = cur
            cur += 1
        remap[i] = remap[root]

    out = remap[labels]
    # Kleine Rausch-Artefakte (< 12 Pixel) filtern
    if cur > 1:
        areas = np.bincount(out.ravel())
        small = areas < 12
        small[0] = False
        out[small[out]] = 0
    return out


def predict_nuclei_instances(tile: np.ndarray, gt_5ch: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extrahiert aus dem Haematoxylin-Kanal (Ruifrok OD) die Zellkern-Instanzen eines (dekodierten) Tiles
    und ordnet sie den 5 PanNuke-Zellklassen zu, um den kompressionsbedingten Masken-/Instanz-Drift
    (Delta Dice, Delta bPQ, Delta mPQ) exakt zu messen.
    """
    od = -np.log10(np.clip(tile.astype(np.float64), 1.0, 255.0) / 255.0)
    h_stain = (od @ _INV_STAIN_MAT)[..., 0]

    gt_union = np.any(gt_5ch > 0, axis=-1)
    target_frac = float(np.mean(gt_union))
    if target_frac <= 0.002:
        thresh = float(np.percentile(h_stain, 92.0))
    else:
        q = np.clip(100.0 * (1.0 - target_frac), 5.0, 99.5)
        thresh = float(np.percentile(h_stain, q))

    bin_pred = h_stain >= thresh
    inst_bin = _fast_connected_components_2d(bin_pred)

    # Fuer Multi-Class mPQ: Klassenzuordnung pro detektierter Instanz
    class_insts: list[np.ndarray] = []
    for c in range(5):
        c_region = gt_5ch[..., c] > 0
        c_map = np.where(c_region, inst_bin, 0)
        class_insts.append(c_map)

    return inst_bin, class_insts


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    tot = float(mask_a.sum() + mask_b.sum())
    return (2.0 * inter / tot) if tot > 0.0 else 1.0


def fast_pq(inst_true: np.ndarray, inst_pred: np.ndarray) -> float:
    """O(H*W) Panoptic Quality (PQ = DQ * SQ) mit striktem IoU > 0.5 Matching."""
    u_t, t_inv = np.unique(inst_true, return_inverse=True)
    u_p, p_inv = np.unique(inst_pred, return_inverse=True)
    t_map = t_inv.reshape(inst_true.shape)
    p_map = p_inv.reshape(inst_pred.shape)

    if u_t[0] != 0:
        t_map = t_map + 1
    if u_p[0] != 0:
        p_map = p_map + 1

    n_true = int(t_map.max())
    n_pred = int(p_map.max())
    if n_true == 0 and n_pred == 0:
        return 1.0
    if n_true == 0 or n_pred == 0:
        return 0.0

    t_areas = np.bincount(t_map.ravel())[1:]
    p_areas = np.bincount(p_map.ravel())[1:]

    fg = (t_map > 0) & (p_map > 0)
    if not np.any(fg):
        return 0.0

    pairs = t_map[fg].astype(np.int64) * (n_pred + 1) + p_map[fg].astype(np.int64)
    uniq_pairs, inter_counts = np.unique(pairs, return_counts=True)

    tp = 0
    iou_sum = 0.0
    matched_t: set[int] = set()
    matched_p: set[int] = set()

    for pair_val, inter in zip(uniq_pairs, inter_counts):
        t_id = int(pair_val // (n_pred + 1))
        p_id = int(pair_val % (n_pred + 1))
        union = float(t_areas[t_id - 1] + p_areas[p_id - 1] - inter)
        iou = float(inter) / union
        if iou > 0.5 and t_id not in matched_t and p_id not in matched_p:
            tp += 1
            iou_sum += iou
            matched_t.add(t_id)
            matched_p.add(p_id)

    fp = n_pred - tp
    fn = n_true - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return (iou_sum / denom) if denom > 0 else 0.0


def psnr_masked(ref: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return psnr_rgb(ref, rec)
    diff = ref[mask].astype(np.float64) - rec[mask].astype(np.float64)
    mse = float(np.mean(diff**2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def run_track_b2(
    data_dir: Path,
    manifest_path: Path,
    out_csv: Path,
    summary_csv: Path,
    plot_path: Path,
    fast_only: bool = True,
    limit: int | None = None,
) -> None:
    u8_path = data_dir / "images_u8.npy"
    masks_path = data_dir / "masks_u16.npz"
    if not u8_path.exists() or not masks_path.exists():
        raise FileNotFoundError("Bitte zuerst `python prepare_track_b2_pannuke.py` ausfuehren.")

    images = np.load(u8_path, mmap_mode="r")
    masks_all = np.load(masks_path)["masks"]
    mf = pd.read_csv(manifest_path)
    if fast_only:
        mf = mf[mf["is_fast_subset"].astype(str).str.lower() == "true"].copy()
    if limit is not None:
        mf = mf.head(limit).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Track B2] Starte Evaluation auf {len(mf)} PanNuke-Kacheln ({mf['organ'].nunique()} Organe, Device={device})...")
    feat_extractor = DeepFeatureExtractor(device)
    codecs = build_codecs(jxl_effort=7, verbose=False)

    # Vorberechnung der Referenz-Features und Referenz-Kerninstanzen auf den Original-Tiles
    ref_cache: dict[str, tuple[int, str, np.ndarray, np.ndarray, torch.Tensor, np.ndarray, list[np.ndarray]]] = {}
    for _, r in mf.iterrows():
        tid = str(r["tile_id"])
        idx = int(r["tile_idx"])
        organ = str(r["organ"])
        tile = np.asarray(images[idx], dtype=np.uint8).copy()
        gt_5ch = masks_all[idx]
        f_ref = feat_extractor.embed(tile)
        ref_inst, ref_class_insts = predict_nuclei_instances(tile, gt_5ch)
        ref_cache[tid] = (idx, organ, tile, gt_5ch, f_ref, ref_inst, ref_class_insts)

    rows: list[B2ResultRow] = []

    def _score_decoded(
        tid: str,
        category: str,
        name: str,
        setting: float,
        target_bpp: float,
        bpp: float,
        within_tol: bool,
        decoded: np.ndarray,
    ) -> B2ResultRow:
        _, organ, ref_tile, gt_5ch, f_ref, ref_inst, ref_class_insts = ref_cache[tid]
        nuc_mask = np.any(gt_5ch > 0, axis=-1)
        f_rec = feat_extractor.embed(decoded)
        cos_sim = feat_extractor.cosine_sim(f_ref, f_rec)

        rec_inst, rec_class_insts = predict_nuclei_instances(decoded, gt_5ch)
        dice_val = compute_dice(ref_inst > 0, rec_inst > 0)
        bpq_val = fast_pq(ref_inst, rec_inst)
        mpq_vals = [
            fast_pq(ref_class_insts[c], rec_class_insts[c])
            for c in range(5)
            if np.any(ref_class_insts[c] > 0) or np.any(rec_class_insts[c] > 0)
        ]
        mpq_val = float(np.mean(mpq_vals)) if mpq_vals else 1.0

        return B2ResultRow(
            tile_id=tid,
            organ=organ,
            category=category,
            codec_or_model=name,
            setting=round(setting, 4),
            target_bpp=target_bpp,
            bpp=round(bpp, 4),
            within_tolerance=within_tol,
            psnr_db=round(psnr_rgb(ref_tile, decoded), 2),
            psnr_y_db=round(psnr_y(ref_tile, decoded), 2),
            psnr_nuclei_db=round(psnr_masked(ref_tile, decoded, nuc_mask), 2),
            ms_ssim=round(ms_ssim_rgb(ref_tile, decoded), 4),
            feature_cos_sim=round(cos_sim, 5),
            dice_vs_ref=round(dice_val, 4),
            bpq_vs_ref=round(bpq_val, 4),
            mpq_vs_ref=round(mpq_val, 4),
        )

    # 1. Konventionelle Codecs
    print("[1/2] Evaluierung der 6 konventionellen Codecs...")
    for i, (tid, (_, _, ref_tile, _, _, _, _)) in enumerate(ref_cache.items(), start=1):
        h, w = ref_tile.shape[:2]
        for codec in codecs:
            t_cache = {}
            for target in DEFAULT_TARGETS_BPP:
                param, enc = match_target_bpp(codec, ref_tile, target, cache=t_cache)
                bpp = bits_per_pixel(enc.num_bytes, h, w)
                tol = abs(bpp - target) / target <= RATE_TOLERANCE
                rows.append(_score_decoded(tid, "conventional", codec.name, param, target, bpp, tol, enc.decoded))
        if i % 5 == 0 or i == len(ref_cache):
            print(f"  Konventionell: [{i}/{len(ref_cache)}] Kacheln fertig.")

    # 2. CompressAI Modelle (Qualities 1..6 fuer schnellen, abdeckenden R-D-Verlauf)
    print("[2/2] Evaluierung der CompressAI-Modelle (bmshj2018-hyperprior & cheng2020-attn)...")
    cpu = torch.device("cpu")
    for model_name in ("bmshj2018-hyperprior", "cheng2020-attn"):
        lo, hi = QUALITY_RANGES[model_name]
        for q in range(lo, min(hi, 6) + 1):
            model = load_model(model_name, q, "mse", cpu)
            for tid, (_, _, ref_tile, _, _, _, _) in ref_cache.items():
                x, hw = pad_to_multiple(tile_to_tensor(ref_tile))
                num_bytes, x_hat, _, _ = real_coding(model, x, hw)
                bpp = bits_per_pixel(num_bytes, *hw)
                rec = tensor_to_tile(x_hat)
                rows.append(_score_decoded(tid, "neural", model_name, float(q), round(bpp, 2), bpp, True, rec))
            print(f"  CompressAI: {model_name} q={q} fertig.")
            del model

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(B2ResultRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    df = pd.DataFrame([asdict(r) for r in rows])
    df_valid = df.groupby(["codec_or_model", "target_bpp"]).filter(
        lambda g: (g["category"].iloc[0] == "neural")
        or (abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15)
    )

    grp_key = ["codec_or_model", "setting"] if "setting" in df_valid.columns else ["codec_or_model", "target_bpp"]
    summary_list = []
    for (name, bucket), grp in df_valid.groupby(
        ["codec_or_model", df_valid["target_bpp"].where(df_valid["category"] == "conventional", df_valid["setting"])]
    ):
        summary_list.append({
            "Codec / Model": name,
            "Bucket": bucket,
            "Mean bpp": round(float(grp["bpp"].mean()), 3),
            "RGB-PSNR [dB]": round(float(grp["psnr_db"].mean()), 2),
            "Nuclei-PSNR [dB]": round(float(grp["psnr_nuclei_db"].mean()), 2),
            "Feature CosSim": round(float(grp["feature_cos_sim"].mean()), 4),
            "Dice (vs Ref)": round(float(grp["dice_vs_ref"].mean()), 4),
            "bPQ (vs Ref)": round(float(grp["bpq_vs_ref"].mean()), 4),
            "mPQ (vs Ref)": round(float(grp["mpq_vs_ref"].mean()), 4),
        })

    sum_df = pd.DataFrame(summary_list).sort_values(["Codec / Model", "Mean bpp"])
    sum_df.to_csv(summary_csv, index=False)
    print(f"\n=== Track B2 (PanNuke Fold 3, {df['organ'].nunique()} Organe) Downstream Summary ===")
    print(sum_df.to_string(index=False))

    # Plot erstellen: 1. bPQ vs. bpp, 2. Dice vs. bpp, 3. Feature Cosine Similarity vs. bpp
    _plot_track_b2(sum_df, len(ref_cache), df["organ"].nunique(), plot_path)


def _plot_track_b2(sum_df: pd.DataFrame, n_tiles: int, n_organs: int, plot_path: Path) -> None:
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
        ax1.plot(grp["Mean bpp"], grp["bPQ (vs Ref)"], **st, linewidth=1.8)
        ax2.plot(grp["Mean bpp"], grp["Dice (vs Ref)"], **st, linewidth=1.8)
        ax3.plot(grp["Mean bpp"], grp["Feature CosSim"], **st, linewidth=1.8)

    for ax in (ax1, ax2, ax3):
        ax.axvline(x=0.28, color="#444444", linestyle="--", linewidth=1.2, alpha=0.75, label="Reference Target (0.28 bpp)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlabel("Bitrate [bpp]", fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    ax1.set_title("Nuclei Instance Stability: Binary PQ (bPQ)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Binary Panoptic Quality (bPQ)", fontsize=11)

    ax2.set_title("Nuclei Mask Overlap: Dice Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Dice Similarity Coefficient", fontsize=11)

    ax3.set_title("Deep Feature Cosine Similarity (ResNet-50)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Cosine Similarity vs. Uncompressed", fontsize=11)

    plt.suptitle(f"Curated Track B2: PanNuke Fold 3 Downstream Stability (n={n_tiles} Tiles across {n_organs} Organs)", fontsize=13.5, y=0.99)
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"\nTrack-B2-Plot gespeichert: {plot_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Track B2 (PanNuke Fold 3) downstream nuclei & feature metrics.")
    p.add_argument("--data-dir", type=Path, default=Path("pannuke_fold3"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b2_pannuke.csv"))
    p.add_argument("--out", type=Path, default=Path("results/curated_v1/results_track_b2_pannuke.csv"))
    p.add_argument("--summary", type=Path, default=Path("results/curated_v1/results_track_b2_summary.csv"))
    p.add_argument("--plot", type=Path, default=Path("results/curated_v1/rd_curves_track_b2_pannuke.png"))
    p.add_argument("--all-tiles", action="store_true", help="Run all 2,722 tiles instead of the 38-tile 19-organ fast subset")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_track_b2(
        args.data_dir, args.manifest, args.out, args.summary, args.plot,
        fast_only=not args.all_tiles, limit=args.limit,
    )