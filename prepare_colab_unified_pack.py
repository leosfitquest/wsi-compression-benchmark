"""
prepare_colab_unified_pack.py
-----------------------------
Builds the compact 440-tile unified Phase-1 benchmark pack for Google Colab GPU
(and VUB IPC3 smoke-testing) from already extracted local datasets (0 new downloads):

  - 300 Lossless H&E (Soft_Tissue)  : IDC RMS (is_fast_subset=True, 30 patients x 10 tiles)
  -  40 Lossless IHC_DAB (Lung)     : ANHIR CTU PNG (is_fast_subset=True, 40 tiles)
  -  25 Lossy H&E (Lymph_Node)      : CAMELYON16 Level-0 (stratified across slides, seed=42)
  -  75 Lossy H&E (Breast/Colon/Lung, 25 each): PanNuke Fold 3 (random seed=42, NOT top-nuclei)

Outputs:
  - tiles_unified_440/*.png              (440 lossless 256x256 uint8 RGB PNGs)
  - tiles_unified_440/manifest_unified_440.csv  (Single Source of Truth: gt_type, organ, stain)
  - colab_bundle_440.zip                 (Ready to upload to Google Drive)
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def measure_8x8_blockiness(tile: np.ndarray) -> float:
    """Ratio of mean luminance gradient on 8-px JPEG block boundaries vs. interior pixels."""
    y = (
        0.2126 * tile[..., 0].astype(np.float64)
        + 0.7152 * tile[..., 1].astype(np.float64)
        + 0.0722 * tile[..., 2].astype(np.float64)
    )
    dv = np.abs(np.diff(y, axis=0))
    dh = np.abs(np.diff(y, axis=1))
    v_idx = np.arange(dv.shape[0])
    h_idx = np.arange(dh.shape[1])
    v_b = dv[(v_idx + 1) % 8 == 0, :].mean()
    v_i = dv[(v_idx + 1) % 8 != 0, :].mean()
    h_b = dh[:, (h_idx + 1) % 8 == 0].mean()
    h_i = dh[:, (h_idx + 1) % 8 != 0].mean()
    denom = 0.5 * (v_i + h_i)
    if denom < 1e-9:
        return 1.0
    return float(0.5 * (v_b + h_b) / denom)


def _infer_anhir_organ(tile_id: str, profile: str) -> str:
    low = f"{tile_id} {profile}".lower()
    if "lung" in low:
        return "Lung"
    if "gastric" in low:
        return "Gastric"
    if "breast" in low or "mammary" in low:
        return "Breast"
    if "kidney" in low:
        return "Kidney"
    return "IHC_Other"


def _sample_stratified(df: pd.DataFrame, group_col: str, n_total: int, seed: int) -> pd.DataFrame:
    if len(df) < n_total:
        raise ValueError(f"Requested {n_total} rows, but only {len(df)} available.")
    rng = np.random.default_rng(seed)
    groups = sorted(df[group_col].astype(str).unique())
    base = n_total // len(groups)
    rem = n_total % len(groups)
    chosen_parts: list[pd.DataFrame] = []
    leftover_parts: list[pd.DataFrame] = []
    for idx, g in enumerate(groups):
        sub = df[df[group_col].astype(str) == g]
        want = base + (1 if idx < rem else 0)
        take = min(want, len(sub))
        perm = rng.permutation(len(sub))
        chosen_parts.append(sub.iloc[perm[:take]])
        if take < len(sub):
            leftover_parts.append(sub.iloc[perm[take:]])
    chosen = pd.concat(chosen_parts, ignore_index=True)
    if len(chosen) < n_total:
        need = n_total - len(chosen)
        pool = pd.concat(leftover_parts, ignore_index=True)
        extra_idx = rng.permutation(len(pool))[:need]
        chosen = pd.concat([chosen, pool.iloc[extra_idx]], ignore_index=True)
    return chosen.reset_index(drop=True)


def _save_tile_png(tile: np.ndarray, dest: Path) -> None:
    if tile.shape != (256, 256, 3) or tile.dtype != np.uint8:
        raise ValueError(f"Invalid tile for {dest.name}: shape={tile.shape}, dtype={tile.dtype}")
    Image.fromarray(tile, mode="RGB").save(dest, format="PNG")


def build_unified_pack(
    out_dir: Path,
    zip_path: Path,
    seed: int = 42,
    n_cam16: int = 25,
    pannuke_organs: tuple[str, ...] = ("Breast", "Colon", "Lung"),
    n_per_pannuke_organ: int = 25,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(out_dir.glob("*.png")) + list(out_dir.glob("*.csv")):
        stale.unlink(missing_ok=True)

    records: list[dict[str, object]] = []

    # 1. Track A: 300 Lossless IDC RMS tiles (Soft_Tissue, H&E)
    rms_mf_path = Path("results/curated_v1/manifest_track_a.csv")
    rms_tiles_dir = Path("tiles_curated_v1/track_a_rms")
    rms_mf = pd.read_csv(rms_mf_path)
    rms_fast = rms_mf[rms_mf["is_fast_subset"].astype(str).str.lower() == "true"].copy()
    if "track" in rms_fast.columns:
        rms_fast = rms_fast[rms_fast["track"].astype(str) == "A"].copy()
    if "tile_size" in rms_fast.columns:
        rms_fast = rms_fast[pd.to_numeric(rms_fast["tile_size"], errors="coerce") == 256].copy()
    assert len(rms_fast) == 300, f"Expected 300 fast 256x256 Track A RMS tiles, found {len(rms_fast)}"

    for _, row in rms_fast.iterrows():
        tid = str(row["tile_id"])
        src_png = rms_tiles_dir / f"{tid}.png"
        if not src_png.exists():
            raise FileNotFoundError(f"Missing RMS tile PNG: {src_png}")
        with Image.open(src_png) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        assert arr.shape == (256, 256, 3), f"Unexpected RMS tile shape {arr.shape} in {src_png}"
        shutil.copy2(src_png, out_dir / f"{tid}.png")
        records.append({
            "tile_id": tid,
            "track": "A",
            "source_track": "A",
            "source_dataset": "IDC_RMS_Lossless",
            "gt_type": "lossless",
            "organ": "Soft_Tissue",
            "stain": "H&E",
            "stain_detail": str(row.get("stain_stratum", "H&E")),
            "patient_id": str(row.get("patient_id", tid.split("_")[0])),
            "blockiness_8x8": round(measure_8x8_blockiness(arr), 4),
            "is_fast_subset": True,
        })
    print(f"[1/4] IDC RMS Lossless H&E (Soft_Tissue): {len(rms_fast)} tiles")

    # 2. Track B3: 40 Lossless ANHIR IHC_DAB tiles
    b3_mf = pd.read_csv("results/curated_v1/manifest_track_b3_ihc.csv")
    b3_imgs = np.load("anhir_data/ihc_tiles_u8.npy", mmap_mode="r")
    b3_fast = b3_mf[b3_mf["is_fast_subset"].astype(str).str.lower() == "true"].copy()
    assert len(b3_fast) == 40, f"Expected 40 fast Track B3 ANHIR tiles, found {len(b3_fast)}"

    for _, row in b3_fast.iterrows():
        tid = str(row["tile_id"])
        idx = int(row["tile_idx"])
        prof = str(row["subset_or_profile"])
        arr = np.asarray(b3_imgs[idx], dtype=np.uint8)
        _save_tile_png(arr, out_dir / f"{tid}.png")
        records.append({
            "tile_id": tid,
            "track": "A",
            "source_track": "B3",
            "source_dataset": str(row.get("source_type", "ANHIR_CTU_PNG")),
            "gt_type": "lossless",
            "organ": _infer_anhir_organ(tid, prof),
            "stain": "IHC_DAB",
            "stain_detail": prof,
            "patient_id": tid.split("_scale-")[0],
            "blockiness_8x8": round(measure_8x8_blockiness(arr), 4),
            "is_fast_subset": True,
        })
    print(f"[2/4] ANHIR Lossless IHC_DAB: {len(b3_fast)} tiles")

    # 3. Track B1: 25 Lossy CAMELYON16 Level-0 tiles (Lymph_Node, stratified across slides)
    b1_mf = pd.read_csv("results/curated_v1/manifest_track_b1_tumor.csv")
    b1_imgs = np.load("camelyon_data/b1_tiles_u8.npy", mmap_mode="r")
    b1_sel = _sample_stratified(b1_mf, "slide_id", n_cam16, seed=seed)

    for _, row in b1_sel.iterrows():
        tid = str(row["tile_id"])
        idx = int(row["tile_idx"])
        arr = np.asarray(b1_imgs[idx], dtype=np.uint8)
        _save_tile_png(arr, out_dir / f"{tid}.png")
        records.append({
            "tile_id": tid,
            "track": "A",
            "source_track": "B1",
            "source_dataset": "CAMELYON16_Level0",
            "gt_type": "lossy",
            "organ": "Lymph_Node",
            "stain": "H&E",
            "stain_detail": f"H&E_{row.get('class_name', 'tissue')}",
            "patient_id": str(row.get("slide_id", "cam16")),
            "blockiness_8x8": round(measure_8x8_blockiness(arr), 4),
            "is_fast_subset": True,
        })
    print(f"[3/4] CAMELYON16 Lossy H&E (Lymph_Node): {len(b1_sel)} tiles across {b1_sel['slide_id'].nunique()} slides")

    # 4. Track B2: 75 Lossy PanNuke Fold 3 tiles (Breast, Colon, Lung x 25, random seed, NOT top-nuclei)
    b2_mf = pd.read_csv("results/curated_v1/manifest_track_b2_pannuke.csv")
    b2_imgs = np.load("pannuke_fold3/images_u8.npy", mmap_mode="r")
    rng_b2 = np.random.default_rng(seed)

    b2_total = 0
    for organ in pannuke_organs:
        raw_pool = b2_mf[b2_mf["organ"].astype(str) == organ].copy()
        # Exclude empty glass / background-dominant patches before random sampling
        keep_mask = []
        for _, r in raw_pool.iterrows():
            nuc_frac = float(r.get("nuclei_area_fraction", 0.0))
            arr_chk = np.asarray(b2_imgs[int(r["tile_idx"])], dtype=np.uint8)
            tissue_frac = float(np.mean(arr_chk.mean(axis=-1) < 225.0))
            keep_mask.append(nuc_frac >= 0.01 and tissue_frac >= 0.50)
        pool = raw_pool[keep_mask].copy()
        assert len(pool) >= n_per_pannuke_organ, (
            f"PanNuke organ '{organ}' has only {len(pool)} tissue-rich tiles out of {len(raw_pool)} (< {n_per_pannuke_organ})"
        )
        perm = rng_b2.permutation(len(pool))[:n_per_pannuke_organ]
        sel = pool.iloc[perm]
        for _, row in sel.iterrows():
            tid = str(row["tile_id"])
            idx = int(row["tile_idx"])
            arr = np.asarray(b2_imgs[idx], dtype=np.uint8)
            _save_tile_png(arr, out_dir / f"{tid}.png")
            records.append({
                "tile_id": tid,
                "track": "A",
                "source_track": "B2",
                "source_dataset": "PanNuke_Fold3",
                "gt_type": "lossy",
                "organ": str(organ),
                "stain": "H&E",
                "stain_detail": "H&E_PanNuke",
                "patient_id": f"pannuke_f3_{organ}",
                "blockiness_8x8": round(measure_8x8_blockiness(arr), 4),
                "is_fast_subset": True,
            })
        b2_total += len(sel)
        print(f"[4/4] PanNuke_Fold3 Lossy H&E ({organ}): {len(sel)} random tiles from {len(pool)}/{len(raw_pool)} tissue-filtered")

    mf_out = pd.DataFrame(records)
    assert mf_out["tile_id"].is_unique, "Duplicate tile_id detected in unified manifest!"
    expected_total = 300 + 40 + n_cam16 + len(pannuke_organs) * n_per_pannuke_organ
    assert len(mf_out) == expected_total, f"Expected {expected_total} rows, got {len(mf_out)}"

    manifest_path = out_dir / "manifest_unified_440.csv"
    mf_out.to_csv(manifest_path, index=False)

    # Create zip bundle for Google Drive upload
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(manifest_path, arcname=f"{out_dir.name}/{manifest_path.name}")
        for png in sorted(out_dir.glob("*.png")):
            zf.write(png, arcname=f"{out_dir.name}/{png.name}")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("\n=== Unified Pack Summary ===")
    print(
        mf_out.groupby(["gt_type", "organ", "stain", "source_dataset"], sort=True)
        .agg(N_Tiles=("tile_id", "count"), Mean_Blockiness=("blockiness_8x8", "mean"))
        .round(4)
        .to_string()
    )
    print(f"\nManifest: {manifest_path.resolve()} ({len(mf_out)} unique tiles)")
    print(f"ZIP ready for Google Drive: {zip_path.resolve()} ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified 440-tile Colab pack + manifest_unified_440.csv.")
    p.add_argument("--out-dir", type=Path, default=Path("tiles_unified_440"))
    p.add_argument("--zip-out", type=Path, default=Path("colab_bundle_440.zip"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_unified_pack(args.out_dir, args.zip_out, seed=args.seed)
