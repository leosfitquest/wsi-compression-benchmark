"""
audit_track_a_forensics.py
--------------------------
1. Forensischer 8x8-Blockartefakt-Check (Ausschluss von DICOM-Metadaten-Fallen
   wie HTAN-Vanderbilt trotz LossyImageCompression == '00').
2. Confounder-freies Generation-Loss-Experiment auf identischen 40x-RMS-Kacheln:
   Vergleicht die R-D-Kurven und BD-Raten auf denselben Kacheln einmal nativ
   verlustfrei (Ground Truth) vs. synthetisch Scanner-vorkomprimiert (JPEG Q=70).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from bdrate_summary import bd_psnr, bd_rate
from benchmark_conventional import build_codecs, match_target_bpp, DEFAULT_TARGETS_BPP, RATE_TOLERANCE
from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb, psnr_y


def measure_8x8_blockiness(tile: np.ndarray) -> float:
    """Misst das Verhaeltnis der Pixel-Differenzen an 8x8-Blockgrenzen vs. Nicht-Blockgrenzen.
    Reine Lossless-Bilder liegen nahe 1.00; JPEG-vorkomprimierte Bilder typischerweise > 1.15.
    """
    gray = tile.astype(np.float64).mean(axis=-1)
    h, w = gray.shape

    # Horizontale und vertikale Nachbardifferenzen
    dh = np.abs(gray[1:, :] - gray[:-1, :])
    dw = np.abs(gray[:, 1:] - gray[:, :-1])

    h_boundary_idx = [i for i in range(7, h - 1, 8)]
    h_interior_idx = [i for i in range(h - 1) if (i + 1) % 8 != 0]

    w_boundary_idx = [j for j in range(7, w - 1, 8)]
    w_interior_idx = [j for j in range(w - 1) if (j + 1) % 8 != 0]

    b_energy = float(dh[h_boundary_idx, :].mean() + dw[:, w_boundary_idx].mean())
    i_energy = float(dh[h_interior_idx, :].mean() + dw[:, w_interior_idx].mean())
    return b_energy / max(i_energy, 1e-8)


def simulate_scanner_q70(tile: np.ndarray, quality: int = 70) -> tuple[np.ndarray, float]:
    """Simuliert eine Aperio-Scanner-Vorkompression (JPEG Q=70, 4:4:4 RGB/YCbCr ohne Subsampling)."""
    buf = io.BytesIO()
    Image.fromarray(tile).save(buf, format="JPEG", quality=quality, subsampling=0, optimize=True)
    data = buf.getvalue()
    bpp_scanner = bits_per_pixel(len(data), tile.shape[0], tile.shape[1])
    with Image.open(io.BytesIO(data)) as im:
        precomp = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return precomp, bpp_scanner


def run_forensic_and_ablation(
    tiles_dir: Path = Path("tiles_curated_v1/track_a_rms"),
    manifest_path: Path = Path("results/curated_v1/manifest_track_a.csv"),
    out_dir: Path = Path("results/curated_v1"),
    max_ablation_tiles: int = 20,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest nicht gefunden: {manifest_path}")

    mf = pd.read_csv(manifest_path)
    mf_a = mf[mf["track"] == "A"].copy()
    fast_df = mf_a[mf_a["is_fast_subset"].astype(str).str.lower() == "true"].head(max_ablation_tiles)

    print(f"=== 1. Forensischer 8x8-Block-Audit auf {len(fast_df)} Track-A-Kacheln ===")
    block_rows = []
    loaded_tiles: list[tuple[str, np.ndarray, np.ndarray]] = []

    for _, r in fast_df.iterrows():
        tid = str(r["tile_id"])
        p = tiles_dir / f"{tid}.png"
        if not p.exists():
            continue
        with Image.open(p) as im:
            tile_gt = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()

        tile_q70, q70_bpp = simulate_scanner_q70(tile_gt, quality=70)
        b_gt = measure_8x8_blockiness(tile_gt)
        b_q70 = measure_8x8_blockiness(tile_q70)

        block_rows.append({
            "tile_id": tid,
            "patient_id": r["patient_id"],
            "transfer_syntax": r["transfer_syntax"],
            "blockiness_lossless_gt": round(b_gt, 4),
            "blockiness_simulated_q70": round(b_q70, 4),
            "scanner_q70_bpp": round(q70_bpp, 3),
            "scanner_q70_psnr_db": round(psnr_rgb(tile_gt, tile_q70), 2),
        })
        loaded_tiles.append((tid, tile_gt, tile_q70))

    df_block = pd.DataFrame(block_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    forensic_csv = out_dir / "forensic_audit_track_a.csv"
    df_block.to_csv(forensic_csv, index=False)

    print(f"Mittlere 8x8-Blockiness (Lossless GT):      {df_block['blockiness_lossless_gt'].mean():.4f} (ideal ~1.00)")
    print(f"Mittlere 8x8-Blockiness ( nach JPEG Q=70):  {df_block['blockiness_simulated_q70'].mean():.4f} (deutliches 8x8-Gitter)")
    print(f"Mittlere Bitrate von JPEG Q=70 auf Gewebe:  {df_block['scanner_q70_bpp'].mean():.2f} bpp "
          f"(PSNR vs. GT: {df_block['scanner_q70_psnr_db'].mean():.2f} dB)")
    print(f"Forensik-Tabelle gespeichert: {forensic_csv.resolve()}\n")

    print(f"=== 2. Confounder-freies Generation-Loss-Experiment ({len(loaded_tiles)} identische 40x-Kacheln) ===")
    codecs = build_codecs(jxl_effort=7, verbose=False)
    ablation_records = []

    for mode_label, idx_tile in [("Lossless_40x_GT", 1), ("PreCompressed_40x_Q70", 2)]:
        for item in loaded_tiles:
            tid = item[0]
            ref_tile = item[idx_tile]
            h, w = ref_tile.shape[:2]
            for codec in codecs:
                cache = {}
                for target in DEFAULT_TARGETS_BPP:
                    param, enc = match_target_bpp(codec, ref_tile, target, cache=cache)
                    bpp = bits_per_pixel(enc.num_bytes, h, w)
                    ablation_records.append({
                        "mode": mode_label,
                        "tile_id": tid,
                        "codec": codec.name,
                        "target_bpp": target,
                        "bpp": bpp,
                        "within_tolerance": abs(bpp - target) / target <= RATE_TOLERANCE,
                        "psnr_db": psnr_rgb(ref_tile, enc.decoded),
                        "psnr_y_db": psnr_y(ref_tile, enc.decoded),
                        "ms_ssim": ms_ssim_rgb(ref_tile, enc.decoded),
                    })

    df_ab = pd.DataFrame(ablation_records)
    summary_rows = []
    for mode_label, grp_m in df_ab.groupby("mode"):
        filtered = grp_m.groupby(["codec", "target_bpp"]).filter(
            lambda g: abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15
        )
        means = filtered.groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "ms_ssim"]].mean().reset_index()
        anchor_df = means[means["codec"] == "JPEG 4:4:4"].sort_values("bpp")
        r_ref, p_ref = anchor_df["bpp"].to_numpy(), anchor_df["psnr_db"].to_numpy()
        s_ref = -10.0 * np.log10(np.clip(1.0 - anchor_df["ms_ssim"].to_numpy(), 1e-10, 1.0))

        for codec_name, c_df in means.groupby("codec"):
            c_df = c_df.sort_values("bpp")
            c_df = c_df.loc[c_df["bpp"].round(3).drop_duplicates(keep="last").index]
            r_t, p_t = c_df["bpp"].to_numpy(), c_df["psnr_db"].to_numpy()
            s_t = -10.0 * np.log10(np.clip(1.0 - c_df["ms_ssim"].to_numpy(), 1e-10, 1.0))
            bdr_p = 0.0 if codec_name == "JPEG 4:4:4" else bd_rate(r_ref, p_ref, r_t, p_t)
            bdp = 0.0 if codec_name == "JPEG 4:4:4" else bd_psnr(r_ref, p_ref, r_t, p_t)
            bdr_s = 0.0 if codec_name == "JPEG 4:4:4" else bd_rate(r_ref, s_ref, r_t, s_t)
            summary_rows.append({
                "Regime (Same 40x Tiles)": mode_label,
                "Codec": codec_name,
                "BD-Rate PSNR [%]": round(bdr_p, 2),
                "BD-PSNR [dB]": round(bdp, 2),
                "BD-Rate MS-SSIM [%]": round(bdr_s, 2),
                "PSNR @ 1.50 bpp [dB]": round(float(p_t[-1]), 2),
            })

    sum_df = pd.DataFrame(summary_rows)
    ablation_csv = out_dir / "generation_loss_ablation.csv"
    sum_df.to_csv(ablation_csv, index=False)
    print(sum_df.to_string(index=False))
    print(f"\nAblations-Tabelle gespeichert: {ablation_csv.resolve()}")


if __name__ == "__main__":
    run_forensic_and_ablation()