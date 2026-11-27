"""
bdrate_summary.py
-----------------
Computes Bjontegaard-Delta Rate (BD-Rate in %) and BD-PSNR (in dB) relative to
JPEG 4:4:4 using the official FAU-LMS `bjontegaard` library (`method="akima"`),
with ITU-T VCEG-AI11 PCHIP retained as an optional comparison column.

Supports:
- Unified multi-codec CSV schema (`rate_mode` in {bisect, fixed_qp, neural_q, brm} + `setting_id`)
- Backward compatibility with legacy pilot & curated_v1 CSVs
- Upper-saturation preservation for bisected codecs on lossy Q=70 slides (e.g. WebP/JPEG 4:2:0)
- Single-source-of-truth manifest join (`gt_type`, `organ`, `stain`) with uniqueness & strict tile-set pairing
- Three Track A views:
  (1) Overall R-D (Stratum-Balanced Macro Mean + Tile-Pooled)
  (2) Lossless vs. Lossy R-D
  (3) R-D by Organ & Stain
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence

import bjontegaard as bj
import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator


def _prepare_monotone_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filters non-finite entries, sorts by x, and averages duplicate x-values."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if len(x_arr) == 0:
        return x_arr, y_arr

    order = np.argsort(x_arr)
    xs, ys = x_arr[order], y_arr[order]
    ux, inv = np.unique(xs, return_inverse=True)
    if len(ux) < len(xs):
        uy = np.array([ys[inv == i].mean() for i in range(len(ux))], dtype=np.float64)
        return ux, uy
    return xs, ys


def _prepare_strict_rd(
    rate: np.ndarray,
    dist: np.ndarray,
    rate_eps: float = 1e-9,
    dist_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sanitizes (rate, dist) pairs so both rate and dist are finite and strictly increasing."""
    r = np.asarray(rate, dtype=np.float64)
    d = np.asarray(dist, dtype=np.float64)
    valid = np.isfinite(r) & np.isfinite(d) & (r > 0.0)
    r, d = r[valid], d[valid]
    if len(r) == 0:
        return r, d

    order = np.argsort(r)
    r, d = r[order], d[order]
    kept_r: list[float] = [float(r[0])]
    kept_d: list[float] = [float(d[0])]
    for ri, di in zip(r[1:], d[1:]):
        if float(ri) > kept_r[-1] + rate_eps and float(di) > kept_d[-1] + dist_eps:
            kept_r.append(float(ri))
            kept_d.append(float(di))
    return np.asarray(kept_r, dtype=np.float64), np.asarray(kept_d, dtype=np.float64)


def bd_psnr_pchip(
    rate_ref: np.ndarray,
    psnr_ref: np.ndarray,
    rate_test: np.ndarray,
    psnr_test: np.ndarray,
    fixed_range: tuple[float, float] | None = None,
    strict_fixed_range: bool = True,
) -> float:
    """Average PSNR difference (dB) via PCHIP over the overlapping or fixed log-rate interval."""
    lr_ref, p_ref = _prepare_monotone_xy(np.log10(np.asarray(rate_ref, dtype=np.float64)), psnr_ref)
    lr_test, p_test = _prepare_monotone_xy(np.log10(np.asarray(rate_test, dtype=np.float64)), psnr_test)
    if len(lr_ref) < 4 or len(lr_test) < 4:
        return float("nan")

    min_lr = max(float(lr_ref[0]), float(lr_test[0]))
    max_lr = min(float(lr_ref[-1]), float(lr_test[-1]))
    if fixed_range is not None:
        lo, hi = float(np.log10(fixed_range[0])), float(np.log10(fixed_range[1]))
        if strict_fixed_range and (min_lr > lo + 0.015 or max_lr < hi - 0.015):
            return float("nan")
        min_lr, max_lr = max(min_lr, lo), min(max_lr, hi)
    if max_lr <= min_lr:
        return float("nan")

    f_ref = PchipInterpolator(lr_ref, p_ref)
    f_test = PchipInterpolator(lr_test, p_test)
    return float((float(f_test.integrate(min_lr, max_lr)) - float(f_ref.integrate(min_lr, max_lr))) / (max_lr - min_lr))


def bd_rate_pchip(rate_ref: np.ndarray, dist_ref: np.ndarray, rate_test: np.ndarray, dist_test: np.ndarray) -> float:
    """Average bitrate savings (%) via PCHIP over the overlapping distortion interval."""
    d_ref, lr_ref = _prepare_monotone_xy(dist_ref, np.log10(np.asarray(rate_ref, dtype=np.float64)))
    d_test, lr_test = _prepare_monotone_xy(dist_test, np.log10(np.asarray(rate_test, dtype=np.float64)))
    if len(d_ref) < 4 or len(d_test) < 4:
        return float("nan")

    min_d = max(float(d_ref[0]), float(d_test[0]))
    max_d = min(float(d_ref[-1]), float(d_test[-1]))
    if max_d <= min_d:
        return float("nan")

    p_ref = PchipInterpolator(d_ref, lr_ref)
    p_test = PchipInterpolator(d_test, lr_test)
    avg_log_diff = (float(p_test.integrate(min_d, max_d)) - float(p_ref.integrate(min_d, max_d))) / (max_d - min_d)
    return float((10.0**avg_log_diff - 1.0) * 100.0)


def bd_rate(
    rate_ref: np.ndarray,
    dist_ref: np.ndarray,
    rate_test: np.ndarray,
    dist_test: np.ndarray,
    method: str = "akima",
    min_overlap_frac: float = 0.25,
) -> float:
    """Average bitrate savings (%) via FAU-LMS `bjontegaard` (default: Akima) with strict safety guards."""
    if method == "pchip":
        return bd_rate_pchip(rate_ref, dist_ref, rate_test, dist_test)

    r_ref, d_ref = _prepare_strict_rd(rate_ref, dist_ref)
    r_test, d_test = _prepare_strict_rd(rate_test, dist_test)
    if len(r_ref) < 4 or len(r_test) < 4:
        return float("nan")

    min_d = max(float(d_ref[0]), float(d_test[0]))
    max_d = min(float(d_ref[-1]), float(d_test[-1]))
    ref_span = float(d_ref[-1] - d_ref[0])
    if max_d <= min_d or ref_span <= 0.0 or (max_d - min_d) < min_overlap_frac * ref_span:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = float(
                bj.bd_rate(
                    r_ref,
                    d_ref,
                    r_test,
                    d_test,
                    method=method,
                    require_matching_points=False,
                    min_overlap=0.0,
                )
            )
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def bd_psnr(
    rate_ref: np.ndarray,
    psnr_ref: np.ndarray,
    rate_test: np.ndarray,
    psnr_test: np.ndarray,
    fixed_range: tuple[float, float] | None = None,
    strict_fixed_range: bool = True,
    method: str = "akima",
    min_overlap_frac: float = 0.25,
) -> float:
    """Average PSNR difference (dB) via FAU-LMS `bjontegaard` (Akima) or Akima fixed-range integration."""
    if method == "pchip":
        return bd_psnr_pchip(rate_ref, psnr_ref, rate_test, psnr_test, fixed_range, strict_fixed_range)

    r_ref, p_ref = _prepare_strict_rd(rate_ref, psnr_ref)
    r_test, p_test = _prepare_strict_rd(rate_test, psnr_test)
    if len(r_ref) < 4 or len(r_test) < 4:
        return float("nan")

    lr_ref = np.log10(r_ref)
    lr_test = np.log10(r_test)
    min_lr = max(float(lr_ref[0]), float(lr_test[0]))
    max_lr = min(float(lr_ref[-1]), float(lr_test[-1]))

    if fixed_range is not None:
        lo, hi = float(np.log10(fixed_range[0])), float(np.log10(fixed_range[1]))
        if strict_fixed_range and (min_lr > lo + 0.015 or max_lr < hi - 0.015):
            return float("nan")
        min_lr, max_lr = max(min_lr, lo), min(max_lr, hi)
        if max_lr <= min_lr:
            return float("nan")
        f_ref = Akima1DInterpolator(lr_ref, p_ref)
        f_test = Akima1DInterpolator(lr_test, p_test)
        return float((float(f_test.integrate(min_lr, max_lr)) - float(f_ref.integrate(min_lr, max_lr))) / (max_lr - min_lr))

    ref_span = float(lr_ref[-1] - lr_ref[0])
    if max_lr <= min_lr or ref_span <= 0.0 or (max_lr - min_lr) < min_overlap_frac * ref_span:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = float(
                bj.bd_psnr(
                    r_ref,
                    p_ref,
                    r_test,
                    p_test,
                    method=method,
                    require_matching_points=False,
                    min_overlap=0.0,
                )
            )
    except Exception:
        return float("nan")
    return val if np.isfinite(val) else float("nan")


def load_codec_csvs(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Loads and normalizes one or more codec CSVs (legacy or unified 12-codec schema)."""
    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        if not p.exists():
            continue
        raw = pd.read_csv(p)
        if raw.empty:
            continue
        df = pd.DataFrame()
        df["tile_id"] = raw["tile_id"].astype(str)

        if "codec" in raw.columns:
            df["codec"] = raw["codec"].astype(str)
        elif "model" in raw.columns:
            df["codec"] = raw["model"].astype(str)
        elif "codec_or_model" in raw.columns:
            df["codec"] = raw["codec_or_model"].astype(str)
        else:
            raise ValueError(f"Missing codec/model column in {p}")

        if "rate_mode" in raw.columns:
            df["rate_mode"] = raw["rate_mode"].astype(str)
        elif "category" in raw.columns:
            df["rate_mode"] = np.where(raw["category"].astype(str) == "conventional", "bisect", "neural_q")
        elif "model" in raw.columns and "quality" in raw.columns:
            df["rate_mode"] = "neural_q"
        elif "target_bpp" in raw.columns:
            df["rate_mode"] = "bisect"
        else:
            df["rate_mode"] = "fixed_qp"

        if "setting_id" in raw.columns:
            df["setting_id"] = raw["setting_id"].astype(str)
        elif "quality" in raw.columns:
            df["setting_id"] = raw["quality"].map(lambda q: f"q_{int(q)}")
        elif "target_bpp" in raw.columns:
            df["setting_id"] = raw["target_bpp"].map(lambda t: f"target_{float(t):.2f}")
        elif "setting" in raw.columns:
            df["setting_id"] = raw["setting"].map(lambda s: f"s_{s}")
        else:
            df["setting_id"] = "default"

        df["target_bpp"] = pd.to_numeric(raw["target_bpp"], errors="coerce") if "target_bpp" in raw.columns else np.nan

        if "bpp_real" in raw.columns and raw["bpp_real"].notna().any():
            df["bpp"] = pd.to_numeric(raw["bpp_real"], errors="coerce")
        elif "bpp" in raw.columns:
            df["bpp"] = pd.to_numeric(raw["bpp"], errors="coerce")
        elif "bpp_est" in raw.columns:
            df["bpp"] = pd.to_numeric(raw["bpp_est"], errors="coerce")
        else:
            raise ValueError(f"Missing bpp column in {p}")

        if "within_tolerance" in raw.columns:
            df["within_tolerance"] = raw["within_tolerance"].astype(bool)
        else:
            df["within_tolerance"] = True

        psnr_col = "psnr_db" if "psnr_db" in raw.columns else "psnr_rgb_db"
        df["psnr_db"] = pd.to_numeric(raw[psnr_col], errors="coerce")
        df["ms_ssim"] = pd.to_numeric(raw["ms_ssim"], errors="coerce")

        for col in ("psnr_y_db", "psnr_u_db", "psnr_v_db", "psnr_yuv_db"):
            df[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan

        frames.append(df.dropna(subset=["bpp", "psnr_db"]))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_curves_from_df(
    df: pd.DataFrame,
    allowed_tiles: set[str] | None = None,
    strict_all_tolerance: bool = False,
    strict_pairing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Aggregates normalized codec rows into per-codec R-D curves respecting `rate_mode`."""
    if df.empty:
        return {}
    work = df.copy()
    if allowed_tiles is not None:
        work = work[work["tile_id"].isin(allowed_tiles)].copy()
    if work.empty:
        return {}

    if strict_pairing:
        codec_tile_sets = {
            str(codec): set(grp["tile_id"].unique()) for codec, grp in work.groupby("codec")
        }
        if len(codec_tile_sets) > 1:
            shared_tiles = set.intersection(*codec_tile_sets.values())
            for codec, tset in codec_tile_sets.items():
                if len(tset) != len(shared_tiles):
                    print(f"[warn] Stratum tile-set pairing: {codec} had {len(tset)} tiles, intersecting to {len(shared_tiles)} shared tiles.")
            work = work[work["tile_id"].isin(shared_tiles)].copy()
            if work.empty:
                return {}

    metric_cols = ["bpp", "psnr_db", "ms_ssim"]
    for opt in ("psnr_y_db", "psnr_u_db", "psnr_v_db", "psnr_yuv_db"):
        if opt in work.columns and work[opt].notna().any():
            metric_cols.append(opt)

    curves: dict[str, pd.DataFrame] = {}
    for codec, cgrp in work.groupby("codec"):
        mode = str(cgrp["rate_mode"].iloc[0])
        if mode == "bisect":
            grouped = (
                cgrp.groupby(["setting_id", "target_bpp"], dropna=False)
                .agg(
                    **{col: (col, "mean") for col in metric_cols},
                    all_tol=("within_tolerance", "all"),
                )
                .reset_index()
                .sort_values("target_bpp")
            )
            kept_rows = []
            for _, row in grouped.iterrows():
                tbpp = float(row["target_bpp"])
                mbpp = float(row["bpp"])
                mpsnr = float(row["psnr_db"])
                if strict_all_tolerance:
                    if bool(row["all_tol"]):
                        kept_rows.append(row)
                    continue
                rel_err = abs(mbpp - tbpp) / tbpp if tbpp > 0 else 0.0
                if rel_err <= 0.15:
                    kept_rows.append(row)
                elif tbpp >= 1.0 and mbpp < tbpp and kept_rows:
                    # Upper saturation guard for pre-compressed Q=70 slides: keep strictly rising top points
                    prev_bpp = float(kept_rows[-1]["bpp"])
                    prev_psnr = float(kept_rows[-1]["psnr_db"])
                    if mbpp >= prev_bpp + 0.01 and mpsnr > prev_psnr:
                        kept_rows.append(row)
            if not kept_rows:
                continue
            out_grp = pd.DataFrame(kept_rows).sort_values("bpp")
            out_grp = out_grp.loc[out_grp["bpp"].round(3).drop_duplicates(keep="last").index]
            curves[str(codec)] = out_grp[metric_cols].reset_index(drop=True)
        else:
            # fixed_qp (HM/VTM), neural_q (CompressAI/MLIC++), brm (JPEG AI VM): never apply 15% filter
            out_grp = cgrp.groupby("setting_id")[metric_cols].mean().reset_index().sort_values("bpp")
            out_grp = out_grp.loc[out_grp["bpp"].round(4).drop_duplicates(keep="last").index]
            curves[str(codec)] = out_grp[metric_cols].reset_index(drop=True)

    return curves


def prepare_curves(
    conv_csv: Path,
    ai_csv: Path | None,
    allowed_tiles: set[str] | None = None,
    strict_all_tolerance: bool = False,
    pair_with_ai_tiles: bool = True,
) -> dict[str, pd.DataFrame]:
    """Backward-compatible wrapper loading conv + optional ai CSVs."""
    paths = [conv_csv]
    if ai_csv is not None and ai_csv.exists():
        paths.append(ai_csv)
    df = load_codec_csvs(paths)
    return build_curves_from_df(
        df,
        allowed_tiles=allowed_tiles,
        strict_all_tolerance=strict_all_tolerance,
        strict_pairing=pair_with_ai_tiles,
    )


def score_curves_table(
    cohort_name: str,
    curves: dict[str, pd.DataFrame],
    anchor: str = "JPEG 4:4:4",
    include_fixed_interval: bool = False,
    include_pchip_col: bool = True,
) -> pd.DataFrame | None:
    if anchor not in curves:
        return None
    ref = curves[anchor]
    r_ref = ref["bpp"].to_numpy()
    p_ref = ref["psnr_db"].to_numpy()
    has_y = "psnr_y_db" in ref.columns and ref["psnr_y_db"].notna().all()
    has_yuv = "psnr_yuv_db" in ref.columns and ref["psnr_yuv_db"].notna().all()
    py_ref = ref["psnr_y_db"].to_numpy() if has_y else None
    pyuv_ref = ref["psnr_yuv_db"].to_numpy() if has_yuv else None
    s_ref = -10.0 * np.log10(np.clip(1.0 - ref["ms_ssim"].to_numpy(), 1e-10, 1.0))

    rows = []
    for name, df_c in curves.items():
        r_t = df_c["bpp"].to_numpy()
        p_t = df_c["psnr_db"].to_numpy()
        s_t = -10.0 * np.log10(np.clip(1.0 - df_c["ms_ssim"].to_numpy(), 1e-10, 1.0))

        if name == anchor:
            bdr_rgb = bdp = bdp_fixed = bdr_y = bdr_yuv = bdr_ssim = bdr_pchip = 0.0
        else:
            bdr_rgb = bd_rate(r_ref, p_ref, r_t, p_t, method="akima")
            bdp = bd_psnr(r_ref, p_ref, r_t, p_t, method="akima")
            bdp_fixed = bd_psnr(r_ref, p_ref, r_t, p_t, fixed_range=(0.25, 0.90), strict_fixed_range=True, method="akima")
            bdr_ssim = bd_rate(r_ref, s_ref, r_t, s_t, method="akima")
            bdr_y = bd_rate(r_ref, py_ref, r_t, df_c["psnr_y_db"].to_numpy(), method="akima") if (has_y and py_ref is not None and "psnr_y_db" in df_c.columns) else float("nan")
            bdr_yuv = bd_rate(r_ref, pyuv_ref, r_t, df_c["psnr_yuv_db"].to_numpy(), method="akima") if (has_yuv and pyuv_ref is not None and "psnr_yuv_db" in df_c.columns) else float("nan")
            bdr_pchip = bd_rate_pchip(r_ref, p_ref, r_t, p_t) if include_pchip_col else float("nan")

        row_dict: dict[str, object] = {
            "Cohort": cohort_name,
            "Codec / Model": name,
            "BD-Rate RGB-PSNR [%]": round(bdr_rgb, 2) if np.isfinite(bdr_rgb) else float("nan"),
            "BD-PSNR [dB]": round(bdp, 2) if np.isfinite(bdp) else float("nan"),
        }
        if include_fixed_interval:
            row_dict["BD-PSNR [0.25-0.90] [dB]"] = round(bdp_fixed, 2) if np.isfinite(bdp_fixed) else float("nan")
        if has_y:
            row_dict["BD-Rate Y-PSNR [%]"] = round(bdr_y, 2) if np.isfinite(bdr_y) else float("nan")
        if has_yuv:
            row_dict["BD-Rate YUV-PSNR [%]"] = round(bdr_yuv, 2) if np.isfinite(bdr_yuv) else float("nan")
        row_dict["BD-Rate MS-SSIM [%]"] = round(bdr_ssim, 2) if np.isfinite(bdr_ssim) else float("nan")
        if include_pchip_col:
            row_dict["BD-Rate RGB PCHIP [%]"] = round(bdr_pchip, 2) if np.isfinite(bdr_pchip) else float("nan")
        row_dict["Min bpp"] = round(float(r_t.min()), 3)
        row_dict["Max bpp"] = round(float(r_t.max()), 3)
        rows.append(row_dict)

    return pd.DataFrame(rows).sort_values("BD-Rate RGB-PSNR [%]")


def evaluate_cohort(
    cohort_name: str,
    conv_csv: Path,
    ai_csv: Path | None,
    out_csv: Path,
    anchor: str = "JPEG 4:4:4",
    allowed_tiles: set[str] | None = None,
    strict_all_tolerance: bool = False,
    include_fixed_interval: bool = False,
    pair_with_ai_tiles: bool = True,
) -> pd.DataFrame | None:
    curves = prepare_curves(
        conv_csv,
        ai_csv,
        allowed_tiles=allowed_tiles,
        strict_all_tolerance=strict_all_tolerance,
        pair_with_ai_tiles=pair_with_ai_tiles,
    )
    res_df = score_curves_table(cohort_name, curves, anchor=anchor, include_fixed_interval=include_fixed_interval)
    if res_df is None:
        return None
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"\n=== {cohort_name} (Anchor: {anchor}, Method: Akima via bjontegaard) ===")
    print(res_df.to_string(index=False))
    print(f"Saved to: {out_csv.resolve()}")
    return res_df


def load_unified_manifest(manifest_csv: Path) -> pd.DataFrame:
    """Loads single-source-of-truth manifest, verifies unique tile_id, and supplies safe defaults for legacy manifests."""
    mf = pd.read_csv(manifest_csv)
    mf["tile_id"] = mf["tile_id"].astype(str)
    assert mf["tile_id"].is_unique, f"Duplicate tile_id entries detected in {manifest_csv}!"
    if "gt_type" not in mf.columns:
        mf["gt_type"] = "lossless"
    if "organ" not in mf.columns:
        mf["organ"] = "Soft_Tissue"
    if "stain" not in mf.columns:
        mf["stain"] = "H&E"
    return mf


def evaluate_unified_benchmark(
    csv_paths: Sequence[Path],
    manifest_csv: Path,
    out_dir: Path,
    anchor: str = "JPEG 4:4:4",
) -> None:
    """Runs all 3 required Track A evaluations via central manifest join:
    (1) Overall (Stratum-Balanced Macro Mean + Tile-Pooled)
    (2) Lossless vs. Lossy (`gt_type`)
    (3) Organ & Stain (`gt_type`, `organ`, `stain`)
    """
    mf = load_unified_manifest(manifest_csv)
    df_all = load_codec_csvs(csv_paths)
    if df_all.empty:
        raise ValueError("No codec rows loaded from provided CSV paths.")

    meta_cols = [c for c in ("tile_id", "gt_type", "organ", "stain", "stain_detail", "patient_id") if c in mf.columns]
    joined = df_all.merge(mf[meta_cols], on="tile_id", how="inner", validate="many_to_one")
    if joined.empty:
        raise ValueError("Manifest join yielded 0 rows ? check tile_id prefixes between CSVs and manifest.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # (3) Per (gt_type, organ, stain) breakdown
    stratum_tables: list[pd.DataFrame] = []
    for (gt_type, organ, stain), sgrp in joined.groupby(["gt_type", "organ", "stain"], sort=True):
        n_t = int(sgrp["tile_id"].nunique())
        curves = build_curves_from_df(sgrp, strict_pairing=True)
        tbl = score_curves_table(f"{gt_type} | {organ} | {stain} (n={n_t})", curves, anchor=anchor)
        if tbl is not None:
            tbl.insert(1, "gt_type", str(gt_type))
            tbl.insert(2, "organ", str(organ))
            tbl.insert(3, "stain", str(stain))
            tbl.insert(4, "N_Tiles", n_t)
            stratum_tables.append(tbl)

    if stratum_tables:
        org_df = pd.concat(stratum_tables, ignore_index=True)
        org_df.to_csv(out_dir / "results_bdrate_by_organ_stain.csv", index=False)
        print("\n=== (3) R-D by Organ & Stain (Akima) ===")
        print(org_df.to_string(index=False))

    # (2) Lossless vs. Lossy breakdown
    gt_tables: list[pd.DataFrame] = []
    for gt_type, ggrp in joined.groupby("gt_type", sort=True):
        n_t = int(ggrp["tile_id"].nunique())
        curves = build_curves_from_df(ggrp, strict_pairing=True)
        tbl = score_curves_table(f"GT={gt_type} (n={n_t})", curves, anchor=anchor)
        if tbl is not None:
            tbl.insert(1, "gt_type", str(gt_type))
            tbl.insert(2, "N_Tiles", n_t)
            gt_tables.append(tbl)

    if gt_tables:
        gt_df = pd.concat(gt_tables, ignore_index=True)
        gt_df.to_csv(out_dir / "results_bdrate_lossless_vs_lossy.csv", index=False)
        print("\n=== (2) Lossless vs. Lossy R-D (Akima) ===")
        print(gt_df.to_string(index=False))

    # (1) Overall: Tile-Pooled + Stratum-Balanced Macro Mean
    tile_meta = joined[["tile_id", "gt_type"]].drop_duplicates()
    n_lossless = int((tile_meta["gt_type"] == "lossless").sum())
    n_lossy = int((tile_meta["gt_type"] == "lossy").sum())
    overall_curves = build_curves_from_df(joined, strict_pairing=True)
    pooled_df = score_curves_table(
        f"Overall (Tile-Pooled, n_lossless={n_lossless}, n_lossy={n_lossy})",
        overall_curves,
        anchor=anchor,
    )

    overall_parts: list[pd.DataFrame] = []
    if stratum_tables:
        org_all = pd.concat(stratum_tables, ignore_index=True)
        num_cols = [c for c in org_all.columns if c.startswith("BD-") or c in ("Min bpp", "Max bpp")]
        macro = org_all.groupby("Codec / Model", as_index=False)[num_cols].mean()
        macro[num_cols] = macro[num_cols].round(2)
        macro.insert(0, "Cohort", f"Overall (Stratum-Balanced Macro Mean, {len(stratum_tables)} strata)")
        macro = macro.sort_values("BD-Rate RGB-PSNR [%]")
        overall_parts.append(macro)
    if pooled_df is not None:
        overall_parts.append(pooled_df)

    if overall_parts:
        overall_df = pd.concat(overall_parts, ignore_index=True)
        overall_df.to_csv(out_dir / "results_bdrate_overall.csv", index=False)
        print("\n=== (1) Overall R-D: Stratum-Balanced Macro Mean & Tile-Pooled (Akima) ===")
        print(overall_df.to_string(index=False))


def evaluate_per_patient_summary(
    conv_csv: Path,
    ai_csv: Path,
    manifest_csv: Path,
    out_csv: Path,
    anchor: str = "JPEG 4:4:4",
) -> None:
    """Computes Akima BD-Rate and BD-PSNR per patient on the paired tiles and reports Mean +/- Std."""
    mf = pd.read_csv(manifest_csv)
    if "track" in mf.columns:
        mf = mf[mf["track"] == "A"].copy()
    patients = sorted(mf["patient_id"].astype(str).unique())
    if len(patients) < 2:
        return

    per_pat_records: list[dict[str, object]] = []
    for pat in patients:
        pat_tiles = set(mf[mf["patient_id"].astype(str) == pat]["tile_id"].astype(str))
        curves = prepare_curves(conv_csv, ai_csv, allowed_tiles=pat_tiles, strict_all_tolerance=False, pair_with_ai_tiles=True)
        if anchor not in curves:
            continue
        ref = curves[anchor]
        r_ref, p_ref = ref["bpp"].to_numpy(), ref["psnr_db"].to_numpy()
        s_ref = -10.0 * np.log10(np.clip(1.0 - ref["ms_ssim"].to_numpy(), 1e-10, 1.0))

        for name, df_c in curves.items():
            if name == anchor:
                continue
            r_t, p_t = df_c["bpp"].to_numpy(), df_c["psnr_db"].to_numpy()
            s_t = -10.0 * np.log10(np.clip(1.0 - df_c["ms_ssim"].to_numpy(), 1e-10, 1.0))
            per_pat_records.append({
                "patient_id": pat,
                "Codec / Model": name,
                "bdr_psnr": bd_rate(r_ref, p_ref, r_t, p_t, method="akima"),
                "bdp": bd_psnr(r_ref, p_ref, r_t, p_t, method="akima"),
                "bdp_fixed": bd_psnr(r_ref, p_ref, r_t, p_t, fixed_range=(0.25, 0.90), strict_fixed_range=True, method="akima"),
                "bdr_ssim": bd_rate(r_ref, s_ref, r_t, s_t, method="akima"),
            })

    if not per_pat_records:
        return

    pat_df = pd.DataFrame(per_pat_records)
    summary_rows = []
    for name, grp in pat_df.groupby("Codec / Model"):
        n_p = int(grp["bdr_psnr"].notna().sum())
        n_fixed = int(grp["bdp_fixed"].notna().sum())
        summary_rows.append({
            "Codec / Model": name,
            "N_Patients": n_p,
            "BD-Rate RGB-PSNR Mean [%]": round(float(grp["bdr_psnr"].mean()), 2) if n_p > 0 else float("nan"),
            "BD-Rate RGB-PSNR Std [%]": round(float(grp["bdr_psnr"].std(ddof=1)), 2) if n_p > 1 else 0.0,
            "BD-PSNR Mean [dB]": round(float(grp["bdp"].mean()), 2) if n_p > 0 else float("nan"),
            "BD-PSNR Std [dB]": round(float(grp["bdp"].std(ddof=1)), 2) if n_p > 1 else 0.0,
            "N_Patients_Fixed": n_fixed,
            "BD-PSNR [0.25-0.90] Mean [dB]": round(float(grp["bdp_fixed"].mean()), 2) if n_fixed >= 5 else float("nan"),
            "BD-Rate MS-SSIM Mean [%]": round(float(grp["bdr_ssim"].mean()), 2) if n_p > 0 else float("nan"),
            "BD-Rate MS-SSIM Std [%]": round(float(grp["bdr_ssim"].std(ddof=1)), 2) if n_p > 1 else 0.0,
        })

    sum_df = pd.DataFrame(summary_rows).sort_values("BD-Rate RGB-PSNR Mean [%]")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(out_csv, index=False)
    print(f"\n=== Curated Track A - Patient-Level Summary (Mean +/- Std across {len(patients)} Patients, Akima) ===")
    print(sum_df.to_string(index=False))
    print(f"Saved to: {out_csv.resolve()}")


def _run_self_test() -> None:
    r_ref = np.array([0.25, 0.35, 0.50, 0.75, 1.00, 1.50])
    p_ref = np.array([22.0, 24.0, 26.0, 28.0, 29.5, 31.0])
    # Test codec needs half the bitrate for same PSNR (~ -50% BD-Rate)
    r_test = r_ref * 0.5
    p_test = p_ref.copy()
    bdr_ak = bd_rate(r_ref, p_ref, r_test, p_test, method="akima")
    bdr_pc = bd_rate_pchip(r_ref, p_ref, r_test, p_test)
    assert abs(bdr_ak - (-50.0)) < 1.5, f"Expected ~ -50% Akima BD-Rate, got {bdr_ak}"
    assert abs(bdr_ak - bdr_pc) < 1.5, f"Akima ({bdr_ak}) vs PCHIP ({bdr_pc}) diverged unexpectedly"

    # Guard tests: unequal point counts, non-monotonic dip, < 4 points, mini-overlap
    r_unequal = np.array([0.20, 0.40, 0.60, 0.80, 1.20])
    p_unequal = np.array([23.0, 26.0, 25.99, 28.5, 30.8])  # slight dip at 0.60
    val_unequal = bd_rate(r_ref, p_ref, r_unequal, p_unequal, method="akima")
    assert np.isfinite(val_unequal), "Expected finite Akima BD-Rate after monotonic sanitization"
    assert np.isnan(bd_rate(r_ref[:3], p_ref[:3], r_test, p_test)), "Must return NaN for < 4 points"
    assert np.isnan(bd_rate(r_ref, p_ref, r_test, p_test + 20.0)), "Must return NaN for zero overlap"
    print(f"[SELF-TEST OK] Akima={bdr_ak:.2f}%, PCHIP={bdr_pc:.2f}%, unequal+dip={val_unequal:.2f}%, guards passed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Akima/PCHIP BD-Rate & Unified Track A Manifest Evaluator.")
    p.add_argument("--self-test", action="store_true", help="Run internal Akima/guard sanity tests")
    p.add_argument("--csvs", type=Path, nargs="+", help="One or more codec result CSVs for unified evaluation")
    p.add_argument("--manifest", type=Path, help="Unified manifest CSV containing tile_id, gt_type, organ, stain")
    p.add_argument("--out-dir", type=Path, default=Path("results/unified_v1"), help="Output directory for unified tables")
    p.add_argument("--anchor", default="JPEG 4:4:4", help="Anchor codec name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        _run_self_test()
    elif args.csvs and args.manifest:
        evaluate_unified_benchmark(args.csvs, args.manifest, args.out_dir, anchor=args.anchor)
    else:
        rms_conv = Path("results/pilot/results_rms_conv.csv")
        rms_ai = Path("results/pilot/results_rms_ai.csv")
        if rms_conv.exists():
            evaluate_cohort(
                "IDC RMS (Lossless Ground-Truth)", rms_conv, rms_ai,
                Path("results/pilot/results_bdrate_rms.csv"), strict_all_tolerance=True,
            )

        tcga_conv = Path("results/pilot/results_tcga256_conv.csv")
        tcga_ai = Path("results/pilot/results_tcga256_ai.csv")
        if tcga_conv.exists():
            evaluate_cohort(
                "TCGA-LUAD (Scanner Q=70 Pre-Compressed)", tcga_conv, tcga_ai,
                Path("results/pilot/results_bdrate_tcga.csv"), strict_all_tolerance=True,
            )

        cur_conv = Path("results/curated_v1/results_rms_conv.csv")
        cur_ai = Path("results/curated_v1/results_rms_ai.csv")
        manifest = Path("results/curated_v1/manifest_track_a.csv")
        if cur_conv.exists():
            evaluate_cohort(
                "Curated Track A - Paired 300 Tiles (Conv + CompressAI)",
                cur_conv, cur_ai,
                Path("results/curated_v1/results_bdrate_track_a_all.csv"),
                include_fixed_interval=True,
                pair_with_ai_tiles=True,
            )
            evaluate_cohort(
                "Curated Track A - Full 6,000 Tiles (Conventional Only)",
                cur_conv, None,
                Path("results/curated_v1/results_bdrate_track_a_conv6k.csv"),
                include_fixed_interval=True,
                pair_with_ai_tiles=False,
            )
            if manifest.exists():
                mf_df = pd.read_csv(manifest)
                for stratum in ("S1_eosinophilic", "S2_basophilic"):
                    t_set = set(mf_df[mf_df["stain_stratum"] == stratum]["tile_id"].astype(str))
                    if t_set:
                        evaluate_cohort(
                            f"Curated Track A - {stratum} (Paired)",
                            cur_conv, cur_ai,
                            Path(f"results/curated_v1/results_bdrate_track_a_{stratum[:2].lower()}.csv"),
                            allowed_tiles=t_set,
                            include_fixed_interval=True,
                            pair_with_ai_tiles=True,
                        )
                evaluate_per_patient_summary(
                    cur_conv, cur_ai, manifest,
                    Path("results/curated_v1/results_bdrate_track_a_per_patient.csv"),
                )
