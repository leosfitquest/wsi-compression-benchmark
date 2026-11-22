"""
bdrate_summary.py
-----------------
Computes Bjontegaard-Delta Rate (BD-Rate in %) and BD-PSNR (in dB)
relative to JPEG 4:4:4 using ITU-T VCEG-AI11 / JPEG AI CTC standard PCHIP
(Piecewise Cubic Hermite Interpolating Polynomial, scipy.interpolate.PchipInterpolator).

Features for Curated Test Set v1 (Track A):
- Paired Apples-to-Apples BD-Rate on the exact same 300 tiles when comparing Conventional vs. CompressAI
- Full 6,000-tile BD-Rate table for Conventional codecs (`results_bdrate_track_a_conv6k.csv`)
- Full-overlap BD-Rate (RGB-PSNR, Y-PSNR, MS-SSIM) and BD-PSNR
- Strict fixed-interval BD-PSNR on [0.25, 0.90] bpp (returns NaN if interval is not covered)
- Ruifrok stain stratification (S1_eosinophilic vs. S2_basophilic)
- Patient-level Mean +/- Std aggregation across patients (excluding NaNs from N_Patients)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def _prepare_monotone_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filters non-finite entries, sorts by x, and averages duplicate x-values for strict monotonicity."""
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


def bd_psnr(
    rate_ref: np.ndarray,
    psnr_ref: np.ndarray,
    rate_test: np.ndarray,
    psnr_test: np.ndarray,
    fixed_range: tuple[float, float] | None = None,
    strict_fixed_range: bool = True,
) -> float:
    """Average PSNR difference (dB) via PCHIP over the overlapping (or strictly covered fixed) log-rate interval."""
    lr_ref, p_ref = _prepare_monotone_xy(np.log10(np.asarray(rate_ref, dtype=np.float64)), psnr_ref)
    lr_test, p_test = _prepare_monotone_xy(np.log10(np.asarray(rate_test, dtype=np.float64)), psnr_test)
    if len(lr_ref) < 4 or len(lr_test) < 4:
        return float("nan")

    min_lr = max(float(lr_ref[0]), float(lr_test[0]))
    max_lr = min(float(lr_ref[-1]), float(lr_test[-1]))
    if fixed_range is not None:
        lo = float(np.log10(fixed_range[0]))
        hi = float(np.log10(fixed_range[1]))
        # 0.015 log10 (~3.5% rate tolerance) permits tiny boundary jitter (e.g. 0.254 vs 0.250),
        # while rejecting severe gaps (e.g. WebP starting at 0.499 bpp -> NaN instead of silent clipping).
        if strict_fixed_range and (min_lr > lo + 0.015 or max_lr < hi - 0.015):
            return float("nan")
        min_lr = max(min_lr, lo)
        max_lr = min(max_lr, hi)
    if max_lr <= min_lr:
        return float("nan")

    f_ref = PchipInterpolator(lr_ref, p_ref)
    f_test = PchipInterpolator(lr_test, p_test)
    avg_diff = (float(f_test.integrate(min_lr, max_lr)) - float(f_ref.integrate(min_lr, max_lr))) / (max_lr - min_lr)
    return float(avg_diff)


def bd_rate(rate_ref: np.ndarray, dist_ref: np.ndarray, rate_test: np.ndarray, dist_test: np.ndarray) -> float:
    """Average bitrate savings (%) via PCHIP over the overlapping distortion (PSNR or MS-SSIM-dB) interval."""
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


def prepare_curves(
    conv_csv: Path,
    ai_csv: Path | None,
    allowed_tiles: set[str] | None = None,
    strict_all_tolerance: bool = False,
    pair_with_ai_tiles: bool = True,
) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}

    df_ai: pd.DataFrame | None = None
    if ai_csv is not None and ai_csv.exists():
        df_ai = pd.read_csv(ai_csv)
        if pair_with_ai_tiles and not df_ai.empty:
            ai_tiles = set(df_ai["tile_id"].astype(str).unique())
            allowed_tiles = ai_tiles if allowed_tiles is None else (allowed_tiles & ai_tiles)

    df_conv = pd.read_csv(conv_csv)
    if allowed_tiles is not None:
        df_conv = df_conv[df_conv["tile_id"].astype(str).isin(allowed_tiles)].copy()
    if df_conv.empty:
        return curves

    metric_cols = ["bpp", "psnr_db", "ms_ssim"] + (["psnr_y_db"] if "psnr_y_db" in df_conv.columns else [])
    if strict_all_tolerance:
        filtered = df_conv.groupby(["codec", "target_bpp"]).filter(lambda g: g["within_tolerance"].all())
    else:
        filtered = df_conv.groupby(["codec", "target_bpp"]).filter(
            lambda g: abs(g["bpp"].mean() - g["target_bpp"].iloc[0]) / g["target_bpp"].iloc[0] <= 0.15
        )

    conv_grouped = filtered.groupby(["codec", "target_bpp"])[metric_cols].mean().reset_index()
    for codec, grp in conv_grouped.groupby("codec"):
        grp = grp.sort_values("bpp")
        grp = grp.loc[grp["bpp"].round(3).drop_duplicates(keep="last").index]
        curves[str(codec)] = grp[metric_cols].copy()

    if df_ai is not None and not df_ai.empty:
        if allowed_tiles is not None:
            df_ai = df_ai[df_ai["tile_id"].astype(str).isin(allowed_tiles)].copy()
        if not df_ai.empty:
            if "bpp_real" in df_ai.columns and df_ai["bpp_real"].notna().any():
                bpp_col = "bpp_real"
            elif "bpp_est" in df_ai.columns and df_ai["bpp_est"].notna().any():
                bpp_col = "bpp_est"
            else:
                bpp_col = "bpp"
            ai_cols = [bpp_col, "psnr_db", "ms_ssim"] + (["psnr_y_db"] if "psnr_y_db" in df_ai.columns else [])
            ai_grouped = df_ai.dropna(subset=[bpp_col]).groupby(["model", "quality"])[ai_cols].mean().reset_index()
            for model, grp in ai_grouped.groupby("model"):
                grp = grp.sort_values(bpp_col).rename(columns={bpp_col: "bpp"})
                curves[str(model)] = grp[metric_cols if "psnr_y_db" in grp.columns else ["bpp", "psnr_db", "ms_ssim"]].copy()

    return curves


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
    if anchor not in curves:
        return None

    ref = curves[anchor]
    r_ref = ref["bpp"].to_numpy()
    p_ref = ref["psnr_db"].to_numpy()
    has_y = "psnr_y_db" in ref.columns
    py_ref = ref["psnr_y_db"].to_numpy() if has_y else None
    s_ref = -10.0 * np.log10(np.clip(1.0 - ref["ms_ssim"].to_numpy(), 1e-10, 1.0))

    rows = []
    for name, df_c in curves.items():
        r_test = df_c["bpp"].to_numpy()
        p_test = df_c["psnr_db"].to_numpy()
        s_test = -10.0 * np.log10(np.clip(1.0 - df_c["ms_ssim"].to_numpy(), 1e-10, 1.0))

        if name == anchor:
            bdr_psnr, bdp, bdp_fixed, bdr_ssim, bdr_y = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            bdr_psnr = bd_rate(r_ref, p_ref, r_test, p_test)
            bdp = bd_psnr(r_ref, p_ref, r_test, p_test)
            bdp_fixed = bd_psnr(r_ref, p_ref, r_test, p_test, fixed_range=(0.25, 0.90), strict_fixed_range=True)
            bdr_ssim = bd_rate(r_ref, s_ref, r_test, s_test)
            if has_y and py_ref is not None and "psnr_y_db" in df_c.columns:
                bdr_y = bd_rate(r_ref, py_ref, r_test, df_c["psnr_y_db"].to_numpy())
            else:
                bdr_y = float("nan")

        row_dict: dict[str, object] = {
            "Cohort": cohort_name,
            "Codec / Model": name,
            "BD-Rate PSNR [%]": round(bdr_psnr, 2) if np.isfinite(bdr_psnr) else float("nan"),
            "BD-PSNR [dB]": round(bdp, 2) if np.isfinite(bdp) else float("nan"),
        }
        if include_fixed_interval:
            row_dict["BD-PSNR [0.25-0.90] [dB]"] = round(bdp_fixed, 2) if np.isfinite(bdp_fixed) else float("nan")
        if has_y:
            row_dict["BD-Rate Y-PSNR [%]"] = round(bdr_y, 2) if np.isfinite(bdr_y) else float("nan")
        row_dict["BD-Rate MS-SSIM [%]"] = round(bdr_ssim, 2) if np.isfinite(bdr_ssim) else float("nan")
        row_dict["Min bpp"] = round(float(r_test.min()), 3)
        row_dict["Max bpp"] = round(float(r_test.max()), 3)
        rows.append(row_dict)

    res_df = pd.DataFrame(rows).sort_values("BD-Rate PSNR [%]")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)

    print(f"\n=== {cohort_name} (Anchor: {anchor}, Method: PCHIP) ===")
    print(res_df.to_string(index=False))
    print(f"Saved to: {out_csv.resolve()}")
    return res_df


def evaluate_per_patient_summary(
    conv_csv: Path,
    ai_csv: Path,
    manifest_csv: Path,
    out_csv: Path,
    anchor: str = "JPEG 4:4:4",
) -> None:
    """Computes BD-Rate and BD-PSNR per patient on the paired patient tiles and reports Mean +/- Std."""
    mf = pd.read_csv(manifest_csv)
    mf = mf[mf["track"] == "A"].copy()
    patients = sorted(mf["patient_id"].astype(str).unique())
    if len(patients) < 2:
        return

    per_pat_records: list[dict[str, object]] = []
    for pat in patients:
        pat_tiles = set(mf[mf["patient_id"].astype(str) == pat]["tile_id"].astype(str))
        curves = prepare_curves(
            conv_csv, ai_csv, allowed_tiles=pat_tiles, strict_all_tolerance=False, pair_with_ai_tiles=True
        )
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
                "bdr_psnr": bd_rate(r_ref, p_ref, r_t, p_t),
                "bdp": bd_psnr(r_ref, p_ref, r_t, p_t),
                "bdp_fixed": bd_psnr(r_ref, p_ref, r_t, p_t, fixed_range=(0.25, 0.90), strict_fixed_range=True),
                "bdr_ssim": bd_rate(r_ref, s_ref, r_t, s_t),
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
            "BD-Rate PSNR Mean [%]": round(float(grp["bdr_psnr"].mean()), 2) if n_p > 0 else float("nan"),
            "BD-Rate PSNR Std [%]": round(float(grp["bdr_psnr"].std(ddof=1)), 2) if n_p > 1 else 0.0,
            "BD-PSNR Mean [dB]": round(float(grp["bdp"].mean()), 2) if n_p > 0 else float("nan"),
            "BD-PSNR Std [dB]": round(float(grp["bdp"].std(ddof=1)), 2) if n_p > 1 else 0.0,
            "N_Patients_Fixed": n_fixed,
            "BD-PSNR [0.25-0.90] Mean [dB]": round(float(grp["bdp_fixed"].mean()), 2) if n_fixed >= 5 else float("nan"),
            "BD-Rate MS-SSIM Mean [%]": round(float(grp["bdr_ssim"].mean()), 2) if n_p > 0 else float("nan"),
            "BD-Rate MS-SSIM Std [%]": round(float(grp["bdr_ssim"].std(ddof=1)), 2) if n_p > 1 else 0.0,
        })

    sum_df = pd.DataFrame(summary_rows).sort_values("BD-Rate PSNR Mean [%]")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(out_csv, index=False)
    print(f"\n=== Curated Track A - Patient-Level Summary (Mean +/- Std across {len(patients)} Patients, PCHIP) ===")
    print(sum_df.to_string(index=False))
    print(f"Saved to: {out_csv.resolve()}")


if __name__ == "__main__":
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