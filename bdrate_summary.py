"""
bdrate_summary.py
-----------------
Computes Bjontegaard-Delta Rate (BD-Rate in %) and BD-PSNR (in dB)
relative to JPEG 4:4:4 for all evaluated conventional and neural codecs.

Features for Curated Test Set v1 (Track A):
  - Full-overlap BD-Rate (RGB-PSNR, Y-PSNR, MS-SSIM) and BD-PSNR
  - Fixed-interval BD-PSNR on [0.25, 0.90] bpp (eliminates rate-range bias)
  - Ruifrok stain stratification (S1_eosinophilic vs. S2_basophilic)
  - Patient-level Mean +/- Std aggregation across patients
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def bd_psnr(
    rate_ref: np.ndarray,
    psnr_ref: np.ndarray,
    rate_test: np.ndarray,
    psnr_test: np.ndarray,
    fixed_range: tuple[float, float] | None = None,
) -> float:
    """Average PSNR difference (dB) over the overlapping (or fixed) log-rate interval."""
    lr_ref = np.log10(rate_ref)
    lr_test = np.log10(rate_test)

    min_lr = max(float(lr_ref.min()), float(lr_test.min()))
    max_lr = min(float(lr_ref.max()), float(lr_test.max()))
    if fixed_range is not None:
        min_lr = max(min_lr, float(np.log10(fixed_range[0])))
        max_lr = min(max_lr, float(np.log10(fixed_range[1])))
    if max_lr <= min_lr:
        return float("nan")

    deg_ref = min(3, len(lr_ref) - 1)
    deg_test = min(3, len(lr_test) - 1)
    if deg_ref < 1 or deg_test < 1:
        return float("nan")
    p_ref = np.poly1d(np.polyfit(lr_ref, psnr_ref, deg_ref))
    p_test = np.poly1d(np.polyfit(lr_test, psnr_test, deg_test))

    int_ref = p_ref.integ()
    int_test = p_test.integ()
    avg_diff = (int_test(max_lr) - int_test(min_lr)) - (int_ref(max_lr) - int_ref(min_lr))
    return float(avg_diff / (max_lr - min_lr))


def bd_rate(rate_ref: np.ndarray, dist_ref: np.ndarray, rate_test: np.ndarray, dist_test: np.ndarray) -> float:
    """Average bitrate savings (%) over the overlapping distortion (PSNR or MS-SSIM) interval."""
    lr_ref = np.log10(rate_ref)
    lr_test = np.log10(rate_test)

    min_d = max(float(dist_ref.min()), float(dist_test.min()))
    max_d = min(float(dist_ref.max()), float(dist_test.max()))
    if max_d <= min_d:
        return float("nan")

    deg_ref = min(3, len(dist_ref) - 1)
    deg_test = min(3, len(dist_test) - 1)
    if deg_ref < 1 or deg_test < 1:
        return float("nan")
    p_ref = np.poly1d(np.polyfit(dist_ref, lr_ref, deg_ref))
    p_test = np.poly1d(np.polyfit(dist_test, lr_test, deg_test))

    int_ref = p_ref.integ()
    int_test = p_test.integ()
    avg_log_diff = ((int_test(max_d) - int_test(min_d)) - (int_ref(max_d) - int_ref(min_d))) / (max_d - min_d)
    return float((10.0**avg_log_diff - 1.0) * 100.0)


def prepare_curves(
    conv_csv: Path,
    ai_csv: Path,
    allowed_tiles: set[str] | None = None,
    strict_all_tolerance: bool = False,
) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}

    df_conv = pd.read_csv(conv_csv)
    if allowed_tiles is not None:
        df_conv = df_conv[df_conv["tile_id"].isin(allowed_tiles)].copy()
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

    if ai_csv.exists():
        df_ai = pd.read_csv(ai_csv)
        if allowed_tiles is not None:
            df_ai = df_ai[df_ai["tile_id"].isin(allowed_tiles)].copy()
        if not df_ai.empty:
            ai_cols = ["bpp_real", "psnr_db", "ms_ssim"] + (["psnr_y_db"] if "psnr_y_db" in df_ai.columns else [])
            ai_grouped = df_ai.groupby(["model", "quality"])[ai_cols].mean().reset_index()
            for model, grp in ai_grouped.groupby("model"):
                grp = grp.sort_values("bpp_real").rename(columns={"bpp_real": "bpp"})
                curves[str(model)] = grp[metric_cols if "psnr_y_db" in grp.columns else ["bpp", "psnr_db", "ms_ssim"]].copy()

    return curves


def evaluate_cohort(
    cohort_name: str,
    conv_csv: Path,
    ai_csv: Path,
    out_csv: Path,
    anchor: str = "JPEG 4:4:4",
    allowed_tiles: set[str] | None = None,
    strict_all_tolerance: bool = False,
    include_fixed_interval: bool = False,
) -> pd.DataFrame | None:
    curves = prepare_curves(
        conv_csv, ai_csv, allowed_tiles=allowed_tiles, strict_all_tolerance=strict_all_tolerance
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
            bdp_fixed = bd_psnr(r_ref, p_ref, r_test, p_test, fixed_range=(0.25, 0.90))
            bdr_ssim = bd_rate(r_ref, s_ref, r_test, s_test)
            if has_y and py_ref is not None and "psnr_y_db" in df_c.columns:
                bdr_y = bd_rate(r_ref, py_ref, r_test, df_c["psnr_y_db"].to_numpy())
            else:
                bdr_y = float("nan")

        row_dict: dict[str, object] = {
            "Cohort": cohort_name,
            "Codec / Model": name,
            "BD-Rate PSNR [%]": round(bdr_psnr, 2),
            "BD-PSNR [dB]": round(bdp, 2),
        }
        if include_fixed_interval:
            row_dict["BD-PSNR [0.25-0.90] [dB]"] = round(bdp_fixed, 2)
        if has_y:
            row_dict["BD-Rate Y-PSNR [%]"] = round(bdr_y, 2)
        row_dict["BD-Rate MS-SSIM [%]"] = round(bdr_ssim, 2)
        row_dict["Min bpp"] = round(float(r_test.min()), 3)
        row_dict["Max bpp"] = round(float(r_test.max()), 3)
        rows.append(row_dict)

    res_df = pd.DataFrame(rows).sort_values("BD-Rate PSNR [%]")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)

    print(f"\n=== {cohort_name} (Anchor: {anchor}) ===")
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
    """Computes BD-Rate and BD-PSNR per patient and reports Mean +/- Std across patients."""
    mf = pd.read_csv(manifest_csv)
    mf = mf[mf["track"] == "A"].copy()
    patients = sorted(mf["patient_id"].astype(str).unique())
    if len(patients) < 2:
        return

    per_pat_records: list[dict[str, object]] = []
    for pat in patients:
        pat_tiles = set(mf[mf["patient_id"].astype(str) == pat]["tile_id"].astype(str))
        curves = prepare_curves(conv_csv, ai_csv, allowed_tiles=pat_tiles, strict_all_tolerance=False)
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
                "bdp_fixed": bd_psnr(r_ref, p_ref, r_t, p_t, fixed_range=(0.25, 0.90)),
                "bdr_ssim": bd_rate(r_ref, s_ref, r_t, s_t),
            })

    if not per_pat_records:
        return

    pat_df = pd.DataFrame(per_pat_records)
    summary_rows = []
    for name, grp in pat_df.groupby("Codec / Model"):
        n_p = len(grp)
        summary_rows.append({
            "Codec / Model": name,
            "N_Patients": n_p,
            "BD-Rate PSNR Mean [%]": round(float(grp["bdr_psnr"].mean()), 2),
            "BD-Rate PSNR Std [%]": round(float(grp["bdr_psnr"].std(ddof=1)), 2) if n_p > 1 else 0.0,
            "BD-PSNR Mean [dB]": round(float(grp["bdp"].mean()), 2),
            "BD-PSNR Std [dB]": round(float(grp["bdp"].std(ddof=1)), 2) if n_p > 1 else 0.0,
            "BD-PSNR [0.25-0.90] Mean [dB]": round(float(grp["bdp_fixed"].mean()), 2),
            "BD-Rate MS-SSIM Mean [%]": round(float(grp["bdr_ssim"].mean()), 2),
            "BD-Rate MS-SSIM Std [%]": round(float(grp["bdr_ssim"].std(ddof=1)), 2) if n_p > 1 else 0.0,
        })

    sum_df = pd.DataFrame(summary_rows).sort_values("BD-Rate PSNR Mean [%]")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(out_csv, index=False)
    print(f"\n=== Curated Track A - Patient-Level Summary (Mean +/- Std across {len(patients)} Patients) ===")
    print(sum_df.to_string(index=False))
    print(f"Saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    # 1. Pilot Benchmark Evaluation (strict_all_tolerance=True keeps exact v1.0-pilot numbers)
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

    # 2. Curated Test Set v1 Evaluation (Track A + S1/S2 Stratification + Patient-Level Mean/Std)
    cur_conv = Path("results/curated_v1/results_rms_conv.csv")
    cur_ai = Path("results/curated_v1/results_rms_ai.csv")
    manifest = Path("results/curated_v1/manifest_track_a.csv")
    if cur_conv.exists():
        evaluate_cohort(
            "Curated Track A - All Tiles",
            cur_conv, cur_ai,
            Path("results/curated_v1/results_bdrate_track_a_all.csv"),
            include_fixed_interval=True,
        )
        if manifest.exists():
            mf_df = pd.read_csv(manifest)
            for stratum in ("S1_eosinophilic", "S2_basophilic"):
                t_set = set(mf_df[mf_df["stain_stratum"] == stratum]["tile_id"].astype(str))
                if t_set:
                    evaluate_cohort(
                        f"Curated Track A - {stratum}",
                        cur_conv, cur_ai,
                        Path(f"results/curated_v1/results_bdrate_track_a_{stratum[:2].lower()}.csv"),
                        allowed_tiles=t_set,
                        include_fixed_interval=True,
                    )
            evaluate_per_patient_summary(
                cur_conv, cur_ai, manifest,
                Path("results/curated_v1/results_bdrate_track_a_per_patient.csv"),
            )