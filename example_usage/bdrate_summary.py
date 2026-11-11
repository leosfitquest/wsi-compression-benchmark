"""
bdrate_summary.py
-----------------
Computes Bjontegaard-Delta Rate (BD-Rate in %) and BD-PSNR (in dB)
relative to JPEG 4:4:4 for all evaluated conventional and neural codecs
on the lossless IDC RMS ground-truth cohort (and optionally TCGA).

Follows ITU-T VCEG-M33 (cubic polynomial integration in log-rate domain).
Depends on numpy and pandas only.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def bd_psnr(rate_ref: np.ndarray, psnr_ref: np.ndarray, rate_test: np.ndarray, psnr_test: np.ndarray) -> float:
    """Average PSNR difference (dB) over the overlapping log-rate interval."""
    lr_ref = np.log10(rate_ref)
    lr_test = np.log10(rate_test)

    min_lr = max(float(lr_ref.min()), float(lr_test.min()))
    max_lr = min(float(lr_ref.max()), float(lr_test.max()))
    if max_lr <= min_lr:
        return float("nan")

    deg_ref = min(3, len(lr_ref) - 1)
    deg_test = min(3, len(lr_test) - 1)
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
    p_ref = np.poly1d(np.polyfit(dist_ref, lr_ref, deg_ref))
    p_test = np.poly1d(np.polyfit(dist_test, lr_test, deg_test))

    int_ref = p_ref.integ()
    int_test = p_test.integ()
    avg_log_diff = ((int_test(max_d) - int_test(min_d)) - (int_ref(max_d) - int_ref(min_d))) / (max_d - min_d)
    return float((10.0**avg_log_diff - 1.0) * 100.0)


def prepare_curves(conv_csv: Path, ai_csv: Path) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}

    df_conv = pd.read_csv(conv_csv)
    conv_grouped = df_conv.groupby(["codec", "target_bpp"])[["bpp", "psnr_db", "ms_ssim"]].mean().reset_index()
    for codec, grp in conv_grouped.groupby("codec"):
        grp = grp.sort_values("bpp")
        grp = grp.loc[grp["bpp"].round(3).drop_duplicates(keep="last").index]
        curves[str(codec)] = grp[["bpp", "psnr_db", "ms_ssim"]].copy()

    if ai_csv.exists():
        df_ai = pd.read_csv(ai_csv)
        ai_grouped = df_ai.groupby(["model", "quality"])[["bpp_real", "psnr_db", "ms_ssim"]].mean().reset_index()
        for model, grp in ai_grouped.groupby("model"):
            grp = grp.sort_values("bpp_real").rename(columns={"bpp_real": "bpp"})
            curves[str(model)] = grp[["bpp", "psnr_db", "ms_ssim"]].copy()

    return curves


def evaluate_cohort(cohort_name: str, conv_csv: Path, ai_csv: Path, out_csv: Path, anchor: str = "JPEG 4:4:4") -> None:
    curves = prepare_curves(conv_csv, ai_csv)
    if anchor not in curves:
        raise ValueError(f"Anchor codec '{anchor}' not found in {conv_csv}")

    ref = curves[anchor]
    r_ref = ref["bpp"].to_numpy()
    p_ref = ref["psnr_db"].to_numpy()
    # Linearize MS-SSIM saturation near 1.0 via dB transform: -10 * log10(1 - MS-SSIM)
    s_ref = -10.0 * np.log10(np.clip(1.0 - ref["ms_ssim"].to_numpy(), 1e-10, 1.0))

    rows = []
    for name, df_c in curves.items():
        r_test = df_c["bpp"].to_numpy()
        p_test = df_c["psnr_db"].to_numpy()
        s_test = -10.0 * np.log10(np.clip(1.0 - df_c["ms_ssim"].to_numpy(), 1e-10, 1.0))

        if name == anchor:
            bdr_psnr, bdp, bdr_ssim = 0.0, 0.0, 0.0
        else:
            bdr_psnr = bd_rate(r_ref, p_ref, r_test, p_test)
            bdp = bd_psnr(r_ref, p_ref, r_test, p_test)
            bdr_ssim = bd_rate(r_ref, s_ref, r_test, s_test)

        rows.append({
            "Cohort": cohort_name,
            "Codec / Model": name,
            "BD-Rate PSNR [%]": round(bdr_psnr, 2),
            "BD-PSNR [dB]": round(bdp, 2),
            "BD-Rate MS-SSIM [%]": round(bdr_ssim, 2),
            "Min bpp": round(float(r_test.min()), 3),
            "Max bpp": round(float(r_test.max()), 3),
        })

    res_df = pd.DataFrame(rows).sort_values("BD-Rate PSNR [%]")
    res_df.to_csv(out_csv, index=False)

    print(f"\n=== {cohort_name} (Anchor: {anchor}) ===")
    print(res_df.to_string(index=False))
    print(f"Saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    rms_conv = Path("results_rms_conv.csv")
    rms_ai = Path("results_rms_ai.csv")
    if rms_conv.exists():
        evaluate_cohort("IDC RMS (Lossless Ground-Truth)", rms_conv, rms_ai, Path("results_bdrate_rms.csv"))

    tcga_conv = Path("results_tcga256_conv.csv")
    tcga_ai = Path("results_tcga256_ai.csv")
    if tcga_conv.exists():
        evaluate_cohort("TCGA-LUAD (Scanner Q=70 Pre-Compressed)", tcga_conv, tcga_ai, Path("results_bdrate_tcga.csv"))