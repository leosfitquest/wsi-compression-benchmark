"""
prepare_track_a_rms.py
----------------------
Baut Track A des Curated Test Sets (v1) gemaess Sheet 3 auf:
  - Local CPU Pilot: 30 Patienten x 200 Level-0-Tiles (256x256) = 6.000 Tiles + 30 x 2048x2048 ROIs
  - VUB GPU Full Run: 100 Patienten x 1.000 Level-0-Tiles (256x256) = 100.000 Tiles
  - Strikter pydicom VOLUME-Audit (TransferSyntax .90 oder .2.1, LossyImageCompression == '00')
  - 10x Speedup durch Vorfilterung leerer Glasflaechen auf der niedrigsten Pyramidenstufe
  - Echte Ruifrok & Johnston (2001) H&E Farb-Dekonvolution fuer 50/50 Balancierung
    zwischen S1 (Eosinophilic / Pink) und S2 (Basophilic / Blue)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import openslide
import pydicom
from idc_index import index
from PIL import Image

LOSSLESS_TS = {
    "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless (.90)
    "1.2.840.10008.1.2.1",     # Explicit VR Little Endian Uncompressed (.2.1)
}

# Ruifrok & Johnston (2001) Standard-OD-Vektoren fuer H&E (normiert)
_H_VEC = np.array([0.650, 0.704, 0.286], dtype=np.float64)
_E_VEC = np.array([0.072, 0.990, 0.105], dtype=np.float64)
_H_VEC /= np.linalg.norm(_H_VEC)
_E_VEC /= np.linalg.norm(_E_VEC)
_R_VEC = np.cross(_H_VEC, _E_VEC)
_R_VEC /= np.linalg.norm(_R_VEC)
_STAIN_MAT = np.stack([_H_VEC, _E_VEC, _R_VEC], axis=0)
_INV_STAIN_MAT = np.linalg.inv(_STAIN_MAT)

MANIFEST_FIELDS = [
    "tile_id",
    "track",
    "patient_id",
    "series_uid",
    "sop_uid_short",
    "fold",
    "stain_stratum",
    "h_ratio",
    "tissue_fraction",
    "mpp_um",
    "tile_x",
    "tile_y",
    "tile_size",
    "transfer_syntax",
    "is_fast_subset",
]


def ruifrok_he_ratio(tile_rgb: np.ndarray, tissue_mask: np.ndarray) -> float:
    """Berechnet den relativen Haematoxylin-Anteil H / (H + E) auf den Gewebepixeln nach Ruifrok (2001)."""
    fg_pixels = tile_rgb[tissue_mask].astype(np.float64)
    if fg_pixels.size == 0:
        return 0.5
    od = -np.log10(np.clip(fg_pixels, 1.0, 255.0) / 255.0)
    stains = od @ _INV_STAIN_MAT
    h_mean = float(np.mean(np.maximum(stains[:, 0], 0.0)))
    e_mean = float(np.mean(np.maximum(stains[:, 1], 0.0)))
    return h_mean / (h_mean + e_mean + 1e-8)


def get_verified_volumes(dcm_root: Path) -> list[tuple[int, Path, str, str]]:
    """Findet alle verifizierten verlustfreien Level-0 VOLUME-Instanzen (>= 5000 px) in dcm_root."""
    vols: list[tuple[int, Path, str, str]] = []
    for p in dcm_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
        except Exception:
            continue
        img_type = [str(x).upper() for x in getattr(ds, "ImageType", [])]
        if "VOLUME" not in img_type:
            continue
        cols = int(getattr(ds, "TotalPixelMatrixColumns", 0))
        rows = int(getattr(ds, "TotalPixelMatrixRows", 0))
        ts = str(ds.file_meta.TransferSyntaxUID).strip()
        lossy = str(ds.get("LossyImageCompression", "")).strip()
        if ts not in LOSSLESS_TS or lossy != "00":
            continue
        if cols < 5000 or rows < 5000:
            continue
        im_width_mm = float(getattr(ds, "ImagedVolumeWidth", 0.0) or 0.0)
        mpp = round((im_width_mm * 1000.0) / cols, 4) if im_width_mm > 0 and cols > 0 else 0.2526
        vols.append((cols * rows, p, ts, f"{mpp:.4f}"))

    vols.sort(key=lambda t: t[0], reverse=True)
    return vols


def _prefilter_coords_via_pyramid(
    slide: openslide.OpenSlide,
    dcm_dir: Path,
    coords: list[tuple[int, int]],
    w_l0: int,
    h_l0: int,
    min_tissue_fraction: float,
) -> list[tuple[int, int]]:
    """Filtert leere Glas-Koordinaten in <0.1s ueber die kleinste Pyramidenstufe vor."""
    thumb_arr = None

    # 1. Versuch: Niedrigste Pyramidenstufe direkt aus OpenSlide (falls vorhanden und <= 4096 px)
    for lvl in range(slide.level_count - 1, 0, -1):
        wl, hl = slide.level_dimensions[lvl]
        if wl <= 4096 and hl <= 4096:
            thumb_rgb = slide.read_region((0, 0), level=lvl, size=(wl, hl)).convert("RGB")
            thumb_arr = np.asarray(thumb_rgb, dtype=np.uint8)
            break

    # 2. Fallback: Kleinste VOLUME- oder OVERVIEW-DICOM-Datei im selben Serienordner nutzen
    if thumb_arr is None:
        small_dcms = []
        for p in dcm_dir.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True)
                c = int(getattr(ds, "TotalPixelMatrixColumns", 0))
                r = int(getattr(ds, "TotalPixelMatrixRows", 0))
                if 256 <= c <= 4096 and 256 <= r <= 4096:
                    small_dcms.append((c * r, p))
            except Exception:
                continue
        if small_dcms:
            small_dcms.sort(key=lambda t: t[0])
            try:
                with openslide.OpenSlide(str(small_dcms[0][1])) as s_low:
                    wl, hl = s_low.dimensions
                    thumb_rgb = s_low.read_region((0, 0), level=0, size=(wl, hl)).convert("RGB")
                    thumb_arr = np.asarray(thumb_rgb, dtype=np.uint8)
            except Exception:
                thumb_arr = None

    if thumb_arr is None:
        return coords

    hl, wl = thumb_arr.shape[:2]
    scale_x = wl / float(w_l0)
    scale_y = hl / float(h_l0)
    tw = max(1, int(round(256 * scale_x)))
    th = max(1, int(round(256 * scale_y)))

    pixel_means = thumb_arr.mean(axis=-1)
    tissue_map = (pixel_means >= 30.0) & (pixel_means <= 220.0)

    # Etwas grosszuegigerer Schwellwert auf dem Thumbnail (0.50 statt 0.70), Level-0 prueft danach exakt >= 0.70
    low_thresh = max(0.45, min_tissue_fraction - 0.20)
    kept: list[tuple[int, int]] = []
    for x, y in coords:
        lx = min(int(x * scale_x), wl - tw)
        ly = min(int(y * scale_y), hl - th)
        patch_mask = tissue_map[ly : ly + th, lx : lx + tw]
        if patch_mask.size > 0 and float(np.mean(patch_mask)) >= low_thresh:
            kept.append((x, y))

    print(f"  [Fast-Mask] Pyramiden-Vorfilter ({wl}x{hl} px): {len(kept)}/{len(coords)} Gewebe-Positionen behalten.")
    return kept if len(kept) >= 50 else coords


def sample_patient(
    patient_id: str,
    series_uid: str,
    fold: int,
    volumes: list[tuple[int, Path, str, str]],
    tiles_dir: Path,
    roi_dir: Path,
    tiles_per_patient: int,
    fast_per_patient: int,
    min_tissue_fraction: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    candidates: list[tuple[float, float, int, int, str, str, str, np.ndarray]] = []
    roi_saved = False
    roi_rows: list[dict[str, object]] = []

    target_pool = int(max(tiles_per_patient * 1.2, tiles_per_patient + 10))

    for _, dcm_path, ts_uid, mpp_str in volumes:
        if len(candidates) >= target_pool:
            break
        sop_short = dcm_path.stem[:12]
        with openslide.OpenSlide(str(dcm_path)) as slide:
            w_l0, h_l0 = slide.dimensions
            print(f"  [OpenSlide] {dcm_path.name} ({w_l0}x{h_l0} px, TS={ts_uid}, MPP={mpp_str})")

            xs = range(0, max(1, w_l0 - 256), 256)
            ys = range(0, max(1, h_l0 - 256), 256)
            coords = [(x, y) for y in ys for x in xs]
            rng.shuffle(coords)

            # Schneller Pyramiden-Vorfilter verwirft weisses Glas in <0.1 Sekunden
            coords = _prefilter_coords_via_pyramid(
                slide, dcm_path.parent, coords, w_l0, h_l0, min_tissue_fraction
            )

            for x, y in coords:
                if len(candidates) >= target_pool:
                    break
                tile_rgb = slide.read_region((x, y), level=0, size=(256, 256)).convert("RGB")
                tile_arr = np.asarray(tile_rgb, dtype=np.uint8)

                pixel_means = tile_arr.mean(axis=-1)
                tissue_mask = (pixel_means >= 30.0) & (pixel_means <= 220.0)
                t_frac = float(np.mean(tissue_mask))
                if t_frac < min_tissue_fraction:
                    continue

                h_ratio = ruifrok_he_ratio(tile_arr, tissue_mask)
                candidates.append((h_ratio, t_frac, x, y, sop_short, ts_uid, mpp_str, tile_arr))

                if len(candidates) == 1 or len(candidates) % 40 == 0 or len(candidates) == target_pool:
                    print(
                        f"    [{len(candidates)}/{target_pool}] Gewebe-Kandidat @ ({x}, {y}) "
                        f"| Tissue: {100.0 * t_frac:.0f}% | Ruifrok H/(H+E): {h_ratio:.3f}"
                    )

            if not roi_saved and candidates and w_l0 >= 2048 and h_l0 >= 2048:
                _, _, cx, cy, _, _, _, _ = candidates[0]
                rx = min(max(0, cx - 896), w_l0 - 2048)
                ry = min(max(0, cy - 896), h_l0 - 2048)
                roi_rgb = slide.read_region((rx, ry), level=0, size=(2048, 2048)).convert("RGB")
                roi_id = f"{patient_id}_{sop_short}_x{rx}_y{ry}_s2048"
                roi_rgb.save(roi_dir / f"{roi_id}.png", format="PNG")
                roi_rows.append({
                    "tile_id": roi_id,
                    "track": "A_ROI",
                    "patient_id": patient_id,
                    "series_uid": series_uid,
                    "sop_uid_short": sop_short,
                    "fold": fold,
                    "stain_stratum": "H&E_ROI_2048",
                    "h_ratio": 0.5,
                    "tissue_fraction": 1.0,
                    "mpp_um": mpp_str,
                    "tile_x": rx,
                    "tile_y": ry,
                    "tile_size": 2048,
                    "transfer_syntax": ts_uid,
                    "is_fast_subset": True,
                })
                roi_saved = True

    if not candidates:
        return []

    selected = candidates[:tiles_per_patient]
    selected.sort(key=lambda item: item[0])
    half = len(selected) // 2
    fast_s1 = fast_per_patient // 2
    fast_s2 = fast_per_patient - fast_s1

    out_rows: list[dict[str, object]] = []
    for idx, (h_ratio, t_frac, x, y, sop_short, ts_uid, mpp_str, tile_arr) in enumerate(selected):
        if idx < half:
            stratum = "S1_eosinophilic"
            is_fast = idx < fast_s1
        else:
            stratum = "S2_basophilic"
            is_fast = (idx - half) < fast_s2

        tile_id = f"{patient_id}_{sop_short}_x{x}_y{y}_s256"
        Image.fromarray(tile_arr).save(tiles_dir / f"{tile_id}.png", format="PNG")
        out_rows.append({
            "tile_id": tile_id,
            "track": "A",
            "patient_id": patient_id,
            "series_uid": series_uid,
            "sop_uid_short": sop_short,
            "fold": fold,
            "stain_stratum": stratum,
            "h_ratio": round(h_ratio, 4),
            "tissue_fraction": round(t_frac, 3),
            "mpp_um": mpp_str,
            "tile_x": x,
            "tile_y": y,
            "tile_size": 256,
            "transfer_syntax": ts_uid,
            "is_fast_subset": is_fast,
        })

    s1_n = sum(1 for r in out_rows if r["stain_stratum"] == "S1_eosinophilic")
    s2_n = sum(1 for r in out_rows if r["stain_stratum"] == "S2_basophilic")
    print(f"  -> Gespeichert: {len(out_rows)} Tiles (S1_pink={s1_n}, S2_blue={s2_n}) + {len(roi_rows)} ROI fuer {patient_id}")
    return out_rows + roi_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream & extract Track A (IDC RMS lossless) curated tiles.")
    p.add_argument("--num-patients", type=int, default=30, help="30 for Local CPU Pilot, 100 for VUB GPU Full Run")
    p.add_argument("--tiles-per-patient", type=int, default=200, help="200 for CPU Pilot (6,000 total), 1000 for GPU Full")
    p.add_argument("--fast-per-patient", type=int, default=10, help="Tiles per patient flagged for fast CPU neural subset")
    p.add_argument("--min-tissue", type=float, default=0.70, help="Minimum tissue pixel fraction per tile (default 0.70)")
    p.add_argument("--tiles-dir", type=Path, default=Path("tiles_curated_v1/track_a_rms"))
    p.add_argument("--roi-dir", type=Path, default=Path("tiles_curated_v1/track_a_roi_2048"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_a.csv"))
    p.add_argument("--local-rms-dir", type=Path, default=Path("idc_rms_lossless"), help="Existing local DICOM folder")
    p.add_argument("--tmp-dcm-dir", type=Path, default=Path("idc_tmp_download"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.tiles_dir.mkdir(parents=True, exist_ok=True)
    args.roi_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    done_patients: set[str] = set()
    if args.manifest.exists() and args.manifest.stat().st_size > 0:
        with args.manifest.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_patients.add(row["patient_id"])
        print(f"[Resume] {len(done_patients)} Patienten bereits in {args.manifest.name} -> ueberspringe diese.")

    write_header = not args.manifest.exists() or args.manifest.stat().st_size == 0
    mf = args.manifest.open("a" if not write_header else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
    if write_header:
        writer.writeheader()
        mf.flush()

    try:
        completed = len(done_patients)

        # 1. Lokalen Patienten in idc_rms_lossless zuerst verarbeiten
        if args.local_rms_dir.exists() and completed < args.num_patients:
            local_vols = get_verified_volumes(args.local_rms_dir)
            if local_vols:
                ds0 = pydicom.dcmread(local_vols[0][1], stop_before_pixels=True)
                pat0 = str(ds0.get("PatientID", "RMS2402")).strip()
                ser0 = str(ds0.get("SeriesInstanceUID", "local_series")).strip()
                if pat0 not in done_patients:
                    print(f"\n[{completed + 1}/{args.num_patients}] Nutze lokalen Patienten {pat0} aus '{args.local_rms_dir}'...")
                    rows = sample_patient(
                        pat0, ser0, (completed % 5) + 1, local_vols,
                        args.tiles_dir, args.roi_dir, args.tiles_per_patient,
                        args.fast_per_patient, args.min_tissue, rng,
                    )
                    if rows:
                        writer.writerows(rows)
                        mf.flush()
                        done_patients.add(pat0)
                        completed += 1

        if completed >= args.num_patients:
            print(f"\nZiel erreicht ({completed}/{args.num_patients} Patienten). Manifest: {args.manifest.resolve()}")
            return

        # 2. Weitere Patienten deterministisch ueber idc-index streamen
        print("\n[IDC] Initialisiere IDCClient fuer weitere Patienten...")
        client = index.IDCClient()
        df = client.index
        rms_df = df[(df["collection_id"].str.lower() == "rms_mutation_prediction") & (df["Modality"] == "SM")].copy()

        ts_cols = [c for c in rms_df.columns if "syntax" in c.lower()]
        if ts_cols:
            ts_series = rms_df[ts_cols[0]].astype(str)
            has_lossless = ts_series.str.contains(r"1\.2\.840\.10008\.1\.2\.4\.90|1\.2\.840\.10008\.1\.2\.1", regex=True, na=False)
            has_lossy = ts_series.str.contains(r"1\.2\.840\.10008\.1\.2\.4\.91|1\.2\.840\.10008\.1\.2\.4\.50", regex=True, na=False)
            rms_df = rms_df[has_lossless & (~has_lossy)]

        if "series_size_MB" in rms_df.columns:
            rms_df = rms_df[(rms_df["series_size_MB"] >= 300) & (rms_df["series_size_MB"] <= 2800)].sort_values(by=["PatientID", "series_size_MB"], ascending=[True, True])
        else:
            rms_df = rms_df.sort_values(by=["PatientID"])

        patient_groups = list(rms_df.groupby("PatientID", sort=False))
        perm = rng.permutation(len(patient_groups))
        shuffled_patients = [patient_groups[i] for i in perm]

        for pat_id, grp in shuffled_patients:
            if completed >= args.num_patients:
                break
            pat_str = str(pat_id).strip()
            if pat_str in done_patients:
                continue

            fold = (completed % 5) + 1
            for _, series_row in grp.iterrows():
                series_uid = str(series_row["SeriesInstanceUID"])
                size_mb = float(series_row.get("series_size_MB", 0.0))
                print(f"\n[{completed + 1}/{args.num_patients}] Lade Patient {pat_str} (Serie {series_uid[:18]}..., {size_mb:.1f} MB)...")

                if args.tmp_dcm_dir.exists():
                    shutil.rmtree(args.tmp_dcm_dir, ignore_errors=True)
                args.tmp_dcm_dir.mkdir(parents=True, exist_ok=True)

                try:
                    client.download_from_selection(
                        seriesInstanceUID=[series_uid],
                        downloadDir=str(args.tmp_dcm_dir),
                        use_s5cmd_sync=True,
                    )
                    vols = get_verified_volumes(args.tmp_dcm_dir)
                    if not vols:
                        print("  [Gatekeeper] Keine verlustfreie Level-0 VOLUME-Instanz in dieser Serie -> naechste Serie.")
                        continue

                    rows = sample_patient(
                        pat_str, series_uid, fold, vols,
                        args.tiles_dir, args.roi_dir, args.tiles_per_patient,
                        args.fast_per_patient, args.min_tissue, rng,
                    )
                    if len([r for r in rows if r["track"] == "A"]) >= min(50, args.tiles_per_patient):
                        writer.writerows(rows)
                        mf.flush()
                        done_patients.add(pat_str)
                        completed += 1
                        break
                except Exception as exc:
                    print(f"  [warn] Fehler bei Serie {series_uid}: {exc}")
                finally:
                    shutil.rmtree(args.tmp_dcm_dir, ignore_errors=True)

        print(f"\nFertig! {completed}/{args.num_patients} Patienten vorbereitet. Manifest: {args.manifest.resolve()}")
    finally:
        mf.close()


if __name__ == "__main__":
    main()