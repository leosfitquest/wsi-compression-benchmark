"""
prepare_track_b2_pannuke.py
---------------------------
Bereitet Track B2 (PanNuke Fold 3, n=2.722 native 256x256 Tiles ueber 19 Organe) vor:
  1. Laedt offizielles fold_3.zip von Warwick TIA herunter (falls noch nicht vorhanden).
  2. Konvertiert images.npy (3.99 GB float64) verlustfrei in images_u8.npy (510.4 MB uint8)
     und masks.npy (7.97 GB float64, 6 Kanaele) in masks_u16.npz (~40 MB uint16),
     um ~11.4 GB SSD-Speicherplatz zu sparen.
  3. Erstellt manifest_track_b2_pannuke.csv mit Zellkern-Anzahlen der 5 PanNuke-Klassen
     (Neoplastic, Inflammatory, Connective, Dead, Epithelial), Forensik-Blockiness und
     einem balancierten Fast-Subset (2 kernreichste Kacheln pro Organ = 38 Kacheln ueber alle 19 Organe).
"""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from audit_track_a_forensics import measure_8x8_blockiness
from prepare_track_a_rms import ruifrok_he_ratio

WARWICK_FOLD3_URL = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip"
CELL_CLASSES = ("Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial")

MANIFEST_B2_FIELDS = [
    "tile_id",
    "tile_idx",
    "track",
    "fold",
    "organ",
    "total_nuclei",
    "n_neoplastic",
    "n_inflammatory",
    "n_connective",
    "n_dead",
    "n_epithelial",
    "nuclei_area_fraction",
    "h_ratio",
    "blockiness_8x8",
    "is_fast_subset",
]


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Download] Lade {url} nach {dest} ...")

    def _hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and count % 200 == 0:
            mb = (count * block_size) / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            pct = min(100.0, 100.0 * mb / tot_mb)
            print(f"\r  Fortschritt: {pct:5.1f}% ({mb:.1f} / {tot_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print("\n[Download] Abgeschlossen.")


def _extract_member_by_suffix(zf: zipfile.ZipFile, suffix: str, out_path: Path) -> Path:
    matches = [m for m in zf.namelist() if m.lower().endswith(suffix.lower())]
    if not matches:
        raise FileNotFoundError(f"Keine Datei mit Endung '{suffix}' in {zf.filename} gefunden.")
    member = matches[0]
    print(f"[Unzip] Extrahiere '{member}' -> {out_path.name} ...")
    with zf.open(member) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out_path


def prepare_pannuke_fold3(
    data_dir: Path,
    manifest_path: Path,
    fast_per_organ: int = 2,
    keep_zip: bool = False,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    u8_img_path = data_dir / "images_u8.npy"
    types_path = data_dir / "types.npy"
    masks_npz_path = data_dir / "masks_u16.npz"
    zip_path = data_dir / "fold_3.zip"

    # 1. Falls images_u8.npy oder masks_u16.npz noch fehlen: aus lokalem Ordner oder fold_3.zip konvertieren
    if not (u8_img_path.exists() and types_path.exists() and masks_npz_path.exists()):
        raw_img = next(data_dir.rglob("images.npy"), None)
        raw_types = next(data_dir.rglob("types.npy"), None)
        raw_masks = next(data_dir.rglob("masks.npy"), None)

        if raw_img is None or raw_masks is None:
            if not zip_path.exists():
                _download_with_progress(WARWICK_FOLD3_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                if not types_path.exists():
                    _extract_member_by_suffix(zf, "types.npy", types_path)
                if not u8_img_path.exists():
                    tmp_img = data_dir / "_tmp_images_f64.npy"
                    _extract_member_by_suffix(zf, "images.npy", tmp_img)
                    print("[Convert] Konvertiere images.npy (float64 -> uint8, 510.4 MB)...")
                    arr_img = np.load(tmp_img, mmap_mode="r")
                    u8_arr = np.empty(arr_img.shape, dtype=np.uint8)
                    for i in range(len(arr_img)):
                        u8_arr[i] = np.clip(np.round(arr_img[i]), 0, 255).astype(np.uint8)
                    np.save(u8_img_path, u8_arr)
                    del arr_img, u8_arr
                    tmp_img.unlink(missing_ok=True)

                if not masks_npz_path.exists():
                    tmp_msk = data_dir / "_tmp_masks_f64.npy"
                    _extract_member_by_suffix(zf, "masks.npy", tmp_msk)
                    print("[Convert] Konvertiere masks.npy (float64 -> komprimiertes uint16 .npz)...")
                    arr_msk = np.load(tmp_msk, mmap_mode="r")
                    u16_msk = np.empty((len(arr_msk), 256, 256, 5), dtype=np.uint16)
                    for i in range(len(arr_msk)):
                        u16_msk[i] = np.asarray(arr_msk[i, ..., :5], dtype=np.uint16)
                    np.savez_compressed(masks_npz_path, masks=u16_msk)
                    del arr_msk, u16_msk
                    tmp_msk.unlink(missing_ok=True)

            if not keep_zip and zip_path.exists():
                zip_path.unlink(missing_ok=True)
        else:
            if raw_types and raw_types != types_path:
                shutil.copy2(raw_types, types_path)
            if not u8_img_path.exists():
                arr_img = np.load(raw_img, mmap_mode="r")
                u8_arr = np.clip(np.round(arr_img), 0, 255).astype(np.uint8)
                np.save(u8_img_path, u8_arr)
            if not masks_npz_path.exists():
                arr_msk = np.load(raw_masks, mmap_mode="r")
                u16_msk = np.asarray(arr_msk[..., :5], dtype=np.uint16)
                np.savez_compressed(masks_npz_path, masks=u16_msk)

    # 2. Manifest mit allen 2.722 Kacheln, 19 Organen und Zellkern-Statistiken aufbauen
    print("[Manifest] Analysiere 2.722 PanNuke Fold-3 Kacheln und 5-Klassen-Kernmasken...")
    images = np.load(u8_img_path, mmap_mode="r")
    organs = np.load(types_path, allow_pickle=True)
    masks_5ch = np.load(masks_npz_path)["masks"]  # (2722, 256, 256, 5) uint16

    rows: list[dict[str, object]] = []
    for idx in range(len(images)):
        tile = np.asarray(images[idx], dtype=np.uint8)
        organ = str(organs[idx]).strip().replace(" ", "_")
        msk = masks_5ch[idx]  # (256, 256, 5)

        class_counts = []
        for c in range(5):
            uniq = np.unique(msk[..., c])
            class_counts.append(int(np.sum(uniq > 0)))

        tot_nuclei = sum(class_counts)
        any_nucleus = np.any(msk > 0, axis=-1)
        nuc_frac = float(np.mean(any_nucleus))

        pixel_means = tile.mean(axis=-1)
        tissue_mask = (pixel_means >= 20.0) & (pixel_means <= 235.0)
        h_ratio = ruifrok_he_ratio(tile, tissue_mask)
        b8 = measure_8x8_blockiness(tile)

        rows.append({
            "tile_id": f"pannuke_{idx:04d}_{organ}",
            "tile_idx": idx,
            "track": "B2",
            "fold": 3,
            "organ": organ,
            "total_nuclei": tot_nuclei,
            "n_neoplastic": class_counts[0],
            "n_inflammatory": class_counts[1],
            "n_connective": class_counts[2],
            "n_dead": class_counts[3],
            "n_epithelial": class_counts[4],
            "nuclei_area_fraction": round(nuc_frac, 4),
            "h_ratio": round(h_ratio, 4),
            "blockiness_8x8": round(b8, 4),
            "is_fast_subset": False,
        })

    # Pro Organ die `fast_per_organ` kernreichsten Kacheln als Fast-Subset markieren (19 x 2 = 38 Tiles)
    by_organ: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_organ.setdefault(str(r["organ"]), []).append(r)

    fast_count = 0
    for organ, grp in sorted(by_organ.items()):
        grp_sorted = sorted(grp, key=lambda x: (int(x["total_nuclei"]), float(x["nuclei_area_fraction"])), reverse=True)
        for item in grp_sorted[:fast_per_organ]:
            item["is_fast_subset"] = True
            fast_count += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_B2_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total_nuc = sum(int(r["total_nuclei"]) for r in rows)
    mean_b8 = float(np.mean([float(r["blockiness_8x8"]) for r in rows]))
    print(f"\n[Fertig] PanNuke Fold 3 vorbereitet ({len(rows)} Kacheln, {len(by_organ)} Organe, {total_nuc} Zellkerne).")
    print(f"  Speicherplatz images_u8.npy: {u8_img_path.stat().st_size / 1e6:.1f} MB | masks_u16.npz: {masks_npz_path.stat().st_size / 1e6:.1f} MB")
    print(f"  Mittlere 8x8-Blockiness (Lossy-Scanner/Upsampling-Herkunft): {mean_b8:.4f}")
    print(f"  Balanciertes Fast-Subset: {fast_count} Kacheln ({fast_per_organ} pro Organ ueber alle {len(by_organ)} Organe)")
    print(f"  Manifest gespeichert: {manifest_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Track B2 (PanNuke Fold 3) in compact uint8/uint16 format.")
    p.add_argument("--data-dir", type=Path, default=Path("pannuke_fold3"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b2_pannuke.csv"))
    p.add_argument("--fast-per-organ", type=int, default=2, help="Tiles per organ for fast subset (default 2 -> 38 tiles)")
    p.add_argument("--keep-zip", action="store_true", help="Keep fold_3.zip after conversion")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_pannuke_fold3(args.data_dir, args.manifest, args.fast_per_organ, args.keep_zip)