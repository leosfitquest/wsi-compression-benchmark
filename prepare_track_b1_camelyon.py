"""
prepare_track_b1_camelyon.py
----------------------------
100% Original-Daten-Pipeline fuer Track B1 (CAMELYON16 Lymph Node Metastasis):
  - KEINE synthetischen Daten, KEINE Proxy-Kohorten.
  - Fragt automatisch das offizielle AWS-S3-Bucket-Verzeichnis (camelyon-dataset)
    sowie die GigaDB-100439-Mirrors (Wasabi S3 CDN & CNGB HTTP) ab, um die exakten
    Pfade fuer `lesion_annotations.zip` / `tumor_*.xml` und `tumor_*.tif` / `normal_*.tif`
    aufzuloesen.
  - Extrahiert mit OpenSlide auf Level 0 exakt 256x256 Kacheln innerhalb der
    verifizierten XML-Metastasen-Polygone (label=1) und aus Normal-Slides (label=0).
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw

from audit_track_a_forensics import measure_8x8_blockiness
from prepare_track_a_rms import ruifrok_he_ratio

S3_BUCKET_ROOT = "https://camelyon-dataset.s3.amazonaws.com"
WASABI_GIGADB = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100439"
CNGB_GIGADB = "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439"

DEFAULT_TUMOR_SLIDES = ("tumor_001", "tumor_002", "tumor_003", "tumor_004", "tumor_005")
DEFAULT_NORMAL_SLIDES = ("normal_001", "normal_002", "normal_003", "normal_004", "normal_005")

MANIFEST_B1_FIELDS = [
    "tile_id",
    "tile_idx",
    "track",
    "slide_id",
    "label",
    "class_name",
    "level0_x",
    "level0_y",
    "h_ratio",
    "blockiness_8x8",
    "is_fast_subset",
]


def _discover_s3_keys() -> list[str]:
    """Liest das offizielle AWS S3 Bucket-Inhaltsverzeichnis aus, um die exakten Datei-Keys zu ermitteln."""
    keys: list[str] = []
    for url in (f"{S3_BUCKET_ROOT}/?list-type=2&max-keys=1000", f"{S3_BUCKET_ROOT}/"):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            found = [el.text for el in root.iter() if el.tag.endswith("Key") and el.text]
            if found:
                keys.extend(found)
                print(f"[S3 Discovery] {len(found)} Originaldateien im CAMELYON-Bucket gefunden.")
                break
        except Exception:
            continue
    return keys


def _download_file(urls: list[str], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            print(f"[Download] Pruefe & lade Originaldatei: {url} ...")

            def _hook(count: int, block_size: int, total_size: int) -> None:
                if total_size > 0 and count % 200 == 0:
                    mb = (count * block_size) / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    pct = min(100.0, 100.0 * mb / tot_mb)
                    print(f"\r  Fortschritt: {pct:5.1f}% ({mb:.1f} / {tot_mb:.1f} MB)", end="", flush=True)

            urllib.request.urlretrieve(url, dest, reporthook=_hook)
            if dest.exists() and dest.stat().st_size > 1024:
                print(f"\n[OK] Gespeichert unter {dest} ({dest.stat().st_size / 1e6:.2f} MB)")
                return dest
        except Exception as exc:
            print(f"\n  -> Nicht unter diesem Pfad ({exc})")
            last_err = exc
            dest.unlink(missing_ok=True)
    raise RuntimeError(
        f"Download fuer {dest.name} fehlgeschlagen ({last_err}).\n"
        f"Bitte lege die Original-CAMELYON16-Dateien direkt in '{dest.parent.resolve()}' ab."
    )


def _ensure_xml_annotations(xml_dir: Path, required_slides: tuple[str, ...], s3_keys: list[str]) -> None:
    xml_dir.mkdir(parents=True, exist_ok=True)
    if all((xml_dir / f"{sid}.xml").exists() for sid in required_slides):
        return

    # 1. Versuche zuerst das komplette lesion_annotations.zip (~10 MB) zu laden und zu entpacken
    s3_zip_urls = [
        f"{S3_BUCKET_ROOT}/{k}" for k in s3_keys
        if k.lower().endswith(".zip") and ("lesion" in k.lower() or "annot" in k.lower())
    ]
    zip_candidates = s3_zip_urls + [
        f"{WASABI_GIGADB}/CAMELYON16/training/lesion_annotations.zip",
        f"{WASABI_GIGADB}/training/lesion_annotations.zip",
        f"{WASABI_GIGADB}/lesion_annotations.zip",
        f"{CNGB_GIGADB}/CAMELYON16/training/lesion_annotations.zip",
        f"{CNGB_GIGADB}/training/lesion_annotations.zip",
        f"{CNGB_GIGADB}/lesion_annotations.zip",
        f"{S3_BUCKET_ROOT}/CAMELYON16/training/lesion_annotations.zip",
        f"{S3_BUCKET_ROOT}/CAMELYON16/lesion_annotations.zip",
    ]

    anno_zip = xml_dir.parent / "lesion_annotations.zip"
    try:
        _download_file(zip_candidates, anno_zip)
        with zipfile.ZipFile(anno_zip, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith(".xml"):
                    (xml_dir / Path(member).name).write_bytes(zf.read(member))
        anno_zip.unlink(missing_ok=True)
    except Exception:
        anno_zip.unlink(missing_ok=True)

    # 2. Falls einzelne XMLs noch fehlen: Einzel-XML-Download ueber gefundene S3-Keys und Mirrors
    for sid in required_slides:
        target_xml = xml_dir / f"{sid}.xml"
        if target_xml.exists():
            continue
        s3_xml_urls = [f"{S3_BUCKET_ROOT}/{k}" for k in s3_keys if k.lower().endswith(f"{sid.lower()}.xml")]
        xml_urls = s3_xml_urls + [
            f"{S3_BUCKET_ROOT}/CAMELYON16/annotations/{sid}.xml",
            f"{S3_BUCKET_ROOT}/CAMELYON16/lesion_annotations/{sid}.xml",
            f"{S3_BUCKET_ROOT}/CAMELYON16/training/lesion_annotations/{sid}.xml",
            f"{WASABI_GIGADB}/CAMELYON16/training/lesion_annotations/{sid}.xml",
            f"{WASABI_GIGADB}/lesion_annotations/{sid}.xml",
        ]
        _download_file(xml_urls, target_xml)


def _build_slide_urls(sid: str, is_tumor: bool, s3_keys: list[str]) -> list[str]:
    sub = "tumor" if is_tumor else "normal"
    s3_matches = [f"{S3_BUCKET_ROOT}/{k}" for k in s3_keys if k.lower().endswith(f"{sid.lower()}.tif")]
    return s3_matches + [
        f"{WASABI_GIGADB}/CAMELYON16/training/{sub}/{sid}.tif",
        f"{WASABI_GIGADB}/training/{sub}/{sid}.tif",
        f"{WASABI_GIGADB}/{sub}/{sid}.tif",
        f"{CNGB_GIGADB}/CAMELYON16/training/{sub}/{sid}.tif",
        f"{CNGB_GIGADB}/training/{sub}/{sid}.tif",
        f"{S3_BUCKET_ROOT}/CAMELYON16/images/{sid}.tif",
        f"{S3_BUCKET_ROOT}/CAMELYON16/training/{sub}/{sid}.tif",
    ]


def parse_camelyon_xml_polygons(xml_path: Path) -> list[list[tuple[float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons: list[list[tuple[float, float]]] = []
    for anno in root.findall(".//Annotation"):
        coords = []
        for coord in anno.findall(".//Coordinate"):
            coords.append((float(coord.attrib["X"]), float(coord.attrib["Y"])))
        if len(coords) >= 3:
            polygons.append(coords)
    return polygons


def build_lowres_tumor_mask(
    slide: openslide.OpenSlide, polygons: list[list[tuple[float, float]]], downsample: int = 32
) -> np.ndarray:
    w0, h0 = slide.dimensions
    wm, hm = max(1, w0 // downsample), max(1, h0 // downsample)
    mask_img = Image.new("L", (wm, hm), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        scaled = [(x / downsample, y / downsample) for x, y in poly]
        draw.polygon(scaled, outline=1, fill=1)
    return np.asarray(mask_img, dtype=np.uint8) > 0


def extract_tumor_tiles_from_slide(
    slide_path: Path, xml_path: Path, slide_id: str, max_tiles: int, seed: int = 42
) -> list[tuple[str, str, int, int, np.ndarray]]:
    slide = openslide.OpenSlide(str(slide_path))
    polygons = parse_camelyon_xml_polygons(xml_path)
    if not polygons:
        slide.close()
        raise ValueError(f"Keine Tumor-Polygone in {xml_path} gefunden!")

    ds = 32
    tumor_mask = build_lowres_tumor_mask(slide, polygons, downsample=ds)
    ys, xs = np.where(tumor_mask)
    if len(xs) == 0:
        slide.close()
        raise ValueError(f"Leere Tumor-Maske fuer {slide_id}!")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))
    extracted: list[tuple[str, str, int, int, np.ndarray]] = []
    seen_coords: set[tuple[int, int]] = set()

    for idx in order:
        if len(extracted) >= max_tiles:
            break
        x0 = int((xs[idx] * ds) // 256) * 256
        y0 = int((ys[idx] * ds) // 256) * 256
        if (x0, y0) in seen_coords:
            continue
        seen_coords.add((x0, y0))

        patch = np.asarray(slide.read_region((x0, y0), 0, (256, 256)).convert("RGB"), dtype=np.uint8)
        pm = patch.mean(axis=-1)
        if float(np.mean((pm >= 25.0) & (pm <= 225.0))) < 0.70:
            continue
        tid = f"cam16_{slide_id}_x{x0}_y{y0}"
        extracted.append((tid, slide_id, x0, y0, patch))

    slide.close()
    return extracted


def extract_normal_tiles_from_slide(
    slide_path: Path, slide_id: str, max_tiles: int, seed: int = 42
) -> list[tuple[str, str, int, int, np.ndarray]]:
    slide = openslide.OpenSlide(str(slide_path))
    best_level = min(slide.level_count - 1, 5)
    ds = float(slide.level_downsamples[best_level])
    thumb = np.asarray(slide.read_region((0, 0), best_level, slide.level_dimensions[best_level]).convert("RGB"))
    gray = thumb.mean(axis=-1)
    tissue_ys, tissue_xs = np.where((gray >= 30.0) & (gray <= 215.0))

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tissue_xs))
    extracted: list[tuple[str, str, int, int, np.ndarray]] = []
    seen_coords: set[tuple[int, int]] = set()

    for idx in order:
        if len(extracted) >= max_tiles:
            break
        x0 = int((tissue_xs[idx] * ds) // 256) * 256
        y0 = int((tissue_ys[idx] * ds) // 256) * 256
        if (x0, y0) in seen_coords:
            continue
        seen_coords.add((x0, y0))

        patch = np.asarray(slide.read_region((x0, y0), 0, (256, 256)).convert("RGB"), dtype=np.uint8)
        pm = patch.mean(axis=-1)
        if float(np.mean((pm >= 25.0) & (pm <= 220.0))) < 0.75:
            continue
        tid = f"cam16_{slide_id}_x{x0}_y{y0}"
        extracted.append((tid, slide_id, x0, y0, patch))

    slide.close()
    return extracted


def prepare_real_camelyon16(
    data_dir: Path,
    manifest_out: Path,
    num_slides_per_class: int = 2,
    tiles_per_slide: int = 200,
    fast_per_class: int = 20,
    keep_wsi: bool = False,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    wsi_dir = data_dir / "wsi_tmp"
    xml_dir = data_dir / "lesion_annotations"
    out_npy = data_dir / "b1_tiles_u8.npy"

    tumor_slides = DEFAULT_TUMOR_SLIDES[:num_slides_per_class]
    normal_slides = DEFAULT_NORMAL_SLIDES[:num_slides_per_class]

    s3_keys = _discover_s3_keys()
    _ensure_xml_annotations(xml_dir, tumor_slides, s3_keys)

    tumor_tiles: list[tuple[str, str, int, int, np.ndarray]] = []
    normal_tiles: list[tuple[str, str, int, int, np.ndarray]] = []

    for sid in tumor_slides:
        xml_path = xml_dir / f"{sid}.xml"
        tif_path = wsi_dir / f"{sid}.tif"
        if not tif_path.exists():
            _download_file(_build_slide_urls(sid, is_tumor=True, s3_keys=s3_keys), tif_path)
        print(f"[Extraktion] Schneide {tiles_per_slide} echte Metastasen-Kacheln aus {sid}.tif (innerhalb XML-Polygon)...")
        t_list = extract_tumor_tiles_from_slide(tif_path, xml_path, sid, max_tiles=tiles_per_slide)
        tumor_tiles.extend(t_list)
        if not keep_wsi:
            tif_path.unlink(missing_ok=True)

    for sid in normal_slides:
        tif_path = wsi_dir / f"{sid}.tif"
        if not tif_path.exists():
            _download_file(_build_slide_urls(sid, is_tumor=False, s3_keys=s3_keys), tif_path)
        print(f"[Extraktion] Schneide {tiles_per_slide} echte Normal-Kacheln aus {sid}.tif...")
        n_list = extract_normal_tiles_from_slide(tif_path, sid, max_tiles=tiles_per_slide)
        normal_tiles.extend(n_list)
        if not keep_wsi:
            tif_path.unlink(missing_ok=True)

    total_n = len(tumor_tiles) + len(normal_tiles)
    u8_stack = np.empty((total_n, 256, 256, 3), dtype=np.uint8)
    rows: list[dict[str, object]] = []

    fast_tum_step = max(1, len(tumor_tiles) // fast_per_class)
    fast_norm_step = max(1, len(normal_tiles) // fast_per_class)

    idx = 0
    for i, (tid, sid, x0, y0, patch) in enumerate(tumor_tiles):
        u8_stack[idx] = patch
        pm = patch.mean(axis=-1)
        h_rat = ruifrok_he_ratio(patch, (pm >= 25.0) & (pm <= 225.0))
        rows.append({
            "tile_id": tid,
            "tile_idx": idx,
            "track": "B1",
            "slide_id": sid,
            "label": 1,
            "class_name": "tumor",
            "level0_x": x0,
            "level0_y": y0,
            "h_ratio": round(h_rat, 4),
            "blockiness_8x8": round(measure_8x8_blockiness(patch), 4),
            "is_fast_subset": (i % fast_tum_step == 0) and (i // fast_tum_step < fast_per_class),
        })
        idx += 1

    for i, (tid, sid, x0, y0, patch) in enumerate(normal_tiles):
        u8_stack[idx] = patch
        pm = patch.mean(axis=-1)
        h_rat = ruifrok_he_ratio(patch, (pm >= 25.0) & (pm <= 225.0))
        rows.append({
            "tile_id": tid,
            "tile_idx": idx,
            "track": "B1",
            "slide_id": sid,
            "label": 0,
            "class_name": "normal",
            "level0_x": x0,
            "level0_y": y0,
            "h_ratio": round(h_rat, 4),
            "blockiness_8x8": round(measure_8x8_blockiness(patch), 4),
            "is_fast_subset": (i % fast_norm_step == 0) and (i // fast_norm_step < fast_per_class),
        })
        idx += 1

    np.save(out_npy, u8_stack)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_B1_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Fertig] Echte CAMELYON16-Daten vorbereitet: {len(tumor_tiles)} Tumor + {len(normal_tiles)} Normal.")
    print(f"  Mittlere 8x8-Blockiness (echter Scanner Q=80 Nachweis): {np.mean([r['blockiness_8x8'] for r in rows]):.4f}")
    print(f"  Manifest: {manifest_out.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare 100% real CAMELYON16 tiles for Track B1.")
    p.add_argument("--data-dir", type=Path, default=Path("camelyon_data"))
    p.add_argument("--manifest-out", type=Path, default=Path("results/curated_v1/manifest_track_b1_tumor.csv"))
    p.add_argument("--slides-per-class", type=int, default=2)
    p.add_argument("--tiles-per-slide", type=int, default=200)
    p.add_argument("--fast-per-class", type=int, default=20)
    p.add_argument("--keep-wsi", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_real_camelyon16(
        args.data_dir, args.manifest_out, args.slides_per_class,
        args.tiles_per_slide, args.fast_per_class, args.keep_wsi,
    )