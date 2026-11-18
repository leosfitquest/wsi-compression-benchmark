"""
prepare_track_b3_anhir_ihc.py
-----------------------------
Strikt verifizierte Original-Daten-Pipeline fuer Track B3 (ANHIR IHC / DAB):
  - KEIN synthetischer Fallback, KEINE .jpg-Vorschaubilder (scale-5pc/10pc/25pc).
  - Akzeptiert ausschliesslich echte, hochaufloesende .png-Originaldateien
    (scale-100pc, mind. 3000x3000 Pixel) aus `anhir_data/raw_slides/`,
    heruntergeladen von https://anhir.grand-challenge.org/Download/
    (Subsets: lung-lesions, lung-lobes, mammary-glands, mice-kidney).
  - Prueft jede extrahierte 256x256 Kachel forensisch auf 8x8-Blockfreiheit
    (blockiness_8x8 <= 1.03), um vorkomprimierte Scanner-Slides auszuschliessen.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from audit_track_a_forensics import measure_8x8_blockiness

# Erlaubte grossformatige PNGs benoetigen erhöhtes DecompressionBomb-Limit
Image.MAX_IMAGE_PIXELS = None

# Ruifrok & Johnston (2001) Standard-OD-Vektoren fuer H-DAB
_H_VEC = np.array([0.650, 0.704, 0.286], dtype=np.float64)
_DAB_VEC = np.array([0.268, 0.570, 0.776], dtype=np.float64)
_H_VEC /= np.linalg.norm(_H_VEC)
_DAB_VEC /= np.linalg.norm(_DAB_VEC)
_HDAB_RES = np.cross(_H_VEC, _DAB_VEC)
_HDAB_RES /= np.linalg.norm(_HDAB_RES)
_INV_HDAB_MAT = np.linalg.inv(np.stack([_H_VEC, _DAB_VEC, _HDAB_RES], axis=0))

MANIFEST_B3_FIELDS = [
    "tile_id",
    "tile_idx",
    "track",
    "source_type",
    "subset_or_profile",
    "dab_mean_od",
    "h_mean_od",
    "dab_positive_fraction",
    "chroma_std",
    "blockiness_8x8",
    "is_fast_subset",
]


def compute_ihc_stats(tile: np.ndarray) -> tuple[float, float, float, float]:
    od = -np.log10(np.clip(tile.astype(np.float64), 1.0, 255.0) / 255.0)
    stains = od @ _INV_HDAB_MAT
    h_od = np.maximum(stains[..., 0], 0.0)
    dab_od = np.maximum(stains[..., 1], 0.0)
    dab_pos_frac = float(np.mean(dab_od >= 0.15))

    r, g, b = tile[..., 0].astype(np.float64), tile[..., 1].astype(np.float64), tile[..., 2].astype(np.float64)
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    chroma_std = float(np.sqrt(np.var(cb) + np.var(cr)))
    return float(np.mean(dab_od)), float(np.mean(h_od)), dab_pos_frac, chroma_std


def _detect_stain_name(filename: str) -> str:
    low = filename.lower()
    for st in ("cd31", "ki67", "prospc", "cc10", "her2", "er", "pr", "pas", "mas", "he"):
        if st in low:
            return f"ANHIR_{st.upper()}"
    return "ANHIR_IHC"


def prepare_verified_anhir_b3(
    data_dir: Path,
    manifest_path: Path,
    max_tiles: int = 2000,
    fast_subset_size: int = 40,
    min_slide_dim: int = 3000,
    max_blockiness: float = 1.03,
) -> None:
    raw_dir = data_dir / "raw_slides"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_npy = data_dir / "ihc_tiles_u8.npy"

    # Suche ausschliesslich nach .png-Dateien (schliesst scale-5pc/10pc/25pc .jpg strikt aus)
    png_files = [
        p for p in sorted(raw_dir.rglob("*.png"))
        if not any(s in str(p).lower() for s in ("scale-5pc", "scale-10pc", "scale-25pc"))
    ]

    if not png_files:
        raise FileNotFoundError(
            f"\n[STOP] Keine hochaufloesenden ANHIR .png-Originaldateien in '{raw_dir.resolve()}' gefunden!\n"
            f"Da ANHIR einen Login erfordert (https://anhir.grand-challenge.org/Download/),\n"
            f"lade dort bitte die 'scale-100pc/*.png' Dateien der verlustfreien Mikroskop-Sets\n"
            f"(z. B. lesions_1..3, lung-lobes_1..4, mammary-gland_1..2 mit CD31, Ki67, proSPC, Cc10)\n"
            f"herunter und lege sie in '{raw_dir.resolve()}' ab."
        )

    collected: list[tuple[str, str, str, np.ndarray]] = []
    skipped_small = 0
    skipped_blocky = 0

    for p in png_files:
        if len(collected) >= max_tiles:
            break
        with Image.open(p) as im:
            w, h = im.size
            if min(w, h) < min_slide_dim:
                skipped_small += 1
                continue
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)

        stain = _detect_stain_name(p.stem)
        for y in range(0, h - 256 + 1, 256):
            for x in range(0, w - 256 + 1, 256):
                if len(collected) >= max_tiles:
                    break
                patch = arr[y : y + 256, x : x + 256]
                pm = patch.mean(axis=-1)
                if float(np.mean((pm >= 25.0) & (pm <= 225.0))) < 0.70:
                    continue
                if measure_8x8_blockiness(patch) > max_blockiness:
                    skipped_blocky += 1
                    continue
                tid = f"anhir_{p.stem}_x{x}_y{y}"
                collected.append((tid, "ANHIR_Scale100pc_PNG", stain, patch.copy()))

    if not collected:
        raise RuntimeError(
            f"Keine gueltigen verlustfreien Kacheln gefunden (uebersprungene kleine Bilder: {skipped_small}, "
            f"uebersprungene 8x8-blockige Kacheln: {skipped_blocky})."
        )

    u8_stack = np.empty((len(collected), 256, 256, 3), dtype=np.uint8)
    rows: list[dict[str, object]] = []
    for idx, (tid, src_type, stain, tile_arr) in enumerate(collected):
        u8_stack[idx] = tile_arr
        dab_m, h_m, dab_pos, c_std = compute_ihc_stats(tile_arr)
        rows.append({
            "tile_id": tid,
            "tile_idx": idx,
            "track": "B3",
            "source_type": src_type,
            "subset_or_profile": stain,
            "dab_mean_od": round(dab_m, 4),
            "h_mean_od": round(h_m, 4),
            "dab_positive_fraction": round(dab_pos, 4),
            "chroma_std": round(c_std, 3),
            "blockiness_8x8": round(measure_8x8_blockiness(tile_arr), 4),
            "is_fast_subset": False,
        })

    np.save(out_npy, u8_stack)

    by_stain: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_stain.setdefault(str(r["subset_or_profile"]), []).append(r)

    per_group = max(1, fast_subset_size // max(1, len(by_stain)))
    for stain, grp in sorted(by_stain.items()):
        grp_sorted = sorted(grp, key=lambda x: float(x["chroma_std"]), reverse=True)
        for item in grp_sorted[:per_group]:
            item["is_fast_subset"] = True

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_B3_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Fertig] {len(rows)} verifizierte verlustfreie ANHIR scale-100pc PNG-Kacheln gespeichert.")
    print(f"  Mittlere 8x8-Blockiness: {np.mean([r['blockiness_8x8'] for r in rows]):.4f}")
    print(f"  Faerbungen: {sorted(by_stain.keys())}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare strictly verified scale-100pc ANHIR PNG tiles for Track B3.")
    p.add_argument("--data-dir", type=Path, default=Path("anhir_data"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b3_ihc.csv"))
    p.add_argument("--max-tiles", type=int, default=2000)
    p.add_argument("--fast-subset-size", type=int, default=40)
    args = p.parse_args()
    prepare_verified_anhir_b3(args.data_dir, args.manifest, args.max_tiles, args.fast_subset_size)