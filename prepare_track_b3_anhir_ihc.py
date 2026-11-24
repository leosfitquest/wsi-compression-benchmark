"""
prepare_track_b3_anhir_ihc.py
-----------------------------
100% Original-Daten-Pipeline fuer Track B3 (ANHIR IHC / DAB):
- VUB GPU Default (`--mag-mode 40x_only`): Ausschliesslich die beiden echten 40x
  verlustfreien Mikroskop-Subsets nach Borovec et al. (IEEE TMI 2020, Table I & S.I):
    * `gastric` (Leica DM LB2 Mikroskop, 40x human, 0.2528 um/px, 9 Sets: CD4, CD8, CD68, CD1a, LMP-1/EBV)
    * `lung-lesion` (Zeiss Axio Imager M1, 40x Maus, 0.174 um/px, 3 Sets: CD31, CC10, Ki67, Pro-SPC)
- Option `--mag-mode all_microscope`: Schaltet zusaetzlich die 10x-Mikroskop-Subsets
  (`lung-lobes` 1.274 um/px & `mammary-gland` 2.294 um/px) fuer lokale Piloten frei.
- Schliesst WSI-Scanner-Subsets (`human-breast` Aperio AT2, `mice-kidney` Hamamatsu,
  `coad` 3DHistech, `kidney` Aperio) sowie reine H&E-Schnitte strikt aus.
- Deterministischer, dateinamen-basierter RNG-Seed (sha256) und Regex-Wortgrenzen fuer Marker.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from audit_track_a_forensics import measure_8x8_blockiness

Image.MAX_IMAGE_PIXELS = None

ANHIR_BASE_URL = "http://ptak.felk.cvut.cz/Medical/dataset_ANHIR/images"
ANHIR_CSV_URL = f"{ANHIR_BASE_URL}/dataset_medium.csv"

# Verifizierte optische Mikroskop-Subsets (Borovec et al., IEEE TMI 2020, Table I & S.I)
LOSSLESS_40X_SUBSETS = ("gastric", "lung-lesion")
LOSSLESS_ALL_MICROSCOPE_SUBSETS = ("gastric", "lung-lesion", "lung-lobes", "mammary-gland")
# Hart ausgeschlossene klinische WSI-Scanner-Subsets (lossy beim Scan)
EXCLUDED_WSI_SUBSETS = ("breast", "mice-kidney", "kidney", "coad")
NON_IHC_STAINS = {"ANHIR_HE", "ANHIR_PAS", "ANHIR_MAS"}

_STAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ANHIR_PROSPC", re.compile(r"(?:^|[^a-z0-9])pro[-_]?spc(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_CD31",   re.compile(r"(?:^|[^a-z0-9])cd[-_]?31(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_KI67",   re.compile(r"(?:^|[^a-z0-9])ki[-_]?67(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_CC10",   re.compile(r"(?:^|[^a-z0-9])cc[-_]?10(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_HER2",   re.compile(r"(?:^|[^a-z0-9])her[-_]?2(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_CD68",   re.compile(r"(?:^|[^a-z0-9])cd[-_]?68(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_CD1A",   re.compile(r"(?:^|[^a-z0-9])cd[-_]?1a(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_CD4",    re.compile(r"(?:^|[^a-z0-9])cd[-_]?4(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_CD8",    re.compile(r"(?:^|[^a-z0-9])cd[-_]?8(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_LMP1",   re.compile(r"(?:^|[^a-z0-9])(lmp[-_]?1|ebv)(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_ER",     re.compile(r"(?:^|[^a-z0-9])er(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_PR",     re.compile(r"(?:^|[^a-z0-9])pr(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_PAS",    re.compile(r"(?:^|[^a-z0-9])pas(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_MAS",    re.compile(r"(?:^|[^a-z0-9])(mas|masson)(?:$|[^a-z0-9])", re.I)),
    ("ANHIR_HE",     re.compile(r"(?:^|[^a-z0-9])(h&?e|he)(?:$|[^a-z0-9])", re.I)),
]

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


def _allowed_subsets(mag_mode: str) -> tuple[str, ...]:
    if mag_mode == "40x_only":
        return LOSSLESS_40X_SUBSETS
    if mag_mode == "all_microscope":
        return LOSSLESS_ALL_MICROSCOPE_SUBSETS
    raise ValueError(f"Unbekannter mag_mode='{mag_mode}'. Erlaubt: '40x_only', 'all_microscope'.")


def _is_allowed_slide_path(path_str: str, allowed_subsets: tuple[str, ...]) -> bool:
    low = path_str.lower()
    if any(ex in low for ex in EXCLUDED_WSI_SUBSETS):
        return False
    return any(sub in low for sub in allowed_subsets)


def _detect_stain_name(filename: str) -> str:
    """Erkennt den IHC-Marker per Regex-Token-Grenzen (kein False-Match auf 'upper_HE' oder 'other')."""
    for label, pat in _STAIN_PATTERNS:
        if pat.search(filename):
            return label
    return "ANHIR_IHC"


def _slide_rng(filename: str, base_seed: int = 42) -> np.random.Generator:
    """Deterministischer RNG abgeleitet aus Dateiname + Seed (unabhaengig von slide_idx)."""
    digest = hashlib.sha256(f"{base_seed}:{filename}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _fetch_with_auth(url: str, user: str | None, password: str | None, timeout: int = 30) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download_individual_anhir_pngs(
    raw_dir: Path,
    user: str | None,
    password: str | None,
    allowed_subsets: tuple[str, ...],
    max_slides: int = 60,
) -> list[Path]:
    if not user or not password:
        return []

    print(f"[ANHIR Server] Lade Dateiliste von {ANHIR_CSV_URL} (Subsets: {allowed_subsets}) ...")
    csv_bytes = _fetch_with_auth(ANHIR_CSV_URL, user, password)
    csv_text = csv_bytes.decode("utf-8", errors="replace")

    rel_paths: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9_\-/\.]+?\.png", csv_text):
        clean = token.lstrip("/")
        if clean.startswith("images/"):
            clean = clean[len("images/"):]
        if clean not in seen and _is_allowed_slide_path(clean, allowed_subsets):
            if _detect_stain_name(Path(clean).stem) in NON_IHC_STAINS:
                continue
            seen.add(clean)
            rel_paths.append(clean)

    downloaded: list[Path] = []
    for rel in rel_paths:
        if len(downloaded) >= max_slides:
            break
        scale100_rel = re.sub(r"scale-\d+pc", "scale-100pc", rel)
        candidates = [scale100_rel, rel] if scale100_rel != rel else [rel]

        for cand in candidates:
            safe_name = cand.replace("/", "_")
            dest = raw_dir / safe_name
            if dest.exists() and dest.stat().st_size > 1024:
                downloaded.append(dest)
                break
            url = f"{ANHIR_BASE_URL}/{cand}"
            try:
                print(f"[Download {len(downloaded)+1}/{max_slides}] Lade {url} ...")
                img_bytes = _fetch_with_auth(url, user, password, timeout=60)
                if len(img_bytes) > 1024:
                    dest.write_bytes(img_bytes)
                    downloaded.append(dest)
                    print(f"  -> Gespeichert: {dest.name} ({len(img_bytes)/1e6:.2f} MB)")
                    break
            except Exception:
                continue

    return downloaded


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


def prepare_verified_anhir_b3(
    data_dir: Path,
    manifest_path: Path,
    user: str | None = None,
    password: str | None = None,
    mag_mode: str = "40x_only",
    max_tiles: int = 2000,
    fast_subset_size: int = 40,
    max_slides: int = 60,
    min_slide_dim: int = 1024,
    max_blockiness: float | None = None,
) -> None:
    allowed_subsets = _allowed_subsets(mag_mode)
    raw_dir = data_dir / "raw_slides"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_npy = data_dir / "ihc_tiles_u8.npy"

    all_pngs = sorted(raw_dir.rglob("*.png"))
    ihc_files = [
        p for p in all_pngs
        if _is_allowed_slide_path(str(p.relative_to(raw_dir)), allowed_subsets) and _detect_stain_name(p.stem) not in NON_IHC_STAINS
    ]

    if not ihc_files and user and password:
        _download_individual_anhir_pngs(
            raw_dir, user=user, password=password, allowed_subsets=allowed_subsets, max_slides=max_slides
        )
        all_pngs = sorted(raw_dir.rglob("*.png"))
        ihc_files = [
            p for p in all_pngs
            if _is_allowed_slide_path(str(p.relative_to(raw_dir)), allowed_subsets) and _detect_stain_name(p.stem) not in NON_IHC_STAINS
        ]

    if not ihc_files:
        raise FileNotFoundError(
            f"\n[STOP] Keine ANHIR IHC .png-Originaldateien fuer mag_mode='{mag_mode}' "
            f"(Subsets: {allowed_subsets}) in '{raw_dir.resolve()}' gefunden!"
        )

    print(
        f"[ANHIR IHC] {len(ihc_files)} echte IHC/DAB-Slides fuer mag_mode='{mag_mode}' "
        f"(Subsets: {allowed_subsets}; WSI-Scanner & H&E ausgeschlossen)."
    )

    collected: list[tuple[str, str, str, np.ndarray]] = []
    skipped_small = 0
    skipped_blocky = 0

    for slide_idx, p in enumerate(ihc_files):
        remaining_slides = len(ihc_files) - slide_idx
        quota = int(np.ceil((max_tiles - len(collected)) / max(1, remaining_slides)))
        if quota <= 0:
            break

        with Image.open(p) as im:
            w, h = im.size
            if min(w, h) < min_slide_dim:
                skipped_small += 1
                continue
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)

        stain = _detect_stain_name(p.stem)
        coords = [
            (y, x)
            for y in range(0, h - 256 + 1, 256)
            for x in range(0, w - 256 + 1, 256)
        ]
        rng = _slide_rng(p.name, base_seed=42)
        order = rng.permutation(len(coords))

        slide_count = 0
        for idx in order:
            if slide_count >= quota or len(collected) >= max_tiles:
                break
            y, x = coords[int(idx)]
            patch = arr[y : y + 256, x : x + 256]
            pm = patch.mean(axis=-1)
            if float(np.mean((pm >= 25.0) & (pm <= 225.0))) < 0.70:
                continue
            if max_blockiness is not None and measure_8x8_blockiness(patch) > max_blockiness:
                skipped_blocky += 1
                continue
            tid = f"anhir_{p.stem}_x{x}_y{y}"
            collected.append((tid, f"ANHIR_CTU_PNG_{mag_mode}", stain, patch.copy()))
            slide_count += 1

        print(f"  [{slide_idx+1}/{len(ihc_files)}] {p.name} ({stain}): {slide_count} Kacheln extrahiert.")

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

    counts_str = ", ".join(f"{k}: {len(v)}" for k, v in sorted(by_stain.items()))
    print(f"\n[Fertig] {len(rows)} verifizierte verlustfreie ANHIR IHC-Kacheln gespeichert ({counts_str}).")
    print(f"  Mittlere 8x8-Blockiness: {np.mean([r['blockiness_8x8'] for r in rows]):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download & prepare strictly verified ANHIR IHC PNG tiles from CTU Prague server.")
    p.add_argument("--data-dir", type=Path, default=Path("anhir_data"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b3_ihc.csv"))
    p.add_argument("--user", type=str, default=None)
    p.add_argument("--password", type=str, default=None)
    p.add_argument(
        "--mag-mode",
        choices=("40x_only", "all_microscope"),
        default="40x_only",
        help="40x_only = Gastric (Leica 40x) + Lung lesions (Zeiss 40x) [VUB GPU default]; all_microscope adds 10x lung-lobes/mammary-gland.",
    )
    p.add_argument("--max-tiles", type=int, default=2000)
    p.add_argument("--fast-subset-size", type=int, default=40)
    p.add_argument("--max-slides", type=int, default=60)
    p.add_argument("--max-blockiness", type=float, default=None, help="Optional blockiness filter (None avoids truncation bias on lossless PNGs)")
    args = p.parse_args()
    prepare_verified_anhir_b3(
        args.data_dir, args.manifest, user=args.user, password=args.password,
        mag_mode=args.mag_mode, max_tiles=args.max_tiles,
        fast_subset_size=args.fast_subset_size, max_slides=args.max_slides,
        max_blockiness=args.max_blockiness,
    )