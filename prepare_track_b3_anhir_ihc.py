"""
prepare_track_b3_anhir_ihc.py
-----------------------------
Bereitet Track B3 (IHC / DAB Color Stress Test, n=2.000 lossless 256x256 Tiles) vor:
  1. Sucht zuerst nach lokalen ANHIR-PNG-Slides in `anhir_data/` oder streamt gezielt
     verlustfreie Mikroskopie-PNGs (Lung Lobes, Breast, Kidney, Mammary Gland) per
     HTTP-Range-Requests direkt aus dem offiziellen Zenodo-Archiv (Record 2597699),
     ohne das gesamte Multi-Gigabyte-ZIP herunterladen zu muessen.
  2. Verifiziert jede Quelle forensisch auf 8x8-Blockfreiheit (blockiness <= 1.03),
     um verlustbehaftete Scanner-Subsets strikt auszuschliessen (Fix 1.4).
  3. Ergaenzt fehlende Kacheln bis n=2.000 deterministisch ueber kalibrierte
     Beer-Lambert Ruifrok (2001) H-DAB-Immunhistochemie-Transformationen auf den
     verifizierten Level-0-Optical-GT-Kacheln (CD31_DAB, Ki67_DAB, HER2_DAB, ER_PR_DAB).
  4. Speichert alle 2.000 Kacheln kompakt in `anhir_data/ihc_tiles_u8.npy` (~393 MB)
     sowie `results/curated_v1/manifest_track_b3_ihc.csv` (inkl. 40-Tile Fast-Subset).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from audit_track_a_forensics import measure_8x8_blockiness

ZENODO_RECORD_API = "https://zenodo.org/api/records/2597699"

# Ruifrok & Johnston (2001) Standard-OD-Vektoren fuer H&E und H-DAB
_H_VEC = np.array([0.650, 0.704, 0.286], dtype=np.float64)
_E_VEC = np.array([0.072, 0.990, 0.105], dtype=np.float64)
_DAB_VEC = np.array([0.268, 0.570, 0.776], dtype=np.float64)
_H_VEC /= np.linalg.norm(_H_VEC)
_E_VEC /= np.linalg.norm(_E_VEC)
_DAB_VEC /= np.linalg.norm(_DAB_VEC)

_HE_RES = np.cross(_H_VEC, _E_VEC)
_HE_RES /= np.linalg.norm(_HE_RES)
_INV_HE_MAT = np.linalg.inv(np.stack([_H_VEC, _E_VEC, _HE_RES], axis=0))

_HDAB_RES = np.cross(_H_VEC, _DAB_VEC)
_HDAB_RES /= np.linalg.norm(_HDAB_RES)
_INV_HDAB_MAT = np.linalg.inv(np.stack([_H_VEC, _DAB_VEC, _HDAB_RES], axis=0))

IHC_PROFILES: tuple[tuple[str, float, float], ...] = (
    ("CD31_DAB_vascular", 1.35, 0.90),
    ("Ki67_DAB_nuclear", 1.20, 1.05),
    ("HER2_DAB_membrane", 1.45, 0.80),
    ("ER_PR_DAB_strong", 1.25, 0.95),
)

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


class HttpRangeFile:
    """Seekable HTTP-Datei mit 4-MB-Tail-Cache + 2-MB-Lesefenster fuer selektives ZIP-Streaming."""

    def __init__(self, url: str, size: int, tail_size: int = 4 * 1024 * 1024, chunk_size: int = 2 * 1024 * 1024) -> None:
        self.url = url
        self.size = size
        self.pos = 0
        self.chunk_size = chunk_size
        self.tail_start = max(0, self.size - tail_size)
        self.tail_buf = self._fetch(self.tail_start, self.size - 1)
        self.win_start = -1
        self.win_end = -1
        self.win_buf = b""

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        elif whence == os.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.pos, self.size))
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if size < 0 or self.pos + size > self.size:
            size = self.size - self.pos
        if size == 0:
            return b""
        start = self.pos
        end = start + size - 1
        self.pos += size

        if start >= self.tail_start:
            rel = start - self.tail_start
            return self.tail_buf[rel : rel + size]
        if self.win_start <= start and end <= self.win_end:
            rel = start - self.win_start
            return self.win_buf[rel : rel + size]
        if size <= self.chunk_size:
            fetch_end = min(self.size - 1, start + self.chunk_size - 1)
            self.win_buf = self._fetch(start, fetch_end)
            self.win_start = start
            self.win_end = start + len(self.win_buf) - 1
            return self.win_buf[:size]
        return self._fetch(start, end)


def compute_ihc_stats(tile: np.ndarray) -> tuple[float, float, float, float]:
    """Berechnet mittlere DAB-OD, Haematoxylin-OD, DAB-Positivitaets-Anteil und Chroma-Std."""
    od = -np.log10(np.clip(tile.astype(np.float64), 1.0, 255.0) / 255.0)
    stains = od @ _INV_HDAB_MAT
    h_od = np.maximum(stains[..., 0], 0.0)
    dab_od = np.maximum(stains[..., 1], 0.0)
    dab_pos_frac = float(np.mean(dab_od >= 0.18))

    r, g, b = tile[..., 0].astype(np.float64), tile[..., 1].astype(np.float64), tile[..., 2].astype(np.float64)
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b
    chroma_std = float(np.sqrt(np.var(cb) + np.var(cr)))
    return float(np.mean(dab_od)), float(np.mean(h_od)), dab_pos_frac, chroma_std


def restain_he_to_hdab(tile_he: np.ndarray, dab_gain: float, h_gain: float) -> np.ndarray:
    """Exakte Beer-Lambert-Transformation einer verlustfreien Level-0 H&E-Kachel in den H-DAB IHC-Farbraum."""
    od = -np.log10(np.clip(tile_he.astype(np.float64), 1.0, 255.0) / 255.0)
    stains = od @ _INV_HE_MAT
    h_conc = np.maximum(stains[..., 0], 0.0) * h_gain
    dab_conc = np.maximum(stains[..., 1], 0.0) * dab_gain
    res_conc = stains[..., 2] * 0.20

    od_hdab = (
        h_conc[..., None] * _H_VEC[None, None, :]
        + dab_conc[..., None] * _DAB_VEC[None, None, :]
        + res_conc[..., None] * _HDAB_RES[None, None, :]
    )
    rgb = 255.0 * (10.0 ** (-np.clip(od_hdab, 0.0, 3.0)))
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _stream_anhir_png_tiles(target_count: int, max_blockiness: float = 1.03) -> list[tuple[str, str, np.ndarray]]:
    """Versucht verlustfreie Mikroskopie-PNGs (Lung Lobes, Breast, Kidney) per HTTP Range aus Zenodo 2597699 zu streamen."""
    extracted: list[tuple[str, str, np.ndarray]] = []
    allowed_prefixes = ("lung-lobes", "breast", "kidney", "mice-kidney", "mammary-gland")
    try:
        req = urllib.request.Request(ZENODO_RECORD_API, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        files = meta.get("files", [])
        zip_entries = [f for f in files if str(f.get("key", "")).endswith(".zip")]
        if not zip_entries:
            return extracted

        zip_info = zip_entries[0]
        zip_url = zip_info.get("links", {}).get("self") or zip_info.get("links", {}).get("download")
        zip_size = int(zip_info.get("size", 0))
        if not zip_url or zip_size <= 0:
            return extracted

        print(f"[Zenodo Range] Verbinde mit {zip_info.get('key')} ({zip_size / 1e9:.2f} GB) per HTTP-Range...")
        range_fp = HttpRangeFile(zip_url, zip_size)
        with zipfile.ZipFile(range_fp, "r") as zf:
            png_members = [
                m for m in zf.namelist()
                if m.lower().endswith(".png")
                and any(p in m.lower() for p in allowed_prefixes)
            ]
            for member in png_members[:25]:
                if len(extracted) >= target_count:
                    break
                raw_bytes = zf.read(member)
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                h, w = arr.shape[:2]
                subset_name = Path(member).parts[0] if Path(member).parts else "anhir_microscopy"
                for y in range(0, h - 256 + 1, 256):
                    for x in range(0, w - 256 + 1, 256):
                        if len(extracted) >= target_count:
                            break
                        patch = arr[y : y + 256, x : x + 256]
                        pm = patch.mean(axis=-1)
                        if float(np.mean((pm >= 25.0) & (pm <= 225.0))) < 0.70:
                            continue
                        if measure_8x8_blockiness(patch) > max_blockiness:
                            continue
                        tid = f"anhir_{subset_name}_x{x}_y{y}"
                        extracted.append((tid, subset_name, patch.copy()))
    except Exception as exc:
        print(f"[Info] Zenodo HTTP-Range-Stream nicht verfuegbar ({exc}) -> nutze kalibrierte Level-0 H-DAB Transformation.")
    return extracted


def prepare_track_b3(
    data_dir: Path,
    rms_tiles_dir: Path,
    manifest_path: Path,
    num_tiles: int = 2000,
    fast_subset_size: int = 40,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    out_npy = data_dir / "ihc_tiles_u8.npy"

    collected: list[tuple[str, str, str, np.ndarray]] = []

    # 1. Lokale ANHIR-PNGs in anhir_data/ pruefen (falls vorhanden)
    local_pngs = [p for p in sorted(data_dir.rglob("*.png")) if p.name != "ihc_tiles_u8.png"]
    for p in local_pngs:
        if len(collected) >= num_tiles:
            break
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            h, w = arr.shape[:2]
            for y in range(0, h - 256 + 1, 256):
                for x in range(0, w - 256 + 1, 256):
                    if len(collected) >= num_tiles:
                        break
                    patch = arr[y : y + 256, x : x + 256]
                    pm = patch.mean(axis=-1)
                    if float(np.mean((pm >= 25.0) & (pm <= 225.0))) < 0.70:
                        continue
                    if measure_8x8_blockiness(patch) <= 1.03:
                        collected.append((f"anhir_{p.stem}_x{x}_y{y}", "ANHIR_Local_PNG", p.parent.name or "ANHIR_IHC", patch.copy()))
        except Exception:
            continue

    # 2. Optionaler Zenodo-Range-Stream fuer ANHIR-Mikroskopie-PNGs
    if len(collected) < num_tiles:
        streamed = _stream_anhir_png_tiles(num_tiles - len(collected))
        for tid, subset_name, patch in streamed:
            collected.append((tid, "ANHIR_Zenodo_PNG", subset_name, patch))

    # 3. Kalibrierte Beer-Lambert Ruifrok (2001) H-DAB IHC-Transformation auf verifizierten Level-0 Optical-GT-Kacheln
    if len(collected) < num_tiles:
        needed = num_tiles - len(collected)
        rms_pngs = sorted(rms_tiles_dir.glob("*.png"))
        if not rms_pngs:
            raise FileNotFoundError(f"Keine verlustfreien Level-0-Kacheln in {rms_tiles_dir} gefunden.")
        print(f"[Ruifrok H-DAB] Erzeuge {needed} kalibrierte verlustfreie H-DAB IHC-Kacheln aus Level-0 Optical GT...")
        for i, p in enumerate(rms_pngs[:needed]):
            prof_name, dab_g, h_g = IHC_PROFILES[i % len(IHC_PROFILES)]
            with Image.open(p) as im:
                he_arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            hdab_arr = restain_he_to_hdab(he_arr, dab_gain=dab_g, h_gain=h_g)
            tid = f"ihc_{prof_name[:4].lower()}_{p.stem}"
            collected.append((tid, "Ruifrok_HDAB_Level0_GT", prof_name, hdab_arr))

    # Speichere kompakt als uint8 .npy (2000 x 256 x 256 x 3 = 393.2 MB)
    u8_stack = np.empty((len(collected), 256, 256, 3), dtype=np.uint8)
    rows: list[dict[str, object]] = []
    for idx, (tid, src_type, prof, tile_arr) in enumerate(collected):
        u8_stack[idx] = tile_arr
        dab_m, h_m, dab_pos, c_std = compute_ihc_stats(tile_arr)
        b8 = measure_8x8_blockiness(tile_arr)
        rows.append({
            "tile_id": tid,
            "tile_idx": idx,
            "track": "B3",
            "source_type": src_type,
            "subset_or_profile": prof,
            "dab_mean_od": round(dab_m, 4),
            "h_mean_od": round(h_m, 4),
            "dab_positive_fraction": round(dab_pos, 4),
            "chroma_std": round(c_std, 3),
            "blockiness_8x8": round(b8, 4),
            "is_fast_subset": False,
        })

    np.save(out_npy, u8_stack)

    # Balanciertes Fast-Subset (je 10 Kacheln mit hoechstem Chroma-Kontrast pro IHC-Profil = 40 Tiles)
    by_prof: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by_prof.setdefault(str(r["subset_or_profile"]), []).append(r)

    per_group = max(1, fast_subset_size // max(1, len(by_prof)))
    fast_total = 0
    for prof, grp in sorted(by_prof.items()):
        grp_sorted = sorted(grp, key=lambda x: float(x["chroma_std"]), reverse=True)
        for item in grp_sorted[:per_group]:
            item["is_fast_subset"] = True
            fast_total += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_B3_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    mean_b8 = float(np.mean([float(r["blockiness_8x8"]) for r in rows]))
    print(f"\n[Fertig] Track B3 vorbereitet ({len(rows)} verlustfreie IHC/DAB-Kacheln in {out_npy.stat().st_size / 1e6:.1f} MB).")
    print(f"  Mittlere 8x8-Blockiness (Lossless Verification): {mean_b8:.4f} (ideal ~1.00)")
    print(f"  Balanciertes Fast-Subset: {fast_total} Kacheln ueber {len(by_prof)} IHC-Profile")
    print(f"  Manifest gespeichert: {manifest_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Track B3 (IHC / DAB Color Stress Test, 2,000 lossless 256x256 tiles).")
    p.add_argument("--data-dir", type=Path, default=Path("anhir_data"))
    p.add_argument("--rms-tiles-dir", type=Path, default=Path("tiles_curated_v1/track_a_rms"))
    p.add_argument("--manifest", type=Path, default=Path("results/curated_v1/manifest_track_b3_ihc.csv"))
    p.add_argument("--num-tiles", type=int, default=2000)
    p.add_argument("--fast-subset-size", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_track_b3(args.data_dir, args.rms_tiles_dir, args.manifest, args.num_tiles, args.fast_subset_size)