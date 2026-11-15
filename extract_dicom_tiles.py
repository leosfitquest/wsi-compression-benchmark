"""
extract_dicom_tiles.py
Extrahiert native 256x256 Level-0 Kacheln aus verifiziert verlustfreien DICOM WSIs.
Prueft vorab per pydicom die Level-0 VOLUME-Instanz auf LossyImageCompression == '00'
und nutzt einen deterministischen Zufalls-Seed sowie einen Pixel-Gewebeanteil-Filter.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import openslide
import pydicom

LOSSLESS_TS = {
    "1.2.840.10008.1.2.4.90",  # JPEG 2000 Lossless
    "1.2.840.10008.1.2.1",     # Explicit VR Little Endian (Uncompressed)
}


def level0_lossless_volume(series_dir: Path) -> Path:
    """Gibt die Level-0 VOLUME-Instanz einer Serie zurueck; wirft RuntimeError, falls nicht verlustfrei."""
    vols = []
    for p in series_dir.glob("*.dcm"):
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        img_type = [str(x).upper() for x in getattr(ds, "ImageType", [])]
        if "VOLUME" in img_type:
            cols = int(getattr(ds, "TotalPixelMatrixColumns", 0))
            rows = int(getattr(ds, "TotalPixelMatrixRows", 0))
            vols.append((cols * rows, p, ds))
    if not vols:
        raise RuntimeError(f"Keine VOLUME-Instanz in {series_dir}")
    _, path, ds = max(vols, key=lambda t: t[0])
    ts = str(ds.file_meta.TransferSyntaxUID).strip()
    lossy = str(ds.get("LossyImageCompression", "")).strip()
    if ts not in LOSSLESS_TS or lossy != "00":
        raise RuntimeError(f"{path.name}: TS={ts}, Lossy={lossy!r} -> NICHT verlustfrei, Abbruch!")
    return path


def extract_lossless_tiles(
    dcm_path: Path | str,
    output_dir: Path | str,
    tile_size: int = 256,
    max_tiles: int = 10,
    background_threshold: float = 220.0,
    min_tissue_fraction: float = 0.70,
    seed: int = 42,
) -> None:
    dcm_path = Path(dcm_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OpenSlide] Oeffne verifiziertes DICOM VOLUME-Slide: {dcm_path.name}")
    with openslide.OpenSlide(str(dcm_path)) as slide:
        w_l0, h_l0 = slide.dimensions
        print(f"Level-0 Abmessungen: {w_l0} x {h_l0} Pixel")

        vendor = slide.properties.get("openslide.vendor", "unbekannt")
        print(f"Vendor: {vendor} | Properties: {len(slide.properties)} Eintraege")

        # Alle nicht-ueberlappenden Rasterpositionen erzeugen und mit festem Seed mischen,
        # damit die Kacheln ueber das gesamte Gewebe verteilt sind (statt nur in einer Zeile).
        step = tile_size * 4
        xs = range(0, max(1, w_l0 - tile_size), step)
        ys = range(0, max(1, h_l0 - tile_size), step)
        coords = [(x, y) for y in ys for x in xs]

        rng = np.random.default_rng(seed)
        rng.shuffle(coords)

        extracted = 0
        for x, y in coords:
            if extracted >= max_tiles:
                break

            # Level-0 Kachel auslesen (Rueckgabe ist PIL.Image RGBA)
            tile_rgba = slide.read_region((x, y), level=0, size=(tile_size, tile_size))
            tile_rgb = tile_rgba.convert("RGB")
            tile_arr = np.asarray(tile_rgb, dtype=np.uint8)

            # Pixelweiser Gewebe-Filter: Ein Pixel zaehlt als Gewebe, wenn seine mittlere
            # Intensitaet zwischen 30 (kein schwarzer Rand) und background_threshold (kein reines Glas) liegt.
            pixel_means = tile_arr.mean(axis=-1)
            tissue_mask = (pixel_means >= 30.0) & (pixel_means <= background_threshold)
            tissue_fraction = float(np.mean(tissue_mask))

            if tissue_fraction < min_tissue_fraction:
                continue  # Ueberspringe Kacheln mit zu hohem Glas-/Hintergrundanteil

            mean_intensity = float(pixel_means.mean())
            tile_filename = f"{dcm_path.stem}_x{x}_y{y}_s{tile_size}.png"
            tile_rgb.save(out_dir / tile_filename, format="PNG")
            extracted += 1
            print(
                f"[{extracted}/{max_tiles}] Gespeichert: {tile_filename} "
                f"(Tissue: {100.0 * tissue_fraction:.0f}%, Mean: {mean_intensity:.1f})"
            )

    print(f"\nExtraktion abgeschlossen: {extracted} Kacheln in '{out_dir}' gespeichert.")


if __name__ == "__main__":
    root_dir = Path("idc_rms_lossless")
    series_dirs = sorted({p.parent for p in root_dir.rglob("*.dcm")})
    if not series_dirs:
        raise FileNotFoundError("Keine .dcm Datei in ./idc_rms_lossless gefunden.")

    target_directory = Path("tiles_rms_lossless_256")
    for s_dir in series_dirs:
        dicom_file = level0_lossless_volume(s_dir)
        print(f"Verifizierte Level-0 VOLUME-Instanz: {dicom_file.name} ({dicom_file.stat().st_size / 1e6:.1f} MB)")
        extract_lossless_tiles(dicom_file, target_directory, tile_size=256, max_tiles=5)