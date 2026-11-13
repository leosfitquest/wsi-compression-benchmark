"""
extract_dicom_tiles.py
Extrahiert native 256x256 Level-0 Kacheln aus einem verlustfreien DICOM WSI.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import openslide

def extract_lossless_tiles(
    dcm_path: Path | str,
    output_dir: Path | str,
    tile_size: int = 256,
    max_tiles: int = 10,
    background_threshold: float = 220.0
) -> None:
    dcm_path = Path(dcm_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[OpenSlide] Oeffne DICOM Slide: {dcm_path.name}")
    with openslide.OpenSlide(str(dcm_path)) as slide:
        w_l0, h_l0 = slide.dimensions
        print(f"Level-0 Abmessungen: {w_l0} x {h_l0} Pixel")
        
        # Pruefe Transfer Syntax / Properties
        comment = slide.properties.get("openslide.comment", "")
        vendor = slide.properties.get("openslide.vendor", "unbekannt")
        print(f"Vendor: {vendor} | Properties: {len(slide.properties)} Eintraege")

        extracted = 0
        # Schrittweite fest auf Kachelgroesse setzen (kein Overlap)
        step = tile_size
        
        # Iteration ueber das Gewebe
        for y in range(0, h_l0 - tile_size, step * 4):  # Subsampling der Positionen fuer Diversitaet
            for x in range(0, w_l0 - tile_size, step * 4):
                if extracted >= max_tiles:
                    break
                
                # Level-0 Kachel auslesen (Rueckgabe ist PIL.Image RGBA)
                tile_rgba = slide.read_region((x, y), level=0, size=(tile_size, tile_size))
                tile_rgb = tile_rgba.convert("RGB")
                tile_arr = np.asarray(tile_rgb)

                # Hintergrund-Filter: Reines Glas hat hohe Intensitaet (nahe 255)
                mean_intensity = float(tile_arr.mean())
                if mean_intensity > background_threshold or mean_intensity < 30.0:
                    continue  # Ueberspringe Hintergrund oder Artefakte/Scannerraender

                # Verlustfrei als PNG speichern
                tile_filename = f"{dcm_path.stem}_x{x}_y{y}_s{tile_size}.png"
                tile_rgb.save(out_dir / tile_filename, format="PNG")
                extracted += 1
                print(f"[{extracted}/{max_tiles}] Gespeichert: {tile_filename} (Mean: {mean_intensity:.1f})")
            
            if extracted >= max_tiles:
                break

    print(f"\nExtraktion abgeschlossen: {extracted} Kacheln in '{out_dir}' gespeichert.")

if __name__ == "__main__":
    from pathlib import Path
    
    # Sucht im neuen Ordner idc_rms_lossless nach der groessten Level-0 DICOM-Datei
    dcm_candidates = sorted(Path("idc_rms_lossless").rglob("*.dcm"))
    if not dcm_candidates:
        raise FileNotFoundError("Keine .dcm Datei in ./idc_rms_lossless gefunden.")
    
    dicom_file = max(dcm_candidates, key=lambda p: p.stat().st_size)
    print(f"Ausgewaehlte Level-0 DICOM-Instanz: {dicom_file.name} ({dicom_file.stat().st_size / 1e6:.1f} MB)")
    
    target_directory = Path("tiles_rms_lossless_256")
    extract_lossless_tiles(dicom_file, target_directory, tile_size=256, max_tiles=5)