"""
merge_plots.py
Fuegt rd_curves_tcga_precompressed.png und rd_curves_rms_groundtruth.png
vertikal zu einer zusammenfassenden Vergleichsgrafik zusammen.
"""
from PIL import Image
from pathlib import Path

tcga_img = Path("rd_curves_tcga_precompressed.png")
rms_img = Path("rd_curves_rms_groundtruth.png")

if tcga_img.exists() and rms_img.exists():
    im1 = Image.open(tcga_img)
    im2 = Image.open(rms_img)
    
    # Gleiche Breite sicherstellen, untereinander stapeln
    w = max(im1.width, im2.width)
    total_h = im1.height + im2.height
    
    merged = Image.new("RGB", (w, total_h), (255, 255, 255))
    merged.paste(im1, (0, 0))
    merged.paste(im2, (0, im1.height))
    
    out_path = Path("rd_curves_combined_comparison.png")
    merged.save(out_path, dpi=(300, 300))
    print(f"Kombinierter Vergleichs-Plot gespeichert: {out_path.resolve()}")