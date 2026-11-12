"""
make_he_showcase.py
-------------------
Creates a visual H&E tissue comparison banner (he_showcase.png) for the README:
Original Lossless Ground-Truth vs. JPEG 4:4:4, JPEG 2000, and AVIF at 0.25 bpp.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from benchmark_conventional import CodecAVIF, Jpeg2000, JpegPillow, match_target_bpp
from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb
from tiles import tiles_from_dir

tiles = list(tiles_from_dir(Path("tiles_rms_lossless_256"), "*.png", limit=1))
if not tiles:
    raise FileNotFoundError("No PNG tiles found in tiles_rms_lossless_256.")

tile_id, tile = tiles[0]
h, w = tile.shape[:2]
target_bpp = 0.25

panels = [("Original (Lossless .90)", tile, "24.00 bpp | Ground Truth")]
codecs = [JpegPillow("4:4:4"), Jpeg2000(), CodecAVIF()]

for codec in codecs:
    if codec.available():
        _, enc = match_target_bpp(codec, tile, target_bpp)
        bpp = bits_per_pixel(enc.num_bytes, h, w)
        psnr = psnr_rgb(tile, enc.decoded)
        msssim = ms_ssim_rgb(tile, enc.decoded)
        panels.append((codec.name, enc.decoded, f"{bpp:.2f} bpp | {psnr:.1f} dB | MS-SSIM {msssim:.3f}"))

fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.8), dpi=250)
for ax, (title, img, subtitle) in zip(axes, panels):
    ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=16)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=8.5, color="#333333")
    ax.axis("off")

plt.suptitle("H&E Whole Slide Image Tile (256x256 px, Level-0) â€” Visual Comparison @ ~0.25 bpp", fontsize=12, fontweight="bold", y=1.08)
plt.tight_layout()

out_path = Path("he_showcase.png")
plt.savefig(out_path, dpi=250, bbox_inches="tight")
print(f"H&E showcase saved to: {out_path.resolve()}")