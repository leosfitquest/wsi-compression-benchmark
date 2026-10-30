"""
tiles.py
--------
Tile sources for the codec benchmarks. Tile extraction is deliberately kept
separate from encoding (benchmark_*.py) and scoring (metrics.py).

Every source yields (tile_id, tile) pairs; tile is a uint8 RGB array (H, W, 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from numpy.typing import NDArray
from PIL import Image

Tile = NDArray[np.uint8]

# Reference tiles must be lossless; a JPEG reference would re-introduce generation loss.
LOSSY_SUFFIXES = frozenset({".jpg", ".jpeg", ".jfif"})


def _smooth(a: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Separable Gaussian blur with reflect padding (numpy only)."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    h, w = a.shape
    p = np.pad(a, radius, mode="reflect")
    rows = np.zeros((h + 2 * radius, w))
    for i in range(g.size):
        rows += g[i] * p[:, i : i + w]
    out = np.zeros((h, w))
    for i in range(g.size):
        out += g[i] * rows[i : i + h, :]
    return out


def synthetic_he_tile(size: int = 256, seed: int = 0) -> Tile:
    """H&E-like test tile: pink stroma texture, white lumina, purple elliptical nuclei.

    Only for smoke-testing the pipeline. The R-D numbers it produces carry no
    scientific meaning.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)

    texture = _smooth(rng.normal(0.0, 1.0, (size, size)), sigma=6.0)
    texture = (texture - texture.min()) / (np.ptp(texture) + 1e-9)
    img = np.array([235.0, 170.0, 200.0])[None, None, :] - 45.0 * texture[..., None] * np.array([0.3, 1.0, 0.6])

    lumen = _smooth(rng.normal(0.0, 1.0, (size, size)), sigma=14.0)
    img[lumen > np.quantile(lumen, 0.9)] = [245.0, 240.0, 245.0]

    nuclei = np.zeros((size, size), dtype=bool)
    for _ in range(size * size // 900):
        cy, cx = rng.uniform(0, size, 2)
        ry, rx = rng.uniform(3.0, 8.0, 2)
        theta = rng.uniform(0.0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        nuclei |= (u / rx) ** 2 + (v / ry) ** 2 <= 1.0
    img[nuclei] = np.array([90.0, 50.0, 140.0]) + rng.normal(0.0, 12.0, (int(nuclei.sum()), 3))

    img = np.stack([_smooth(img[..., c], 0.8) for c in range(3)], axis=-1)  # optical blur
    img += rng.normal(0.0, 3.0, img.shape)  # sensor noise
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def synthetic_tiles(n: int, size: int = 256, seed: int = 0) -> Iterator[tuple[str, Tile]]:
    for i in range(n):
        yield f"synthetic_{i:03d}", synthetic_he_tile(size, seed + i)


def tiles_from_dir(directory: Path | str, pattern: str = "*.png", limit: int | None = None) -> Iterator[tuple[str, Tile]]:
    """Load pre-extracted LOSSLESS tiles (PNG/TIFF), e.g. RMS tiles exported via wsi_io.py."""
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    for path in paths[:limit]:
        if path.suffix.lower() in LOSSY_SUFFIXES:
            raise ValueError(f"{path.name}: JPEG reference tiles are not allowed (already lossy).")
        with Image.open(path) as im:
            yield path.stem, np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def tiles_from_wsi(
    slide_path: Path | str,
    coords_level0: Iterable[tuple[int, int]],
    size: int = 256,
    level: int = 0,
) -> Iterator[tuple[str, Tile]]:
    """Read tiles straight from a slide with OpenSlide.

    Adapter placeholder: replace the body with the equivalent call from your
    wsi_io.py so that the benchmark uses the validated extraction pipeline.
    Coordinates are in the level-0 reference frame (OpenSlide convention).
    """
    import openslide  # imported lazily: only needed for this source

    stem = Path(slide_path).stem
    with openslide.OpenSlide(str(slide_path)) as slide:
        for x, y in coords_level0:
            region = slide.read_region((x, y), level, (size, size)).convert("RGB")
            yield f"{stem}_x{x}_y{y}", np.asarray(region, dtype=np.uint8).copy()
