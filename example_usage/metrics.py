"""
metrics.py
----------
Rate and distortion metrics shared by every codec benchmark in this folder.

All functions take uint8 RGB arrays of shape (H, W, 3). Conventional and learned
codecs are therefore scored by exactly the same code, which is what makes their
rate-distortion curves comparable.

Depends on numpy only. Run `python metrics.py` for a quick self-test.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

U8Image = NDArray[np.uint8]
F64Array = NDArray[np.float64]

# Wang, Simoncelli & Bovik (2003), "Multi-scale structural similarity for image quality assessment".
MS_SSIM_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _check_pair(ref: U8Image, rec: U8Image) -> None:
    if ref.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {rec.shape}")
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB arrays, got {ref.shape}")
    if ref.dtype != np.uint8 or rec.dtype != np.uint8:
        raise TypeError("Expected uint8 arrays (8-bit RGB).")


def bits_per_pixel(num_bytes: int, height: int, width: int) -> float:
    """Rate: total compressed bits divided by the pixel count of the ORIGINAL tile."""
    return num_bytes * 8.0 / float(height * width)


def psnr_rgb(ref: U8Image, rec: U8Image) -> float:
    """PSNR in dB, MSE pooled over all three RGB channels, peak value 255."""
    _check_pair(ref, rec)
    mse = float(np.mean((ref.astype(np.float64) - rec.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def _gaussian_window(size: int, sigma: float) -> F64Array:
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    return g / g.sum()


def _filter_valid(img: F64Array, g: F64Array) -> F64Array:
    """Separable Gaussian filter without padding ('valid' mode, as in reference MS-SSIM)."""
    k = g.size
    h, w = img.shape
    rows = np.zeros((h, w - k + 1), dtype=np.float64)
    for i in range(k):
        rows += g[i] * img[:, i : i + w - k + 1]
    out = np.zeros((h - k + 1, w - k + 1), dtype=np.float64)
    for i in range(k):
        out += g[i] * rows[i : i + h - k + 1, :]
    return out


def _ssim_and_cs(x: F64Array, y: F64Array, g: F64Array, data_range: float = 255.0) -> tuple[float, float]:
    """Mean SSIM and mean contrast-structure term for one channel at one scale."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = _filter_valid(x, g)
    mu_y = _filter_valid(y, g)
    var_x = _filter_valid(x * x, g) - mu_x**2
    var_y = _filter_valid(y * y, g) - mu_y**2
    cov_xy = _filter_valid(x * y, g) - mu_x * mu_y
    cs_map = (2.0 * cov_xy + c2) / (var_x + var_y + c2)
    luminance = (2.0 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    return float(np.mean(luminance * cs_map)), float(np.mean(cs_map))


def _downsample_2x(img: F64Array) -> F64Array:
    h, w = (img.shape[0] // 2) * 2, (img.shape[1] // 2) * 2
    img = img[:h, :w]
    return 0.25 * (img[0::2, 0::2] + img[1::2, 0::2] + img[0::2, 1::2] + img[1::2, 1::2])


def ms_ssim_rgb(
    ref: U8Image,
    rec: U8Image,
    weights: tuple[float, ...] = MS_SSIM_WEIGHTS,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """MS-SSIM (5 scales, 11x11 Gaussian window, sigma 1.5), averaged over R, G, B.

    Follows the same convention as `pytorch_msssim.ms_ssim(..., data_range=255)`,
    the implementation used by CompressAI's evaluation scripts.
    Each side must be >= win_size * 2**(scales - 1) = 176 px (224 and 256 tiles are fine).
    """
    _check_pair(ref, rec)
    levels = len(weights)
    min_side = win_size * 2 ** (levels - 1)
    if min(ref.shape[:2]) < min_side:
        raise ValueError(f"MS-SSIM with {levels} scales needs tiles >= {min_side} px per side.")

    g = _gaussian_window(win_size, sigma)
    w = np.asarray(weights, dtype=np.float64)
    per_channel: list[float] = []
    for c in range(3):
        x = ref[..., c].astype(np.float64)
        y = rec[..., c].astype(np.float64)
        factors: list[float] = []
        for level in range(levels):
            ssim_val, cs_val = _ssim_and_cs(x, y, g)
            if level < levels - 1:
                factors.append(max(cs_val, 0.0))  # contrast-structure at coarser scales
                x, y = _downsample_2x(x), _downsample_2x(y)
            else:
                factors.append(max(ssim_val, 0.0))  # full SSIM (incl. luminance) at last scale
        per_channel.append(float(np.prod(np.asarray(factors) ** w)))
    return float(np.mean(per_channel))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert psnr_rgb(ref, ref) == math.inf
    assert abs(ms_ssim_rgb(ref, ref) - 1.0) < 1e-9
    noise = rng.integers(-5, 6, ref.shape)
    noisy = np.clip(ref.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    print("identical tiles : PSNR = inf, MS-SSIM = 1.0  (OK)")
    print(f"+/-5 noise      : PSNR = {psnr_rgb(ref, noisy):.2f} dB, MS-SSIM = {ms_ssim_rgb(ref, noisy):.4f}")
    print(f"2048 B @ 256x256: {bits_per_pixel(2048, 256, 256):.4f} bpp (expected 0.2500)")
