"""
metrics.py
----------
Rate and distortion metrics shared by every codec benchmark in this repository.

All distortion functions take uint8 RGB arrays of shape (H, W, 3) and follow
ISO/IEC JPEG AI CTC v8.0 / JVET CTC ITU-R BT.709 Full-Range (0..255) in float64
without intermediate uint8 rounding (peak 255.0, capped at 100.0 dB).

Also provides lossless 8-bit RGB <-> 10-bit Full-Range BT.709 YUV444p10le
conversion helpers for HM / VTM All-Intra reference encoders.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

U8Image = NDArray[np.uint8]
U16Array = NDArray[np.uint16]
F64Array = NDArray[np.float64]

# ITU-R BT.709 luma coefficients
BT709_KR: float = 0.2126
BT709_KG: float = 0.7152
BT709_KB: float = 0.0722
PSNR_CAP_DB: float = 100.0

# Wang, Simoncelli & Bovik (2003), "Multi-scale structural similarity for image quality assessment".
MS_SSIM_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

_LPIPS_CACHE: dict[tuple[str, str], Any] = {}


def _check_pair(ref: U8Image, rec: U8Image) -> None:
    if ref.shape != rec.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} vs {rec.shape}")
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB arrays, got {ref.shape}")
    if ref.dtype != np.uint8 or rec.dtype != np.uint8:
        raise TypeError("Expected uint8 arrays (8-bit RGB).")


def _mse_to_psnr(mse: float, peak: float = 255.0) -> float:
    if mse <= 0.0:
        return PSNR_CAP_DB
    return min(PSNR_CAP_DB, 10.0 * math.log10(peak**2 / mse))


def bits_per_pixel(num_bytes: int, height: int, width: int) -> float:
    """Rate: total compressed bits divided by the pixel count of the ORIGINAL tile."""
    return num_bytes * 8.0 / float(height * width)


def psnr_rgb(ref: U8Image, rec: U8Image) -> float:
    """PSNR in dB, MSE pooled over all three RGB channels, peak 255, capped at 100 dB."""
    _check_pair(ref, rec)
    mse = float(np.mean((ref.astype(np.float64) - rec.astype(np.float64)) ** 2))
    return _mse_to_psnr(mse)


def rgb_to_yuv_bt709_f64(img: U8Image) -> tuple[F64Array, F64Array, F64Array]:
    """ITU-R BT.709 Full-Range (0..255) conversion in float64 without intermediate rounding."""
    rgb = img.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = BT709_KR * r + BT709_KG * g + BT709_KB * b
    u = (0.5 / (1.0 - BT709_KB)) * (b - y) + 128.0
    v = (0.5 / (1.0 - BT709_KR)) * (r - y) + 128.0
    return y, u, v


def psnr_yuv_channels(ref: U8Image, rec: U8Image) -> tuple[float, float, float, float]:
    """Returns (Y-PSNR, U-PSNR, V-PSNR, 6:1:1 weighted YUV-PSNR) in BT.709 Full-Range."""
    _check_pair(ref, rec)
    y_ref, u_ref, v_ref = rgb_to_yuv_bt709_f64(ref)
    y_rec, u_rec, v_rec = rgb_to_yuv_bt709_f64(rec)
    py = _mse_to_psnr(float(np.mean((y_ref - y_rec) ** 2)))
    pu = _mse_to_psnr(float(np.mean((u_ref - u_rec) ** 2)))
    pv = _mse_to_psnr(float(np.mean((v_ref - v_rec) ** 2)))
    pyuv = (6.0 * py + pu + pv) / 8.0
    return py, pu, pv, pyuv


def psnr_y(ref: U8Image, rec: U8Image) -> float:
    """Luminance PSNR in dB (ITU-R BT.709 Full-Range Y-channel, capped at 100 dB)."""
    py, _, _, _ = psnr_yuv_channels(ref, rec)
    return py


def psnr_yuv(ref: U8Image, rec: U8Image) -> float:
    """Weighted 6:1:1 YUV-PSNR in dB (ITU-R BT.709 Full-Range, JVET / JPEG AI CTC)."""
    _, _, _, pyuv = psnr_yuv_channels(ref, rec)
    return pyuv


def psnr_chroma_bt709(ref: U8Image, rec: U8Image) -> float:
    """Pooled chrominance PSNR in dB over ITU-R BT.709 Full-Range U and V channels."""
    _check_pair(ref, rec)
    _, u_ref, v_ref = rgb_to_yuv_bt709_f64(ref)
    _, u_rec, v_rec = rgb_to_yuv_bt709_f64(rec)
    mse = 0.5 * (float(np.mean((u_ref - u_rec) ** 2)) + float(np.mean((v_ref - v_rec) ** 2)))
    return _mse_to_psnr(mse)


def rgb_to_yuv444p10le(img: U8Image) -> U16Array:
    """Convert uint8 RGB (H, W, 3) to planar uint16 BT.709 Full-Range 10-bit YUV444 (3, H, W) in [0, 1023]."""
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB (H, W, 3), got {img.shape} {img.dtype}")
    rgb = img.astype(np.float64) * (1023.0 / 255.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = BT709_KR * r + BT709_KG * g + BT709_KB * b
    u = (0.5 / (1.0 - BT709_KB)) * (b - y) + 512.0
    v = (0.5 / (1.0 - BT709_KR)) * (r - y) + 512.0
    yuv = np.stack([y, u, v], axis=0)
    return np.clip(np.rint(yuv), 0, 1023).astype(np.uint16)


def yuv444p10le_to_rgb(yuv: U16Array) -> U8Image:
    """Convert planar uint16 BT.709 Full-Range 10-bit YUV444 (3, H, W) back to uint8 RGB (H, W, 3)."""
    if yuv.ndim != 3 or yuv.shape[0] != 3:
        raise ValueError(f"Expected planar YUV (3, H, W), got {yuv.shape}")
    y = yuv[0].astype(np.float64)
    u = yuv[1].astype(np.float64) - 512.0
    v = yuv[2].astype(np.float64) - 512.0
    r = y + (2.0 * (1.0 - BT709_KR)) * v
    b = y + (2.0 * (1.0 - BT709_KB)) * u
    g = (y - BT709_KR * r - BT709_KB * b) / BT709_KG
    rgb = np.stack([r, g, b], axis=-1) * (255.0 / 1023.0)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


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
    """MS-SSIM (5 scales, 11x11 Gaussian window, sigma 1.5), averaged over R, G, B."""
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
                factors.append(max(cs_val, 0.0))
                x, y = _downsample_2x(x), _downsample_2x(y)
            else:
                factors.append(max(ssim_val, 0.0))
        per_channel.append(float(np.prod(np.asarray(factors) ** w)))
    return float(np.mean(per_channel))


def lpips_rgb(ref: U8Image, rec: U8Image, net: str = "alex", device: str = "cpu") -> float:
    """Learned Perceptual Image Patch Similarity (Zhang et al., CVPR 2018; lower is better)."""
    _check_pair(ref, rec)
    import torch
    import lpips  # type: ignore[import-untyped]

    key = (net, device)
    if key not in _LPIPS_CACHE:
        loss_fn = lpips.LPIPS(net=net, verbose=False).eval().to(device)
        _LPIPS_CACHE[key] = loss_fn
    loss_fn = _LPIPS_CACHE[key]

    t_ref = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1.0).to(device)
    t_rec = torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1.0).to(device)
    with torch.inference_mode():
        val = float(loss_fn(t_ref, t_rec).item())
    return max(0.0, val)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert psnr_rgb(ref, ref) == 100.0
    assert psnr_yuv_channels(ref, ref) == (100.0, 100.0, 100.0, 100.0)
    assert psnr_chroma_bt709(ref, ref) == 100.0
    assert abs(ms_ssim_rgb(ref, ref) - 1.0) < 1e-9
    rec_10bit = yuv444p10le_to_rgb(rgb_to_yuv444p10le(ref))
    assert np.array_equal(ref, rec_10bit), "10-bit BT.709 Full-Range roundtrip must be lossless!"
    noise = rng.integers(-5, 6, ref.shape)
    noisy = np.clip(ref.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    py, pu, pv, pyuv = psnr_yuv_channels(ref, noisy)
    print("identical tiles : RGB/Y/U/V/YUV-PSNR = 100.0 dB (capped), MS-SSIM = 1.0, 10-bit roundtrip = lossless (OK)")
    print(f"+/-5 noise      : RGB={psnr_rgb(ref, noisy):.2f} dB, Y={py:.2f}, U={pu:.2f}, V={pv:.2f}, YUV(6:1:1)={pyuv:.2f} dB, MS-SSIM={ms_ssim_rgb(ref, noisy):.4f}")
    print(f"2048 B @ 256x256: {bits_per_pixel(2048, 256, 256):.4f} bpp (expected 0.2500)")
