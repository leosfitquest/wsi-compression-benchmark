"""
benchmark_compressai.py
-----------------------
Rate-distortion benchmark of pretrained CompressAI models on RGB tiles.

For each tile and quality level the script reports
    bpp_est   rate estimated from the model's likelihoods (-sum log2 p)
    bpp_real  rate of the actual bitstream produced by model.compress()
    PSNR / MS-SSIM of the tile decoded from that real bitstream
using the same metrics module as benchmark_conventional.py.

Device policy: the likelihood forward pass runs on the selected device (GPU if
available); real entropy coding always runs on CPU, because GPU float
non-determinism can desynchronise encoder and decoder (CompressAI itself warns
about this for autoregressive models). Timings are therefore CPU timings, the
same setting as Fischer et al. (Fig. 7).

Usage (run from this folder, or use VS Code "Run Python File"):
    python benchmark_compressai.py --synthetic 2
    python benchmark_compressai.py --tiles-dir path/to/tiles --models bmshj2018-hyperprior --qualities 1 2 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    import torch.nn.functional as F
    from compressai import zoo
except ImportError as exc:
    sys.exit(f"Missing dependency '{exc.name}'. Install with: pip install torch torchvision compressai")

from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb
from tiles import Tile, synthetic_tiles, tiles_from_dir

# Pretrained quality levels available in compressai.zoo
QUALITY_RANGES: dict[str, tuple[int, int]] = {
    "bmshj2018-factorized": (1, 8),
    "bmshj2018-hyperprior": (1, 8),
    "mbt2018-mean": (1, 8),
    "cheng2020-attn": (1, 6),
}
# g_a downsamples by 2^4, h_a by a further 2^2 -> spatial dims must be multiples of 64.
PAD_MULTIPLE = 64


@dataclass(frozen=True)
class ResultRow:
    tile_id: str
    model: str
    quality: int
    est_device: str
    bpp_est: float
    bpp_real: float | None
    psnr_db: float
    ms_ssim: float
    enc_s: float | None
    dec_s: float | None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available -> using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_model(name: str, quality: int, metric: str, device: torch.device) -> torch.nn.Module:
    """Download (first call only, cached by torch.hub) and prepare a pretrained model."""
    builder = getattr(zoo, name.replace("-", "_"))
    model = builder(quality=quality, metric=metric, pretrained=True, progress=True)
    model = model.eval().to(device)
    model.update()  # ensures the CDF tables needed by compress()/decompress() exist
    return model


def tile_to_tensor(tile: Tile) -> torch.Tensor:
    """uint8 (H, W, 3) -> float32 (1, 3, H, W) in [0, 1]."""
    return torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().div(255.0)


def tensor_to_tile(x: torch.Tensor) -> Tile:
    """float (1, 3, H, W) in [0, 1] -> uint8 (H, W, 3), i.e. the 8-bit image a viewer would show."""
    return x.squeeze(0).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def pad_to_multiple(x: torch.Tensor, multiple: int = PAD_MULTIPLE) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicate-pad bottom/right so H and W are multiples of `multiple` (224 -> 256)."""
    h, w = x.shape[-2:]
    pad_h, pad_w = (-h) % multiple, (-w) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (h, w)


@torch.inference_mode()
def estimate_rate(model: torch.nn.Module, x_padded: torch.Tensor, hw: tuple[int, int]) -> tuple[float, torch.Tensor]:
    """bpp from likelihoods: bits = -sum(log2 p) over all latents (y, and z if present)."""
    out = model(x_padded)
    total_bits = sum(float(-torch.log2(lik).sum()) for lik in out["likelihoods"].values())
    h, w = hw
    return total_bits / (h * w), out["x_hat"][..., :h, :w]


@torch.inference_mode()
def real_coding(model_cpu: torch.nn.Module, x_padded_cpu: torch.Tensor, hw: tuple[int, int]) -> tuple[int, torch.Tensor, float, float]:
    """Actual rANS bitstream: returns (bytes, decoded tensor, encode s, decode s)."""
    t0 = time.perf_counter()
    enc = model_cpu.compress(x_padded_cpu)
    t1 = time.perf_counter()
    dec = model_cpu.decompress(enc["strings"], enc["shape"])
    t2 = time.perf_counter()
    num_bytes = sum(len(s) for per_latent in enc["strings"] for s in per_latent)
    h, w = hw
    return num_bytes, dec["x_hat"][..., :h, :w], t1 - t0, t2 - t1


HEADER = (f"{'tile':<14} {'model':<22} {'q':>2} {'bpp_est':>8} {'bpp_real':>8} "
          f"{'PSNR':>7} {'MS-SSIM':>8} {'enc s':>7} {'dec s':>7}")


def _fmt_row(r: ResultRow) -> str:
    real = f"{r.bpp_real:>8.4f}" if r.bpp_real is not None else f"{'-':>8}"
    enc = f"{r.enc_s:>7.3f}" if r.enc_s is not None else f"{'-':>7}"
    dec = f"{r.dec_s:>7.3f}" if r.dec_s is not None else f"{'-':>7}"
    return (f"{r.tile_id:<14} {r.model:<22} {r.quality:>2} {r.bpp_est:>8.4f} {real} "
            f"{r.psnr_db:>7.2f} {r.ms_ssim:>8.4f} {enc} {dec}")


def run_benchmark(
    tiles: Iterable[tuple[str, Tile]],
    models: Sequence[str],
    qualities: Sequence[int],
    metric: str,
    device: torch.device,
    do_real_coding: bool,
) -> list[ResultRow]:
    cpu = torch.device("cpu")
    tile_list = list(tiles)  # each model is loaded once and applied to all tiles
    rows: list[ResultRow] = []
    print(f"estimation device: {device} | entropy coding device: cpu")
    print(HEADER)
    print("-" * len(HEADER))
    for name in models:
        lo, hi = QUALITY_RANGES[name]
        for q in qualities:
            if not lo <= q <= hi:
                print(f"[skip] {name} has no quality {q} (valid range {lo}-{hi})")
                continue
            est_model = load_model(name, q, metric, device)
            code_model = est_model if device.type == "cpu" else load_model(name, q, metric, cpu)
            for tile_id, tile in tile_list:
                x, hw = pad_to_multiple(tile_to_tensor(tile))
                bpp_est, x_hat = estimate_rate(est_model, x.to(device), hw)
                bpp_real = enc_s = dec_s = None
                if do_real_coding:
                    num_bytes, x_hat, enc_s, dec_s = real_coding(code_model, x, hw)
                    bpp_real = bits_per_pixel(num_bytes, *hw)
                rec = tensor_to_tile(x_hat)
                row = ResultRow(tile_id, name, q, device.type, bpp_est, bpp_real,
                                psnr_rgb(tile, rec), ms_ssim_rgb(tile, rec), enc_s, dec_s)
                rows.append(row)
                print(_fmt_row(row))
            del est_model, code_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def write_csv(rows: Sequence[ResultRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(ResultRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(f"\n{len(rows)} rows written to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CompressAI pretrained-model R-D benchmark on RGB tiles.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic H&E-like tiles (default 2)")
    src.add_argument("--tiles-dir", type=Path, help="folder of lossless PNG/TIFF tiles")
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--limit", type=int)
    p.add_argument("--tile-size", type=int, default=256, help="synthetic tile size (224 or 256)")
    p.add_argument("--models", nargs="+", choices=sorted(QUALITY_RANGES),
                   default=["bmshj2018-hyperprior", "cheng2020-attn"])
    p.add_argument("--qualities", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--metric", choices=["mse", "ms-ssim"], default="mse",
                   help="distortion the pretrained weights were optimised for")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--estimate-only", action="store_true", help="skip real entropy coding (faster)")
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results_compressai.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiles_dir:
        tiles = tiles_from_dir(args.tiles_dir, args.pattern, args.limit)
    else:
        tiles = synthetic_tiles(args.synthetic or 2, args.tile_size)
    rows = run_benchmark(tiles, args.models, args.qualities, args.metric,
                         resolve_device(args.device), not args.estimate_only)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
