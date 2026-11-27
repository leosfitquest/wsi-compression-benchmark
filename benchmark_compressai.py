"""
benchmark_compressai.py
-----------------------
Rate-distortion benchmark of pretrained CompressAI models on RGB tiles.

For each tile and quality level the script reports
    bpp_est   rate estimated from the model's likelihoods (-sum log2 p)
    bpp_real  rate of the actual bitstream produced by model.compress()
    PSNR / Y-PSNR / MS-SSIM of the tile decoded from that real bitstream
using the same metrics module as benchmark_conventional.py.

Supports automatic checkpointing/resume and optional filtering via --manifest
and --fast-subset-only for rapid CPU evaluation across patients.
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

from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb, psnr_yuv_channels
from tiles import Tile, synthetic_tiles, tiles_from_dir

QUALITY_RANGES: dict[str, tuple[int, int]] = {
    "bmshj2018-factorized": (1, 8),
    "bmshj2018-hyperprior": (1, 8),
    "mbt2018-mean": (1, 8),
    "cheng2020-attn": (1, 6),
}
PAD_MULTIPLE = 64


@dataclass(frozen=True)
class ResultRow:
    tile_id: str
    model: str
    rate_mode: str
    setting_id: str
    quality: int
    est_device: str
    bpp_est: float
    bpp_real: float | None
    psnr_db: float
    psnr_y_db: float
    psnr_u_db: float
    psnr_v_db: float
    psnr_yuv_db: float
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
    model.update()
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
def real_coding(model: torch.nn.Module, x_padded: torch.Tensor, hw: tuple[int, int]) -> tuple[int, torch.Tensor, float, float]:
    """Actual rANS bitstream on model device (CPU or CUDA): returns (bytes, decoded tensor on CPU, encode s, decode s)."""
    dev = next(model.parameters()).device
    x_dev = x_padded.to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    enc = model.compress(x_dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    dec = model.decompress(enc["strings"], enc["shape"])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t2 = time.perf_counter()
    num_bytes = sum(len(s) for per_latent in enc["strings"] for s in per_latent)
    h, w = hw
    return num_bytes, dec["x_hat"][..., :h, :w].cpu(), t1 - t0, t2 - t1


HEADER = (f"{'tile':<14} {'model':<22} {'q':>2} {'bpp_est':>8} {'bpp_real':>8} "
          f"{'PSNR':>7} {'Y-PSNR':>7} {'YUV-PSNR':>8} {'MS-SSIM':>8} {'enc s':>7} {'dec s':>7}")


def _fmt_row(r: ResultRow) -> str:
    real = f"{r.bpp_real:>8.4f}" if r.bpp_real is not None else f"{'-':>8}"
    enc = f"{r.enc_s:>7.3f}" if r.enc_s is not None else f"{'-':>7}"
    dec = f"{r.dec_s:>7.3f}" if r.dec_s is not None else f"{'-':>7}"
    return (f"{r.tile_id:<14} {r.model:<22} {r.quality:>2} {r.bpp_est:>8.4f} {real} "
            f"{r.psnr_db:>7.2f} {r.psnr_y_db:>7.2f} {r.psnr_yuv_db:>8.2f} {r.ms_ssim:>8.4f} {enc} {dec}")


def _load_completed_keys(out_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not out_path.exists() or out_path.stat().st_size == 0:
        return done
    expected_cols = [fl.name for fl in fields(ResultRow)]
    with out_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_cols:
            raise ValueError(
                f"CSV schema mismatch in {out_path}: found {reader.fieldnames}, expected {expected_cols}. "
                "Use a new --out path or pass --no-resume to overwrite."
            )
        for row in reader:
            try:
                done.add((row["tile_id"], row["model"], int(row["quality"])))
            except (KeyError, TypeError, ValueError):
                continue
    return done


def _load_allowed_tiles_from_manifest(manifest_path: Path, fast_only: bool) -> set[str]:
    allowed: set[str] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("track") != "A":
                continue
            if fast_only and str(row.get("is_fast_subset", "")).lower() != "true":
                continue
            allowed.add(row["tile_id"])
    return allowed


def run_benchmark(
    tiles: Iterable[tuple[str, Tile]],
    models: Sequence[str],
    qualities: Sequence[int],
    metric: str,
    device: torch.device,
    do_real_coding: bool,
    out_path: Path | None = None,
    resume: bool = True,
) -> list[ResultRow]:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    tile_list = list(tiles)
    rows: list[ResultRow] = []

    done_keys = _load_completed_keys(out_path) if (out_path and resume) else set()
    if done_keys and out_path is not None:
        print(f"[Resume] Found {len(done_keys)} completed entries in {out_path.name} -> skipping them.")

    csv_file = None
    writer = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not done_keys
        csv_file = out_path.open("a" if done_keys else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=[fl.name for fl in fields(ResultRow)])
        if write_header:
            writer.writeheader()
            csv_file.flush()

    print(f"estimation device: {device} | entropy coding device: {device} | tiles: {len(tile_list)}")
    print(HEADER)
    print("-" * len(HEADER))
    try:
        for name in models:
            lo, hi = QUALITY_RANGES[name]
            for q in qualities:
                if not lo <= q <= hi:
                    print(f"[skip] {name} has no quality {q} (valid range {lo}-{hi})")
                    continue
                pending_tiles = [(tid, t) for tid, t in tile_list if (tid, name, q) not in done_keys]
                if not pending_tiles:
                    continue
                est_model = load_model(name, q, metric, device)
                code_model = est_model  # Keep real_coding on active device (CPU/CUDA) for GPU acceleration on VUB IPC3
                for tile_id, tile in pending_tiles:
                    x, hw = pad_to_multiple(tile_to_tensor(tile))
                    bpp_est, x_hat = estimate_rate(est_model, x.to(device), hw)
                    bpp_real = enc_s = dec_s = None
                    if do_real_coding:
                        num_bytes, x_hat, enc_s, dec_s = real_coding(code_model, x, hw)
                        bpp_real = bits_per_pixel(num_bytes, *hw)
                    rec = tensor_to_tile(x_hat)
                    p_rgb = psnr_rgb(tile, rec)
                    if do_real_coding and p_rgb < 12.0:
                        raise RuntimeError(f"Potential GPU rANS desync on {tile_id} ({name} q={q}): RGB-PSNR={p_rgb:.2f} dB")
                    py, pu, pv, pyuv = psnr_yuv_channels(tile, rec)
                    row = ResultRow(
                        tile_id=tile_id, model=name, rate_mode="neural_q", setting_id=f"q_{q}",
                        quality=q, est_device=device.type, bpp_est=bpp_est, bpp_real=bpp_real,
                        psnr_db=p_rgb, psnr_y_db=py, psnr_u_db=pu, psnr_v_db=pv, psnr_yuv_db=pyuv,
                        ms_ssim=ms_ssim_rgb(tile, rec), enc_s=enc_s, dec_s=dec_s,
                    )
                    rows.append(row)
                    if writer is not None and csv_file is not None:
                        writer.writerow(asdict(row))
                        csv_file.flush()
                    print(_fmt_row(row))
                del est_model, code_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if csv_file is not None:
            csv_file.close()

    if out_path is not None:
        print(f"\n+{len(rows)} new rows written to {out_path} (total: {len(done_keys) + len(rows)})")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CompressAI pretrained-model R-D benchmark on RGB tiles.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic H&E-like tiles (default 2)")
    src.add_argument("--tiles-dir", type=Path, help="folder of lossless PNG/TIFF tiles")
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--limit", type=int)
    p.add_argument("--manifest", type=Path, help="optional manifest_track_a.csv to filter tiles")
    p.add_argument("--fast-subset-only", action="store_true", help="only run tiles with is_fast_subset=True in manifest")
    p.add_argument("--tile-size", type=int, default=256, help="synthetic tile size (224 or 256)")
    p.add_argument("--models", nargs="+", choices=sorted(QUALITY_RANGES),
                   default=["bmshj2018-hyperprior", "cheng2020-attn"])
    p.add_argument("--qualities", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--metric", choices=["mse", "ms-ssim"], default="mse",
                   help="distortion the pretrained weights were optimised for")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--estimate-only", action="store_true", help="skip real entropy coding (faster)")
    p.add_argument("--no-resume", action="store_true", help="overwrite output CSV from scratch")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "curated_v1" / "results_rms_ai.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiles_dir:
        tiles = list(tiles_from_dir(args.tiles_dir, args.pattern, None))
        if args.manifest and args.manifest.exists():
            allowed = _load_allowed_tiles_from_manifest(args.manifest, args.fast_subset_only)
            tiles = [(tid, t) for tid, t in tiles if tid in allowed]
        if args.limit:
            tiles = tiles[: args.limit]
    else:
        tiles = list(synthetic_tiles(args.synthetic or 2, args.tile_size))

    run_benchmark(
        tiles, args.models, args.qualities, args.metric,
        resolve_device(args.device), not args.estimate_only,
        out_path=args.out, resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()