"""
complexity_tracker.py
---------------------
Misst die rechentechnische Komplexitaet (Computational Complexity) aller
konventionellen Codecs und neuronalen CompressAI-Modelle auf RGB-Kacheln:

    - Enkodier- und Dekodierzeit (ms) ohne Bisektions- und Profiler-Overhead
    - Durchsatz in Megapixel pro Sekunde (MPix/s)
    - Peak-Speicherallokation im RAM (MB) via Python tracemalloc
    - Modellgroesse (MParams) und Rechenoperationen (kMACs/px) via PyTorch Hooks

Aufruf im Ordner example_usage:
    python complexity_tracker.py --tiles-dir tiles_rms_lossless_256 --limit 3
"""

from __future__ import annotations

import argparse
import csv
import gc
import tracemalloc
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from benchmark_compressai import (
    load_model,
    pad_to_multiple,
    real_coding,
    tensor_to_tile,
    tile_to_tensor,
)
from benchmark_conventional import (
    Codec,
    CodecUnavailableError,
    RateControlError,
    build_codecs,
    match_target_bpp,
)
from metrics import bits_per_pixel, psnr_rgb
from tiles import Tile, synthetic_tiles, tiles_from_dir


@dataclass(frozen=True)
class ComplexityRow:
    codec_or_model: str
    category: str
    backend: str
    setting: str
    bpp: float
    psnr_db: float
    enc_ms: float
    dec_ms: float
    enc_mpix_s: float
    dec_mpix_s: float
    peak_ram_mb: float
    params_m: float | None
    kmacs_per_px: float | None


def count_macs_and_params(
    model: nn.Module, x_padded: torch.Tensor, orig_hw: tuple[int, int]
) -> tuple[float, float]:
    """Berechnet Parameter (in Mio.) und kMACs pro Original-Pixel via PyTorch-Hooks."""
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    macs_list: list[int] = []

    def conv_hook(module: nn.Module, inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            b, c_out, h_out, w_out = out.shape
            k_h, k_w = module.kernel_size
            c_in = module.in_channels
            groups = module.groups
            macs = b * c_out * h_out * w_out * (c_in // groups) * k_h * k_w
            macs_list.append(int(macs))
        elif isinstance(module, nn.Linear):
            macs = inp[0].numel() * module.out_features
            macs_list.append(int(macs))

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(m.register_forward_hook(conv_hook))

    with torch.inference_mode():
        model(x_padded)

    for h in hooks:
        h.remove()

    h_orig, w_orig = orig_hw
    kmacs_per_px = sum(macs_list) / (float(h_orig * w_orig) * 1000.0)
    return round(total_params, 3), round(kmacs_per_px, 2)


def measure_conventional(
    tiles: Sequence[tuple[str, Tile]],
    codecs: Sequence[Codec],
    target_bpp: float,
    repeats: int = 3,
) -> list[ComplexityRow]:
    rows: list[ComplexityRow] = []

    for codec in codecs:
        bpp_vals, psnr_vals, enc_vals, dec_vals, ram_vals = [], [], [], [], []
        last_param = 0.0
        backend_str = ""

        for tile_id, tile in tiles:
            h, w = tile.shape[:2]
            try:
                # 1. Optimalen Parameter fuer Ziel-Bitrate bestimmen
                best_param, _ = match_target_bpp(codec, tile, target_bpp)
            except (CodecUnavailableError, RateControlError) as exc:
                print(f"[skip] {codec.name} auf {tile_id}: {exc}")
                continue

            last_param = best_param

            # 2a. Reine Zeitmessung OHNE tracemalloc-Overhead
            t_enc_list, t_dec_list = [], []
            enc_res = None
            for _ in range(repeats):
                enc_res = codec.encode_decode(tile, best_param)
                t_enc_list.append(enc_res.enc_s)
                t_dec_list.append(enc_res.dec_s)

            # 2b. Separater Durchlauf fuer die RAM-Messung
            gc.collect()
            tracemalloc.start()
            _ = codec.encode_decode(tile, best_param)
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert enc_res is not None
            backend_str = enc_res.backend
            bpp_vals.append(bits_per_pixel(enc_res.num_bytes, h, w))
            psnr_vals.append(psnr_rgb(tile, enc_res.decoded))
            enc_vals.append(float(np.median(t_enc_list)))
            dec_vals.append(float(np.median(t_dec_list)))
            ram_vals.append(peak_bytes / (1024.0 * 1024.0))

        if not bpp_vals:
            continue

        mean_enc_s = float(np.mean(enc_vals))
        mean_dec_s = float(np.mean(dec_vals))
        h0, w0 = tiles[0][1].shape[:2]
        mpix = (h0 * w0) / 1e6

        row = ComplexityRow(
            codec_or_model=codec.name,
            category="conventional",
            backend=backend_str,
            setting=f"target={target_bpp:.2f}bpp ({codec.param_name}={last_param:g})",
            bpp=round(float(np.mean(bpp_vals)), 4),
            psnr_db=round(float(np.mean(psnr_vals)), 2),
            enc_ms=round(mean_enc_s * 1e3, 2),
            dec_ms=round(mean_dec_s * 1e3, 2),
            enc_mpix_s=round(mpix / max(mean_enc_s, 1e-9), 2),
            dec_mpix_s=round(mpix / max(mean_dec_s, 1e-9), 2),
            peak_ram_mb=round(float(np.max(ram_vals)), 2),
            params_m=None,
            kmacs_per_px=None,
        )
        rows.append(row)
        print(_fmt_row(row))

    return rows


def measure_compressai(
    tiles: Sequence[tuple[str, Tile]],
    models: Sequence[str],
    quality: int,
    metric: str = "mse",
) -> list[ComplexityRow]:
    rows: list[ComplexityRow] = []
    cpu = torch.device("cpu")

    for name in models:
        model = load_model(name, quality, metric, cpu)

        # MACs und Parameter auf der ersten Kachel bestimmen
        first_x, first_hw = pad_to_multiple(tile_to_tensor(tiles[0][1]))
        params_m, kmacs_px = count_macs_and_params(model, first_x, first_hw)

        bpp_vals, psnr_vals, enc_vals, dec_vals, ram_vals = [], [], [], [], []

        for idx, (_, tile) in enumerate(tiles):
            x, hw = pad_to_multiple(tile_to_tensor(tile))

            # 1. Reine Zeitmessung OHNE tracemalloc-Overhead
            num_bytes, x_hat, enc_s, dec_s = real_coding(model, x, hw)

            # 2. Separater RAM-Check auf der ersten Kachel
            if idx == 0:
                gc.collect()
                tracemalloc.start()
                _ = real_coding(model, x, hw)
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                ram_vals.append(peak_bytes / (1024.0 * 1024.0))

            rec = tensor_to_tile(x_hat)
            bpp_vals.append(bits_per_pixel(num_bytes, *hw))
            psnr_vals.append(psnr_rgb(tile, rec))
            enc_vals.append(enc_s)
            dec_vals.append(dec_s)

        mean_enc_s = float(np.mean(enc_vals))
        mean_dec_s = float(np.mean(dec_vals))
        h0, w0 = tiles[0][1].shape[:2]
        mpix = (h0 * w0) / 1e6

        row = ComplexityRow(
            codec_or_model=name,
            category="neural",
            backend="CompressAI/CPU (rANS)",
            setting=f"quality={quality}",
            bpp=round(float(np.mean(bpp_vals)), 4),
            psnr_db=round(float(np.mean(psnr_vals)), 2),
            enc_ms=round(mean_enc_s * 1e3, 2),
            dec_ms=round(mean_dec_s * 1e3, 2),
            enc_mpix_s=round(mpix / max(mean_enc_s, 1e-9), 3),
            dec_mpix_s=round(mpix / max(mean_dec_s, 1e-9), 3),
            peak_ram_mb=round(float(np.max(ram_vals)), 2),
            params_m=params_m,
            kmacs_per_px=kmacs_px,
        )
        rows.append(row)
        print(_fmt_row(row))
        del model

    return rows


HEADER = (
    f"{'Codec / Model':<22} {'bpp':>6} {'PSNR':>6} {'enc ms':>8} {'dec ms':>8} "
    f"{'enc MP/s':>9} {'dec MP/s':>9} {'RAM MB':>7} {'MParams':>8} {'kMAC/px':>8}"
)


def _fmt_row(r: ComplexityRow) -> str:
    pm = f"{r.params_m:>8.2f}" if r.params_m is not None else f"{'-':>8}"
    km = f"{r.kmacs_per_px:>8.1f}" if r.kmacs_per_px is not None else f"{'-':>8}"
    return (
        f"{r.codec_or_model:<22} {r.bpp:>6.3f} {r.psnr_db:>6.2f} "
        f"{r.enc_ms:>8.1f} {r.dec_ms:>8.1f} {r.enc_mpix_s:>9.2f} {r.dec_mpix_s:>9.2f} "
        f"{r.peak_ram_mb:>7.2f} {pm} {km}"
    )


def write_csv(rows: Sequence[ComplexityRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(ComplexityRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(f"\n{len(rows)} Zeilen gespeichert in: {path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Computational Complexity Benchmark (Conventional + CompressAI).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", type=int, metavar="N", help="N synthetische Kacheln nutzen (Default 2)")
    src.add_argument("--tiles-dir", type=Path, help="Ordner mit verlustfreien PNG-Kacheln")
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--limit", type=int, default=3, help="Anzahl der Kacheln fuer den Durchsatz-Mittelwert")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--target-bpp", type=float, default=0.36, help="Iso-Bitrate passend zu ai-quality=3 (~0.36 bpp)")
    p.add_argument("--ai-quality", type=int, default=3, help="Quality-Stufe fuer CompressAI (1-6)")
    p.add_argument("--models", nargs="+", default=["bmshj2018-hyperprior", "cheng2020-attn"])
    p.add_argument("--jxl-effort", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results_complexity.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiles_dir:
        tiles = list(tiles_from_dir(args.tiles_dir, args.pattern, args.limit))
    else:
        tiles = list(synthetic_tiles(args.synthetic or 2, args.tile_size))

    if not tiles:
        raise RuntimeError("Keine Kacheln zum Benchmarken gefunden.")

    print(f"Benchmarking Complexity auf {len(tiles)} Kachel(n) @ {tiles[0][1].shape[:2]} px...\n")
    print(HEADER)
    print("-" * len(HEADER))

    codecs = build_codecs(args.jxl_effort)
    rows: list[ComplexityRow] = []
    rows.extend(measure_conventional(tiles, codecs, args.target_bpp))
    rows.extend(measure_compressai(tiles, args.models, args.ai_quality))

    write_csv(rows, args.out)


if __name__ == "__main__":
    main()