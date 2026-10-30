"""
benchmark_conventional.py
-------------------------
Rate-distortion benchmark of conventional codecs on RGB tiles.

    JPEG       Pillow (bundled libjpeg-turbo), chroma 4:4:4 and 4:2:0
    JPEG 2000  OpenJPEG CLI (opj_compress / opj_decompress); Pillow+OpenJPEG fallback
    JPEG XL    libjxl CLI (cjxl / djxl) only. Skipped if missing; never substituted.

For every tile, codec and target bit rate, the codec parameter (JPEG quality,
J2K compression ratio, JXL distance) is bisected until the compressed file lands
on the target bpp. The decoded tile is then scored with PSNR and MS-SSIM.

Usage (run from this folder, or use VS Code "Run Python File"):
    python benchmark_conventional.py --synthetic 2
    python benchmark_conventional.py --tiles-dir path/to/lossless_png_tiles --limit 50
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import PIL
from PIL import Image, features

from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb
from tiles import Tile, synthetic_tiles, tiles_from_dir

# Rate points (bpp). 0.12-0.75 follow the JPEG AI core set; 1.0/1.5 cover the
# near-lossless regime that Fischer et al. evaluate (up to ~1.75 bpp).
# Confirm against WG1N100600 before freezing the protocol.
DEFAULT_TARGETS_BPP: tuple[float, ...] = (0.12, 0.25, 0.50, 0.75, 1.00, 1.50)
RATE_TOLERANCE = 0.15  # accepted relative deviation of achieved vs. target bpp


class CodecUnavailableError(RuntimeError):
    """No working backend exists for this codec."""


class RateControlError(RuntimeError):
    """A backend silently ignored the requested rate."""


@dataclass(frozen=True)
class Encoded:
    num_bytes: int
    decoded: Tile
    backend: str
    enc_s: float
    dec_s: float


@dataclass(frozen=True)
class ResultRow:
    tile_id: str
    codec: str
    backend: str
    param_name: str
    param: float
    target_bpp: float
    bpp: float
    within_tolerance: bool
    psnr_db: float
    ms_ssim: float
    enc_s: float
    dec_s: float


def _run(cmd: Sequence[object]) -> None:
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


def _decode_file(path: Path) -> Tile:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def _decode_bytes(data: bytes) -> Tile:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Codecs: one tunable parameter that monotonically controls the rate
# ---------------------------------------------------------------------------

class Codec(ABC):
    name: str
    param_name: str
    param_lo: float
    param_hi: float
    integer_param: bool
    rate_increases_with_param: bool

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def encode_decode(self, tile: Tile, param: float) -> Encoded: ...


class JpegPillow(Codec):
    """Baseline JPEG (IJG quality scale 1-100), Huffman tables optimised."""

    param_name = "quality"
    param_lo, param_hi = 1.0, 100.0
    integer_param = True
    rate_increases_with_param = True
    _SUBSAMPLING = {"4:4:4": 0, "4:2:0": 2}

    def __init__(self, subsampling: str = "4:4:4") -> None:
        if subsampling not in self._SUBSAMPLING:
            raise ValueError(f"subsampling must be one of {list(self._SUBSAMPLING)}")
        self.subsampling = subsampling
        self.name = f"JPEG {subsampling}"

    def available(self) -> bool:
        return bool(features.check("jpg"))

    def encode_decode(self, tile: Tile, param: float) -> Encoded:
        buf = io.BytesIO()
        t0 = time.perf_counter()
        Image.fromarray(tile).save(
            buf, format="JPEG", quality=int(param),
            subsampling=self._SUBSAMPLING[self.subsampling], optimize=True,
        )
        t1 = time.perf_counter()
        data = buf.getvalue()
        decoded = _decode_bytes(data)
        t2 = time.perf_counter()
        return Encoded(len(data), decoded, "Pillow/libjpeg", t1 - t0, t2 - t1)


class Jpeg2000(Codec):
    """Irreversible 9/7 DWT + ICT colour transform; raw .j2k codestream (no JP2 box).

    Parameter = compression ratio w.r.t. raw 24-bit RGB, i.e. requested bpp = 24 / ratio.
    """

    name = "JPEG 2000"
    param_name = "ratio"
    param_lo, param_hi = 2.0, 1000.0
    integer_param = False
    rate_increases_with_param = False
    num_resolutions = 5

    def __init__(self) -> None:
        self._opj_c = shutil.which("opj_compress")
        self._opj_d = shutil.which("opj_decompress")
        self._pillow = bool(features.check("jpg_2000"))
        self._cli_failed = False

    def available(self) -> bool:
        return bool(self._opj_c and self._opj_d) or self._pillow

    def encode_decode(self, tile: Tile, param: float) -> Encoded:
        if self._opj_c and self._opj_d and not self._cli_failed:
            try:
                return self._encode_cli(tile, param)
            except subprocess.CalledProcessError as exc:
                self._cli_failed = True
                msg = (exc.stderr or b"").decode(errors="ignore").strip()[:200]
                print(f"[warn] opj_compress failed ({msg}); switching to Pillow backend.")
        if self._pillow:
            return self._encode_pillow(tile, param)
        raise CodecUnavailableError("No JPEG 2000 backend (install OpenJPEG CLI or Pillow with OpenJPEG).")

    def _encode_cli(self, tile: Tile, ratio: float) -> Encoded:
        with tempfile.TemporaryDirectory() as tmp:
            src, bitstream, dec = Path(tmp) / "in.ppm", Path(tmp) / "out.j2k", Path(tmp) / "dec.ppm"
            Image.fromarray(tile).save(src)
            t0 = time.perf_counter()
            # -I: irreversible 9/7 wavelet (the opj_compress default is the reversible 5/3)
            _run([self._opj_c, "-i", src, "-o", bitstream, "-r", f"{ratio:.4f}",
                  "-I", "-mct", "1", "-n", self.num_resolutions])
            t1 = time.perf_counter()
            _run([self._opj_d, "-i", bitstream, "-o", dec])
            t2 = time.perf_counter()
            return Encoded(bitstream.stat().st_size, _decode_file(dec), "OpenJPEG CLI", t1 - t0, t2 - t1)

    def _encode_pillow(self, tile: Tile, ratio: float) -> Encoded:
        buf = io.BytesIO()
        t0 = time.perf_counter()
        # Pillow silently ignores unknown options, so `mct`/`no_jp2` need a recent Pillow.
        Image.fromarray(tile).save(
            buf, format="JPEG2000", no_jp2=True, irreversible=True, mct=1,
            quality_mode="rates", quality_layers=[float(ratio)],
            num_resolutions=self.num_resolutions,
        )
        t1 = time.perf_counter()
        data = buf.getvalue()
        decoded = _decode_bytes(data)
        t2 = time.perf_counter()
        requested = 24.0 / ratio
        achieved = bits_per_pixel(len(data), tile.shape[0], tile.shape[1])
        if achieved > 2.0 * requested + 0.05:  # +0.05 bpp slack for the codestream header
            raise RateControlError(
                f"Pillow {PIL.__version__} ignored rate control: requested {requested:.3f} bpp, "
                f"got {achieved:.3f} bpp. Install the OpenJPEG CLI."
            )
        return Encoded(len(data), decoded, f"Pillow {PIL.__version__}/OpenJPEG", t1 - t0, t2 - t1)


class JpegXL(Codec):
    """JPEG XL via the libjxl reference CLI. Parameter = Butteraugli distance (-d)."""

    name = "JPEG XL"
    param_name = "distance"
    param_lo, param_hi = 0.05, 25.0
    integer_param = False
    rate_increases_with_param = False

    def __init__(self, effort: int = 7) -> None:
        self.effort = effort
        self._cjxl = shutil.which("cjxl")
        self._djxl = shutil.which("djxl")

    def available(self) -> bool:
        return bool(self._cjxl and self._djxl)

    def encode_decode(self, tile: Tile, param: float) -> Encoded:
        if not self.available():
            raise CodecUnavailableError("cjxl/djxl not found on PATH.")
        with tempfile.TemporaryDirectory() as tmp:
            src, bitstream, dec = Path(tmp) / "in.png", Path(tmp) / "out.jxl", Path(tmp) / "dec.png"
            Image.fromarray(tile).save(src, format="PNG")
            t0 = time.perf_counter()
            _run([self._cjxl, src, bitstream, "-d", f"{param:.4f}", "-e", self.effort])
            t1 = time.perf_counter()
            _run([self._djxl, bitstream, dec])
            t2 = time.perf_counter()
            return Encoded(bitstream.stat().st_size, _decode_file(dec), f"libjxl CLI (e{self.effort})", t1 - t0, t2 - t1)


# ---------------------------------------------------------------------------
# Rate matching and benchmark loop
# ---------------------------------------------------------------------------

def match_target_bpp(codec: Codec, tile: Tile, target_bpp: float, max_iter: int = 14) -> tuple[float, Encoded]:
    """Bisect the codec parameter so the file size lands as close as possible to target_bpp.

    Continuous parameters are bisected on a log scale; JPEG quality on the integer grid.
    """
    h, w = tile.shape[:2]
    cache: dict[float, Encoded] = {}

    def rate(p: float) -> float:
        if p not in cache:
            cache[p] = codec.encode_decode(tile, p)
        return bits_per_pixel(cache[p].num_bytes, h, w)

    lo, hi = codec.param_lo, codec.param_hi
    for _ in range(max_iter):
        if codec.integer_param:
            if hi - lo <= 1:
                break
            mid = float((int(lo) + int(hi)) // 2)
        else:
            mid = math.sqrt(lo * hi)
        r = rate(mid)
        if abs(r - target_bpp) <= 0.01 * target_bpp:
            break
        if (r > target_bpp) == codec.rate_increases_with_param:
            hi = mid
        else:
            lo = mid
    if codec.integer_param and hi - lo <= 1:
        rate(lo)
        rate(hi)
    best = min(cache, key=lambda p: abs(rate(p) - target_bpp))
    return best, cache[best]


def _fmt_row(r: ResultRow) -> str:
    ok = "yes" if r.within_tolerance else "NO"
    return (f"{r.tile_id:<14} {r.codec:<10} {r.param_name + '=' + format(r.param, 'g'):<16} "
            f"{r.target_bpp:>6.2f} {r.bpp:>7.3f} {ok:>4} {r.psnr_db:>7.2f} {r.ms_ssim:>8.4f} "
            f"{1e3 * r.enc_s:>7.1f} {1e3 * r.dec_s:>7.1f}  {r.backend}")


HEADER = (f"{'tile':<14} {'codec':<10} {'param':<16} {'target':>6} {'bpp':>7} {'ok':>4} "
          f"{'PSNR':>7} {'MS-SSIM':>8} {'enc ms':>7} {'dec ms':>7}  backend")


def run_benchmark(tiles: Iterable[tuple[str, Tile]], codecs: Sequence[Codec], targets: Sequence[float]) -> list[ResultRow]:
    rows: list[ResultRow] = []
    print(HEADER)
    print("-" * len(HEADER))
    for tile_id, tile in tiles:
        h, w = tile.shape[:2]
        for codec in codecs:
            for target in targets:
                try:
                    param, enc = match_target_bpp(codec, tile, target)
                except (CodecUnavailableError, RateControlError) as exc:
                    print(f"[skip] {codec.name} @ {target} bpp on {tile_id}: {exc}")
                    continue
                bpp = bits_per_pixel(enc.num_bytes, h, w)
                row = ResultRow(
                    tile_id=tile_id, codec=codec.name, backend=enc.backend,
                    param_name=codec.param_name, param=round(param, 4),
                    target_bpp=target, bpp=bpp,
                    within_tolerance=abs(bpp - target) / target <= RATE_TOLERANCE,
                    psnr_db=psnr_rgb(tile, enc.decoded), ms_ssim=ms_ssim_rgb(tile, enc.decoded),
                    enc_s=enc.enc_s, dec_s=enc.dec_s,
                )
                rows.append(row)
                print(_fmt_row(row))
    return rows


def write_csv(rows: Sequence[ResultRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(ResultRow)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    print(f"\n{len(rows)} rows written to {path}")


def build_codecs(jxl_effort: int) -> list[Codec]:
    active: list[Codec] = []
    for codec in (JpegPillow("4:4:4"), JpegPillow("4:2:0"), Jpeg2000(), JpegXL(jxl_effort)):
        if codec.available():
            active.append(codec)
        else:
            print(f"[info] {codec.name}: no backend found -> skipped (no substitute codec is used).")
    return active


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conventional codec R-D benchmark on RGB tiles.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic H&E-like tiles (default 2)")
    src.add_argument("--tiles-dir", type=Path, help="folder of lossless PNG/TIFF tiles")
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--limit", type=int)
    p.add_argument("--tile-size", type=int, default=256, help="synthetic tile size (224 or 256)")
    p.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS_BPP))
    p.add_argument("--jxl-effort", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path(__file__).with_name("results_conventional.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiles_dir:
        tiles = tiles_from_dir(args.tiles_dir, args.pattern, args.limit)
    else:
        tiles = synthetic_tiles(args.synthetic or 2, args.tile_size)
    codecs = build_codecs(args.jxl_effort)
    rows = run_benchmark(tiles, codecs, args.targets)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
