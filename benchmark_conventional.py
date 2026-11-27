"""
benchmark_conventional.py
-------------------------
Rate-distortion benchmark of conventional codecs on RGB tiles.

    JPEG       Pillow (bundled libjpeg-turbo), chroma 4:4:4 and 4:2:0
    JPEG 2000  OpenJPEG CLI (opj_compress / opj_decompress); Pillow+OpenJPEG fallback
    JPEG XL    imagecodecs (libjxl) with fallback to libjxl CLI (cjxl / djxl)
    WebP       Pillow (bundled libwebp, YUV 4:2:0)
    AVIF       imagecodecs (libavif, explicitly pinned YUV 4:4:4)

For every tile, codec and target bit rate, the codec parameter (JPEG quality,
J2K compression ratio, JXL distance) is bisected until the compressed file lands
on the target bpp. The decoded tile is then scored with RGB-PSNR, Y-PSNR, and MS-SSIM.

Supports automatic checkpointing/resume and multi-core CPU execution (--workers).
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

from metrics import bits_per_pixel, ms_ssim_rgb, psnr_rgb, psnr_yuv_channels
from tiles import Tile, synthetic_tiles, tiles_from_dir

DEFAULT_TARGETS_BPP: tuple[float, ...] = (0.12, 0.25, 0.35, 0.50, 0.75, 1.00, 1.50)
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
    rate_mode: str
    setting_id: str
    backend: str
    param_name: str
    param: float
    target_bpp: float
    bpp: float
    within_tolerance: bool
    psnr_db: float
    psnr_y_db: float
    psnr_u_db: float
    psnr_v_db: float
    psnr_yuv_db: float
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
    """Irreversible 9/7 DWT + ICT colour transform; raw .j2k codestream (no JP2 box)."""

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
            _run([self._opj_c, "-i", src, "-o", bitstream, "-r", f"{ratio:.4f}",
                  "-I", "-mct", "1", "-n", self.num_resolutions])
            t1 = time.perf_counter()
            _run([self._opj_d, "-i", bitstream, "-o", dec])
            t2 = time.perf_counter()
            return Encoded(bitstream.stat().st_size, _decode_file(dec), "OpenJPEG CLI", t1 - t0, t2 - t1)

    def _encode_pillow(self, tile: Tile, ratio: float) -> Encoded:
        buf = io.BytesIO()
        t0 = time.perf_counter()
        Image.fromarray(tile).save(
            buf, format="JPEG2000", no_jp2=True, irreversible=True, mct=1,
            quality_mode="rates", quality_layers=[float(ratio)],
            num_resolutions=self.num_resolutions,
        )
        t1 = time.perf_counter()
        data = buf.getvalue()
        decoded = _decode_bytes(data)
        t2 = time.perf_counter()
        return Encoded(len(data), decoded, f"Pillow {PIL.__version__}/OpenJPEG", t1 - t0, t2 - t1)


class JpegXL(Codec):
    """JPEG XL via imagecodecs (libjxl) mit Fallback auf libjxl CLI."""

    name = "JPEG XL"
    param_name = "distance"
    param_lo, param_hi = 0.05, 25.0
    integer_param = False
    rate_increases_with_param = False

    def __init__(self, effort: int = 7) -> None:
        self.effort = effort
        self._backend = "none"
        try:
            import imagecodecs
            self._ic = imagecodecs
            self._backend = "imagecodecs"
        except ImportError:
            self._ic = None
            if shutil.which("cjxl") and shutil.which("djxl"):
                self._backend = "cli"

    def available(self) -> bool:
        return self._backend != "none"

    def encode_decode(self, tile: Tile, param: float) -> Encoded:
        if self._backend == "imagecodecs":
            t0 = time.perf_counter()
            encoded_bytes = self._ic.jpegxl_encode(
                tile,
                distance=float(param),
                effort=self.effort,
                planar=False,
                numthreads=1,
            )
            t1 = time.perf_counter()
            decoded = self._ic.jpegxl_decode(encoded_bytes, numthreads=1)
            t2 = time.perf_counter()
            return Encoded(len(encoded_bytes), decoded, f"imagecodecs/libjxl (e{self.effort})", t1 - t0, t2 - t1)

        elif self._backend == "cli":
            with tempfile.TemporaryDirectory() as tmp:
                src, bitstream, dec = Path(tmp) / "in.png", Path(tmp) / "out.jxl", Path(tmp) / "dec.png"
                Image.fromarray(tile).save(src, format="PNG")
                t0 = time.perf_counter()
                _run(["cjxl", src, bitstream, "-d", f"{param:.4f}", "-e", self.effort, "--num_threads=1"])
                t1 = time.perf_counter()
                _run(["djxl", bitstream, dec, "--num_threads=1"])
                t2 = time.perf_counter()
                return Encoded(bitstream.stat().st_size, _decode_file(dec), f"libjxl CLI (e{self.effort})", t1 - t0, t2 - t1)

        raise CodecUnavailableError("Weder imagecodecs noch cjxl/djxl CLI verfuegbar.")


class CodecWebP(Codec):
    """WebP Benchmark (Pillow integriertes libwebp, YUV 4:2:0)."""

    name = "WebP"
    param_name = "quality"
    param_lo, param_hi = 0.0, 100.0
    integer_param = True
    rate_increases_with_param = True

    def available(self) -> bool:
        return bool(features.check("webp"))

    def encode_decode(self, tile: Tile, param: float) -> Encoded:
        im = Image.fromarray(tile)
        buf = io.BytesIO()

        t0 = time.perf_counter()
        im.save(buf, format="WEBP", quality=int(param), method=4)
        t1 = time.perf_counter()

        data = buf.getvalue()
        buf.seek(0)
        with Image.open(buf) as dec_im:
            decoded = np.asarray(dec_im.convert("RGB"), dtype=np.uint8).copy()
        t2 = time.perf_counter()

        return Encoded(len(data), decoded, "Pillow/libwebp", t1 - t0, t2 - t1)


class CodecAVIF(Codec):
    """AVIF Implementation via imagecodecs (libavif) mit explizit gepinntem pixelformat (Default: YUV 4:4:4)."""

    name = "AVIF"
    param_name = "quality"
    param_lo, param_hi = 1.0, 100.0
    integer_param = True
    rate_increases_with_param = True

    def __init__(self, speed: int = 6, subsampling: str = "4:4:4") -> None:
        self.speed = speed
        self.subsampling = subsampling
        self._ic = None
        self._pixelformat = None
        try:
            import imagecodecs
            if hasattr(imagecodecs, "avif_encode") and hasattr(imagecodecs, "avif_decode"):
                self._ic = imagecodecs
                pf_enum = getattr(getattr(imagecodecs, "AVIF", None), "PIXEL_FORMAT", None)
                enum_key = {"4:4:4": "YUV444", "4:2:2": "YUV422", "4:2:0": "YUV420"}.get(subsampling, "YUV444")
                self._pixelformat = getattr(pf_enum, enum_key, f"yuv{subsampling.replace(':', '')}")
        except ImportError:
            pass

    def available(self) -> bool:
        return self._ic is not None

    def encode_decode(self, tile: Tile, param: float) -> Encoded:
        if self._ic is None:
            raise CodecUnavailableError("imagecodecs mit AVIF-Support ist nicht verfuegbar.")

        t0 = time.perf_counter()
        encoded_bytes = self._ic.avif_encode(
            tile,
            level=int(param),
            speed=self.speed,
            pixelformat=self._pixelformat,
            numthreads=1,
        )
        t1 = time.perf_counter()

        decoded = self._ic.avif_decode(encoded_bytes, numthreads=1)
        t2 = time.perf_counter()

        return Encoded(len(encoded_bytes), decoded, f"imagecodecs/libavif ({self.subsampling})", t1 - t0, t2 - t1)


# ---------------------------------------------------------------------------
# Fast rate matching (with boundary short-circuit & shared per-tile cache)
# ---------------------------------------------------------------------------

def match_target_bpp(
    codec: Codec,
    tile: Tile,
    target_bpp: float,
    max_iter: int = 9,
    cache: dict[float, Encoded] | None = None,
) -> tuple[float, Encoded]:
    """Bisect the codec parameter so the file size lands as close as possible to target_bpp."""
    h, w = tile.shape[:2]
    if cache is None:
        cache = {}

    def rate(p: float) -> float:
        if p not in cache:
            cache[p] = codec.encode_decode(tile, p)
        return bits_per_pixel(cache[p].num_bytes, h, w)

    # Short-circuit if target_bpp is below the codec's minimum achievable bitrate
    low_rate_param = codec.param_lo if codec.rate_increases_with_param else codec.param_hi
    if target_bpp <= rate(low_rate_param) * 1.01:
        return low_rate_param, cache[low_rate_param]

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
            f"{r.target_bpp:>6.2f} {r.bpp:>7.3f} {ok:>4} {r.psnr_db:>7.2f} {r.psnr_y_db:>7.2f} {r.psnr_yuv_db:>8.2f} "
            f"{r.ms_ssim:>8.4f} {1e3 * r.enc_s:>7.1f} {1e3 * r.dec_s:>7.1f}  {r.backend}")


HEADER = (f"{'tile':<14} {'codec':<10} {'param':<16} {'target':>6} {'bpp':>7} {'ok':>4} "
          f"{'PSNR':>7} {'Y-PSNR':>7} {'YUV-PSNR':>8} {'MS-SSIM':>8} {'enc ms':>7} {'dec ms':>7}  backend")


def _load_completed_keys(out_path: Path) -> set[tuple[str, str, float]]:
    done: set[tuple[str, str, float]] = set()
    if not out_path.exists() or out_path.stat().st_size == 0:
        return done
    try:
        with out_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["tile_id"], row["codec"], round(float(row["target_bpp"]), 4)))
    except Exception:
        pass
    return done


def _process_single_tile(
    args_tuple: tuple[str, Tile, int, list[tuple[str, float]]]
) -> list[ResultRow]:
    """Worker function for parallel tile processing across CPU cores."""
    tile_id, tile, jxl_effort, pending_tasks = args_tuple
    codecs_map = {c.name: c for c in build_codecs(jxl_effort, verbose=False)}
    caches: dict[str, dict[float, Encoded]] = {}
    h, w = tile.shape[:2]
    tile_rows: list[ResultRow] = []
    for codec_name, target in pending_tasks:
        codec = codecs_map.get(codec_name)
        if codec is None:
            continue
        tile_cache = caches.setdefault(codec_name, {})
        try:
            param, enc = match_target_bpp(codec, tile, target, cache=tile_cache)
        except (CodecUnavailableError, RateControlError):
            continue
        bpp = bits_per_pixel(enc.num_bytes, h, w)
        py, pu, pv, pyuv = psnr_yuv_channels(tile, enc.decoded)
        tile_rows.append(
            ResultRow(
                tile_id=tile_id, codec=codec.name, rate_mode="bisect",
                setting_id=f"target_{target:.2f}", backend=enc.backend,
                param_name=codec.param_name, param=round(param, 4),
                target_bpp=target, bpp=bpp,
                within_tolerance=abs(bpp - target) / target <= RATE_TOLERANCE,
                psnr_db=psnr_rgb(tile, enc.decoded),
                psnr_y_db=py, psnr_u_db=pu, psnr_v_db=pv, psnr_yuv_db=pyuv,
                ms_ssim=ms_ssim_rgb(tile, enc.decoded),
                enc_s=enc.enc_s, dec_s=enc.dec_s,
            )
        )
    return tile_rows


def run_benchmark(
    tiles: Iterable[tuple[str, Tile]],
    codecs: Sequence[Codec],
    targets: Sequence[float],
    out_path: Path | None = None,
    resume: bool = True,
    workers: int = 1,
    jxl_effort: int = 7,
) -> list[ResultRow]:
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

    print(HEADER)
    print("-" * len(HEADER))
    try:
        if workers <= 1:
            for tile_id, tile in tiles:
                h, w = tile.shape[:2]
                for codec in codecs:
                    tile_cache: dict[float, Encoded] = {}
                    for target in targets:
                        key = (tile_id, codec.name, round(float(target), 4))
                        if key in done_keys:
                            continue
                        try:
                            param, enc = match_target_bpp(codec, tile, target, cache=tile_cache)
                        except (CodecUnavailableError, RateControlError) as exc:
                            print(f"[skip] {codec.name} @ {target} bpp on {tile_id}: {exc}")
                            continue
                        bpp = bits_per_pixel(enc.num_bytes, h, w)
                        py, pu, pv, pyuv = psnr_yuv_channels(tile, enc.decoded)
                        row = ResultRow(
                            tile_id=tile_id, codec=codec.name, rate_mode="bisect",
                            setting_id=f"target_{target:.2f}", backend=enc.backend,
                            param_name=codec.param_name, param=round(param, 4),
                            target_bpp=target, bpp=bpp,
                            within_tolerance=abs(bpp - target) / target <= RATE_TOLERANCE,
                            psnr_db=psnr_rgb(tile, enc.decoded),
                            psnr_y_db=py, psnr_u_db=pu, psnr_v_db=pv, psnr_yuv_db=pyuv,
                            ms_ssim=ms_ssim_rgb(tile, enc.decoded),
                            enc_s=enc.enc_s, dec_s=enc.dec_s,
                        )
                        rows.append(row)
                        if writer is not None and csv_file is not None:
                            writer.writerow(asdict(row))
                            csv_file.flush()
                        print(_fmt_row(row))
        else:
            import itertools

            def _iter_work_items():
                for tile_id, tile in tiles:
                    pending = [
                        (c.name, float(t))
                        for c in codecs
                        for t in targets
                        if (tile_id, c.name, round(float(t), 4)) not in done_keys
                    ]
                    if pending:
                        yield (tile_id, tile, jxl_effort, pending)

            work_iter = _iter_work_items()
            chunk_size = max(workers * 4, 32)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                while True:
                    chunk = list(itertools.islice(work_iter, chunk_size))
                    if not chunk:
                        break
                    for tile_rows in pool.map(_process_single_tile, chunk):
                        for row in tile_rows:
                            rows.append(row)
                            if writer is not None and csv_file is not None:
                                writer.writerow(asdict(row))
                            print(_fmt_row(row))
                        if csv_file is not None:
                            csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    if out_path is not None:
        print(f"\n+{len(rows)} new rows written to {out_path} (total: {len(done_keys) + len(rows)})")
    return rows


def build_codecs(jxl_effort: int, verbose: bool = True) -> list[Codec]:
    active: list[Codec] = []
    candidates = [
        JpegPillow("4:4:4"),
        JpegPillow("4:2:0"),
        Jpeg2000(),
        JpegXL(jxl_effort),
        CodecWebP(),
        CodecAVIF(),
    ]
    for codec in candidates:
        if codec.available():
            active.append(codec)
        elif verbose:
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
    p.add_argument("--workers", type=int, default=1, help="number of parallel CPU worker processes (e.g. 8)")
    p.add_argument("--no-resume", action="store_true", help="overwrite output CSV from scratch")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "curated_v1" / "results_rms_conv.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.tiles_dir:
        tiles = tiles_from_dir(args.tiles_dir, args.pattern, args.limit)
    else:
        tiles = synthetic_tiles(args.synthetic or 2, args.tile_size)
    codecs = build_codecs(args.jxl_effort)
    run_benchmark(
        tiles, codecs, args.targets,
        out_path=args.out, resume=not args.no_resume,
        workers=args.workers, jxl_effort=args.jxl_effort,
    )


if __name__ == "__main__":
    main()