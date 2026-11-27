"""
benchmark_video_intra.py
------------------------
All-Intra still-image benchmark wrapper for standard video reference codecs:
  - HM All-Intra  (HEVC Test Model, 4:4:4 10-bit via `encoder_intra_main_rext.cfg`)
  - VTM All-Intra (VVC Test Model,  4:4:4 10-bit via `encoder_intra_vtm.cfg`)

Design principles (ISO/IEC JPEG AI CTC v8.0 & JVET CTC compliant):
- Lossless 8-bit RGB <-> 10-bit Full-Range BT.709 planar YUV444 (`<u2` little-endian)
  via `metrics.py` (`rgb_to_yuv444p10le` / `yuv444p10le_to_rgb`) without FFmpeg.
- Fixed QP sweep (default: 17, 22, 27, 32, 37, 42), `rate_mode="fixed_qp"`,
  `setting_id=f"qp_{qp}"`, `--SEIDecodedPictureHash=0` (no 16-20 B SEI hash overhead).
- RAM-disk temporary directories (`/dev/shm` on Linux/Colab/IPC3) + bounded
  `ProcessPoolExecutor` chunking + schema-checked CSV checkpoint/resume.
- Optional `--b3-out`: directly scores any `IHC_DAB` / `anhir_*` tiles in memory
  using `evaluate_track_b3_ihc.B3ResultRow` so HM and VTM integrate seamlessly
  into Track B3 DAB ablation tables without encoding twice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from metrics import (
    bits_per_pixel,
    ms_ssim_rgb,
    psnr_chroma_bt709,
    psnr_rgb,
    psnr_yuv_channels,
    rgb_to_yuv444p10le,
    yuv444p10le_to_rgb,
)
from tiles import Tile, tiles_from_dir

DEFAULT_QPS: tuple[int, ...] = (17, 22, 27, 32, 37, 42)
CODEC_DISPLAY_NAMES: dict[str, str] = {
    "hm": "HM All-Intra",
    "vtm": "VTM All-Intra",
}


@dataclass(frozen=True)
class VideoIntraRow:
    tile_id: str
    codec: str
    rate_mode: str
    setting_id: str
    qp: int
    backend: str
    bpp: float
    psnr_db: float
    psnr_y_db: float
    psnr_u_db: float
    psnr_v_db: float
    psnr_yuv_db: float
    ms_ssim: float
    enc_s: float
    dec_s: float


def _ram_tmp_root() -> str | None:
    shm = Path("/dev/shm")
    return str(shm) if shm.is_dir() else None


def build_encode_decode_cmds(
    family: str,
    encoder_bin: Path,
    decoder_bin: Path,
    cfg_path: Path,
    in_yuv: Path,
    out_bin: Path,
    enc_rec_yuv: Path,
    dec_yuv: Path,
    width: int,
    height: int,
    qp: int,
    disable_lmcs: bool = False,
) -> tuple[list[str], list[str]]:
    """Constructs HM or VTM All-Intra 4:4:4 10-bit CLI commands."""
    fam = family.lower()
    enc_rec_target = "/dev/null" if os.name == "posix" else str(enc_rec_yuv)
    if fam == "hm":
        enc_cmd = [
            str(encoder_bin),
            "-c", str(cfg_path),
            "-i", str(in_yuv),
            "-b", str(out_bin),
            "-o", enc_rec_target,
            "-wdt", str(width),
            "-hgt", str(height),
            "-fr", "1",
            "-f", "1",
            "-ts", "1",
            "-q", str(qp),
            "--Profile=main-RExt",
            "--ConformanceWindowMode=1",
            "--InputChromaFormat=444",
            "--ChromaFormatIDC=444",
            "--InputBitDepth=10",
            "--InternalBitDepth=10",
            "--OutputBitDepth=10",
            "--CostMode=lossy",
            "--SEIDecodedPictureHash=0",
        ]
        dec_cmd = [
            str(decoder_bin),
            "-b", str(out_bin),
            "-o", str(dec_yuv),
            "-d", "10",
        ]
    elif fam == "vtm":
        enc_cmd = [
            str(encoder_bin),
            "-c", str(cfg_path),
            "-i", str(in_yuv),
            "-b", str(out_bin),
            "-o", enc_rec_target,
            "-wdt", str(width),
            "-hgt", str(height),
            "-fr", "1",
            "-f", "1",
            "-ts", "1",
            "-q", str(qp),
            "--Profile=auto",
            "--ConformanceWindowMode=1",
            "--InputChromaFormat=444",
            "--ChromaFormatIDC=444",
            "--InputBitDepth=10",
            "--InternalBitDepth=10",
            "--OutputBitDepth=10",
            "--SEIDecodedPictureHash=0",
        ]
        if disable_lmcs:
            enc_cmd.append("--LMCSEnable=0")
        dec_cmd = [
            str(decoder_bin),
            "-b", str(out_bin),
            "-o", str(dec_yuv),
            "-d", "10",
        ]
    else:
        raise ValueError(f"Unsupported codec family '{family}'. Expected 'hm' or 'vtm'.")
    return enc_cmd, dec_cmd


def _run_checked(cmd: Sequence[str]) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout[-800:]}\nSTDERR:\n{proc.stderr[-800:]}"
        )
    return dt


def encode_decode_tile_video_intra(
    tile: Tile,
    family: str,
    encoder_bin: Path,
    decoder_bin: Path,
    cfg_path: Path,
    qp: int,
    disable_lmcs: bool = False,
) -> tuple[int, Tile, float, float]:
    """Encodes & decodes one uint8 RGB tile through HM/VTM in 10-bit BT.709 Full-Range YUV444."""
    h, w = tile.shape[:2]
    expected_samples = 3 * h * w
    with tempfile.TemporaryDirectory(prefix=f"{family}_intra_", dir=_ram_tmp_root()) as td:
        tdir = Path(td)
        in_yuv = tdir / "in.yuv"
        out_bin = tdir / "out.bin"
        enc_rec_yuv = tdir / "enc_rec.yuv"
        dec_yuv = tdir / "dec.yuv"

        rgb_to_yuv444p10le(tile).astype("<u2").tofile(in_yuv)
        enc_cmd, dec_cmd = build_encode_decode_cmds(
            family, encoder_bin, decoder_bin, cfg_path,
            in_yuv, out_bin, enc_rec_yuv, dec_yuv, w, h, qp,
            disable_lmcs=disable_lmcs,
        )
        enc_s = _run_checked(enc_cmd)
        dec_s = _run_checked(dec_cmd)

        num_bytes = out_bin.stat().st_size
        raw_dec = np.fromfile(dec_yuv, dtype="<u2")
        if raw_dec.size != expected_samples:
            raise RuntimeError(
                f"Decoded YUV size mismatch for {family} qp={qp}: got {raw_dec.size} uint16 samples, expected {expected_samples}"
            )
        decoded_rgb = yuv444p10le_to_rgb(raw_dec.reshape(3, h, w))
        return num_bytes, decoded_rgb, enc_s, dec_s


def _load_completed_keys(out_path: Path, expected_cols: list[str], key_cols: tuple[str, str, str]) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not out_path.exists() or out_path.stat().st_size == 0:
        return done
    with out_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_cols:
            raise ValueError(
                f"CSV schema mismatch in {out_path}: found {reader.fieldnames}, expected {expected_cols}. "
                "Use a new output path or pass --no-resume."
            )
        k0, k1, k2 = key_cols
        for row in reader:
            try:
                done.add((str(row[k0]), str(row[k1]), str(row[k2])))
            except KeyError:
                continue
    return done


def _process_single_tile_worker(args: tuple) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tile_id, tile, family, encoder_bin, decoder_bin, cfg_path, backend_label, pending_qps, b3_profile, disable_lmcs = args
    h, w = tile.shape[:2]
    codec_name = CODEC_DISPLAY_NAMES[family]
    a_rows: list[dict[str, object]] = []
    b3_rows: list[dict[str, object]] = []

    dab_ref = None
    lab_ref = None
    b3_mod = None
    if b3_profile is not None:
        import evaluate_track_b3_ihc as b3_mod_imp
        b3_mod = b3_mod_imp
        dab_ref = b3_mod.extract_dab_od(tile)
        lab_ref = b3_mod._rgb_to_lab(tile)

    for qp in pending_qps:
        num_bytes, decoded, enc_s, dec_s = encode_decode_tile_video_intra(
            tile, family, encoder_bin, decoder_bin, cfg_path, int(qp), disable_lmcs=disable_lmcs
        )
        bpp = bits_per_pixel(num_bytes, h, w)
        p_rgb = psnr_rgb(tile, decoded)
        py, pu, pv, pyuv = psnr_yuv_channels(tile, decoded)
        ssim = ms_ssim_rgb(tile, decoded)
        sid = f"qp_{int(qp)}"

        a_row = VideoIntraRow(
            tile_id=tile_id,
            codec=codec_name,
            rate_mode="fixed_qp",
            setting_id=sid,
            qp=int(qp),
            backend=backend_label,
            bpp=float(bpp),
            psnr_db=float(p_rgb),
            psnr_y_db=float(py),
            psnr_u_db=float(pu),
            psnr_v_db=float(pv),
            psnr_yuv_db=float(pyuv),
            ms_ssim=float(ssim),
            enc_s=float(enc_s),
            dec_s=float(dec_s),
        )
        a_rows.append(asdict(a_row))

        if b3_profile is not None and b3_mod is not None and dab_ref is not None and lab_ref is not None:
            p_chr = psnr_chroma_bt709(tile, decoded)
            p_dab, d_dice = b3_mod.dab_psnr_and_dice(dab_ref, decoded)
            de_val = b3_mod.cie_delta_e(lab_ref, decoded)
            b3_row = b3_mod.B3ResultRow(
                tile_id=tile_id,
                profile=str(b3_profile),
                category="conventional",
                rate_mode="fixed_qp",
                setting_id=sid,
                codec_or_model=codec_name,
                chroma_mode=b3_mod.CHROMA_MODES.get(codec_name, "4:4:4 (Video Intra)"),
                setting=float(qp),
                target_bpp=round(float(bpp), 2),
                bpp=float(bpp),
                within_tolerance=True,
                psnr_rgb_db=float(p_rgb),
                psnr_y_db=float(py),
                psnr_u_db=float(pu),
                psnr_v_db=float(pv),
                psnr_yuv_db=float(pyuv),
                psnr_chroma_db=float(p_chr),
                psnr_dab_db=float(p_dab) if np.isfinite(p_dab) else float("nan"),
                dab_dice=float(d_dice) if np.isfinite(d_dice) else float("nan"),
                cie_delta_e=float(de_val),
                ms_ssim=float(ssim),
            )
            b3_rows.append(asdict(b3_row))

    return a_rows, b3_rows


def _load_b3_profiles(manifest_path: Path | None) -> dict[str, str]:
    profiles: dict[str, str] = {}
    if manifest_path is None or not manifest_path.exists():
        return profiles
    mf = pd.read_csv(manifest_path)
    if "stain" in mf.columns:
        ihc = mf[mf["stain"].astype(str) == "IHC_DAB"]
        for _, r in ihc.iterrows():
            profiles[str(r["tile_id"])] = str(r.get("stain_detail", "IHC_DAB"))
    elif "subset_or_profile" in mf.columns:
        for _, r in mf.iterrows():
            profiles[str(r["tile_id"])] = str(r["subset_or_profile"])
    return profiles


def run_video_intra_benchmark(
    tiles: Iterable[tuple[str, Tile]],
    family: str,
    encoder_bin: Path,
    decoder_bin: Path,
    cfg_path: Path,
    qps: Sequence[int],
    out_path: Path,
    b3_out_path: Path | None = None,
    manifest_path: Path | None = None,
    workers: int = 2,
    resume: bool = True,
    disable_lmcs: bool = False,
) -> None:
    fam = family.lower()
    codec_name = CODEC_DISPLAY_NAMES[fam]
    backend_label = f"{encoder_bin.name} ({cfg_path.name})"
    a_cols = [fl.name for fl in fields(VideoIntraRow)]

    done_a = _load_completed_keys(out_path, a_cols, ("tile_id", "codec", "setting_id")) if resume else set()
    if done_a:
        print(f"[Resume] Found {len(done_a)} completed rows in {out_path.name} -> skipping them.")

    b3_cols: list[str] | None = None
    done_b3: set[tuple[str, str, str]] = set()
    b3_profiles = _load_b3_profiles(manifest_path) if b3_out_path is not None else {}
    if b3_out_path is not None:
        from evaluate_track_b3_ihc import B3ResultRow
        b3_cols = [fl.name for fl in fields(B3ResultRow)]
        if resume:
            done_b3 = _load_completed_keys(b3_out_path, b3_cols, ("tile_id", "codec_or_model", "setting_id"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    a_mode = "a" if (resume and out_path.exists() and out_path.stat().st_size > 0) else "w"
    a_file = out_path.open(a_mode, newline="", encoding="utf-8")
    a_writer = csv.DictWriter(a_file, fieldnames=a_cols)
    if a_mode == "w":
        a_writer.writeheader()
        a_file.flush()

    b3_file = None
    b3_writer = None
    if b3_out_path is not None and b3_cols is not None:
        b3_out_path.parent.mkdir(parents=True, exist_ok=True)
        b3_mode = "a" if (resume and b3_out_path.exists() and b3_out_path.stat().st_size > 0) else "w"
        b3_file = b3_out_path.open(b3_mode, newline="", encoding="utf-8")
        b3_writer = csv.DictWriter(b3_file, fieldnames=b3_cols)
        if b3_mode == "w":
            b3_writer.writeheader()
            b3_file.flush()

    def _iter_work():
        for tile_id, tile in tiles:
            b3_prof = None
            if b3_out_path is not None:
                if tile_id in b3_profiles:
                    b3_prof = b3_profiles[tile_id]
                elif tile_id.lower().startswith("anhir_"):
                    b3_prof = "ANHIR_IHC"
            pending = [
                int(qp) for qp in qps
                if (tile_id, codec_name, f"qp_{int(qp)}") not in done_a
                or (b3_prof is not None and (tile_id, codec_name, f"qp_{int(qp)}") not in done_b3)
            ]
            if pending:
                yield (tile_id, tile, fam, encoder_bin, decoder_bin, cfg_path, backend_label, pending, b3_prof, disable_lmcs)

    print(f"[{codec_name}] encoder={encoder_bin} | cfg={cfg_path.name} | QPs={list(qps)} | workers={workers} | tmp={_ram_tmp_root() or 'system-tmp'}")
    written_a = 0
    written_b3 = 0
    try:
        work_iter = _iter_work()
        if workers <= 1:
            for item in work_iter:
                a_rows, b3_rows = _process_single_tile_worker(item)
                for r in a_rows:
                    key = (str(r["tile_id"]), str(r["codec"]), str(r["setting_id"]))
                    if key not in done_a:
                        a_writer.writerow(r)
                        done_a.add(key)
                        written_a += 1
                a_file.flush()
                if b3_writer is not None and b3_file is not None and b3_rows:
                    for r in b3_rows:
                        key = (str(r["tile_id"]), str(r["codec_or_model"]), str(r["setting_id"]))
                        if key not in done_b3:
                            b3_writer.writerow(r)
                            done_b3.add(key)
                            written_b3 += 1
                    b3_file.flush()
                if a_rows:
                    last = a_rows[-1]
                    print(f"  {last['tile_id']:<34} {codec_name:<14} qp={last['qp']:>2} bpp={last['bpp']:.3f} RGB={last['psnr_db']:.2f} YUV={last['psnr_yuv_db']:.2f} enc={last['enc_s']:.2f}s")
        else:
            chunk_size = max(workers * 4, 16)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                while True:
                    chunk = list(itertools.islice(work_iter, chunk_size))
                    if not chunk:
                        break
                    for a_rows, b3_rows in pool.map(_process_single_tile_worker, chunk):
                        for r in a_rows:
                            key = (str(r["tile_id"]), str(r["codec"]), str(r["setting_id"]))
                            if key not in done_a:
                                a_writer.writerow(r)
                                done_a.add(key)
                                written_a += 1
                        if b3_writer is not None and b3_file is not None and b3_rows:
                            for r in b3_rows:
                                key = (str(r["tile_id"]), str(r["codec_or_model"]), str(r["setting_id"]))
                                if key not in done_b3:
                                    b3_writer.writerow(r)
                                    done_b3.add(key)
                                    written_b3 += 1
                        if a_rows:
                            last = a_rows[-1]
                            print(f"  {last['tile_id']:<34} {codec_name:<14} qp={last['qp']:>2} bpp={last['bpp']:.3f} RGB={last['psnr_db']:.2f} YUV={last['psnr_yuv_db']:.2f} enc={last['enc_s']:.2f}s")
                    a_file.flush()
                    if b3_file is not None:
                        b3_file.flush()
    finally:
        a_file.close()
        if b3_file is not None:
            b3_file.close()

    print(f"\n[{codec_name}] Done: +{written_a} Track-A rows -> {out_path.resolve()}")
    if b3_out_path is not None:
        print(f"[{codec_name}] Done: +{written_b3} Track-B3 IHC rows -> {b3_out_path.resolve()}")


def _run_self_test() -> None:
    rng = np.random.default_rng(7)
    tile = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory(dir=_ram_tmp_root()) as td:
        p = Path(td) / "roundtrip.yuv"
        rgb_to_yuv444p10le(tile).astype("<u2").tofile(p)
        assert p.stat().st_size == 3 * 256 * 256 * 2
        rec = yuv444p10le_to_rgb(np.fromfile(p, dtype="<u2").reshape(3, 256, 256))
        assert np.array_equal(tile, rec), "10-bit little-endian file roundtrip mismatch!"

    for fam in ("hm", "vtm"):
        ecmd, dcmd = build_encode_decode_cmds(
            fam, Path(f"/bin/{fam}_enc"), Path(f"/bin/{fam}_dec"), Path(f"/cfg/{fam}.cfg"),
            Path("/tmp/in.yuv"), Path("/tmp/out.bin"), Path("/tmp/rec.yuv"), Path("/tmp/dec.yuv"),
            256, 256, 27,
        )
        assert "--InputChromaFormat=444" in ecmd and "--SEIDecodedPictureHash=0" in ecmd
        assert "--ConformanceWindowMode=1" in ecmd and "-ts" in ecmd
        assert "-d" in dcmd and "10" in dcmd
    print("[SELF-TEST OK] 10-bit <u2 file roundtrip lossless, HM/VTM CLI flags verified.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HM / VTM All-Intra 4:4:4 10-bit WSI benchmark wrapper.")
    p.add_argument("--self-test", action="store_true", help="Run local YUV10le I/O and CLI flag self-test")
    p.add_argument("--codec", choices=["hm", "vtm"], help="Reference codec family (hm or vtm)")
    p.add_argument("--encoder-bin", type=Path, help="Path to TAppEncoderStatic (HM) or EncoderApp (VTM)")
    p.add_argument("--decoder-bin", type=Path, help="Path to TAppDecoderStatic (HM) or DecoderApp (VTM)")
    p.add_argument("--cfg", type=Path, help="Path to encoder_intra_main_rext.cfg (HM) or encoder_intra_vtm.cfg (VTM)")
    p.add_argument("--tiles-dir", type=Path, help="Directory of lossless 256x256 PNG tiles")
    p.add_argument("--pattern", default="*.png")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--qps", type=int, nargs="+", default=list(DEFAULT_QPS))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, help="Output Track-A CSV path")
    p.add_argument("--b3-out", type=Path, default=None, help="Optional Track-B3 CSV path for IHC_DAB / anhir_* tiles")
    p.add_argument("--manifest", type=Path, default=None, help="Optional manifest CSV to filter tiles and identify IHC_DAB profiles")
    p.add_argument("--fast-only", action="store_true", help="With --manifest, restrict to rows with is_fast_subset=True")
    p.add_argument("--disable-lmcs", action="store_true", help="Pass --LMCSEnable=0 for older VTM builds on 4:4:4")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        _run_self_test()
    else:
        missing = [k for k in ("codec", "encoder_bin", "decoder_bin", "cfg", "tiles_dir", "out") if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"Missing required arguments: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        allowed_ids: set[str] | None = None
        if args.manifest is not None:
            mf = pd.read_csv(args.manifest)
            if args.fast_only and "is_fast_subset" in mf.columns:
                mf = mf[mf["is_fast_subset"].astype(str).str.lower() == "true"]
            allowed_ids = set(mf["tile_id"].astype(str))
            print(f"[Manifest] Filtering tiles to {len(allowed_ids)} IDs from {args.manifest.name}")
        raw_tiles = tiles_from_dir(args.tiles_dir, args.pattern, limit=None if allowed_ids else args.limit)
        if allowed_ids is not None:
            tiles = ((tid, t) for tid, t in raw_tiles if tid in allowed_ids)
            if args.limit is not None:
                tiles = itertools.islice(tiles, args.limit)
        else:
            tiles = raw_tiles
        run_video_intra_benchmark(
            tiles=tiles,
            family=args.codec,
            encoder_bin=args.encoder_bin,
            decoder_bin=args.decoder_bin,
            cfg_path=args.cfg,
            qps=args.qps,
            out_path=args.out,
            b3_out_path=args.b3_out,
            manifest_path=args.manifest,
            workers=args.workers,
            resume=not args.no_resume,
            disable_lmcs=args.disable_lmcs,
        )
