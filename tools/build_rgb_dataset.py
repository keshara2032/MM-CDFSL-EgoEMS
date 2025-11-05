#!/usr/bin/env python3
"""
Parallel RGB frame extractor for MM-CDFSL layout.

Creates:
<output_root>/<dataset_name>/RGB_frames/<clip_id>/frame_0000000001.jpg
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
from multiprocessing import Pool, cpu_count
from functools import partial


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate RGB frames in MM-CDFSL structure (multiprocess).")
    p.add_argument("--video-list", required=True, type=Path,
                   help="Text file: one video path per line (absolute or relative).")
    p.add_argument("--output-root", required=True, type=Path,
                   help="Root directory that will contain the dataset folder.")
    p.add_argument("--dataset-name", required=True,
                   help="Name of the dataset directory to create under the output root.")
    p.add_argument("--frame-rate-stride", type=int, default=1,
                   help="Sample every Nth frame (default: 1 = all frames).")
    p.add_argument("--digits", type=int, default=10,
                   help="Zero-pad width for frame file names (default: 10).")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, existing frame folders will be overwritten.")
    p.add_argument("--max-procs", type=int, default=0,
                   help="Max parallel processes (default: use all logical cores).")
    p.add_argument("--jpg-quality", type=int, default=95,
                   help="JPEG quality [0–100] for saved frames (default: 95).")
    p.add_argument("--chunksize", type=int, default=1,
                   help="Pool.imap_unordered chunksize (default: 1).")
    return p.parse_args()


# --------------------------- IO ---------------------------

def read_video_paths(list_file: Path) -> List[Path]:
    if not list_file.exists():
        raise FileNotFoundError(f"Video list file not found: {list_file}")
    with list_file.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return [Path(ln) for ln in lines]


def prepare_clip_dir(clip_dir: Path, overwrite: bool) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for jpg in clip_dir.glob("*.jpg"):
            try:
                jpg.unlink()
            except Exception:
                pass


# ----------------------- Core extract ----------------------

def extract_frames_one(
    video_path: Path,
    clip_dir: Path,
    stride: int,
    digits: int,
    jpg_quality: int
) -> int:
    """Extract frames from a single video into clip_dir. Returns #frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_counter = 1
    sampled_frames = 0
    frame_index = 0

    # OpenCV imwrite JPEG params
    jpg_quality = int(max(0, min(100, jpg_quality)))
    imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % stride == 0:
            fn = f"frame_{frame_counter:0{digits}d}.jpg"
            target = clip_dir / fn
            if not cv2.imwrite(str(target), frame, imwrite_params):
                raise RuntimeError(f"Could not write frame to {target}")
            frame_counter += 1
            sampled_frames += 1

        frame_index += 1

    cap.release()

    if sampled_frames == 0:
        raise RuntimeError(f"No frames written for {video_path}")
    return sampled_frames


# ----------------------- Worker wrapper --------------------

def _worker(job: Tuple[str, str, int, int, int, bool]) -> Tuple[str, bool, str]:
    """
    job = (video_path, clip_dir, stride, digits, jpg_quality, overwrite)
    Returns (clip_id, ok, message)
    """
    # Tame oversubscription inside each process
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    video_path_s, clip_dir_s, stride, digits, jpg_quality, overwrite = job
    video_path = Path(video_path_s)
    clip_dir = Path(clip_dir_s)
    clip_id = clip_dir.name

    try:
        if not video_path.exists():
            return (clip_id, False, f"Missing video: {video_path}")

        prepare_clip_dir(clip_dir, overwrite)
        n = extract_frames_one(
            video_path=video_path,
            clip_dir=clip_dir,
            stride=stride,
            digits=digits,
            jpg_quality=jpg_quality
        )
        return (clip_id, True, f"{n} frames")
    except Exception as e:
        return (clip_id, False, f"{type(e).__name__}: {e}")


# --------------------------- Main --------------------------

def main() -> None:
    args = parse_args()

    video_paths = read_video_paths(args.video_list)
    if not video_paths:
        print("No video paths found in the provided list.", file=sys.stderr)
        sys.exit(1)

    dataset_root = args.output_root / args.dataset_name / "RGB_frames"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Build jobs
    jobs = []
    for vp in video_paths:
        if not vp.exists():
            print(f"[WARN] Skip missing video: {vp}", file=sys.stderr)
            continue
        clip_id = vp.stem
        clip_dir = dataset_root / clip_id
        jobs.append((
            str(vp),
            str(clip_dir),
            int(args.frame_rate_stride),
            int(args.digits),
            int(args.jpg_quality),
            bool(args.overwrite),
        ))

    if not jobs:
        print("No valid jobs to run.", file=sys.stderr)
        sys.exit(1)

    procs = args.max_procs if args.max_procs and args.max_procs > 0 else cpu_count()
    procs = max(1, min(procs, cpu_count()))
    print(f"[INFO] Launching pool with {procs} processes for {len(jobs)} videos…")

    # One video per task; maxtasksperchild=1 keeps memory stable for very long batches
    ok_count = 0
    fail_count = 0
    with Pool(processes=procs, maxtasksperchild=1) as pool:
        for clip_id, ok, msg in pool.imap_unordered(_worker, jobs, chunksize=args.chunksize):
            if ok:
                ok_count += 1
                print(f"[DONE] {clip_id}: {msg}")
            else:
                fail_count += 1
                print(f"[FAIL] {clip_id}: {msg}", file=sys.stderr)

    print(f"[SUMMARY] Ok: {ok_count}  Fail: {fail_count}  Total: {ok_count + fail_count}")


if __name__ == "__main__":
    main()
