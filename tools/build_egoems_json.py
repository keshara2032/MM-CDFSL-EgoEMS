#!/usr/bin/env python3
"""
Build MM-CDFSL-style annotation JSON from pre-segmented ego clips.

Assumptions about filenames (examples):
  ms1_t1_ks11_67.823_74.825_ego.mp4
  ms1_cardiac_arrest_t0_ks11_65.205_70.523_ego.mp4
  ng3_cardiac_arrest_t11_ks11_81.297_83.901_ego.mp4

We extract:
  participant_id  := first token before the first underscore (e.g., "ms1", "ng3")
  action_idx      := digits after "ks" (e.g., ks11 -> 11)
  start_timestamp := second-from-last numeric token (seconds in original video)
  stop_timestamp  := last numeric token (seconds in original video)

Since these files are already clipped to the segment, we set:
  start_frame = 0
  stop_frame  = (num_frames - 1)   # computed via OpenCV on the clip itself

Fields we don’t have (verb/noun/narration/video_id) are set to reasonable placeholders.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

try:
    import cv2  # pip install opencv-python
except ImportError as e:
    print("ERROR: This script needs OpenCV (pip install opencv-python).", file=sys.stderr)
    raise

# --------- CONFIG ---------
VAL_ROOT = Path("/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/TimeSformer_Format/ego/val_root")
OUTPUT_JSON = Path("../cdfsl/target/egoems/ego_val_mm_cdfsl.json")
# make sure directory exists output
if not OUTPUT_JSON.parent.exists():
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


SPLIT_NAME = "val"
DESCRIPTION = "EgoEMS few-shot style JSON generated from pre-segmented clips"
VERSION = 1.0

# Glob pattern (recurses class folders like administer_shock_aed/*)
GLOB = "**/*.mp4"

# --------- HELPERS ---------
def secs_to_hhmmss(secs: float) -> str:
    """Convert 81.297 -> '00:01:21.297' (millisecond precision retained)."""
    whole = int(secs)
    ms = int(round((secs - whole) * 1000))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def get_video_length_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if frames <= 0:
        raise RuntimeError(f"Frame count unavailable for: {path}")
    return frames

def parse_from_filename(stem: str):
    """
    Returns (participant_id, action_idx, start_sec, end_sec).
    Robust to optional middle tokens like 'cardiac_arrest' and 't<number>'.

    Strategy:
      - participant_id = first token
      - action_idx from token containing 'ks\\d+'
      - start/end seconds from the last two numeric tokens before the final 'ego' token
    """
    toks = stem.split("_")

    # participant_id
    participant_id = toks[0]

    # action_idx from 'ksNN'
    action_idx = None
    for t in toks:
        m = re.fullmatch(r"ks(\d+)", t)
        if m:
            action_idx = int(m.group(1))
            break
    if action_idx is None:
        raise ValueError(f"Cannot find action_idx 'ksNN' in: {stem}")

    # find numeric tokens near the end (handle ..._<start>_<end>_ego)
    numeric = [t for t in toks if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    if len(numeric) < 2:
        # fallback: try to read the last 3 tokens like '81.297_83.901_ego'
        # but usually the two numeric tokens will be collected above
        raise ValueError(f"Cannot find start/end timestamps in: {stem}")

    start_sec = float(numeric[-2])
    end_sec = float(numeric[-1])

    return participant_id, action_idx, start_sec, end_sec

# --------- MAIN ---------
def main():
    clips = []
    uid = 0
    
    unique_action_ids = set()

    for mp4 in sorted(VAL_ROOT.glob(GLOB)):
        stem = mp4.stem  # e.g., ng3_cardiac_arrest_t11_ks11_81.297_83.901_ego
        participant_id, action_idx, start_sec, end_sec = parse_from_filename(stem)
        
        unique_action_ids.add(action_idx)

        # class name from parent dir (e.g., 'administer_shock_aed') — useful as narration placeholder
        class_name = mp4.parent.name

        # Frames from the actual (clipped) video
        total_frames = get_video_length_frames(mp4)
        start_frame = 0
        stop_frame = max(0, total_frames - 1)

        # Placeholders for fields you don't have
        narration = class_name.replace("_", " ")
        verb = ""
        noun = ""
        verb_label = -1
        noun_label = -1

        # Use file stem as a stable video_id if no original video id is available
        video_id = stem

        entry = {
            "uid": str(uid),
            "participant_id": participant_id,
            "video_id": video_id,
            "narration": narration,
            "start_timestamp": secs_to_hhmmss(start_sec),
            "stop_timestamp": secs_to_hhmmss(end_sec),
            "start_frame": str(start_frame),
            "stop_frame": str(stop_frame),
            "verb": verb,
            "noun": noun,
            "verb_label": str(verb_label),
            "noun_label": str(noun_label),
            "action_idx": int(action_idx),
            "path": str(mp4),  # extra convenience field (not in EPIC JSON, but handy)
        }
        clips.append(entry)
        uid += 1

    out = {
        "version": VERSION,
        "date": int(datetime.now().strftime("%y%m%d")),
        "description": DESCRIPTION,
        "split": SPLIT_NAME,
        "num_actions": len(unique_action_ids),
        "clips": clips,
    }
    
    print(f"Found {len(unique_action_ids)} unique action IDs.")
    print(unique_action_ids)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(clips)} clips -> {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
