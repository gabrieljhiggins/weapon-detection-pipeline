#!/usr/bin/env python3
"""Join all clips from one camera folder into one long MP4.

Cameras record in short clips to avoid losing footage if a clip is corrupted, primarily  due to manual stopping of the recording process. 
This code stitches all clips from one camera into one long MP4 for frame extraction.
This code is provided for reference only and documents the video merging process.
The full videos would subsequently be used for frame extraction, model training and evaluation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = ROOT / "recordings"


def list_clips(folder: Path) -> list[Path]:
    clips = sorted(
        [p for p in folder.glob("*.mp4") if p.is_file() and p.stat().st_size > 1024]
    )
    return clips


def stitch(folder: Path, out: Path) -> int:
    clips = list_clips(folder)
    if not clips:
        print(f"No clips in {folder}")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.with_suffix(".concat.txt")
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")

    print(f"{folder.name}: {len(clips)} clips → {out}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(out),
    ]
    result = subprocess.run(cmd)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"ffmpeg failed for {folder.name}")
        return result.returncode
    print(f"Saved {out}  ({out.stat().st_size / (1024 * 1024):.1f} MB)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Stitch camera clips into one video")
    p.add_argument("camera", nargs="?", help="cam1, cam2, cam3, or a folder path")
    p.add_argument("--all", action="store_true", help="Stitch every camera folder")
    p.add_argument("--recordings", default=str(DEFAULT_RECORDINGS))
    p.add_argument("--out", help="Output MP4 (single camera only)")
    args = p.parse_args()

    rec = Path(args.recordings)
    if args.all:
        folders = sorted([d for d in rec.iterdir() if d.is_dir()])
        if not folders:
            print(f"No camera folders in {rec}")
            return 1
        code = 0
        for folder in folders:
            out = rec / f"{folder.name}_full.mp4"
            code |= stitch(folder, out)
        return code

    if not args.camera:
        print("Pass a camera name (cam1) or --all")
        return 1

    cam = Path(args.camera)
    folder = cam if cam.is_dir() else rec / args.camera
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        return 1

    out = Path(args.out) if args.out else rec / f"{folder.name}_full.mp4"
    return stitch(folder, out)


if __name__ == "__main__":
    raise SystemExit(main())