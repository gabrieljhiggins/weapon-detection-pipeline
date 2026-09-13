#!/usr/bin/env python3
"""Extract frames from stitched *_full.mp4 videos.

  python3 extract_frames.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def video_duration_sec(path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def extract_one(video: Path, name: str, out_root: Path, fps: float, quality: int) -> int:
    cam_dir = out_root / name
    cam_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(cam_dir / f"{name}_%06d.jpg")

    duration = video_duration_sec(video)
    expected = int(duration * fps) if duration else 0
    print(f"{name}: {video}  ~{duration/60:.1f} min  →  ~{expected} frames @ {fps} fps")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(quality),
        pattern,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ffmpeg failed on {video}")
        return 1

    frames = sorted(cam_dir.glob(f"{name}_*.jpg"))
    meta = []
    for i, fp in enumerate(frames, start=1):
        t = (i - 1) / fps
        mm, ss = divmod(t, 60)
        hh, mm = divmod(mm, 60)
        meta.append(
            {
                "file": fp.name,
                "index": i,
                "time_sec": round(t, 3),
                "wall_time": f"{int(hh):02d}:{int(mm):02d}:{ss:06.3f}",
                "source": video.name,
            }
        )

    with open(cam_dir / "metadata.json", "w") as f:
        json.dump(
            {"fps": fps, "source": str(video), "count": len(frames), "frames": meta},
            f,
            indent=2,
        )

    print(f"  saved {len(frames)} frames → {cam_dir}")
    return 0


def find_full_videos(search_roots: list[Path]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*_full.mp4")):
            name = p.name.replace("_full.mp4", "")
            if name not in seen:
                found.append((p, name))
                seen.add(name)
        for p in sorted(root.glob("*/*_full.mp4")):
            name = p.name.replace("_full.mp4", "")
            if name not in seen:
                found.append((p, name))
                seen.add(name)
    return found


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("frames"))
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--quality", type=int, default=3)
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    search = [args.input_dir] if args.input_dir else [root, root / "recordings"]
    jobs = find_full_videos(search)
    if not jobs:
        print("No *_full.mp4 files found.")
        print("Looked in:", ", ".join(str(s) for s in search))
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir.resolve()}")
    print(f"FPS:    {args.fps}\n")

    code = 0
    for video, name in jobs:
        code |= extract_one(video, name, args.output_dir, args.fps, args.quality)

    print("\nDone. Frames are in:", args.output_dir.resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())