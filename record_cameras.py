#!/usr/bin/env python3
"""Continuously record RTSP cameras to disk (dataset collection).

Uses ffmpeg stream copy; splits into segments (see cameras.json).
Stop however you like — last partial segment may be corrupt and can be discarded.

  python3 record_cameras.py
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "cameras.json"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_cmd(cam: dict, out_dir: Path, segment_seconds: int) -> list[str]:
    name = cam["name"]
    rtsp = cam["rtsp"]
    cam_dir = out_dir / name
    cam_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(cam_dir / f"{name}_%Y%m%d_%H%M%S.mp4")

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "+genpts+discardcorrupt",
        "-use_wallclock_as_timestamps",
        "1",
        "-i",
        rtsp,
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-segment_format",
        "mp4",
        "-reset_timestamps",
        "1",
        "-strftime",
        "1",
        pattern,
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Record cameras continuously")
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to cameras.json")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 1

    cfg = load_config(cfg_path)
    cameras = cfg.get("cameras") or []
    if not cameras:
        print("No cameras in config.")
        return 1

    segment = int(cfg.get("segment_seconds", 60))
    out_dir = Path(cfg.get("output_dir", "recordings"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []

    def stop_all(*_args) -> None:
        print("\nStopping...")
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    print(f"Output:  {out_dir.resolve()}")
    print(f"Segment: {segment}s (~{max(segment // 60, 1)} min) files")
    print(f"Cameras: {len(cameras)}")
    print()

    for cam in cameras:
        name = cam.get("name", "cam")
        proc = subprocess.Popen(build_cmd(cam, out_dir, segment))
        procs.append(proc)
        time.sleep(2)
        if proc.poll() is None:
            print(f"  Camera {name} - LIVE")
        else:
            print(f"  Camera {name} - FAILED (exit {proc.returncode})")

    print("  Recording in progress...\n")

    while True:
        time.sleep(3)
        for i, cam in enumerate(cameras):
            if procs[i].poll() is not None:
                name = cam.get("name", "cam")
                print(f"  Camera {name} - DOWN — restarting...")
                procs[i] = subprocess.Popen(build_cmd(cam, out_dir, segment))
                time.sleep(2)
                if procs[i].poll() is None:
                    print(f"  Camera {name} - LIVE")


if __name__ == "__main__":
    raise SystemExit(main())