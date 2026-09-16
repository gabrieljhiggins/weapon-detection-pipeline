"""Always-on RTSP recorders. 30s segments under data/recordings/<cam>/."""

from __future__ import annotations

import subprocess
from pathlib import Path

REC_ROOT = Path("/home/gjh/weapon-detection-pipeline/data/recordings")


def start(cams: list[dict]) -> list[subprocess.Popen]:
    REC_ROOT.mkdir(parents=True, exist_ok=True)
    procs = []
    for cam in cams:
        d = REC_ROOT / cam["name"]
        d.mkdir(parents=True, exist_ok=True)
        out = str(d / (cam["name"] + "_%Y%m%d_%H%M%S.mp4"))
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "+genpts+discardcorrupt",
            "-i",
            cam["rtsp"],
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            "30",
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            out,
        ]
        procs.append(subprocess.Popen(cmd))
    return procs


def stop(procs: list[subprocess.Popen]):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
