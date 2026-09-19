"""One MP4 per camera, written from the same RTSP stream as detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

REC_ROOT = Path("/home/gjh/weapon-detection-pipeline/data/recordings")
FPS = 10.0


def start(cams: list[dict]) -> list[dict]:
    """Create output paths. The writer is opened on the first frame."""
    REC_ROOT.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    for cam in cams:
        d = REC_ROOT / cam["name"]
        d.mkdir(parents=True, exist_ok=True)
        jobs.append({
            "name": cam["name"],
            "path": d / ("%s_%s.mp4" % (cam["name"], tag)),
            "writer": None,
        })
    return jobs


def attach(jobs: list[dict], name: str, frame) -> None:
    """Write a frame to the MP4 file for the given camera name."""
    job = None
    for item in jobs:
        if item["name"] == name:
            job = item
            break
    if job is None or frame is None:
        return
    if job["writer"] is None:
        h, w = frame.shape[:2]
        job["writer"] = cv2.VideoWriter(
            str(job["path"]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (w, h),
        )
    writer = job["writer"]
    if writer is not None:
        writer.write(frame)


def stop(jobs: list) -> None:
    """Release all MP4 writers."""
    for job in jobs:
        writer = job.get("writer") if isinstance(job, dict) else None
        if writer is not None:
            writer.release()
            job["writer"] = None
