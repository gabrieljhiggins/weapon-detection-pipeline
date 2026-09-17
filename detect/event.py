"""Annotated frames per person id. One suspect clip with boxes + a JSON proof log."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

EVENT_ROOT = Path("/home/gjh/weapon-detection-pipeline/data/events")
KEEP_S = 240.0
FPS = 8
W, H = 640, 360


def _resize(frame):
    out = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(out)


class Bank:
    def __init__(self):
        self.by_id = defaultdict(deque)
        self.last = {}

    def add(self, tid, cam, vis):
        if tid is None:
            return
        now = datetime.now().timestamp()
        if now - self.last.get((tid, cam), 0) < 0.08:
            return
        self.last[(tid, cam)] = now
        ok, buf = cv2.imencode(".jpg", _resize(vis), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        q = self.by_id[tid]
        q.append((now, cam, buf.tobytes()))
        while q and now - q[0][0] > KEEP_S:
            q.popleft()

    def absorb(self, src, dst):
        if src is None or dst is None or src == dst:
            return
        merged = list(self.by_id.get(dst, [])) + list(self.by_id.get(src, []))
        merged.sort(key=lambda x: x[0])
        self.by_id[dst] = deque(merged)
        self.by_id.pop(src, None)

    def encode(self, tid) -> Path | None:
        rows = list(self.by_id.get(tid, []))
        rows.sort(key=lambda x: x[0])
        if not rows:
            return None
        EVENT_ROOT.mkdir(parents=True, exist_ok=True)
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = EVENT_ROOT / ("event_%s_id%s.mp4" % (tag, tid))
        tmp = EVENT_ROOT / ("event_%s_id%s.tmp.mp4" % (tag, tid))
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "%dx%d" % (W, H), "-r", str(FPS),
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            str(tmp),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        stdin = proc.stdin
        if stdin is None:
            proc.kill()
            return None
        n = 0
        try:
            for _, _cam, blob in rows:
                frame = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                stdin.write(_resize(frame).tobytes())
                n += 1
        finally:
            stdin.close()
            proc.wait(timeout=30)
        if proc.returncode != 0 or n == 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            if tmp.exists():
                tmp.unlink()
            return None
        tmp.rename(path)
        return path


def write_log(tid, spans, clip: Path | None) -> Path | None:
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EVENT_ROOT / ("event_%s_id%s.json" % (tag, tid))
    rows = []
    for s in sorted(spans, key=lambda x: x["t0"]):
        rows.append({
            "cam": s["cam"],
            "t0": s["t0"].strftime("%H:%M:%S"),
            "t1": s["t1"].strftime("%H:%M:%S"),
            "armed": bool(s.get("armed")),
        })
    path.write_text(json.dumps({
        "id": tid,
        "clip": str(clip) if clip else None,
        "spans": rows,
    }, indent=2))
    return path
