"""Cut suspect intervals from always-on recordings into one mp4."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

REC_ROOT = Path("/home/gjh/weapon-detection-pipeline/data/recordings")
EVENT_ROOT = Path("/home/gjh/weapon-detection-pipeline/data/events")
PAD = 2.0


def _parse_start(path: Path):
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return datetime.strptime(parts[-2] + parts[-1], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _files(cam: str):
    d = REC_ROOT / cam
    if not d.exists():
        return []
    rows = []
    for p in sorted(d.glob("*.mp4")):
        t0 = _parse_start(p)
        if t0:
            rows.append((t0, p))
    return rows


def _extract(src: Path, ss: float, dur: float, dest: Path) -> bool:
    if dur <= 0.05:
        return False
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "%.2f" % max(0, ss),
        "-i",
        str(src),
        "-t",
        "%.2f" % dur,
        "-c",
        "copy",
        "-y",
        str(dest),
    ]
    return subprocess.run(cmd).returncode == 0 and dest.exists() and dest.stat().st_size > 1000


def stitch(tid, spans, tag: str):
    EVENT_ROOT.mkdir(parents=True, exist_ok=True)
    work = EVENT_ROOT / ("_work_%s" % tag)
    work.mkdir(parents=True, exist_ok=True)
    bits = []
    n = 0
    for s in sorted(spans, key=lambda x: x["t0"]):
        cam = s["cam"]
        a = s["t0"] - timedelta(seconds=PAD)
        b = s["t1"] + timedelta(seconds=PAD)
        for t0, path in _files(cam):
            t1 = t0 + timedelta(seconds=35)
            if t1 < a or t0 > b:
                continue
            ss = max(0.0, (a - t0).total_seconds())
            end = min(t1, b)
            dur = (end - (t0 + timedelta(seconds=ss))).total_seconds()
            part = work / ("%03d_%s.mp4" % (n, cam))
            if _extract(path, ss, dur, part):
                bits.append(part)
                n += 1
    if not bits:
        return None
    lst = work / "list.txt"
    lst.write_text("".join("file '%s'\n" % p.name for p in bits))
    out = EVENT_ROOT / ("event_%s_id%s.mp4" % (tag, tid))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(lst),
        "-c",
        "copy",
        "-y",
        str(out),
    ]
    ok = subprocess.run(cmd, cwd=str(work)).returncode == 0
    for p in work.iterdir():
        p.unlink()
    work.rmdir()
    if ok and out.exists() and out.stat().st_size > 1000:
        return out
    return None
