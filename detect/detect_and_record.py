#!/usr/bin/env python3
"""Watch one camera with YOLO. On detection, save a 5s clip.

Watched classes: person, air rifle, person - armed
Model:           yolo26n_best.pt  (place next to this script)
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000",
)

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).resolve().parent / "yolo26n_best.pt"
DEFAULT_SOURCE = os.environ.get(
    "CAMERA_RTSP",
    "rtsp://admin:admin123@192.168.50.81:554/h264Preview_01_sub",
)
DEFAULT_OUT = Path(__file__).resolve().parent / "recordings"
CLIP_SECONDS = 5.0
CONF = 0.35

WATCH = {"person", "air rifle", "person - armed"}


def normalize(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


WATCH_NORM = {normalize(n) for n in WATCH}


def is_watched(name: str) -> bool:
    return normalize(name) in WATCH_NORM


def open_cam(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    return cap


def main() -> int:
    p = argparse.ArgumentParser(description="Detect classes and auto-record 5s clips")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--duration", type=float, default=CLIP_SECONDS)
    p.add_argument("--once", action="store_true")
    p.add_argument("--list-classes", action="store_true")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Copy yolo26n_best.pt next to this script.")
        return 1

    print(f"Loading {model_path.name} ...")
    model = YOLO(str(model_path))
    print("Model classes:", dict(model.names))

    if args.list_classes:
        for i, name in model.names.items():
            mark = "  [WATCH]" if is_watched(str(name)) else ""
            print(f"  {int(i):3d}  {name}{mark}")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Watching: {', '.join(sorted(WATCH))}  (conf >= {args.conf})")
    print(f"Source:   {args.source}")
    print(f"Clips →   {out_dir.resolve()}  ({args.duration:.0f}s each)")
    print("Ctrl+C to stop.\n")

    cap = open_cam(args.source)
    ok, frame = cap.read()
    if not ok:
        print("No frames from source.")
        return 1

    h, w = frame.shape[:2]
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps < 5 or fps > 60:
        fps = 15.0

    writer = None
    clip_until = 0.0
    clip_path = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.25)
                cap.release()
                cap = open_cam(args.source)
                continue

            result = model.predict(frame, conf=args.conf, verbose=False)[0]
            hits = []
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < args.conf:
                        continue
                    label = str(result.names[int(box.cls[0])])
                    if not is_watched(label):
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    hits.append((label, conf, (x1, y1, x2, y2)))

            for label, conf, (x1, y1, x2, y2) in hits:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 220), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 220),
                    2,
                )

            now = time.monotonic()

            if hits and writer is None:
                top_label, top_conf, _ = max(hits, key=lambda t: t[1])
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                safe = top_label.replace(" ", "_").replace("-", "_")
                clip_path = out_dir / f"{stamp}_{safe}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    print(f"Failed to open writer for {clip_path}")
                    writer = None
                    continue
                clip_until = now + args.duration
                writer.write(frame)
                print(f"REC  {top_label}  {top_conf:.2f}  →  {clip_path.name}")
                continue

            if writer is not None:
                remaining = clip_until - now
                cv2.circle(frame, (w - 24, 24), 10, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"REC {max(remaining, 0):.1f}s",
                    (w - 140, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                writer.write(frame)
                if now >= clip_until:
                    writer.release()
                    print(f"SAVED {clip_path.resolve()}")
                    writer = None
                    clip_path = None
                    if args.once:
                        print("First clip done. Stopping.")
                        break

            if args.preview:
                cv2.imshow("detect_and_record", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if writer is not None:
            writer.release()
            if clip_path is not None:
                print(f"Partial clip saved: {clip_path}")
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())