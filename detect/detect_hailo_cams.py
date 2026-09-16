#!/usr/bin/env python3
"""One Hailo-8 HEF, three Reolink sub streams.

Cameras are listed in config/cameras.json.
Frames are inferred round-robin on a single VDevice.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

REPO = Path("/home/gjh/weapon-detection-pipeline")
HEF_PATH = REPO / "models" / "yolo26s_cam123_best.hef"
CAMERAS_JSON = REPO / "config" / "cameras.json"
FRAME_ROOT = REPO / "data" / "frames"
PERSON_CONF = 0.55
WEAPON_CONF = 0.35
WEAPON_NAMES = {"knife", "axe", "pistol", "assault_rifle", "shotgun"}
CLASSES = ["person", "knife", "axe", "pistol", "assault_rifle", "shotgun"]


def load_cameras() -> list[dict]:
    cfg = json.loads(CAMERAS_JSON.read_text())
    cams = []
    for cam in cfg.get("cameras") or []:
        if cam.get("enabled", True) and cam.get("rtsp") and "REPLACE_" not in cam["rtsp"]:
            cams.append({"name": cam["name"], "rtsp": cam["rtsp"]})
    if not cams:
        raise SystemExit(f"No usable cameras in {CAMERAS_JSON}")
    return cams


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def unwrap_nms(hailo_out):
    if isinstance(hailo_out, dict):
        hailo_out = next(iter(hailo_out.values()))
    if isinstance(hailo_out, (list, tuple)) and len(hailo_out) == 1:
        hailo_out = hailo_out[0]
    return hailo_out


def iter_boxes(hailo_out):
    classes = unwrap_nms(hailo_out)
    if classes is None:
        return
    for cls_id, boxes in enumerate(classes):
        if boxes is None:
            continue
        if isinstance(boxes, np.ndarray) and boxes.size == 0:
            continue
        try:
            items = list(boxes)
        except TypeError:
            items = [boxes]
        for item in items:
            row = np.asarray(item, dtype=float).reshape(-1)
            if row.size >= 5:
                yield cls_id, row, float(row[-1])


def peaks_from(hailo_out, n=6):
    peaks = [0.0] * n
    for cls_id, _, score in iter_boxes(hailo_out):
        if 0 <= cls_id < n:
            peaks[cls_id] = max(peaks[cls_id], score)
    return peaks


def draw(frame, hailo_out, scale, left, top):
    vis = frame.copy()
    for cls_id, row, score in iter_boxes(hailo_out):
        name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"id{cls_id}"
        need = PERSON_CONF if name == "person" else WEAPON_CONF
        if score < need:
            continue
        ymin, xmin, ymax, xmax = row[:4]
        x1 = int((xmin * 640 - left) / scale)
        y1 = int((ymin * 640 - top) / scale)
        x2 = int((xmax * 640 - left) / scale)
        y2 = int((ymax * 640 - top) / scale)
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"id{cls_id}:{name} {score:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return vis


def grabber(name: str, rtsp: str, q: Queue, stop: threading.Event):
    cap = None
    fails = 0
    while not stop.is_set():
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"  Camera {name} - DOWN — retry")
                time.sleep(1.0)
                continue
            print(f"  Camera {name} - LIVE")
            fails = 0
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1
            if fails >= 15:
                print(f"  Camera {name} - DOWN — reconnect")
                cap.release()
                cap = None
                fails = 0
            time.sleep(0.05)
            continue
        fails = 0
        try:
            while q.qsize() > 0:
                q.get_nowait()
            q.put_nowait(frame)
        except Exception:
            pass
    if cap is not None:
        cap.release()


def main():
    cams = load_cameras()
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    for cam in cams:
        (FRAME_ROOT / cam["name"]).mkdir(parents=True, exist_ok=True)

    print("HEF ", HEF_PATH)
    print("JSON", CAMERAS_JSON)
    print("Cams", ", ".join(c["name"] for c in cams))
    print("Ctrl+C to stop")

    stop = threading.Event()
    queues = {c["name"]: Queue(maxsize=1) for c in cams}
    threads = []
    for cam in cams:
        t = threading.Thread(
            target=grabber,
            args=(cam["name"], cam["rtsp"], queues[cam["name"]], stop),
            daemon=True,
        )
        t.start()
        threads.append(t)

    hef = HEF(str(HEF_PATH))
    frames = 0
    per_cam = {c["name"]: 0 for c in cams}
    t0 = time.time()

    with VDevice() as target:
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, params)[0]
        ng_params = network_group.create_params()
        in_name = hef.get_input_vstream_infos()[0].name
        in_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.UINT8
        )
        out_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        with InferVStreams(network_group, in_params, out_params) as infer:
            with network_group.activate(ng_params):
                try:
                    while True:
                        got_any = False
                        for cam in cams:
                            name = cam["name"]
                            try:
                                frame = queues[name].get(timeout=0.05)
                            except Empty:
                                continue
                            got_any = True
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            inp, scale, left, top = letterbox(rgb, 640)
                            raw = infer.infer({in_name: np.expand_dims(inp, 0)})
                            frames += 1
                            per_cam[name] += 1
                            peaks = peaks_from(raw, n=len(CLASSES))
                            hits = [
                                (CLASSES[i], peaks[i])
                                for i in range(len(CLASSES))
                                if CLASSES[i] in WEAPON_NAMES and peaks[i] >= WEAPON_CONF
                            ]
                            if not hits:
                                continue
                            vis = draw(frame, raw, scale, left, top)
                            hit_s = " ".join(f"{n}={s:.2f}" for n, s in hits)
                            print(f"WEAPON {name} person={peaks[0]:.2f} {hit_s}")
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            best_name = max(hits, key=lambda t: t[1])[0]
                            out = FRAME_ROOT / name / f"{name}_{ts}_{best_name}.jpg"
                            cv2.imwrite(str(out), vis)
                        if not got_any:
                            time.sleep(0.02)
                except KeyboardInterrupt:
                    print("\nStopped")
                finally:
                    stop.set()

    elapsed = max(time.time() - t0, 1e-6)
    print(f"frames={frames} {frames / elapsed:.1f} FPS total")
    for name, n in per_cam.items():
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
