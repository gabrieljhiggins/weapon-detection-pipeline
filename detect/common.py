"""Shared camera / HEF / box helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/gjh/weapon-detection-pipeline")
HEF_PATH = REPO / "models" / "yolo26s_cam123_best.hef"
REID_HEF = Path("/usr/local/hailo/resources/models/hailo8/repvgg_a0_person_reid_512.hef")
CAMERAS_JSON = REPO / "config" / "cameras.json"
FRAME_ROOT = REPO / "data" / "frames"
ALERT_ROOT = REPO / "data" / "alerts"

PERSON_CONF = 0.55
WEAPON_CONF = 0.45
CLASSES = ["person", "knife", "axe", "pistol", "assault_rifle", "shotgun"]
WEAPON_NAMES = {"knife", "axe", "pistol", "assault_rifle", "shotgun"}


def load_cameras() -> list[dict]:
    cfg = json.loads(CAMERAS_JSON.read_text())
    cams = []
    for cam in cfg.get("cameras") or []:
        if cam.get("enabled", True) and cam.get("rtsp") and "REPLACE_" not in str(cam.get("rtsp", "")):
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


def box_xyxy(row, scale, left, top, fw, fh):
    ymin, xmin, ymax, xmax = row[:4]
    x1 = int((xmin * 640 - left) / scale)
    y1 = int((ymin * 640 - top) / scale)
    x2 = int((xmax * 640 - left) / scale)
    y2 = int((ymax * 640 - top) / scale)
    x1 = max(0, min(fw - 1, x1))
    x2 = max(0, min(fw - 1, x2))
    y1 = max(0, min(fh - 1, y1))
    y2 = max(0, min(fh - 1, y2))
    return x1, y1, x2, y2


def detections(hailo_out, scale, left, top, fw, fh):
    people, weapons = [], []
    for cls_id, row, score in iter_boxes(hailo_out):
        name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"id{cls_id}"
        box = box_xyxy(row, scale, left, top, fw, fh)
        if name == "person" and score >= PERSON_CONF:
            people.append({"name": "person", "score": score, "box": box, "cls_id": cls_id})
        elif name in WEAPON_NAMES and score >= WEAPON_CONF:
            weapons.append({"name": "weapon", "score": score, "box": box, "cls_id": cls_id})
    return people, weapons


def draw(frame, people, weapons):
    vis = frame.copy()
    for p in people:
        x1, y1, x2, y2 = p["box"]
        tid = p.get("track_id", "?")
        armed = p.get("armed")
        color = (0, 0, 255) if armed else (0, 255, 0)
        tag = "ARMED" if armed else "person"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"id{tid} {tag} {p['score']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    for w in weapons:
        x1, y1, x2, y2 = w["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(
            vis,
            f"weapon {w['score']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            2,
        )
    return vis
