#!/usr/bin/env python3
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    HailoStreamInterface,
)

HEF_PATH = "/home/gjh/weapon-detection-pipeline/models/yolo26s_cam123_best.hef"
RTSP = "rtsp://admin:admin123@192.168.50.64:554/h264Preview_01_sub"
FRAME_DIR = Path("/home/gjh/weapon-detection-pipeline/data/frames")
CONF = 0.20
SAVE_EVERY = 5
CLASSES = ["person", "knife", "axe", "pistol", "assault_rifle", "shotgun"]


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
    # Hailo NMS BY CLASS: [batch][class][boxes]
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


def dump_nms(raw):
    classes = unwrap_nms(raw)
    print("NMS classes", type(classes).__name__, "len", len(classes) if hasattr(classes, "__len__") else None)
    for i, b in enumerate(classes):
        try:
            n = 0 if b is None else len(b)
        except TypeError:
            n = -1
        print(" class", i, type(b).__name__, "nboxes", n)


def draw(frame, hailo_out, scale, left, top):
    vis = frame.copy()
    for cls_id, row, score in iter_boxes(hailo_out):
        if score < CONF:
            continue
        name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else f"id{cls_id}"
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
            0.55,
            color,
            2,
        )
    return vis


def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    print("Opening", RTSP)
    print("Saving to", FRAME_DIR)
    print("Ctrl+C to stop")

    cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit(1)

    hef = HEF(HEF_PATH)
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

        frames = 0
        t0 = time.time()
        with InferVStreams(network_group, in_params, out_params) as infer:
            with network_group.activate(ng_params):
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            print("Frame grab failed")
                            time.sleep(0.2)
                            continue
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inp, scale, left, top = letterbox(rgb, 640)
                        raw = infer.infer({in_name: np.expand_dims(inp, 0)})
                        frames += 1
                        if frames == 1:
                            dump_nms(raw)
                        peaks = peaks_from(raw, n=len(CLASSES))
                        vis = draw(frame, raw, scale, left, top)
                        if frames % 5 == 0:
                            peak_s = " | ".join(
                                f"{i}:{CLASSES[i]}={peaks[i]:.2f}" for i in range(len(CLASSES))
                            )
                            print(f"{frames:5d} {peak_s}")
                        if frames % SAVE_EVERY == 0:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            best = int(np.argmax(peaks))
                            cv2.imwrite(str(FRAME_DIR / f"cam_{ts}_best{best}.jpg"), vis)
                except KeyboardInterrupt:
                    print("\nStopped")
        print(f"frames={frames} {frames / max(time.time() - t0, 1e-6):.1f} FPS")
    cap.release()


if __name__ == "__main__":
    main()