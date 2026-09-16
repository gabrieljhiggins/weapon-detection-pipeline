#!/usr/bin/env python3
"""Record all cameras. One event clip = suspect only, all views, in time order."""

from __future__ import annotations

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
    HailoSchedulingAlgorithm,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import associate
import backtrack
import common
import record_all
import reid
import stitch
import track


def grabber(name, rtsp, q, stop, live):
    cap = None
    fails = 0
    while not stop.is_set():
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                live[name] = False
                time.sleep(1.0)
                continue
            live[name] = True
            fails = 0
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1
            if fails >= 15:
                live[name] = False
                cap.release()
                cap = None
                fails = 0
            time.sleep(0.05)
            continue
        fails = 0
        live[name] = True
        try:
            while q.qsize() > 0:
                q.get_nowait()
            q.put_nowait(frame)
        except Exception:
            pass
    if cap is not None:
        cap.release()


def open_net(target, hef_path):
    hef = HEF(str(hef_path))
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = target.configure(hef, params)[0]
    in_name = hef.get_input_vstream_infos()[0].name
    in_params = InputVStreamParams.make_from_network_group(
        ng, quantized=False, format_type=FormatType.UINT8
    )
    out_params = OutputVStreamParams.make_from_network_group(
        ng, quantized=False, format_type=FormatType.FLOAT32
    )
    return ng, in_name, in_params, out_params


def finish(tid, history):
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    spans = history.spans.get(tid, [])
    path = stitch.stitch(tid, spans, tag)
    if path:
        print("Event clip: %s" % path)
    else:
        print("Event clip: failed (no overlapping recordings)")


def main():
    cams = common.load_cameras()
    recs = record_all.start(cams)

    stop = threading.Event()
    queues = {c["name"]: Queue(maxsize=1) for c in cams}
    live = {c["name"]: False for c in cams}
    for cam in cams:
        threading.Thread(
            target=grabber,
            args=(cam["name"], cam["rtsp"], queues[cam["name"]], stop, live),
            daemon=True,
        ).start()

    t0 = time.time()
    while time.time() - t0 < 20 and not all(live.values()):
        time.sleep(0.2)
    print("Cams live: " + " ".join(c["name"] for c in cams if live[c["name"]]))

    trackers = track.Trackers()
    history = backtrack.Backtrack(seconds=3600, gap=3.0)
    suspect = None
    last_seen = 0.0

    vd_params = VDevice.create_params()
    vd_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    with VDevice(vd_params) as target:
        det_ng, det_in, det_ip, det_op = open_net(target, common.HEF_PATH)
        reid_ng, reid_in, reid_ip, reid_op = open_net(target, common.REID_HEF)
        with InferVStreams(det_ng, det_ip, det_op) as det_infer, InferVStreams(
            reid_ng, reid_ip, reid_op
        ) as reid_infer:
            gallery = reid.ReID(reid_infer.infer, reid_in, match_thr=0.45)
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
                        inp, scale, left, top = common.letterbox(rgb, 640)
                        raw = det_infer.infer({det_in: np.expand_dims(inp, 0)})
                        fh, fw = frame.shape[:2]
                        people, weapons = common.detections(raw, scale, left, top, fw, fh)
                        people = trackers.update(name, people)
                        people = gallery.assign(frame, people)
                        assoc = associate.run(people, weapons)
                        for item in assoc["armed"]:
                            gallery.mark_attacker(item["person"].get("track_id"))
                            history.add(
                                name,
                                item["person"].get("track_id"),
                                item["person"]["box"],
                                True,
                                "weapon",
                            )
                        for p in assoc["idle"]:
                            history.add(name, p.get("track_id"), p["box"], False, None)
                        if assoc["armed"] and suspect is None:
                            suspect = assoc["armed"][0]["person"].get("track_id")
                            print(
                                "Alert: %s (weapon detected - tracking suspect)"
                                % name
                            )
                        if suspect is not None:
                            for p in people:
                                if p.get("track_id") == suspect:
                                    last_seen = time.time()
                            if last_seen and time.time() - last_seen > 8:
                                finish(suspect, history)
                                suspect = None
                                last_seen = 0.0
                    if not got_any:
                        time.sleep(0.02)
            except KeyboardInterrupt:
                print("\nStopped")
            finally:
                stop.set()
                if suspect is not None:
                    finish(suspect, history)
                record_all.stop(recs)

    print("Raw recordings in data/recordings/ — delete after you copy the event clip")


if __name__ == "__main__":
    main()
