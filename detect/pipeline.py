#!/usr/bin/env python3
"""Record all cameras. One boxed event clip = suspect only, all views, in time order."""

from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict, deque
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
import event
import record_all
import reid
import track

ARM_WINDOW = 48
ARM_NEED = 20
IDLE_S = 20.0
LOOSE_GAP = 2.0


def grabber(name, rtsp, q, stop, live, recs):
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
        record_all.attach(recs, name, frame)
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


def cam_label(name):
    s = str(name).strip()
    low = s.lower()
    if low.startswith("cam") and low[3:].isdigit():
        return "Camera %s" % low[3:]
    if low.startswith("camera") and low[6:].strip().isdigit():
        return "Camera %s" % low[6:].strip()
    return s


def finish(tid, history, bank):
    path = bank.encode(tid)
    spans = history.dump("", tid)
    event.write_log(tid, spans or [], path)


def main():
    cams = common.load_cameras()
    print("Starting weapon detection pipeline...")
    recs = record_all.start(cams)
    bank = event.Bank()

    stop = threading.Event()
    queues = {c["name"]: Queue(maxsize=1) for c in cams}
    live = {c["name"]: False for c in cams}
    for cam in cams:
        threading.Thread(
            target=grabber,
            args=(cam["name"], cam["rtsp"], queues[cam["name"]], stop, live, recs),
            daemon=True,
        ).start()

    print("Waiting for cameras...")
    t0 = time.time()
    shown = set()
    while time.time() - t0 < 20 and not all(live.values()):
        for c in cams:
            if live[c["name"]] and c["name"] not in shown:
                print("  %s - LIVE" % cam_label(c["name"]))
                shown.add(c["name"])
        time.sleep(0.2)
    for c in cams:
        if live[c["name"]] and c["name"] not in shown:
            print("  %s - LIVE" % cam_label(c["name"]))
    down = [cam_label(c["name"]) for c in cams if not live[c["name"]]]
    if down:
        print("Not live: " + " ".join(down))

    trackers = track.Trackers()
    history = backtrack.Backtrack(seconds=3600, gap=3.0)
    suspect = None
    last_seen = 0.0
    arm_hist = defaultdict(lambda: deque(maxlen=ARM_WINDOW))
    last_loose = {}
    wrote = False

    vd_params = VDevice.create_params()
    vd_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    with VDevice(vd_params) as target:
        det_ng, det_in, det_ip, det_op = open_net(target, common.HEF_PATH)
        reid_ng, reid_in, reid_ip, reid_op = open_net(target, common.REID_HEF)
        with InferVStreams(det_ng, det_ip, det_op) as det_infer, InferVStreams(
            reid_ng, reid_ip, reid_op
        ) as reid_infer:
            gallery = reid.ReID(reid_infer.infer, reid_in)
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
                        people, remaps = gallery.assign(frame, people, cam=name)
                        for src, dst in remaps:
                            bank.absorb(src, dst)
                            history.absorb(src, dst)
                            if src in arm_hist:
                                arm_hist[dst].extend(arm_hist.pop(src))
                        if suspect is not None and len(people) == 1:
                            tid = people[0].get("track_id")
                            if tid is not None and tid != suspect:
                                bank.absorb(tid, suspect)
                                history.absorb(tid, suspect)
                                if tid in arm_hist:
                                    arm_hist[suspect].extend(arm_hist.pop(tid))
                                people[0]["track_id"] = suspect
                        assoc = associate.run(people, weapons)
                        armed_ids = set()
                        for item in assoc["armed"]:
                            item["person"]["armed"] = True
                            tid = item["person"].get("track_id")
                            armed_ids.add(tid)
                            if tid is not None:
                                arm_hist[tid].append(True)
                            history.add(name, tid, item["person"]["box"], True, "weapon")
                        for p in assoc["idle"]:
                            tid = p.get("track_id")
                            if tid is not None:
                                arm_hist[tid].append(False)
                            history.add(name, tid, p["box"], False, None)

                        if assoc["loose"]:
                            now = time.time()
                            if now - last_loose.get(name, 0) >= LOOSE_GAP:
                                last_loose[name] = now
                                loose_dir = common.ALERT_ROOT / "unassigned"
                                loose_dir.mkdir(parents=True, exist_ok=True)
                                snap = common.draw(frame, people, assoc["loose"])
                                snap_path = loose_dir / (
                                    "%s_%s_weapon.jpg"
                                    % (name, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                                )
                                cv2.imwrite(str(snap_path), snap)

                        if suspect is not None:
                            for p in people:
                                if p.get("track_id") == suspect:
                                    p["armed"] = True

                        vis = common.draw(frame, people, weapons)
                        for p in people:
                            bank.add(p.get("track_id"), name, vis)
                            if suspect is not None and p.get("track_id") == suspect:
                                last_seen = time.time()

                        if suspect is None:
                            for item in assoc["armed"]:
                                tid = item["person"].get("track_id")
                                hist = arm_hist.get(tid, ())
                                if tid is None or sum(hist) < ARM_NEED:
                                    continue
                                suspect = tid
                                gallery.mark_attacker(tid, item["person"].get("embedding"))
                                last_seen = time.time()
                                wrote = False
                                print(
                                    "Alert: %s (person armed - tracking suspect ID %s)"
                                    % (cam_label(name), suspect)
                                )
                                break

                    if (
                        suspect is not None
                        and not wrote
                        and last_seen
                        and time.time() - last_seen > IDLE_S
                    ):
                        finish(suspect, history, bank)
                        wrote = True
                        gallery.clear_attacker()
                        suspect = None
                        last_seen = 0.0
                        arm_hist.clear()
                    if not got_any:
                        time.sleep(0.02)
            except KeyboardInterrupt:
                print("\nStopped")
            finally:
                stop.set()
                if suspect is not None and not wrote:
                    finish(suspect, history, bank)
                record_all.stop(recs)

    print("All-camera archive: data/recordings")
    print("Event reconstruction: data/events")


if __name__ == "__main__":
    main()
