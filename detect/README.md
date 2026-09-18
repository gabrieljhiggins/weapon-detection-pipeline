# Live detection

This folder is the edge pipeline that runs on the Raspberry Pi 5 with the Hailo-8 HAT. Three Reolink sub-streams are read at once. Each frame is analyzed, people are given unique IDs, in the case of a weapon being detected, it is associated to one person, and that suspect is then tracked across cameras for complete event reconstruction - even in the case of discarding or concealing the weapon. When the suspect leaves the scene, one boxed clip and a short JSON log are written under `data/events/`. Raw footage from every camera is stored under `data/recordings/` for later review.

## Files

| File | Role |
| --- | --- |
| `pipeline.py` | Wires the modules, owns the Hailo device, prints status. |
| `common.py` | `cameras.json` loader, class names, confidence thresholds, letterbox, Hailo NMS unwrap, box draw. |
| `associate.py` | Weapon-person association logic; leftover weapons stay “unassigned”. |
| `track.py` | Separate tracker per camera. |
| `reid.py` | Cross-camera identity; freezes the suspect embedding. |
| `backtrack.py` | Open/close time spans per `(id, camera)` when a gap exceeds 3 s. |
| `event.py` | Keeps ~4 minutes of boxed JPEGs per ID and encodes the event MP4. |
| `record_all.py` | Opens one `VideoWriter` per camera from the same frames the detector sees. |

## Video Processing

`pipeline.py` starts one grabber thread per camera. The thread pulls RTSP, appends the frame to that camera’s archive MP4, and keeps only the latest frame in a size-1 queue so Hailo never sits on stale video.

The main loop takes cameras in turn, letterboxes the frame to 640×640, and runs two networks on the same Hailo device:

1. Custom YOLO HEF (`models/yolo26s_cam123_best.hef`) → person and weapon detection.
2. Official person Re-ID HEF (`repvgg_a0_person_reid_512.hef`) → a 512-D embedding per person crop.

`track.py` keeps a short-term ID on each camera. `reid.py` then maps those local IDs onto a global gallery so the same person keeps one ID when they walk across camera views. Once an alert fires, that embedding is frozen and later views refer to that ID.

`associate.py` decides who is holding the weapon. A rifle or shotgun often sits beside the body, so the code does not rely on IoU alone. It tries, in order: weapon centre inside a padded person box, any real overlap, then distance compared with person height. Unmatched weapons are stored under `data/alerts/unassigned/`.

A person becomes the suspect only after they look armed on enough recent frames (`ARM_NEED` of `ARM_WINDOW`, currently 20 of 48 frames). The console prints:

```
Alert: Camera 2 (person armed - tracking suspect ID 1)
```

For the next 20s after they disappear, history is closed and `event.py` builds one clip that stays on a camera until another view takes over. `backtrack.py` prints the same intervals and writes them into the JSON log.

## Run

Clears the last run's output, then starts the pipeline. Use system Python so Hailo packages resolve.

```bash
rm -rf ~/weapon-detection-pipeline/data/{alerts,events,recordings}
mkdir -p ~/weapon-detection-pipeline/data/{alerts,events,recordings}
/usr/bin/python3 ~/weapon-detection-pipeline/detect/pipeline.py
```

Stop with Ctrl+C. That closes the archive writers. After a run:

```
data/recordings/cam1|cam2|cam3/*.mp4   full session, one file per camera
data/events/event_*_id*.mp4            complete event reconstruction of the tracked suspect and their activity before during and after the attack.
data/events/event_*_id*.json           camera timeline for that ID
data/alerts/unassigned/                weapon with no person
```

Cameras come from `config/cameras.json`. The recorder service in `config/` must stay off while this process runs so two programs do not open the same RTSP URLs.

# References
- [Intersection over Union (IoU) in Object Detection & Segmentation](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/)