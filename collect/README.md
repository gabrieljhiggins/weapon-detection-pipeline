# Dataset collection

This folder documents how camera footage was recorded and turned into a labelled training set. The scripts are kept for reference. They are not part of the live detection pipeline.

The sequence is:

1. Record each camera to short clips.
2. Stop recording cleanly.
3. Stitch each camera’s clips into one video.
4. Extract frames for annotation.
5. Label frames in Label Studio with upright bounding boxes.
6. After Label Studio export, merge Camera 1/2/3 into one YOLO dataset with train / val / test splits.

```
cameras_record.py  →  videos_merge.py  →  frames_extract.py  →  Label Studio  →  dataset_build.py
```

## Layout

```
collect/
  cameras_record.py
  stop_recording.sh
  videos_merge.py
  frames_extract.py
  dataset_build.py
  README.md
```

Camera names in `config/cameras.json` should stay `cam1`, `cam2`, `cam3`. Label Studio export folders are expected as `Camera 1`, `Camera 2`, `Camera 3`.

## 1. Record (`cameras_record.py`)

Pulls each Reolink RTSP stream and writes video plus audio. Footage is stored as short MP4 segments so a hard stop only corrupts the last clip.

```bash
python3 collect/cameras_record.py --config config/cameras.json
```

`config/cameras.json` controls the output folder (`output_dir`, default `data/recordings`) and clip length (`segment_seconds`, default 60). Each camera uses `-c:v copy` (no re-encode) and AAC audio at 64 kb/s.

On start the script prints `Camera <name> - LIVE`. If an ffmpeg process dies it is restarted.

## 2. Stop (`stop_recording.sh`)

Use this instead of pulling power.

```bash
bash collect/stop_recording.sh
```

If `record-cameras.service` is enabled it is stopped with systemd. Otherwise a `STOP` file is created in the repo root and the recorder exits on the next loop.

Wait a minute after stopping before copying files. The last segment may still be short or incomplete; do not use that clip for training.

## 3. Merge clips (`videos_merge.py`)

Joins every valid MP4 in a camera folder into one file named `<camera>_full.mp4`. Concatenation uses ffmpeg stream copy.

```bash
python3 collect/videos_merge.py --all --recordings data/recordings
# or one camera
python3 collect/videos_merge.py cam1 --recordings data/recordings
```

Clips smaller than 1 KB are skipped. Output examples:

- `data/recordings/cam1_full.mp4`
- `data/recordings/cam2_full.mp4`
- `data/recordings/cam3_full.mp4`

## 4. Extract frames (`frames_extract.py`)

Reads each `*_full.mp4` and writes JPEGs plus a `metadata.json` (frame index and timestamp). Default rate is 2 frames per second to keep things manageable given the combined collected footage for this project was around 6 hours.

```bash
python3 collect/frames_extract.py \
  --input-dir data/recordings \
  --output-dir data/frames \
  --fps 2 \
  --quality 3
```

`--quality` is ffmpeg `-q:v` (lower is better). These frames are what go into Label Studio.

## 5. Label Studio

One project per camera. Keep the three projects separate so `dataset_build.py` can prefix files (`camera1_…`, `camera2_…`, `camera3_…`) without name clashes.

Suggested project names: **Camera 1**, **Camera 2**, **Camera 3**.

### Install and start

```bash
python3 -m venv .venv-ls
source .venv-ls/bin/activate
pip install label-studio
```

For a large local frame set, start Label Studio with local-file serving enabled. Do not drag thousands of JPEGs through the Import button.

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/data
label-studio start
```

Open http://localhost:8080 and create an account.

### Create a project

1. **Create Project**.
2. Name it `Camera 1` (then repeat for 2 and 3).
3. Skip the small file-upload tab.
4. Under labeling setup, choose **Computer Vision → Object Detection with Bounding Boxes**, then replace the config with the block below.
5. Save.

Use only `RectangleLabels`. Do not add `PolygonLabels`, `BrushLabels`, or rotated-box tools. YOLO export stores axis-aligned boxes (`class x_center y_center width height`). A rotated box is flattened on export and no longer matches the object.

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="#3b82f6"/>
    <Label value="knife" background="#ffe500"/>
    <Label value="axe" background="#64a500"/>
    <Label value="pistol" background="#f30a0a"/>
    <Label value="assault_rifle" background="#ff8500"/>
    <Label value="shotgun" background="#c026d3"/>
  </RectangleLabels>
</View>
```

Use the same class strings, in the same order, on all three projects. `dataset_build.py` takes `classes.txt` from Camera 1.

### Import all frames at once

The web Import dialog is meant for small batches. For this project, attach the frame folder as **Local Files** storage and sync it.

1. Project **Settings → Cloud Storage → Add Source Storage**.
2. Storage type: **Local Files**.
3. Absolute local path must sit under `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`, for example:
   - root: `/home/gjh/weapon-detection-pipeline/data`
   - path: `/home/gjh/weapon-detection-pipeline/data/frames/cam1`
4. Import method: **Files**.
5. File filter regex: `.*\.(jpg|jpeg)$`
6. Treat every object as a source file: **on**.
7. Check connection, Save, then **Sync Storage**.

Label Studio creates one task per JPEG. Sync again if new frames are added later.

On macOS / Windows the same pattern applies: set `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` to a parent folder, then point storage at the camera frame directory.

### How to draw boxes

1. Select the class first (`person`, `assault_rifle`, …).
2. Click and drag a rectangle. Two clicks, no rotation.
3. Keep every box axis-aligned (sides parallel to the image). Do not use the three-point rotated box gesture.
4. One box per object. A person holding a rifle is two labels: `person` and `assault_rifle`.
5. If the weapon is long and diagonal, still use an upright box that covers it. That is what YOLO trains on.
6. Skip empty frames or leave them with no boxes (they export as empty `.txt` files and are valid background).
7. Submit / Update after each image.

Do not mix polygons or brushes in these projects. Those types do not export cleanly to the YOLO format used by `dataset_build.py`.

### Export

1. **Export**.
2. Format: **YOLO** (not YOLO OBB).
3. Unzip into a folder named exactly `Camera 1`, `Camera 2`, or `Camera 3`.

Expected layout per camera:

```
Camera 1/
  classes.txt
  images/
  labels/
  notes.json
```

If the YOLO zip has labels but no images, copy the original JPEGs into `images/` so the filenames match the `.txt` stems.

## 6. Build the YOLO set (`dataset_build.py`)

After each camera is annotated and exported in YOLO format, this script copies images and labels into one folder and splits them:

- train 70%
- val 20%
- test 10%

```bash
python3 collect/dataset_build.py \
  --root /path/to/frames \
  --out /path/to/train_cam123 \
  --seed 42
```

`--root` must contain:

```
Camera 1/images/  Camera 1/labels/  Camera 1/classes.txt
Camera 2/images/  Camera 2/labels/  Camera 2/classes.txt
Camera 3/images/  Camera 3/labels/  Camera 3/classes.txt
```

`classes.txt` from Camera 1 is used for `data.yaml`. Images are prefixed (`camera1_…`) so names from different cameras do not collide. Missing label files become empty `.txt` files (background frames).

## Notes

- Recording and live detection should not share the same RTSP session if you need both at once; this is for offline dataset work.
- Keep class names identical across the three Label Studio projects before running `dataset_build.py`.
- The random split is seeded (`--seed 42`) so the same run can be reproduced.

## References

- [Label Studio – Install and upgrade Label Studio](https://labelstud.io/guide/install)
- [Label Studio – Create and configure projects](https://labelstud.io/guide/setup_project)
- [Label Studio – Configure the labeling interface](https://labelstud.io/guide/setup)
- [Label Studio – Object Detection with Bounding Boxes (`RectangleLabels`)](https://labelstud.io/templates/image_bbox.html)
- [Label Studio – Import data](https://labelstud.io/guide/tasks.html)
- [Label Studio – Set up local storage](https://labelstud.io/guide/storage_local)
- [Label Studio – Cloud and external storage](https://labelstud.io/guide/storage.html)
- [Label Studio – Export annotations](https://labelstud.io/guide/export.html)
- [Label Studio – Label and annotate data](https://labelstud.io/guide/labeling/)
