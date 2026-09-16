# Detect

Live Hailo-8 inference on one Reolink RTSP stream.

## Script

`detect_hailo_cam1.py`

- Model: `models/yolo26s_cam123_best.hef` (Hailo-8, NMS on chip)
- Camera: `rtsp://admin:admin123@192.168.50.64:554/h264Preview_01_sub`
- Classes: person, knife, axe, pistol, assault_rifle, shotgun
- Writes overlay JPEGs to `data/frames/`

## Run

```bash
rm -f /home/gjh/weapon-detection-pipeline/data/frames/*.jpg
/usr/bin/python3 /home/gjh/weapon-detection-pipeline/detect/detect_hailo_cam1.py
```

Ctrl+C to stop.

## Notes

Use the **sub** stream for live tests. The main stream overloads the Pi and drops H264 slices.

NMS layout from Hailo is `[batch][class][boxes]`. The script unwraps the batch dim, then reads six class lists.
