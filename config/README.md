# Configuration

This folder holds the camera list used by both dataset recording and live detection, plus the systemd unit that can start recording at boot. Paths and passwords live here so the Python scripts do not hard-code device details.

```
config/
  cameras.json
  cameras.json.example
  record-cameras.service
  README.md
```

## RTSP

RTSP (Real Time Streaming Protocol) is how a client asks an IP camera for a live video stream. The camera keeps sending H.264 frames until the client disconnects. A Reolink URL looks like:

```
rtsp://username:password@ip:554/h264Preview_01_sub
```

Port **554** is the default. `_sub` is the low-resolution fluent stream used in this project; `_main` is the full 2560×1920 stream and is not used for Hailo inference. RTSP must be switched on in the Reolink app (network / advanced / server settings) before any of these files will work.

## cameras.json

The live config on the Pi. `collect/cameras_record.py` and `detect/common.py` both read it.

| Field | Purpose |
| --- | --- |
| `segment_seconds` | Length of each clip written by the collector (60 s). |
| `output_dir` | Folder for those clips (`data/recordings`). |
| `cameras` | The three sources. `name` must stay `cam1`, `cam2`, `cam3`. `rtsp` is the stream URL. |


## record-cameras.service

Systemd unit for automatic dataset collection on startup - developed for field dataset recording. It waits a while after boot (so the PoE switch and cameras are up), then runs `collect/cameras_record.py`.

This unit is only for recording in the inital parts of the project development. It is left disabled while writing or running `detect/pipeline.py`.

```bash
sudo cp config/record-cameras.service /etc/systemd/system/
sudo systemctl enable --now record-cameras.service   # field recording
sudo systemctl disable --now record-cameras.service  # development
```

`collect/stop_recording.sh` stops the unit when it is active.

## References

- [Introduction to RTSP](https://support.reolink.com/articles/900000630706-Introduction-to-RTSP/)
