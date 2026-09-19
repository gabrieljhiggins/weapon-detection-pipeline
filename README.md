# Real-Time Weapon Detection and Event Reconstruction in Multi-Camera Surveillance Systems

Edge inference on a **Raspberry Pi 5** with a **Hailo-8 AI HAT+ (26 TOPS)** and three **Reolink RLC-520A** PoE cameras. Detection, person–weapon association, Re-ID, and event reconstruction run on the device.

- Dataset: [Multi-Camera Weapon Detection Dataset](https://platform.ultralytics.com/gabriel-higgins/datasets/multi-camera-weapon-detection-dataset) (Ultralytics Platform, Research / Academic Use Only)
- Training artefacts: [Google Drive project archive](https://drive.google.com/drive/folders/1cRBccHCCE-pXnpMpHOmpousu_coObOzT)

---

## Introduction

This project implements a real-time surveillance pipeline built around IoT visual sensors on a Raspberry Pi, accelerated by a Hailo-8 processor. A custom multi-weapon dataset was labelled for this work. Training and compilation run in the cloud; the resulting Hailo Executable Format (HEF) is stored and executed on the edge device.

Weapon detection is coupled to association logic that links weapon boxes to tracked persons. In a multi-person scene that step is what stops an alert from meaning only “a weapon is in the frame”. The alert logs the individual suspect who is likely armed.

Persistent tracking and cross-camera backtracking use Hailo Re-ID. Identity is kept across cameras when a weapon is concealed or discarded. Those trajectories, with association attached, produce a threat alert and a time-stamped event timeline. Detection, association, and tracking run on the device for improved latency and privacy. Training stays in the cloud because it needs a GPU and an x86 Dataflow Compiler. Full streams and event clips are archived for review.

This repository is that weapon detection pipeline:

1. Record three camera views and build a labelled YOLO dataset ([collect/](collect/README.md)).
2. Train YOLO26 on cloud and compile the selected weights to Hailo-8 ([models/](models/README.md)).
3. Run detection, association, tracking, Re-ID, and alerting on the Pi ([detect/](detect/README.md)).
4. Keep camera URLs and the optional boot recorder in one place ([config/](config/README.md)).

Identity uses a **full-body** Hailo RepVGG-A0 512-d embedding, not a face model.

---

## Results

Trained on the merged three-camera set (70 / 20 / 10). Figures below are **mAP50** per class; the first two rows are aggregate mAP50 and mAP50–95 on the validation split.

| Class         | YOLO26n | YOLO26s | YOLO26m |
| ------------- | ------: | ------: | ------: |
| all mAP50     |   0.658 |   0.721 |   0.740 |
| all mAP50–95  |   0.408 |   0.444 |   0.452 |
| person        |   0.989 |   0.983 |   0.991 |
| knife         |   0.184 |   0.267 |   0.238 |
| axe           |   0.528 |   0.629 |   0.674 |
| pistol        |   0.717 |   0.788 |   0.815 |
| assault_rifle |   0.765 |   0.824 |   0.854 |
| shotgun       |   0.762 |   0.832 |   0.869 |

**YOLO26s** is what the Pi runs (`yolo26s_cam123_best.hef`). It is +0.063 mAP50 over nano and close to medium, at a size Hailo-8 can compile and sustain. `person` is effectively solved. Assault rifle, shotgun, pistol and axe yield modest results, while `knife` stays weak (small object, limited views).

On-device HEF throughput on synthetic frames (hailortcli run) is **37.76 FPS**. The live pipeline - three RTSP pulls, overlay, and archive writers - runs at about **10 FPS**, which matches the Reolink sub-stream.

Classes: `person`, `knife`, `axe`, `pistol`, `assault_rifle`, `shotgun`.

---

## Hardware

| Item | Role |
| ---- | ---- |
| Raspberry Pi 5 8GB | Edge device |
| Official Active Cooler | Keeps the SoC in budget under Hailo load |
| Raspberry Pi AI HAT+ 26 TOPS | Hailo-8 on the Pi 5 PCIe connector |
| Official 27W USB-C PSU | Required for Pi 5 + HAT |
| 3× Reolink RLC-520A | PoE cameras, H.264 sub-stream |
| Reolink RLA-PS1E | PoE switch |
| 4× Cat6 Ethernet cables | Three cameras → switch, one switch uplink → Pi `eth0` |
| SanDisk MicroSDXC Ultra 64 GB | Raspberry Pi OS, code repository, and local recordings |

## Network
| Link | Role |
| ---- | ---- |
| Isolated camera LAN | Pi `eth0` = `192.168.50.1/24`, DHCP via dnsmasq |
| Pi Wi-Fi | SSH and GitHub only |

Cameras stay on the wired LAN (`eth0`). The Pi is the DHCP server on Ethernet; Wi-Fi is only the uplink for SSH and Git.

---

## Raspberry Pi 5 and Hailo setup

Official mechanical steps: [AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) and [AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#step1).

### 1. Image the card

On another computer, use [Raspberry Pi Imager](https://www.raspberrypi.com/software/). Choose Raspberry Pi 5 and Raspberry Pi OS (64-bit). In OS customisation set hostname, user, Wi-Fi, locale, and **enable SSH**. Write, eject, insert the card in the Pi.

### 2. Cooler + HAT

Fit the Active Cooler first (thermal pads, push-pins, fan JST). Then:

1. Fit the four spacers on the Pi 5 mounting holes.
2. Seat the GPIO stacking header if the HAT kit includes one.
3. Insert the FPC into the Pi 5 PCIe connector (contacts toward the USB ports) and lock the clip.
4. Seat the HAT on the spacers and screw it down.
5. Insert the other end of the FPC into the HAT and lock the clip.

Do not peel the cooler off after it is seated. Use only the official 27W supply.

### 3. First boot and SSH

Power on. From the laptop:

```bash
ssh <user>@<hostname>.local
```

Update the system, then install Hailo packages from Raspberry Pi OS:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install hailo-all -y
sudo reboot
```

After reboot:

```bash
lspci | grep -i hailo
hailortcli fw-control identify
```

You should see a Hailo-8 function and a firmware version. PCIe link speed on this unit is `8.0 GT/s`.

Optional vendor tree for GStreamer Hailo elements and the official body Re-ID HEF:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

Use the **person** Re-ID HEF (`repvgg_a0_person_reid_512`), not a face Re-ID sample.

### 4. Camera network

Give `eth0` a static address and run dnsmasq for `192.168.50.50–100`. Uplink of the PoE switch goes to the Pi Ethernet port; cameras go to the switch. Confirm:

```bash
sudo nmap -sn 192.168.50.0/24
```

Enable RTSP on each camera in the Reolink app. Stream path used here:

```
rtsp://USER:PASS@192.168.50.x:554/h264Preview_01_sub
```

Copy `config/cameras.json.example` to `config/cameras.json` and fill in credentials. Details: [config/README.md](config/README.md).

---

## Pipeline

```
Cameras  →  collect/          dataset clips and frames
         →  Label Studio      annotated dataset
         →  collect/          train/val/test merge
         →  Colab + DFC       YOLO26s .pt → .hef
         →  detect/pipeline   live Hailo run
```

**Collection.** `collect/cameras_record.py` writes 60s segments per camera so a hard stop only damages the last file. Clips are stitched, frames extracted at 2 fps, labelled per camera, then merged 70 / 20 / 10. Full procedure: [collect/README.md](collect/README.md).

**Training and compile.** Fine-tune on Google Colab (T4). Compile ONNX → HEF on x86_64 with Hailo DFC 3.34.0 (AWS EC2 in this project). Metrics, HEF I/O, and device checks: [models/README.md](models/README.md).

**Live run.** `detect/pipeline.py` starts one grabber thread per RTSP URL. Each frame is archived and the newest frame is queued for Hailo. Two networks share the chip:

- custom detector HEF → person and weapon boxes
- official person Re-ID HEF → 512-d body embedding

A per-camera tracker assigns short-term IDs. Re-ID maps those onto one global ID across views. Association links a weapon to one person using padded containment, overlap, then distance — not IoU alone, because a long gun often sits beside the body. A person becomes the suspect only after they look armed on enough recent frames. When that person has been gone for 20s, one boxed clip and a JSON timeline are written under `data/events/`. Module-by-module description: [detect/README.md](detect/README.md).

---

## Run the live pipeline

Keep `record-cameras.service` disabled so two programs do not open the same RTSP URLs.

```bash
cd ~/weapon-detection-pipeline
rm -rf data/{alerts,events,recordings}
mkdir -p data/{alerts,events,recordings}
/usr/bin/python3 detect/pipeline.py
```

Stop with Ctrl+C.

| Output | Contents |
| ------ | -------- |
| `data/recordings/cam*/` | Full session, one MP4 per camera |
| `data/events/event_*_id*.mp4` | Boxed reconstruction of the tracked suspect |
| `data/events/event_*_id*.json` | Camera timeline for that ID |
| `data/alerts/unassigned/` | Weapons with no associated person |

---

## References

- [Raspberry Pi AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [Raspberry Pi AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html#step1)
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- [Hailo-8 / 8L software downloads](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l)
- [Hailo Dataflow Compiler v3.34.0](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-34-0/?sp_referrer=install/install.html)
- [Ultralytics Hailo integration](https://docs.ultralytics.com/integrations/hailo)
- [Multi-Camera Weapon Detection Dataset](https://platform.ultralytics.com/gabriel-higgins/datasets/multi-camera-weapon-detection-dataset)
- [Project archive (Drive)](https://drive.google.com/drive/folders/1cRBccHCCE-pXnpMpHOmpousu_coObOzT)
- [collect/](collect/README.md) · [config/](config/README.md) · [detect/](detect/README.md) · [models/](models/README.md)