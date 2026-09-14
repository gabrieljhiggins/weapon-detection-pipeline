# Weapon Detection Pipeline

Edge surveillance on **Raspberry Pi 5 + Hailo-8 (AI HAT+ 26 TOPS)** and three **Reolink RLC-520A** cameras.

Detection, person–weapon association, full-body Re-ID, and alerting run on the device. Raw video can be archived off-device. Faces are **not** used for identity.

Viewers are advised to open this repo and see the whole path. **Hailo Apps** stays an install step on the Pi; this repo documents that step and contains pipeline code.

---

## Pipeline

```
1. Collect    3× RTSP → 1-minute clips           collect/cameras_record.py
2. Frames     clips → jpg                        collect/frames_extract.py
3. Label      Label Studio (per camera)
4. Split      merge cams → train/val/test 70/20/10
                                                 collect/dataset_build.py
5. Train      YOLO26s (cloud / PC)               models/  (weights not in git)
6. Compile    .pt → .hef  (x86 DFC, not the Pi)
7. Edge       detect + track + body Re-ID
              + weapon-person link + backtrack   edge/
8. Review     dashboard / clip export            (later)
```

---

## Hardware

| Item | Notes |
|------|--------|
| Raspberry Pi 5 8GB | Active Cooler + official 27W PSU |
| Raspberry Pi AI HAT+ | Hailo-8, 26 TOPS |
| 3× Reolink RLC-520A | PoE cameras |
| Reolink RLA-PS1E | PoE switch |
| Isolated camera LAN | Pi `eth0` = `192.168.50.1/24`, DHCP via dnsmasq |
| Pi Wi-Fi | SSH / GitHub only; cameras never join home Wi-Fi |

---

## Repo layout

```
collect/     capture, stitch, extract, dataset split
config/      cameras.json, systemd unit
detect/      early CPU YOLO test (not the final demo)
edge/        Hailo live pipeline (association + Re-ID)
models/      README only; .pt / .hef stay on disk / Drive
data/        recordings and frames (gitignored)
```

---

## Classes

`person`, `knife`, `axe`, `pistol`, `assault_rifle`, `shotgun`

Identity uses a **full-body** person Re-ID embedding (Hailo RepVGG-A0 512-d), not face recognition. This is something that could be developed in the future.

---

## What is not in git

- `data/recordings`, `data/frames`, Label Studio projects
- `*.pt`, `*.hef` (too large; listed in [`models/README.md`](models/README.md))
- `~/hailo-apps` (vendor SDK installed on the Pi)

---

## Pi software (markers: install once)

Already on the project Pi:

- Raspberry Pi OS Trixie
- HailoRT 4.23
- `hailo-all` 5.1.1
- TAPPAS 5.1.0

Then:

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
source setup_env.sh
```

That install provides GStreamer Hailo elements and the official `repvgg_a0_person_reid_512` body-ReID HEF.

Do **not** use `hailo-reid` — that sample is face-based.

Custom detector HEF is compiled on **x86 Linux** with the Hailo Dataflow Compiler, then copied to `models/`.

---

## Dataset

1. `python3 collect/cameras_record.py` (or systemd `record-cameras.service`)
2. `python3 collect/frames_extract.py`
3. Drop empty frames where there is no action
4. Label each camera in Label Studio, export YOLO
5. `python3 collect/dataset_build.py --root .../frames`

Output: `train_cam123/images/{train,val,test}` + `data.yaml`

---

## Status

| Stage | State |
|-------|--------|
| Hardware, SSH, isolated camera net, continuous recording | Done |
| Dataset + YOLO26s training | Done off-device |
| Hailo Apps + body Re-ID HEF + custom HEF + live association | In progress |

`edge/` will land next so the repo contains the actual association code, not only a description.

---

## Related files

- [`.gitignore`](.gitignore) — recordings, frames, weights, venv, vendor tree
- [`models/README.md`](models/README.md) — weight filenames and where they live (Drive / Pi path)
