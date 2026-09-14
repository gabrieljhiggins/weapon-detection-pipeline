# Models

Trained on the merged three-camera dataset (`train_cam123`) with a 70/20/10 train–validation–test split.

**Classes:** `person`, `knife`, `axe`, `pistol`, `assault_rifle`, `shotgun`.

## Files

| File                      | Role |
| ------------------------- | ---- |
| `yolo26n_cam123_best.pt`  | YOLO26 Nano baseline |
| `yolo26s_cam123_best.pt`  | Selected live-pipeline weights |
| `yolo26m_cam123_best.pt`  | Highest-accuracy variant; heaviest |
| `yolo26s_cam123_best.hef` | Hailo-8 compiled network for the Raspberry Pi 5 AI HAT+ |

The edge device runs the Hailo Executable Format (`.hef`).

## Validation

Per-class scores below are **mAP50**. Aggregate rows report mAP50 and mAP50–95 on the held-out validation split.

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

`person` is effectively solved. Long guns and axe improve with model size. `knife` stays weak on all three (small object, few / hard views).

## Deployment choice

**YOLO26s** is used on the Pi. It gains +0.063 mAP50 over YOLO26n and is close to YOLO26m while remaining lighter for Hailo-8 compilation and edge inference.

## Hybrid edge–cloud workflow

Training and compilation are conducted on the cloud. While detection, association and alerting are executed on edge devices.

| Stage | Where | Why |
| ----- | ----- | --- |
| Dataset collection | Raspberry Pi + Reolink cameras | Footage is captured on site, relevant frames are kept for training |
| Supervised training | Google Colab (T4 GPU) | Fine-tuning YOLO26 needs a more capable GPU then what the Pi offers |
| ONNX → HEF compile | AWS EC2 x86_64 + Hailo DFC 3.34.0 | The Dataflow Compiler is x86_64-only |
| Live inference | Raspberry Pi 5 + Hailo-8 | Low latency & accurate setup |

This is the hybrid edge–cloud architecture used in the project: compute intensive, infrequent work (train, quantize, compile) runs in the cloud; continuous inference stays on-device for latency and privacy.

### Training compute (Google Colab)

Fine-tuning of YOLO26n / YOLO26s / YOLO26m was run in **Google Colab** from `FP_yolo26.ipynb`, using the merged `train_cam123` set (70/20/10). Colab provides a NVIDIA GPU; the Pi is used only after a `.pt` or `.hef` exists.

Typical flow:

1. Upload or mount `train_cam123.zip` to Google Drive.
2. Train each model in the notebook (Ultralytics YOLO26, 640 input).
3. Export validation metrics and figures.
4. Save `*_best.pt` plus `train_curves/` and `test_plots/` to Drive.

The Pi is not used as a training host.

### Project archive (Google Drive)

Training artefacts that are too large for GitHub are stored [here](https://drive.google.com/drive/folders/1cRBccHCCE-pXnpMpHOmpousu_coObOzT?usp=sharing).

| Item | Size (approx.) | Contents |
| ---- | -------------: | -------- |
| `train_cam123.zip` | 1.12 GB | Merged three-camera dataset used for train/val/test |
| `FP_yolo26.ipynb` | 237 KB | Colab training notebook |
| `yolo26n_cam123_best.pt` | 5.1 MB | Nano model |
| `yolo26s_cam123_best.pt` | 19.4 MB | Small model |
| `yolo26m_cam123_best.pt` | 42 MB | Medium model |
| `train_curves/` | — | Loss / metric curves from training |
| `test_plots/` | — | Qualitative and test-set figures |

### Compilation compute (AWS EC2)

After training, `yolo26s_cam123_best.pt` was exported to ONNX and compiled to Hailo-8 on a **Ubuntu 24.04 LTS (x86_64)** EC2 instance (about 4 vCPU / 16 GB). The resulting `.hef` was copied to `models/` on the Pi and the instance was stopped.

## Hailo-8 compilation

`yolo26s_cam123_best.hef` targets **Hailo-8** (`--hw-arch hailo8`) and was produced with Hailo Dataflow Compiler **3.34.0**.

| Step     | Detail |
| -------- | ------ |
| Export   | Ultralytics ONNX export from `yolo26s_cam123_best.pt` |
| Parse    | DFC ONNX parser. Hailo fused NMS was **not** attached (automatic NMS configuration failed) |
| Calib    | 400 validation frames resized to 640×640 RGB and stored as `.npy` |
| Optimize | Quantization at optimization level 0 (CPU-only VM; DFC used a 64-frame subset) |
| Compile  | Four contexts; compiled artefact ≈ 20 MB |

Box decoding and non-maximum suppression run on the Raspberry Pi CPU after Hailo inference. A later compile on a GPU instance with ≥1024 calibration images could raise quantization quality; it is not required to run the current `.hef`.

### AWS EC2 Procedure

1. Launch Ubuntu 24.04 x86_64 and connect with the AWS `.pem` key (`ubuntu@<public-ip>`).
2. Install DFC 3.34.0 in a Python virtual environment from the Hailo Developer Zone package.
3. If the installer reports missing GPU or Hailo PCIe hardware, set `export HAILO_SKIP_SYSTEM_REQUIREMENTS=1`.
4. Copy `yolo26s_cam123_best.onnx` and the calibration frames to the VM.
5. Parse ONNX to `.har`. Decline automatic NMS if the parser prompt would attach a broken post-process.
6. Convert JPEG calibration images to 640×640 `.npy` tensors.
7. Run `hailo optimize` with `--calib-set-path`, then `hailo compiler`.
8. Save `yolo26s_cam123_best.hef`.
9. Stop the EC2 instance.

```bash
export HAILO_SKIP_SYSTEM_REQUIREMENTS=1
hailo parser onnx yolo26s_cam123_best.onnx --hw-arch hailo8 --har-path yolo26s_cam123.har
hailo optimize yolo26s_cam123.har --hw-arch hailo8 --calib-set-path calib_npy --output-har-path yolo26s_cam123_optimized.har
hailo compiler yolo26s_cam123_optimized.har --hw-arch hailo8
```

### Device check (Raspberry Pi 5 + Hailo-8)

```bash
hailortcli parse-hef models/yolo26s_cam123_best.hef
hailortcli run models/yolo26s_cam123_best.hef
```

Measured on the target device with `hailortcli`. `run` uses synthetic frames (no camera).

| Item | Value |
| ---- | ----- |
| File | `models/yolo26s_cam123_best.hef` (20 MB) |
| Target | Hailo-8 |
| Network | `yolo26s_cam123_best` |
| Contexts | 4 |
| Input | `input_layer1`, UINT8, NHWC 640×640×3 |
| Output | `concat23`, UINT8, FCR 1×8400×10 |
| Frames (`hailortcli run`) | 197 |
| Throughput | **39.15 FPS** |
| Send rate | 384.85 Mbit/s |
| Receive rate | 42.09 Mbit/s |

The output layout `1×8400×10` is consistent with 8400 candidate boxes and 10 values per box (4 box parameters + 6 class scores). NMS is applied on the Pi CPU after this tensor is read from Hailo.

## References

- [Google Drive (dataset, Colab notebook, plots)](https://drive.google.com/drive/folders/1cRBccHCCE-pXnpMpHOmpousu_coObOzT?usp=sharing)
- [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/tree/master/docs)
- [Hailo-8 / 8L software downloads](https://hailo.ai/developer-zone/software-downloads/?product=ai_accelerators&device=hailo_8_8l)
- [Ultralytics Hailo integration](https://docs.ultralytics.com/integrations/hailo)
- [Hailo Dataflow Compiler v3.34.0](https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-34-0/?sp_referrer=install/install.html)