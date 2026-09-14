# Models

Trained on the merged three-camera set (`train_cam123`, 70/20/10).
Classes: person, knife, axe, pistol, assault_rifle, shotgun.

## Files in this folder

| File | Notes |
|---|---|
| `yolo26n_cam123_best.pt` | Nano baseline |
| `yolo26s_cam123_best.pt` | Chosen for the Pi |
| `yolo26m_cam123_best.pt` | Highest mAP, heaviest |

## Validation (mAP)

| Class | YOLO26n | YOLO26s | YOLO26m |
|---|---:|---:|---:|
| all mAP50 | 0.658 | 0.721 | 0.740 |
| all mAP50-95 | 0.408 | 0.444 | 0.452 |
| person | 0.989 | 0.983 | 0.991 |
| knife | 0.184 | 0.267 | 0.238 |
| axe | 0.528 | 0.629 | 0.674 |
| pistol | 0.717 | 0.788 | 0.815 |
| assault_rifle | 0.765 | 0.824 | 0.854 |
| shotgun | 0.762 | 0.832 | 0.869 |

Per-class rows are mAP50.

Person is effectively solved. Long guns and axe improve with model size.
Knife stays weak on all three (small object, few / hard views).

## Choice for the live pipeline

Use **YOLO26s**: +0.063 mAP50 over n, close to m, cheaper to run than m.
HEF compile and Re-ID weights are not in the repo yet.