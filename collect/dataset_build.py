#!/usr/bin/env python3
"""Merge Label Studio YOLO exports for Camera 1/2/3 into one dataset.

This code is provided for reference only and documents the dataset building process. 
Training (70%), Valuation (20%), Test (10%) splits are created.
The annotated frames would subsequently be used for model training and evaluation. 
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

DEFAULT_ROOT = Path("/Users/gabrielhiggins/Desktop/frames")
CAMERAS = ("Camera 1", "Camera 2", "Camera 3")


def read_classes(cam_dir: Path) -> list[str]:
    p = cam_dir / "classes.txt"
    if not p.is_file():
        raise SystemExit(f"Missing {p}")
    names = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if not names:
        raise SystemExit(f"Empty {p}")
    return names


def list_images(cam_dir: Path) -> list[Path]:
    img_dir = cam_dir / "images"
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not imgs:
        raise SystemExit(f"No images in {img_dir}")
    return imgs


def pick_split(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.70:
        return "train"
    if r < 0.90:
        return "val"
    return "test"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    root = args.root
    out = args.out or (root / "train_cam123")
    cams = [root / name for name in CAMERAS]
    for d in cams:
        if not d.is_dir():
            raise SystemExit(f"Missing {d}")

    classes = read_classes(cams[0])
    if out.exists():
        shutil.rmtree(out)

    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0, "test": 0}

    for cam in cams:
        tag = cam.name.replace(" ", "").lower()  # camera1
        for img in list_images(cam):
            split = pick_split(rng)
            img_dst_dir = out / "images" / split
            lab_dst_dir = out / "labels" / split
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            lab_dst_dir.mkdir(parents=True, exist_ok=True)

            dst_img = img_dst_dir / f"{tag}_{img.name}"
            shutil.copy2(img, dst_img)

            src_lab = cam / "labels" / (img.stem + ".txt")
            dst_lab = lab_dst_dir / f"{tag}_{img.stem}.txt"
            if src_lab.is_file():
                shutil.copy2(src_lab, dst_lab)
            else:
                dst_lab.write_text("")

            counts[split] += 1

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names}\n"
    )

    print(classes)
    print(f"train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()