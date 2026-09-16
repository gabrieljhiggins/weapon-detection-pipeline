"""Hailo person Re-ID gallery (repvgg_a0_person_reid_512)."""

from __future__ import annotations

import cv2
import numpy as np


def crop_256x128(frame, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
    return np.ascontiguousarray(cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR))


def l2norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def cosine(a, b):
    return float(np.dot(a, b))


class ReID:
    def __init__(self, infer_fn, in_name: str, match_thr: float = 0.45):
        self.infer_fn = infer_fn
        self.in_name = in_name
        self.match_thr = match_thr
        self.next_id = 1
        self.gallery = {}
        self.attacker_id = None

    def embed(self, frame, box):
        crop = crop_256x128(frame, box)
        if crop is None:
            return None
        raw = self.infer_fn({self.in_name: np.expand_dims(crop, 0)})
        if isinstance(raw, dict):
            raw = next(iter(raw.values()))
        return l2norm(raw)

    def assign(self, frame, people):
        for p in people:
            emb = self.embed(frame, p["box"])
            p["embedding"] = emb
            if emb is None:
                continue
            gid, best = None, self.match_thr
            for existing, vec in self.gallery.items():
                s = cosine(emb, vec)
                if s > best:
                    best, gid = s, existing
            if gid is None:
                gid = self.next_id
                self.next_id += 1
            self.gallery[gid] = emb
            p["track_id"] = gid
        return people

    def mark_attacker(self, gid):
        if gid is not None:
            self.attacker_id = gid

    def is_attacker(self, gid):
        return self.attacker_id is not None and gid == self.attacker_id
