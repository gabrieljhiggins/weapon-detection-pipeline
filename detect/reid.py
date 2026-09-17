"""Hailo person Re-ID. Freeze the suspect embedding; prefer that id across cams."""

from __future__ import annotations

import time

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
    def __init__(self, infer_fn, in_name: str, match_thr: float = 0.40, suspect_thr: float = 0.22, hold_s: float = 8.0):
        self.infer_fn = infer_fn
        self.in_name = in_name
        self.match_thr = match_thr
        self.suspect_thr = suspect_thr
        self.hold_s = hold_s
        self.next_id = 1
        self.gallery = {}
        self.attacker_id = None
        self.frozen = None
        self.last_t = 0.0
        self.last_cam = None

    def embed(self, frame, box):
        crop = crop_256x128(frame, box)
        if crop is None:
            return None
        raw = self.infer_fn({self.in_name: np.expand_dims(crop, 0)})
        if isinstance(raw, dict):
            raw = next(iter(raw.values()))
        return l2norm(raw)

    def assign(self, frame, people, cam=None):
        now = time.time()
        n = len(people)
        recent = self.attacker_id is not None and (now - self.last_t) <= self.hold_s
        hop = cam is not None and cam != self.last_cam
        for p in people:
            emb = self.embed(frame, p["box"])
            p["embedding"] = emb
            gid = None
            if self.attacker_id is not None and self.frozen is not None and emb is not None:
                s = cosine(emb, self.frozen)
                if s >= self.suspect_thr:
                    gid = self.attacker_id
                elif recent and hop and n == 1 and s >= self.suspect_thr - 0.06:
                    gid = self.attacker_id
            if gid is None and emb is not None:
                best_id, best = None, self.match_thr
                for existing, vec in self.gallery.items():
                    if existing == self.attacker_id:
                        continue
                    score = cosine(emb, vec)
                    if score > best:
                        best, best_id = score, existing
                gid = best_id
            if gid is None and recent and hop and n == 1:
                gid = self.attacker_id
            if gid is None:
                gid = self.next_id
                self.next_id += 1
            if emb is not None and gid != self.attacker_id:
                self.gallery[gid] = emb
            p["track_id"] = gid
            if gid == self.attacker_id:
                self.last_t = now
                if cam is not None:
                    self.last_cam = cam
        return people

    def mark_attacker(self, gid, emb=None):
        if gid is None:
            return
        self.attacker_id = gid
        if emb is not None:
            self.frozen = l2norm(emb)
            self.gallery[gid] = self.frozen
        elif gid in self.gallery:
            self.frozen = self.gallery[gid]
        self.last_t = time.time()

    def clear_attacker(self):
        self.attacker_id = None
        self.frozen = None

    def is_attacker(self, gid):
        return self.attacker_id is not None and gid == self.attacker_id
