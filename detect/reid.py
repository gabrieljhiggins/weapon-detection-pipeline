"""Hailo person Re-ID. Freeze the suspect embedding and match it to other frames. Assign a unique id to each person."""

from __future__ import annotations

import time

import cv2
import numpy as np


def crop_256x128(frame, box):
    """Person crop at the Re-ID HEF size: 128x256 RGB."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
    return np.ascontiguousarray(cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR))


def l2norm(v):
    """L2-normalize a vector."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def cosine(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b))


class ReID:
    def __init__(
        self,
        infer_fn,
        in_name: str,
        match_thr: float = 0.32,
        suspect_thr: float = 0.10,
        hold_s: float = 90.0,
    ):
        self.infer_fn = infer_fn
        self.in_name = in_name
        self.match_thr = match_thr      # gallery match
        self.suspect_thr = suspect_thr  # match against frozen attacker
        self.hold_s = hold_s            # keep suspect id after last sighting
        self.next_id = 1
        self.gallery = {}
        self.attacker_id = None
        self.frozen = None
        self.last_t = 0.0
        self.last_cam = None

    def embed(self, frame, box):
        """512-d unit vector for one person box."""
        crop = crop_256x128(frame, box)
        if crop is None:
            return None
        raw = self.infer_fn({self.in_name: np.expand_dims(crop, 0)})
        if isinstance(raw, dict):
            raw = next(iter(raw.values()))
        return l2norm(raw)

    def assign(self, frame, people, cam=None):
        """Assign a unique track_id to each person in the frame. Remap old ids to new ones."""
        remaps = []
        now = time.time()
        n = len(people)
        recent = self.attacker_id is not None and (now - self.last_t) <= self.hold_s
        hop = cam is not None and cam != self.last_cam

        for p in people:
            emb = self.embed(frame, p["box"])
            p["embedding"] = emb
            gid = None
            if recent and n == 1:
                gid = self.attacker_id
            elif self.attacker_id is not None and self.frozen is not None and emb is not None:
                s = cosine(emb, self.frozen)
                floor = self.suspect_thr - 0.05 if (recent and hop and n <= 2) else self.suspect_thr
                if s >= floor:
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
            if gid is None:
                gid = self.next_id
                self.next_id += 1
            if emb is not None and gid != self.attacker_id:
                self.gallery[gid] = emb
            old = p.get("track_id")
            p["track_id"] = gid
            if old is not None and old != gid:
                remaps.append((old, gid))
            if gid == self.attacker_id:
                self.last_t = now
                if cam is not None:
                    self.last_cam = cam

        if self.attacker_id is not None and recent:
            have = any(p.get("track_id") == self.attacker_id for p in people)
            if not have and people:
                best_p, best_s = None, -1.0
                for p in people:
                    emb = p.get("embedding")
                    if emb is None or self.frozen is None:
                        continue
                    s = cosine(emb, self.frozen)
                    if s > best_s:
                        best_s, best_p = s, p
                take = None
                if n == 1:
                    take = people[0]
                elif hop and n <= 2 and best_p is not None and best_s >= 0.06:
                    take = best_p
                elif best_p is not None and best_s >= 0.10:
                    take = best_p
                if take is not None:
                    old = take.get("track_id")
                    take["track_id"] = self.attacker_id
                    if old is not None and old != self.attacker_id:
                        remaps.append((old, self.attacker_id))
                    self.last_t = now
                    if cam is not None:
                        self.last_cam = cam

        return people, remaps

    def mark_attacker(self, gid, emb=None):
        """Mark a person id as the attacker. Freeze their embedding for future matching."""
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
        """Clear the attacker id and frozen embedding."""
        self.attacker_id = None
        self.frozen = None

    def is_attacker(self, gid):
        """Return True if the given id is the attacker."""
        return self.attacker_id is not None and gid == self.attacker_id