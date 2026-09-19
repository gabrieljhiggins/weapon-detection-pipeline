"""Per-camera tracker for person boxes."""

from __future__ import annotations


def iou(a, b):
    """Compute the intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class CameraTracker:
    def __init__(self, cam: str, max_age: int = 20, min_iou: float = 0.25):
        self.cam = cam
        self.max_age = max_age
        self.min_iou = min_iou
        self.next_id = 1
        self.tracks = {}  # id -> {box, age, hits}

    def update(self, people: list[dict]) -> list[dict]:
        """Greedy IoU match. New boxes get a new local id."""
        assigned = set()
        used_det = set()
        for tid, tr in list(self.tracks.items()):
            best_j, best = None, self.min_iou
            for j, p in enumerate(people):
                if j in used_det:
                    continue
                v = iou(tr["box"], p["box"])
                if v > best:
                    best, best_j = v, j
            if best_j is None:
                tr["age"] += 1
                continue
            people[best_j]["track_id"] = tid
            tr["box"] = people[best_j]["box"]
            tr["age"] = 0
            tr["hits"] += 1
            assigned.add(tid)
            used_det.add(best_j)
        for j, p in enumerate(people):
            if j in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            p["track_id"] = tid
            self.tracks[tid] = {"box": p["box"], "age": 0, "hits": 1}
            assigned.add(tid)
        dead = [tid for tid, tr in self.tracks.items() if tr["age"] > self.max_age]
        for tid in dead:
            del self.tracks[tid]
        return people


class Trackers:
    """Maintain a CameraTracker for each camera."""

    def __init__(self):
        self.by_cam = {}

    def update(self, cam: str, people: list[dict]) -> list[dict]:
        """Update the tracker for a given camera and return the people with track_ids."""
        if cam not in self.by_cam:
            self.by_cam[cam] = CameraTracker(cam)
        return self.by_cam[cam].update(people)
