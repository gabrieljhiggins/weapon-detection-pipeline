"""Camera timeline per person id. A gap > 3s on the same camera starts a new span."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime


class Backtrack:
    def __init__(self, seconds: int = 300, gap: float = 3.0):
        self.seconds = seconds
        self.gap = gap
        self.spans = defaultdict(list)  # tid -> [{cam,t0,t1,armed}]

    def add(self, cam: str, track_id, box, armed: bool, weapon=None):
        if track_id is None:
            return
        now = datetime.now()
        spans = self.spans[track_id]
        cur = None
        for s in reversed(spans):
            if s["cam"] != cam:
                continue
            if (now - s["t1"]).total_seconds() <= self.gap:
                cur = s
            break
        if cur is None:
            cur = {"cam": cam, "t0": now, "t1": now, "armed": bool(armed), "weapon": weapon}
            spans.append(cur)
        else:
            cur["t1"] = now
            if armed:
                cur["armed"] = True
                cur["weapon"] = weapon
        self._trim(now)

    def _trim(self, now):
        cutoff = now.timestamp() - self.seconds
        for tid in list(self.spans):
            keep = [s for s in self.spans[tid] if s["t1"].timestamp() >= cutoff]
            if keep:
                self.spans[tid] = keep
            else:
                del self.spans[tid]

    def absorb(self, src, dst):
        if src is None or dst is None or src == dst:
            return
        merged = list(self.spans.get(dst, [])) + list(self.spans.get(src, []))
        merged.sort(key=lambda s: s["t0"])
        self.spans[dst] = merged
        self.spans.pop(src, None)

    def dump(self, cam: str, track_id):
        rows = sorted(self.spans.get(track_id, []), key=lambda s: s["t0"])
        if not rows:
            print(f"BACKTRACK id={track_id} (no history)")
            return rows
        cams = []
        for r in rows:
            if r["cam"] not in cams:
                cams.append(r["cam"])
        print(f"BACKTRACK id={track_id} cams={'/'.join(cams)}")
        for r in rows:
            t0 = r["t0"].strftime("%H:%M:%S")
            t1 = r["t1"].strftime("%H:%M:%S")
            flag = "weapon" if r["armed"] else "person"
            if t0 == t1:
                print(f"  {r['cam']} {t0} {flag}")
            else:
                print(f"  {r['cam']} {t0} to {t1} {flag}")
        return rows