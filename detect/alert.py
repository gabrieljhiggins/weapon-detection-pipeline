"""Local alert when a tracked person is armed."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from common import ALERT_ROOT


def run(cam: str, assoc: dict, frame, save: bool = True) -> list[dict]:
    alerts = []
    ALERT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for item in assoc["armed"]:
        person = item["person"]
        weapon = item["weapon"]
        tid = person.get("track_id")
        event = {
            "time": ts,
            "cam": cam,
            "track_id": tid,
            "weapon": weapon["name"],
            "weapon_score": weapon["score"],
            "person_score": person["score"],
        }
        line = (
            f"ALERT {ts} {cam} id={tid} "
            f"{weapon['name']}={weapon['score']:.2f} person={person['score']:.2f}"
        )
        print(line)
        if save and frame is not None:
            out = ALERT_ROOT / f"{cam}_{file_ts}_id{tid}_{weapon['name']}.jpg"
            cv2.imwrite(str(out), frame)
            event["file"] = str(out)
        alerts.append(event)
    return alerts
