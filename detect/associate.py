"""Link each weapon box to one person box."""


def centre(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def contains(person_box, pt):
    x1, y1, x2, y2 = person_box
    px, py = pt
    return x1 <= px <= x2 and y1 <= py <= y2


def iou(a, b):
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


def run(people, weapons):
    armed = []
    loose = []
    used = set()
    for w in weapons:
        owner = None
        for i, p in enumerate(people):
            if contains(p["box"], centre(w["box"])):
                owner = i
                break
        if owner is None:
            best_i, best = None, 0.15
            for i, p in enumerate(people):
                v = iou(p["box"], w["box"])
                if v > best:
                    best, best_i = v, i
            owner = best_i
        if owner is None:
            loose.append(w)
        else:
            used.add(owner)
            person = people[owner]
            armed.append({
                "person": person,
                "weapon": w,
                "track_id": person.get("track_id"),
            })
    idle = [p for i, p in enumerate(people) if i not in used]
    return {"armed": armed, "loose": loose, "idle": idle}
