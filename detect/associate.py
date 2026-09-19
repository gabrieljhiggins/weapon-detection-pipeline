"""Weapon-person assocaition logic. Match weapons to people by proximity and containment.

Long guns (shotgun / rifle) often sit beside the body: the weapon centre is
outside the person box and IoU is tiny. Match by padded containment, any
real overlap, then proximity vs person height.
"""


def centre(box):
    """Return the centre point of a box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def contains(person_box, pt):
    """Return True if the point is inside the person box."""
    x1, y1, x2, y2 = person_box
    px, py = pt
    return x1 <= px <= x2 and y1 <= py <= y2


def inflate(box, frac):
    """Return a box inflated by a fraction of its height."""
    x1, y1, x2, y2 = box
    h = max(1.0, y2 - y1)
    p = h * frac
    return (x1 - p, y1 - 0.15 * p, x2 + p, y2 + 0.15 * p)


def iou(a, b):
    """Return the intersection-over-union of two boxes."""
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


def dist_to_box(pt, box):
    """Return the distance from a point to the nearest edge of a box."""
    px, py = pt
    x1, y1, x2, y2 = box
    cx = min(max(px, x1), x2)
    cy = min(max(py, y1), y2)
    dx, dy = px - cx, py - cy
    return (dx * dx + dy * dy) ** 0.5


def area(box):
    """Return the area of a box."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def owner_for(weapon, people):
    """Return the index of the person who owns the weapon, or None.
    
    Order: centre in a padded person box, then any real overlap, then
    proximity vs person height. Skip people much smaller than the weapon.
    """
    wc = centre(weapon["box"])
    w_area = area(weapon["box"])
    best_i, best = None, None
    for i, p in enumerate(people):
        pb = p["box"]
        ph = max(1.0, pb[3] - pb[1])
        if w_area > 3.0 * max(area(pb), 1.0):
            continue
        d = dist_to_box(wc, pb)
        if contains(inflate(pb, 0.40), wc):
            key = (0, d)
        elif iou(pb, weapon["box"]) > 0.02:
            key = (1, d)
        elif d < 0.50 * ph:
            key = (2, d)
        else:
            continue
        if best is None or key < best:
            best, best_i = key, i
    return best_i


def run(people, weapons):
    """armed = person+weapon, loose = unmatched weapons, idle = unmatched people."""
    armed = []
    loose = []
    used = set()
    for w in weapons:
        owner = owner_for(w, people)
        if owner is None:
            loose.append(w)
            continue
        used.add(owner)
        person = people[owner]
        armed.append({
            "person": person,
            "weapon": w,
            "track_id": person.get("track_id"),
        })
    idle = [p for i, p in enumerate(people) if i not in used]
    return {"armed": armed, "loose": loose, "idle": idle}