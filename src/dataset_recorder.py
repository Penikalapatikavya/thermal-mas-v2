"""Automatic YOLO labels + depth targets from simulated sensors.

Class numbering matches the FLIR-trained model: 0 person, 1 car, 2 bike.
Per-object boxes come from Webots recognition objects (which object is which),
tightened to the visible pixels of that object's colour in the segmentation image.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = {0: "person", 1: "car", 2: "bike"}
CLASS_COLOURS = {(255, 0, 0): 0, (110, 28, 25): 1}
COLOUR_TOLERANCE = 2   # Webots colours are stored as 0-1 floats, allow rounding
MIN_PIXELS = 15        # skip objects that are almost fully hidden or tiny
MAX_DEPTH_M = 100.0


def _match_class(rgb):
    r, g, b = rgb
    if r >= 200 and g <= 5 and b <= 5:   # any shade of red = person
        return 0
    for colour, cls in CLASS_COLOURS.items():
        if all(abs(a - c) <= COLOUR_TOLERANCE for a, c in zip(rgb, colour)):
            return cls
    return None


def labels_from_recognition(seg, objects, with_ids=False):
    """Return a list of (class, x_center, y_center, width, height), normalised 0-1."""
    h, w = seg.shape[:2]
    labels = []
    for obj in objects:
        if obj.getNumberOfColors() < 1:
            continue
        cols = obj.getColors()
        own = np.array([cols[0] * 255, cols[1] * 255, cols[2] * 255])
        cls = _match_class(tuple(own))
        if cls is None:
            continue
        pos = obj.getPositionOnImage()
        size = obj.getSizeOnImage()
        cx, cy = pos[0], pos[1]
        bw, bh = size[0], size[1]
        x0, x1 = max(0, int(cx - bw / 2) - 1), min(w, int(cx + bw / 2) + 2)
        y0, y1 = max(0, int(cy - bh / 2) - 1), min(h, int(cy + bh / 2) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.all(np.abs(seg[y0:y1, x0:x1].astype(np.int16) - own) <= 3, axis=2)
        if mask.sum() < MIN_PIXELS:
            continue
        ys, xs = np.nonzero(mask)
        bx0, bx1 = x0 + xs.min(), x0 + xs.max() + 1
        by0, by1 = y0 + ys.min(), y0 + ys.max() + 1
        lab = (cls, (bx0 + bx1) / 2 / w, (by0 + by1) / 2 / h,
               (bx1 - bx0) / w, (by1 - by0) / h)
        if with_ids:
            pos = obj.getPosition()
            labels.append(lab + (obj.getId(), (pos[0], pos[1], pos[2])))
        else:
            labels.append(lab)
    return labels


def save_sample(root, name, thermal_u8, labels, depth_m, seg=None):
    """Write images/<name>.png, labels/<name>.txt, depth/<name>.npy under root."""
    for sub in ("images", "labels", "depth", "seg"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    Image.fromarray(thermal_u8).save(os.path.join(root, "images", name + ".png"))
    if seg is not None:
        Image.fromarray(np.ascontiguousarray(seg)).save(os.path.join(root, "seg", name + ".png"))
    with open(os.path.join(root, "labels", name + ".txt"), "w") as f:
        for cls, x, y, bw, bh in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
    d = np.nan_to_num(np.asarray(depth_m, np.float32), nan=MAX_DEPTH_M,
                      posinf=MAX_DEPTH_M, neginf=0.0)
    d = d[::2, ::2]  # half resolution for depth training
    np.save(os.path.join(root, "depth", name + ".npy"),
            np.clip(d, 0, MAX_DEPTH_M).astype(np.float16))


def draw_preview(thermal_u8, labels, path):
    """Save the thermal image with boxes drawn, for checking labels by eye."""
    img = Image.fromarray(thermal_u8).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for cls, x, y, bw, bh in labels:
        x0, y0 = (x - bw / 2) * w, (y - bh / 2) * h
        x1, y1 = (x + bw / 2) * w, (y + bh / 2) * h
        colour = (0, 255, 0) if cls == 0 else (255, 160, 0)
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)
        draw.text((x0, max(0, y0 - 11)), CLASS_NAMES[cls], fill=colour)
    img.save(path)
