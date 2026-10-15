"""Compare person map positions: learned depth vs ground-plane geometry (feet ray hits the ground)."""
import glob
import json
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from geometry import camera_rotation
import config

LOG = os.path.expanduser(config.LOG_ROOT)
W, H = 640, 512
FX = (W / 2) / math.tan(1.2 / 2)
truth = [json.loads(l) for l in open(sorted(glob.glob(f"{LOG}/truth_*.jsonl"))[-1]) if l.strip()]
tt = np.array([r["t"] for r in truth])


def ground_point(cam, R, u, v):
    d = R @ np.array([1.0, -(u - W / 2) / FX, -(v - H / 2) / FX])
    if d[2] > -1e-3:
        return None
    return cam + (-cam[2] / d[2]) * d


rows = []
for agent in ("pole_cam", "ugv", "uav"):
    path = sorted(glob.glob(f"{LOG}/{agent}_*.jsonl"))[-1]
    learned, ground = [], []
    for fr in (json.loads(l) for l in open(path) if l.strip()):
        if "pose" not in fr:
            continue
        cam = np.array(fr["pose"]["pos"])
        R = camera_rotation(*fr["pose"]["rpy"], config.CAMERA_MOUNT_PITCH[agent])
        r = truth[int(np.argmin(np.abs(tt - fr["t"])))]
        people = [np.array(o["pos"][:2]) for o in r["objects"] if o["cls"] == 0]
        for p in fr["pred"]:
            if p["cls"] != "person" or "p_world" not in p or not np.all(np.isfinite(p["p_world"])):
                continue
            x0, y0, x1, y1 = p["xyxy"]
            g = ground_point(cam, R, (x0 + x1) / 2, y1)
            if g is None:
                continue
            # which true person is this? the one closest to the viewing direction (sideways error is tiny)
            c = R @ np.array([1.0, -((x0 + x1) / 2 - W / 2) / FX, 0.0])
            dir2 = c[:2] / np.linalg.norm(c[:2])
            best, best_perp = None, 1.0
            for tp in people:
                rel = tp - cam[:2]
                perp = abs(rel[0] * dir2[1] - rel[1] * dir2[0])
                if rel @ dir2 > 0 and perp < best_perp:
                    best, best_perp = tp, perp
            if best is None:
                continue
            learned.append(np.linalg.norm(np.array(p["p_world"][:2]) - best))
            ground.append(np.linalg.norm(g[:2] - best))
    rows.append((agent, np.array(learned), np.array(ground)))

print("| Agent | People seen | Learned depth: median / mean / 90th pct / >2 m | Ground plane: median / mean / 90th pct / >2 m |")
print("|---|---|---|---|")
for agent, a, b in rows:
    f = lambda x: f"{np.median(x):.2f} / {x.mean():.2f} / {np.percentile(x, 90):.2f} m / {(x > 2).mean()*100:.0f}%"
    print(f"| {agent} | {len(a)} | {f(a)} | {f(b)} |")
