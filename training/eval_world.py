"""World-frame person localisation: geometry check + full pipeline error."""
import glob
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
from mot_metrics import iou
from geometry import camera_rotation, cam_to_world
import config

LOG_DIR = os.path.expanduser(config.LOG_ROOT)
OUT = os.path.expanduser("~/thermal-mas-v2/outputs/results/world_localization_results.md")

truth_path = sorted(glob.glob(os.path.join(LOG_DIR, "truth_*.jsonl")))[-1]
truth = [json.loads(l) for l in open(truth_path) if l.strip()]
truth_t = np.array([r["t"] for r in truth])


def true_pos(obj_id, t):
    r = truth[int(np.argmin(np.abs(truth_t - t)))]
    for o in r["objects"]:
        if o["id"] == obj_id:
            return np.array(o["pos"])
    return None


rows = []
for agent in ("pole_cam", "ugv", "uav"):
    path = sorted(glob.glob(os.path.join(LOG_DIR, f"{agent}_*.jsonl")))[-1]
    geo, pipe, pipe_xy = [], [], []
    for fr in (json.loads(l) for l in open(path) if l.strip()):
        if "pose" not in fr:
            continue
        R = camera_rotation(*fr["pose"]["rpy"], config.CAMERA_MOUNT_PITCH[agent])
        gts = [g for g in fr["gt"] if g["cls"] == 0]
        for g in gts:
            tp = true_pos(g["id"], fr["t"])
            if tp is not None:
                geo.append(np.linalg.norm(cam_to_world(g["p_cam"], fr["pose"]["pos"], R) - tp))
        preds = [p for p in fr["pred"] if p["cls"] == "person" and "p_world" in p and np.all(np.isfinite(p["p_world"]))]
        if not gts or not preds:
            continue
        cost = np.array([[1 - iou(g["xyxy"], p["xyxy"]) for p in preds] for g in gts])
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=0.7)
        for gi, pi in enumerate(x):
            if pi < 0:
                continue
            tp = true_pos(gts[gi]["id"], fr["t"])
            if tp is None:
                continue
            pw = np.array(preds[pi]["p_world"])
            pipe.append(np.linalg.norm(pw - tp))
            pipe_xy.append(np.linalg.norm(pw[:2] - tp[:2]))
    rows.append((agent, np.array(geo), np.array(pipe), np.array(pipe_xy)))

lines = [f"Truth log: {os.path.basename(truth_path)}", "",
         "| Agent | Geometry check median (m) | Matched people | Map error median (m) | Map error mean (m) | Map error 90th pct (m) | Ground-plane error median (m) |",
         "|---|---|---|---|---|---|---|"]
for agent, geo, pipe, pxy in rows:
    lines.append(f"| {agent} | {np.median(geo):.2f} | {len(pipe)} | {np.median(pipe):.2f} | {pipe.mean():.2f} | "
                 f"{np.percentile(pipe, 90):.2f} | {np.median(pxy):.2f} |")
allp = np.concatenate([r[2] for r in rows])
allxy = np.concatenate([r[3] for r in rows])
allg = np.concatenate([r[1] for r in rows])
lines.append(f"| all | {np.median(allg):.2f} | {len(allp)} | {np.median(allp):.2f} | {allp.mean():.2f} | "
             f"{np.percentile(allp, 90):.2f} | {np.median(allxy):.2f} |")
table = "\n".join(lines)
print(table)
with open(OUT, "w") as f:
    f.write("# Deliverable 3 (step 2): person positions on the world map (GPS + inertial unit + depth)\n\n" + table + "\n")
print("saved", OUT)
