"""3D localisation error: tracked person positions vs Webots ground truth (camera frame)."""
import glob
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
from mot_metrics import iou

LOG_DIR = os.path.expanduser("~/thermal-mas-v2/outputs/live_logs")
OUT = os.path.expanduser("~/thermal-mas-v2/outputs/results/localization_results.md")


def to_rub(p):
    """our (forward, left, up) -> graphics convention (right, up, backward)."""
    f, l, u = p
    return np.array([-l, u, -f])


rows = []
for agent in ("pole_cam", "ugv", "uav"):
    logs = sorted(glob.glob(os.path.join(LOG_DIR, f"{agent}_*.jsonl")))
    frames = [json.loads(l) for l in open(logs[-1]) if l.strip()] if logs else []
    err = {k: [] for k in ("range", "planar_flu", "eucl_flu", "planar_rub", "eucl_rub", "true_range")}
    for fr in frames:
        gts = [g for g in fr["gt"] if g["cls"] == 0 and "p_cam" in g]
        preds = [p for p in fr["pred"] if p["cls"] == "person" and "p_eucl" in p and np.isfinite(p["dist"])]
        if not gts or not preds:
            continue
        cost = np.array([[1 - iou(g["xyxy"], p["xyxy"]) for p in preds] for g in gts])
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=0.7)
        for gi, pi in enumerate(x):
            if pi < 0:
                continue
            g, p = np.array(gts[gi]["p_cam"]), preds[pi]
            pl, eu = np.array(p["p_planar"]), np.array(p["p_eucl"])
            err["true_range"].append(np.linalg.norm(g))
            err["range"].append(abs(p["dist"] - np.linalg.norm(g)))
            err["planar_flu"].append(np.linalg.norm(pl - g))
            err["eucl_flu"].append(np.linalg.norm(eu - g))
            err["planar_rub"].append(np.linalg.norm(to_rub(pl) - g))
            err["eucl_rub"].append(np.linalg.norm(to_rub(eu) - g))
    if err["range"]:
        rows.append((agent, len(logs[-1]) and os.path.basename(logs[-1]), {k: np.array(v) for k, v in err.items()}))

print("Which convention fits best (median 3D error in metres, lower = correct):")
for agent, _, e in rows:
    print(f"  {agent:<8} planar+FLU {np.median(e['planar_flu']):6.2f}  eucl+FLU {np.median(e['eucl_flu']):6.2f}  "
          f"planar+RUB {np.median(e['planar_rub']):6.2f}  eucl+RUB {np.median(e['eucl_rub']):6.2f}")

all_e = {k: np.concatenate([e[k] for _, _, e in rows]) for k in rows[0][2]}
best = min(("planar_flu", "eucl_flu", "planar_rub", "eucl_rub"), key=lambda k: np.median(all_e[k]))
print("\nbest overall:", best)

lines = [f"Convention selected: {best}", "",
         "| Agent | Matched people | Typical true distance (m) | Median 3D error (m) | Mean 3D error (m) | 90th pct (m) | Median rel. error |",
         "|---|---|---|---|---|---|---|"]
for agent, _, e in rows + [("all", None, all_e)]:
    b = e[best]
    lines.append(f"| {agent} | {len(b)} | {np.median(e['true_range']):.1f} | {np.median(b):.2f} | {b.mean():.2f} | "
                 f"{np.percentile(b, 90):.2f} | {np.median(b / e['true_range']):.3f} |")
table = "\n".join(lines)
print("\n" + table)
with open(OUT, "w") as f:
    f.write("# Deliverable 3 (step 1): 3D person localisation from thermal detection + depth (camera frame)\n\n" + table + "\n")
print("saved", OUT)
