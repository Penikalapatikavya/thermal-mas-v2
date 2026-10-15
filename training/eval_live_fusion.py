"""Score each agent's own live fused map (built from radio messages) against true positions."""
import glob
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
import config

LOG = os.path.expanduser(config.LOG_ROOT)
OUT = os.path.expanduser("~/thermal-mas-v2/outputs/results/live_fusion_results.md")
MATCH_M = 2.0
truth = [json.loads(l) for l in open(sorted(glob.glob(f"{LOG}/truth_*.jsonl"))[-1]) if l.strip()]
tt = np.array([r["t"] for r in truth])

lines = [f"COMM_LOSS = {config.COMM_LOSS}", "",
         "| Agent's own map | Frames | Coverage | Map error median (m) | Map error mean (m) | False tracks | ID switches | Track IDs | Msgs sent | Msgs received |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for agent in ("pole_cam", "ugv", "uav"):
    frames = [json.loads(l) for l in open(sorted(glob.glob(f"{LOG}/{agent}_*.jsonl"))[-1]) if l.strip()]
    frames = [f for f in frames if "fused" in f]
    errs, cov, false, idsw, last, ids = [], [], [], 0, {}, set()
    for fr in frames:
        r = truth[int(np.argmin(np.abs(tt - fr["t"])))]
        people = [o for o in r["objects"] if o["cls"] == 0]
        tracks = fr["fused"]
        if people and tracks:
            cost = np.array([[np.linalg.norm(np.array(o["pos"][:2]) - np.array(tr["xy"])) for tr in tracks] for o in people])
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=MATCH_M)
        else:
            x = [-1] * len(people)
        n = 0
        for i, o in enumerate(people):
            if x[i] >= 0:
                n += 1
                tid = tracks[x[i]]["id"]
                errs.append(cost[i, x[i]]); ids.add(tid)
                if o["id"] in last and last[o["id"]] != tid:
                    idsw += 1
                last[o["id"]] = tid
        cov.append(n / max(1, len(people)))
        false.append(len(tracks) - n)
    e = np.array(errs)
    c = frames[-1]["comm"]
    lines.append(f"| {agent} | {len(frames)} | {np.mean(cov):.3f} | {np.median(e):.2f} | {e.mean():.2f} | "
                 f"{np.mean(false):.2f} | {idsw} | {len(ids)} | {c['sent']} | {c['received']} |")
table = "\n".join(lines)
print(table)
with open(OUT, "w") as f:
    f.write("# Deliverable 3: live decentralised fusion over the radio (each agent's own map)\n\n" + table + "\n")
print("saved", OUT)
