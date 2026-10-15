"""Offline fusion from logged live runs: single agents vs pairs vs all three."""
import glob
import json
import os
import sys
import itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
from mot_metrics import iou
from fusion import make_observation, WorldTracker
import config

LOG = os.path.expanduser(config.LOG_ROOT)
OUT = os.path.expanduser("~/thermal-mas-v2/outputs/results/fusion_results.md")
AGENTS = ("pole_cam", "ugv", "uav")
MATCH_M = 2.0

logs = {a: [json.loads(l) for l in open(sorted(glob.glob(f"{LOG}/{a}_*.jsonl"))[-1]) if l.strip()] for a in AGENTS}
truth = [json.loads(l) for l in open(sorted(glob.glob(f"{LOG}/truth_*.jsonl"))[-1]) if l.strip()]
truth = [r for r in truth if 1.0 <= r["t"] <= config.LIVE_SECONDS]
tt = np.array([r["t"] for r in truth])

# 1) error model including bias (robust RMS of relative depth error)
err_model = {}
for a in AGENTS:
    rel, lat = [], []
    for fr in logs[a]:
        gts = [g for g in fr["gt"] if g["cls"] == 0]
        preds = [p for p in fr["pred"] if p["cls"] == "person" and "p_planar" in p and np.all(np.isfinite(p["p_planar"]))]
        if not gts or not preds:
            continue
        cost = np.array([[1 - iou(g["xyxy"], p["xyxy"]) for p in preds] for g in gts])
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=0.7)
        for gi, pi in enumerate(x):
            if pi >= 0:
                g, p = np.array(gts[gi]["p_cam"]), np.array(preds[pi]["p_planar"])
                r = np.linalg.norm(g); u = g / r; d = p - g
                rel.append((d @ u) / r); lat.append(np.linalg.norm(d - (d @ u) * u))
    err_model[a] = {"range_rel_sigma": round(float(1.4826 * np.median(np.abs(rel))), 4),
                    "lateral_sigma_m": round(max(float(1.4826 * np.median(lat)), 0.1), 3)}
    print(f"{a}: depth sigma {err_model[a]['range_rel_sigma']*100:.1f}% (bias included), sideways {err_model[a]['lateral_sigma_m']:.2f} m")
c = os.path.join(os.path.dirname(__file__), "..", "src", "config.py")
s = open(c).read()
start = s.index("ERROR_MODEL = ")
s = s[:start] + f"ERROR_MODEL = {json.dumps(err_model)}\n"
# config is no longer overwritten here

# 2) observations per agent
obs_by_agent = {}
for a in AGENTS:
    batches = []
    for fr in logs[a]:
        batch = [make_observation(a, fr["t"], p["p_world"], fr["pose"]["pos"], err_model[a])
                 for p in fr["pred"] if p["cls"] == "person" and "p_world" in p and np.all(np.isfinite(p["p_world"]))]
        batches.append((fr["t"], a, batch))
    obs_by_agent[a] = batches


def run(agents, drop=0.0, receiver=None, seed=0):
    rng = np.random.default_rng(seed)
    events = sorted([b for a in agents for b in obs_by_agent[a]], key=lambda e: (e[0], e[1]))
    if receiver is not None and drop > 0:
        events = [ev for ev in events if ev[1] == receiver or rng.random() >= drop]
    trk = WorldTracker()
    ei = 0
    errs, covered, false_tracks, id_switch, false_dist = [], [], [], 0, []
    last_id, ids_used = {}, set()
    for r in truth:
        while ei < len(events) and events[ei][0] <= r["t"]:
            trk.step(events[ei][0], events[ei][2]); ei += 1
        people = [o for o in r["objects"] if o["cls"] == 0]
        tracks = trk.confirmed(r["t"])
        if people and tracks:
            cost = np.array([[np.linalg.norm(np.array(o["pos"][:2]) - xy) for _, xy in tracks] for o in people])
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=MATCH_M)
        else:
            x = [-1] * len(people)
        n_match = 0
        for i, o in enumerate(people):
            if x[i] >= 0:
                n_match += 1
                tid = tracks[x[i]][0]
                errs.append(cost[i, x[i]])
                ids_used.add(tid)
                if o["id"] in last_id and last_id[o["id"]] != tid:
                    id_switch += 1
                last_id[o["id"]] = tid
        covered.append(n_match / max(1, len(people)))
        matched_tracks = {x[i] for i in range(len(people)) if x[i] >= 0}
        for k, (_, xy) in enumerate(tracks):
            if k not in matched_tracks and people:
                false_dist.append(min(np.linalg.norm(np.array(o['pos'][:2]) - xy) for o in people))
        false_tracks.append(len(tracks) - n_match)
    e = np.array(errs) if errs else np.array([np.nan])
    fd = np.array(false_dist) if false_dist else np.array([np.nan])
    return dict(dup=float(np.mean(fd < 3.0)) if false_dist else 0.0, fdist=float(np.median(fd)),
                coverage=np.mean(covered), err_med=np.median(e), err_mean=np.mean(e),
                false=np.mean(false_tracks), idsw=id_switch, ids=len(ids_used))


configs = [(a,) for a in AGENTS] + list(itertools.combinations(AGENTS, 2)) + [AGENTS]
lines = ["| Agents | Coverage of people | Map error median (m) | Map error mean (m) | False tracks per moment | of which duplicates (<3 m) | ID switches | Track IDs (4 people) |",
         "|---|---|---|---|---|---|---|---|"]
for cfg in configs:
    m = run(cfg)
    lines.append(f"| {' + '.join(cfg)} | {m['coverage']:.3f} | {m['err_med']:.2f} | {m['err_mean']:.2f} | "
                 f"{m['false']:.2f} | {m['dup']:.2f} | {m['idsw']} | {m['ids']} |")
    print(lines[-1])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("# Deliverable 3: belief fusion on the world map (offline replay of one live run)\n\n")
    f.write(f"Error model (bias included): {json.dumps(err_model)}\n\n")
    f.write("\n".join(lines) + "\n")
print("saved", OUT)

# communication loss: each agent keeps its own observations, others' messages are dropped with probability p
loss_lines = ["| Receiving agent | Message loss | Coverage | Map error median (m) | Map error mean (m) | False tracks | ID switches |",
              "|---|---|---|---|---|---|---|"]
for rcv in AGENTS:
    for p in (0.0, 0.2, 0.5, 0.8, 1.0):
        ms = [run(AGENTS, drop=p, receiver=rcv, seed=k) for k in range(5 if 0 < p < 1 else 1)]
        avg = {key: float(np.nanmean([m[key] for m in ms])) for key in ("coverage", "err_med", "err_mean", "false", "idsw")}
        row = (f"| {rcv} | {int(p*100)}% | {avg['coverage']:.3f} | {avg['err_med']:.2f} | {avg['err_mean']:.2f} | "
               f"{avg['false']:.2f} | {avg['idsw']:.1f} |")
        loss_lines.append(row)
        print(row)
with open(OUT, "a") as f:
    f.write("\n## Communication loss (5 random seeds per loss level, averaged)\n\n" + "\n".join(loss_lines) + "\n")
print("appended to", OUT)
