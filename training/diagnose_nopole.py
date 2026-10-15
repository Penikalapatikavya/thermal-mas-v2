"""For failed runs of a batch: was each missed person ever on the robot's map, and what was the robot doing?
Usage: python3 training/diagnose_nopole.py rule_nopole 1 20"""
import glob, json, math, os, re, sys
import numpy as np

R = os.path.expanduser("~/thermal-mas-v2/outputs")
TRUTHS = sorted(glob.glob(f"{R}/live_logs/truth_*.jsonl"))
ts = lambda p: re.search(r"(\d{8}_\d{6})", os.path.basename(p)).group(1)
WALKERS = ("pedestrian(1)", "pedestrian(2)")
label, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
summary = {"never on map": 0, "on map but not reached": 0}
for seed in range(first, first + n):
    mp = sorted(glob.glob(f"{R}/missions/mission_*_{label}_s{seed}_*.json"))[-1]
    m = json.load(open(mp))
    tp = [p for p in TRUTHS if ts(p) <= ts(m["perception_log"])][-1]
    truth = [json.loads(l) for l in open(tp) if l.strip()]
    tt = np.array([r["t"] for r in truth])
    frames = [json.loads(l) for l in open(m["perception_log"]) if l.strip()]
    names = {o["id"]: o["name"] for o in truth[0]["objects"] if o["cls"] == 0}
    reached = set()
    for r in truth:
        if r["t"] > m["end_t"]:
            break
        ugv = next(o["pos"][:2] for o in r["objects"] if o["name"] == "ugv")
        reached |= {o["id"] for o in r["objects"] if o["cls"] == 0 and math.dist(o["pos"][:2], ugv) <= 2.0}
    missed = [pid for pid in names if pid not in reached]
    if not missed:
        continue
    print(f"=== seed {seed}: end t={m['end_t']:.0f}s ({m['reason']}), rescued claims at "
          f"{[round(c['t']) for c in m['rescues']]}")
    for pid in missed:
        on_map, first_seen, closest = 0, None, float("inf")
        for f in frames:
            if f["t"] > m["end_t"]:
                break
            r = truth[int(np.argmin(np.abs(tt - f["t"])))]
            p = next(o["pos"][:2] for o in r["objects"] if o["id"] == pid)
            ugv = next(o["pos"][:2] for o in r["objects"] if o["name"] == "ugv")
            closest = min(closest, math.dist(p, ugv))
            if any(math.dist(tr["xy"], p) < 2.0 for tr in f.get("fused", [])):
                on_map += 1
                first_seen = first_seen if first_seen is not None else f["t"]
        kind = "walking" if names[pid] in WALKERS else "standing"
        share = on_map / max(1, len([f for f in frames if f["t"] <= m["end_t"]]))
        summary["never on map" if on_map == 0 else "on map but not reached"] += 1
        print(f"   missed {names[pid]} ({kind}): on robot's map {share*100:.0f}% of the time, first seen "
              f"{'never' if first_seen is None else f't={first_seen:.0f}s'}, robot came as close as {closest:.1f} m")
    sizes = [len(f.get("fused", [])) for f in frames if m["end_t"] - 30 <= f["t"] <= m["end_t"]]
    print(f"   people on robot's map in the last 30 s: min {min(sizes)}, max {max(sizes)}")
print("totals:", summary)
