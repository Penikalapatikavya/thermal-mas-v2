"""List robot-caused collisions in a batch: python3 training/collision_report.py rule_v2 1 20"""
import glob, json, math, os, re, sys
import numpy as np

R = os.path.expanduser("~/thermal-mas-v2/outputs")
TRUTHS = sorted(glob.glob(f"{R}/live_logs/truth_*.jsonl"))
ts = lambda p: re.search(r"(\d{8}_\d{6})", os.path.basename(p)).group(1)
label, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
counts = {"walking": 0, "standing": 0, "during approach": 0, "other": 0}
for seed in range(first, first + n):
    mp = sorted(glob.glob(f"{R}/missions/mission_*_{label}_s{seed}_*.json"))[-1]
    m = json.load(open(mp))
    tp = [p for p in TRUTHS if ts(p) <= ts(m["perception_log"])][-1]
    truth = [json.loads(l) for l in open(tp) if l.strip()]
    tt = np.array([r["t"] for r in truth])
    in_coll = False
    for r in truth:
        if r["t"] > m["end_t"]:
            break
        ugv = next(o["pos"][:2] for o in r["objects"] if o["name"] == "ugv")
        close = [o for o in r["objects"] if o["cls"] == 0 and math.dist(o["pos"][:2], ugv) < 0.6]
        if close and not in_coll:
            prev = truth[int(np.argmin(np.abs(tt - (r["t"] - 1.0))))]
            pu = next(o["pos"][:2] for o in prev["objects"] if o["name"] == "ugv")
            o = close[0]
            pp = next(q["pos"][:2] for q in prev["objects"] if q["id"] == o["id"])
            to_p = np.array(o["pos"][:2]) - np.array(pu)
            approach = float(np.dot(np.array(ugv) - np.array(pu), to_p / max(np.linalg.norm(to_p), 1e-6)))
            if approach > 0.2:
                kind = "walking" if o["name"] in ("pedestrian(1)", "pedestrian(2)") else "standing"
                nxt = [c["t"] for c in m["rescues"] if c["t"] >= r["t"] - 1.0]
                phase = "during approach" if nxt and nxt[0] - r["t"] < 6 else "other"
                counts[kind] += 1
                counts[phase] += 1
                print(f"seed {seed:2d} t={r['t']:5.1f}s {o['name']} ({kind}) dist {math.dist(o['pos'][:2], ugv):.2f} m, "
                      f"robot moved {approach:.2f} m toward them, person moved {math.dist(o['pos'][:2], pp):.2f} m, "
                      f"next rescue claim {('t=' + format(nxt[0], '.1f') + 's') if nxt else 'none'} -> {phase}")
        in_coll = bool(close)
print("totals:", counts)
