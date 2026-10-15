"""Robot path and nearby people every 5 s for one batch run: python3 training/show_path.py 11"""
import glob, json, math, os, re, sys
import numpy as np
R = os.path.expanduser("~/thermal-mas-v2/outputs")
ts = lambda p: re.search(r"(\d{8}_\d{6})", os.path.basename(p)).group(1)
seed = sys.argv[1]
m = json.load(open(sorted(glob.glob(f"{R}/missions/mission_rule_rule_s{seed}_*.json"))[-1]))
tp = [p for p in sorted(glob.glob(f"{R}/live_logs/truth_*.jsonl")) if ts(p) <= ts(m["perception_log"])][-1]
truth = [json.loads(l) for l in open(tp) if l.strip()]
next_t = 0
for r in truth:
    if r["t"] < next_t or r["t"] > m["end_t"]:
        continue
    next_t = r["t"] + 5
    ugv = next(o["pos"] for o in r["objects"] if o["name"] == "ugv")
    car = next((o["pos"] for o in r["objects"] if o["name"] == "vehicle"), None)
    near = min((math.dist(o["pos"][:2], ugv[:2]), o["name"]) for o in r["objects"] if o["cls"] == 0)
    car_d = math.dist(car[:2], ugv[:2]) if car else float("nan")
    print(f"t={r['t']:5.0f}s robot ({ugv[0]:5.1f}, {ugv[1]:5.1f}) z {ugv[2]:.2f} | nearest person "
          f"{near[1]} {near[0]:.1f} m | car {car_d:.1f} m")
