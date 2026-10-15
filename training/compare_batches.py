"""Paired comparison of two batches on the same seeds: python3 training/compare_batches.py rule_v5 rl_v1"""
import csv, os, sys
import numpy as np
R = os.path.expanduser("~/thermal-mas-v2/outputs/results")
a, b = sys.argv[1], sys.argv[2]
load = lambda l: {int(r["seed"]): r for r in csv.DictReader(open(f"{R}/batch_{l}.csv")) if r.get("sim_fault") != "True"}
A, B = load(a), load(b)
seeds = sorted(set(A) & set(B))
t = lambda r: float(r["time_all"]) if r["success"] == "True" else 180.0
d = np.array([t(B[s]) - t(A[s]) for s in seeds])
boot = [np.random.default_rng(k).choice(d, len(d)).mean() for k in range(5000)]
print(f"{len(seeds)} paired seeds | success {a}: {sum(A[s]['success']=='True' for s in seeds)}, "
      f"{b}: {sum(B[s]['success']=='True' for s in seeds)}")
print(f"time ({b} - {a}), failures counted as 180 s: {d.mean():+.2f} s, 95% CI "
      f"[{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}] | {b} faster on {np.sum(d < -0.5)}, "
      f"slower on {np.sum(d > 0.5)}, same on {np.sum(abs(d) <= 0.5)} seeds")
