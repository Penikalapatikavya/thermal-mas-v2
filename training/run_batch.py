"""Run many headless missions in a row and score each one against the truth.
Usage: python3 training/run_batch.py <label> <first_seed> <number_of_runs>"""
import csv
import glob
import os
import subprocess
import sys
import time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mission_metrics import evaluate

WEBOTS = "/Applications/Webots.app/Contents/MacOS/webots"
ROOT = os.path.expanduser("~/thermal-mas-v2")
WORLD = os.path.join(ROOT, "worlds", "rescue_city.wbt")
label, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out_csv = os.path.join(ROOT, "outputs", "results", f"batch_{label}.csv")
log_dir = os.path.join(ROOT, "outputs", "batch_logs")
os.makedirs(log_dir, exist_ok=True)
rows = []
for seed in range(first, first + n):
    run_id = f"{label}_s{seed}"
    env = dict(os.environ, RUN_ID=run_id, SEED=str(seed), BATCH="1")
    t0 = time.time()
    with open(os.path.join(log_dir, f"{run_id}.log"), "w") as lf:
        try:
            subprocess.run([WEBOTS, "--mode=fast", "--no-rendering", "--minimize", "--batch",
                            "--stdout", "--stderr", WORLD], env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=2400)
        except subprocess.TimeoutExpired:
            print(f"{run_id}: TIMEOUT")
    missions = sorted(glob.glob(os.path.join(ROOT, "outputs", "missions", f"mission_*_{run_id}_*.json")))
    truths = [p for p in glob.glob(os.path.join(ROOT, "outputs", "live_logs", "truth_*.jsonl")) if os.path.getmtime(p) >= t0]
    if not missions or not truths:
        print(f"{run_id}: no mission result (see outputs/batch_logs/{run_id}.log)")
        rows.append({"run_id": run_id, "seed": seed, "success": False, "reached": 0})
        continue
    m = evaluate(missions[-1], sorted(truths)[-1])
    m["wall_s"] = round(time.time() - t0)
    rows.append(m)
    print(f"{run_id}: reached {m['reached']}/{m['people']}, time {m['time_all']}, ghosts {m['ghost_rescues']}, "
          f"robot collisions {m['robot_collisions']}, walk-ins {m['walkin_collisions']}, path {m['path_m']} m, "
          f"fog {m['fog']}  ({m['wall_s']} s real time)")
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
print("saved", out_csv)
