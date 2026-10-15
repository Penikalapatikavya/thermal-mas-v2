"""Summarise one or more batch CSV files: python3 training/summarize_batch.py rule [rule_nocomm ...]"""
import csv
import os
import sys
import numpy as np

ROOT = os.path.expanduser("~/thermal-mas-v2/outputs/results")
print("| Setup | Runs | Success (all reached) | People reached | Time to reach all (s) | Ghost rescues | Robot-caused collisions | Walk-ins | Path (m) |")
print("|---|---|---|---|---|---|---|---|---|")
for label in sys.argv[1:]:
    rows = list(csv.DictReader(open(os.path.join(ROOT, f"batch_{label}.csv"))))
    faults = sum(r.get('sim_fault') == 'True' for r in rows)
    rows = [r for r in rows if r.get('sim_fault') != 'True']
    label = f"{label} ({faults} sim fault{'s' if faults != 1 else ''} excluded)" if faults else label
    f = lambda k: [float(r[k]) for r in rows if r.get(k) not in (None, "", "None")]
    succ = [r["success"] == "True" for r in rows]
    times = f("time_all")
    reached, people = f("reached"), f("people")
    print(f"| {label} | {len(rows)} | {np.mean(succ)*100:.0f}% | {np.sum(reached):.0f}/{np.sum(people):.0f} | "
          f"{np.mean(times):.1f} ± {np.std(times):.1f} | {np.sum(f('ghost_rescues')):.0f} | "
          f"{np.sum(f('robot_collisions')):.0f} | {np.sum(f('walkin_collisions')):.0f} | {np.mean(f('path_m')):.1f} |")
