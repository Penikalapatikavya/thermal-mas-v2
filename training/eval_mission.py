"""Score a rescue mission against true positions (people + ground robot)."""
import glob
import json
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import config

LOG = os.path.expanduser(config.LOG_ROOT)
EVAL_RADIUS = 2.0   # fixed: independent of the robot's own setting
MIS = os.path.expanduser(config.MISSION_LOG_DIR)
mission_path = sorted(glob.glob(f"{MIS}/mission_*.json"))[-1]
mission = json.load(open(mission_path))
truth_path = sorted(glob.glob(f"{LOG}/truth_*.jsonl"))[-1]
truth = [json.loads(l) for l in open(truth_path) if l.strip()]
tt = np.array([r["t"] for r in truth])


def snapshot(t):
    r = truth[int(np.argmin(np.abs(tt - t)))]
    people = {o["id"]: o["pos"][:2] for o in r["objects"] if o["cls"] == 0}
    ugv = next((o["pos"][:2] for o in r["objects"] if o["name"] == "ugv"), None)
    return people, ugv


reached, min_dist, collisions, in_collision = {}, float("inf"), 0, False
kinds = []
for r in truth:
    if r["t"] > mission["end_t"]:
        break
    people, ugv = snapshot(r["t"])
    if ugv is None:
        continue
    for pid, p in people.items():
        d = math.dist(p, ugv)
        min_dist = min(min_dist, d)
        if d <= EVAL_RADIUS and pid not in reached:
            reached[pid] = r["t"]
    close = [(pid, math.dist(p, ugv)) for pid, p in people.items() if math.dist(p, ugv) < 0.6]
    if close and not in_collision:
        collisions += 1
        names = {o["id"]: o["name"] for o in r["objects"] if o["cls"] == 0}
        prev = snapshot(r["t"] - 1.0)
        ugv_speed = math.dist(ugv, prev[1]) if prev[1] is not None else float("nan")
        for pid, dd in close:
            nm = names.get(pid, "?")
            moved = math.dist(people[pid], prev[0].get(pid, people[pid]))
            kind = "ROBOT-CAUSED" if ugv_speed > 0.2 else "person walked into stationary robot"
            kinds.append(kind)
            print(f"   COLLISION [{kind}] t={r['t']:.1f}s with {nm} ({'walking' if nm in ('pedestrian(1)', 'pedestrian(2)') else 'standing'}), "
                  f"distance {dd:.2f} m, robot moved {ugv_speed:.2f} m and person moved {moved:.2f} m in the last second")
    in_collision = bool(close)

ghosts = []
for res in mission["rescues"]:
    people, _ = snapshot(res["t"])
    nearest = min(math.dist(res["xy"], p) for p in people.values())
    if nearest > 2.5:
        ghosts.append((res["t"], res["xy"], nearest))

n_people = len(snapshot(1.0)[0])
print(f"mission file: {os.path.basename(mission_path)}  (truth: {os.path.basename(truth_path)})")
print(f"real people reached: {len(reached)}/{n_people}  at t = {sorted(round(v, 1) for v in reached.values())}")
print(f"time to reach all: {max(reached.values()):.1f} s" if len(reached) == n_people else "time to reach all: NOT all reached")
print(f"claimed rescues: {len(mission['rescues'])}, ghost rescues: {len(ghosts)}")
for t, xy, d in ghosts:
    print(f"   ghost at t={t:.1f}s map ({xy[0]:.1f}, {xy[1]:.1f}), nearest real person {d:.1f} m away")
print(f"closest approach to any person: {min_dist:.2f} m, collisions (<0.6 m): {collisions} "
      f"(robot-caused: {kinds.count('ROBOT-CAUSED')}, walked into stationary robot: {len(kinds) - kinds.count('ROBOT-CAUSED')})")
print(f"path length: {mission['path_len_m']:.1f} m, sonar near-misses: {mission['sonar_near_misses']}")

# ghost tracks on the ground robot's own map
frames = [json.loads(l) for l in open(mission["perception_log"]) if l.strip()]
far = {}
for fr in frames:
    people, _ = snapshot(fr["t"])
    for tr in fr.get("fused", []):
        d = min(math.dist(tr["xy"], p) for p in people.values())
        if d > 2.5:
            far.setdefault(tr["id"], []).append((fr["t"], tr["xy"], d))
print("ghost tracks on the ground robot's map (seen > 2 s):")
for tid, obs in far.items():
    if len(obs) * (1 / config.PERCEPTION_HZ) > 2:
        xy = np.mean([o[1] for o in obs], 0)
        print(f"   map id {tid}: {len(obs) / config.PERCEPTION_HZ:.0f} s, around ({xy[0]:.1f}, {xy[1]:.1f}), "
              f"{np.median([o[2] for o in obs]):.1f} m from nearest real person, t={obs[0][0]:.0f}-{obs[-1][0]:.0f}s")
