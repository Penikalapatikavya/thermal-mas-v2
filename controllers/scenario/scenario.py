"""Scenario manager: rearranges the scene every episode and logs true object positions."""
import json
import math
import os
import random
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
import config
from controller import Supervisor

EPISODE_S = config.SCENARIO_EPISODE_S
AREA_X = (36.0, 54.0)
AREA_Y = (-54.0, -36.0)
MIN_GAP_M = 2.5
STANDING = ["pedestrian", "pedestrian(3)"]
ALL_PEOPLE = ["pedestrian", "pedestrian(1)", "pedestrian(2)", "pedestrian(3)"]
UGV_X = (41.0, 49.0)
UGV_Y = (-19.0, -12.0)
CENTRE = (45.0, -45.0)
TRUTH_PERIOD = 1.0 / config.PERCEPTION_HZ

sup = Supervisor()
ts = int(sup.getBasicTimeStep())
root = sup.getRoot().getField("children")


def find(name):
    for i in range(root.getCount()):
        node = root.getMFNode(i)
        f = node.getField("name")
        if f is not None and f.getSFString() == name:
            return node
    return None


people = {n: find(n) for n in ALL_PEOPLE}
car = find("vehicle")
ugv = find("ugv")
print("scenario: found", [n for n, nd in people.items() if nd], "| car", bool(car), "| ugv", bool(ugv))

log_dir = os.path.expanduser(config.LOG_ROOT)
os.makedirs(log_dir, exist_ok=True)
truth_path = os.path.join(log_dir, f"truth_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
truth = open(truth_path, "w")
print("scenario: logging true positions to", truth_path)

rng = random.Random(config.SEED)
done_path = os.path.join(os.path.expanduser(config.MISSION_LOG_DIR), f"done_{config.RUN_ID}")
if os.path.exists(done_path):
    os.remove(done_path)
next_episode, next_truth, episode = 0.0, 0.0, 0
while sup.step(ts) != -1:
    t = sup.getTime()
    if config.BATCH and (os.path.exists(done_path) or t > config.MISSION_SECONDS + 10):
        truth.close()
        print(f"scenario: run {config.RUN_ID} finished at t={t:.0f}s, quitting")
        sup.simulationQuit(0)
        break
    if t >= next_episode:
        next_episode = t + EPISODE_S
        episode += 1
        placed = []
        for name in STANDING:
            node = people.get(name)
            if node is None:
                continue
            for _ in range(100):
                x, y = rng.uniform(*AREA_X), rng.uniform(*AREA_Y)
                if all(math.hypot(x - a, y - b) >= MIN_GAP_M for a, b in placed):
                    break
            placed.append((x, y))
            node.getField("translation").setSFVec3f([x, y, 1.27])
            node.getField("rotation").setSFRotation([0, 0, 1, rng.uniform(-math.pi, math.pi)])
        if ugv is not None:
            ux, uy = rng.uniform(*UGV_X), rng.uniform(*UGV_Y)
            yaw = math.atan2(CENTRE[1] - uy, CENTRE[0] - ux) + rng.uniform(-0.35, 0.35)
            ugv.getField("translation").setSFVec3f([ux, uy, 0.0975])
            ugv.getField("rotation").setSFRotation([0, 0, 1, yaw])
            ugv.resetPhysics()
        print(f"scenario: episode {episode} at t={t:.0f}s | standing people "
              + ", ".join(f"({x:.1f}, {y:.1f})" for x, y in placed))
    if t >= next_truth and t <= config.LIVE_SECONDS + 1:
        next_truth = t + TRUTH_PERIOD
        objs = [{"name": n, "id": nd.getId(), "cls": 0, "pos": list(nd.getPosition())}
                for n, nd in people.items() if nd is not None]
        if car is not None:
            objs.append({"name": "vehicle", "id": car.getId(), "cls": 1, "pos": list(car.getPosition())})
        if ugv is not None:
            objs.append({"name": "ugv", "id": ugv.getId(), "cls": -1, "pos": list(ugv.getPosition())})
        truth.write(json.dumps({"t": round(t, 3), "objects": objs}) + "\n")
        truth.flush()
