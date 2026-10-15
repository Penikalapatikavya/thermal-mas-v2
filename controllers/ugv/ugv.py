"""Ground robot: rule-based rescue mission on the shared map."""
import json
import math
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from controller import Robot
from thermal_sensor import ThermalSensor
from agent_runtime import LivePerception
from navigation import Navigator
import config
if config.POLICY == "rl":
    from rl_policy import RLPolicy

robot = Robot()
timestep = int(robot.getBasicTimeStep())
sensor = ThermalSensor(robot, "ugv")
live = LivePerception(robot, sensor, "ugv")
nav = Navigator(robot)
policy = RLPolicy() if config.POLICY == "rl" else None
next_decision, rl_target_id, pending_id = 0.0, None, None

reached_ids, reached_pos, rescues = set(), [], []
close_since = {}
idle_since, done, step = None, False, 0
patrol_i = 0


def already_reached(p):
    return p["id"] in reached_ids or any(math.dist(p["xy"], r) < 1.0 for r in reached_pos)


def finish(t, reason):
    log_dir = os.path.expanduser(config.MISSION_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"mission_{config.POLICY}_{config.RUN_ID}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"policy": config.POLICY, "uav_policy": config.UAV_POLICY, "run_id": config.RUN_ID,
               "seed": config.SEED, "pole": config.POLE, "conditions": sensor.fixed[2], "comm_loss": config.COMM_LOSS,
               "end_t": t, "reason": reason,
               "rescues": rescues, "path_len_m": nav.path_len, "sonar_near_misses": nav.near_misses,
               "perception_log": live.log_path}, open(path, "w"), indent=1)
    open(os.path.join(log_dir, f"done_{config.RUN_ID}"), "w").close()
    print(f"ugv: MISSION END ({reason}) t={t:.1f}s rescued={len(rescues)} path={nav.path_len:.1f} m -> {path}")


while robot.step(timestep) != -1:
    sensor.maybe_record()
    live.step()
    step += 1
    t = robot.getTime()
    # yield: scripted walkers do not avoid the robot, so back away from anyone closer than 1 m
    if t >= 1.0 and live.last_fused:
        px, py, pyaw = nav.pose()
        if math.isfinite(px):
            near = min(live.last_fused, key=lambda q: math.dist(q["xy"], (px, py)))
            dn = math.dist(near["xy"], (px, py))
            if dn < 1.0:
                ahead = (near["xy"][0] - px) * math.cos(pyaw) + (near["xy"][1] - py) * math.sin(pyaw)
                v = -3.0 if ahead > 0 else 3.0
                nav.left.setVelocity(v)
                nav.right.setVelocity(v)
                continue
            elif done:
                nav.stop()
    if done or t < 1.0:
        continue
    x, y, _ = nav.pose()
    if not (math.isfinite(x) and math.isfinite(y)):
        continue
    people = live.last_fused
    for p in people:
        if already_reached(p):
            continue
        if p.get("age", 0.0) > 1.0:                    # only a recently seen person can be rescued
            close_since.pop(p["id"], None)
            continue
        if math.dist(p["xy"], (x, y)) > config.RESCUE_RADIUS:
            close_since.pop(p["id"], None)
            continue
        close_since.setdefault(p["id"], t)
        if t - close_since[p["id"]] >= 1.0:          # stayed close for 1 s: a real rescue, not a passing moment
            reached_ids.add(p["id"]); reached_pos.append(p["xy"])
            live.world.mark_rescued(p["id"])
            rescues.append({"t": round(t, 2), "track_id": p["id"], "xy": p["xy"]})
            print(f"ugv: RESCUED person (map id {p['id']}) at t={t:.1f}s  [{len(rescues)} so far]")
    todo = [p for p in people if not already_reached(p)]
    if t > config.MISSION_SECONDS:
        nav.stop(); finish(t, "time limit"); done = True; continue
    if len(rescues) >= config.EXPECTED_PEOPLE:
        nav.stop(); finish(t, "all expected people rescued"); done = True; continue
    live.rescued_info = {"count": len(rescues), "xy": reached_pos}
    target = None
    if todo and policy is None:
        target = min(todo, key=lambda p: math.dist(p["xy"], (x, y)))
    elif todo:
        if t >= next_decision:                         # RL decides once per second, as in training
            next_decision = t + 1.0
            slots = sorted(todo, key=lambda p: math.dist(p["xy"], (x, y)))[:4]
            choice = policy.ugv_choose(policy.obs(live, 0, t, (x, y), slots, len(rescues)), len(slots))
            new_id = slots[choice]["id"] if choice < len(slots) else None
            cur_ok = rl_target_id is not None and any(q["id"] == rl_target_id for q in todo)
            if config.RL_HYSTERESIS and cur_ok and new_id != rl_target_id:
                if pending_id == new_id:                   # same new choice twice in a row: switch
                    rl_target_id, pending_id = new_id, None
                else:
                    pending_id = new_id
            else:
                rl_target_id, pending_id = new_id, None
        target = next((p for p in todo if p["id"] == rl_target_id), None)
    if target is None:                                 # nobody chosen: patrol the area
        wp = config.PATROL[patrol_i]
        if math.dist(wp, (x, y)) < 2.0:
            patrol_i = (patrol_i + 1) % len(config.PATROL)
        nav.drive_to(config.PATROL[patrol_i], [p["xy"] for p in people])
        continue
    idle_since = None
    avoid = [p["xy"] for p in people if p is not target]
    nav.drive_to(target["xy"], avoid)
    if step % 500 == 0:
        print(f"ugv: t={t:.0f}s pos ({x:.1f}, {y:.1f}) heading to map id {target['id']} at "
              f"({target['xy'][0]:.1f}, {target['xy'][1]:.1f}), {len(todo)} left on map")
