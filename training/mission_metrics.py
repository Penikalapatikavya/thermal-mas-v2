"""Score one rescue mission against true positions (importable)."""
import json
import math
import numpy as np

EVAL_RADIUS = 2.0
WALKERS = ("pedestrian(1)", "pedestrian(2)")


def evaluate(mission_path, truth_path):
    mission = json.load(open(mission_path))
    truth = [json.loads(l) for l in open(truth_path) if l.strip()]
    tt = np.array([r["t"] for r in truth])

    def snap(t):
        r = truth[int(np.argmin(np.abs(tt - t)))]
        people = {o["id"]: o["pos"][:2] for o in r["objects"] if o["cls"] == 0}
        ugv = next((o["pos"][:2] for o in r["objects"] if o["name"] == "ugv"), None)
        return people, ugv, r

    reached, min_d, robot_coll, walkin_coll, in_coll = {}, float("inf"), 0, 0, False
    sim_fault = any(o['name'] == 'ugv' and o['pos'][2] > 0.5 for r in truth for o in r['objects'])
    for r in truth:
        if r["t"] > mission["end_t"]:
            break
        people, ugv, _ = snap(r["t"])
        if ugv is None:
            continue
        for pid, p in people.items():
            d = math.dist(p, ugv)
            min_d = min(min_d, d)
            if d <= EVAL_RADIUS and pid not in reached:
                reached[pid] = r["t"]
        close = [p for p in people.values() if math.dist(p, ugv) < 0.6]
        if close and not in_coll:
            prev = snap(r["t"] - 1.0)[1]
            p = close[0]
            approach = 0.0
            if prev is not None:
                to_p = np.array(p) - np.array(prev)
                approach = float(np.dot(np.array(ugv) - np.array(prev), to_p / max(np.linalg.norm(to_p), 1e-6)))
            if approach > 0.2:          # robot drove at least 0.2 m toward the person in the last second
                robot_coll += 1
            else:
                walkin_coll += 1
        in_coll = bool(close)
    ghosts = 0
    for res in mission["rescues"]:
        people, _, _ = snap(res["t"])
        if min(math.dist(res["xy"], p) for p in people.values()) > 2.5:
            ghosts += 1
    n_people = len(snap(1.0)[0])
    return {"run_id": mission.get("run_id"), "seed": mission.get("seed"), "policy": mission.get("policy"),
            "fog": (mission.get("conditions") or {}).get("fog"), "comm_loss": mission.get("comm_loss"),
            "sim_fault": sim_fault, "people": n_people, "reached": len(reached), "success": len(reached) == n_people,
            "time_all": max(reached.values()) if len(reached) == n_people else None,
            "claimed": len(mission["rescues"]), "ghost_rescues": ghosts,
            "robot_collisions": robot_coll, "walkin_collisions": walkin_coll,
            "min_dist": round(min_d, 2), "path_m": round(mission["path_len_m"], 1), "end_t": round(mission["end_t"], 1)}
