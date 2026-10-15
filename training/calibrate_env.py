"""Find training-world parameters that reproduce the Webots rule-based results (with and without pole)."""
import csv, itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from rl_env import RescueEnv, rule_ugv, rule_uav, N_PEOPLE, PARAMS_FILE

RES = os.path.expanduser("~/thermal-mas-v2/outputs/results")


def webots(label):
    rows = list(csv.DictReader(open(f"{RES}/batch_{label}.csv")))
    rows = [r for r in rows if r.get("sim_fault") != "True"]
    times = [float(r["time_all"]) for r in rows if r["time_all"] not in ("", "None")]
    return {"success": np.mean([r["success"] == "True" for r in rows]),
            "reached": sum(float(r["reached"]) for r in rows) / sum(float(r["people"]) for r in rows),
            "time": float(np.mean(times))}


def simulate(params, pole, n=100):
    env = RescueEnv(pole=pole, params=params)
    times, succ, reached = [], 0, 0
    for ep in range(n):
        env.reset(seed=5000 + ep)
        done = False
        while not done:
            _, _, _, done = env.step((rule_ugv(env), rule_uav(env)))
        reached += len(env.rescued)
        if len(env.rescued) == N_PEOPLE:
            succ += 1
            times.append(max(env.rescue_t.values()))
    return {"success": succ / n, "reached": reached / (n * N_PEOPLE), "time": float(np.mean(times)) if times else 999.0}


on, off = webots("rule_v3"), webots("rule_nopole")
print(f"Webots targets  pole ON: time {on['time']:.1f} s | pole OFF: success {off['success']*100:.0f}%, "
      f"reached {off['reached']*100:.0f}%, time {off['time']:.1f} s")
results, t0 = [], time.time()
for speed, ugv_r, uav_r in itertools.product([0.80, 0.85, 0.90, 0.95], [8, 12, 16, 20], [4, 6, 8]):
    prm = {"speed": speed, "ugv_view_r": ugv_r, "uav_view_r": uav_r}
    a, b = simulate(prm, True), simulate(prm, False)
    loss = ((a["time"] - on["time"]) / 10) ** 2 + ((b["success"] - off["success"]) / 0.1) ** 2 \
         + ((b["reached"] - off["reached"]) / 0.05) ** 2 + ((b["time"] - off["time"]) / 10) ** 2
    results.append((loss, prm, a, b))
results.sort(key=lambda r: r[0])
print(f"tested {len(results)} combinations in {time.time() - t0:.0f} s; best 5:")
for loss, prm, a, b in results[:5]:
    print(f"  loss {loss:5.2f}  {prm} -> pole ON time {a['time']:.1f} s | pole OFF success {b['success']*100:.0f}%, "
          f"reached {b['reached']*100:.0f}%, time {b['time']:.1f} s")
best = results[0][1]
json.dump(best, open(PARAMS_FILE, "w"), indent=1)
print("saved", PARAMS_FILE)
