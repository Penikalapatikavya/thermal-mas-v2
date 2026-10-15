"""Held-out paired comparison (rule vs MAPPO on the SAME fresh missions) + behaviour analysis."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
import torch.nn as nn
from rl_env import RescueEnv, rule_ugv, rule_uav, N_PEOPLE, OBS_DIM

POLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N = 500


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


ck = torch.load(os.path.expanduser(f"~/thermal-mas-v2/weights/mappo_pole{POLE}.pt"))
ugv, uav = mlp(OBS_DIM, N_PEOPLE + 1, 128), mlp(OBS_DIM, 16, 128)
ugv.load_state_dict({k.replace("net.", ""): v for k, v in ck["ugv"].items()})
uav.load_state_dict({k.replace("net.", ""): v for k, v in ck["uav"].items()})


def rl_actions(env, obs):
    with torch.no_grad():
        l0 = ugv(torch.as_tensor(obs[0]))
        l0[~torch.as_tensor(env.ugv_mask())] = -1e9
        return int(l0.argmax()), int(uav(torch.as_tensor(obs[1])).argmax())


def run(policy, seed, stats=None):
    env = RescueEnv(pole=bool(POLE))
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        if policy == "rule":
            a = (rule_ugv(env), rule_uav(env))
        else:
            a = rl_actions(env, obs)
            if stats is not None:
                sl = env.slots()
                if len(sl) >= 2:
                    stats["choices"] += 1
                    if a[0] != 0:
                        stats["not_nearest"] += 1
                        stats["not_nearest_walker"] += sl[a[0]] >= 2 if a[0] < len(sl) else 0
                    stats["nearest_is_walker"] += sl[0] >= 2
                stats["uav_cells"][a[1]] += 1
        obs, _, _, done = env.step(a)
    ok = len(env.rescued) == N_PEOPLE
    return ok, (max(env.rescue_t.values()) if ok else 180.0)


stats = {"choices": 0, "not_nearest": 0, "not_nearest_walker": 0, "nearest_is_walker": 0, "uav_cells": np.zeros(16)}
rule = [run("rule", 20000 + k) for k in range(N)]
rl = [run("rl", 20000 + k, stats) for k in range(N)]
rt, lt = np.array([t for _, t in rule]), np.array([t for _, t in rl])
diff = lt - rt                                      # negative = RL faster on that mission
rng = np.random.default_rng(0)
boot = [rng.choice(diff, N).mean() for _ in range(5000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"held-out missions: {N} (seeds 20000+), pole {'ON' if POLE else 'OFF'}")
print(f"rule:  success {np.mean([o for o, _ in rule])*100:.1f}%, time {rt.mean():.1f} ± {rt.std():.1f} s")
print(f"MAPPO: success {np.mean([o for o, _ in rl])*100:.1f}%, time {lt.mean():.1f} ± {lt.std():.1f} s")
print(f"paired difference (RL - rule): {diff.mean():+.2f} s, 95% CI [{lo:+.2f}, {hi:+.2f}] s  "
      f"-> RL faster in {np.mean(diff < -0.5)*100:.0f}% of missions, slower in {np.mean(diff > 0.5)*100:.0f}%, "
      f"same in {np.mean(abs(diff) <= 0.5)*100:.0f}%")
c = stats["choices"]
print(f"ground robot, decisions with 2+ people on the map: {c}")
print(f"   RL chose someone other than the nearest: {stats['not_nearest']/c*100:.1f}% of them "
      f"({stats['not_nearest_walker']} of those choices were a walker)")
print(f"   the nearest person was a walker in {stats['nearest_is_walker']/c*100:.1f}% of those decisions")
top = np.argsort(stats["uav_cells"])[::-1][:4]
print(f"drone: most chosen cells {list(top)} = {[f'{stats['uav_cells'][i]/stats['uav_cells'].sum()*100:.0f}%' for i in top]} of its choices")
