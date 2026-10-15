"""Sanity-check the 2-D world with the rule-based policies, with and without the pole camera."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from rl_env import RescueEnv, rule_ugv, rule_uav, N_PEOPLE

for pole in (True, False):
    env = RescueEnv(pole=pole)
    times, succ, t0 = [], 0, time.time()
    for ep in range(300):
        env.reset(seed=1000 + ep)
        done = False
        while not done:
            _, _, _, done = env.step((rule_ugv(env), rule_uav(env)))
        if len(env.rescued) == N_PEOPLE:
            succ += 1
            times.append(max(env.rescue_t.values()))
    print(f"pole {'ON ' if pole else 'OFF'}: success {succ / 3:.0f}%  time to reach all {np.mean(times):.1f} ± "
          f"{np.std(times):.1f} s  ({300 / (time.time() - t0):.0f} missions per second)")
