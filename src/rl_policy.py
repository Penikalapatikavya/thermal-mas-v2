"""Run the trained MAPPO actors inside Webots: builds the same 50-number observation as the training world."""
import math
import numpy as np
import torch
import torch.nn as nn
import config
from rl_env import (N_PEOPLE, OBS_DIM, T_MAX, FORGET_S, AREA_X, AREA_Y, UAV_X, UAV_Y, GRID,
                    VIEW_OFF, UAV_READY, DEFAULT_PARAMS, norm_xy)

CELLS = np.array([(UAV_X[0] + (i + 0.5) * (UAV_X[1] - UAV_X[0]) / GRID,
                   UAV_Y[0] + (j + 0.5) * (UAV_Y[1] - UAV_Y[0]) / GRID) for i in range(GRID) for j in range(GRID)])
INSIDE = (CELLS[:, 0] > AREA_X[0] - 4) & (CELLS[:, 0] < AREA_X[1] + 4) & \
         (CELLS[:, 1] > AREA_Y[0] - 4) & (CELLS[:, 1] < AREA_Y[1] + 4)


def update_cells(live, t):
    """Same freshness bookkeeping as the training world (pole area + drone camera footprint)."""
    if config.POLE:
        live.cell_seen[INSIDE] = t
    pose = live.own_pose if live.agent == "uav" else (
        (live.others["uav"]["pos"], live.others["uav"]["yaw"]) if "uav" in live.others else None)
    if pose is not None and t >= UAV_READY:
        g = pose[0] + VIEW_OFF * np.array([math.cos(pose[1]), math.sin(pose[1])])
        live.cell_seen[np.linalg.norm(CELLS - g, axis=1) < DEFAULT_PARAMS["uav_view_r"]] = t


def _mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class RLPolicy:
    def __init__(self, path=config.RL_WEIGHTS):
        ck = torch.load(path, map_location="cpu")
        self.ugv, self.uav = _mlp(OBS_DIM, N_PEOPLE + 1, 128), _mlp(OBS_DIM, 16, 128)
        self.ugv.load_state_dict({k.replace("net.", ""): v for k, v in ck["ugv"].items()})
        self.uav.load_state_dict({k.replace("net.", ""): v for k, v in ck["uav"].items()})
        print(f"rl_policy: loaded {path}")

    def obs(self, live, me, t, ugv_xy, slots, n_rescued):
        """me: 0 = ground robot, 1 = drone. slots: up to 4 map entries, nearest to the ground robot first."""
        own, own_yaw = live.own_pose
        other_name = "uav" if me == 0 else "ugv"
        other = live.others[other_name]["pos"] if other_name in live.others else own
        o = list(norm_xy(own)) + [math.cos(own_yaw), math.sin(own_yaw)] + list(norm_xy(other)) + [t / T_MAX]
        for k in range(N_PEOPLE):
            if k < len(slots):
                p = slots[k]
                v = p.get("v", [0.0, 0.0])
                o += [1.0, (p["xy"][0] - own[0]) / 15, (p["xy"][1] - own[1]) / 15,
                      min(p.get("age", 0.0), FORGET_S) / FORGET_S, v[0], v[1]]
            else:
                o += [0.0] * 6
        o += [min(n_rescued, N_PEOPLE) / N_PEOPLE]
        o += list(np.minimum(t - live.cell_seen, 30.0) / 30.0)
        o += [1.0, 0.0] if me == 0 else [0.0, 1.0]
        return torch.as_tensor(np.array(o, dtype=np.float32))

    def ugv_choose(self, ob, n_slots):
        with torch.no_grad():
            logits = self.ugv(ob)
        mask = torch.zeros(N_PEOPLE + 1, dtype=torch.bool)
        mask[:n_slots] = True
        mask[N_PEOPLE] = True
        logits[~mask] = -1e9
        return int(logits.argmax())

    def uav_choose(self, ob):
        with torch.no_grad():
            return int(self.uav(ob).argmax())
