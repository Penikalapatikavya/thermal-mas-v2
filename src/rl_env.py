"""Fast 2-D abstraction of the Webots rescue mission for multi-agent RL (ground robot + drone).
Copied from the Webots setup: area, walker paths and speeds, robot speed and turning, careful approach,
drone speed and camera footprint, 8 s forgetting on the shared map."""
import math
import json
import os
import numpy as np

AREA_X, AREA_Y = (36.0, 54.0), (-54.0, -36.0)          # where standing people are placed
UAV_X, UAV_Y = (32.0, 58.0), (-58.0, -30.0)            # drone search area (same grid as Webots)
WALKERS = [((38.0, -52.0), (38.0, -46.0), 0.6), ((53.0, -38.0), (47.0, -38.0), 0.8)]
DT, DECIDE_EVERY, T_MAX = 0.25, 4, 180.0                # agents choose a new action every 1 s
GRID, N_PEOPLE, FORGET_S = 4, 4, 8.0
UGV_SPEED, UGV_TURN = 0.585, 1.5                        # m/s (6 rad/s x 0.0975 m wheel), rad/s
UAV_SPEED, UAV_ALT, UAV_READY = 3.0, 15.0, 9.0          # drone needs ~9 s to take off
VIEW_OFF = UAV_ALT / math.tan(0.9)                      # camera looks 0.9 rad below horizontal
UAV_VIEW_R, UGV_VIEW_R, UGV_HALF_FOV = 8.0, 25.0, 0.6
OBS_DIM = 50
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "rl_env_params.json")
DEFAULT_PARAMS = {"speed": 1.0, "ugv_view_r": 25.0, "ugv_p": 0.8, "uav_view_r": 8.0, "uav_p": 0.95}
if os.path.exists(PARAMS_FILE):
    DEFAULT_PARAMS.update(json.load(open(PARAMS_FILE)))
PATROL = [(40.0, -40.0), (50.0, -40.0), (50.0, -50.0), (40.0, -50.0)]
STATE_DIM = 2 * OBS_DIM + 3 * N_PEOPLE


def norm_xy(p):
    return np.array([(p[0] - 45.0) / 15.0, (p[1] + 45.0) / 15.0])


class RescueEnv:
    def __init__(self, pole=True, seed=None, params=None, randomize=False, hysteresis=False):
        self.pole = pole
        self.randomize, self.hysteresis = randomize, hysteresis
        self.p = dict(DEFAULT_PARAMS, **(params or {}))
        self.rng = np.random.default_rng(seed)
        self.cells = np.array([(UAV_X[0] + (i + 0.5) * (UAV_X[1] - UAV_X[0]) / GRID,
                                UAV_Y[0] + (j + 0.5) * (UAV_Y[1] - UAV_Y[0]) / GRID)
                               for i in range(GRID) for j in range(GRID)])

    # ---------- world ----------
    def walker(self, i, t):
        a, b, v = WALKERS[i]
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        s = (t * v) % (2 * L)
        f = s / L if s <= L else 2 - s / L
        return a + f * (b - a)

    def people(self, t):
        return [self.standing[0], self.standing[1], self.walker(0, t), self.walker(1, t)]

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t, self.step_i = 0.0, 0
        pts = []
        while len(pts) < 2:
            p = np.array([self.rng.uniform(*AREA_X), self.rng.uniform(*AREA_Y)])
            if all(np.linalg.norm(p - q) >= 2.5 for q in pts):
                pts.append(p)
        self.standing = pts
        x, y = self.rng.uniform(41, 49), self.rng.uniform(-19, -12)
        self.ugv = np.array([x, y])
        self.ugv_yaw = math.atan2(-45 - y, 45 - x) + self.rng.uniform(-0.35, 0.35)
        self.uav = np.array([40.0, -22.0])
        self.uav_yaw = -math.pi / 2
        self.known = {}             # person -> [estimated xy, last seen t, estimated velocity]
        self.ever_seen, self.rescued, self.close_since = set(), set(), {}
        self.rescue_t = {}
        self.cell_seen = np.full(len(self.cells), -30.0)
        self.ugv_goal, self.uav_goal = None, None
        self.patrol_i = 0
        if self.randomize:                                 # domain randomisation per mission
            self.ep_speed = self.p["speed"] * self.rng.uniform(0.88, 1.12)
            self.ep_noise = self.rng.uniform(0.0, 0.4)
            self.ep_pole_miss = self.rng.uniform(0.0, 0.15)
        else:
            self.ep_speed, self.ep_noise, self.ep_pole_miss = self.p["speed"], 0.0, 0.0
        self.pending = None
        self._observe()
        return self.obs(), self.state()

    def uav_ground(self):
        return self.uav + VIEW_OFF * np.array([math.cos(self.uav_yaw), math.sin(self.uav_yaw)])

    def _observe(self):
        t = self.t
        P = self.people(t)
        g = self.uav_ground()
        for pid, p in enumerate(P):
            noise = 0.1 + self.ep_noise
            seen = self.pole and (self.ep_pole_miss == 0.0 or self.rng.random() >= self.ep_pole_miss)
            if t >= UAV_READY and np.linalg.norm(p - g) < self.p["uav_view_r"] and self.rng.random() < self.p["uav_p"]:
                seen = True
            d = np.linalg.norm(p - self.ugv)
            bearing = math.atan2(p[1] - self.ugv[1], p[0] - self.ugv[0]) - self.ugv_yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            if d < self.p["ugv_view_r"] and abs(bearing) < UGV_HALF_FOV and self.rng.random() < self.p["ugv_p"]:
                seen = True
                if not self.pole:
                    noise = max(noise, 0.05 * d)
            if seen:
                est = p + self.rng.normal(0, noise, 2)
                if pid in self.known and t > self.known[pid][1]:
                    vel = 0.7 * self.known[pid][2] + 0.3 * (est - self.known[pid][0]) / (t - self.known[pid][1])
                else:
                    vel = np.zeros(2)
                self.known[pid] = [est, t, vel]
                self.ever_seen.add(pid)
        for pid in list(self.known):
            if t - self.known[pid][1] > FORGET_S:
                del self.known[pid]
        if self.pole:
            inside = (self.cells[:, 0] > AREA_X[0] - 4) & (self.cells[:, 0] < AREA_X[1] + 4) & \
                     (self.cells[:, 1] > AREA_Y[0] - 4) & (self.cells[:, 1] < AREA_Y[1] + 4)
            self.cell_seen[inside] = t
        if t >= UAV_READY:
            self.cell_seen[np.linalg.norm(self.cells - g, axis=1) < self.p["uav_view_r"]] = t

    # ---------- what each agent knows ----------
    def slots(self):
        """Known, not-yet-rescued people sorted by distance to the ground robot (same rule in Webots)."""
        ids = [pid for pid in self.known if pid not in self.rescued]
        ids.sort(key=lambda pid: np.linalg.norm(self.known[pid][0] - self.ugv))
        return ids[:N_PEOPLE]

    def _agent_obs(self, me):
        own, own_yaw, other = (self.ugv, self.ugv_yaw, self.uav) if me == 0 else (self.uav, self.uav_yaw, self.ugv)
        o = list(norm_xy(own)) + [math.cos(own_yaw), math.sin(own_yaw)] + list(norm_xy(other)) + [self.t / T_MAX]
        sl = self.slots()
        for k in range(N_PEOPLE):
            if k < len(sl):
                est, ts, vel = self.known[sl[k]]
                o += [1.0, (est[0] - own[0]) / 15, (est[1] - own[1]) / 15, (self.t - ts) / FORGET_S, vel[0], vel[1]]
            else:
                o += [0.0] * 6
        o += [len(self.rescued) / N_PEOPLE]
        o += list(np.minimum(self.t - self.cell_seen, 30.0) / 30.0)
        o += [1.0, 0.0] if me == 0 else [0.0, 1.0]
        return np.array(o, dtype=np.float32)

    def obs(self):
        return [self._agent_obs(0), self._agent_obs(1)]

    def state(self):
        P = self.people(self.t)
        extra = []
        for pid, p in enumerate(P):
            extra += list(norm_xy(p)) + [1.0 if pid in self.rescued else 0.0]
        return np.concatenate([self._agent_obs(0), self._agent_obs(1), np.array(extra, dtype=np.float32)])

    # ---------- acting ----------
    def _ugv_goal_from_action(self, a):
        sl = self.slots()
        if a < N_PEOPLE and a < len(sl):
            return ("person", sl[a])
        wp = np.array(PATROL[self.patrol_i])               # nobody chosen: patrol (same loop as Webots)
        if np.linalg.norm(wp - self.ugv) < 2.0:
            self.patrol_i = (self.patrol_i + 1) % len(PATROL)
            wp = np.array(PATROL[self.patrol_i])
        return ("point", wp)

    def _move_ugv(self):
        if self.ugv_goal is None:
            return
        kind, g = self.ugv_goal
        if kind == "person":
            if g not in self.known or g in self.rescued:
                return
            target, stop = self.known[g][0], 1.4
        else:
            target, stop = g, 1.0
        d = np.linalg.norm(target - self.ugv)
        err = math.atan2(target[1] - self.ugv[1], target[0] - self.ugv[0]) - self.ugv_yaw
        err = (err + math.pi) % (2 * math.pi) - math.pi
        self.ugv_yaw += max(-UGV_TURN * DT, min(UGV_TURN * DT, err))
        if d <= stop:
            return
        speed = UGV_SPEED * self.ep_speed * max(0.0, math.cos(err))
        if kind == "person" and d < 3.0:
            speed *= min(1.0, max(0.25, (d - 1.0) / 2.0))
        self.ugv = self.ugv + speed * DT * np.array([math.cos(self.ugv_yaw), math.sin(self.ugv_yaw)])

    def _move_uav(self):
        if self.t < UAV_READY or self.uav_goal is None:
            return
        cell = self.cells[self.uav_goal]
        ang = math.atan2(cell[1] - self.uav[1], cell[0] - self.uav[0])
        view = cell - VIEW_OFF * np.array([math.cos(ang), math.sin(ang)])
        d = np.linalg.norm(view - self.uav)
        if d < 0.5:
            self.uav_yaw = ang
            return
        step = min(d, UAV_SPEED * DT)
        self.uav_yaw = math.atan2(view[1] - self.uav[1], view[0] - self.uav[0])
        self.uav = self.uav + step * (view - self.uav) / d

    def step(self, actions):
        """actions = (ground robot action 0..4, drone action 0..15), applied for DECIDE_EVERY sim steps."""
        new_goal = self._ugv_goal_from_action(int(actions[0]))
        cur = self.ugv_goal
        cur_ok = cur is not None and cur[0] == "person" and cur[1] in self.known and cur[1] not in self.rescued
        if self.hysteresis and cur_ok and (new_goal[0] != "person" or new_goal[1] != cur[1]):
            key = new_goal[1] if new_goal[0] == "person" else "explore"
            if self.pending == key:                       # same new choice twice in a row: switch
                self.ugv_goal, self.pending = new_goal, None
            else:
                self.pending = key
        else:
            self.ugv_goal, self.pending = new_goal, None
        self.uav_goal = int(actions[1])
        reward, seen_before = 0.0, set(self.ever_seen)
        for _ in range(DECIDE_EVERY):
            self.t += DT
            self._move_ugv()
            self._move_uav()
            self._observe()
            for pid, p in enumerate(self.people(self.t)):
                if pid in self.rescued or pid not in self.known:
                    continue
                if np.linalg.norm(p - self.ugv) <= 1.5:
                    self.close_since.setdefault(pid, self.t)
                    if self.t - self.close_since[pid] >= 1.0:
                        self.rescued.add(pid)
                        self.rescue_t[pid] = self.t
                        reward += 1.0
                else:
                    self.close_since.pop(pid, None)
            reward -= 0.01
            if len(self.rescued) == N_PEOPLE or self.t >= T_MAX:
                break
        reward += 0.2 * len(self.ever_seen - seen_before)
        done = len(self.rescued) == N_PEOPLE or self.t >= T_MAX
        if len(self.rescued) == N_PEOPLE:
            reward += 2.0
        return self.obs(), self.state(), reward, done


# ---------- rule-based policies (same rules as in Webots) ----------
def rule_ugv(env):
    return 0 if env.slots() else N_PEOPLE                  # nearest known person, else explore


def rule_uav(env):
    stale = np.minimum(env.t - env.cell_seen, 30.0)
    oldest = stale.max()
    options = np.where(stale >= oldest - 1.0)[0]
    return int(options[np.argmin(np.linalg.norm(env.cells[options] - env.uav, axis=1))])


def _ugv_mask(self):
    """Valid ground-robot actions: people currently on the map (nearest first) + explore."""
    m = np.zeros(N_PEOPLE + 1, dtype=bool)
    m[:len(self.slots())] = True
    m[N_PEOPLE] = True
    return m


RescueEnv.ugv_mask = _ugv_mask
