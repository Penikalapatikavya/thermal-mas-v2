"""Belief fusion: observations with uncertainty -> Kalman-filter tracks on the world map."""
import numpy as np
import lap

H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def make_observation(agent, t, p_world, cam_pos, err):
    """Map position (x, y) + 2x2 covariance from range/sideways uncertainty along the line of sight."""
    p, c = np.asarray(p_world, float), np.asarray(cam_pos, float)
    v = p - c
    r = np.linalg.norm(v)
    u = v / r
    s_r = max(err["range_rel_sigma"] * r, 0.2)
    s_l = err["lateral_sigma_m"]
    cov3 = s_r ** 2 * np.outer(u, u) + s_l ** 2 * (np.eye(3) - np.outer(u, u))
    return {"agent": agent, "t": t, "z": p[:2], "R": cov3[:2, :2]}


def make_observation_2d(agent, t, xy, cam_xy, sigma_r, sigma_l):
    """Ground position + covariance: sigma_r along the line of sight, sigma_l sideways."""
    xy, c = np.asarray(xy, float), np.asarray(cam_xy, float)
    u = (xy - c) / max(np.linalg.norm(xy - c), 1e-6)
    R = sigma_r ** 2 * np.outer(u, u) + sigma_l ** 2 * (np.eye(2) - np.outer(u, u))
    return {"agent": agent, "t": t, "z": xy, "R": R}


class Track:
    def __init__(self, tid, obs):
        self.id, self.t, self.hits = tid, obs["t"], 1
        self.last_update = obs["t"]
        self.x = np.array([obs["z"][0], obs["z"][1], 0.0, 0.0])
        self.P = np.diag([1.0, 1.0, 1.0, 1.0])
        self.P[:2, :2] = obs["R"]
        self.agents = {obs["agent"]}
        self.confirmed = False
        self.rescued = False

    def predicted(self, t, accel_sigma):
        dt = max(0.0, t - self.t)
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = accel_sigma ** 2
        Q = q * np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                          [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]])
        return F @ self.x, F @ self.P @ F.T + Q

    def predict(self, t, accel_sigma):
        self.x, self.P = self.predicted(t, accel_sigma)
        self.t = max(self.t, t)
        if self.t - self.last_update > 2.0:
            self.x[2:] = 0.0

    def update(self, obs):
        S = H @ self.P @ H.T + obs["R"]
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (obs["z"] - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1
        self.last_update = obs["t"]
        self.agents.add(obs["agent"])


class WorldTracker:
    def __init__(self, accel_sigma=0.7, gate=9.21, confirm_hits=2, max_age=15.0, confirm_std=1.5):
        self.accel_sigma, self.gate = accel_sigma, gate          # gate: chi-square 99%, 2 dof
        self.confirm_hits, self.max_age = confirm_hits, max_age
        self.confirm_std = confirm_std                         # show a person only once their position is known this well (m)
        self.tracks, self.next_id = [], 1

    def step(self, t, batch):
        """batch: observations from ONE agent at time t."""
        for tr in self.tracks:
            tr.predict(t, self.accel_sigma)
        if batch and self.tracks:
            cost = np.zeros((len(batch), len(self.tracks)))
            for i, o in enumerate(batch):
                for j, tr in enumerate(self.tracks):
                    S = H @ tr.P @ H.T + o["R"]
                    y = o["z"] - H @ tr.x
                    cost[i, j] = y @ np.linalg.solve(S, y)
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=self.gate)
        else:
            x = [-1] * len(batch)
        for i, o in enumerate(batch):
            if x[i] >= 0:
                self.tracks[x[i]].update(o)
            else:
                self.tracks.append(Track(self.next_id, o))
                self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.rescued or t - tr.last_update <= self.max_age]
        self._merge_duplicates()

    def _merge_duplicates(self, merge_gate=5.99, max_dist=6.0):
        """Merge tracks that are statistically the same person (chi-square 95%, 2 dof)."""
        merged = True
        while merged:
            merged = False
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    a, b = self.tracks[i], self.tracks[j]
                    d = a.x[:2] - b.x[:2]
                    if np.linalg.norm(d) > max_dist:
                        continue
                    if d @ np.linalg.solve(a.P[:2, :2] + b.P[:2, :2], d) > merge_gate:
                        continue
                    ia, ib = np.linalg.inv(a.P), np.linalg.inv(b.P)
                    P = np.linalg.inv(ia + ib)
                    keep, drop = (a, b) if a.id < b.id else (b, a)
                    keep.x, keep.P = P @ (ia @ a.x + ib @ b.x), P
                    keep.hits += drop.hits
                    keep.last_update = max(a.last_update, b.last_update)
                    keep.agents |= drop.agents
                    keep.confirmed |= drop.confirmed
                    keep.rescued |= drop.rescued
                    self.tracks.remove(drop)
                    merged = True
                    break
                if merged:
                    break

    def confirmed(self, t):
        out = []
        for tr in self.tracks:
            if not tr.confirmed and tr.hits >= self.confirm_hits:
                _, Pp = tr.predicted(t, self.accel_sigma)
                tr.confirmed = np.sqrt(np.linalg.eigvalsh(Pp[:2, :2]).max()) <= self.confirm_std
            if tr.confirmed:
                xp, _ = tr.predicted(min(t, tr.last_update + 2.0), self.accel_sigma)
                out.append((tr.id, xp[:2]))
        return out


def _mark_rescued(self, track_id):
    for tr in self.tracks:
        if tr.id == track_id:
            tr.rescued = True


WorldTracker.mark_rescued = _mark_rescued
