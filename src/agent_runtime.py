"""Live perception for an agent: detection + tracking, logged with ground truth."""
import json
import os
import time
import numpy as np
from PIL import Image
import config
from perception import ThermalTracker
from depth_estimator import ThermalDepth
from geometry import camera_rotation, cam_to_world
from fusion import make_observation, make_observation_2d, WorldTracker
import math
from thermal_sensor import random_conditions


class LivePerception:
    def __init__(self, robot, sensor, agent):
        self.robot, self.sensor, self.agent = robot, sensor, agent
        self.tracker = ThermalTracker()
        self.depth_est = ThermalDepth()
        self.w, self.h = sensor.w, sensor.h
        self.fx = (self.w / 2) / math.tan(sensor.cam.getFov() / 2)
        self.fy = self.fx
        self.gps = robot.getDevice("gps")
        self.gps.enable(config.SENSOR_PERIOD_MS)
        self.imu = robot.getDevice("inertial unit")
        self.imu.enable(config.SENSOR_PERIOD_MS)
        self.mount_pitch = config.CAMERA_MOUNT_PITCH[agent]
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(config.SENSOR_PERIOD_MS)
        self.world = WorldTracker()
        self.comm_rng = np.random.default_rng(config.LIVE_SEED + sum(map(ord, agent)))
        self.comm_stats = {"sent": 0, "received": 0, "dropped": 0}
        self.others, self.own_pose = {}, None
        self.cell_seen = np.full(16, -30.0)
        self.rescued_info = {"count": 0, "xy": []}
        self.remote_rescued = {"count": 0, "xy": []}
        sensor.fixed = random_conditions(np.random.default_rng(config.LIVE_SEED))
        if os.environ.get("FOG") is not None:              # fog sweep: force one fog level for all agents
            import dataclasses
            fog = float(os.environ["FOG"])
            cfg, mats, info = sensor.fixed
            keys = [k for k in vars(cfg) if "fog" in k.lower() or "attenuation" in k.lower()]
            if not keys:
                raise RuntimeError("FOG override: no fog/attenuation field found in the thermal config")
            for k in keys:
                try:
                    setattr(cfg, k, fog)
                except Exception:
                    cfg = dataclasses.replace(cfg, **{k: fog})
            sensor.fixed = (cfg, mats, dict(info, fog=fog))
            print(f"{agent}: fog forced to {fog} (fields set: {keys})")
        self.period = 1.0 / config.PERCEPTION_HZ
        self.next_t = 1.0
        log_dir = os.path.expanduser(config.LOG_ROOT)
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{agent}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self.log = open(self.log_path, "w")
        self.frame_dir = self.log_path[:-6] + "_frames"
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frames = 0
        self.done = False
        self.last_tracks = []
        self.last_fused = []
        self.next_dump = 1.0
        print(f"{agent}: live perception on, conditions {sensor.fixed[2]}, logging to {self.log_path}")

    def step(self):
        t = self.robot.getTime()
        if self.done or t < self.next_t:
            return self.last_tracks
        if t > config.LIVE_SECONDS:
            self.done = True
            self.log.close()
            print(f"{self.agent}: live run done ({self.frames} frames)")
            return self.last_tracks
        self.next_t = t + self.period
        thermal, gt, _ = self.sensor.perceive_inputs()
        tracks, _ = self.tracker.update(thermal)
        cam_pos = list(self.gps.getValues())
        rpy = list(self.imu.getRollPitchYaw())
        R = camera_rotation(rpy[0], rpy[1], rpy[2], self.mount_pitch)
        self.own_pose = (np.array(cam_pos[:2]), rpy[2])
        dmap = self.depth_est.depth_map(thermal)
        for trk in tracks:
            x0, y0, x1, y1 = trk["xyxy"]
            dist = ThermalDepth.box_distance(dmap, trk["xyxy"], (self.w, self.h))
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            ray = np.array([1.0, -(u - self.w / 2) / self.fx, -(v - self.h / 2) / self.fy])
            trk["dist"] = dist
            trk["p_planar"] = (ray * dist).tolist()                       # if depth is along the view axis
            trk["p_eucl"] = (ray / np.linalg.norm(ray) * dist).tolist()  # if depth is straight-line distance
            trk["p_world"] = cam_to_world(trk["p_planar"], cam_pos, R).tolist()
            cam = np.array(cam_pos)
            em = config.ERROR_MODEL[self.agent]
            dir_c = R @ ray
            dir2 = dir_c[:2] / max(np.linalg.norm(dir_c[:2]), 1e-6)
            ests = []
            r_l = float(np.linalg.norm(np.array(trk["p_world"][:2]) - cam[:2]))
            if np.isfinite(r_l):
                ests.append((r_l, max(em["learned_rel"] * r_l, 0.2)))
            if y1 < self.h - 2:
                dg = R @ np.array([1.0, -(u - self.w / 2) / self.fx, -(y1 - self.h / 2) / self.fy])
                if dg[2] < -1e-3:
                    g = cam + (-cam[2] / dg[2]) * dg
                    r_g = float(np.linalg.norm(g[:2] - cam[:2]))
                    trk["p_ground"] = g.tolist()
                    ests.append((r_g, max(em["ground_rel"] * r_g, 0.1)))
            if ests:
                wsum = sum(1.0 / sg ** 2 for _, sg in ests)
                r_f = sum(rr / sg ** 2 for rr, sg in ests) / wsum
                trk["xy_fused"] = (cam[:2] + dir2 * r_f).tolist()
                trk["sigma_r"] = float(1.0 / math.sqrt(wsum))
        own = [make_observation_2d(self.agent, t, trk["xy_fused"], cam_pos[:2], trk["sigma_r"],
                                   config.ERROR_MODEL[self.agent]["lateral_sigma_m"])
               for trk in tracks if trk["cls"] == "person" and "xy_fused" in trk]
        self.emitter.send(json.dumps({"agent": self.agent, "t": t, "pos": list(cam_pos[:2]), "yaw": rpy[2],
                                      "rescued": self.rescued_info if self.agent == "ugv" else None,
                                      "obs": [{"z": o["z"].tolist(), "R": o["R"].tolist()} for o in own]}).encode())
        self.comm_stats["sent"] += 1
        while self.receiver.getQueueLength() > 0:
            raw = self.receiver.getBytes() if hasattr(self.receiver, "getBytes") else self.receiver.getString()
            self.receiver.nextPacket()
            msg = json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
            if msg["agent"] == self.agent:
                continue
            if self.comm_rng.random() < config.COMM_LOSS:
                self.comm_stats["dropped"] += 1
                continue
            self.comm_stats["received"] += 1
            self.others[msg["agent"]] = {"pos": np.array(msg["pos"]), "yaw": msg["yaw"], "t": msg["t"]}
            if msg.get("rescued"):
                self.remote_rescued = msg["rescued"]
            self.world.step(msg["t"], [{"agent": msg["agent"], "t": msg["t"], "z": np.array(o["z"]),
                                        "R": np.array(o["R"])} for o in msg["obs"]])
        self.world.step(t, own)
        fused = self.world.confirmed_with_age(t)
        self.last_fused = fused
        if os.environ.get("RGB_DUMP") == "1" and t >= self.next_dump:      # RGB baseline recording
            self.next_dump = t + 1.0
            d = os.path.expanduser(f"~/thermal-mas-v2/data/rgb_dump/{self.agent}")
            os.makedirs(d, exist_ok=True)
            cam = self.sensor.cam
            rgb = np.frombuffer(cam.getImage(), np.uint8).reshape(cam.getHeight(), cam.getWidth(), 4)[:, :, [2, 1, 0]]
            _, _, dep_full, seg = self.sensor.read()
            name = f"s{config.SEED:03d}_{self.frames:05d}"
            Image.fromarray(np.ascontiguousarray(rgb)).save(f"{d}/{name}_rgb.png")
            Image.fromarray(np.ascontiguousarray(seg)).save(f"{d}/{name}_seg.png")
            np.save(f"{d}/{name}_depth.npy", np.nan_to_num(np.asarray(dep_full, np.float32), nan=100.0,
                                                           posinf=100.0).clip(0, 100).astype(np.float16))
            hh, ww = thermal.shape
            boxes = [[(g[1] - g[3] / 2) * ww, (g[2] - g[4] / 2) * hh, (g[1] + g[3] / 2) * ww, (g[2] + g[4] / 2) * hh]
                     for g in gt if int(g[0]) == 0]
            json.dump(boxes, open(f"{d}/{name}_boxes.json", "w"))
        if config.POLICY == "rl":
            from rl_policy import update_cells
            update_cells(self, t)
        h, w = thermal.shape
        gt_out = [{"id": int(g[5]), "cls": int(g[0]), "p_cam": list(g[6]),
                   "xyxy": [(g[1] - g[3] / 2) * w, (g[2] - g[4] / 2) * h,
                            (g[1] + g[3] / 2) * w, (g[2] + g[4] / 2) * h]} for g in gt]
        frame_name = f"{self.frames:05d}.png"
        if config.SAVE_FRAMES:
            Image.fromarray(thermal).save(os.path.join(self.frame_dir, frame_name))
        self.log.write(json.dumps({"t": round(t, 3), "frame": frame_name, "pose": {"pos": cam_pos, "rpy": rpy},
                                   "gt": gt_out, "pred": tracks, "fused": fused,
                                   "comm": dict(self.comm_stats)}) + "\n")
        self.log.flush()
        self.frames += 1
        if self.frames % 40 == 0:
            print(f"{self.agent}: live frame {self.frames} t={t:.0f}s tracks={len(tracks)} truth={len(gt)} "
                  f"map={len(fused)} comm={self.comm_stats}")
        self.last_tracks = tracks
        return tracks
