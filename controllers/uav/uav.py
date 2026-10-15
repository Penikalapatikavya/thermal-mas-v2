"""UAV agent: kinematic flight to random targets + thermal sensing on the gimbal.

The drone is moved along its path directly (supervisor), assuming an onboard
flight controller handles stabilisation. Sensing uses the real simulated sensors."""
import math
import os
import random
import sys
from controller import Supervisor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from thermal_sensor import ThermalSensor
import config
if config.POLICY == "rl":
    from rl_policy import RLPolicy
from agent_runtime import LivePerception

TAKEOFF_ALTITUDE = 15.0  # m
CLIMB_SPEED = 1.5        # m/s
CRUISE_SPEED = 3.0       # m/s
TURN_RATE = 1.5          # rad/s
AREA_X = (32.0, 58.0)
AREA_Y = (-58.0, -30.0)
ALTITUDE_RANGE = (10.0, 20.0)


class UAV(Supervisor):
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        self.sensor = ThermalSensor(self, "uav")
        self.live = LivePerception(self, self.sensor, "uav")
        self.getDevice("camera pitch").setPosition(0.0)
        for name in ("front left propeller", "front right propeller",
                     "rear left propeller", "rear right propeller"):
            m = self.getDevice(name)
            m.setPosition(float("inf"))
            m.setVelocity(100)  # visual only

        self.node = self.getSelf()
        self.t_field = self.node.getField("translation")
        self.r_field = self.node.getField("rotation")
        self.pos = list(self.t_field.getSFVec3f())
        r = self.r_field.getSFRotation()
        self.yaw = r[3] * (1 if r[2] >= 0 else -1)

        self.rng = random.Random()
        n = config.COVERAGE_GRID
        self.cells = [((AREA_X[0] + (i + 0.5) * (AREA_X[1] - AREA_X[0]) / n),
                       (AREA_Y[0] + (j + 0.5) * (AREA_Y[1] - AREA_Y[0]) / n)) for i in range(n) for j in range(n)]
        self.last_seen = [-1e9] * len(self.cells)
        self.cover_cell = None
        self.policy = RLPolicy() if config.POLICY == "rl" else None
        self.next_rl, self.rl_target = 0.0, None
        self.airborne = False
        self.target = None

    def mark_observed(self):
        x, y, z = self.pos
        off = z / math.tan(0.9)               # camera looks 0.9 rad below horizontal
        gx, gy = x + math.cos(self.yaw) * off, y + math.sin(self.yaw) * off
        t = self.getTime()
        for k, (cx, cy) in enumerate(self.cells):
            if math.hypot(cx - gx, cy - gy) < 8.0:
                self.last_seen[k] = t

    def new_target(self):
        if self.policy is not None and self.rl_target is not None:
            self.target = self.rl_target
            return
        if config.UAV_POLICY == "coverage":
            if self.cover_cell is not None:          # arrived at the viewpoint: that cell is now observed
                self.last_seen[self.cover_cell] = self.getTime()
            oldest = min(self.last_seen)
            options = [c for c, ls in zip(self.cells, self.last_seen) if ls <= oldest + 1.0]
            x, y = self.pos[0], self.pos[1]
            cx, cy = min(options, key=lambda c: math.hypot(c[0] - x, c[1] - y))
            self.cover_cell = self.cells.index((cx, cy))
            # stand off so the tilted camera looks AT the cell, approaching from the current side
            ang = math.atan2(cy - y, cx - x)
            off = 15.0 / math.tan(0.9)
            self.target = (cx - math.cos(ang) * off, cy - math.sin(ang) * off, 15.0)
        else:
            self.target = (self.rng.uniform(*AREA_X), self.rng.uniform(*AREA_Y),
                           self.rng.uniform(*ALTITUDE_RANGE))
        tx, ty, tz = self.target
        print(f"uav: new target ({tx:.1f}, {ty:.1f}) at {tz:.1f} m")

    def fly(self):
        x, y, z = self.pos
        if not self.airborne:
            z = min(TAKEOFF_ALTITUDE, z + CLIMB_SPEED * self.dt)
            if z >= TAKEOFF_ALTITUDE - 0.05:
                self.airborne = True
                print(f"uav: reached {z:.1f} m altitude")
                self.new_target()
        else:
            tx, ty, tz = self.target
            dz = tz - z
            z += max(-CLIMB_SPEED * self.dt, min(CLIMB_SPEED * self.dt, dz))
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < 0.3 and abs(dz) < 0.1:
                self.new_target()
            elif dist >= 0.3:
                step = min(dist, CRUISE_SPEED * self.dt)
                x += dx / dist * step
                y += dy / dist * step
                target_yaw = math.atan2(dy, dx)
                diff = (target_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
                self.yaw += max(-TURN_RATE * self.dt, min(TURN_RATE * self.dt, diff))
        self.pos = [x, y, z]
        self.t_field.setSFVec3f(self.pos)
        self.r_field.setSFRotation([0, 0, 1, self.yaw])
        self.node.resetPhysics()

    def run(self):
        step = 0
        while self.step(self.time_step) != -1:
            step += 1
            self.fly()
            if step % 250 == 0:
                x, y, z = self.pos
                print(f"uav: t={self.getTime():.1f}s pos ({x:.1f}, {y:.1f}) altitude {z:.1f} m")
            if self.airborne:
                self.mark_observed()
                self.sensor.maybe_record()
                self.live.step()
                t_now = self.getTime()
                if self.policy is not None and self.live.own_pose is not None and t_now >= self.next_rl:
                    self.next_rl = t_now + 1.0
                    ugv = self.live.others.get("ugv")
                    ugv_xy = ugv["pos"] if ugv is not None else self.live.own_pose[0]
                    resc = self.live.remote_rescued
                    todo = [q for q in self.live.last_fused
                            if all(math.dist(q["xy"], r) >= 1.0 for r in resc["xy"])]
                    slots = sorted(todo, key=lambda q: math.dist(q["xy"], ugv_xy))[:4]
                    cell = self.policy.uav_choose(self.policy.obs(self.live, 1, t_now, ugv_xy, slots, resc["count"]))
                    cx, cy = self.cells[cell]
                    ang = math.atan2(cy - self.pos[1], cx - self.pos[0])
                    off = 15.0 / math.tan(0.9)
                    self.rl_target = (cx - math.cos(ang) * off, cy - math.sin(ang) * off, 15.0)
                    self.target = self.rl_target


UAV().run()
