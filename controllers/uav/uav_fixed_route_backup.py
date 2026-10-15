"""UAV agent: kinematic waypoint flight + thermal sensing on the gimbal.

The drone is moved along its path directly (supervisor), assuming an onboard
flight controller handles stabilisation. Sensing uses the real simulated sensors."""
import math
import os
import sys
import numpy as np
from PIL import Image
from controller import Supervisor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from thermal_sensor import ThermalSensor

ALTITUDE = 15.0      # m
CLIMB_SPEED = 1.5    # m/s
CRUISE_SPEED = 3.0   # m/s
TURN_RATE = 1.5      # rad/s
WAYPOINTS = [[38, -32], [52, -45], [45, -55]]


class UAV(Supervisor):
    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        self.sensor = ThermalSensor(self, "uav")
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
        self.wp_index = 0


    def fly(self):
        x, y, z = self.pos
        if z < ALTITUDE - 0.05:
            z = min(ALTITUDE, z + CLIMB_SPEED * self.dt)
            at_altitude = False
        else:
            at_altitude = True
            tx, ty = WAYPOINTS[self.wp_index]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if dist < 0.3:
                self.wp_index = (self.wp_index + 1) % len(WAYPOINTS)
                print("uav: waypoint reached, next:", WAYPOINTS[self.wp_index])
            else:
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
        return at_altitude

    def run(self):
        step = 0
        reached = False
        while self.step(self.time_step) != -1:
            step += 1
            at_alt = self.fly()
            if at_alt and not reached:
                print(f"uav: reached {self.pos[2]:.1f} m altitude")
                reached = True
            if step % 250 == 0:
                x, y, z = self.pos
                print(f"uav: t={self.getTime():.1f}s pos ({x:.1f}, {y:.1f}) altitude {z:.1f} m")
            if at_alt:
                self.sensor.maybe_record()


UAV().run()
