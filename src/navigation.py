"""Ground robot: drive to a map point with sonar-based obstacle avoidance."""
import math
import numpy as np
import config

SONAR_NAMES = [f"so{i}" for i in range(16)]
# Pioneer 3-DX sonar directions (deg, 0 = straight ahead, positive = left)
SONAR_ANGLES = [90, 50, 30, 10, -10, -30, -50, -90, -90, -130, -150, -170, 170, 150, 130, 90]


class Navigator:
    def __init__(self, robot):
        self.robot = robot
        p = config.SENSOR_PERIOD_MS
        self.left = robot.getDevice("left wheel")
        self.right = robot.getDevice("right wheel")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.sonars = [robot.getDevice(n) for n in SONAR_NAMES]
        for s in self.sonars:
            s.enable(p)
        self.gps = robot.getDevice("gps")
        self.imu = robot.getDevice("inertial unit")
        self.path_len = 0.0
        self.last_xy = None
        self.near_misses = 0
        self._was_close = False

    def sonar_raw(self):
        return [s.getValue() for s in self.sonars]

    def sonar_dist(self):
        """Sonar values -> metres using each sensor's own lookup table (max range if nothing seen)."""
        out = []
        for s in self.sonars:
            lut = s.getLookupTable()          # flat list: distance, value, noise, distance, value, noise, ...
            pts = sorted((lut[i + 1], lut[i]) for i in range(0, len(lut), 3))
            vals, dists = [p[0] for p in pts], [p[1] for p in pts]
            out.append(float(np.interp(s.getValue(), vals, dists)))
        return out

    def pose(self):
        x, y, _ = self.gps.getValues()
        return x, y, self.imu.getRollPitchYaw()[2]

    def drive_to(self, goal, avoid_points=()):
        """One control step towards goal. Returns distance to goal."""
        x, y, yaw = self.pose()
        if not (math.isfinite(x) and math.isfinite(y)):
            return float("inf")
        if self.last_xy is not None:
            self.path_len += math.hypot(x - self.last_xy[0], y - self.last_xy[1])
        self.last_xy = (x, y)
        dx, dy = goal[0] - x, goal[1] - y
        dist = math.hypot(dx, dy)
        if dist < config.UGV_GOAL_TOLERANCE:
            self.stop()
            return dist
        vx, vy = dx / dist, dy / dist
        closest_person = float("inf")
        blocked = False
        for ox, oy in avoid_points:
            rx, ry = x - ox, y - oy
            r = math.hypot(rx, ry)
            closest_person = min(closest_person, r)
            if 0.01 < r < config.AVOID_RADIUS:
                w = 2.0 * (config.AVOID_RADIUS - r) / config.AVOID_RADIUS
                vx += w * rx / r
                vy += w * ry / r
                ox, oy = -rx / r, -ry / r                      # direction robot -> person
                if ox * dx / dist + oy * dy / dist > 0.5:      # person roughly between robot and goal
                    tx, ty = -oy, ox                           # go around them, on the side closer to the goal
                    if tx * dx + ty * dy < 0:
                        tx, ty = -tx, -ty
                    vx += 1.5 * w * tx
                    vy += 1.5 * w * ty
                if r < 0.9 and ox * math.cos(yaw) + oy * math.sin(yaw) > math.cos(math.radians(70)):
                    blocked = True
        heading_err = (math.atan2(vy, vx) - yaw + math.pi) % (2 * math.pi) - math.pi

        d = self.sonar_dist()
        front = [d[i] for i in range(1, 7)]
        nearest = min(front)
        close = nearest < 0.5
        if close and not self._was_close:
            self.near_misses += 1
        self._was_close = close

        # repulsion from obstacles in front: push the heading away from close sonars
        avoid = 0.0
        for i in range(1, 7):
            if d[i] < 1.5:
                strength = (1.5 - d[i]) / 1.5
                avoid -= strength * math.copysign(1.0, SONAR_ANGLES[i] if SONAR_ANGLES[i] != 0 else 1)
        turn = 1.5 * heading_err + 2.5 * avoid
        speed = config.UGV_MAX_SPEED * max(0.0, math.cos(heading_err)) * min(1.0, max(0.15, (nearest - 0.3) / 1.2))
        if closest_person < 1.5:
            speed *= 0.3
        if dist < 3.0:                                 # careful final approach to the person being rescued
            speed *= min(1.0, max(0.25, (dist - 1.0) / 2.0))
        tx_, ty_ = dx / dist, dy / dist
        if dist < 1.0 and tx_ * math.cos(yaw) + ty_ * math.sin(yaw) > math.cos(math.radians(70)):
            blocked = True                             # the target itself is right in front
        if blocked:
            speed = 0.0                                    # someone right in front: only turn
        turn = max(-1.0, min(1.0, turn)) * config.UGV_MAX_SPEED * 0.6
        self.left.setVelocity(max(-config.UGV_MAX_SPEED, min(config.UGV_MAX_SPEED, speed - turn)))
        self.right.setVelocity(max(-config.UGV_MAX_SPEED, min(config.UGV_MAX_SPEED, speed + turn)))
        return dist

    def stop(self):
        self.left.setVelocity(0.0)
        self.right.setVelocity(0.0)
