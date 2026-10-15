"""UAV agent: Webots Mavic 2 Pro patrol flight code + thermal sensing on the gimbal.
Flight logic adapted from Webots' mavic2pro_patrol.py (Apache 2.0, Cyberbotics)."""
import os
import sys
import numpy as np
from PIL import Image
from controller import Robot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from thermal_camera import synthesize, to_uint8


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)


class Mavic(Robot):
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0
    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    target_precision = 0.5

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.cam = self.getDevice("thermal_cam")
        self.cam.enable(128)
        self.cam.recognitionEnable(128)
        self.cam.enableRecognitionSegmentation()
        self.depth = self.getDevice("depth_cam")
        self.depth.enable(128)

        self.getDevice("camera pitch").setPosition(0.9)

        self.motors = [self.getDevice(n) for n in (
            "front left propeller", "front right propeller",
            "rear left propeller", "rear right propeller")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(1)

        self.current_pose = 6 * [0]
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0

        self.out_dir = os.path.expanduser("~/thermal-mas-v2/outputs/uav_live")
        os.makedirs(self.out_dir, exist_ok=True)
        self.rng = np.random.default_rng()
        self.frames_saved = 0

    def move_to_target(self, waypoints):
        if self.target_position[0:2] == [0, 0]:
            self.target_position[0:2] = waypoints[0]
        if all(abs(a - b) < self.target_precision
               for a, b in zip(self.target_position, self.current_pose[0:2])):
            self.target_index = (self.target_index + 1) % len(waypoints)
            self.target_position[0:2] = waypoints[self.target_index]
            print("uav: waypoint reached, next:", self.target_position[0:2])
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1],
            self.target_position[0] - self.current_pose[0])
        angle_left = self.target_position[2] - self.current_pose[5]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if angle_left > np.pi:
            angle_left -= 2 * np.pi
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        pitch_disturbance = clamp(np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)
        return yaw_disturbance, pitch_disturbance

    def capture(self):
        w, h = self.cam.getWidth(), self.cam.getHeight()
        seg = np.frombuffer(self.cam.getRecognitionSegmentationImage(), np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0]]
        dep = np.array(self.depth.getRangeImage(), dtype=np.float32).reshape(
            self.depth.getHeight(), self.depth.getWidth())
        thermal = synthesize(seg, dep, rng=self.rng)
        self.frames_saved += 1
        Image.fromarray(to_uint8(thermal)).save(
            os.path.join(self.out_dir, f"thermal_{self.frames_saved:03d}.png"))
        print(f"uav: frame {self.frames_saved} saved, mean brightness {thermal.mean():.3f}")
        if self.frames_saved == 3:
            print("uav: test done, stopping saves (still flying)")

    def run(self):
        t1 = self.getTime()
        roll_d = pitch_d = yaw_d = 0
        waypoints = [[38, -32], [52, -45], [45, -55]]
        self.target_altitude = 15
        reached = False
        step = 0

        while self.step(self.time_step) != -1:
            step += 1
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x, y, altitude = self.gps.getValues()
            roll_acc, pitch_acc, _ = self.gyro.getValues()
            self.current_pose = [x, y, altitude, roll, pitch, yaw]
            if step % 250 == 0:
                print(f"uav: t={self.getTime():.1f}s altitude {altitude:.1f} m")

            if altitude > self.target_altitude - 1:
                if not reached:
                    print(f"uav: reached {altitude:.1f} m altitude")
                    reached = True
                if self.getTime() - t1 > 0.1:
                    yaw_d, pitch_d = self.move_to_target(waypoints)
                    t1 = self.getTime()
                if self.frames_saved < 3 and step % 125 == 0:
                    self.capture()

            roll_in = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acc + roll_d
            pitch_in = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acc + pitch_d
            yaw_in = yaw_d
            diff = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vert_in = self.K_VERTICAL_P * pow(diff, 3.0)

            fl = self.K_VERTICAL_THRUST + vert_in - yaw_in + pitch_in - roll_in
            fr = self.K_VERTICAL_THRUST + vert_in + yaw_in + pitch_in + roll_in
            rl = self.K_VERTICAL_THRUST + vert_in + yaw_in - pitch_in - roll_in
            rr = self.K_VERTICAL_THRUST + vert_in - yaw_in - pitch_in + roll_in
            if step % 250 == 0:
                print(f"uav: roll {roll:.2f} pitch {pitch:.2f} yaw {yaw:.2f} | motors {fl:.0f} {fr:.0f} {rl:.0f} {rr:.0f}")
            self.motors[0].setVelocity(fl)
            self.motors[1].setVelocity(-fr)
            self.motors[2].setVelocity(-rl)
            self.motors[3].setVelocity(rr)


Mavic().run()
