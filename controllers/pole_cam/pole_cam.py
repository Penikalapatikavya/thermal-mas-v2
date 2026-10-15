import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from controller import Robot
from thermal_sensor import ThermalSensor
from agent_runtime import LivePerception

robot = Robot()
timestep = int(robot.getBasicTimeStep())
import config as _cfg
if not _cfg.POLE:
    print('pole_cam: disabled for this run (POLE=0)')
    while robot.step(timestep) != -1:
        pass
    raise SystemExit
sensor = ThermalSensor(robot, "pole_cam")
live = LivePerception(robot, sensor, "pole_cam")
while robot.step(timestep) != -1:
    sensor.maybe_record()
    live.step()
