"""Shared settings for all agents."""
import os as _os
RUN_ID = _os.environ.get("RUN_ID", "manual")      # set by the batch runner
SEED = int(_os.environ.get("SEED", "7"))           # decides layout + thermal conditions
BATCH = _os.environ.get("BATCH", "0") == "1"       # headless batch run: auto-quit, no frame saving
SAVE_FRAMES = not BATCH
SENSOR_PERIOD_MS = 128      # sensors update ~8 times per second
RECORD_EVERY_S = 1.0        # save one sample per second per agent
MAX_SAMPLES = 0             # dataset recording off (dataset already built)
PREVIEW_SAMPLES = 20        # how many samples also get a preview with boxes
DATA_ROOT = "~/thermal-mas-v2/data/sim_raw"

# live perception (detection + tracking) settings
PERCEPTION_HZ = 4           # detections per second per agent
LIVE_SECONDS = 180          # length of a live evaluation run
LIVE_SEED = SEED            # same thermal conditions for all agents in a run
LOG_ROOT = "~/thermal-mas-v2/outputs/live_logs"
SCENARIO_EPISODE_S = 100000  # no re-arranging during missions     # how often the scenario manager rearranges the scene

# camera tilt (rad, positive = looking down) relative to the agent body
CAMERA_MOUNT_PITCH = {"pole_cam": 0.4, "ugv": 0.0, "uav": 0.9}

# measured observation uncertainty per agent (fitted from a logged live run)
ERROR_MODEL = {"pole_cam": {"learned_rel": 0.0453, "ground_rel": 0.0059, "lateral_sigma_m": 0.123, "range_rel_sigma": 0.0453}, "ugv": {"learned_rel": 0.4403, "ground_rel": 0.067, "lateral_sigma_m": 0.1, "range_rel_sigma": 0.4403}, "uav": {"learned_rel": 0.0393, "ground_rel": 0.0205, "lateral_sigma_m": 0.539, "range_rel_sigma": 0.0393}}

# probability that a message from another agent is lost
COMM_LOSS = float(_os.environ.get("COMM_LOSS", "0.0"))

# ground robot driving
UGV_MAX_SPEED = 6.0             # wheel speed, rad/s (Pioneer max ~12.3)
UGV_GOAL_TOLERANCE = 1.4        # m: robot stops this close to its target

# rescue mission (rule-based decisions)
MISSION_SECONDS = 180           # time limit
RESCUE_RADIUS = 1.5             # m: robot claims a rescue within this distance (evaluation uses 2.0 m)
AVOID_RADIUS = 3.0              # m: other people closer than this push the ground robot away
UAV_POLICY = _os.environ.get("UAV_POLICY", "coverage")   # "coverage" or "random"
COVERAGE_GRID = 4               # cells per side over the rescue area
MISSION_LOG_DIR = "~/thermal-mas-v2/outputs/missions"

# mission setup switches
POLE = _os.environ.get("POLE", "1") == "1"          # POLE=0: the fixed pole camera is out of action
EXPECTED_PEOPLE = 4                                 # reported number of stranded people
PATROL = [(40.0, -40.0), (50.0, -40.0), (50.0, -50.0), (40.0, -50.0)]   # ground robot search loop

# decision policy: "rule" (hand-written) or "rl" (trained MAPPO actors)
POLICY = _os.environ.get("POLICY", "rule")
RL_TAG = _os.environ.get("RL_TAG", "")          # "" = first model (v1), "_dr" = randomised + commitment
RL_WEIGHTS = _os.path.expanduser(f"~/thermal-mas-v2/weights/mappo_pole{1 if POLE else 0}{RL_TAG}.pt")
RL_HYSTERESIS = RL_TAG == "_dr"
