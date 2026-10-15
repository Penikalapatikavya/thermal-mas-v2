"""One thermal sensor (camera + depth) shared by every agent."""
import os
import time
import numpy as np
from dataclasses import replace
from thermal_camera import synthesize, to_uint8, ThermalConfig, MATERIALS, Material
from dataset_recorder import labels_from_recognition, save_sample, draw_preview
import config


def random_conditions(rng):
    """Domain randomisation: different night, camera and weather for each sample."""
    ambient = float(rng.uniform(278.0, 298.0))
    shift = ambient - 288.0
    fog = 0.0 if rng.random() < 0.5 else float(rng.uniform(0.005, 0.04))
    cfg = ThermalConfig(
        ambient_k=ambient,
        reflected_k=275.0 + shift,
        netd_k=float(rng.uniform(0.03, 0.10)),
        attenuation_per_m=fog,
        display_range_k=(280.0 + shift, 310.0 + shift * 0.5),
    )
    materials = {}
    for colour, m in MATERIALS.items():
        if m.name == "person":
            t = 306.0 + float(rng.uniform(-3.0, 3.0))
        else:
            t = m.temperature_k + shift * 0.7 + float(rng.uniform(-1.0, 1.0))
        materials[colour] = replace(m, temperature_k=t)
    import thermal_camera
    untagged = replace(thermal_camera.UNTAGGED_GROUND,
                       temperature_k=290.0 + shift * 0.7 + float(rng.uniform(-1.0, 1.0)))
    sky = replace(thermal_camera.SKY, temperature_k=260.0 + shift * 0.5)
    return cfg, (materials, untagged, sky), {"ambient_k": round(ambient, 1), "fog": round(fog, 3)}


class ThermalSensor:
    def __init__(self, robot, agent_name, cam_name="thermal_cam", depth_name="depth_cam"):
        self.robot = robot
        self.agent = agent_name
        p = config.SENSOR_PERIOD_MS
        self.cam = robot.getDevice(cam_name)
        self.cam.enable(p)
        self.cam.recognitionEnable(p)
        self.cam.enableRecognitionSegmentation()
        self.depth = robot.getDevice(depth_name)
        self.depth.enable(p)
        self.w, self.h = self.cam.getWidth(), self.cam.getHeight()
        self.rng = np.random.default_rng()
        self.fixed = None  # set to fixed conditions for live runs

        self.root = os.path.join(os.path.expanduser(config.DATA_ROOT), agent_name)
        self.preview_dir = os.path.join(self.root, "previews")
        os.makedirs(self.preview_dir, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.saved = 0
        self.next_record_t = 1.0  # give sensors a moment to start

    def read(self):
        seg = np.frombuffer(self.cam.getRecognitionSegmentationImage(), np.uint8) \
            .reshape(self.h, self.w, 4)[:, :, [2, 1, 0]]
        person = (seg[..., 0] >= 200) & (seg[..., 1] <= 5) & (seg[..., 2] <= 5)
        seg_thermal = seg.copy()
        seg_thermal[person] = (255, 0, 0)
        dep = np.array(self.depth.getRangeImage(), dtype=np.float32) \
            .reshape(self.depth.getHeight(), self.depth.getWidth())
        labels = labels_from_recognition(seg, self.cam.getRecognitionObjects())
        car_boxes = [(int((x - bw / 2) * self.w), int((y - bh / 2) * self.h),
                      int(np.ceil((x + bw / 2) * self.w)), int(np.ceil((y + bh / 2) * self.h)))
                     for cls, x, y, bw, bh in labels if cls == 1]
        cfg, materials, self.last_conditions = (self.fixed if self.fixed is not None
                                                else random_conditions(self.rng))
        import thermal_camera
        mats, untagged, sky = materials
        saved = (thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY)
        thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY = mats, untagged, sky
        try:
            thermal = to_uint8(synthesize(seg_thermal, dep, cfg=cfg, rng=self.rng, car_boxes=car_boxes))
        finally:
            thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY = saved
        return thermal, labels, dep, seg

    def maybe_record(self):
        """Call every step; saves a sample when it's time. Returns True when finished."""
        if self.saved >= config.MAX_SAMPLES:
            return True
        t = self.robot.getTime()
        if t < self.next_record_t:
            return False
        self.next_record_t = t + config.RECORD_EVERY_S
        thermal, labels, dep, seg = self.read()
        name = f"{self.agent}_{self.run_id}_{self.saved:05d}"
        save_sample(self.root, name, thermal, labels, dep, seg=seg)
        if self.saved < config.PREVIEW_SAMPLES:
            draw_preview(thermal, labels, os.path.join(self.preview_dir, name + ".png"))
        self.saved += 1
        counts = {}
        for l in labels:
            counts[l[0]] = counts.get(l[0], 0) + 1
        print(f"{self.agent}: sample {self.saved} t={t:.1f}s persons={counts.get(0, 0)} cars={counts.get(1, 0)} {self.last_conditions}")
        if self.saved >= config.MAX_SAMPLES:
            print(f"{self.agent}: recording done ({self.saved} samples)")
            return True
        return False

    def perceive_inputs(self):
        """Thermal image, ground-truth boxes with object IDs (evaluation only), depth."""
        thermal, labels, dep, seg = self.read()
        gt = labels_from_recognition(seg, self.cam.getRecognitionObjects(), with_ids=True)
        return thermal, gt, dep
