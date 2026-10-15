"""Thermal detection + tracking (YOLOv8s fine-tuned + BoT-SORT with camera-motion compensation)."""
import os
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = os.path.expanduser("~/thermal-mas-v2/weights/yolo_thermal_finetuned.pt")


class ThermalTracker:
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.25, device="mps"):
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device
        self.names = self.model.names

    def update(self, thermal_u8):
        """thermal_u8: (H, W) uint8. Returns list of dicts and the YOLO result."""
        frame = np.repeat(thermal_u8[:, :, None], 3, axis=2)
        res = self.model.track(frame, persist=True, tracker="botsort.yaml",
                               conf=self.conf, device=self.device, verbose=False)[0]
        tracks = []
        b = res.boxes
        if b is not None and len(b) > 0:
            ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
            for tid, cls, cf, xyxy in zip(ids, b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist()):
                tracks.append({"id": tid, "cls": self.names[cls], "conf": cf, "xyxy": xyxy})
        return tracks, res
