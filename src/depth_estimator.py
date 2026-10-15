"""Monocular thermal depth (U-Net, ResNet-34 encoder) + per-person distance."""
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from PIL import Image

DEFAULT_WEIGHTS = os.path.expanduser("~/thermal-mas-v2/weights/thermal_depth_unet.pt")
MAX_D = 100.0
NET_SIZE = (320, 256)          # width, height used in training


class ThermalDepth:
    def __init__(self, weights=DEFAULT_WEIGHTS, device="mps"):
        self.device = device
        self.model = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(device).eval()

    @torch.no_grad()
    def depth_map(self, thermal_u8):
        """thermal_u8: (H, W) uint8 at any size. Returns depth in metres at 320x256."""
        img = Image.fromarray(thermal_u8)
        if img.size != NET_SIZE:
            img = img.resize(NET_SIZE, Image.BILINEAR)
        x = torch.from_numpy(np.array(img)).float().div(255).sub(0.5).div(0.25)[None, None].to(self.device)
        return torch.exp(self.model(x)).clamp(0.1, MAX_D)[0, 0].cpu().numpy()

    @staticmethod
    def box_distance(depth, xyxy, image_size, percentile=20):
        """Distance to an object from its box (pixel xyxy in the original image)."""
        h, w = depth.shape
        sx, sy = w / image_size[0], h / image_size[1]
        x0, y0 = int(xyxy[0] * sx), int(xyxy[1] * sy)
        x1, y1 = int(np.ceil(xyxy[2] * sx)), int(np.ceil(xyxy[3] * sy))
        patch = depth[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        return float(np.percentile(patch, percentile)) if patch.size else float("nan")
