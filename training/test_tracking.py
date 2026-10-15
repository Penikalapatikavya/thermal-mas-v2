"""Offline tracking test on 60 consecutive pole camera frames."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from PIL import Image
from perception import ThermalTracker

src = os.path.expanduser("~/thermal-mas-v2/data/sim_raw/pole_cam/images")
frames = sorted(os.listdir(src))[100:160]
tracker = ThermalTracker()
out, ids_seen = [], set()
for i, f in enumerate(frames):
    img = np.array(Image.open(os.path.join(src, f)))
    tracks, res = tracker.update(img)
    ids_seen.update(t["id"] for t in tracks if t["id"] is not None)
    if i % 10 == 0:
        print(f"frame {i:2d}: " + ", ".join(f'{t["cls"]}#{t["id"]}' for t in tracks))
    out.append(Image.fromarray(res.plot()[:, :, ::-1]).resize((480, 384)))
os.makedirs(os.path.expanduser("~/thermal-mas-v2/outputs/tracking"), exist_ok=True)
gif = os.path.expanduser("~/thermal-mas-v2/outputs/tracking/pole_cam_tracking.gif")
out[0].save(gif, save_all=True, append_images=out[1:], duration=300, loop=0)
print("unique track IDs:", sorted(ids_seen))
print("saved", gif)
