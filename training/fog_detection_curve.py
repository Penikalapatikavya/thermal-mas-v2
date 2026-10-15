"""Re-render the test images at increasing fog and measure person detection (recall / precision)."""
import dataclasses, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
from PIL import Image
from ultralytics import YOLO
import thermal_camera
from thermal_camera import synthesize, to_uint8
from thermal_sensor import random_conditions

ROOT = os.path.expanduser("~/thermal-mas-v2")
OUT = f"{ROOT}/outputs/results/fog_detection_curve.md"
FOGS = [0.0, 0.03, 0.06, 0.09, 0.15, 0.25, 0.40, 0.60]
model = YOLO(f"{ROOT}/weights/yolo_thermal_finetuned.pt")
names = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{ROOT}/data/sim_yolo/images/test/*.png"))]


def agent_of(n):
    return "pole_cam" if n.startswith("pole_cam") else n.split("_")[0]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


samples = []
for n in names:
    ag = agent_of(n)
    seg_p, dep_p = f"{ROOT}/data/sim_raw/{ag}/seg/{n}.png", f"{ROOT}/data/sim_raw/{ag}/depth/{n}.npy"
    if not (os.path.exists(seg_p) and os.path.exists(dep_p)):
        continue
    seg = np.array(Image.open(seg_p).convert("RGB"))
    person = (seg[..., 0] >= 200) & (seg[..., 1] <= 5) & (seg[..., 2] <= 5)
    seg[person] = (255, 0, 0)
    dep = np.load(dep_p).astype(np.float32)
    dep = np.repeat(np.repeat(dep, 2, axis=0), 2, axis=1)[:seg.shape[0], :seg.shape[1]]
    H, W = seg.shape[:2]
    gts = []
    for line in open(f"{ROOT}/data/sim_yolo/labels/test/{n}.txt"):
        c, x, y, bw, bh = map(float, line.split()[:5])
        if int(c) == 0:
            gts.append([(x - bw/2) * W, (y - bh/2) * H, (x + bw/2) * W, (y + bh/2) * H])
    samples.append((seg, dep, gts))
print(f"{len(samples)} test images, {sum(len(g) for _, _, g in samples)} people")

rng = np.random.default_rng(0)
cfg0, materials, _ = random_conditions(np.random.default_rng(7))
mats, untagged, sky = materials
saved = (thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY)
lines = ["| Fog (attenuation per m) | Transmission at 25 m | Person recall | Precision |", "|---|---|---|---|"]
thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY = mats, untagged, sky
try:
    for fog in FOGS:
        cfg = dataclasses.replace(cfg0, attenuation_per_m=fog)
        tp = fp = fn = 0
        for seg, dep, gts in samples:
            img = to_uint8(synthesize(seg, dep, cfg=cfg, rng=rng))
            r = model.predict(np.repeat(img[:, :, None], 3, axis=2), conf=0.25, device="mps", verbose=False)[0]
            preds = [b for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()) if model.names[c] == "person"]
            m = 0
            if gts and preds:
                cost = np.array([[1 - iou(g, p) for p in preds] for g in gts])
                _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=0.5)
                m = int(sum(1 for j in x if j >= 0))
            tp += m; fn += len(gts) - m; fp += len(preds) - m
        row = f"| {fog:.2f} | {np.exp(-fog * 25) * 100:.0f}% | {tp / max(1, tp + fn):.3f} | {tp / max(1, tp + fp):.3f} |"
        print(row)
        lines.append(row)
finally:
    thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY = saved
open(OUT, "w").write("# Thermal person detection vs fog (270 test images re-rendered, fine-tuned YOLOv8s, conf 0.25)\n\n"
                     + "\n".join(lines) + "\n")
print("saved", OUT)
