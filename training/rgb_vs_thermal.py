"""RGB (COCO-pretrained YOLOv8s) vs thermal (ours) on the SAME recorded frames, day/night x fog."""
import dataclasses, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
from PIL import Image, ImageDraw
from ultralytics import YOLO
import thermal_camera
from thermal_camera import synthesize, to_uint8
from thermal_sensor import random_conditions

ROOT = os.path.expanduser("~/thermal-mas-v2")
OUT = f"{ROOT}/outputs/results/rgb_vs_thermal.md"
FOGS = [0.0, 0.03, 0.09, 0.15]
NIGHT_LIGHT, READ_NOISE = 0.10, 0.01           # night: 10% light; sensor noise amplified by 1/0.10 after auto-gain
rgb_model = YOLO("yolov8s.pt")
th_model = YOLO(f"{ROOT}/weights/yolo_thermal_finetuned.pt")
rng = np.random.default_rng(0)

frames = sorted(glob.glob(f"{ROOT}/data/rgb_dump/*/*_rgb.png"))[::2]
print(f"{len(frames)} frames")


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def score(model, img3, gts, acc):
    r = model.predict(img3, conf=0.25, device="mps", verbose=False)[0]
    preds = [b for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()) if model.names[c] == "person"]
    m = 0
    if gts and preds:
        cost = np.array([[1 - iou(g, p) for p in preds] for g in gts])
        _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=0.5)
        m = int(sum(1 for j in x if j >= 0))
    acc[0] += m; acc[1] += len(gts) - m; acc[2] += len(preds) - m
    return preds


def rgb_condition(rgb, depth, fog, night):
    J = rgb.astype(np.float32) / 255.0
    light = NIGHT_LIGHT if night else 1.0
    airlight = 0.05 if night else 0.8
    t = np.exp(-fog * depth)[..., None]
    I = (J * light) * t + airlight * light * (1 - t)
    I = I + rng.normal(0, READ_NOISE, I.shape)
    I = I / light                                                  # camera auto-gain
    return (np.clip(I, 0, 1) * 255).astype(np.uint8)


cfg0, materials, _ = random_conditions(np.random.default_rng(7))
mats, untagged, sky = materials
saved = (thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY)
acc = {}
example = None
thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY = mats, untagged, sky
try:
    for k, fp in enumerate(frames):
        base = fp[:-8]
        rgb = np.array(Image.open(fp).convert("RGB"))
        seg = np.array(Image.open(base + "_seg.png").convert("RGB"))
        person = (seg[..., 0] >= 200) & (seg[..., 1] <= 5) & (seg[..., 2] <= 5)
        seg[person] = (255, 0, 0)
        depth = np.load(base + "_depth.npy").astype(np.float32)
        gts = json.load(open(base + "_boxes.json"))
        for fog in FOGS:
            for night in (False, True):
                img = rgb_condition(rgb, depth, fog, night)
                p = score(rgb_model, img, gts, acc.setdefault(("rgb", fog, night), [0, 0, 0]))
                if example is None and night and fog == 0.09 and len(gts) >= 3:
                    example = {"rgb_day": rgb, "rgb_night_fog": img, "rgb_pred": p, "gts": gts, "base": base}
            th = to_uint8(synthesize(seg, depth, cfg=dataclasses.replace(cfg0, attenuation_per_m=fog), rng=rng))
            p = score(th_model, np.repeat(th[:, :, None], 3, axis=2), gts, acc.setdefault(("thermal", fog), [0, 0, 0]))
            if example is not None and example["base"] == base and fog == 0.09:
                example["th"], example["th_pred"] = th, p
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(frames)} frames done")
finally:
    thermal_camera.MATERIALS, thermal_camera.UNTAGGED_GROUND, thermal_camera.SKY = saved

f = lambda a: f"{a[0] / max(1, a[0] + a[1]):.3f} / {a[0] / max(1, a[0] + a[2]):.3f}"
lines = ["| Condition | RGB, COCO YOLOv8s (recall / precision) | Thermal, ours (recall / precision) |", "|---|---|---|"]
for night in (False, True):
    for fog in FOGS:
        lines.append(f"| {'Night' if night else 'Day'}, fog {fog:.2f} | {f(acc[('rgb', fog, night)])} | {f(acc[('thermal', fog)])} |")
table = "\n".join(lines)
print(table)
open(OUT, "w").write(f"# RGB vs thermal person detection, {len(frames)} identical frames from 3 Webots missions\n\n"
                     "Night = 10% light with 10x amplified sensor noise. Fog: same attenuation for RGB and thermal "
                     "(generous to RGB; visible light is scattered more than LWIR in real fog).\n\n" + table + "\n")
if example and "th" in example:
    tiles = []
    for img, preds, title in ((example["rgb_day"], None, "RGB day"), (example["rgb_night_fog"], example["rgb_pred"], "RGB night, fog 0.09"),
                              (np.repeat(example["th"][:, :, None], 3, axis=2), example["th_pred"], "Thermal, fog 0.09")):
        im = Image.fromarray(img).convert("RGB")
        d = ImageDraw.Draw(im)
        for g in example["gts"]:
            d.rectangle(g, outline=(0, 120, 255), width=1)
        for b in preds or []:
            d.rectangle(b, outline=(0, 255, 0), width=2)
        d.rectangle([0, 0, 230, 18], fill=(0, 0, 0))
        d.text((5, 3), title, fill=(255, 255, 255))
        tiles.append(im)
    grid = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
    x = 0
    for t in tiles:
        grid.paste(t, (x, 0))
        x += t.width
    grid.save(f"{ROOT}/outputs/results/rgb_vs_thermal_example.png")
    print("example image saved: outputs/results/rgb_vs_thermal_example.png (blue = true people, green = detections)")
print("saved", OUT)
