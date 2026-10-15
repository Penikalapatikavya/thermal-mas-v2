"""Build a YOLO dataset from data/sim_raw with a leakage-free block split.

Consecutive frames are near-duplicates, so frames are grouped into blocks of
BLOCK consecutive samples per agent, and whole blocks go to one split.
Depth maps are not copied; split lists are written for depth training instead.
"""
import os
import shutil
from collections import Counter

ROOT = os.path.expanduser("~/thermal-mas-v2/data")
RAW = os.path.join(ROOT, "sim_raw")
OUT = os.path.join(ROOT, "sim_yolo")
AGENTS = ["pole_cam", "ugv", "uav"]
BLOCK = 30


def split_for(index):
    slot = (index // BLOCK) % 20
    if slot < 14:
        return "train"
    if slot < 17:
        return "val"
    return "test"


if os.path.exists(OUT):
    shutil.rmtree(OUT)
stats = {s: Counter() for s in ("train", "val", "test")}
depth_lists = {s: [] for s in stats}

for agent in AGENTS:
    names = sorted(f[:-4] for f in os.listdir(os.path.join(RAW, agent, "images")) if f.endswith(".png"))
    for name in names:
        idx = int(name.rsplit("_", 1)[1])
        split = split_for(idx)
        for sub, ext in (("images", ".png"), ("labels", ".txt")):
            dst_dir = os.path.join(OUT, sub, split)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(os.path.join(RAW, agent, sub, name + ext), os.path.join(dst_dir, name + ext))
        lines = open(os.path.join(RAW, agent, "labels", name + ".txt")).read().split("\n")
        classes = [int(l.split()[0]) for l in lines if l.strip()]
        st = stats[split]
        st["images"] += 1
        st[f"images_{agent}"] += 1
        st["persons"] += classes.count(0)
        st["cars"] += classes.count(1)
        if not classes:
            st["empty_images"] += 1
        depth_lists[split].append(os.path.join(agent, "depth", name + ".npy"))

with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n")
    f.write("names:\n  0: person\n  1: car\n  2: bike\n")

for split, files in depth_lists.items():
    with open(os.path.join(RAW, f"depth_{split}.txt"), "w") as f:
        f.write("\n".join(files) + "\n")

print(f"{'split':<6} {'images':>7} {'pole':>5} {'ugv':>5} {'uav':>5} {'persons':>8} {'cars':>5} {'empty':>6}")
for split, st in stats.items():
    print(f"{split:<6} {st['images']:>7} {st['images_pole_cam']:>5} {st['images_ugv']:>5} {st['images_uav']:>5} "
          f"{st['persons']:>8} {st['cars']:>5} {st['empty_images']:>6}")
print("written:", OUT)
