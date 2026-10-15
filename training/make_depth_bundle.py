"""Package thermal images + depth (+ boxes) for depth training on Colab."""
import os
import shutil
import numpy as np
from PIL import Image

RAW = os.path.expanduser("~/thermal-mas-v2/data/sim_raw")
OUT = os.path.expanduser("~/thermal-mas-v2/colab_upload")
TMP = os.path.join(OUT, "depth_data")
SIZE = (320, 256)

if os.path.exists(TMP):
    shutil.rmtree(TMP)
counts = {}
for split in ("train", "val", "test"):
    entries = [l.strip() for l in open(os.path.join(RAW, f"depth_{split}.txt")) if l.strip()]
    for sub in ("images", "depth", "boxes"):
        os.makedirs(os.path.join(TMP, split, sub))
    for rel in entries:
        agent, _, fname = rel.split("/")
        name = fname[:-4]
        img = Image.open(os.path.join(RAW, agent, "images", name + ".png")).resize(SIZE, Image.BILINEAR)
        img.save(os.path.join(TMP, split, "images", name + ".png"))
        d = np.load(os.path.join(RAW, agent, "depth", name + ".npy")).astype(np.float32)
        cm = np.clip(np.round(d * 100), 0, 65535).astype(np.uint16)
        Image.fromarray(cm).save(os.path.join(TMP, split, "depth", name + ".png"))
        shutil.copy(os.path.join(RAW, agent, "labels", name + ".txt"),
                    os.path.join(TMP, split, "boxes", name + ".txt"))
    counts[split] = len(entries)
shutil.make_archive(TMP, "zip", TMP)
shutil.rmtree(TMP)
print("depth_data.zip done:", counts)
