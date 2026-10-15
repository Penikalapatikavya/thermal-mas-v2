"""Package the files needed for fine-tuning on Google Colab."""
import os
import random
import shutil

HOME = os.path.expanduser("~")
OUT = os.path.join(HOME, "thermal-mas-v2", "colab_upload")
SIM = os.path.join(HOME, "thermal-mas-v2", "data", "sim_yolo")
FLIR = os.path.join(HOME, "thermal-mas", "data", "yolo_thermal")
WEIGHTS = os.path.join(HOME, "thermal-mas-v2", "weights", "best_kaggle.pt")
FLIR_TRAIN_SUBSET = 3000

os.makedirs(OUT, exist_ok=True)

shutil.make_archive(os.path.join(OUT, "sim_yolo"), "zip", SIM)
print("sim_yolo.zip done")

tmp = os.path.join(OUT, "flir_subset")
if os.path.exists(tmp):
    shutil.rmtree(tmp)
random.seed(0)
counts = {}
for split, limit in (("train", FLIR_TRAIN_SUBSET), ("val", None)):
    img_dir = os.path.join(FLIR, "images", split)
    names = sorted(os.listdir(img_dir))
    if limit:
        names = random.sample(names, min(limit, len(names)))
    os.makedirs(os.path.join(tmp, "images", split))
    os.makedirs(os.path.join(tmp, "labels", split))
    for n in names:
        shutil.copy(os.path.join(img_dir, n), os.path.join(tmp, "images", split, n))
        lab = os.path.splitext(n)[0] + ".txt"
        src = os.path.join(FLIR, "labels", split, lab)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tmp, "labels", split, lab))
    counts[split] = len(names)
with open(os.path.join(tmp, "data.yaml"), "w") as f:
    f.write("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: car\n  2: bike\n")
shutil.make_archive(os.path.join(OUT, "flir_subset"), "zip", tmp)
shutil.rmtree(tmp)
print("flir_subset.zip done:", counts)

shutil.copy(WEIGHTS, os.path.join(OUT, "best_kaggle.pt"))
print("best_kaggle.pt copied")
